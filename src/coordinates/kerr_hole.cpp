//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file kerr_hole.cpp
//! \brief quasi-stationary evolution of the fixed Kerr hole (see kerr_hole.hpp).

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "athena.hpp"
#include "globals.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "mesh/meshblock.hpp"
#include "mesh/meshblock_pack.hpp"
#include "mesh/nghbr_index.hpp"
#include "driver/driver.hpp"
#include "eos/eos.hpp"
#include "hydro/hydro.hpp"
#include "mhd/mhd.hpp"
#include "coordinates.hpp"
#include "cartesian_ks.hpp"
#include "cell_locations.hpp"
#include "kerr_hole.hpp"
#include "gravity/gr_selfgravity.hpp"

#if MPI_PARALLEL_ENABLED
#include <mpi.h>
#endif

namespace kerr_hole {

namespace {

//----------------------------------------------------------------------------------------
// The process holds one hole at a time.  Its bookkept state outlives every Coordinates
// object (a regrid deletes and rebuilds Coordinates from the deck), so it lives here and
// the Coordinates constructor takes the live metric from it (RestoreHole).  It ends with
// the MeshBlockPack (Release); the device views are also released by a Kokkos finalize
// hook, before the backend goes away.

struct HoleState {
  bool initialized = false;   // the first RestoreHole has run
  bool evolves = false;
  bool hook_registered = false;
  Mesh *pmesh = nullptr;
  // bookkept hole: total energy and z angular momentum, accumulated exactly
  Real m_true = 1.0, j_true = 0.0;
  // the metric's hole (mirrors CoordData::bh_mass/bh_spin) and the excision radii
  Real m_g = 1.0, a_g = 0.0;
  Real rexcise = 0.0, flux_excise_r = 0.0;

  // Per-block slots (nmb, 2): (dM, dJz) since the last reduction.
  DvceArray2D<Real> slots_fluid, slots_rad;
  bool pending = false;       // a stage has written the slots since the last reduction

  // Hole faces of the fluid, grouped by block: (m, k, j, i, code), code = 2*dir + side,
  // side 1 when the excised cell is on the +dir side of the face.  face_offset(m) is
  // the first row of block m, face_offset(nmb) the count.
  DvceArray2D<int> faces;
  DvceArray1D<int> face_offset;
  int nmb_layout = -1;
  bool layout_valid = false;
  std::uint64_t layout_topology = ~std::uint64_t{0};
  const bool *layout_mask = nullptr;
};

HoleState &State() {
  static HoleState state;
  return state;
}

constexpr Real kRefreshLag = 1.0e-5;   // metric lag that triggers a refresh, in M_g

[[noreturn]] void Fatal(const char *file, int line, const std::string &msg) {
  std::cout << "### FATAL ERROR in " << file << " at line " << line << std::endl
            << msg << std::endl;
  std::exit(EXIT_FAILURE);
}

//----------------------------------------------------------------------------------------
// The Kerr bound |a| < M.  An accreted state beyond it is physically impossible, so it
// signals a defect in the bookkeeping; it is never clamped.
void CheckKerrBound(const Real m, const Real j, const char *where) {
  if (!(std::isfinite(m) && std::isfinite(j) && m > 0.0 && std::abs(j) < m*m)) {
    std::ostringstream msg;
    msg.precision(17);
    msg << "Kerr hole " << where << " violates the Kerr bound or is not finite: M = "
        << m << ", J = " << j << " (|a|/M = " << ((m > 0.0) ? std::abs(j)/(m*m) : 0.0)
        << ").  The hole's mass and spin come from exact face sums of the fluxes into "
        << "the excised region, so this is a bookkeeping defect, not a physical state.";
    Fatal(__FILE__, __LINE__, msg.str());
  }
}

//----------------------------------------------------------------------------------------
// Whether a fluid hole face of block m sits on a block face whose
// neighbour is at another refinement level (or, with lat_same_level, on any block face).
// There a reflux would correct the gas cell after this module read the flux, so the
// face sum would no longer be what the update applied.
void CheckBlockFace(MeshBlockPack *pmbp, const int m, const int dir, const int side) {
  Mesh *pm = pmbp->pmesh;
  const int ix = (dir == 0) ? (side ? 1 : -1) : 0;
  const int iy = (dir == 1) ? (side ? 1 : -1) : 0;
  const int iz = (dir == 2) ? (side ? 1 : -1) : 0;
  const int base = NeighborIndex(ix, iy, iz, 0, 0);
  const int mylev = pmbp->pmb->mb_lev.h_view(m);
  for (int n = base; n < base + 4; ++n) {
    const auto &nb = pmbp->pmb->nghbr.h_view(m, n);
    if (nb.gid < 0) continue;
    if (nb.lev != mylev || pm->hydro_lat_same_level) {
      std::ostringstream msg;
      msg << "The excised region of the Kerr hole reaches the boundary of MeshBlock "
          << pmbp->pmb->mb_gid.h_view(m) << " (level " << mylev << ") toward "
          << ((nb.lev != mylev) ? "a MeshBlock at level " : "a same-level MeshBlock")
          << ((nb.lev != mylev) ? std::to_string(nb.lev) : std::string())
          << ((nb.lev != mylev) ? "" : " under <time>/lat_same_level")
          << ".  The hole's mass and spin are the fluxes into the excised cells, and a "
          << "reflux at that face would change the gas after they were counted.  Refine "
          << "so that the excision sits inside one level with same-level neighbours.";
      Fatal(__FILE__, __LINE__, msg.str());
    }
  }
}

//----------------------------------------------------------------------------------------
// Rebuild the fluid hole-face list and the slots when the topology or the mask moved, and
// assert that no hole face meets a reflux (the fine/coarse-face assertion).  Host only.
void EnsureLayout(MeshBlockPack *pmbp) {
  HoleState &g = State();
  Mesh *pm = pmbp->pmesh;
  const auto &floor = pmbp->pcoord->excision_floor;
  const std::uint64_t topo = pm->topology_version;
  if (g.layout_valid && g.layout_topology == topo && g.layout_mask == floor.data() &&
      g.nmb_layout == pmbp->nmb_thispack) {
    return;
  }
  if (g.pending && g.layout_topology != topo) {
    Fatal(__FILE__, __LINE__,
          "The mesh topology changed while the Kerr hole's face sums were not yet "
          "reduced: kerr_hole::Synchronize must run before every regrid and load "
          "balance, or the energy and angular momentum they hold are lost.");
  }

  const int nmb = pmbp->nmb_thispack;
  const auto &indcs = pm->mb_indcs;
  const int is = indcs.is, ie = indcs.ie;
  const int js = indcs.js, je = indcs.je;
  const int ks = indcs.ks, ke = indcs.ke;
  const bool multi_d = pm->multi_d, three_d = pm->three_d;
  auto hfloor = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), floor);
  std::vector<int> rows;
  std::vector<int> offset(nmb + 1, 0);
  for (int m = 0; m < nmb; ++m) {
    offset[m] = static_cast<int>(rows.size()/5);
    for (int k = ks; k <= ke; ++k) {
      for (int j = js; j <= je; ++j) {
        for (int i = is; i <= ie; ++i) {
          if (hfloor(m, k, j, i)) continue;
          auto add = [&](const int dir, const int side, const int kf, const int jf,
                         const int iff) {
            rows.push_back(m);
            rows.push_back(kf);
            rows.push_back(jf);
            rows.push_back(iff);
            rows.push_back(2*dir + side);
          };
          // face index = the cell on the face's + side
          if (hfloor(m, k, j, i-1)) {
            if (i == is) CheckBlockFace(pmbp, m, 0, 0);
            add(0, 0, k, j, i);
          }
          if (hfloor(m, k, j, i+1)) {
            if (i == ie) CheckBlockFace(pmbp, m, 0, 1);
            add(0, 1, k, j, i+1);
          }
          if (multi_d) {
            if (hfloor(m, k, j-1, i)) {
              if (j == js) CheckBlockFace(pmbp, m, 1, 0);
              add(1, 0, k, j, i);
            }
            if (hfloor(m, k, j+1, i)) {
              if (j == je) CheckBlockFace(pmbp, m, 1, 1);
              add(1, 1, k, j+1, i);
            }
          }
          if (three_d) {
            if (hfloor(m, k-1, j, i)) {
              if (k == ks) CheckBlockFace(pmbp, m, 2, 0);
              add(2, 0, k, j, i);
            }
            if (hfloor(m, k+1, j, i)) {
              if (k == ke) CheckBlockFace(pmbp, m, 2, 1);
              add(2, 1, k+1, j, i);
            }
          }
        }
      }
    }
  }
  offset[nmb] = static_cast<int>(rows.size()/5);

  const int nface = offset[nmb];
  g.faces = DvceArray2D<int>("kerr_hole_faces", std::max(nface, 1), 5);
  auto hfaces = Kokkos::create_mirror_view(g.faces);
  for (int f = 0; f < nface; ++f) {
    for (int c = 0; c < 5; ++c) hfaces(f, c) = rows[5*f + c];
  }
  Kokkos::deep_copy(g.faces, hfaces);
  g.face_offset = DvceArray1D<int>("kerr_hole_face_offset", nmb + 1);
  auto hoff = Kokkos::create_mirror_view(g.face_offset);
  for (int m = 0; m <= nmb; ++m) hoff(m) = offset[m];
  Kokkos::deep_copy(g.face_offset, hoff);

  // A new topology starts with empty slots (the reduction above emptied the old ones).
  if (g.layout_topology != topo || g.nmb_layout != nmb ||
      g.slots_fluid.extent_int(0) != nmb) {
    g.slots_fluid = DvceArray2D<Real>("kerr_hole_slots_fluid", nmb, 2);
    g.slots_rad = DvceArray2D<Real>("kerr_hole_slots_rad", nmb, 2);
  }
  g.nmb_layout = nmb;
  g.layout_topology = topo;
  g.layout_mask = floor.data();
  g.layout_valid = true;
}

//----------------------------------------------------------------------------------------
// Sum the slots over all blocks of all ranks in global block-id order, so the totals do
// not depend on the rank layout, and empty them.  Collective.
void ReduceSlots(MeshBlockPack *pmbp, Real &dm, Real &dj) {
  HoleState &g = State();
  Mesh *pm = pmbp->pmesh;
  const int nmb = pmbp->nmb_thispack;
  auto hf = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), g.slots_fluid);
  auto hr = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), g.slots_rad);
  const int ntot = pm->nmb_total;
  std::vector<Real> all(2*static_cast<std::size_t>(ntot), 0.0);
  const int gid0 = pmbp->gids;
  for (int m = 0; m < nmb; ++m) {
    // fluid of one block first, then blocks in gid order
    all[2*(gid0 + m)] = hf(m, 0) + hr(m, 0);
    all[2*(gid0 + m) + 1] = hf(m, 1) + hr(m, 1);
  }
#if MPI_PARALLEL_ENABLED
  std::vector<int> counts(global_variable::nranks), displs(global_variable::nranks);
  for (int r = 0; r < global_variable::nranks; ++r) {
    counts[r] = 2*pm->nmb_eachrank[r];
    displs[r] = 2*pm->gids_eachrank[r];
  }
  MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, all.data(), counts.data(),
                 displs.data(), MPI_ATHENA_REAL, MPI_COMM_WORLD);
#endif
  dm = 0.0;
  dj = 0.0;
  for (int n = 0; n < ntot; ++n) {
    dm += all[2*n];
    dj += all[2*n + 1];
  }
  // the slots hold code mass, the hole geometric mass: G of the code units converts
  const Real g_code = grsg::GravConstant(pmbp);
  dm *= g_code;
  dj *= g_code;
  Kokkos::deep_copy(g.slots_fluid, 0.0);
  Kokkos::deep_copy(g.slots_rad, 0.0);
  g.pending = false;
}

//----------------------------------------------------------------------------------------
// The S_t impulse on the fluid: IEN += 1/2 T^{mu nu} [g_{mu nu}(new) - g_{mu nu}(old)],
// with T the full gas + EM tensor of the synced primitives on the old metric.  D, the
// momenta, B and the scalars are unchanged.  Active non-excised cells; the ghosts are
// rebuilt by the exchange of the recovery that follows.
void ApplyFluidImpulse(MeshBlockPack *pmbp, const KerrHole &h_old, const KerrHole &h_new) {
  const bool is_mhd = (pmbp->pmhd != nullptr);
  if (!is_mhd && pmbp->phydro == nullptr) return;
  auto u0 = is_mhd ? pmbp->pmhd->u0 : pmbp->phydro->u0;
  auto w0 = is_mhd ? pmbp->pmhd->w0 : pmbp->phydro->w0;
  // The hole evolves only on the fixed metric (grsg::HoleEvolves), where the MHD fluid
  // always carries a field; the field-free fluid (dynamical metric only) would read b = 0.
  const bool has_field = is_mhd && pmbp->pmhd->HasField();
  DvceArray5D<Real> bcc;
  if (has_field) bcc = pmbp->pmhd->bcc0;
  const Real gamma_eos = is_mhd ? pmbp->pmhd->peos->eos_data.gamma :
                                  pmbp->phydro->peos->eos_data.gamma;
  auto excised = pmbp->pcoord->excision_floor;
  auto &indcs = pmbp->pmesh->mb_indcs;
  const int is = indcs.is, ie = indcs.ie;
  const int js = indcs.js, je = indcs.je;
  const int ks = indcs.ks, ke = indcs.ke;
  const int nx1 = indcs.nx1, nx2 = indcs.nx2, nx3 = indcs.nx3;
  auto size = pmbp->pmb->mb_size;
  const int nmb1 = pmbp->nmb_thispack - 1;
  const KerrHole hold = h_old, hnew = h_new;

  par_for("kerr_hole_impulse", DevExeSpace(), 0, nmb1, ks, ke, js, je, is, ie,
  KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
    if (excised(m, k, j, i)) return;
    const Real x = CellCenterX(i-is, nx1, size.d_view(m).x1min, size.d_view(m).x1max);
    const Real y = CellCenterX(j-js, nx2, size.d_view(m).x2min, size.d_view(m).x2max);
    const Real z = CellCenterX(k-ks, nx3, size.d_view(m).x3min, size.d_view(m).x3max);

    Real g[4][4], gi[4][4], gn[4][4], gni[4][4];
    ComputeMetricAndInverse(x, y, z, hold, g, gi);
    ComputeMetricAndInverse(x, y, z, hnew, gn, gni);

    const Real rho = w0(m, IDN, k, j, i);
    const Real uu1 = w0(m, IVX, k, j, i);
    const Real uu2 = w0(m, IVY, k, j, i);
    const Real uu3 = w0(m, IVZ, k, j, i);
    const Real gamma_gas = gamma_eos;
    const Real pgas = (gamma_gas - 1.0)*w0(m, IEN, k, j, i);

    // u^mu on the old metric
    const Real uu_sq = g[1][1]*uu1*uu1 + 2.0*g[1][2]*uu1*uu2 + 2.0*g[1][3]*uu1*uu3
                     + g[2][2]*uu2*uu2 + 2.0*g[2][3]*uu2*uu3 + g[3][3]*uu3*uu3;
    const Real alpha = sqrt(-1.0/gi[0][0]);
    const Real lor = sqrt(1.0 + uu_sq);
    Real u[4];
    u[0] = lor/alpha;
    u[1] = uu1 - alpha*lor*gi[0][1];
    u[2] = uu2 - alpha*lor*gi[0][2];
    u[3] = uu3 - alpha*lor*gi[0][3];

    // b^mu from the densitized cell field (sqrt(-g) = 1)
    Real b[4] = {0.0, 0.0, 0.0, 0.0};
    if (has_field) {
      const Real bb1 = bcc(m, IBX, k, j, i);
      const Real bb2 = bcc(m, IBY, k, j, i);
      const Real bb3 = bcc(m, IBZ, k, j, i);
      Real ul[4];
      for (int a = 1; a < 4; ++a) {
        ul[a] = g[a][0]*u[0] + g[a][1]*u[1] + g[a][2]*u[2] + g[a][3]*u[3];
      }
      b[0] = ul[1]*bb1 + ul[2]*bb2 + ul[3]*bb3;
      b[1] = (bb1 + b[0]*u[1])/u[0];
      b[2] = (bb2 + b[0]*u[2])/u[0];
      b[3] = (bb3 + b[0]*u[3])/u[0];
    }
    Real b_sq = 0.0;
    for (int a = 0; a < 4; ++a) {
      for (int c = 0; c < 4; ++c) b_sq += g[a][c]*b[a]*b[c];
    }
    const Real wtot = rho + gamma_gas/(gamma_gas - 1.0)*pgas + b_sq;
    const Real ptot = pgas + 0.5*b_sq;

    // 1/2 T^{mu nu} Delta g_{mu nu}
    Real t_dg = 0.0;
    for (int a = 0; a < 4; ++a) {
      for (int c = 0; c < 4; ++c) {
        const Real tmn = wtot*u[a]*u[c] + ptot*gi[a][c] - b[a]*b[c];
        t_dg += tmn*(gn[a][c] - g[a][c]);
      }
    }
    u0(m, IEN, k, j, i) += 0.5*t_dg;
  });
}

//----------------------------------------------------------------------------------------
// The fluid share of a moved excision mask, added to the block slots: s_in times the
// content of the interior cells that are excised now but were not in old_floor, s_out
// times that of the cells that left it.  Energy with rest mass -C_t = D - IEN and z
// angular momentum x C_y - y C_x, the quantities the face sums book.  One team per block.
void BookFluidMaskChange(MeshBlockPack *pmbp, const DvceArray4D<bool> &old_floor,
                         const Real s_in, const Real s_out) {
  HoleState &g = State();
  const bool is_mhd = (pmbp->pmhd != nullptr);
  if (!is_mhd && pmbp->phydro == nullptr) return;
  const int nmb = pmbp->nmb_thispack;
  if (nmb <= 0) return;
  auto u0 = is_mhd ? pmbp->pmhd->u0 : pmbp->phydro->u0;
  auto now = pmbp->pcoord->excision_floor;
  auto &indcs = pmbp->pmesh->mb_indcs;
  const int is = indcs.is, js = indcs.js, ks = indcs.ks;
  const int nx1 = indcs.nx1, nx2 = indcs.nx2, nx3 = indcs.nx3;
  const int ncell = nx1*nx2*nx3;
  auto size = pmbp->pmb->mb_size;
  auto slots = g.slots_fluid;

  Kokkos::TeamPolicy<> policy(DevExeSpace(), nmb, Kokkos::AUTO);
  Kokkos::parallel_for("kerr_hole_mask_change", athenak_lw(policy),
  KOKKOS_LAMBDA(TeamMember_t tmember) {
    const int m = tmember.league_rank();
    const Real vol = size.d_view(m).dx1*size.d_view(m).dx2*size.d_view(m).dx3;
    auto weight = [&](const int c, int &k, int &j, int &i) {
      k = ks + c/(nx1*nx2);
      j = js + (c/nx1)%nx2;
      i = is + c%nx1;
      const bool in = now(m, k, j, i);
      return (old_floor(m, k, j, i) == in) ? 0.0 : (in ? s_in : s_out);
    };
    Real dm = 0.0;
    Kokkos::parallel_reduce(Kokkos::TeamThreadRange(tmember, ncell),
    [&](const int c, Real &sum) {
      int k, j, i;
      const Real w = weight(c, k, j, i);
      if (w != 0.0) sum += w*(u0(m, IDN, k, j, i) - u0(m, IEN, k, j, i))*vol;
    }, dm);
    Real dj = 0.0;
    Kokkos::parallel_reduce(Kokkos::TeamThreadRange(tmember, ncell),
    [&](const int c, Real &sum) {
      int k, j, i;
      const Real w = weight(c, k, j, i);
      if (w == 0.0) return;
      const Real x = CellCenterX(i-is, nx1, size.d_view(m).x1min, size.d_view(m).x1max);
      const Real y = CellCenterX(j-js, nx2, size.d_view(m).x2min, size.d_view(m).x2max);
      sum += w*(x*u0(m, IM2, k, j, i) - y*u0(m, IM1, k, j, i))*vol;
    }, dj);
    Kokkos::single(Kokkos::PerTeam(tmember), [&]() {
      slots(m, 0) += dm;
      slots(m, 1) += dj;
    });
  });
}

//----------------------------------------------------------------------------------------
// Refresh the metric to (m_new, a_new) at a global sync (design Sect. 7.4).
void Refresh(Driver *pdrive, Mesh *pm, const Real m_new, const Real a_new) {
  HoleState &g = State();
  MeshBlockPack *pmbp = pm->pmb_pack;
  CoordData &cd = pmbp->pcoord->coord_data;
  CoordData cd_new = cd;
  cd_new.bh_mass = m_new;
  cd_new.bh_spin = a_new;
  const KerrHole h_old = cd.Hole();
  const KerrHole h_new = cd_new.Hole();

  // 1. the S_t impulse on the fluid, on the old metric
  ApplyFluidImpulse(pmbp, h_old, h_new);

  // 2. the new metric; kernels take it at their next launch
  cd.bh_mass = m_new;
  cd.bh_spin = a_new;
  g.m_g = m_new;
  g.a_g = a_new;

  // 3. Excision radii.  A radius at or inside the old horizon follows the horizon
  // inward and never grows back (r <- min(r, r_+)).  A radius the deck put outside the
  // horizon (a user flux_excise_r) is the deck's numerical choice and is kept.  The
  // 1e-12 only absorbs the rounding of two evaluations of the same horizon.
  const Real rp_old = KerrHorizonRadius(h_old);
  const Real rp_new = KerrHorizonRadius(h_new);
  auto follow = [rp_old, rp_new](const Real r) {
    return (r <= rp_old*(1.0 + 1.0e-12)) ? std::min(r, rp_new) : r;
  };
  cd.rexcise = follow(cd.rexcise);
  cd.flux_excise_r = follow(cd.flux_excise_r);
  g.rexcise = cd.rexcise;
  g.flux_excise_r = cd.flux_excise_r;

  // 4. Excision masks.  Kerr-Schild r at a fixed point depends on the spin (r^4 - (R^2 -
  // a^2) r^2 - a^2 z^2 = 0), so the masks follow the hole at every refresh, not only when
  // a radius moves; a mask left on the old spin would also differ from the one a restart
  // builds from the restored hole.  A cell may enter the mask (spin-up) or leave it, so
  // its content is booked: what enters goes to the hole, at the content left after the
  // recovery resets it to the excision state, and the excision state of a cell that
  // leaves joins the gas, paid by the hole.  M_true + E_gas stays exact.
  Coordinates *pc = pmbp->pcoord;
  const auto &fl = pc->excision_floor;
  DvceArray4D<bool> old_floor("kerr_hole_old_floor", fl.extent(0), fl.extent(1),
                              fl.extent(2), fl.extent(3));
  Kokkos::deep_copy(old_floor, fl);
  pc->SetExcisionMasks(pc->excision_floor, pc->excision_flux);
  g.layout_valid = false;
  BookFluidMaskChange(pmbp, old_floor, 1.0, 0.0);

  // 5. primitives on the new metric, through the regrid's entry: exchange and
  // recovery.  It resets the cells that entered
  // the mask, whose remaining content stays out of the hole.  Phi is not invalidated.
  pdrive->InitBoundaryValuesAndPrimitives(pm);
  BookFluidMaskChange(pmbp, old_floor, -1.0, -1.0);
  Real dm = 0.0, dj = 0.0;
  ReduceSlots(pmbp, dm, dj);
  g.m_true += dm;
  g.j_true += dj;
  CheckKerrBound(g.m_true, g.j_true, "state");
}

void ReleaseDeviceState() {
  HoleState &g = State();
  g.slots_fluid = DvceArray2D<Real>();
  g.slots_rad = DvceArray2D<Real>();
  g.faces = DvceArray2D<int>();
  g.face_offset = DvceArray1D<int>();
  g.layout_valid = false;
}

}  // namespace

//----------------------------------------------------------------------------------------
bool Evolves() { return State().evolves; }

//----------------------------------------------------------------------------------------
void Release() {
  HoleState &g = State();
  const bool hook_registered = g.hook_registered;
  g = HoleState();
  g.hook_registered = hook_registered;
}

//----------------------------------------------------------------------------------------
void RestoreHole(ParameterInput *pin, MeshBlockPack *ppack, bool fixed_gr, CoordData &cd) {
  HoleState &g = State();
  if (!fixed_gr || cd.is_minkowski) return;
  const bool gravity_on = pin->DoesBlockExist("gravity") &&
                          pin->GetOrAddBoolean("gravity", "self_gravity", true);
  const bool evolves = gravity_on && cd.bh_excise;

  if (!g.initialized) {
    g.initialized = true;
    g.pmesh = ppack->pmesh;
    // A checkpoint of an evolving hole carries its state; the metric of the checkpointed
    // gas is (M_g, a_g) whether or not this run evolves it further.
    if (pin->DoesParameterExist("coord", "bh_mass_live")) {
      g.m_g = pin->GetReal("coord", "bh_mass_live");
      g.a_g = pin->GetReal("coord", "bh_spin_live");
      g.m_true = pin->GetReal("coord", "bh_mass_true");
      g.j_true = pin->GetReal("coord", "bh_angmom_true");
      g.rexcise = pin->GetReal("coord", "bh_rexcise_live");
      g.flux_excise_r = pin->GetReal("coord", "bh_flux_excise_r_live");
    } else {
      g.m_g = cd.bh_mass;
      g.a_g = cd.bh_spin;
      g.m_true = cd.bh_mass;
      g.j_true = cd.bh_mass*cd.bh_spin;
      g.rexcise = cd.rexcise;
      g.flux_excise_r = cd.flux_excise_r;
    }
    if (evolves) {
      CheckKerrBound(g.m_g, g.m_g*g.a_g, "metric");
      CheckKerrBound(g.m_true, g.j_true, "state");
    }
    if (evolves && pin->DoesBlockExist("radiation")) {
      Fatal(__FILE__, __LINE__,
            "The Kerr hole evolves under GR self-gravity, but the angle-resolved "
            "<radiation> module has neither hole fluxes nor the refresh impulse.  Run "
            "without self-gravity.");
    }
    if (!g.hook_registered) {
      Kokkos::push_finalize_hook(ReleaseDeviceState);
      g.hook_registered = true;
    }
  }
  g.evolves = evolves;
  g.pmesh = ppack->pmesh;
  cd.bh_mass = g.m_g;
  cd.bh_spin = g.a_g;
  cd.rexcise = g.rexcise;
  cd.flux_excise_r = g.flux_excise_r;
}

//----------------------------------------------------------------------------------------
void AccumulateFaceSums(MeshBlockPack *pack, const BandFaceFld5D<Real> &uflx,
                        int ien, int idn, Real w_dt) {
  HoleState &g = State();
  if (!g.evolves || w_dt == 0.0) return;
  EnsureLayout(pack);
  const bool lat_enabled = pack->lat_active_mask_enabled;
  const int nwork = lat_enabled ? pack->lat_nactive_thispack : pack->nmb_thispack;
  g.pending = true;
  if (nwork <= 0) return;

  const bool lat_per_block_dt = pack->lat_per_block_timestep;
  const Real mesh_dt = pack->pmesh->dt;
  const Real w_rate = (lat_per_block_dt && mesh_dt != 0.0) ? w_dt/mesh_dt : 0.0;
  auto active_indices = pack->lat_active_indices.d_view;
  auto lat_step_dt = pack->lat_step_dt.d_view;
  auto &indcs = pack->pmesh->mb_indcs;
  const int is = indcs.is, js = indcs.js;
  const int nx1 = indcs.nx1, nx2 = indcs.nx2;
  auto size = pack->pmb->mb_size;
  auto faces = g.faces;
  auto off = g.face_offset;
  auto slots = g.slots_fluid;
  const auto f1 = uflx.x1f, f2 = uflx.x2f, f3 = uflx.x3f;
  const int ie_ = ien, id_ = idn;

  Kokkos::TeamPolicy<> policy(DevExeSpace(), nwork, Kokkos::AUTO);
  Kokkos::parallel_for("kerr_hole_face_sums", athenak_lw(policy),
  KOKKOS_LAMBDA(TeamMember_t tmember) {
    const int a = tmember.league_rank();
    const int m = lat_enabled ? active_indices(a) : a;
    const int fbeg = off(m), fend = off(m+1);
    if (fend <= fbeg) return;
    const Real scale = lat_per_block_dt ? w_rate*lat_step_dt(m) : w_dt;
    const Real x1min = size.d_view(m).x1min, x1max = size.d_view(m).x1max;
    const Real x2min = size.d_view(m).x2min, x2max = size.d_view(m).x2max;
    const Real dx1 = size.d_view(m).dx1, dx2 = size.d_view(m).dx2;
    const Real dx3 = size.d_view(m).dx3;

    // The flux component F_v . n dA of face f, n from the gas into the hole.
    auto flux_n = [&](const int f, const int v) {
      const int k = faces(f, 1), j = faces(f, 2), i = faces(f, 3);
      const int code = faces(f, 4);
      const int dir = code >> 1;
      const Real sgn = (code & 1) ? 1.0 : -1.0;
      if (dir == 0) return sgn*dx2*dx3*f1(m, v, k, j, i);
      if (dir == 1) return sgn*dx1*dx3*f2(m, v, k, j, i);
      return sgn*dx1*dx2*f3(m, v, k, j, i);
    };

    Real dm = 0.0;
    Kokkos::parallel_reduce(Kokkos::TeamThreadRange(tmember, fbeg, fend),
    [&](const int f, Real &sum) {
      sum -= flux_n(f, ie_) - flux_n(f, id_);
    }, dm);
    Real dj = 0.0;
    Kokkos::parallel_reduce(Kokkos::TeamThreadRange(tmember, fbeg, fend),
    [&](const int f, Real &sum) {
      const int j = faces(f, 2), i = faces(f, 3);
      const int dir = faces(f, 4) >> 1;
      const Real xf = (dir == 0) ? LeftEdgeX(i-is, nx1, x1min, x1max) :
                                   CellCenterX(i-is, nx1, x1min, x1max);
      const Real yf = (dir == 1) ? LeftEdgeX(j-js, nx2, x2min, x2max) :
                                   CellCenterX(j-js, nx2, x2min, x2max);
      sum += xf*flux_n(f, IM2) - yf*flux_n(f, IM1);
    }, dj);
    Kokkos::single(Kokkos::PerTeam(tmember), [&]() {
      slots(m, 0) += scale*dm;
      slots(m, 1) += scale*dj;
    });
  });
}

void AccumulateFaceSums(MeshBlockPack *pack, const DvceFaceFld5D<Real> &uflx,
                        int ien, int idn, Real w_dt) {
  if (!State().evolves || w_dt == 0.0) return;
  if (pack->pmhd != nullptr && uflx.x1f.data() == pack->pmhd->uflx.x1f.data()) {
    AccumulateFaceSums(pack, pack->pmhd->FluxBand(uflx), ien, idn, w_dt);
  } else if (pack->phydro != nullptr &&
             uflx.x1f.data() == pack->phydro->uflx.x1f.data()) {
    AccumulateFaceSums(pack, pack->phydro->FluxBand(uflx), ien, idn, w_dt);
  } else {
    Fatal(__FILE__, __LINE__,
          "kerr_hole::AccumulateFaceSums: the flux register is neither the MHD nor the "
          "hydro uflx, so its band origin is unknown; pass its FluxBand view.");
  }
}

//----------------------------------------------------------------------------------------
void Synchronize(Driver *pdrive) {
  HoleState &g = State();
  if (!g.evolves) return;
  Synchronize(pdrive, g.pmesh);
}

void Synchronize(Driver *pdrive, Mesh *pm) {
  HoleState &g = State();
  if (!g.evolves) return;
  MeshBlockPack *pmbp = pm->pmb_pack;
  if (pmbp->lat_active_mask_enabled) {
    Fatal(__FILE__, __LINE__,
          "kerr_hole::Synchronize ran with a LAT active set installed; it belongs at a "
          "global sync, after ClearHydroLAT, where every MeshBlock is at one time.");
  }
  EnsureLayout(pmbp);

  // reduce the face sums into the bookkept hole
  Real dm = 0.0, dj = 0.0;
  ReduceSlots(pmbp, dm, dj);
  g.m_true += dm;
  g.j_true += dj;
  CheckKerrBound(g.m_true, g.j_true, "state");

  // refresh rule: the metric lags the bookkept hole by at most 1e-5 M_g
  const Real a_true = g.j_true/g.m_true;
  if (std::abs(g.m_true - g.m_g) > kRefreshLag*g.m_g ||
      std::abs(a_true - g.a_g) > kRefreshLag*g.m_g) {
    Refresh(pdrive, pm, g.m_true, a_true);
    CheckKerrBound(g.m_g, g.m_g*g.a_g, "metric");
  }
}

//----------------------------------------------------------------------------------------
void StoreRestartKeys(ParameterInput *pin) {
  HoleState &g = State();
  if (!g.initialized || !g.evolves) return;
  // A checkpoint written between a stage and the sync would drop unreduced sums, so
  // reduce here as well.  Unconditionally: the restart dump is collective, while whether
  // a rank's slots were written is not (a LAT bin runs only on the ranks that own it).
  // Empty slots add an exact 0.
  EnsureLayout(g.pmesh->pmb_pack);
  Real dm = 0.0, dj = 0.0;
  ReduceSlots(g.pmesh->pmb_pack, dm, dj);
  g.m_true += dm;
  g.j_true += dj;
  CheckKerrBound(g.m_true, g.j_true, "state");
  pin->SetReal("coord", "bh_mass_live", g.m_g);
  pin->SetReal("coord", "bh_spin_live", g.a_g);
  pin->SetReal("coord", "bh_mass_true", g.m_true);
  pin->SetReal("coord", "bh_angmom_true", g.j_true);
  pin->SetReal("coord", "bh_rexcise_live", g.rexcise);
  pin->SetReal("coord", "bh_flux_excise_r_live", g.flux_excise_r);
}

//----------------------------------------------------------------------------------------
void HistoryValues(Real &bh_m, Real &bh_a) {
  const HoleState &g = State();
  bh_m = g.m_true;
  bh_a = g.j_true/(g.m_true*g.m_true);
}

}  // namespace kerr_hole
