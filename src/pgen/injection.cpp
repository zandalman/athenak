//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file injection.cpp
//! \brief Problem generator for a global TDE debris-stream injection simulation.
//!
//! Streams are injected near the self-intersection point and the collision-induced
//! outflow falls back onto the black hole.  Newtonian, adiabatic hydro in Cartesian
//! coordinates, with the black hole represented by the Tejeda & Rosswog (2013, TR13)
//! generalized Newtonian potential applied as a user source term.
//!
//! Units: G = M_BH = c = 1 (lengths in r_g, times in r_g/c).
//!
//! Current contents:
//!   - hydrostatic, zero-Bernoulli polytropic background atmosphere in the static TR13
//!     potential, normalized by <problem>/rho_atm at <problem>/r_atm
//!   - TR13 gravity source term (LAT-safe)
//!   - excision for r < <problem>/r_in, held at the atmosphere state at r_in
//!   - AMR: every <refined_regionN> is kept as a minimum level (they would otherwise be
//!     derefined under adaptive refinement), and inside the injection box
//!     <problem>/inj_x?min..inj_x?max blocks whose max density exceeds
//!     <problem>/amr_rho_min are refined to <problem>/amr_level
//!   - stream injection: <problem>/nstreams streams, each a mass/momentum/energy source
//!     uniform over a cylinder (radius H, length H, axis along the stream velocity) centred
//!     on (x, y, 0), injecting mdot with velocity (xdot, ydot, 0) and radiation
//!     temperature temp [K] (e = a_rad T^4; the gas is radiation dominated)
//!   - LAT factor cap: blocks at the injection cylinders and their neighbours step at
//!     factor 1
//!
//! References:
//!    Tejeda & Rosswog 2013, MNRAS 433, 1930 (TR13)

#include <math.h>

#if MPI_PARALLEL_ENABLED
#include <mpi.h>
#endif

#include <iostream>
#include <string>
#include <vector>

#include "athena.hpp"
#include "globals.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "mesh/mesh_refinement.hpp"
#include "coordinates/cell_locations.hpp"
#include "eos/eos.hpp"
#include "hydro/hydro.hpp"
#include "pgen/pgen.hpp"
#include "units/units.hpp"
#include "injection.hpp"

namespace {

// Problem parameters, read in UserProblem (also on restart) and used by the source terms.
struct InjectionParams {
  Real r_in;      // inner (excision) radius in r_g; must be > 2
  Real rho_atm;   // atmosphere density at r_atm
  Real r_atm;     // normalization radius of the atmosphere
  Real rho_exc;   // excision density   (atmosphere at r_in)
  Real eint_exc;  // excision internal energy density (atmosphere at r_in)
};
InjectionParams params;

// Axis-aligned box with a (physical, root = 0) refinement level.
struct RefBox {
  Real x1min, x1max, x2min, x2max, x3min, x3max;
  int level;
};
std::vector<RefBox> smr_boxes;  // <refined_regionN> blocks, kept as minimum levels
RefBox inj_box;                 // injection region; level = AMR target level
Real amr_rho_min;               // density threshold for AMR inside inj_box

// Injected streams.  Plain struct with a fixed-size array so it is captured by value.
constexpr int kMaxStreams = 4;
struct Stream {
  Real x, y;      // centre of the injection cylinder (z = 0)
  Real vx, vy;    // injection velocity
  Real nx, ny;    // unit vector along the velocity (cylinder axis)
  Real h;         // stream radius = cylinder radius = cylinder length
  Real q;         // mass injection rate per unit volume, mdot / V_cyl
  Real espec;     // specific internal energy of injected gas, a_rad T^4 / rho_s
};
struct Streams {
  int n;
  Stream s[kMaxStreams];
};
Streams streams;

KOKKOS_INLINE_FUNCTION
bool InStream(const Stream &s, Real x, Real y, Real z) {
  const Real dx = x - s.x, dy = y - s.y;
  const Real along = dx*s.nx + dy*s.ny;
  const Real perp2 = dx*dx + dy*dy + z*z - along*along;
  return fabs(along) <= 0.5*s.h && perp2 <= s.h*s.h;
}

void InjectionSources(Mesh *pm, const Real bdt);
void CapLATFactorsAtInjection(Mesh *pm, int max_factor, int *lat_factor_eachmb);
void ExcisionFixup(MeshBlockPack *pmbp, const Real time);
void InjectionRefinement(MeshBlockPack *pmbp);

//----------------------------------------------------------------------------------------
//! \fn AtmosphereDensity
//! \brief Zero-Bernoulli polytrope in the static TR13 potential.
//! Hydrostatic equilibrium with P = K rho^gamma gives enthalpy
//!   h = gamma/(gamma-1) P/rho = -Phi_s(r),
//! so rho(r) = rho_atm [Phi_s(r)/Phi_s(r_atm)]^n with n = 1/(gamma-1).  The profile is
//! frozen at r_in for r < r_in (that region will be excised).

KOKKOS_INLINE_FUNCTION
Real AtmosphereDensity(Real r, Real r_in, Real rho_atm, Real phi_atm, Real n) {
  const Real r_eff = fmax(r, r_in);
  return rho_atm*pow(TR13StaticPot(r_eff)/phi_atm, n);
}

}  // namespace

//----------------------------------------------------------------------------------------
//! \fn ProblemGenerator::UserProblem()
//! \brief Sets up the background atmosphere and enrolls the TR13 gravity source.

void ProblemGenerator::UserProblem(ParameterInput *pin, const bool restart) {
  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;

  // Read parameters (needed on restart too, for the source terms).
  params.r_in = pin->GetReal("problem", "r_in");
  params.rho_atm = pin->GetReal("problem", "rho_atm");
  params.r_atm = pin->GetReal("problem", "r_atm");

  // Injection region and AMR settings.
  inj_box.x1min = pin->GetReal("problem", "inj_x1min");
  inj_box.x1max = pin->GetReal("problem", "inj_x1max");
  inj_box.x2min = pin->GetReal("problem", "inj_x2min");
  inj_box.x2max = pin->GetReal("problem", "inj_x2max");
  inj_box.x3min = pin->GetReal("problem", "inj_x3min");
  inj_box.x3max = pin->GetReal("problem", "inj_x3max");
  inj_box.level = pin->GetInteger("problem", "amr_level");
  amr_rho_min = pin->GetReal("problem", "amr_rho_min");
  smr_boxes.clear();
  for (auto &b : pin->block) {
    if (b.block_name.compare(0, 14, "refined_region") != 0) continue;
    const std::string &n = b.block_name;
    smr_boxes.push_back({pin->GetReal(n, "x1min"), pin->GetReal(n, "x1max"),
                         pin->GetReal(n, "x2min"), pin->GetReal(n, "x2max"),
                         pin->GetReal(n, "x3min"), pin->GetReal(n, "x3max"),
                         pin->GetInteger(n, "level")});
  }
  if (pmy_mesh_->adaptive) user_ref_func = InjectionRefinement;

  // Atmosphere: P = kappa rho^gamma with kappa from h(r_atm) = -Phi_s(r_atm),
  // i.e. kappa = -Phi_s(r_atm) (gamma-1)/gamma / rho_atm^(gamma-1).
  const Real gamma = pmbp->phydro->peos->eos_data.gamma;
  const Real gm1 = gamma - 1.0;
  const Real n = 1.0/gm1;
  const Real r_in = params.r_in;
  const Real rho_atm = params.rho_atm;
  const Real phi_atm = TR13StaticPot(params.r_atm);
  const Real kappa = -phi_atm*gm1/gamma/pow(rho_atm, gm1);
  params.rho_exc = AtmosphereDensity(r_in, r_in, rho_atm, phi_atm, n);
  params.eint_exc = kappa*pow(params.rho_exc, gamma)/gm1;

  // Streams.  The source density q = mdot/V uses the discrete volume of the cells whose
  // centres lie in the cylinder, so exactly mdot is injected.  The mesh there is static
  // (SMR boxes around the injection points), so V is fixed for the run.
  auto &indcs = pmy_mesh_->mb_indcs;
  const int is = indcs.is, ie = indcs.ie;
  const int js = indcs.js, je = indcs.je;
  const int ks = indcs.ks, ke = indcs.ke;
  auto &size = pmbp->pmb->mb_size;
  const int nmb = pmbp->nmb_thispack;
  const int nkji = indcs.nx3*indcs.nx2*indcs.nx1, nji = indcs.nx2*indcs.nx1;
  streams.n = pin->GetInteger("problem", "nstreams");
  for (int ns = 0; ns < streams.n; ++ns) {
    const std::string id = std::to_string(ns + 1);
    Stream &s = streams.s[ns];
    s.x = pin->GetReal("problem", "x" + id);
    s.y = pin->GetReal("problem", "y" + id);
    s.vx = pin->GetReal("problem", "xdot" + id);
    s.vy = pin->GetReal("problem", "ydot" + id);
    s.h = pin->GetReal("problem", "H" + id);
    const Real mdot = pin->GetReal("problem", "mdot" + id);
    const Real temp = pin->GetReal("problem", "temp" + id);
    const Real v = sqrt(s.vx*s.vx + s.vy*s.vy);
    s.nx = s.vx/v;
    s.ny = s.vy/v;
    // Radiation-dominated gas: internal energy density e = a_rad T^4 (= 3P for gamma=4/3),
    // per unit mass of the stream density rho_s = mdot/(pi H^2 v).
    const Real rho_s = mdot/(M_PI*s.h*s.h*v);
    const Real e_rad = units::Units::rad_constant_cgs*pow(temp, 4)/
                       pmbp->punit->pressure_cgs();
    s.espec = e_rad/rho_s;

    const Stream sl = s;
    Real vol = 0.0;
    Kokkos::parallel_reduce("inj_stream_vol",
        Kokkos::RangePolicy<>(DevExeSpace(), 0, nmb*nkji),
    KOKKOS_LAMBDA(const int idx, Real &sum) {
      const int m = idx/nkji;
      const int k = (idx - m*nkji)/nji;
      const int j = (idx - m*nkji - k*nji)/indcs.nx1;
      const int i = idx - m*nkji - k*nji - j*indcs.nx1;
      const Real x = CellCenterX(i, indcs.nx1, size.d_view(m).x1min, size.d_view(m).x1max);
      const Real y = CellCenterX(j, indcs.nx2, size.d_view(m).x2min, size.d_view(m).x2max);
      const Real z = CellCenterX(k, indcs.nx3, size.d_view(m).x3min, size.d_view(m).x3max);
      if (InStream(sl, x, y, z)) {
        sum += size.d_view(m).dx1*size.d_view(m).dx2*size.d_view(m).dx3;
      }
    }, vol);
#if MPI_PARALLEL_ENABLED
    MPI_Allreduce(MPI_IN_PLACE, &vol, 1, MPI_ATHENA_REAL, MPI_SUM, MPI_COMM_WORLD);
#endif
    s.q = mdot/vol;
    if (global_variable::my_rank == 0) {
      std::cout << "Stream " << id << ": V_cyl = " << vol << " (analytic "
                << M_PI*s.h*s.h*s.h << "), rho = mdot/(pi H^2 v) = " << rho_s
                << ", P = a T^4/3 = " << e_rad/3.0 << std::endl;
    }
  }

  // Enroll the source terms (TR13 gravity + stream injection).  They obey the LAT
  // contract (active blocks only, per-block dt, no MPI collectives, time-independent),
  // see docs/notes.md.
  user_srcs = true;
  user_srcs_func = InjectionSources;
  user_srcs_lat_safe = true;
  user_srcs_lat_union_safe = true;
  // LAT fixes a block's step for a whole window from its state at the window start, but
  // the injection source creates fast gas in blocks that held static background: keep the
  // injection blocks and their neighbours at factor 1.
  user_hydro_lat_factor_cap_func = CapLATFactorsAtInjection;
  // Excision: reset r < r_in every stage (idempotent, so safe under the union predictor).
  user_hydro_state_fixup_func = ExcisionFixup;
  user_hydro_state_fixup_lat_union_safe = true;
  // Same-level LAT ghosts: use the linear time chord instead of the SSPRK2 dense output,
  // as the built-in gravity sources do (docs/wiki/Local-Adaptive-Time-Stepping.md).
  pmbp->phydro->ConfigureLATDenseOutput(false);

  if (restart) return;

  // Initialize the hydrostatic atmosphere ------------------------------------------------
  auto &w0 = pmbp->phydro->w0;

  par_for("pgen_injection_atm", DevExeSpace(), 0, (pmbp->nmb_thispack-1),
          ks, ke, js, je, is, ie,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    const Real x = CellCenterX(i-is, indcs.nx1, size.d_view(m).x1min,
                               size.d_view(m).x1max);
    const Real y = CellCenterX(j-js, indcs.nx2, size.d_view(m).x2min,
                               size.d_view(m).x2max);
    const Real z = CellCenterX(k-ks, indcs.nx3, size.d_view(m).x3min,
                               size.d_view(m).x3max);
    const Real r = sqrt(x*x + y*y + z*z);
    const Real rho = AtmosphereDensity(r, r_in, rho_atm, phi_atm, n);
    const Real pres = kappa*pow(rho, gamma);
    w0(m,IDN,k,j,i) = rho;
    w0(m,IVX,k,j,i) = 0.0;
    w0(m,IVY,k,j,i) = 0.0;
    w0(m,IVZ,k,j,i) = 0.0;
    w0(m,IEN,k,j,i) = pres/gm1;
  });

  pmbp->phydro->peos->PrimToCons(w0, pmbp->phydro->u0, is, ie, js, je, ks, ke);
}

namespace {

//----------------------------------------------------------------------------------------
//! \fn InjectionSources
//! \brief User source terms, evaluated on the stage primitives:
//!  - TR13 gravity: d(rho v)/dt = rho a,  dE/dt = rho v.a, with a = TR13Accel(x, v).
//!    Work only, so a dual-energy internal-energy field is untouched.  Skipped for r < r_in.
//!  - stream injection: in each stream's cylinder, d(rho)/dt = q, d(rho v)/dt = q v_s,
//!    dE/dt = q (v_s^2/2 + espec), with q = mdot/V_cyl.
//!
//! LAT: loops over the active (due) blocks only and uses each block's own step, following
//! TDEExternalGravitySource in tde_external.cpp.

void InjectionSources(Mesh *pm, const Real bdt) {
  MeshBlockPack *pmbp = pm->pmb_pack;
  auto &indcs = pm->mb_indcs;
  const int is = indcs.is, ie = indcs.ie;
  const int js = indcs.js, je = indcs.je;
  const int ks = indcs.ks, ke = indcs.ke;
  const int nmb = pmbp->nmb_thispack;
  auto &size = pmbp->pmb->mb_size;

  // LAT bookkeeping: active blocks and per-block step.
  const bool lat_enabled = pmbp->lat_active_mask_enabled;
  const bool lat_per_block_dt = pmbp->lat_per_block_timestep;
  const Real mesh_dt = pm->dt;
  const Real stage_weight = (lat_per_block_dt && mesh_dt > 0.0) ? (bdt/mesh_dt) : 0.0;
  auto lat_step_dt = pmbp->lat_step_dt.d_view;
  auto active_indices = pmbp->lat_active_indices.d_view;
  const int nwork1 = lat_enabled ? (pmbp->lat_nactive_thispack - 1) : (nmb - 1);
  if (nwork1 < 0) return;

  auto &w0 = pmbp->phydro->w0;
  auto &u0 = pmbp->phydro->u0;
  const Real r_in2 = params.r_in*params.r_in;
  const Streams st = streams;

  par_for("injection_srcs", DevExeSpace(), 0, nwork1, ks, ke, js, je, is, ie,
  KOKKOS_LAMBDA(int a, int k, int j, int i) {
    const int m = lat_enabled ? active_indices(a) : a;
    const Real block_bdt = lat_per_block_dt ? stage_weight*lat_step_dt(m) : bdt;

    const Real x = CellCenterX(i-is, indcs.nx1, size.d_view(m).x1min,
                               size.d_view(m).x1max);
    const Real y = CellCenterX(j-js, indcs.nx2, size.d_view(m).x2min,
                               size.d_view(m).x2max);
    const Real z = CellCenterX(k-ks, indcs.nx3, size.d_view(m).x3min,
                               size.d_view(m).x3max);
    if (x*x + y*y + z*z < r_in2) return;

    const Real rho = w0(m,IDN,k,j,i);
    const Real vx = w0(m,IVX,k,j,i);
    const Real vy = w0(m,IVY,k,j,i);
    const Real vz = w0(m,IVZ,k,j,i);
    Real ax, ay, az;
    TR13Accel(x, y, z, vx, vy, vz, ax, ay, az);

    const Real src = block_bdt*rho;
    u0(m,IM1,k,j,i) += src*ax;
    u0(m,IM2,k,j,i) += src*ay;
    u0(m,IM3,k,j,i) += src*az;
    u0(m,IEN,k,j,i) += src*(vx*ax + vy*ay + vz*az);

    for (int n = 0; n < st.n; ++n) {
      const Stream &s = st.s[n];
      if (!InStream(s, x, y, z)) continue;
      const Real dm = block_bdt*s.q;
      u0(m,IDN,k,j,i) += dm;
      u0(m,IM1,k,j,i) += dm*s.vx;
      u0(m,IM2,k,j,i) += dm*s.vy;
      u0(m,IEN,k,j,i) += dm*(0.5*(s.vx*s.vx + s.vy*s.vy) + s.espec);
    }
  });
}

//----------------------------------------------------------------------------------------
//! \fn CapLATFactorsAtInjection
//! \brief LAT factor cap: every block whose bounding box overlaps an injection cylinder
//! (bounded by its circumscribed sphere, radius sqrt(H^2 + (H/2)^2)), and all of its
//! neighbours, take factor 1.  lat_factor_eachmb is indexed by global ID; each rank caps
//! its own blocks and the result is MIN-reduced (this hook runs on the synchronized path,
//! so the collective is allowed).

void CapLATFactorsAtInjection(Mesh *pm, int max_factor, int *lat_factor_eachmb) {
  MeshBlockPack *pmbp = pm->pmb_pack;
  auto &size = pmbp->pmb->mb_size;
  auto &mb_gid = pmbp->pmb->mb_gid;
  auto &nghbr = pmbp->pmb->nghbr;
  const int nnghbr = pmbp->pmb->nnghbr;
  const int nmb_total = pm->nmb_total;
  auto cap = [&](const int gid) {
    if (gid >= 0 && gid < nmb_total) lat_factor_eachmb[gid] = 1;
  };

  for (int m = 0; m < pmbp->nmb_thispack; ++m) {
    const RegionSize &b = size.h_view(m);
    for (int ns = 0; ns < streams.n; ++ns) {
      const Stream &s = streams.s[ns];
      const Real rad = sqrt(1.25)*s.h;
      if (b.x1max > s.x - rad && b.x1min < s.x + rad &&
          b.x2max > s.y - rad && b.x2min < s.y + rad &&
          b.x3max > -rad && b.x3min < rad) {
        cap(mb_gid.h_view(m));
        for (int n = 0; n < nnghbr; ++n) cap(nghbr.h_view(m,n).gid);
      }
    }
  }

#if MPI_PARALLEL_ENABLED
  if (global_variable::nranks > 1) {
    MPI_Allreduce(MPI_IN_PLACE, lat_factor_eachmb, nmb_total, MPI_INT, MPI_MIN,
                  MPI_COMM_WORLD);
  }
#endif
}

//----------------------------------------------------------------------------------------
//! \fn ExcisionFixup
//! \brief Excision: owned cells with r < r_in are reset to the background atmosphere
//! state at r_in (rho_exc, eint_exc) with v = 0, so the excision pressure-supports the
//! hydrostatic atmosphere.  Acts on the conserved u0 before boundary exchange and primitive
//! recovery.  Under LAT only the active (due) blocks are touched.

void ExcisionFixup(MeshBlockPack *pmbp, const Real time) {
  auto &indcs = pmbp->pmesh->mb_indcs;
  const int is = indcs.is, ie = indcs.ie;
  const int js = indcs.js, je = indcs.je;
  const int ks = indcs.ks, ke = indcs.ke;
  auto &size = pmbp->pmb->mb_size;

  const bool lat_enabled = pmbp->lat_active_mask_enabled;
  auto active_indices = pmbp->lat_active_indices.d_view;
  const int nwork1 = lat_enabled ? (pmbp->lat_nactive_thispack - 1)
                                 : (pmbp->nmb_thispack - 1);
  if (nwork1 < 0) return;

  auto &u0 = pmbp->phydro->u0;
  const Real rho_exc = params.rho_exc;
  const Real eint_exc = params.eint_exc;
  const Real r_in2 = params.r_in*params.r_in;

  par_for("injection_excision", DevExeSpace(), 0, nwork1, ks, ke, js, je, is, ie,
  KOKKOS_LAMBDA(int a, int k, int j, int i) {
    const int m = lat_enabled ? active_indices(a) : a;
    const Real x = CellCenterX(i-is, indcs.nx1, size.d_view(m).x1min,
                               size.d_view(m).x1max);
    const Real y = CellCenterX(j-js, indcs.nx2, size.d_view(m).x2min,
                               size.d_view(m).x2max);
    const Real z = CellCenterX(k-ks, indcs.nx3, size.d_view(m).x3min,
                               size.d_view(m).x3max);
    if (x*x + y*y + z*z >= r_in2) return;
    u0(m,IDN,k,j,i) = rho_exc;
    u0(m,IM1,k,j,i) = 0.0;
    u0(m,IM2,k,j,i) = 0.0;
    u0(m,IM3,k,j,i) = 0.0;
    u0(m,IEN,k,j,i) = eint_exc;
  });
}

//----------------------------------------------------------------------------------------
//! \fn InjectionRefinement
//! \brief User AMR criterion (<amr_criterion> method = user).  The target level of a
//! block is the highest of
//!   - the level of every <refined_regionN> box it overlaps (static minimum levels), and
//!   - inj_box.level, if it overlaps the injection box and its max density > amr_rho_min.
//! Blocks below the target are flagged to refine, above it to derefine (the mesh only
//! derefines when all siblings agree, and enforces 2:1 level balance itself).

void InjectionRefinement(MeshBlockPack *pmbp) {
  Mesh *pm = pmbp->pmesh;
  auto &refine_flag = pm->pmr->refine_flag;
  const int mbs = pm->gids_eachrank[global_variable::my_rank];
  const int nmb = pmbp->nmb_thispack;
  auto &indcs = pm->mb_indcs;
  const int is = indcs.is, js = indcs.js, ks = indcs.ks;
  const int nx1 = indcs.nx1, nx2 = indcs.nx2, nx3 = indcs.nx3;
  const int nkji = nx3*nx2*nx1, nji = nx2*nx1;

  // max density over the owned cells of each block
  DualArray1D<Real> rho_max("inj_amr_rho_max", nmb);
  auto rmax = rho_max.d_view;
  auto &w0 = pmbp->phydro->w0;
  par_for_outer("inj_amr_rho_max", DevExeSpace(), 0, 0, 0, (nmb-1),
  KOKKOS_LAMBDA(TeamMember_t tmember, const int m) {
    Real team_max = 0.0;
    Kokkos::parallel_reduce(Kokkos::TeamThreadRange(tmember, nkji),
    [&](const int idx, Real &rmx) {
      const int k = idx/nji;
      const int j = (idx - k*nji)/nx1;
      const int i = idx - k*nji - j*nx1;
      rmx = fmax(rmx, w0(m,IDN,k+ks,j+js,i+is));
    }, Kokkos::Max<Real>(team_max));
    Kokkos::single(Kokkos::PerTeam(tmember), [&]() { rmax(m) = team_max; });
  });
  rho_max.template modify<DevExeSpace>();
  rho_max.template sync<HostMemSpace>();

  auto &size = pmbp->pmb->mb_size;
  auto &lev = pmbp->pmb->mb_lev;
  auto overlaps = [](const RegionSize &s, const RefBox &b) {
    return s.x1max > b.x1min && s.x1min < b.x1max &&
           s.x2max > b.x2min && s.x2min < b.x2max &&
           s.x3max > b.x3min && s.x3min < b.x3max;
  };
  for (int m = 0; m < nmb; ++m) {
    const RegionSize &s = size.h_view(m);
    int target = 0;
    for (const auto &b : smr_boxes) {
      if (b.level > target && overlaps(s, b)) target = b.level;
    }
    if (inj_box.level > target && overlaps(s, inj_box) && rho_max.h_view(m) > amr_rho_min) {
      target = inj_box.level;
    }
    const int level = lev.h_view(m) - pm->root_level;
    refine_flag.h_view(m + mbs) = (level < target) ? 1 : ((level > target) ? -1 : 0);
  }
  refine_flag.template modify<HostMemSpace>();
  refine_flag.template sync<DevExeSpace>();
}

}  // namespace
