//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file gr_selfgravity.cpp
//! \brief Potential service of the weak-field self-gravity of the GR paths; see
//! gr_selfgravity.hpp.

// C++ headers
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>

// Athena++ headers
#include "athena.hpp"
#include "coordinates/cell_locations.hpp"
#include "coordinates/coordinates.hpp"
#include "coordinates/kerr_hole.hpp"
#include "driver/driver.hpp"
#include "driver/lat_weights.hpp"
#include "dyn_grmhd/dyn_grmhd.hpp"
#include "eos/eos.hpp"
#include "globals.hpp"
#include "gravity/gr_selfgravity.hpp"
#include "gravity/gr_selfgravity_fluid.hpp"
#include "gravity/gravity.hpp"
#include "hydro/hydro.hpp"
#include "mesh/mesh.hpp"
#include "mesh/meshblock_pack.hpp"
#include "mhd/mhd.hpp"
#include "parameter_input.hpp"
#include "units/units.hpp"
#include "utils/gravity_weight.hpp"

#if MPI_PARALLEL_ENABLED
#include <mpi.h>
#endif

namespace grsg {

namespace {

[[noreturn]] void Fatal(const char *file, int line, const char *msg) {
  std::cout << "### FATAL ERROR in " << file << " at line " << line << std::endl
            << msg << std::endl;
  std::exit(EXIT_FAILURE);
}

}  // namespace

//----------------------------------------------------------------------------------------
//! \fn Real GravConstant

Real GravConstant(const MeshBlockPack *pmbp) {
  return (pmbp != nullptr && pmbp->punit != nullptr) ? pmbp->punit->grav_constant() : 1.0;
}

//----------------------------------------------------------------------------------------
//! \fn bool RelativisticPack, Enabled, HoleEvolves

bool RelativisticPack(const MeshBlockPack *pmbp) {
  return pmbp != nullptr && pmbp->pcoord != nullptr &&
         (pmbp->pcoord->is_general_relativistic ||
          pmbp->pcoord->is_dynamical_relativistic);
}

bool Enabled(const MeshBlockPack *pmbp) {
  return pmbp != nullptr && pmbp->pgrav != nullptr && pmbp->pgrav->relativistic;
}

bool HoleEvolves(const MeshBlockPack *pmbp) {
  return Enabled(pmbp) && pmbp->pcoord->is_general_relativistic &&
         !pmbp->pcoord->is_dynamical_relativistic &&
         !pmbp->pcoord->coord_data.is_minkowski;
}

//----------------------------------------------------------------------------------------
//! \fn void ValidateDeck
//! \brief Deck guards of the GR branch (design section 10).  The pack's MeshBlock size is
//! known: the guard runs from the Gravity constructor.

void ValidateDeck(const MeshBlockPack *pmbp, ParameterInput *pin) {
  if (pin->GetOrAddBoolean("gravity", "lat_time_centered_work", false)) {
    Fatal(__FILE__, __LINE__,
          "<gravity>/lat_time_centered_work is the Newtonian work-refund ledger of the "
          "mass-flux gravity source and does not exist in GR: the GR force is pointwise "
          "and cell-centred, with no gravity reflux.");
  }
  const Real four_pi_G = kFourPi*GravConstant(pmbp);
  if (pin->DoesParameterExist("gravity", "four_pi_G")) {
    const Real deck = pin->GetReal("gravity", "four_pi_G");
    if (!(std::abs(deck - four_pi_G) <= 1.0e-12*four_pi_G)) {
      Fatal(__FILE__, __LINE__,
            "<gravity>/four_pi_G must be 4 pi G (or absent) in GR, with G = G_cgs rho_u "
            "t_u^2 the gravitational constant of the <units> code units.");
    }
  } else {
    pin->SetReal("gravity", "four_pi_G", four_pi_G);
  }
  const auto &indcs = pmbp->pmesh->mb_indcs;
  if (indcs.nx1 != indcs.nx2 || indcs.nx1 != indcs.nx3 ||
      (indcs.nx1 & (indcs.nx1 - 1)) != 0) {
    Fatal(__FILE__, __LINE__,
          "GR self-gravity runs on the multigrid solver, which needs cubic MeshBlocks "
          "with a power-of-two number of cells (a 40^3 torus deck becomes 32^3 or 64^3): "
          "an odd factor would leave the octet coarsening with half cells.");
  }
  const CoordData &cd = pmbp->pcoord->coord_data;
  if (!cd.is_minkowski && !cd.bh_excise) {
    Fatal(__FILE__, __LINE__,
          "GR self-gravity with a hole requires <coord>/excise = true: without "
          "excision the source inside the horizon is undefined.");
  }
}

//----------------------------------------------------------------------------------------
//! \fn PhiView MakePhiView

PhiView MakePhiView(const MeshBlockPack *pmbp) {
  PhiView pv;
  pv.a_frame[0] = pv.a_frame[1] = pv.a_frame[2] = 0.0;
  pv.rho_min = 0.0;
  pv.dfloor = 0.0;
  pv.active = false;
  if (!Enabled(pmbp)) return pv;
  const gravity::Gravity *pg = pmbp->pgrav;
  if (!pg->phi_valid || !pg->self_phi_time_valid) {
    Fatal(__FILE__, __LINE__,
          "GR self-gravity force requested without a valid numerical potential. "
          "Run the multigrid solve before the stage sources.");
  }
  pv.phi = pg->phi;
  for (int d = 0; d < 3; ++d) pv.a_frame[d] = pg->a_frame[d];
  pv.rho_min = pg->pmgd->RhoGravMin();
  pv.dfloor = pg->pmgd->RhoGravFloor();
  pv.active = true;
  return pv;
}

//----------------------------------------------------------------------------------------
//! \fn void AssembleSource
//! \brief rho_eff of Sect. 3.2: hooks, then the weight of the rest-mass density and the
//! excision mask.

void AssembleSource(MeshBlockPack *pmbp, DvceArray5D<Real> &rho_eff, const Real rho_min,
                    const Real rho_floor, const Real mask_radius,
                    const Real mask_origin[3]) {
  Kokkos::deep_copy(rho_eff, 0.0);
  AddGravitatingDensityFixed(pmbp, rho_eff);

  // The weight is the ramp of the rest-mass density, never of rho_eff: in a magnetically
  // dominated cell rho_eff ~ b^2 >> rho and the ramp would misfire.  The force adapters
  // apply the same weight (PhiView::rho_min, dfloor).
  const auto &indcs = pmbp->pmesh->mb_indcs;
  const int is = indcs.is, ie = indcs.ie, js = indcs.js, je = indcs.je;
  const int ks = indcs.ks, ke = indcs.ke;
  const int nmb = pmbp->nmb_thispack;
  const int nx1 = indcs.nx1, nx2 = indcs.nx2, nx3 = indcs.nx3;
  const DvceArray5D<Real> w0 = (pmbp->phydro != nullptr) ? pmbp->phydro->w0
                                                          : pmbp->pmhd->w0;
  const bool excise = pmbp->pcoord->coord_data.bh_excise;
  const DvceArray4D<bool> excision = pmbp->pcoord->excision_floor;
  const auto size = pmbp->pmb->mb_size.d_view;
  const Real rmask2 = mask_radius*mask_radius;
  const bool masked_radius = mask_radius > 0.0;
  const Real ox = mask_origin[0], oy = mask_origin[1], oz = mask_origin[2];
  DvceArray5D<Real> rho = rho_eff;
  par_for("grsg_source_weight", DevExeSpace(), 0, nmb-1, ks, ke, js, je, is, ie,
  KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
    Real w = gravity_weight::Weight(w0(m,IDN,k,j,i), rho_floor, rho_min);
    if (excise && excision(m,k,j,i)) w = 0.0;
    if (masked_radius) {
      const Real x = CellCenterX(i-is, nx1, size(m).x1min, size(m).x1max);
      const Real y = CellCenterX(j-js, nx2, size(m).x2min, size(m).x2max);
      const Real z = CellCenterX(k-ks, nx3, size(m).x3min, size(m).x3max);
      if (SQR(x-ox) + SQR(y-oy) + SQR(z-oz) > rmask2) w = 0.0;
    }
    rho(m,0,k,j,i) *= w;
  });
}

//----------------------------------------------------------------------------------------
//! \fn void ReflexAcceleration

void ReflexAcceleration(const MeshBlockPack *pmbp, const DvceArray5D<Real> &rho_eff,
                        Real a_frame[3]) {
  a_frame[0] = a_frame[1] = a_frame[2] = 0.0;
  if (!HoleEvolves(pmbp)) return;
  const auto &indcs = pmbp->pmesh->mb_indcs;
  const int nmb = pmbp->nmb_thispack;
  const int nx1 = indcs.nx1, nx2 = indcs.nx2, nx3 = indcs.nx3;
  const int is = indcs.is, js = indcs.js, ks = indcs.ks;
  Real total[3] = {0.0, 0.0, 0.0};
  if (nmb > 0) {
    const auto size = pmbp->pmb->mb_size.d_view;
    const DvceArray5D<Real> rho = rho_eff;
    DvceArray3D<Real> rows("grsg_reflex_rows", nmb, nx3, 3);
    DvceArray2D<Real> blocks("grsg_reflex_blocks", nmb, 3);
    // One row (m, k) per thread, summed in index order, then the rows of a block in k
    // order: the result does not depend on the scheduling.
    par_for("grsg_reflex_rows", DevExeSpace(), 0, nmb-1, 0, nx3-1,
    KOKKOS_LAMBDA(const int m, const int kk) {
      const Real dx = size(m).dx1;
      const Real dy = size(m).dx2;
      const Real dz = size(m).dx3;
      const Real vol = dx*dy*dz;
      const int k = ks + kk;
      const Real z = CellCenterX(kk, nx3, size(m).x3min, size(m).x3max);
      Real sx = 0.0, sy = 0.0, sz = 0.0;
      for (int jj = 0; jj < nx2; ++jj) {
        const Real y = CellCenterX(jj, nx2, size(m).x2min, size(m).x2max);
        for (int ii = 0; ii < nx1; ++ii) {
          const Real x = CellCenterX(ii, nx1, size(m).x1min,
                                     size(m).x1max);
          const Real s = rho(m,0,k,js+jj,is+ii);
          const Real r2 = x*x + y*y + z*z;
          if (s != 0.0 && r2 > 0.0) {
            const Real f = s*vol/(r2*sqrt(r2));
            sx += f*x;
            sy += f*y;
            sz += f*z;
          }
        }
      }
      rows(m,kk,0) = sx;
      rows(m,kk,1) = sy;
      rows(m,kk,2) = sz;
    });
    par_for("grsg_reflex_blocks", DevExeSpace(), 0, nmb-1, 0, 2,
    KOKKOS_LAMBDA(const int m, const int c) {
      Real sum = 0.0;
      for (int kk = 0; kk < nx3; ++kk) sum += rows(m,kk,c);
      blocks(m,c) = sum;
    });
    auto blocks_h = Kokkos::create_mirror_view(blocks);
    Kokkos::deep_copy(blocks_h, blocks);
    for (int m = 0; m < nmb; ++m) {
      for (int c = 0; c < 3; ++c) total[c] += blocks_h(m,c);
    }
  }
#if MPI_PARALLEL_ENABLED
  MPI_Allreduce(MPI_IN_PLACE, total, 3, MPI_ATHENA_REAL, MPI_SUM, MPI_COMM_WORLD);
#endif
  const Real g = GravConstant(pmbp);
  for (int c = 0; c < 3; ++c) a_frame[c] = g*total[c];
}

//----------------------------------------------------------------------------------------
//! \fn void WarnIfStrongField

void WarnIfStrongField(gravity::Gravity *pg) {
  if (pg->phi_validity_warned) return;
  const MeshBlockPack *pmbp = pg->pmy_pack;
  const auto &indcs = pmbp->pmesh->mb_indcs;
  const int nmb = pmbp->nmb_thispack;
  Real phimin = std::numeric_limits<Real>::max();
  if (nmb > 0) {
    const DvceArray5D<Real> phi = pg->phi;
    const int is = indcs.is, ie = indcs.ie, js = indcs.js, je = indcs.je;
    const int ks = indcs.ks, ke = indcs.ke;
    Kokkos::parallel_reduce("grsg_phi_min",
        Kokkos::MDRangePolicy<DevExeSpace, Kokkos::Rank<4>>(
            {0, ks, js, is}, {nmb, ke+1, je+1, ie+1}),
    KOKKOS_LAMBDA(const int m, const int k, const int j, const int i, Real &lmin) {
      lmin = (phi(m,0,k,j,i) < lmin) ? phi(m,0,k,j,i) : lmin;
    }, Kokkos::Min<Real>(phimin));
  }
#if MPI_PARALLEL_ENABLED
  MPI_Allreduce(MPI_IN_PLACE, &phimin, 1, MPI_ATHENA_REAL, MPI_MIN, MPI_COMM_WORLD);
#endif
  if (phimin < -0.1) {
    pg->phi_validity_warned = true;
    if (global_variable::my_rank == 0) {
      std::cout << "### WARNING in GR self-gravity: min Phi = " << phimin
                << " < -0.1, outside the weak-field range |Phi| << 1 of the model "
                << "(errors are O(v^2 Phi)); printed once." << std::endl;
    }
  }
}

}  // namespace grsg
