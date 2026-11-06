#ifndef DRIVER_LAT_WEIGHTS_HPP_
#define DRIVER_LAT_WEIGHTS_HPP_
//========================================================================================
// AthenaK astrophysical fluid dynamics and numerical relativity code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file lat_weights.hpp
//! \brief the pure LAT flux-bookkeeping helpers and predicates shared by every physics
//! module.
//!
//! These used to be file-static copies: LATFinalFluxWeight in EIGHT byte-comparable
//! copies (hydro_tasks.cpp, hydro_update.cpp, mhd_tasks.cpp) and LATFluxFaceCount/LATFluxFaceNeighborIndex in four.  They
//! HAD already drifted -- the two hydro copies carried an rk3 branch the other six did
//! not -- and the failure mode of that drift is a silent 50 % conservation error at every
//! coarse/fine face, so they are hoisted here (review A2/F19).
//!
//! The rk3 branch is kept in the single surviving definition and is unreachable in
//! practice: Driver::Execute refuses time/lat for any integrator that is not rk1 or
//! rk2-equivalent (driver.cpp:673-679), and these weights are consumed only on LAT paths.
//! Keeping it means no former copy loses a case.

#include "athena.hpp"
#include "driver/driver.hpp"
#include "mesh/mesh.hpp"

namespace lat {

//----------------------------------------------------------------------------------------
//! \fn Real lat::FinalFluxWeight
//! \brief per-stage weight of the final flux in the window integral.  imex2 shares rk2's
//! EXPLICIT tableau (gam0={1,0.5}, gam1={0,0.5}, beta={1,0.5}), so the weight is 0.5 per
//! stage; the beta[stage-1] fallback would integrate 1.5*dt per window instead of dt, a
//! 50 % over-correction.

inline Real FinalFluxWeight(Driver *pdrive, int stage) {
  if (pdrive->integrator == "rk1") return 1.0;
  if (pdrive->integrator_rk2_equiv) return 0.5;
  if (pdrive->integrator == "rk3") {
    return (stage < 3) ? (1.0/6.0) : (2.0/3.0);
  }
  return pdrive->beta[stage-1];
}

//----------------------------------------------------------------------------------------
//! \fn int lat::FluxFaceCount / lat::FluxFaceNeighborIndex
//! \brief the neighbour slots that carry a face (x1: 0-7, x2: 8-15, x3: 24-31), packed
//! into a contiguous team index.  Pure functions of nnghbr; device callable.

KOKKOS_INLINE_FUNCTION
int FluxFaceCount(const int nnghbr) {
  int count = (nnghbr < 16) ? nnghbr : 16;
  if (nnghbr > 24) count += ((nnghbr - 24) < 8) ? (nnghbr - 24) : 8;
  return count;
}

KOKKOS_INLINE_FUNCTION
int FluxFaceNeighborIndex(const int slot, const int nnghbr) {
  const int first_count = (nnghbr < 16) ? nnghbr : 16;
  return (slot < first_count) ? slot : 24 + (slot - first_count);
}

//----------------------------------------------------------------------------------------
//! \fn bool lat::FluxExchangeActive
//! \brief Does the flux half of a module's boundary exchange have anything to carry?
//! Coarse/fine faces (multilevel) and, under LAT on a uniform grid, same-level faces
//! between blocks in different bins.  The same-level pack/unpack is driven by the mesh
//! flag alone (flux_correct_cc.cpp, iflux_same / iflxs_ndat), so gating the send/recv
//! pairs on multilevel alone would leave a uniform-grid LAT run with the accumulators
//! armed and the exchange never happening: a silent conservation error at every
//! mixed-cadence same-level face.  Both flags are mesh-wide, so the predicate is
//! rank-uniform, which every gate PAIR sharing it requires (MPI requests posted by one
//! half go unmatched in the other otherwise).

inline bool FluxExchangeActive(const MeshBlockPack *pp) {
  return pp->pmesh->multilevel ||
         (pp->lat_active_mask_enabled && pp->pmesh->hydro_lat_same_level);
}

//----------------------------------------------------------------------------------------
//! \fn bool lat::SkipUnionStage1StateExchange
//! \brief The union stage-1 predictor advances every due bin in one pass, so no receiver
//! could consume a stage-1 state publication: each block's endpoint is at its own
//! t + fine_dt*factor(m).  The corrector replaces that exchange with
//! Driver::RefreshHydroLATBoundaries at each factor's own target time, which publishes
//! the predictor endpoint.  Only the state half is skipped: the flux half stays armed,
//! as delayed refluxing accumulates during the predictor.

inline bool SkipUnionStage1StateExchange(const Driver *pdrive, const int stage) {
  return pdrive != nullptr && pdrive->hydro_lat_union_stage1_active && stage == 1;
}

//----------------------------------------------------------------------------------------
//! \fn bool lat::RefluxNeeded
//! \brief Is there a mixed-cadence topology, i.e. anything for the delayed LAT reflux to
//! correct: coarse/fine faces, or same-level bins.  The storage predicate of every
//! module's reflux registers (with the module's own "LAT requested").

inline bool RefluxNeeded(const Mesh *pm) {
  return pm->multilevel || pm->hydro_lat_same_level;
}

}  // namespace lat

#endif  // DRIVER_LAT_WEIGHTS_HPP_
