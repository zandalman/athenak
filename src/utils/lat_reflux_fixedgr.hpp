#ifndef UTILS_LAT_REFLUX_FIXEDGR_HPP_
#define UTILS_LAT_REFLUX_FIXEDGR_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file lat_reflux_fixedgr.hpp
//! \brief Admissibility of the LAT window-end reflux on the fixed-metric GR hydro backend.
//!
//! The energy slot of this backend holds T^t_t + D, which has no sign bound, so a
//! near-total reflux of a vacuum coarse cell leaves (D, S, E) as a difference of nearly
//! equal large numbers that passes the density test of lat_reflux::Admissible and that
//! the recovery cannot invert.  The corrected state is therefore tested with the
//! recovery's own single-cell inversion on the cell's own metric: the GR->SR transform
//! and SingleC2P_IdealSRHyd that IdealGRHydro::ConsToPrim calls.  The inversion gives the
//! recovery's verdict (the kRecovery* bits of lat_reflux_limiter.hpp): a root, a density
//! floor, a Lorentz-factor ceiling.  The fraction is lat_reflux::RecoveryAdmissibleFraction
//! of that verdict.

#include <math.h>

#include "athena.hpp"
#include "coordinates/cartesian_ks.hpp"
#include "coordinates/cell_locations.hpp"
#include "eos/eos.hpp"
#include "eos/ideal_c2p_hyd.hpp"
#include "mesh/mesh.hpp"
#include "utils/lat_reflux_limiter.hpp"

namespace lat_reflux {

//----------------------------------------------------------------------------------------
//! \fn int FixedGRInversionVerdict
//! \brief The recovery's verdict (kRecovery* bits) on the SR inversion of one cell: a root
//! with finite primitives, the density floor, the Lorentz-factor ceiling gamma_max.

KOKKOS_INLINE_FUNCTION
int FixedGRInversionVerdict(const KSNullForm &nf, const HydPrim1D &w,
                            const bool dfloor_used, const bool c2p_failure,
                            const Real gamma_max) {
  if (c2p_failure) return 0;
  if (!(isfinite(w.d) && isfinite(w.e) && isfinite(w.vx) && isfinite(w.vy) &&
        isfinite(w.vz))) return 0;
  int verdict = kRecoveryInverts;
  if (dfloor_used) verdict |= kRecoveryFloored;
  const Real lor = sqrt(1.0 + KSSpatialNormSq(nf, 1, 2, 3, w.vx, w.vy, w.vz));
  if (lor > gamma_max) verdict |= kRecoveryLimited;
  return verdict;
}

//----------------------------------------------------------------------------------------
//! \fn int FixedGRHydroVerdict
//! \brief The verdict of IdealGRHydro::ConsToPrim on the state u on metric nf.  Kept out
//! of line on the device so that the root solve is compiled once, not at every reflux
//! call site.

KOKKOS_INLINE_FUNCTION GR_ENTROPY_DEVICE_NOINLINE
int FixedGRHydroVerdict(const EOS_Data &eos, const KSNullForm nf, const HydCons1D u) {
  HydCons1D u_sr;
  Real s2;
  TransformToSRHyd(u, nf, s2, u_sr);
  HydPrim1D w;
  bool dfloor_used = false, efloor_used = false, c2p_failure = false;
  int iter_used = 0;
  SingleC2P_IdealSRHyd(u_sr, eos, s2, w, dfloor_used, efloor_used, c2p_failure,
                       iter_used);
  return FixedGRInversionVerdict(nf, w, dfloor_used, c2p_failure, eos.gamma_max);
}

//----------------------------------------------------------------------------------------
//! \fn KSNullForm CellNullForm
//! \brief The metric at the centre of cell (i,j,k) of a block, as ConsToPrim takes it.

KOKKOS_INLINE_FUNCTION
KSNullForm CellNullForm(const KerrHole &hole, const RegionIndcs &indcs,
                        const RegionSize &size, const int i, const int j, const int k) {
  KSNullForm nf;
  ComputeKSNullForm(CellCenterX(i - indcs.is, indcs.nx1, size.x1min, size.x1max),
                    CellCenterX(j - indcs.js, indcs.nx2, size.x2min, size.x2max),
                    CellCenterX(k - indcs.ks, indcs.nx3, size.x3min, size.x3max),
                    hole, nf);
  return nf;
}

//----------------------------------------------------------------------------------------
//! \fn Real FixedGRHydroRefluxFraction
//! \brief Largest fraction of the pending correction dcons that cell (i,j,k) of a
//! fixed-metric GR hydro block can absorb and still invert (RecoveryAdmissibleFraction;
//! the energy slot T^t_t + D has no sign bound, so only D >= dfloor is bounded).

KOKKOS_INLINE_FUNCTION
Real FixedGRHydroRefluxFraction(const EOS_Data &eos, const KerrHole &hole,
                                const RegionIndcs &indcs, const RegionSize &size,
                                const int i, const int j, const int k,
                                const Real cons[], const Real dcons[]) {
  const KSNullForm nf = CellNullForm(hole, indcs, size, i, j, k);
  auto verdict = [&](const Real f) {
    return FixedGRHydroVerdict(eos, nf, HydCons1D{
        cons[IDN] + f*dcons[IDN], cons[IM1] + f*dcons[IM1], cons[IM2] + f*dcons[IM2],
        cons[IM3] + f*dcons[IM3], cons[IEN] + f*dcons[IEN]});
  };
  return RecoveryAdmissibleFraction(verdict, eos.dfloor, false, cons, dcons);
}

} // namespace lat_reflux

#endif // UTILS_LAT_REFLUX_FIXEDGR_HPP_
