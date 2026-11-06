#ifndef UTILS_MASS_DRAIN_HPP_
#define UTILS_MASS_DRAIN_HPP_
//========================================================================================
// AthenaK astrophysical fluid dynamics & numerical relativity code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the AthenaK collaboration
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file mass_drain.hpp
//! \brief Whether a stage update empties a cell: the test of the first-order flux
//! correction that keeps the GR gas-entropy column a convex combination.

#include "athena.hpp"

//----------------------------------------------------------------------------------------
//! \fn StageDrainsCell
//! \brief The stage update leaves the cell the remainder R = w D - (what its six faces
//! carry out), with D the cell's mass density, w the weight of the state the fluxes were
//! computed from (the Shu-Osher gam0, or gam0 + gam1 at the first stage, where both
//! registers hold that state) and the six face mass fluxes in units of dt/dx.  The
//! gas-entropy column is updated by the same fluxes with the donor value s_i on the
//! outgoing faces, so its new value is (R s_i + incoming F_D s_j + older state)/D' and
//! lies between the values the column holds iff R >= 0 (Larrouturou 1991, J. Comput.
//! Phys. 95, 59: consistency with continuity).  For R < 0 the update extrapolates away
//! from the neighbours without bound, and the entropy root is exponential in s.  Returns
//! true iff R < 0.
//!
//! `held` is w D; `dtodxN` is the stage's beta dt/dx in direction N.  A first-order
//! Rusanov face carries out at most lambda a D per cell and direction, so a cell whose
//! faces are all first order has R >= D (1 - sum_d lambda_d a_d) >= 0 for cfl <= 1/3 in
//! 3D when a <= 1.  In smooth flow the test never fires (R/D >= 1 - sqrt(3) cfl |V|).  It
//! reads the mass flux only; the entropy slot stays out of every admissibility test.

template <class FluxView>
KOKKOS_INLINE_FUNCTION
bool StageDrainsCell(const FluxView &flx1, const FluxView &flx2, const FluxView &flx3,
                     const int m, const int k, const int j, const int i,
                     const bool multi_d, const bool three_d,
                     const Real dtodx1, const Real dtodx2, const Real dtodx3,
                     const Real held) {
  Real outflow = dtodx1*(fmax(flx1(m,IDN,k,j,i+1), 0.0) + fmax(-flx1(m,IDN,k,j,i), 0.0));
  if (multi_d) {
    outflow += dtodx2*(fmax(flx2(m,IDN,k,j+1,i), 0.0) + fmax(-flx2(m,IDN,k,j,i), 0.0));
  }
  if (three_d) {
    outflow += dtodx3*(fmax(flx3(m,IDN,k+1,j,i), 0.0) + fmax(-flx3(m,IDN,k,j,i), 0.0));
  }
  return held < outflow;
}

//----------------------------------------------------------------------------------------
//! \fn FlagDrainedCells
//! \brief Flags, in `flag`, every unflagged owned cell of the active blocks that the stage
//! drains on the face fluxes now in the registers, and returns how many it flagged.
//!
//! The test of the flux task (StageDrainsCell inside the FOFC trial) runs on the
//! high-order fluxes, before any face is replaced.  A face a flagged neighbour replaces
//! by its first-order flux is shared with an unflagged cell, whose remainder
//!   R = w D - (what its six faces carry out)
//! then changes: the first-order flux of a face carries out at most about lambda a D, but
//! the cell's other, high-order faces were admitted by the test on the strength of the
//! old face.  R can be negative afterwards (WENOZ_DESIGN U2; measured: a hole of density
//! 4e-14 beside four flagged cells, R = -0.17 D, whose entropy label then left the range
//! of its neighbours by a factor 3).  A cell with every face first order has R >= D (1 -
//! sum_d lambda_d a_d) >= 0, so the flagged set only has to grow: flag the cells that
//! fail on the replaced fluxes, replace their faces, test again, until no cell is added
//! (the set is monotone and finite).  The caller exchanges the new flags and replaces
//! the faces of every round.  The excised cells a hard excision does not update are
//! skipped; `held` is `weight` times the cell's mass density, the weight of the state
//! the fluxes were computed from, exactly as in the first test.

template <class FluxView, class MassView, class FlagView, class SizeView,
          class IndexView, class StepView, class ExciseView>
int FlagDrainedCells(const FluxView &flx1, const FluxView &flx2, const FluxView &flx3,
                     const MassView &cons0, const FlagView &flag, const SizeView &size,
                     const IndexView &active_indices, const StepView &lat_step_dt,
                     const ExciseView &excised, const bool skip_excised,
                     const bool lat_enabled, const bool lat_per_block_dt,
                     const int nwork, const int is, const int ie, const int js,
                     const int je, const int ks, const int ke, const bool multi_d,
                     const bool three_d, const Real beta_stage, const Real beta_dt,
                     const Real weight) {
  const int ni = ie - is + 1;
  const int nji = (je - js + 1)*ni;
  const int nkji = (ke - ks + 1)*nji;
  int added = 0;
  Kokkos::parallel_reduce("fofc_drain_close",
      Kokkos::RangePolicy<>(DevExeSpace(), 0, nwork*nkji),
  KOKKOS_LAMBDA(const int idx, int &count) {
    const int a = idx/nkji;
    const int rem = idx - a*nkji;
    const int k = ks + rem/nji;
    const int ji = rem - (k - ks)*nji;
    const int j = js + ji/ni;
    const int i = is + ji - (j - js)*ni;
    const int m = lat_enabled ? active_indices(a) : a;
    if (flag(m,k,j,i)) return;
    if (skip_excised && excised(m,k,j,i)) return;
    const Real block_beta_dt = lat_per_block_dt ? beta_stage*lat_step_dt(m) : beta_dt;
    if (StageDrainsCell(flx1, flx2, flx3, m, k, j, i, multi_d, three_d,
                        block_beta_dt/size(m).dx1, block_beta_dt/size(m).dx2,
                        block_beta_dt/size(m).dx3, weight*cons0(m,IDN,k,j,i))) {
      flag(m,k,j,i) = true;
      ++count;
    }
  }, Kokkos::Sum<int>(added));
  return added;
}

#endif  // UTILS_MASS_DRAIN_HPP_
