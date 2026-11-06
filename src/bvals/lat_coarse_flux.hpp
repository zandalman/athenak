#ifndef BVALS_LAT_COARSE_FLUX_HPP_
#define BVALS_LAT_COARSE_FLUX_HPP_
//========================================================================================
// AthenaK astrophysical fluid dynamics & numerical relativity code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file lat_coarse_flux.hpp
//! \brief the delayed LAT reflux accumulator of the stage-coupled modules: its storage, and the record of a block's OWN time-integrated flux on
//! every face where a finer (or, with same-level LAT, faster) neighbour ships back its
//! own integral.

#include "athena.hpp"

class Driver;
class MeshBlockPack;
class MeshBoundaryValuesCC;

namespace lat {

//! (Re)allocate the two boundary bands of each direction of a reflux register and zero
//! them.
void AllocateRefluxBands(DvceFaceFld5D<Real> &acc, int nmb, int nvar, int n3, int n2,
                         int n1);

//! Called from the module's SendFlux, before the pack.  Adds weight*dt*F of the module's
//! flux bands flx1..3 into acc on the faces of the active blocks that take a delayed
//! reflux.  ledger: the GR module's kernel, which also adds into *own when it is given
//! (its lat_diagnostics ledger copy); the Newtonian module passes false and nullptr.
//! Equal-factor fine faces are skipped: they take the direct_fine replacement path inside
//! RecvAndAccumulateFluxCC (flux_correct_cc.cpp) and never touch the accumulator.
void AccumulateCoarseFluxes(MeshBlockPack *pmbp, MeshBoundaryValuesCC *pbval,
                            Driver *pdrive, int stage, const BandView5D<Real> &flx1,
                            const BandView5D<Real> &flx2, const BandView5D<Real> &flx3,
                            const DvceFaceFld5D<Real> &acc,
                            const DvceFaceFld5D<Real> *own, bool ledger, int nvar,
                            const char *label);

}  // namespace lat

#endif  // BVALS_LAT_COARSE_FLUX_HPP_
