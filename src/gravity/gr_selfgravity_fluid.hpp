#ifndef GRAVITY_GR_SELFGRAVITY_FLUID_HPP_
#define GRAVITY_GR_SELFGRAVITY_FLUID_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file gr_selfgravity_fluid.hpp
//! \brief Fixed-metric (Cartesian Kerr-Schild) fluid adapters of the weak-field GR
//! self-gravity: the active mass varrho = E + S and the four-force of MHD and GR hydro.
//! The physics is gr_selfgravity_core.hpp's; these kernels only gather the cell state.

#include "athena.hpp"
#include "gravity/gr_selfgravity_core.hpp"

class MeshBlockPack;

namespace grsg {

//----------------------------------------------------------------------------------------
//! \fn void AddGravitatingDensityFixed
//! \brief Adds the gas term varrho = E + gamma_ij S^ij (paper Eq. 19) of the stage
//! primitives of the pack's MHD or GR hydro to rho_eff(m, 0, k, j, i), non-densitized
//! and unweighted, over the active cells of every block.  Cells of the excision mask
//! are skipped.  The pack must run the fixed Cartesian Kerr-Schild metric.

void AddGravitatingDensityFixed(MeshBlockPack *pack, DvceArray5D<Real> &rho_eff);

//----------------------------------------------------------------------------------------
//! \fn void AddFluidForceFixed
//! \brief Adds beta_dt times the self-gravity four-force of Eqs. (1)-(3) to u0(IM, IEN)
//! of the pack's MHD (ideal, variable gamma / two-temperature) or GR hydro, from the
//! stage primitives w0 and bcc0 and the frozen potential of `pv`: the momentum pointwise
//! (Eq. 2), the energy as the face work of the stage's final numerical fluxes `flux`
//! (Eq. 3a, F[C_t] = F_IEN - F_IDN).  `dt` is the stage beta_dt.  Every owned cell of the
//! active blocks, as Coordinates::CoordSrcTerms.  A no-op when !pv.active.

void AddFluidForceFixed(MeshBlockPack *pack, const PhiView &pv, const Real dt,
                        const BandFaceFld5D<Real> &flux, DvceArray5D<Real> &u0);

//----------------------------------------------------------------------------------------
//! \fn void AddFaceWorkDyn
//! \brief The dynamical-GR counterpart of the energy part: adds beta_dt times the face
//! work of F[tau + D] = F_IEN + F_IDN (Eq. 3a) to u0(IEN) = tau.  The pointwise remainder
//! -sqrt(gamma) epsbar beta.dPhi and the momentum are added by the fused coordinate
//! source (DynGRMHDPS::AddCoordTermsEOSImpl).  Same cells, weight and per-block step.

void AddFaceWorkDyn(MeshBlockPack *pack, const PhiView &pv, const Real dt,
                    const BandFaceFld5D<Real> &flux, DvceArray5D<Real> &u0);

}  // namespace grsg

#endif  // GRAVITY_GR_SELFGRAVITY_FLUID_HPP_
