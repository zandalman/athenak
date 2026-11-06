#ifndef COORDINATES_KERR_HOLE_HPP_
#define COORDINATES_KERR_HOLE_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file kerr_hole.hpp
//! \brief quasi-stationary evolution of the fixed Kerr hole's mass and spin under GR
//! self-gravity (FUGUE, Plonka & Janiuk, arXiv:2610.20869, Sect. III; Gammie et al. 2004).
//!
//! The hole of the fixed Cartesian Kerr-Schild metric gains exactly what the gas loses
//! into the excised region.  For every face between an active non-excised cell and an
//! excision_floor cell, with n pointing from the gas into the hole and dA the face area,
//!     dM  = - sum (F_IEN - F_IDN) . n dA w_s dt_b     (F_IEN - F_IDN = flux of T^t_t)
//!     dJz = + sum (x_f F_IM2 - y_f F_IM1) . n dA w_s dt_b
//! from the final fluxes the update applied (w_s = lat::FinalFluxWeight, dt_b the block
//! step).  IEN = T^t_t + D, so M takes the total energy including rest mass, and
//! M_BH + E_gas is bookkept exactly.  The sums go into per-block slots each stage (no
//! atomics, no collectives) in code mass and are reduced at a global sync, in global
//! block-id order, times G of the code units (the hole's mass is geometric).
//!
//! The metric (M_g, a_g) in CoordData lags the bookkept (M, J) and is refreshed at a
//! global sync once the lag exceeds 1e-5 M_g.  The time integral of the source
//! S_t = 1/2 T^{mu nu} d_t g_{mu nu} over the lag is then applied as the exact impulse
//! 1/2 T^{mu nu} [g_{mu nu}(new) - g_{mu nu}(old)] to the energy alone; D, the momenta,
//! B and every scalar are unchanged (sqrt(-g) = 1 for every M, a in CKS).  The primitives
//! are then recovered with the new metric through the regrid's own entry.
//!
//! The hole evolves on the fixed metric when <gravity> is on, the hole is not Minkowski
//! and it is excised.  There are no deck keys.  The state rides the restart parameter
//! text as <coord>/bh_*_live|true keys that no deck carries.

#include "athena.hpp"

class Driver;
class Mesh;
class MeshBlockPack;
class ParameterInput;
struct CoordData;

namespace kerr_hole {

//! True when the hole's mass and spin evolve in this run.
bool Evolves();

//! The hole ends with its MeshBlockPack: the next one starts again from the restart keys,
//! or from the deck.
void Release();

//! Called by the Coordinates constructor (first construction, every regrid) after the
//! excision radii are set and before the masks are built.  The first call reads the
//! restart keys when present; every call writes the live (M_g, a_g) and excision radii
//! into cd, so a regrid's new Coordinates continues the evolved metric.
void RestoreHole(ParameterInput *pin, MeshBlockPack *ppack, bool fixed_gr,
                 CoordData &cd);

//! Add one stage's hole face sums of the fluid into the per-block slots.
//! uflx: the fluid's final face fluxes (pmhd/phydro->uflx, or their FluxBand view);
//! ien, idn: the energy and density indices; w_dt = lat::FinalFluxWeight(pdrive, stage)
//! * pmesh->dt, zero for a stage whose fluxes are not applied.  Under per-block LAT
//! steps each block's weight is w_dt * lat_step_dt(m) / pmesh->dt.  Loops the
//! LAT-active blocks only.  Call after the stage's update, at the end of its source pass.
void AccumulateFaceSums(MeshBlockPack *pack, const BandFaceFld5D<Real> &uflx,
                        int ien, int idn, Real w_dt);
void AccumulateFaceSums(MeshBlockPack *pack, const DvceFaceFld5D<Real> &uflx,
                        int ien, int idn, Real w_dt);

//! Global sync (C4): reduce the slots, apply the refresh rule and, on a refresh, the
//! fluid impulse, the excision-radius rule and the primitive recovery.  Called
//! by the driver on every rank after the last stage's C2P and before outputs, the
//! restart dump and AMR, with every block active (after ClearHydroLAT) and no LAT send
//! latched.  A no-op when the hole does not evolve.
void Synchronize(Driver *pdrive);
void Synchronize(Driver *pdrive, Mesh *pm);

//! Writes the restart keys into pin (restart output, before the parameter dump).
void StoreRestartKeys(ParameterInput *pin);

//! The bookkept hole for the history: M and the dimensionless spin J/M^2.
void HistoryValues(Real &bh_m, Real &bh_a);

}  // namespace kerr_hole

#endif  // COORDINATES_KERR_HOLE_HPP_
