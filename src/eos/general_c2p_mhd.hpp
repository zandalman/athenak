#ifndef EOS_GENERAL_C2P_MHD_HPP_
#define EOS_GENERAL_C2P_MHD_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file general_c2p_mhd.hpp
//! \brief EOS-aware nonrel MHD primitive/conserved helpers shared by gamma-law and
//! tabulated EOS paths.

#include <cmath>

#include "athena.hpp"
#include "eos/eos.hpp"
#include "mhd/mhd.hpp"

namespace eos_general {

KOKKOS_INLINE_FUNCTION
void SingleP2C_GeneralMHD(const MHDPrim1D &w, MHDCons1D &u) {
  u.d = w.d;
  u.mx = w.d*w.vx;
  u.my = w.d*w.vy;
  u.mz = w.d*w.vz;
  u.e = w.e + 0.5*w.d*(SQR(w.vx) + SQR(w.vy) + SQR(w.vz)) +
        0.5*(SQR(w.bx) + SQR(w.by) + SQR(w.bz));
  u.by = w.by;
  u.bz = w.bz;
}

KOKKOS_INLINE_FUNCTION
void SingleP2C_GeneralMHD(const MHDPrim1D &w, HydCons1D &u) {
  u.d = w.d;
  u.mx = w.d*w.vx;
  u.my = w.d*w.vy;
  u.mz = w.d*w.vz;
  u.e = w.e + 0.5*w.d*(SQR(w.vx) + SQR(w.vy) + SQR(w.vz)) +
        0.5*(SQR(w.bx) + SQR(w.by) + SQR(w.bz));
}

// Dual-energy selection for Newtonian MHD: the internal energy is read off the energy
// channel, eint = E - KE - B^2/2, only where it is not lost in the subtraction.  This is
// the ONE statement of that rule, hydro's (HydroEnergyChannelResolvesEint) with the
// magnetic energy in the split.  The inversions call it to choose the pressure, and the
// owner's decision before the exchange (MHD::ResolveDualEnergyChannel) calls it too, so
// the auxiliary is only ever overwritten by a value the inversion would itself accept.
// With emag = 0 it is the hydro test.
//
// Two thresholds, because E carries two truncation errors of different size.  The eta1
// test of Bryan et al. (1995) bounds the kinetic one.  The magnetic one is first order
// in the field perturbation, because the Riemann flux of E and the CT update of B are
// two discretisations of the magnetic energy that agree only to leading order in B.dB:
// per stage E - B^2/2 is off by ~ CFL*(dv/v_A)*B^2/2, and for any motion slower than
// sound (dv < c_s) that is ~ CFL*sqrt(beta)*B^2/2.  It exceeds the thermal energy,
// ~ beta*B^2/2, below beta ~ CFL^2 ~ 0.1, however quiet the flow.  A low-beta star
// atmosphere (beta ~ 1e-3) measured it at 0.3-1% of B^2/2 per stage against a thermal
// energy of 0.03-0.06%, which eta1 = 1e-3 of E ~ B^2/2 cannot reject.  Hence the second
// test, a fixed fraction of the magnetic energy: below it the energy channel is noise
// and the auxiliary, which sees only advection and compression, is the thermal state.
//
// A rejected energy channel must not be kept: the owner's decision and the inversion
// that reject it write E = eint_aux + KE + B^2/2 back.  Left alone, E integrates the
// error of every stage, and since nothing else ever reads it the residual grows until
// it passes any fixed threshold, when the whole of it is released as heat at once.  E is
// therefore not conserved in a rejected cell, by exactly the residual that is not
// thermal energy.
inline constexpr Real kDualEnergyMagneticFraction = 0.1;

KOKKOS_INLINE_FUNCTION
bool MHDEnergyChannelResolvesEint(const Real eint_cons, const Real etot, const Real emag,
                                  const Real dual_eta1) {
  return (eint_cons > 0.0) &&
         ((dual_eta1 <= 0.0) ||
          ((eint_cons > dual_eta1*etot) &&
           (eint_cons > kDualEnergyMagneticFraction*emag)));
}

// The internal energy a Newtonian dual-energy inversion of an owned cell reads off it:
// the energy channel where MHDEnergyChannelResolvesEint accepts it, the auxiliary
// elsewhere.  With emag = 0 the test is the hydro one of SingleC2P_GeneralHydDual.  For
// code that hands a cell's thermal state on outside an inversion (a remap carrying it, a
// frame boost that must leave it unchanged): the inversions now publish that state in the
// auxiliary, but a checkpoint written before they did can hold a different one.
KOKKOS_INLINE_FUNCTION
Real DualEnergyInversionEint(const Real eint_cons, const Real etot, const Real emag,
                             const Real eint_aux, const Real dual_eta1) {
  return MHDEnergyChannelResolvesEint(eint_cons, etot, emag, dual_eta1) ? eint_cons
                                                                        : eint_aux;
}

// The one place the magnetic test is wrong is a shock: a low-beta shock that it rejects
// loses its heating to the reset (a beta = 0.01 Brio-Wu slow shock came out at 0.64 of
// its peak specific energy).  MHD applies the shock rule of eos/general_c2p_hyd.hpp with
// hydro's implementation (eos/dual_energy_shock_rule.hpp, MHD::ApplyDualEnergyFormalism):
// the test runs once per step on the step-start primitives and field and holds through
// the step, reaches nghost - 2 cells to both sides along each axis, and a shocked cell
// copies the energy channel E - E_K - E_M into the auxiliary at every stage before the
// step's last and at the last under the step-final guard.  The owner's decision
// (MHD::ResolveDualEnergyChannel) and every inversion -- pointwise ones included, which
// cannot see a neighbour, and the ghost zones, which read the owner's auxiliary -- then
// take the shock heat through the auxiliary channel.  Its price is the truncation error
// of E in the shocked cells, which the scheme accepted there before the magnetic test
// existed; its first-order magnetic part is not taken (below).
//
// The steady test measures the inflow per MHD family: at the upstream point along each
// axis, against the fast and the slow shock whose velocity jump equals the upstream gas
// sound speed (DualEnergyAddShockInflow), each family summed over the axes as hydro sums
// them; the slow family only where the inflow is slower than the Alfven speed along the
// axis, as a slow shock's is.  A simple wave of either family flows in at its own speed
// and is never flagged, and at B = 0 the test is hydro's, bit for bit.  The threshold is
// the gas sound speed, not the fast speed: a low-beta fast shock weak against c_f still
// heats the gas (beta = 0.06, a jump dv = 0.52 c_f = 2.4 c_s doubles its entropy), and
// the auxiliary carries the gas's energy.
//
// A shocked cell reads the channel against the field E's fluxes carried, not the CT
// field (MHDFieldCarriedByE, mhd_dual_energy.cpp): E's Poynting flux through a face is
// consistent with that face's EMFs, CT moves the field with the edge EMFs, and the two
// cell fields differ to first order in the field change, which is the magnetic error
// above (in 1D, and for planes along the grid, they are the same field).  A cell the
// owner's decision then rejects keeps that channel in its auxiliary and resets E =
// eint_aux + E_K + E_M of the CT field: E is not conserved there by the mismatch,
// truncation error as in every rejected cell.  So the steady test runs in every cell.
// Where it misfires, as in a uniform low-beta atmosphere, where the steady speed
// [rho v]/[rho] of two counter-propagating waves takes any value, the cell takes the
// scheme's damping heat of the waves, second order in their amplitude, not the
// first-order magnetic error: read against the CT field, the steady test (on the fast
// speed) heated the atmosphere of a low-beta star in a static field 2.8-15 times (tst
// test_nr_star_surface_eps_cpu.py).  The stencil test sees the shock itself.
//
// The step-final guard is hydro's (general_c2p_hyd.hpp).  It is known to rectify the
// signed front-cell scatter of a shock smeared at a large bulk speed with PPM4 and rk3 or
// rk4 (colliding +-10 c streams carried at 300 c: 1.52-1.65 K/K_RH, the conservative
// scheme 0.977-0.985); taking the channel at every stage instead removed that, but the
// hydro form of it put about ten times more cold outliers on the TDE stream head, where
// the guard keeps the converging sheet off the temperature floor.  No dual-energy deck
// runs PPM4 with rk3 or rk4.
//
// Measured in a 1D model of the update (rk2): perpendicular colliding +-10 c streams
// carried at 300 and 1000 c keep K/K_RH 0.976-1.010 (beta = 1) and 0.935-1.004 (beta =
// 0.1) with nghost = 2, 0.988-1.011 with nghost = 4; the stencil test alone kept
// 0.63-0.72 and 0.19-0.59.  Open limitation: the low-beta Brio-Wu tube carried at Mach
// 300 smears the slow shock, the contact and the compound wave into one structure of ~85
// cells that forms at t = 0 and is never steady.  Its internal energy reaches 0.87-0.92
// of the same tube at rest (1D model, dx to dx/4; 0.77 with the stencil test alone).
// Taking the channel in every converging cell of that structure (an oracle) gives 1.05,
// but no local test was found that does so without heating compression waves: a flag held
// while the cell keeps converging takes the heat of the compression behind a slow shock,
// 6-11 % above the rest frame.  The conservative scheme is no reference there: at Mach
// 300 it holds 1.28 times the heat at rest (1.10 at dx/4), in cells that do not converge.

KOKKOS_INLINE_FUNCTION
Real ApplyMHDThermalFloors(const EOS_Data &eos, const Real dens, Real eint,
                           bool &efloor_used, bool &tfloor_used) {
  const Real eint_floor = eos.HydroInternalEnergyDensityFloor(dens);
  if (eint < eint_floor) {
    eint = eint_floor;
    efloor_used = true;
  }

  // Tabulated LTE EOS floors already enforce tfloor through the cached table closure.
  if ((eos.tfloor > 0.0) && !eos.UsesTabulatedLTE()) {
    const auto thermo = eos.EvalThermoStateFromRhoEint(dens, eint);
    if (thermo.temperature < eos.tfloor) {
      eint = dens*eos.SpecificEintFromRhoT(dens, eos.tfloor);
      tfloor_used = true;
    }
  }
  const Real eint_limited = eos.ClampHydroInternalEnergyDensity(dens, eint);
  if (eint_limited < eint) {
    eint = eint_limited;
    efloor_used = true;
  }
  return eint;
}

KOKKOS_INLINE_FUNCTION
Real MHDAtmosphereInternalEnergy(const EOS_Data &eos) {
  bool efloor_used = false;
  bool tfloor_used = false;
  return ApplyMHDThermalFloors(eos, eos.dfloor, static_cast<Real>(0.0),
                               efloor_used, tfloor_used);
}

KOKKOS_INLINE_FUNCTION
bool NeedsMHDAtmosphereReset(const EOS_Data &eos, const MHDCons1D &u) {
  return (u.d < eos.dfloor) ||
         ((u.d == eos.dfloor) &&
          ((u.mx != 0.0) || (u.my != 0.0) || (u.mz != 0.0)));
}

KOKKOS_INLINE_FUNCTION
void ResetMHDAtmosphereState(const EOS_Data &eos, MHDCons1D &u, MHDPrim1D &w,
                             bool &dfloor_used) {
  const Real atmosphere_eint = MHDAtmosphereInternalEnergy(eos);
  const Real emag = 0.5*(SQR(u.bx) + SQR(u.by) + SQR(u.bz));
  u.d = eos.dfloor;
  u.mx = 0.0;
  u.my = 0.0;
  u.mz = 0.0;
  u.e = atmosphere_eint + emag;
  w.d = u.d;
  w.vx = 0.0;
  w.vy = 0.0;
  w.vz = 0.0;
  w.e = atmosphere_eint;
  w.bx = u.bx;
  w.by = u.by;
  w.bz = u.bz;
  dfloor_used = true;
}

KOKKOS_INLINE_FUNCTION
void ResetMHDAtmosphereState(const EOS_Data &eos, MHDCons1D &u, HydPrim1D &w,
                             bool &dfloor_used) {
  const Real atmosphere_eint = MHDAtmosphereInternalEnergy(eos);
  const Real emag = 0.5*(SQR(u.bx) + SQR(u.by) + SQR(u.bz));
  u.d = eos.dfloor;
  u.mx = 0.0;
  u.my = 0.0;
  u.mz = 0.0;
  u.e = atmosphere_eint + emag;
  w.d = u.d;
  w.vx = 0.0;
  w.vy = 0.0;
  w.vz = 0.0;
  w.e = atmosphere_eint;
  dfloor_used = true;
}

KOKKOS_INLINE_FUNCTION
void ApplyMHDVelocityCeiling(const EOS_Data &eos, MHDPrim1D &w, bool &vceil_used) {
  const Real v2 = SQR(w.vx) + SQR(w.vy) + SQR(w.vz);
  const Real vmag = sqrt(v2);
  if ((eos.vceil > 0.0) && (vmag > eos.vceil)) {
    const Real fac = eos.vceil/vmag;
    w.vx *= fac;
    w.vy *= fac;
    w.vz *= fac;
    vceil_used = true;
  }
}

KOKKOS_INLINE_FUNCTION
void SingleC2P_GeneralMHD(MHDCons1D &u, const EOS_Data &eos, MHDPrim1D &w,
                          bool &dfloor_used, bool &efloor_used, bool &tfloor_used,
                          bool &vceil_used) {
  if (NeedsMHDAtmosphereReset(eos, u)) {
    ResetMHDAtmosphereState(eos, u, w, dfloor_used);
    return;
  }
  w.d = u.d;

  const Real di = 1.0/u.d;
  w.vx = di*u.mx;
  w.vy = di*u.my;
  w.vz = di*u.mz;
  w.bx = u.bx;
  w.by = u.by;
  w.bz = u.bz;
  w.e = u.e - 0.5*di*(SQR(u.mx) + SQR(u.my) + SQR(u.mz))
              - 0.5*(SQR(u.bx) + SQR(u.by) + SQR(u.bz));

  w.e = ApplyMHDThermalFloors(eos, w.d, w.e, efloor_used, tfloor_used);
  ApplyMHDVelocityCeiling(eos, w, vceil_used);
  SingleP2C_GeneralMHD(w, u);
}

KOKKOS_INLINE_FUNCTION
void SingleC2P_GeneralMHD(MHDCons1D &u, const EOS_Data &eos, HydPrim1D &w,
                          bool &dfloor_used, bool &efloor_used, bool &tfloor_used,
                          bool &vceil_used) {
  if (NeedsMHDAtmosphereReset(eos, u)) {
    ResetMHDAtmosphereState(eos, u, w, dfloor_used);
    return;
  }
  w.d = u.d;

  const Real di = 1.0/u.d;
  w.vx = di*u.mx;
  w.vy = di*u.my;
  w.vz = di*u.mz;
  w.e = u.e - 0.5*di*(SQR(u.mx) + SQR(u.my) + SQR(u.mz))
              - 0.5*(SQR(u.bx) + SQR(u.by) + SQR(u.bz));

  w.e = ApplyMHDThermalFloors(eos, w.d, w.e, efloor_used, tfloor_used);
  const Real v2 = SQR(w.vx) + SQR(w.vy) + SQR(w.vz);
  const Real vmag = sqrt(v2);
  if ((eos.vceil > 0.0) && (vmag > eos.vceil)) {
    const Real fac = eos.vceil/vmag;
    w.vx *= fac;
    w.vy *= fac;
    w.vz *= fac;
    vceil_used = true;
  }
}

// The MHD dual-energy inversion, as SingleC2P_GeneralHydDual: energy_channel_readable
// for the cell's own recovery (an owned cell), which may take E - E_K - E_M; a ghost zone
// or coarse buffer holds a state assembled from owners' states, whose auxiliary is the
// matching combination of their thermal energies, and reads the auxiliary
// (MHD::ResolveDualEnergyChannel).  The caller writes u.e = eint_aux + E_K + E_M back
// where eint_from_aux is set.  eint_aux_out is the published thermal energy, the
// auxiliary after every recovery.
KOKKOS_INLINE_FUNCTION
void SingleC2P_GeneralMHDDual(MHDCons1D &u, const EOS_Data &eos, const Real eint_aux_in,
                              const Real dual_eta1, const bool energy_channel_readable,
                              MHDPrim1D &w, Real &eint_aux_out,
                              bool &eint_from_aux, bool &dfloor_used, bool &efloor_used,
                              bool &tfloor_used, bool &vceil_used) {
  if (NeedsMHDAtmosphereReset(eos, u)) {
    ResetMHDAtmosphereState(eos, u, w, dfloor_used);
    eint_aux_out = w.e;
    eint_from_aux = false;
    return;
  }
  w.d = u.d;

  const Real di = 1.0/u.d;
  w.vx = di*u.mx;
  w.vy = di*u.my;
  w.vz = di*u.mz;
  w.bx = u.bx;
  w.by = u.by;
  w.bz = u.bz;

  const Real emag = 0.5*(SQR(u.bx) + SQR(u.by) + SQR(u.bz));
  const Real eint_cons = u.e - 0.5*di*(SQR(u.mx) + SQR(u.my) + SQR(u.mz)) - emag;
  Real eint_aux = ApplyMHDThermalFloors(eos, w.d, eint_aux_in, efloor_used,
                                        tfloor_used);

  const bool use_cons_e = energy_channel_readable &&
                          MHDEnergyChannelResolvesEint(eint_cons, u.e, emag, dual_eta1);
  eint_from_aux = !use_cons_e;
  w.e = use_cons_e ? eint_cons : eint_aux;
  w.e = ApplyMHDThermalFloors(eos, w.d, w.e, efloor_used, tfloor_used);

  ApplyMHDVelocityCeiling(eos, w, vceil_used);
  SingleP2C_GeneralMHD(w, u);
  eint_aux_out = w.e;
}

KOKKOS_INLINE_FUNCTION
void SingleC2P_GeneralMHDDual(MHDCons1D &u, const EOS_Data &eos, const Real eint_aux_in,
                              const Real dual_eta1, const bool energy_channel_readable,
                              HydPrim1D &w, Real &eint_aux_out,
                              bool &eint_from_aux, bool &dfloor_used, bool &efloor_used,
                              bool &tfloor_used, bool &vceil_used) {
  if (NeedsMHDAtmosphereReset(eos, u)) {
    ResetMHDAtmosphereState(eos, u, w, dfloor_used);
    eint_aux_out = w.e;
    eint_from_aux = false;
    return;
  }
  w.d = u.d;

  const Real di = 1.0/u.d;
  w.vx = di*u.mx;
  w.vy = di*u.my;
  w.vz = di*u.mz;

  const Real emag = 0.5*(SQR(u.bx) + SQR(u.by) + SQR(u.bz));
  const Real eint_cons = u.e - 0.5*di*(SQR(u.mx) + SQR(u.my) + SQR(u.mz)) - emag;
  Real eint_aux = ApplyMHDThermalFloors(eos, w.d, eint_aux_in, efloor_used,
                                        tfloor_used);

  const bool use_cons_e = energy_channel_readable &&
                          MHDEnergyChannelResolvesEint(eint_cons, u.e, emag, dual_eta1);
  eint_from_aux = !use_cons_e;
  w.e = use_cons_e ? eint_cons : eint_aux;
  w.e = ApplyMHDThermalFloors(eos, w.d, w.e, efloor_used, tfloor_used);

  const Real v2 = SQR(w.vx) + SQR(w.vy) + SQR(w.vz);
  const Real vmag = sqrt(v2);
  if ((eos.vceil > 0.0) && (vmag > eos.vceil)) {
    const Real fac = eos.vceil/vmag;
    w.vx *= fac;
    w.vy *= fac;
    w.vz *= fac;
    vceil_used = true;
  }
  eint_aux_out = w.e;
}

}  // namespace eos_general

#endif  // EOS_GENERAL_C2P_MHD_HPP_
