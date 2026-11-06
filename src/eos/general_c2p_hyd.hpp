#ifndef EOS_GENERAL_C2P_HYD_HPP_
#define EOS_GENERAL_C2P_HYD_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file general_c2p_hyd.hpp
//! \brief EOS-aware nonrel hydro primitive/conserved helpers shared by gamma-law and
//! tabulated EOS paths.

#include <cmath>
#include <limits>

#include "athena.hpp"
#include "eos/eos.hpp"
#include "hydro/hydro.hpp"

namespace eos_general {

KOKKOS_INLINE_FUNCTION
void SingleP2C_GeneralHyd(const HydPrim1D &w, HydCons1D &u) {
  u.d = w.d;
  u.mx = w.d*w.vx;
  u.my = w.d*w.vy;
  u.mz = w.d*w.vz;
  u.e = w.e + 0.5*w.d*(SQR(w.vx) + SQR(w.vy) + SQR(w.vz));
}

// The floors and ceiling of one density, evaluated once by the caller
// (EOS_Data::HydroInternalEnergyDensityBounds) when the same density is limited more
// than once.
KOKKOS_INLINE_FUNCTION
Real ApplyHydroThermalFloorsWithBounds(const EOS_Data &eos, const Real dens, Real eint,
                                       const Real eint_floor, const Real eint_ceil,
                                       bool &efloor_used, bool &tfloor_used) {
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
  const Real eint_limited = eos.ClampHydroInternalEnergyDensityTo(eint, eint_ceil);
  if (eint_limited < eint) {
    eint = eint_limited;
    efloor_used = true;
  }
  return eint;
}

KOKKOS_INLINE_FUNCTION
Real ApplyHydroThermalFloors(const EOS_Data &eos, const Real dens, Real eint,
                             bool &efloor_used, bool &tfloor_used) {
  Real eint_floor, eint_ceil;
  eos.HydroInternalEnergyDensityBounds(dens, eint_floor, eint_ceil);
  return ApplyHydroThermalFloorsWithBounds(eos, dens, eint, eint_floor, eint_ceil,
                                           efloor_used, tfloor_used);
}

KOKKOS_INLINE_FUNCTION
Real HydroAtmosphereInternalEnergy(const EOS_Data &eos) {
  bool efloor_used = false;
  bool tfloor_used = false;
  return ApplyHydroThermalFloors(eos, eos.dfloor, static_cast<Real>(0.0),
                                 efloor_used, tfloor_used);
}

KOKKOS_INLINE_FUNCTION
bool NeedsHydroAtmosphereReset(const EOS_Data &eos, const HydCons1D &u) {
  return (u.d < eos.dfloor) ||
         ((u.d == eos.dfloor) &&
          ((u.mx != 0.0) || (u.my != 0.0) || (u.mz != 0.0)));
}

KOKKOS_INLINE_FUNCTION
void ResetHydroAtmosphereState(const EOS_Data &eos, HydCons1D &u, HydPrim1D &w,
                               bool &dfloor_used) {
  const Real atmosphere_eint = HydroAtmosphereInternalEnergy(eos);
  u.d = eos.dfloor;
  u.mx = 0.0;
  u.my = 0.0;
  u.mz = 0.0;
  u.e = atmosphere_eint;
  w.d = u.d;
  w.vx = 0.0;
  w.vy = 0.0;
  w.vz = 0.0;
  w.e = atmosphere_eint;
  dfloor_used = true;
}

KOKKOS_INLINE_FUNCTION
void ApplyHydroVelocityCeiling(const EOS_Data &eos, HydPrim1D &w, bool &vceil_used) {
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
void SingleC2P_GeneralHyd(HydCons1D &u, const EOS_Data &eos, HydPrim1D &w,
                          bool &dfloor_used, bool &efloor_used, bool &tfloor_used,
                          bool &vceil_used) {
  if (NeedsHydroAtmosphereReset(eos, u)) {
    ResetHydroAtmosphereState(eos, u, w, dfloor_used);
    return;
  }
  w.d = u.d;

  const Real di = 1.0/u.d;
  w.vx = di*u.mx;
  w.vy = di*u.my;
  w.vz = di*u.mz;
  w.e = u.e - 0.5*di*(SQR(u.mx) + SQR(u.my) + SQR(u.mz));

  w.e = ApplyHydroThermalFloors(eos, w.d, w.e, efloor_used, tfloor_used);
  ApplyHydroVelocityCeiling(eos, w, vceil_used);

  SingleP2C_GeneralHyd(w, u);
}

// The primitives a recovery leaves in w0.  A recovery that floors or limits a cell writes
// the primitive state back as the cell's conserved state, and the inversion of that state
// is not the identity at round-off (E_K is formed from m^2/(2 rho) here and from rho v^2/2
// there).  A run's w0 must be what a restart computes from the checkpointed u0, or the
// step after a restart starts from different owned primitives with no stored copy to say
// otherwise.  So the cell is inverted again from the state it was written back as, and
// again while that inversion limits it, the conserved state staying the first write-back:
// w0 = c2p(u_written), the restart's own recovery of the checkpoint.  A restart's first
// pass is this loop's second, and the loop is a fixed point of the write-back, so both
// end on the same primitives (the cap is never reached: the write-back of a limited state
// is limited at most once more).  `pass(u, aux, w, aux_out)` inverts u as the recovery
// does, leaves u as the recovery writes it back, and returns whether it did.  Hydro's
// recoveries and the Newtonian MHD dual-energy ones (Cons = MHDCons1D, which carries the
// cell-centred field the inversion leaves unchanged) settle alike.
inline constexpr int kHydroSettlePasses = 8;

template <class Pass, class Cons, class Prim>
KOKKOS_INLINE_FUNCTION
void SettleHydroPrimitives(const Pass &pass, Cons u, const Real aux_written,
                           Prim &w, Real &aux_w) {
  Real aux = aux_written;
  aux_w = aux;
  for (int it = 0; it < kHydroSettlePasses; ++it) {
    Prim wn;
    Real aux_out = aux;
    const bool written = pass(u, aux, wn, aux_out);
    w = wn;
    aux = aux_out;
    aux_w = aux_out;
    if (!written) break;
  }
}

// The Bryan et al. eta1 test of the Newtonian hydro dual-energy inversion: the energy
// channel E - E_K is the internal energy only where it is a resolvable fraction of E.  In
// a hypersonic flow E - E_K carries the truncation error of E and E_K, a fixed fraction
// of E, which there is as large as the thermal energy itself; the auxiliary, which sees
// only advection and p dV (and the energy channel where the cell is shocked, the shock
// rule below), is the thermal state.  The same test decides the inversion,
// the owner's decision before the exchange (Hydro::ResolveDualEnergyChannel) and the
// repair test, so the auxiliary is never overwritten by a channel the inversion rejects.
KOKKOS_INLINE_FUNCTION
bool HydroEnergyChannelResolvesEint(const Real eint_cons, const Real etot,
                                    const Real dual_eta1) {
  // eint_cons = E - E_K > 0 implies E > 0: a pure ratio, with no absolute energy scale
  // (a floor on E would reject exact cold gas at rest wherever E fell below it).
  return (eint_cons > 0.0) &&
         ((dual_eta1 <= 0.0) || (eint_cons > dual_eta1*etot));
}

//----------------------------------------------------------------------------------------
// The dual-energy shock rule, one statement for hydro and MHD.
//
// Shock heat is entropy, which only the energy channel records: the auxiliary sees
// advection and p dV and follows the adiabat through a shock.  So a cell that is being
// shocked takes its thermal state from the energy channel, eint_cons = E - E_K (- E_M),
// whatever the eta1 test (a bound on the truncation error of smooth flow) or the MHD
// magnetic test says, provided eint_cons is resolved above its own rounding error.  The
// channel is taken as it is, above or below the auxiliary: a choice that kept only the
// stages whose E - E_K exceeds the auxiliary would rectify the stage-to-stage scatter of
// the RK update and heat the shocked gas (1.4-3x the Rankine-Hugoniot entropy in a 1D
// model of the hydro update at a bulk Mach number of 300-1000).
//
// Hydro and MHD apply this rule with one implementation (eos/dual_energy_shock_rule.hpp,
// called by Hydro::ApplyDualEnergyFormalism and MHD::ApplyDualEnergyFormalism); they
// differ only in the field: E_M in the energy split, the channel of MHD read against
// the field E's fluxes carried (MHDFieldCarriedByE, mhd_dual_energy.cpp), and the MHD
// families of the steady test below.  With B = 0 MHD reproduces hydro
// (tst/test_suite/nr/test_nr_mhd_dual_hydro_parity_cpu.py).
//
// Both decide which cells are shocked once per step, at the first stage on the
// step-start state, and hold the decision through the step.  A stage's E - E_K is
// not a thermal energy on its own: the energy update does the work of the stage-start
// velocity, so E - E_K misses the kinetic energy rho |dv|^2 / 2 of the cell's velocity
// change over the stage, and an averaging stage adds the kinetic energy that mixing the
// time levels dissipates.  Over a whole RK step the two cancel to the order of the
// scheme; taken at some stages and not at others they leave a bias of about
// rho |dv|^2 / 2 (1D model of the update: taking the channel only at the averaging stage
// gives 1.4-2.9x the Rankine-Hugoniot entropy at a bulk Mach number of 300-1000;
// deciding per stage loses 4 and 7 % of it on the tail side, see DualEnergyShockReach).
//
// The step's last stage takes the step's channel only if it has not fallen
// below the auxiliary as the first stage's channel left it: a shock never lowers the
// entropy, so a cell whose step-final channel lies below that was not heated by a shock
// in that step, whatever its stencil says, and it returns to the adiabat it would have
// followed without the channel of the step's earlier stages.  The reference carries the
// first stage's channel and no other: that of a later, averaging stage holds the mixing
// heat of the time levels, and rk3 with it put the reference above the step's channel
// in real shocks (colliding streams at 300 and 1000 c, nghost = 2: K/K_RH 0.89 and 0.41
// on the tail side, against 0.977 and 0.944 without it); for a two-stage step the
// reference is the auxiliary itself.  With PLM the condition selects nothing in a real
// shock (colliding +-10 c streams at 0-1000 c keep every printed digit with it and
// without it, rk2 and rk3); applied at every stage it would rectify the stages' errors
// (1.4-2.9x the Rankine-Hugoniot entropy), and compared with the bare adiabat it
// rectifies the scatter of the smeared front (K/K_RH 1.09 and 1.16 with rk3).  It guards
// the q = 1e4 TDE restart: at the stream head near the black hole (bulk Mach number
// 100-4000) the stream is one to two level-7 cells thick in z, density 1e-7 between gas
// at 1e-11 that falls onto it from both sides at about seven of its sound speeds; the
// stencil test flags its cells step after step while their channel falls about 0.4 % of
// the thermal energy per stage below the adiabat (the frame-dependent truncation error of
// E - E_K, larger than the heat the infalling gas brings, about 0.1 % per step); taken
// as it was, patches fell to the temperature floor.  The condition is per cell, and a
// captured shock's dissipation is not positive cell by cell where a high-order
// reconstruction spreads it with signed weights: with PPM4 and rk3 or rk4 the front
// cells of a shock smeared at 300 c return to the adiabat step after step and the
// inversion creates the difference (hydro_dshock_dual with PPM4: K/K_RH 1.56/1.04 rk3
// and 1.50/1.04 rk4, conservative 0.988/0.996 and 0.996/0.996; MHD the same, 1.52-1.65
// against 0.977-0.985).  No dual-energy deck runs that combination (all use PLM and
// rk2).  MHD once took the channel at every stage, the step's last included: that
// passed every small deck, but the hydro form of it put about ten times more cold
// outliers on the TDE stream head, so both fluids keep the guard.
//
// A cell is shocked when gas converges into it faster than sound, measured two ways:
// - across its own stencil, converging_dv = sum_d [v_d(-1) - v_d(+1)] faster than the
//   auxiliary's sound speed, which no shock heat has polluted;
// - into the steady compression it sits in (DualEnergySteadyInflowSpeed), faster than a
//   shock whose velocity jump equals the upstream sound speed (DualEnergyShockMachSq).
//   MHD measures that inflow against each family's characteristic speed, fast and slow,
//   with the same jump on the gas sound speed (DualEnergyShockFamilyMachSq,
//   DualEnergyAddShockInflow in eos/dual_energy_shock_rule.hpp).
// The first sees sharp and forming shocks only.  An upwind scheme spreads a shock over
// about |v_grid|/dv cells, because its numerical viscosity grows with the speed of the
// gas across the grid: two colliding +-10 c streams moving at 0, 300 and 1000 c gave
// shocks 2, 6 and 11 cells wide, a stencil velocity jump of 8.7, 3.6 and 2.1 c, and the
// first test alone lost 15-33 % of their entropy (1D model).  The second is independent
// of that width: in a steady compression the mass flux rho (v - s) is the same, so two
// points of the stencil give its speed s, and the flow into it is |v - s| at the
// upstream-most point, where the gas is coolest.  A compression wave has |v - s| = c to
// first order in its amplitude and is left to the auxiliary: the scheme damps such a
// wave at a rate proportional to the bulk speed, and its heat is not physical.  Where
// nghost allows, a converging cell that has a shocked cell within reach on either side
// along an axis is shocked too (DualEnergyShockReach): the post-sonic tail an upwind
// scheme puts beside the front.
//
// The rounding bound of eint_cons: E, every momentum component and rho leave the stage
// update q = g0 q0 + g1 q1 - beta dt div F as sums of 2 + 2d terms of at most E + E_K
// each (d <= 3 dimensions, CFL <= 1), i.e. with at most k = 10 roundings of that size;
// E_K = |m|^2/(2 rho) doubles the relative error of m and adds that of rho and its own 4
// roundings.  So |error| <= (k + 3k + 4 + 1) (eps/2) (E + E_K) < 23 eps (E + E_K), taken
// as 32.  (Measured: about 0.2 eps (E + E_K) per stage, RMS, in an entropy wave carried
// at 1000 c with the energy channel always taken.)  E_M enters like E_K.
inline constexpr Real kDualEnergyChannelRoundingEps = 32.0;

KOKKOS_INLINE_FUNCTION
bool DualEnergyChannelResolved(const Real eint_cons, const Real magnitude_sum) {
  return eint_cons >
         kDualEnergyChannelRoundingEps*std::numeric_limits<Real>::epsilon()*magnitude_sum;
}

// Gas converging across the cell's own stencil, converging_dv = sum_d [v_d(-1) -
// v_d(+1)], faster than sound (cs2 of the auxiliary).
KOKKOS_INLINE_FUNCTION
bool DualEnergyStencilConvergesSupersonically(const Real converging_dv, const Real cs2) {
  return (converging_dv > 0.0) && (converging_dv*converging_dv > cs2);
}

// The squared steady inflow speed, in units of the upstream GAS sound speed c_s, of a
// shock of a family k whose velocity jump equals c_s.  The jump across a family-k shock
// in the hydro form, dv = 2 (u^2 - c_k^2)/((Gamma_k + 1) u), with the family's upstream
// characteristic speed c_k along the normal and its adiabatic index Gamma_k, set equal
// to c_s gives u = c_s (q + sqrt(q^2 + c_k^2/c_s^2)), q = (Gamma_k + 1)/4.  The gas
// shock has c_k = c_s (DualEnergyShockMachSq); MHD's fast and slow shocks take their own
// speeds (eos/dual_energy_shock_rule.hpp).
KOKKOS_INLINE_FUNCTION
Real DualEnergyShockFamilyMachSq(const Real gamma_k, const Real ck2_over_cs2) {
  const Real q = 0.25*(gamma_k + 1.0);
  const Real mach = q + sqrt(q*q + ck2_over_cs2);
  return mach*mach;
}

// The squared Mach number of a shock whose velocity jump equals its upstream sound
// speed, for adiabatic index gamma1 = rho c^2/p of the upstream gas: the Rankine-Hugoniot
// jump dv = 2 c (M^2 - 1)/((gamma1 + 1) M) set equal to c gives
// M = q + sqrt(q^2 + 1), q = (gamma1 + 1)/4 (1.87 for gamma1 = 5/3).
KOKKOS_INLINE_FUNCTION
Real DualEnergyShockMachSq(const Real gamma1) {
  return DualEnergyShockFamilyMachSq(gamma1, 1.0);
}

// The flow into the steady compression a cell sits in, along one axis, from the density
// and velocity component at the stencil points -2..+2 (index 0..4, cell at 2).  Returns
// false unless the gas converges across both the inner (-1, +1) and the outer (-2, +2)
// pair and the outer density difference resolves the speed of the compression,
// s = [rho v]/[rho] over the outer pair (constant mass flux rho (v - s) through a steady
// structure), to a tenth of the sound speed c_ref: its rounding, 2 eps (|rho v|_- +
// |rho v|_+)/|[rho]|, then moves the Mach number by less than a sixth of the smallest
// margin of DualEnergyShockMachSq above 1 (0.62, gamma1 = 1).  Otherwise inflow_speed =
// |v - s| at the stencil point where it is largest (the upstream-most, coolest gas of a
// steady profile), upstream is that point and rho_upstream its density.
KOKKOS_INLINE_FUNCTION
bool DualEnergySteadyInflowSpeed(const Real rho[5], const Real vel[5], const Real c_ref,
                                 Real &inflow_speed, int &upstream, Real &rho_upstream) {
  if (!((vel[1] > vel[3]) && (vel[0] > vel[4]))) return false;
  const Real drho = rho[0] - rho[4];
  const Real flux_m = rho[0]*vel[0];
  const Real flux_p = rho[4]*vel[4];
  const Real rounding = 2.0*std::numeric_limits<Real>::epsilon()*
                        (fabs(flux_m) + fabs(flux_p));
  if (!(fabs(drho)*0.1*c_ref > rounding)) return false;
  const Real speed = (flux_m - flux_p)/drho;
  inflow_speed = 0.0;
  upstream = 0;
  rho_upstream = rho[0];
  for (int p = 0; p < 5; ++p) {
    if (p == 2) continue;
    const Real u = fabs(vel[p] - speed);
    if (u > inflow_speed) {
      inflow_speed = u;
      upstream = p;
      rho_upstream = rho[p];
    }
  }
  return true;
}

// How far the shock test looks for the shock a converging cell belongs to
// (DualEnergyCellShocked, eos/dual_energy_shock_rule.hpp): a converging cell is shocked
// also when a cell up to this many cells away from it on either side along an axis
// passes the test.  An upwind
// scheme at a bulk speed |v_grid| >> c puts part of a shock's dissipation beside the
// cells whose stencil sees the supersonic inflow: the shock's post-sonic tail, 1-8 cells
// from the front, upwind of it on the grid where the shock overtakes its upstream gas
// from behind and downwind where it moves slower than its gas (the reach first looked
// downwind of the grid flow only; MHD PPM4 at 300 c lost 1-3 % of the conservative
// scheme's entropy on that side, hydro no printed digit).  The tail's gas still converges
// and heats, but every point of its stencil is already shocked gas, which flows in at
// 1.1-1.7 sound speeds relative to the stencil's steady speed, below M_c (a threshold low
// enough to admit it takes the damping heat of compression waves: M_c = 1 took 30 % of
// it).  Colliding +-10 c streams carried at 300 and 1000 c lose 0.5 and 3 % of the
// Rankine-Hugoniot entropy there (K/K_RH 0.995, 0.972; 0.961, 0.932 with the shock test
// decided per stage; the other side 0.99-1.01).  The front is visible from the tail only
// through its neighbours, whose test reads two cells further, so the reach is what nghost
// leaves, nghost - 2.  1D model of the update, U0 = 300 / 1000, K/K_RH on that side:
// reach 1 1.007 / 0.993, 2 1.009 / 1.006, 3 1.009 / 1.012, 4 1.009 / 1.015 (at 300 the
// conservative scheme's 1.009); the other side 0.991-1.011.  With nghost = 2 (every PLM
// deck) the reach is 0 and the loss above remains.
KOKKOS_INLINE_FUNCTION
int DualEnergyShockReach(const int nghost) {
  return nghost - 2;
}

// A rejected energy channel must not be kept: the caller writes the returned u.e =
// eint_aux + E_K back wherever eint_from_aux is set.  Left alone, E integrates the
// truncation error of every stage, nothing else reads it, and the residual grows until it
// passes eta1, when the whole of it is released as heat in that cell at once; in a cold
// supersonic stream this makes the thermal state flip between the two channels from cell
// to cell.  E is therefore not conserved in a rejected cell, by exactly the residual that
// is not thermal energy (as in MHD, general_c2p_mhd.hpp).
//
// energy_channel_readable: the cell's own recovery (an owned cell) may take E - E_K.  A
// ghost zone or coarse buffer holds a state assembled from owners' states (a copy, a
// restriction, a LAT time interpolation, a prolongation); its auxiliary is the matching
// combination of their thermal energies, its E - E_K adds the kinetic-energy error of the
// combination, so it reads the auxiliary (Hydro::ResolveDualEnergyChannel).
//
// eint_aux_out is the published thermal energy, the auxiliary after every recovery.
KOKKOS_INLINE_FUNCTION
void SingleC2P_GeneralHydDual(HydCons1D &u, const EOS_Data &eos, const Real eint_aux_in,
                              const Real dual_eta1, const bool energy_channel_readable,
                              HydPrim1D &w, Real &eint_aux_out,
                              bool &eint_from_aux, bool &dfloor_used, bool &efloor_used,
                              bool &tfloor_used, bool &vceil_used) {
  if (NeedsHydroAtmosphereReset(eos, u)) {
    ResetHydroAtmosphereState(eos, u, w, dfloor_used);
    eint_aux_out = w.e;
    eint_from_aux = false;
    return;
  }
  w.d = u.d;

  const Real di = 1.0/u.d;
  w.vx = di*u.mx;
  w.vy = di*u.my;
  w.vz = di*u.mz;

  const Real eint_cons = u.e - 0.5*di*(SQR(u.mx) + SQR(u.my) + SQR(u.mz));
  Real eint_floor, eint_ceil;
  eos.HydroInternalEnergyDensityBounds(w.d, eint_floor, eint_ceil);
  Real eint_aux = ApplyHydroThermalFloorsWithBounds(eos, w.d, eint_aux_in, eint_floor,
                                                    eint_ceil, efloor_used,
                                                    tfloor_used);

  const bool use_cons_e = energy_channel_readable &&
                          HydroEnergyChannelResolvesEint(eint_cons, u.e, dual_eta1);
  eint_from_aux = !use_cons_e;
  w.e = use_cons_e ? eint_cons : eint_aux;
  w.e = ApplyHydroThermalFloorsWithBounds(eos, w.d, w.e, eint_floor, eint_ceil,
                                          efloor_used, tfloor_used);

  ApplyHydroVelocityCeiling(eos, w, vceil_used);
  SingleP2C_GeneralHyd(w, u);
  eint_aux_out = w.e;
}

}  // namespace eos_general

#endif  // EOS_GENERAL_C2P_HYD_HPP_
