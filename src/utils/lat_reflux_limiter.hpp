#ifndef UTILS_LAT_REFLUX_LIMITER_HPP_
#define UTILS_LAT_REFLUX_LIMITER_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file lat_reflux_limiter.hpp
//! \brief Admissibility limiter for the LAT window-end conservative reflux.
//!
//! The window-end reflux replaces a coarse block's own face flux by the time-summed
//! flux its finer neighbor actually used.  The two can differ by far more than the
//! coarse cell holds when the atmosphere floor reseeded that cell partway through the
//! window: the floor invented mass the flux history never carried, so subtracting the
//! true fine-side transport drives the corrected state negative.  This picks the
//! largest fraction of the pending mismatch the cell can take and still invert.
//!
//! The fraction is a single scalar per face cell, shared by every conserved variable of
//! that cell, so the correction is scaled as a whole state and never variable by
//! variable.  Whatever the cell declines is handed to the neighbor behind it, which is
//! an ordinary interior cell of the same block, and the same test is applied there: the
//! pair's sum is the unlimited correction whenever that second cell can take its share,
//! which is the case the limiter was written for.  When it cannot either, the residue is
//! dropped, because an inadmissible state has no inversion at all and a conservative
//! reflux that cannot be inverted is worth less than a small accounting loss.  A cell
//! that can take the whole mismatch returns exactly 1.0 and is left bit-for-bit alone.
//!
//! Both fractions are recorded, because the force-free sidecar refluxes its own four
//! tails across the same face and must scale them by the fraction the conserved state of
//! THAT cell took.  A tail and the conserved state it describes are one physical state:
//! the cold closure divides the transported EM momentum by the same cell's |B|^2, so
//! moving different fractions of one face mismatch into the two systems can put the pair
//! outside the drift cone even though both inputs were inside it.

#include <math.h>

#include "athena.hpp"
#include "eos/eos.hpp"

namespace lat_reflux {

//----------------------------------------------------------------------------------------
//! Components of the gravitational-work companion of the pending reflux
//! (Hydro::lat_grav_reflux, MHD::lat_grav_reflux), one register per coarse/fine face
//! cell.  The source term in SourceTerms::Gravity gates the work of each stage's mass
//! flux on that stage's primitive density, while the window-end refund sees only the
//! summed mismatch, so the stage that took the flux has to record how much of it the
//! gate refused.  Storing the refused part rather than the admitted part keeps a face
//! whose gate never closed at exactly zero, so the refund reads the mass accumulator
//! unchanged.

enum GravWorkComponent {
  kGravWorkSeen = 0,         // mass mismatch the gate has already been asked about
  kGravWorkBlockedSelf = 1,  // part of it the self-gravity gate refused
  kGravWorkBlockedBH = 2,    // part of it the external-BH gate refused
  kGravWorkApplied = 3,      // work the window-end refund paid, for the limiter to scale
  kNGravWork = 4
};

//----------------------------------------------------------------------------------------
//! \enum EnergyForm
//! \brief What the energy slot IEN of the conserved state holds on each backend.
//!
//!   kNewtonian  total energy density (kinetic + magnetic + internal)
//!   kTauFlat    SR tau = E - D, with the flat-space inversion bound E >= |S|
//!   kTauCurved  dynamical-GR tau = sqrt(gamma)(E - D), non-negative for every state the
//!               inversion accepts
//!   kTttPlusD   fixed-metric GR T^t_t + D (ideal_c2p_hyd.hpp, ideal_c2p_mhd.hpp), the
//!               rest mass less the energy at infinity: negative for gas at rest in flat
//!               space and of either sign once the gas is bound, so no sign test on it
//!               is an inversion bound and only D carries a test

enum class EnergyForm { kNewtonian, kTauFlat, kTauCurved, kTttPlusD };

//! \fn EnergyForm EnergyFormOf
//! \brief The IEN convention selected by the coordinate flags.  The dynamical backend
//! clears both the SR and the GR flag (coordinates.cpp), so it is tested first.
inline EnergyForm EnergyFormOf(const bool dynamical_rel, const bool general_rel,
                               const bool special_rel) {
  if (dynamical_rel) return EnergyForm::kTauCurved;
  if (general_rel) return EnergyForm::kTttPlusD;
  if (special_rel) return EnergyForm::kTauFlat;
  return EnergyForm::kNewtonian;
}

//----------------------------------------------------------------------------------------
//! \fn bool Admissible
//! \brief Whether one conserved state can be inverted without tripping a floor.
//!
//! Newtonian gas: rest-mass density above the density floor and, with an energy
//! variable, internal energy above the recovery's internal-energy floor once kinetic and
//! magnetic energy are removed.  Relativistic gas: conserved density above the density floor, and where
//! the energy slot holds tau, non-negative tau, with the flat-space inversion bound
//! added under SR.  Every test uses the whole corrected state.  The dynamical metric does
//! not use this test for the reflux: its kTauCurved branch only bounds D and tau, which
//! the residual of a near-total reflux (a difference of nearly equal large numbers)
//! passes while the recovery cannot invert it, so MHD::ApplyLATCorrections takes the
//! fractions of DynGRMHD::LATRefluxFractions, the recovery's own energy-root test.

KOKKOS_INLINE_FUNCTION
bool Admissible(const EOS_Data &eos, const EnergyForm form, const Real cons[],
                const int nvar, const Real emag) {
  const Real dens = cons[IDN];
  if (!(dens >= eos.dfloor)) return false;
  if (!eos.is_ideal) return true;
  if (nvar <= IEN) return true;
  // Only D is bounded here for the fixed-metric GR energy slot (T^t_t + D has no sign
  // bound); those backends reflux with lat_reflux_fixedgr.hpp instead.
  if (form == EnergyForm::kTttPlusD) return true;
  const Real etot = cons[IEN];
  if (form != EnergyForm::kNewtonian) {
    // cons[IEN] is tau, the conserved energy less the conserved density.  A negative
    // tau has no primitive state behind it under any of the inversions in this code.
    if (!(etot >= 0.0)) return false;
    if (form == EnergyForm::kTauFlat) {
      const Real ssq = cons[IM1]*cons[IM1] + cons[IM2]*cons[IM2] + cons[IM3]*cons[IM3];
      if (!(etot + dens >= sqrt(dens*dens + ssq))) return false;
    }
    return true;
  }
  const Real ekin = 0.5*(cons[IM1]*cons[IM1] + cons[IM2]*cons[IM2] +
                         cons[IM3]*cons[IM3])/dens;
  const Real eint = etot - ekin - emag;
  // The recovery's own internal-energy floor: pfloor/(gamma - 1) (or the entropy floor)
  // for the gamma law, the table's lowest temperature at this density for the LTE table.
  return (eint >= eos.HydroInternalEnergyDensityFloor(dens));
}

//----------------------------------------------------------------------------------------
//! \fn Real AdmissibleFractionBy
//! \brief Largest fraction of a pending conserved correction a cell can absorb, for the
//! admissibility test `test(trial)` of the backend (a callable on a conserved state).
//!
//! Returns exactly 1.0 whenever the unlimited correction is already admissible, so a
//! run that never reseeds a coarse cell next to a fine neighbor is untouched.  When it
//! is not, the admissible set is bracketed from the uncorrected state, which the last
//! conserved-to-primitive pass left admissible, and bisected.  A state that was already
//! inadmissible before the correction is past this limiter's reach, and takes the
//! unlimited correction so the delayed flux stays exact there.

//! The longest state a test takes: the conserved variables and the four tails of the
//! hybrid force-free sidecar.
constexpr int kMaxTestState = IEN + 1 + 4;

template <class Test>
KOKKOS_INLINE_FUNCTION
Real AdmissibleFractionBy(const Test &test, const Real cons[], const Real dcons[],
                          const int ntest) {
  Real trial[kMaxTestState];
  for (int n = 0; n < ntest; ++n) trial[n] = cons[n] + dcons[n];
  if (test(trial)) return 1.0;
  if (!test(cons)) return 1.0;

  Real lo = 0.0, hi = 1.0;
  for (int it = 0; it < 50; ++it) {
    const Real mid = 0.5*(lo + hi);
    for (int n = 0; n < ntest; ++n) trial[n] = cons[n] + mid*dcons[n];
    if (test(trial)) {
      lo = mid;
    } else {
      hi = mid;
    }
  }
  return lo;
}

//! The Admissible() test of one cell as a callable for AdmissibleFractionBy.
struct AdmissibleTest {
  const EOS_Data &eos;
  EnergyForm form;
  int nvar;
  Real emag;
  KOKKOS_INLINE_FUNCTION bool operator()(const Real state[]) const {
    return Admissible(eos, form, state, nvar, emag);
  }
};

//----------------------------------------------------------------------------------------
//! \fn Real AdmissibleFraction
//! \brief Largest fraction of a pending conserved correction the cell can absorb, for
//! the Newtonian, SR and tau-form tests of Admissible() above (AdmissibleFractionBy).
//!
//! Two backends do not use it: the dynamical metric takes the recovery's own energy-root
//! test (DynGRMHD::LATRefluxFractions), and the fixed-metric GR backends
//! (EnergyForm::kTttPlusD) take the single-cell inversion of their own recovery on the
//! cell's metric (FixedGRHydroRefluxFraction, FixedGRMHDRefluxFraction in
//! lat_reflux_fixedgr.hpp), because the energy slot T^t_t + D has no sign bound for
//! Admissible() to test.

KOKKOS_INLINE_FUNCTION
Real AdmissibleFraction(const EOS_Data &eos, const EnergyForm form, const Real cons[],
                        const Real dcons[], const int nvar, const Real emag) {
  const int ntest = (nvar > IEN) ? (IEN + 1) : (IM3 + 1);
  return AdmissibleFractionBy(AdmissibleTest{eos, form, nvar, emag}, cons, dcons, ntest);
}

//----------------------------------------------------------------------------------------
//! The recovery's verdict on one conserved state, as the bits of an int: it inverts by its
//! energy root, it had to floor U first, a ceiling (Lorentz factor, sigma, b^2/u) acted on
//! the result.  The last two are the repairs the recovery makes routinely.
constexpr int kRecoveryInverts = 1;
constexpr int kRecoveryFloored = 2;
constexpr int kRecoveryLimited = 4;

//! Relative distance from a bound within which a state counts as sitting on it.  The
//! recovery republishes U of every floored or ceiling-limited cell at the bound, and
//! inverting that U again reaches the bound only to the accuracy of the round trip
//! (a root tolerance of 1e-15 amplified by the Lorentz factor squared, W^2 <= 2.5e3 for
//! the production ceilings), so an exact comparison with the bound is decided by rounding.
//! 1e-8 is far above that and far below any distance that separates two physical states.
constexpr Real kRepublishTolerance = 1.0e-8;

//----------------------------------------------------------------------------------------
//! \fn Real RecoveryAdmissibleFraction
//! \brief Largest fraction of a pending conserved correction the dynamical-metric
//! recovery can take (tau is bounded when `bounds_tau`, an ideal gas), for `verdict(f)`, the recovery's own verdict (kRecovery* bits) on the
//! state cons + f dcons (densitized, as Admissible() takes it).
//!
//! The correction is limited to what the recovery inverts by its energy root, with
//! D >= dfloor and tau >= 0 kept (Admissible(), the test of the other backends, which
//! passes the residual of a near-total reflux that the recovery cannot invert, so it
//! cannot stand alone).  A floor or a ceiling that the uncorrected state is already on is
//! the recovery's routine repair of that cell, not a reason to decline the correction:
//! the state counts as on a bound when the verdict of cons or of cons moved by
//! kRepublishTolerance of its size along the correction raises it, which does not depend
//! on how a cell republished at the bound rounds.  A trial may raise a flag only then.
//! Where the uncorrected state itself cannot be inverted the recovery's verdict says
//! nothing about the correction, and the D and tau bounds alone decide, as for the
//! other backends.  Exactly 1.0 when the whole correction is admissible.  The state is
//! the first `ntest` entries of cons (the conserved variables, then any tails the
//! verdict reads; at most kMaxTestState).

template <class Verdict>
KOKKOS_INLINE_FUNCTION
Real RecoveryAdmissibleFraction(const Verdict &verdict, const Real dfloor,
                                const bool bounds_tau, const Real cons[],
                                const Real dcons[], const int ntest = IEN + 1) {
  auto bounded_state = [&](const Real state[]) {
    return state[IDN] >= dfloor*(1.0 - kRepublishTolerance) &&
           (!bounds_tau || state[IEN] >= 0.0);
  };
  auto bounded = [&](const Real f) {
    Real state[kMaxTestState];
    for (int n = 0; n < ntest; ++n) state[n] = cons[n] + f*dcons[n];
    return bounded_state(state);
  };
  constexpr int kRepairs = kRecoveryFloored | kRecoveryLimited;
  // One call site of the verdict: the stages below are the cases of one loop, so the device
  // code carries one argument copy of the out-of-line inversion instead of one per case.
  enum Stage { kFull, kZero, kOnBound, kAdmFull, kAdmZero, kBisect };
  Stage stage = kFull;
  Real f = 1.0, lo = 0.0, hi = 1.0;
  int verdict0 = 0, on_bound = 0, iteration = 0;
  while (true) {
    const bool evaluate = (stage == kZero || stage == kOnBound) ? true : bounded(f);
    const int v = evaluate ? verdict(f) : -1;
    const bool admissible = evaluate && (v & kRecoveryInverts) != 0 &&
                            (v & kRepairs & ~on_bound) == 0;
    switch (stage) {
      case kFull:
        // A full correction that inverts with no repair is admissible whatever the bound
        // the uncorrected state sits on (the common case: one verdict).
        if (evaluate && (v & kRecoveryInverts) && !(v & kRepairs)) return 1.0;
        stage = kZero;
        f = 0.0;
        break;
      case kZero:
        verdict0 = v;
        if (!(verdict0 & kRecoveryInverts)) {
          return AdmissibleFractionBy(bounded_state, cons, dcons, ntest);
        } else {
          Real size = 0.0, step = 0.0;
          for (int n = 0; n < ntest; ++n) {
            size += fabs(cons[n]);
            step += fabs(dcons[n]);
          }
          f = (step > kRepublishTolerance*size) ? kRepublishTolerance*size/step :
                                                  static_cast<Real>(1.0);
        }
        stage = kOnBound;
        break;
      case kOnBound:
        on_bound = (verdict0 | v) & kRepairs;
        stage = kAdmFull;
        f = 1.0;
        break;
      case kAdmFull:
        if (admissible) return 1.0;
        stage = kAdmZero;
        f = 0.0;
        break;
      case kAdmZero:
        if (!admissible) return 1.0;
        stage = kBisect;
        f = 0.5*(lo + hi);
        break;
      case kBisect:
        if (admissible) {
          lo = f;
        } else {
          hi = f;
        }
        if (++iteration == 50) return lo;
        f = 0.5*(lo + hi);
        break;
    }
  }
}

//----------------------------------------------------------------------------------------
//! \fn Real GasEntropyCorrection
//! \brief The window-end correction of the GR gas-entropy column (u0 = D s) of one cell:
//! the corrected mass d_dens at the cell's own specific entropy s = dens_s/dens.
//!
//! The column is resynchronized from the published state at the end of every step
//! (KORAL's update_entropy, Hydro/MHD::ResyncGasEntropy), so at the window end a coarse
//! cell no longer holds the entropy its own face flux moved during the window: the mass
//! that flux brought in carries the cell's s now, not the s it arrived with, as a cell
//! the atmosphere floor reseeded no longer holds what the flux history moved
//! (AdmissibleFraction).  The transported mismatch sum(F_fine s) - F_coarse s, added to
//! that state, removes the coarse-moved mass at the wrong entropy.  Where the correction
//! returns the density to near its value before the step (a vacuum cell beside a fine
//! region that a star's surface crosses, whose coarse step moved ten times the cell's
//! mass), (D s)'/D' is the quotient of two unrelated corrections, s ~ 1e2, and the
//! window-end recovery's entropy root turns it into a 1e14 pressure.  Moving mass in or
//! out of a cell does not change the specific entropy of the gas it holds, so the
//! corrected column is D' s; the window-end recovery then publishes s from its state.  A
//! cell without a positive density keeps the transported correction.

KOKKOS_INLINE_FUNCTION
Real GasEntropyCorrection(const Real dens, const Real dens_s, const Real d_dens,
                          const Real transported) {
  if (dens > 0.0 && isfinite(dens_s)) return (dens_s/dens)*d_dens;
  return transported;
}

} // namespace lat_reflux

#endif // UTILS_LAT_REFLUX_LIMITER_HPP_
