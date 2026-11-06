#ifndef EOS_IDEAL_C2P_HYD_HPP_
#define EOS_IDEAL_C2P_HYD_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file ideal_c2p_hyd.hpp
//! \brief Various inline functions that transform a single state of conserved variables
//! into primitive variables (and the reverse, primitive to conserved) for hydrodynamics
//! with an ideal gas EOS. Versions for both non-relativistic and relativistic fluids are
//! provided.

#include "coordinates/cartesian_ks.hpp"
#include <limits>

#include "eos/gr_gas_entropy.hpp"

//----------------------------------------------------------------------------------------
//! \fn Real IdealHydroEintFloor()
//! \brief Returns the larger of the pressure and entropy floors in internal-energy form.

KOKKOS_INLINE_FUNCTION
Real IdealHydroEintFloor(const EOS_Data &eos, const Real dens) {
  const Real gm1 = eos.gamma - 1.0;
  const Real eint_floor = eos.pfloor/gm1;
  const Real entropy_floor = (eos.sfloor > 0.0) ?
      (eos.sfloor*pow(dens, eos.gamma)/gm1) : 0.0;
  return fmax(eint_floor, entropy_floor);
}

//----------------------------------------------------------------------------------------
//! \fn Real IdealGRHydroEintFloor()
//! \brief The lower bound the relativistic hydro inversion holds the internal energy
//! density to at density dens: the pressure and entropy floors, and T = p/rho >= tfloor,
//! as IdealGRMHDEintFloor() does for MHD.

KOKKOS_INLINE_FUNCTION
Real IdealGRHydroEintFloor(const EOS_Data &eos, const Real dens) {
  return fmax(IdealHydroEintFloor(eos, dens), dens*eos.tfloor/(eos.gamma - 1.0));
}

KOKKOS_INLINE_FUNCTION
Real IdealHydroAtmosphereInternalEnergy(const EOS_Data &eos) {
  const Real gm1 = eos.gamma - 1.0;
  Real eint = IdealHydroEintFloor(eos, eos.dfloor);
  if ((eos.tfloor > 0.0) && (gm1*eint/eos.dfloor < eos.tfloor)) {
    eint = eos.dfloor*eos.tfloor/gm1;
  }
  const Real eceil = eos.HydroInternalEnergyDensityCeiling(eos.dfloor);
  if (eint > eceil) {
    eint = eceil;
  }
  return eint;
}

KOKKOS_INLINE_FUNCTION
bool NeedsIdealHydroAtmosphereReset(const EOS_Data &eos, const HydCons1D &u) {
  return (u.d < eos.dfloor) ||
         ((u.d == eos.dfloor) &&
          ((u.mx != 0.0) || (u.my != 0.0) || (u.mz != 0.0)));
}

KOKKOS_INLINE_FUNCTION
void ResetIdealHydroAtmosphereState(const EOS_Data &eos, HydCons1D &u, HydPrim1D &w,
                                    bool &dfloor_used) {
  const Real atmosphere_eint = IdealHydroAtmosphereInternalEnergy(eos);
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

//----------------------------------------------------------------------------------------
//! \fn void SingleC2P_IdealHyd()
//! \brief Converts single state of conserved variables into primitive variables for
//! non-relativistic hydrodynamics with an ideal gas EOS.
//! Conserved = (d,M1,M2,M3,E), Primitive = (d,vx,vy,vz,e)
//! where E=total energy density and e=internal energy density

KOKKOS_INLINE_FUNCTION
void SingleC2P_IdealHyd(HydCons1D &u, const EOS_Data &eos,
                        HydPrim1D &w,
                        bool &dfloor_used, bool &efloor_used, bool &tfloor_used,
                        bool &vceil_used) {
  Real tfloor = eos.tfloor;
  Real gm1 = eos.gamma - 1.0;

  if (NeedsIdealHydroAtmosphereReset(eos, u)) {
    ResetIdealHydroAtmosphereState(eos, u, w, dfloor_used);
    return;
  }
  w.d = u.d;

  // compute velocities
  Real di = 1.0/u.d;
  w.vx = di*u.mx;
  w.vy = di*u.my;
  w.vz = di*u.mz;

  // set internal energy, apply floor, correct total energy (if needed)
  Real e_k = 0.5*di*(SQR(u.mx) + SQR(u.my) + SQR(u.mz));
  w.e = (u.e - e_k);
  Real efloor = IdealHydroEintFloor(eos, w.d);
  if (w.e < efloor) {
    w.e = efloor;
    u.e = efloor + e_k;
    efloor_used = true;
  }
  // apply temperature floor
  if (gm1*w.e*di < tfloor) {
    w.e = w.d*tfloor/gm1;
    u.e = w.e + e_k;
    tfloor_used = true;
  }
  const Real eceil = eos.HydroInternalEnergyDensityCeiling(w.d);
  if (w.e > eceil) {
    w.e = eceil;
    u.e = w.e + e_k;
    efloor_used = true;
  }

  // Apply optional velocity ceiling on |v|.
  Real v2 = SQR(w.vx) + SQR(w.vy) + SQR(w.vz);
  Real vmag = sqrt(v2);
  if ((eos.vceil > 0.0) && (vmag > eos.vceil)) {
    Real fac = eos.vceil/vmag;
    w.vx *= fac;
    w.vy *= fac;
    w.vz *= fac;
    vmag = eos.vceil;
    v2 = vmag*vmag;
    vceil_used = true;
  }

  // Keep conserved variables consistent with any floor-limited primitive state.
  u.mx = w.d*w.vx;
  u.my = w.d*w.vy;
  u.mz = w.d*w.vz;
  u.e = w.e + 0.5*w.d*v2;
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void SingleP2C_IdealHyd()
//! \brief Converts single state of primitive variables into conserved variables for
//! non-relativistic hydrodynamics with an ideal gas EOS.
//! Conserved = (d,M1,M2,M3,E), Primitive = (d,vx,vy,vz,e)
//! where E=total energy density and e=internal energy density

KOKKOS_INLINE_FUNCTION
void SingleP2C_IdealHyd(const HydPrim1D &w, HydCons1D &u) {
  u.d  = w.d;
  u.mx = w.d*w.vx;
  u.my = w.d*w.vy;
  u.mz = w.d*w.vz;
  u.e = w.e + 0.5*w.d*(SQR(w.vx) + SQR(w.vy) + SQR(w.vz));
  return;
}

//----------------------------------------------------------------------------------------
//! \fn Real IdealHydroEntropyFloorDensityThreshold()
//! \brief Returns the density above which the entropy floor exceeds the pressure floor,
//! i.e. the density where sfloor*d^gamma == pfloor.  Computed once per cell so that the
//! root-find inner loop can branch on it instead of evaluating a double precision pow()
//! on every iteration.  The threshold is nudged down by a relative 1.0e-12 so that
//! round-off in this expression can only take the fmax() branch early (where fmax()
//! returns the same value the original expression did), never late.

KOKKOS_INLINE_FUNCTION
Real IdealHydroEntropyFloorDensityThreshold(const EOS_Data &eos) {
  if (!(eos.sfloor > 0.0)) {
    return static_cast<Real>(1.0e300);
  }
  return pow(eos.pfloor/eos.sfloor, 1.0/eos.gamma)*(1.0 - 1.0e-12);
}

//----------------------------------------------------------------------------------------
//! \fn Real EquationC22()
//! \brief Inline function to compute function f(z) defined in eq. C22 of Galeazzi et al.
//! used to convert conserved to primitive variables for relativistic hydrodynamics
//! The ConsToPrim algorithm finds the root of this function f(z)=0
//! `sfloor_dthresh` is the entropy-floor density threshold returned by
//! IdealHydroEntropyFloorDensityThreshold(), computed once per cell by the caller.

KOKKOS_INLINE_FUNCTION
Real EquationC22(const Real z, const Real u_d, const Real q, const Real r,
                 const EOS_Data &eos, const Real sfloor_dthresh) {
  Real const gm1 = eos.gamma - 1.0;
  Real const w = sqrt(1.0 + z*z);         // (C15)
  Real const wd = u_d/w;                  // (C15)
  Real eps = w*q - z*r + (z*z)/(1.0 + w); // (C16)
  // The entropy floor can only exceed the pressure floor above wd=(pfloor/sfloor)^(1/g);
  // below that threshold fmax() returns the pressure term anyway, so the returned value
  // is unchanged while a double precision pow() leaves every root-find iteration.
  Real epsmin = fmax(eos.pfloor/(wd*gm1), eos.tfloor/gm1);
  if (wd > sfloor_dthresh) {
    epsmin = fmax(epsmin, eos.sfloor*pow(wd, gm1)/gm1);
  }
  eps = fmax(eps, epsmin);                // (C18)
  Real const h = 1.0 + eos.gamma*eps;     // (C1) & (C21)
  return (z - r/h); // (C22)
}

//! Rounded operations on the longest chain from the conserved state to eps in
//! SingleC2P_IdealSRHyd, counted as for kEpsRoundingOps of the MHD inversion: lor is 3
//! (z z, 1 + , sqrt), lor q 4, r 2 (sqrt, the quotient) and z r 3, the difference 5,
//! z z/(1 + lor) 5 (the sum 1 + lor is 4) and the final sum 6.
constexpr int kHydEpsRoundingOps = 6;

//----------------------------------------------------------------------------------------
//! \fn void SingleC2P_IdealSRHyd()
//! \brief Converts single state of conserved variables into primitive variables for
//! special relativistic hydrodynamics with an ideal gas EOS.  floor_at_equality = false
//! floors eps only strictly below epsmin, so that with every floor at zero eps = 0 is a
//! root, as in KORAL (IdealSRHydHotInversionFails).
//!
//! `eps_unresolved`, when given, is set where the energy equation holds no significant
//! digit of eps, as in SingleC2P_IdealSRMHD with b = 0: eps = lor q - z r + z^2/(1+lor)
//! is the difference of terms of magnitude lor |q|, z r and z^2/(1+lor) that carry the
//! Lorentz factor, so rounding alone moves it by kHydEpsRoundingOps eps_mach times their
//! sum, and a root whose eps is not above that is noise.  It changes nothing here, it is
//! a flag for the caller's entropy inversion, and is judged on eps before the floors act.

template <bool floor_at_equality = true>
KOKKOS_INLINE_FUNCTION
void SingleC2P_IdealSRHyd(HydCons1D &u, const EOS_Data &eos, const Real s2, HydPrim1D &w,
                          bool &dfloor_used, bool &efloor_used, bool &c2p_failure,
                          int &iter_used, bool *eps_unresolved = nullptr) {
  // Parameters
  const int max_iterations = 25;
  const Real tol = 1.0e-12;
  const Real v_max = 0.9999999999995;  // NOTE(@pdmullen): SQR(v_max) = 1.0 - tol;
  const Real kmax = 2.0*v_max/(1.0 + v_max*v_max);
  const Real gm1 = eos.gamma - 1.0;
  // Density above which the entropy floor can exceed the pressure floor.  Evaluated once
  // here so that EquationC22() does not evaluate a pow() on every iteration.
  const Real sfloor_dthresh = IdealHydroEntropyFloorDensityThreshold(eos);

  // apply density floor, without changing momentum or energy
  if (u.d < eos.dfloor) {
    u.d = eos.dfloor;
    dfloor_used = true;
  }

  // apply energy floor
  if (u.e < eos.pfloor/gm1) {
    u.e = eos.pfloor/gm1;
    efloor_used = true;
  }

  // Recast all variables (eq C2)
  Real q = u.e/u.d;
  Real r = sqrt(s2)/u.d;
  Real kk = r/(1.+q);

  // Enforce lower velocity bound (eq. C13). This bound combined with a floor on
  // the value of p will guarantee "some" result of the inversion
  kk = fmin(kmax, kk);

  // Compute bracket (C23)
  Real zm = 0.5*kk/sqrt(1.0 - 0.25*kk*kk);
  Real zp = kk/sqrt(1.0 - kk*kk);

  // Evaluate master function (eq C22) at bracket values
  Real fm = EquationC22(zm, u.d, q, r, eos, sfloor_dthresh);
  Real fp = EquationC22(zp, u.d, q, r, eos, sfloor_dthresh);

  // For simplicity on the GPU, find roots using the false position method
  int iterations = max_iterations;
  // If bracket within tolerances, don't bother doing any iterations
  if ((fabs(zm-zp) < tol) || ((fabs(fm) + fabs(fp)) < 2.0*tol)) {
    iterations = -1;
  }
  Real z = 0.5*(zm + zp);

  for (iter_used=0; iter_used < iterations; ++iter_used) {
    z =  (zm*fp - zp*fm)/(fp-fm);  // linear interpolation to point f(z)=0
    Real f = EquationC22(z, u.d, q, r, eos, sfloor_dthresh);

    // Quit if convergence reached
    // NOTE: both z and f are of order unity
    if ((fabs(zm-zp) < tol) || (fabs(f) < tol)) {
      break;
    }

    // assign zm-->zp if root bracketed by [z,zp]
    if (f*fp < 0.0) {
      zm = zp;
      fm = fp;
      zp = z;
      fp = f;
    } else {  // assign zp-->z if root bracketed by [zm,z]
      fm = 0.5*fm; // 1/2 comes from "Illinois algorithm" to accelerate convergence
      zp = z;
      fp = f;
    }
  }

  // check if convergence is established within max_iterations.  If not, trigger a C2P
  // failure and return floored density, pressure, and primitive velocities. Future
  // development may trigger averaging of (successfully inverted) neighbors in the event
  // of a C2P failure.
  if (iter_used==max_iterations) {
    w.d = eos.dfloor;
    w.e = IdealHydroAtmosphereInternalEnergy(eos);
    w.vx = 0.0;
    w.vy = 0.0;
    w.vz = 0.0;
    c2p_failure = true;
    return;
  }

  // iterations ended, compute primitives from resulting value of z
  Real const lor = sqrt(1.0 + z*z);  // (C15)

  // compute density then apply floor
  Real dens = u.d/lor;
  if (dens < eos.dfloor) {
    dens = eos.dfloor;
    dfloor_used = true;
  }

  // compute specific internal energy density then apply floor
  Real eps = lor*q - z*r + (z*z)/(1.0 + lor);   // (C16)
  // Same guarded entropy floor as in EquationC22(): below the threshold density the
  // pressure term is the larger of the two, so the value is unchanged.
  Real epsmin = fmax(eos.pfloor/(dens*gm1), eos.tfloor/gm1);
  if (dens > sfloor_dthresh) {
    epsmin = fmax(epsmin, eos.sfloor*pow(dens, gm1)/gm1);
  }
  if (eps_unresolved != nullptr) {
    const Real rounding = static_cast<Real>(kHydEpsRoundingOps)*
        std::numeric_limits<Real>::epsilon()*
        (lor*fabs(q) + fabs(z*r) + (z*z)/(1.0 + lor));
    *eps_unresolved = !(eps > rounding);
  }
  if (floor_at_equality ? (eps <= epsmin) : (eps < epsmin)) {
    eps = epsmin;
    efloor_used = true;
  }

  // set parameters required for velocity inversion
  Real const h = 1.0 + eos.gamma*eps;  // (C21)
  Real const conv = 1.0/h;             // (C26)

  // set primitive variables
  w.d  = dens;
  w.vx = conv*(u.mx/u.d);  // (C26)
  w.vy = conv*(u.my/u.d);  // (C26)
  w.vz = conv*(u.mz/u.d);  // (C26)
  w.e  = dens*eps;

  return;
}

//----------------------------------------------------------------------------------------
//! \fn bool IdealSRHydHotInversionFails()
//! \brief KORAL's U2P_HOT verdict (u2p.c:1024-1348: no floor inside the root, failure
//! -104 where uint < 0, uint = 0 accepted) on a GR hydro cell whose ordinary inversion
//! floored D or tau at entry, or eps after the root: the cell's own SR state is inverted
//! again with every floor at zero, and the verdict is that solve's failure, any floor it
//! still needed (D < 0, tau < 0, eps < 0), an eps with no significant digit
//! (eps_unresolved of SingleC2P_IdealSRHyd), or a non-finite result.  The result is only
//! a verdict and is discarded.  Out of line in device code.

KOKKOS_INLINE_FUNCTION GR_ENTROPY_DEVICE_NOINLINE
bool IdealSRHydHotInversionFails(const HydCons1D u_sr, const Real gamma, const Real s2) {
  EOS_Data raw_eos;
  raw_eos.gamma = gamma;
  HydCons1D raw_u = u_sr;
  HydPrim1D raw_w{};
  bool raw_dfloor = false, raw_efloor = false, raw_failure = false;
  int raw_iter = 0;
  bool raw_unresolved = false;
  SingleC2P_IdealSRHyd<false>(raw_u, raw_eos, s2, raw_w, raw_dfloor, raw_efloor,
                              raw_failure, raw_iter, &raw_unresolved);
  return raw_dfloor || raw_efloor || raw_failure || raw_unresolved ||
      !(isfinite(raw_w.d) && isfinite(raw_w.e) && isfinite(raw_w.vx) &&
        isfinite(raw_w.vy) && isfinite(raw_w.vz));
}

//----------------------------------------------------------------------------------------
//! \fn void FailIdealSRHyd()
//! \brief The state a failed inversion leaves: the atmosphere at rest and c2p_failure
//! (a failure the ordinary solve already flagged keeps its own state).

KOKKOS_INLINE_FUNCTION
void FailIdealSRHyd(const EOS_Data &eos, HydPrim1D &w, bool &c2p_failure) {
  if (!c2p_failure) {
    w.d = eos.dfloor;
    w.e = IdealHydroAtmosphereInternalEnergy(eos);
    w.vx = 0.0;
    w.vy = 0.0;
    w.vz = 0.0;
    c2p_failure = true;
  }
}

//----------------------------------------------------------------------------------------
//! \fn bool ApplyEntropyFallback_IdealSRHyd()
//! \brief KORAL's sequence after U2P_HOT (u2p.c:229-263) on one fixed-metric GR hydro
//! cell, applied to the result of SingleC2P_IdealSRHyd.  Hot failure is c2p_failure or a
//! non-finite result, an energy root whose eps holds no significant digit (eps_unresolved
//! of SingleC2P_IdealSRHyd, as in ApplyEntropyFallback_IdealSRMHD), or, where a floor
//! acted, the zero-floor verdict above.  A cell whose hot inversion did not fail is left
//! exactly as it is.  Where it failed and D > 0,
//! U2P_ENTROPY with B = 0 solves the state from the transported entropy entropy_sr = D s
//! (SR-frame D, gr_entropy::RecoverSRStateFromEntropy); its root replaces the energy
//! state, drops the discarded solve's flags and takes the ordinary density and energy
//! floors.
//! Where it fails too, or D <= 0 leaves neither inversion a state to solve, the cell is a
//! failure (u2p.c:342-357) as in GRMHD (ApplyEntropyFallback_IdealSRMHD): the atmosphere at
//! rest and c2p_failure, which the caller answers with the kept previous primitives
//! (KeepPreviousPrimitives_IdealSRHyd).  A root that only lacks digits of eps stays the
//! cell's when the entropy inversion fails (its floors apply), as in GRMHD.  Returns true
//! if the entropy root was taken.

KOKKOS_INLINE_FUNCTION
bool ApplyEntropyFallback_IdealSRHyd(
    const HydCons1D &u_sr, const EOS_Data &eos, const Real s2, const Real entropy_sr,
    HydPrim1D &w, bool &dfloor_used, bool &efloor_used, bool &c2p_failure,
    int &iter_used, const bool eps_unresolved = false) {
  bool hot_failed = c2p_failure ||
      !(isfinite(w.d) && isfinite(w.e) && isfinite(w.vx) && isfinite(w.vy) &&
        isfinite(w.vz));
  if (!hot_failed && (dfloor_used || efloor_used)) {
    hot_failed = IdealSRHydHotInversionFails(u_sr, eos.gamma, s2);
  }
  if (!(hot_failed || eps_unresolved)) return false;
  if (!(u_sr.d > 0.0)) {
    if (hot_failed) FailIdealSRHyd(eos, w, c2p_failure);
    return false;
  }
  MHDCons1D u_sr_mhd;
  u_sr_mhd.d = u_sr.d;
  u_sr_mhd.mx = u_sr.mx;
  u_sr_mhd.my = u_sr.my;
  u_sr_mhd.mz = u_sr.mz;
  u_sr_mhd.e = u_sr.e;
  u_sr_mhd.bx = 0.0;
  u_sr_mhd.by = 0.0;
  u_sr_mhd.bz = 0.0;
  const gr_entropy::SREntropyRecovery entropy = gr_entropy::RecoverSRStateFromEntropy(
      u_sr_mhd, s2, 0.0, 0.0, entropy_sr, eos.gamma);
  iter_used = (entropy.iterations > iter_used) ? entropy.iterations : iter_used;
  if (!entropy.ok) {
    if (hot_failed) FailIdealSRHyd(eos, w, c2p_failure);
    return false;
  }
  w = entropy.w;
  c2p_failure = false;
  dfloor_used = false;
  efloor_used = false;
  if (w.d < eos.dfloor) {
    w.d = eos.dfloor;
    dfloor_used = true;
  }
  const Real eint_floor = IdealGRHydroEintFloor(eos, w.d);
  if (w.e < eint_floor) {
    w.e = eint_floor;
    efloor_used = true;
  }
  return true;
}

//----------------------------------------------------------------------------------------
//! \fn bool KeepPreviousPrimitives_IdealSRHyd()
//! \brief KORAL's no-fixup branch for a cell neither inversion solves (koral_lite
//! u2p.c:342-357, pp = ppbak): w becomes the primitives the cell last published, under
//! the density and energy floors, and the floor flags of the discarded solve are replaced
//! by those of the floors applied to the kept state.  Left alone (returns false) where the
//! kept primitives are not a gas state: rho or eint not positive, anything not finite.
//! The failure flag is the caller's and stays set.

KOKKOS_INLINE_FUNCTION
bool KeepPreviousPrimitives_IdealSRHyd(
    const EOS_Data &eos, const Real rho_prev, const Real eint_prev, const Real vx_prev,
    const Real vy_prev, const Real vz_prev, HydPrim1D &w, bool &dfloor_used,
    bool &efloor_used) {
  if (!(isfinite(rho_prev) && rho_prev > 0.0 && isfinite(eint_prev) && eint_prev > 0.0 &&
        isfinite(vx_prev) && isfinite(vy_prev) && isfinite(vz_prev))) {
    return false;
  }
  w.d = rho_prev;
  w.e = eint_prev;
  w.vx = vx_prev;
  w.vy = vy_prev;
  w.vz = vz_prev;
  dfloor_used = false;
  efloor_used = false;
  if (w.d < eos.dfloor) {
    w.d = eos.dfloor;
    dfloor_used = true;
  }
  const Real eint_floor = IdealGRHydroEintFloor(eos, w.d);
  if (w.e < eint_floor) {
    w.e = eint_floor;
    efloor_used = true;
  }
  return true;
}

//----------------------------------------------------------------------------------------
//! \fn void SingleP2C_IdealSRHyd()
//! \brief Converts single state of primitive variables into conserved variables for
//! special relativistic hydrodynamics with an ideal gas EOS.

KOKKOS_INLINE_FUNCTION
void SingleP2C_IdealSRHyd(const HydPrim1D &w, const Real gam, HydCons1D &u) {
  // Calculate Lorentz factor
  Real u0 = sqrt(1.0 + SQR(w.vx) + SQR(w.vy) + SQR(w.vz));
  Real wgas_u0 = (w.d + gam*w.e)*u0;

  // Set conserved quantities
  u.d  = w.d * u0;
  u.e  = wgas_u0 * u0 - (gam-1.0)*w.e - u.d;  // In SR, evolve E - D
  u.mx = wgas_u0 * w.vx;            // In SR, vx/y/z are 4-velocity
  u.my = wgas_u0 * w.vy;
  u.mz = wgas_u0 * w.vz;
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void TransformToSRHyd()
//! \brief Converts single state of conserved variables in GR hydro into conserved
//! variables for special relativistic hydrodynamics with an ideal gas EOS. This allows
//! the ConsToPrim() function in GR Hydro to use SinceP2C_IdealSrHyd() function.

KOKKOS_INLINE_FUNCTION
void TransformToSRHyd(const HydCons1D &u, Real glower[][4], Real gupper[][4],
                      Real &s2, HydCons1D &u_sr) {
  // Need to multiply the conserved density by alpha, so that it
  // contains a lorentz factor
  Real alpha = sqrt(-1.0/gupper[0][0]);
  u_sr.d = u.d*alpha;

  // We are evolving T^t_t, but the SR C2P algorithm is only consistent with
  // alpha^2 T^{tt}.  Therefore compute T^{tt} = g^0\mu T^t_\mu
  // We are also evolving T^t_t + D as conserved variable, so must convert to E
  u_sr.e = gupper[0][0]*(u.e - u.d) +
           gupper[0][1]*u.mx + gupper[0][2]*u.my + gupper[0][3]*u.mz;

  // This is only true if sqrt{-g}=1!
  u_sr.e *= (-1./gupper[0][0]);  // Multiply by alpha^2

  // Subtract density for consistency with the rest of the algorithm
  u_sr.e -= u_sr.d;

  // Need to treat the conserved momenta. Also they lack an alpha
  // This is only true if sqrt{-g}=1!
  Real m1l = u.mx*alpha;
  Real m2l = u.my*alpha;
  Real m3l = u.mz*alpha;

  // Need to raise indices on u_sr, which transforms using the spatial 3-metric.
  // This is slightly more involved
  //
  // Gourghoulon says: g^ij = gamma^ij - beta^i beta^j/alpha^2
  //       g^0i = beta^i/alpha^2
  //       g^00 = -1/ alpha^2
  // Hence gamma^ij =  g^ij - g^0i g^0j/g^00
  u_sr.mx = ((gupper[1][1] - gupper[0][1]*gupper[0][1]/gupper[0][0])*m1l +
             (gupper[1][2] - gupper[0][1]*gupper[0][2]/gupper[0][0])*m2l +
             (gupper[1][3] - gupper[0][1]*gupper[0][3]/gupper[0][0])*m3l);  // (C26)

  u_sr.my = ((gupper[2][1] - gupper[0][2]*gupper[0][1]/gupper[0][0])*m1l +
             (gupper[2][2] - gupper[0][2]*gupper[0][2]/gupper[0][0])*m2l +
             (gupper[2][3] - gupper[0][2]*gupper[0][3]/gupper[0][0])*m3l);  // (C26)

  u_sr.mz = ((gupper[3][1] - gupper[0][3]*gupper[0][1]/gupper[0][0])*m1l +
             (gupper[3][2] - gupper[0][3]*gupper[0][2]/gupper[0][0])*m2l +
             (gupper[3][3] - gupper[0][3]*gupper[0][3]/gupper[0][0])*m3l);  // (C26)

  // Return s2 = (S^i S_i) (eqn C2)
  s2 = ((m1l*u_sr.mx) + (m2l*u_sr.my) + (m3l*u_sr.mz));
  return;
}

//--------------------------------------------------------------------------------------
//! \fn void SingleP2C_IdealGRHyd()
//! \brief Converts single set of primitive into conserved variables.

KOKKOS_INLINE_FUNCTION
void SingleP2C_IdealGRHyd(const Real glower[][4], const Real gupper[][4],
                          const HydPrim1D &w, const Real &gam, HydCons1D &u) {
  // Calculate 4-velocity (exploiting symmetry of metric)
  Real q = glower[1][1]*w.vx*w.vx +2.0*glower[1][2]*w.vx*w.vy +2.0*glower[1][3]*w.vx*w.vz
         + glower[2][2]*w.vy*w.vy +2.0*glower[2][3]*w.vy*w.vz
         + glower[3][3]*w.vz*w.vz;
  Real alpha = sqrt(-1.0/gupper[0][0]);
  Real gamma = sqrt(1.0 + q);
  Real u0 = gamma / alpha;
  Real u1 = w.vx - alpha * gamma * gupper[0][1];
  Real u2 = w.vy - alpha * gamma * gupper[0][2];
  Real u3 = w.vz - alpha * gamma * gupper[0][3];

  // lower vector indices
  Real u_0 = glower[0][0]*u0 + glower[0][1]*u1 + glower[0][2]*u2 + glower[0][3]*u3;
  Real u_1 = glower[1][0]*u0 + glower[1][1]*u1 + glower[1][2]*u2 + glower[1][3]*u3;
  Real u_2 = glower[2][0]*u0 + glower[2][1]*u1 + glower[2][2]*u2 + glower[2][3]*u3;
  Real u_3 = glower[3][0]*u0 + glower[3][1]*u1 + glower[3][2]*u2 + glower[3][3]*u3;
  Real wgas_u0 = (w.d + gam * w.e) * u0;

  // set conserved quantities
  u.d  = w.d * u0;
  u.e  = wgas_u0 * u_0 + (gam-1.0)*w.e + u.d;  // evolve T^t_t + D
  u.mx = wgas_u0 * u_1;
  u.my = wgas_u0 * u_2;
  u.mz = wgas_u0 * u_3;
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void TransformToSRHyd()  [Kerr-Schild null-form overload]
//! \brief GR->SR conserved transform from the four-number CKS null form.  Raising the
//! momentum index uses gamma^{ij} = delta^{ij} - [f/(1+f)] l_i l_j, algebraically
//! identical to g^{ij} - g^{0i}g^{0j}/g^{00} but without the two O(f) terms that route
//! forms and then cancels near the horizon.

KOKKOS_INLINE_FUNCTION
void TransformToSRHyd(const HydCons1D &u, const KSNullForm &nf,
                      Real &s2, HydCons1D &u_sr) {
  const Real inv_alpha_sq = KSInvLapseSq(nf);   // = -g^{00} = 1 + f
  const Real alpha = KSAlpha(nf);
  u_sr.d = u.d*alpha;

  const Real ldm = nf.l1*u.mx + nf.l2*u.my + nf.l3*u.mz;
  u_sr.e = -inv_alpha_sq*(u.e - u.d) + nf.f*ldm;
  u_sr.e *= (1.0/inv_alpha_sq);   // multiply by alpha^2 (only true if sqrt(-g)=1)
  u_sr.e -= u_sr.d;

  const Real m1l = u.mx*alpha;
  const Real m2l = u.my*alpha;
  const Real m3l = u.mz*alpha;

  const Real c = KSGammaUpperFactor(nf);
  const Real cldm = c*(nf.l1*m1l + nf.l2*m2l + nf.l3*m3l);
  u_sr.mx = m1l - nf.l1*cldm;    // (C26)
  u_sr.my = m2l - nf.l2*cldm;    // (C26)
  u_sr.mz = m3l - nf.l3*cldm;    // (C26)

  s2 = ((m1l*u_sr.mx) + (m2l*u_sr.my) + (m3l*u_sr.mz));   // (C2)
  return;
}

//--------------------------------------------------------------------------------------
//! \fn void SingleP2C_IdealGRHyd()  [Kerr-Schild null-form overload]

KOKKOS_INLINE_FUNCTION
void SingleP2C_IdealGRHyd(const KSNullForm &nf, const HydPrim1D &w, const Real &gam,
                          HydCons1D &u) {
  Real q = KSSpatialNormSq(nf, 1, 2, 3, w.vx, w.vy, w.vz);
  Real alpha = KSAlpha(nf);
  Real gamma = sqrt(1.0 + q);
  Real u0 = gamma / alpha;
  Real ag_f = (alpha*gamma)*nf.f;    // alpha*gamma*g^{0i} = ag_f * l_i
  Real u1 = w.vx - ag_f*nf.l1;
  Real u2 = w.vy - ag_f*nf.l2;
  Real u3 = w.vz - ag_f*nf.l3;

  // lower vector indices: A_mu = eta_{mu nu} A^nu + f l_mu (l.A)
  Real u_0, u_1, u_2, u_3;
  KSLowerVec(nf, 1, 2, 3, u0, u1, u2, u3, u_0, u_1, u_2, u_3);
  Real wgas_u0 = (w.d + gam * w.e) * u0;

  u.d  = w.d * u0;
  u.e  = wgas_u0 * u_0 + (gam-1.0)*w.e + u.d;  // evolve T^t_t + D
  u.mx = wgas_u0 * u_1;
  u.my = wgas_u0 * u_2;
  u.mz = wgas_u0 * u_3;
  return;
}

#endif // EOS_IDEAL_C2P_HYD_HPP_
