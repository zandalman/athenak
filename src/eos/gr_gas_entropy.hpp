#ifndef EOS_GR_GAS_ENTROPY_HPP_
#define EOS_GR_GAS_ENTROPY_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file gr_gas_entropy.hpp
//! \brief KORAL's total gas entropy for an ideal gas, and its U2P_ENTROPY inversion.
//!
//! The entropy per unit rest mass is KORAL's (physics.c:1457-1475, calc_Sfromu):
//!   s = ln(p/rho^Gamma)/(Gamma-1),  S = rho s,
//! with Gamma the cell's adiabatic index.  Every GR fluid carries D s as one conserved
//! column (the gas-entropy slot) and s as its primitive.  The inversion below is
//! U2P_ENTROPY: it solves the momentum, magnetic and density equations with the
//! transported entropy in place of the energy equation (u2p.c:1024-1383, 2144-2200),
//! and is used only where U2P_HOT fails (u2p.c:229-263).  It is written for any
//! transported adiabat p(rho) (RecoverSRStateOnAdiabat); the species closure of the
//! two-temperature module is the other one (eos/ideal_c2p_variable_gamma.hpp).
//!
//! Everything here is device-safe and depends on athena.hpp only, so every inversion
//! (fixed metric, dynamical metric, hybrid sidecar) shares one copy.

#include <math.h>

#include <limits>

#include "athena.hpp"
#include "eos/kastaun_master.hpp"

//! \def GR_ENTROPY_DEVICE_NOINLINE
//! \brief Keeps a rarely taken inversion out of the calling C2P kernel in DEVICE code:
//! an inlined root find would enlarge the register allocation of every cell.  Arguments
//! and result go by value, so no array of the calling kernel has its address taken.
//! Empty for the host compiler.
#if defined(__CUDACC__)
#define GR_ENTROPY_DEVICE_NOINLINE __noinline__
#else
#define GR_ENTROPY_DEVICE_NOINLINE
#endif

namespace gr_entropy {

//----------------------------------------------------------------------------------------
//! \fn Real SpecificEntropyIdeal()
//! \brief s = ln(p/rho^Gamma)/(Gamma-1) of an ideal gas with internal energy density
//! eint (p = (Gamma-1) eint), KORAL calc_Sfromu (physics.c:1457-1475) divided by rho,
//! in log form.  Not finite where the state has no entropy (rho <= 0, eint <= 0, or
//! Gamma <= 1); the callers decide what that means.

KOKKOS_INLINE_FUNCTION
Real SpecificEntropyIdeal(const Real rho, const Real eint, const Real gamma) {
  const Real gm1 = gamma - 1.0;
  return (log(gm1*eint) - gamma*log(rho))/gm1;
}

//----------------------------------------------------------------------------------------
//! \fn Real ColumnWithHeat()
//! \brief The gas-entropy column D s of a gas whose transported column D s_b is joined by
//! the radiation heat register H = D h (MHD::rad_heat; h the comoving heat per unit rest
//! mass the M1 commits of the step handed the gas, combined by the Runge-Kutta weights of
//! the energy): the entropy of e(rho, s_b) + rho h at rho, taken along the isochore of
//! the gamma law, s = s_b + log1p(rho h/e_b)/(Gamma - 1), e_b = e(rho, s_b).  The entropy
//! root at rho then returns e_b + rho h whatever the digits of the energy root, and the
//! heat enters the entropy as it enters the energy: linearly in the stage weights.  rho is
//! the reader's density (the commit's where the gas has not moved).  The column itself
//! where there is no heat (bitwise) and where the heated state has no entropy
//! (e_b + rho h <= 0, a debit beyond the gas the replay guard and the slaved reservoir
//! bound; the entropy root then returns the gas the column describes).

KOKKOS_INLINE_FUNCTION
Real ColumnWithHeat(const Real column, const Real heat, const Real dens, const Real rho,
                    const Real gamma) {
  if (heat == 0.0) return column;
  const Real gm1 = gamma - 1.0;
  if (!(dens > 0.0) || !(rho > 0.0) || !(gm1 > 0.0)) return column;
  const Real e_b = exp(gamma*log(rho) + gm1*column/dens)/gm1;
  const Real x = rho*heat/(dens*e_b);
  if (!(x > -1.0) || !isfinite(x)) return column;
  return column + dens*log1p(x)/gm1;
}

//----------------------------------------------------------------------------------------
//! \struct GasEntropyAdiabat
//! \brief The adiabat of an ideal gas of index Gamma with the transported specific
//! entropy s = ln(p/rho^Gamma)/(Gamma - 1): p = K rho^Gamma, ln K = (Gamma - 1) s, and
//! eint = p/(Gamma - 1).  The adiabat interface of RecoverSRStateOnAdiabat: At(rho)
//! gives the internal energy density and the pressure of the gas at density rho.

struct GasEntropyAdiabat {
  Real log_k = 0.0;
  Real gamma = 0.0;
  KOKKOS_INLINE_FUNCTION
  void At(const Real rho, Real &eint, Real &pressure) const {
    pressure = exp(log_k + gamma*log(rho));
    eint = pressure/(gamma - 1.0);
  }
};

//! \brief f(mu) = mu - 1/(h/W + rbar mu) of the Kastaun master function (eq. 44) with the
//! specific enthalpy h = 1 + (eint + p)/rho of the adiabat at rho = D/W (normalized
//! b2, rpar, r as in kastaun_master.hpp; d the unnormalized D).
template <typename Adiabat>
KOKKOS_INLINE_FUNCTION
Real AdiabatMasterFunction(const Real mu, const Real b2, const Real rpar, const Real r,
                           const Real d, const Adiabat &adiabat) {
  const Real rbar = KastaunRbar(mu, b2, rpar, r);
  const Real z2 = mu*mu*rbar/fabs(1.0 - mu*mu*rbar);
  const Real lor = sqrt(1.0 + z2);
  const Real rho = d/lor;
  Real eint = 0.0, pressure = 0.0;
  adiabat.At(rho, eint, pressure);
  const Real h = 1.0 + (eint + pressure)/rho;
  return mu - 1.0/(h/lor + rbar*mu);
}

//----------------------------------------------------------------------------------------
//! \fn bool RecoverSRStateOnAdiabat()
//! \brief U2P_ENTROPY (KORAL u2p.c:1024-1383, 2144-2200, called where U2P_HOT failed,
//! u2p.c:229-263): the normal-observer state u_sr = (D, S_i, tau, B^i), S^2 = s2,
//! B^2 = b2, rpar = (B.S)/D, with the transported adiabat p(rho) of the gas in place of
//! the energy equation.  Kastaun, Kalinani & Ciolfi (2021) derive their master function
//! eq. 44 from the momentum, field and density equations; the energy equation enters
//! only through the enthalpy h of the trial state, so with h = h(rho) of the adiabat at
//! rho = D/W(mu) the same f(mu) = mu - 1/(h/W + rbar mu) is U2P_ENTROPY in one unknown.
//! Its root is bracketed by [0, mu+] (eq. 49) whatever the adiabat: f(0) = -1/h < 0, and
//! at mu+ (mu+^2 (1 + rbar) = 1, so W = 1/mu+) f = mu+^2 (h - 1)/(mu+ (h + rbar)) >= 0
//! for any gas with h >= 1, so the inversion has a root for every admissible
//! (D > 0, finite) state and needs no seed; it is false position (Illinois), converged on
//! the density (below), and the pressure is that of the adiabat at it.  KORAL instead
//! runs a damped Newton iteration on W = rho h Gamma^2 from the cell's preceding
//! primitives, which on a host scan of sigma <= 1e6, W <= 50 states found no root for
//! 3-6 % of them; this solve finds all, within the round-off of the state, in 6-20
//! evaluations.  Returns false only for a state without a finite admissible inversion.

template <typename Adiabat>
KOKKOS_INLINE_FUNCTION
bool RecoverSRStateOnAdiabat(const MHDCons1D &u_sr, const Real s2, const Real b2_in,
                             const Real rpar_in, const Adiabat &adiabat,
                             HydPrim1D &primitive, int &iterations_used) {
  primitive = HydPrim1D{};
  const Real d = u_sr.d;
  if (!(isfinite(d) && d > 0.0 && isfinite(s2) && isfinite(b2_in) &&
        isfinite(rpar_in))) {
    return false;
  }
  const int max_iterations = 100;
  const Real tol = 1.0e-12;
  const Real isqrtd = 1.0/sqrt(d);
  const Real r = sqrt(s2)/d;
  const Real b2 = b2_in/d;
  const Real rpar = rpar_in*isqrtd;

  // mu+, the root of eq. 49 (as in SingleC2P_IdealSRMHD)
  Real zm = 0.0, zp = 1.0;
  Real fm = Equation49(zm, b2, rpar, r, 0.0);
  Real fp = Equation49(zp, b2, rpar, r, 0.0);
  Real z = 0.5*(zm + zp);
  int iter = 0;
  const int iterations49 =
      ((fabs(zm - zp) < tol) || ((fabs(fm) + fabs(fp)) < 2.0*tol)) ? -1 : max_iterations;
  #pragma unroll 1
  for (iter = 0; iter < iterations49; ++iter) {
    z = (zm*fp - zp*fm)/(fp - fm);
    const Real f = Equation49(z, b2, rpar, r, 0.0);
    if ((fabs(zm - zp) < KastaunMuTolerance(z, tol)) || (fabs(f) < tol)) break;
    if (f*fp < 0.0) {
      zm = zp; fm = fp; zp = z; fp = f;
    } else {
      fm = 0.5*fm; zp = z; fp = f;
    }
  }
  int iterations = iter;

  // The root of the master function on [0, mu+].  mu+ is known to the tolerance of its
  // own solve only, so an upper end that does not yet change sign is moved up past it
  // in doubling steps, each still on the Kastaun branch.
  zm = 0.0;
  zp = z;
  fm = AdiabatMasterFunction(zm, b2, rpar, r, d, adiabat);
  fp = AdiabatMasterFunction(zp, b2, rpar, r, d, adiabat);
  if (!(fm*fp < 0.0) && fp != 0.0) {
    const Real z49 = zp;
    #pragma unroll 1
    for (Real step = 2.0*KastaunMuTolerance(z49, tol); step <= 4096.0*tol; step *= 2.0) {
      const Real hi = fmin(1.0, z49 + step);
      const Real fhi = AdiabatMasterFunction(hi, b2, rpar, r, d, adiabat);
      if ((fm*fhi < 0.0) && MuInsideKastaunBranch(hi, KastaunRbar(hi, b2, rpar, r))) {
        zp = hi;
        fp = fhi;
        break;
      }
    }
  }
  // The gas is that of the adiabat at rho = D/W(mu), and 1/W^2 = 1 - mu^2 rbar, so the
  // density errs by W^2 times the relative error of mu: the root is converged on mu to
  // tol mu/W^2 (where that is above round-off, 4 ulp of mu), which puts the density, and
  // the gas with it, at tol.  On the fallback cells of the test torus the internal energy
  // is then within 6e-10 of a quad-precision root in all but one per mille of them and
  // within 2.5e-7 in all; the energy inversion's tolerance (KastaunMuTolerance) left 5e-4
  // and 5e-3 there, KORAL's Newton iteration 7.5e-6 and 3.5e-2.
  if (fp == 0.0) {
    z = zp;
  } else if (fm*fp < 0.0) {
    z = 0.5*(zm + zp);
    #pragma unroll 1
    for (iter = 0; iter < max_iterations; ++iter) {
      z = (zm*fp - zp*fm)/(fp - fm);
      const Real f = AdiabatMasterFunction(z, b2, rpar, r, d, adiabat);
      const Real tol_z = fmax(tol*z*(1.0 - z*z*KastaunRbar(z, b2, rpar, r)),
                              4.0*std::numeric_limits<Real>::epsilon()*z);
      if ((fabs(zm - zp) < tol_z) || (fabs(f) < tol_z)) break;
      if (f*fp < 0.0) {
        zm = zp; fm = fp; zp = z; fp = f;
      } else {
        fm = 0.5*fm; zp = z; fp = f;
      }
    }
    if (iter == max_iterations) return false;
    iterations = (iter > iterations) ? iter : iterations;
  } else {
    return false;
  }
  iterations_used = (iterations > iterations_used) ? iterations : iterations_used;

  const Real mu = z;
  const Real rbar = KastaunRbar(mu, b2, rpar, r);
  const Real z2 = mu*mu*rbar/fabs(1.0 - mu*mu*rbar);
  const Real lor = sqrt(1.0 + z2);
  const Real rho = d/lor;
  Real eint = 0.0, pressure = 0.0;
  adiabat.At(rho, eint, pressure);
  const Real h = 1.0 + (eint + pressure)/rho;
  const Real conv = lor/(h*lor + b2);  // (C26)
  primitive.d = rho;
  primitive.e = eint;
  primitive.vx = conv*(u_sr.mx/d + u_sr.bx*isqrtd*rpar/(h*lor));
  primitive.vy = conv*(u_sr.my/d + u_sr.by*isqrtd*rpar/(h*lor));
  primitive.vz = conv*(u_sr.mz/d + u_sr.bz*isqrtd*rpar/(h*lor));
  return isfinite(primitive.d) && primitive.d > 0.0 &&
      isfinite(primitive.e) && primitive.e >= 0.0 &&
      isfinite(primitive.vx) && isfinite(primitive.vy) && isfinite(primitive.vz);
}

//----------------------------------------------------------------------------------------
//! \fn bool RecoverMHDWithTransportedEntropy()
//! \brief U2P_ENTROPY of the transported conserved entropy D s (entropy_conserved, in the
//! frame of u_sr.d) for an ideal gas of index gamma_gas: RecoverSRStateOnAdiabat on its
//! adiabat.  B = 0 is the GRHD solve.

KOKKOS_INLINE_FUNCTION
bool RecoverMHDWithTransportedEntropy(
    const MHDCons1D &u_sr, const Real momentum_sq, const Real magnetic_sq,
    const Real momentum_parallel_per_density, const Real entropy_conserved,
    const Real gamma_gas, HydPrim1D &primitive, int &iterations_used) {
  primitive = HydPrim1D{};
  if (!isfinite(u_sr.d) || !(u_sr.d > 0.0) || !isfinite(entropy_conserved) ||
      !isfinite(gamma_gas) || !(gamma_gas > 1.0)) {
    return false;
  }
  const GasEntropyAdiabat adiabat{(gamma_gas - 1.0)*entropy_conserved/u_sr.d, gamma_gas};
  return RecoverSRStateOnAdiabat(u_sr, momentum_sq, magnetic_sq,
                                 momentum_parallel_per_density, adiabat, primitive,
                                 iterations_used);
}

KOKKOS_INLINE_FUNCTION
bool GasEntropyFromRhoInternalEnergy(const Real rho, const Real internal_energy,
                                     const Real gamma_gas, Real &entropy) {
  entropy = 0.0;
  if (!isfinite(rho) || !(rho > 0.0) || !isfinite(internal_energy) ||
      !(internal_energy > 0.0) || !isfinite(gamma_gas) || !(gamma_gas > 1.0)) {
    return false;
  }

  // Algebraically equivalent to KORAL physics.c:1457-1475, but evaluated in log form
  // to avoid an avoidable pow()/division overflow before taking the logarithm.
  const Real gamma_minus_one = gamma_gas - 1.0;
  const Real log_adiabat = log(gamma_minus_one*internal_energy)
                          - gamma_gas*log(rho);
  entropy = rho*log_adiabat/gamma_minus_one;
  return isfinite(entropy);
}

KOKKOS_INLINE_FUNCTION
bool InternalEnergyFromGasEntropy(const Real rho, const Real entropy,
                                  const Real gamma_gas, Real &internal_energy) {
  internal_energy = 0.0;
  if (!isfinite(rho) || !(rho > 0.0) || !isfinite(entropy) ||
      !isfinite(gamma_gas) || !(gamma_gas > 1.0)) {
    return false;
  }

  // Algebraically equivalent to KORAL physics.c:1494-1514.
  const Real gamma_minus_one = gamma_gas - 1.0;
  const Real log_internal_energy = gamma_gas*log(rho)
      + gamma_minus_one*entropy/rho - log(gamma_minus_one);
  internal_energy = exp(log_internal_energy);
  // KORAL rejects only negative/nonfinite energy here.  Exact underflow to zero is a
  // marker for the authoritative ordinary floor pass, not an algebra failure.
  return isfinite(internal_energy) && internal_energy >= 0.0;
}


//! The entropy inversion's result: the primitives in AthenaK's SR convention
//! (rho, internal energy density, u~^i = W v^i), the iterations used, and whether a root
//! was found.
struct SREntropyRecovery {
  HydPrim1D w{};
  int iterations = 0;
  bool ok = false;
};

//----------------------------------------------------------------------------------------
//! \fn SREntropyRecovery RecoverSRStateFromEntropy()
//! \brief KORAL U2P_ENTROPY (u2p.c:240-263 calls it once U2P_HOT returned < 0) on the
//! normal-observer state u_sr = (D, S^i, tau, B^i), with S^2 = momentum_sq,
//! B^2 = magnetic_sq, (B.S)/D = momentum_parallel_per_density and the entropy
//! entropy_conserved = D s in the same frame (RecoverMHDWithTransportedEntropy).  Out
//! of line in device code.

KOKKOS_INLINE_FUNCTION GR_ENTROPY_DEVICE_NOINLINE
SREntropyRecovery RecoverSRStateFromEntropy(
    const MHDCons1D u_sr, const Real momentum_sq, const Real magnetic_sq,
    const Real momentum_parallel_per_density, const Real entropy_conserved,
    const Real gamma_gas) {
  SREntropyRecovery result;
  result.ok = RecoverMHDWithTransportedEntropy(
      u_sr, momentum_sq, magnetic_sq, momentum_parallel_per_density,
      entropy_conserved, gamma_gas, result.w, result.iterations);
  return result;
}

}  // namespace gr_entropy

#endif  // EOS_GR_GAS_ENTROPY_HPP_
