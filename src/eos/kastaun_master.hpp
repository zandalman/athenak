#ifndef EOS_KASTAUN_MASTER_HPP_
#define EOS_KASTAUN_MASTER_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file kastaun_master.hpp
//! \brief The EOS-independent parts of the Kastaun, Kalinani & Ciolfi (2021) inversion
//! of a special-relativistic MHD state: rbar(mu) (eq. 38), the upper bracket mu+ of the
//! master function (eq. 49), the branch test and the root-find tolerance.  Shared by the
//! energy inversion (eos/ideal_c2p_mhd.hpp) and the inversion on an adiabat
//! (eos/gr_gas_entropy.hpp).  Normalized variables throughout: r = |S|/D, b2 = B^2/D,
//! rpar = (B.S)/D^(3/2).

#include <math.h>

#include "athena.hpp"

//----------------------------------------------------------------------------------------
//! \fn Real Equation49()
//! \brief Inline function to compute function fa(mu) defined in eq. 49 of Kastaun et al.
//! The root fa(mu)==0 of this function corresponds to the upper bracket for
//! solving Equation44

KOKKOS_INLINE_FUNCTION
Real Equation49(const Real mu, const Real b2, const Real rp, const Real r, const Real q) {
  Real const x = 1.0/(1.0 + mu*b2);             // (26)
  Real rbar = (x*x*r*r + mu*x*(1.0 + x)*rp*rp); // (38)
  return mu*sqrt(1.0 + rbar) - 1.0;
}

//----------------------------------------------------------------------------------------
//! \fn Real KastaunRbar()
//! \brief Returns rbar(mu), eq. 38 of Kastaun et al., shared by eqs. 44 and 49.

KOKKOS_INLINE_FUNCTION
Real KastaunRbar(const Real mu, const Real b2, const Real rpar, const Real r) {
  Real const x = 1.0/(1.0 + mu*b2);             // (26)
  return (x*x*r*r + mu*x*(1.0 + x)*rpar*rpar);  // (38)
}

//----------------------------------------------------------------------------------------
//! \fn bool MuInsideKastaunBranch()
//! \brief True if mu is at or below the eq. 49 root mu+, i.e. on the branch where the
//! eq. 44 master function is single valued.  This is fa(mu)<=0 written without the
//! square root: mu*sqrt(1+rbar) <= 1 <=> mu^2*(1+rbar) <= 1 for mu >= 0.

KOKKOS_INLINE_FUNCTION
bool MuInsideKastaunBranch(const Real mu, const Real rbar) {
  return (mu*mu*(1.0 + rbar) <= 1.0);
}

//----------------------------------------------------------------------------------------
//! \fn Real KastaunMuTolerance()
//! \brief The convergence tolerance of the root finds of eqs. 44 and 49 for a root near
//! mu.  The absolute tol presumes mu of order unity.  A hot cell in a strong field has
//! mu = 1/(h W) of order 1e-10 (h ~ 1e9: heat of 1e-4 of the field energy density in
//! 1e-12 of its rest mass), and there the absolute tests accept a mu that is off by a
//! few per cent: |f| < tol holds on the flat left side of the root (f = -1/h there), and
//! a warm-start bracket with |f(lo)| + |f(hi)| < 2 tol is returned unsolved.  The Lorentz
//! factor, set by 1 - v^2 = 1/(1 + z2) with z2 ~ mu^2 rbar/(1 - mu^2 rbar), then changes
//! by ~W^2 times the relative error of mu (W = 31.6 became 41.7 in the colliding-flow
//! tube, a root that misses U by 2 % of the total energy).  Below mu = 1e-3 the tolerance
//! is relative, tol*mu, the accuracy the absolute one gives at mu = 1; at and above it
//! the absolute one stands, so that every root a cell had before is reached by the same
//! iterates.

KOKKOS_INLINE_FUNCTION
Real KastaunMuTolerance(const Real mu, const Real tol) {
  return (mu < 1.0e-3) ? tol*mu : tol;
}

#endif  // EOS_KASTAUN_MASTER_HPP_
