#ifndef PGEN_INJECTION_HPP_
#define PGEN_INJECTION_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file injection.hpp
//! \brief Utility functions for injection.cpp: the Tejeda & Rosswog (2013, TR13)
//! generalized Newtonian potential for a Schwarzschild black hole.
//!
//! Units: G = M = c = 1, so lengths are in r_g = GM/c^2 and the horizon is at r = 2.
//! All expressions are singular at r = 2 and meaningless inside it; callers must keep
//! r > 2 (the problem generator excises r < r_in with r_in > 2).

#include <math.h>

#include "athena.hpp"

//----------------------------------------------------------------------------------------
//! \fn TR13Accel
//! \brief TR13 acceleration: a =-(r-2)^2/r^5 [1 + 3|r x v|^2/(r-2)^2] r
//!                                          + 2 (r.v)/(r^2 (r-2)) v

KOKKOS_INLINE_FUNCTION
void TR13Accel(Real x, Real y, Real z, Real xdot, Real ydot, Real zdot,
               Real &xddot, Real &yddot, Real &zddot) {
  const Real r = sqrt(x*x + y*y + z*z);
  const Real rm2 = r - 2.0;
  const Real rv = x*xdot + y*ydot + z*zdot;
  const Real rxvx = y*zdot - z*ydot;
  const Real rxvy = z*xdot - x*zdot;
  const Real rxvz = x*ydot - y*xdot;
  const Real rxvsq = rxvx*rxvx + rxvy*rxvy + rxvz*rxvz;
  const Real rinv5 = 1.0/(r*r*r*r*r);
  const Real fac1 = -rinv5*(rm2*rm2 + 3.0*rxvsq);
  const Real fac2 = 2.0*rv/(r*r*rm2);
  xddot = x*fac1 + xdot*fac2;
  yddot = y*fac1 + ydot*fac2;
  zddot = z*fac1 + zdot*fac2;
}

//----------------------------------------------------------------------------------------
//! \fn TR13Orb
//! \brief Conserved specific orbital energy eg and angular momentum hg of a test particle.

KOKKOS_INLINE_FUNCTION
void TR13Orb(Real x, Real y, Real z, Real xdot, Real ydot, Real zdot,
             Real &eg, Real &hg) {
  const Real r = sqrt(x*x + y*y + z*z);
  const Real rm2 = r - 2.0;
  const Real rv = x*xdot + y*ydot + z*zdot;
  const Real rxvx = y*zdot - z*ydot;
  const Real rxvy = z*xdot - x*zdot;
  const Real rxvz = x*ydot - y*xdot;
  const Real rxvsq = rxvx*rxvx + rxvy*rxvy + rxvz*rxvz;
  eg = 0.5/(rm2*rm2)*(rv*rv + rm2/r*rxvsq) - 1.0/r;
  hg = r/rm2*sqrt(rxvsq);
}

//----------------------------------------------------------------------------------------
//! \fn TR13Pot
//! \brief Velocity-dependent TR13 potential (Lagrangian form).  Note: its gradient at
//! v = 0 is NOT the static force, because the potential depends on velocity; use
//! TR13StaticPot for hydrostatic equilibrium.

KOKKOS_INLINE_FUNCTION
Real TR13Pot(Real x, Real y, Real z, Real xdot, Real ydot, Real zdot) {
  const Real r = sqrt(x*x + y*y + z*z);
  const Real rm2 = r - 2.0;
  const Real rv = x*xdot + y*ydot + z*zdot;
  const Real rxvx = y*zdot - z*ydot;
  const Real rxvy = z*xdot - x*zdot;
  const Real rxvz = x*ydot - y*xdot;
  const Real rxvsq = rxvx*rxvx + rxvy*rxvy + rxvz*rxvz;
  return -1.0/r - 2.0*(r - 1.0)*rv*rv/(r*r*rm2*rm2) - rxvsq/(r*r*rm2);
}

//----------------------------------------------------------------------------------------
//! \fn TR13StaticPot
//! \brief Potential whose gradient is the TR13 acceleration of matter at rest,
//! g(r) = (1 - 2/r)^2 / r^2 (TR13Accel with v = 0), normalized to zero at infinity:
//!   Phi_s(r) = -1/r + 2/r^2 - 4/(3 r^3).
//! Phi_s < 0 for all r > 0.  Used for the hydrostatic atmosphere.

KOKKOS_INLINE_FUNCTION
Real TR13StaticPot(Real r) {
  const Real u = 1.0/r;
  return -u + 2.0*u*u - (4.0/3.0)*u*u*u;
}

#endif  // PGEN_INJECTION_HPP_
