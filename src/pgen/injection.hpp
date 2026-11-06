#ifndef PGEN_INJECTION_HPP_
#define PGEN_INJECTION_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file injection.hpp
//! \brief utility functions for injection.cpp

#include <math.h>

#include <algorithm>
#include <iostream>
#include <string>

#include "athena.hpp"
#include "parameter_input.hpp"

KOKKOS_INLINE_FUNCTION
void TR13Accel(Real x, Real y, Real z, Real xdot, Real ydot, Real zdot, Real& xddot, Real& yddot, Real &zddot) {
  
  Real r, rm2, rv;
  Real rxvx, rxvy, rxvz, rxvsq;
  Real rinv5, fac1, fac2;
  
  r = sqrt(x*x + y*y + z*z);
  rm2 = r - 2.0;
  rv = x * xdot + y * ydot + z * zdot;
  rxvx = y * zdot - z * ydot;
  rxvy = z * xdot - x * zdot;
  rxvz = x * ydot - y * xdot;
  rxvsq = rxvx*rxvx + rxvy*rxvy + rxvz*rxvz;
  rinv5 = 1.0 / (r*r*r*r*r);
  fac1 = -rinv5 * (rm2*rm2 + 3.0*rxvsq);
  fac2 = 2.0 * rv / (r * r * rm2);
  xddot = x * fac1 + xdot * fac2;
  yddot = y * fac1 + ydot * fac2;
  zddot = z * fac1 + zdot * fac2;
}
