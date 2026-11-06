#ifndef COORDINATES_KERR_HOLE_TYPE_HPP_
#define COORDINATES_KERR_HOLE_TYPE_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file kerr_hole_type.hpp
//! \brief the parameters (mass M, spin length a) of the Cartesian Kerr-Schild metric,
//! passed by value to every metric evaluation and horizon test.
//!
//! The Kerr-Schild metric is scale free: g_{mu nu}(x; M, a) = g_{mu nu}(x/M; 1, a/M), and
//! every spatial derivative of it scales by 1/M.  Threading (M, a) instead of the
//! dimensionless spin alone therefore costs one reciprocal and a few multiplies per
//! evaluation, and at M = 1.0 exactly (x*1.0, d*1.0) reduces to the arithmetic of the
//! M = 1 code bit for bit.

#include "athena.hpp"

//----------------------------------------------------------------------------------------
//! \struct KerrHole
//! \brief metric parameters of the central hole.  m is the mass, a the spin length J/M
//! (equal to the dimensionless spin while m = 1), minkowski switches the hole off.

struct KerrHole {
  bool minkowski;
  Real m;
  Real a;
};

//----------------------------------------------------------------------------------------
//! \fn Real KerrHorizonRadius
//! \brief outer horizon r_+ = M (1 + sqrt(1 - (a/M)^2)) in Kerr-Schild radius.  Written
//! in the scaled form so that for m = 1 it is the same arithmetic as 1 + sqrt(1 - a^2)
//! (a/1 and 1*x are exact); the form M + sqrt(M^2 - a^2) can differ from it by an ulp
//! under FMA contraction when a^2 is not exactly representable.  A spin beyond the Kerr
//! bound has no horizon; the radicand is taken as zero there (r_+ = M, the real part),
//! which is what the radiation module's horizon test always did, instead of returning NaN.

KOKKOS_INLINE_FUNCTION
Real KerrHorizonRadius(const KerrHole &h) {
  return h.m*(1.0 + sqrt(fmax(1.0 - SQR(h.a/h.m), 0.0)));
}

#endif // COORDINATES_KERR_HOLE_TYPE_HPP_
