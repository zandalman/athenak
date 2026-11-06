#ifndef PGEN_BH_FORCE_PAIR_HPP_
#define PGEN_BH_FORCE_PAIR_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file bh_force_pair.hpp
//! \brief The external black-hole potential and its force, shared by every module that
//! applies or measures it.
//!
//! rg = 0 is the softened Newtonian point mass, Phi = -G M / r_s, r_s = sqrt(r^2 +
//! softening^2).  rg = G M / c^2 > 0 is the pseudo-Newtonian potential of Wegg (2012, ApJ
//! 749, 183; arXiv:1202.5336), Eq. 1 with the Table 1 coefficients of its potential C,
//! which reproduces the Schwarzschild apsidal precession of orbits with specific energy
//! small against c^2 to within 1%:
//!   Phi = -alpha G M / r_s - (1 - alpha) G M / (r_s - R_x) - G M R_y / r_s^2,
//!   alpha = -(4/3)(2 + sqrt 6),  R_x = (4 sqrt 6 - 9) r_g,  R_y = -(4/3)(2 sqrt 6 - 3) r_g.
//! It is Newtonian at large r (alpha + 1 - alpha = 1) and diverges at r_s = R_x = 0.80
//! r_g, inside the horizon.  Below r_s = 2 R_x = 1.6 r_g the potential is held at its
//! value there (zero force); that region lies inside the sink every run with rg > 0
//! carries (tde_external requires an excision radius of at least 2 r_g).

#include "athena.hpp"

namespace bh_force_pair {

struct Force {
  Real ax = 0.0, ay = 0.0, az = 0.0, phi = 0.0;
};

// Wegg (2012) potential C; R_x and R_y in units of r_g.
constexpr Real kSqrt6 = 2.449489742783178098;
constexpr Real kWeggAlpha = -(4.0/3.0)*(2.0 + kSqrt6);
constexpr Real kWeggRx = 4.0*kSqrt6 - 9.0;
constexpr Real kWeggRy = -(4.0/3.0)*(2.0*kSqrt6 - 3.0);
constexpr Real kWeggHold = 2.0*kWeggRx;  // r_s/r_g below which Phi is held

// Phi at the softened radius r_s.
KOKKOS_INLINE_FUNCTION
Real PotentialOfRadius(const Real rs_in, const Real mass, const Real newton_g,
                       const Real rg) {
  const Real gm = newton_g*mass;
  if (!(rg > 0.0)) return -gm/rs_in;
  const Real rs = fmax(rs_in, kWeggHold*rg);
  return -gm*(kWeggAlpha/rs + (1.0 - kWeggAlpha)/(rs - kWeggRx*rg) +
              kWeggRy*rg/(rs*rs));
}

// dPhi/dr_s at the softened radius r_s (zero where Phi is held).
KOKKOS_INLINE_FUNCTION
Real PotentialSlope(const Real rs, const Real mass, const Real newton_g, const Real rg) {
  const Real gm = newton_g*mass;
  if (!(rg > 0.0)) return gm/(rs*rs);
  if (rs < kWeggHold*rg) return 0.0;
  const Real d = rs - kWeggRx*rg;
  return gm*(kWeggAlpha/(rs*rs) + (1.0 - kWeggAlpha)/(d*d) +
             2.0*kWeggRy*rg/(rs*rs*rs));
}

// Preserve the existing potential expression, including its singular limit
// when both physical softening and sink radius are zero. Do not introduce an
// undocumented numerical softening through a denominator floor.
KOKKOS_INLINE_FUNCTION
Real Potential(const Real x, const Real y, const Real z,
                const Real bhx, const Real bhy, const Real bhz,
                const Real mass, const Real softening, const Real newton_g,
                const Real sink_radius, const Real rg) {
  const Real dx = x - bhx, dy = y - bhy, dz = z - bhz;
  Real r2 = dx*dx + dy*dy + dz*dz;
  if (sink_radius > 0.0) r2 = fmax(r2, sink_radius*sink_radius);
  return PotentialOfRadius(sqrt(r2 + softening*softening), mass, newton_g, rg);
}

// Analytic derivative, NOT the finite-difference force applied by Gravity().
// A radius-clipped potential is constant strictly inside the clipping sphere.
KOKKOS_INLINE_FUNCTION
Force Evaluate(const Real x, const Real y, const Real z,
                                           const Real bhx, const Real bhy, const Real bhz,
                                           const Real mass, const Real softening,
                                           const Real newton_g,
                                           const Real sink_radius, const Real rg) {
  const Real dx = x - bhx, dy = y - bhy, dz = z - bhz;
  const Real r2 = dx*dx + dy*dy + dz*dz;
  Force result;
  result.phi = Potential(x, y, z, bhx, bhy, bhz, mass, softening, newton_g,
                         sink_radius, rg);
  if (sink_radius > 0.0 && r2 <= sink_radius*sink_radius) return result;
  const Real rs = sqrt(r2 + softening*softening);
  // a = -(dPhi/dr_s) (x - x_BH)/r_s, which is -G M (x - x_BH)/r_s^3 for rg = 0
  const Real factor = -PotentialSlope(rs, mass, newton_g, rg)/rs;
  result.ax = factor*dx;
  result.ay = factor*dy;
  result.az = factor*dz;
  return result;
}

struct AxisStencil {
  Real dpl, dpr;
  KOKKOS_INLINE_FUNCTION
  Real Acceleration(const Real spacing) const { return (dpl + dpr)/(2.0*spacing); }
  KOKKOS_INLINE_FUNCTION
  Real WorkRate(const Real left_mass_flux, const Real right_mass_flux,
                const Real spacing) const {
    return (left_mass_flux*dpl + right_mass_flux*dpr)/(2.0*spacing);
  }
};

// Coordinates of the actual adjacent cell centers come from CellCenterX in the
// caller. This retains the source's grid geometry and floating-point ordering.
KOKKOS_INLINE_FUNCTION
AxisStencil Stencil(const int axis, const Real x, const Real y, const Real z,
                     const Real left, const Real right,
                     const Real bhx, const Real bhy, const Real bhz,
                     const Real mass, const Real softening, const Real newton_g,
                     const Real sink_radius, const Real rg) {
  const Real center = Potential(x, y, z, bhx, bhy, bhz, mass, softening,
                                newton_g, sink_radius, rg);
  const Real pl = Potential(axis == 0 ? left : x, axis == 1 ? left : y,
                            axis == 2 ? left : z, bhx, bhy, bhz, mass, softening,
                            newton_g, sink_radius, rg);
  const Real pr = Potential(axis == 0 ? right : x, axis == 1 ? right : y,
                            axis == 2 ? right : z, bhx, bhy, bhz, mass, softening,
                            newton_g, sink_radius, rg);
  return {-(center - pl), -(pr - center)};
}

}  // namespace bh_force_pair

#endif  // PGEN_BH_FORCE_PAIR_HPP_
