#ifndef PGEN_STELLAR_PROFILE_HPP_
#define PGEN_STELLAR_PROFILE_HPP_
//========================================================================================
// AthenaK astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file stellar_profile.hpp
//! \brief The isentropic star in hydrostatic equilibrium with a given EOS and gravity
//! model, used by the Newtonian (tde_external) TDE problem generator.
//!
//! The structure equations are those of the gravity the run evolves:
//!   dp/dr   = -G m Inertia(rho, p) / r^2,
//!   dm/dr   = 4 pi r^2 ActiveDensity(rho, p),
//!   drho/dr = rho / (Gamma1 p) dp/dr            (isentropic: dp/p = Gamma1 drho/rho).
//! Newtonian gravity has Inertia = ActiveDensity = rho.  A weak-field GR
//! self-gravity model has Inertia = rho h and ActiveDensity = rho (1 + eps) + 3 p, with
//! the G of its code units.
//!
//! A Model supplies, on the host:
//!   Real G() const;
//!   Real MassUnit() const;                               // tolerances: max(mass, this)
//!   bool InDomain(Real rho, Real p) const;               // EOS defined at (rho, p)
//!   bool Coefficients(Real rho, Real p, Real &gamma1, Real &inertia,
//!                     Real &active) const;               // false: no valid state
//!   void Densities(Real rho, Real p, Real &inertia, Real &active) const;
//!   bool HasPressureBounds() const;                      // the EOS is tabulated
//!   Real PressureFloor(Real rho) const;                  // p at the table's T_min
//!   Real PressureCeiling(Real rho) const;                // p at the table's T_max
//!   Real DensityMax() const;                             // top of the density axis
//!
//! The profile is found by two nested shootings: the central pressure that makes the
//! pressure vanish exactly at r_surface (for a given central density), then the central
//! density that gives the surface mass.  It is finally renormalised to the exact mass
//! and the pressure rebuilt by hydrostatic quadrature on the profile's own nodes.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <vector>

#include "athena.hpp"

namespace stellar_profile {

struct RadialProfile {
  std::vector<Real> radius;
  std::vector<Real> density;
  std::vector<Real> pressure;
  Real rho_central = 0.0;
  Real p_central = 0.0;
  Real mass_at_surface = 0.0;
  Real pressure_at_surface = 0.0;
  Real surface_radius = 0.0;
  bool reached_target_surface = false;
};

//----------------------------------------------------------------------------------------
//! \fn Real MassScale
//! \brief The mass the tolerances on masses are relative to: the larger of `mass` and
//! the model's mass unit (the Newtonian star's 1), so that
//! a tolerance does not depend on the code mass unit.

template <class Model>
Real MassScale(const Model &model, const Real mass) {
  return std::max(mass, model.MassUnit());
}

//----------------------------------------------------------------------------------------
//! \fn bool BalancedRHS
//! \brief Right-hand side (dm/dr, drho/dr, dp/dr) of the structure equations.

template <class Model>
bool BalancedRHS(const Model &model, const Real r, const Real mass, const Real rho,
                 const Real p, Real &dmdr, Real &drhodr, Real &dpdr) {
  Real gamma1 = 0.0, inertia = 0.0, active = 0.0;
  if (!model.Coefficients(rho, p, gamma1, inertia, active)) {
    return false;
  }
  const Real r_safe = std::max(r, static_cast<Real>(1.0e-30));
  dmdr = 4.0 * M_PI * r_safe * r_safe * active;
  dpdr = -model.G() * mass * inertia / (r_safe * r_safe);
  drhodr = (rho / std::max(gamma1 * p, static_cast<Real>(1.0e-300))) * dpdr;
  return std::isfinite(dmdr) && std::isfinite(drhodr) && std::isfinite(dpdr);
}

//----------------------------------------------------------------------------------------
//! \fn bool IntegrateStructure
//! \brief RK4 from the centre outwards on nsteps uniform steps to r_surface, stopping at
//! the first zero of p or rho (or exit from the EOS domain, or a state at which the model
//! has no valid structure coefficients), where the surface is put by linear interpolation
//! of p.  reached_target_surface is true when the pressure is still positive at
//! r_surface.
//!
//! A state without valid coefficients (Model::Coefficients false: e.g. a negative Gamma1
//! from a tabulated EOS's cold-dense corner) ends the trial like the edge of the EOS
//! domain does: no hydrostatic structure continues through it, so the trial does not
//! reach r_surface.  It is not an error of the solve, whose brackets probe cold trials
//! far from the star.

template <class Model>
bool IntegrateStructure(const Model &model, const Real r_surface,
                        const Real rho_central, const Real p_central, const int nsteps,
                        RadialProfile *profile, Real &mass_at_surface,
                        Real &pressure_at_surface, Real *surface_radius_out = nullptr,
                        bool *reached_target_surface_out = nullptr) {
  if (!(r_surface > 0.0) || !(rho_central > 0.0) || !(p_central > 0.0) || nsteps < 4) {
    return false;
  }

  const Real dr = r_surface / static_cast<Real>(nsteps);
  Real r = 0.5 * dr;
  Real rho = rho_central;
  Real p = p_central;
  Real inertia_central = 0.0, active_central = 0.0;
  model.Densities(rho_central, p_central, inertia_central, active_central);
  Real mass = (4.0 / 3.0) * M_PI * active_central * r * r * r;

  auto finish_surface = [&](const Real r_prev, const Real mass_prev,
                            const Real rho_prev, const Real p_prev,
                            const Real r_term, const Real mass_term,
                            const Real rho_term, const Real p_term) {
    Real frac = 1.0;
    if ((p_prev > 0.0) && (p_term < p_prev) && std::isfinite(p_prev) &&
        std::isfinite(p_term)) {
      frac = p_prev / std::max(p_prev - p_term, static_cast<Real>(1.0e-300));
      frac = std::min(std::max(frac, static_cast<Real>(0.0)), static_cast<Real>(1.0));
    }

    const Real r_zero =
        r_prev + frac * (r_term - r_prev);
    const Real mass_zero =
        mass_prev + frac * (mass_term - mass_prev);
    const Real rho_zero =
        rho_prev + frac * (rho_term - rho_prev);
    const Real r_store =
        std::min(std::max(r_zero, static_cast<Real>(0.0)), r_surface);
    const Real mass_store = std::max(mass_zero, static_cast<Real>(0.0));
    const Real rho_store = std::max(rho_zero, static_cast<Real>(0.0));

    if (profile != nullptr) {
      if (profile->radius.empty() ||
          r_store > profile->radius.back() + static_cast<Real>(1.0e-15) * r_surface) {
        profile->radius.push_back(r_store);
        profile->density.push_back(rho_store);
        profile->pressure.push_back(0.0);
      }
      profile->rho_central = rho_central;
      profile->p_central = p_central;
      profile->mass_at_surface = mass_store;
      profile->pressure_at_surface = p_term;
      profile->surface_radius = r_store;
      profile->reached_target_surface = false;
    }
    mass_at_surface = mass_store;
    pressure_at_surface = p_term;
    if (surface_radius_out != nullptr) {
      *surface_radius_out = r_store;
    }
    if (reached_target_surface_out != nullptr) {
      *reached_target_surface_out = false;
    }
    return true;
  };

  if (profile != nullptr) {
    profile->radius.clear();
    profile->density.clear();
    profile->pressure.clear();
    profile->radius.reserve(static_cast<std::size_t>(nsteps) + 1);
    profile->density.reserve(static_cast<std::size_t>(nsteps) + 1);
    profile->pressure.reserve(static_cast<std::size_t>(nsteps) + 1);
    profile->radius.push_back(0.0);
    profile->density.push_back(rho_central);
    profile->pressure.push_back(p_central);
  }

  for (int step = 0; step < nsteps; ++step) {
    if (!(rho > 0.0) || !(p > 0.0) || !std::isfinite(rho) || !std::isfinite(p)) {
      return finish_surface(r, mass, rho, p, r, mass, rho, p);
    }
    if (!model.InDomain(rho, p)) {
      if (step == 0) {
        return false;
      }
      return finish_surface(r, mass, rho, p, r, mass, rho, p);
    }

    Real k1_m = 0.0, k1_rho = 0.0, k1_p = 0.0;
    if (!BalancedRHS(model, r, mass, rho, p, k1_m, k1_rho, k1_p)) {
      return finish_surface(r, mass, rho, p, r, mass, rho, p);
    }

    const Real r_k2 = r + static_cast<Real>(0.5) * dr;
    const Real mass_k2 = mass + static_cast<Real>(0.5) * dr * k1_m;
    const Real rho_k2 = rho + static_cast<Real>(0.5) * dr * k1_rho;
    const Real p_k2 = p + static_cast<Real>(0.5) * dr * k1_p;
    if (!std::isfinite(mass_k2) || !std::isfinite(rho_k2) || !std::isfinite(p_k2)) {
      return false;
    }
    if (!(rho_k2 > 0.0) || !(p_k2 > 0.0) || !model.InDomain(rho_k2, p_k2)) {
      return finish_surface(r, mass, rho, p, r_k2, mass_k2, rho_k2, p_k2);
    }

    Real k2_m = 0.0, k2_rho = 0.0, k2_p = 0.0;
    if (!BalancedRHS(model, r_k2, mass_k2, rho_k2, p_k2, k2_m, k2_rho, k2_p)) {
      return finish_surface(r, mass, rho, p, r, mass, rho, p);
    }

    const Real r_k3 = r + static_cast<Real>(0.5) * dr;
    const Real mass_k3 = mass + static_cast<Real>(0.5) * dr * k2_m;
    const Real rho_k3 = rho + static_cast<Real>(0.5) * dr * k2_rho;
    const Real p_k3 = p + static_cast<Real>(0.5) * dr * k2_p;
    if (!std::isfinite(mass_k3) || !std::isfinite(rho_k3) || !std::isfinite(p_k3)) {
      return false;
    }
    if (!(rho_k3 > 0.0) || !(p_k3 > 0.0) || !model.InDomain(rho_k3, p_k3)) {
      return finish_surface(r_k2, mass_k2, rho_k2, p_k2,
                            r_k3, mass_k3, rho_k3, p_k3);
    }

    Real k3_m = 0.0, k3_rho = 0.0, k3_p = 0.0;
    if (!BalancedRHS(model, r_k3, mass_k3, rho_k3, p_k3, k3_m, k3_rho, k3_p)) {
      return finish_surface(r, mass, rho, p, r, mass, rho, p);
    }

    const Real r_k4 = r + dr;
    const Real mass_k4 = mass + dr * k3_m;
    const Real rho_k4 = rho + dr * k3_rho;
    const Real p_k4 = p + dr * k3_p;
    if (!std::isfinite(mass_k4) || !std::isfinite(rho_k4) || !std::isfinite(p_k4)) {
      return false;
    }
    if (!(rho_k4 > 0.0) || !(p_k4 > 0.0) || !model.InDomain(rho_k4, p_k4)) {
      return finish_surface(r_k3, mass_k3, rho_k3, p_k3,
                            r_k4, mass_k4, rho_k4, p_k4);
    }

    Real k4_m = 0.0, k4_rho = 0.0, k4_p = 0.0;
    if (!BalancedRHS(model, r_k4, mass_k4, rho_k4, p_k4, k4_m, k4_rho, k4_p)) {
      return finish_surface(r, mass, rho, p, r, mass, rho, p);
    }

    const Real mass_next =
        mass + (dr / 6.0) * (k1_m + 2.0 * k2_m + 2.0 * k3_m + k4_m);
    const Real rho_next =
        rho + (dr / 6.0) * (k1_rho + 2.0 * k2_rho + 2.0 * k3_rho + k4_rho);
    const Real p_next =
        p + (dr / 6.0) * (k1_p + 2.0 * k2_p + 2.0 * k3_p + k4_p);
    const Real r_next = r + dr;

    if (!(mass_next >= 0.0) || !std::isfinite(mass_next) || !std::isfinite(rho_next) ||
        !std::isfinite(p_next)) {
      return false;
    }

    mass = mass_next;
    rho = rho_next;
    p = p_next;
    r = r_next;

    if (profile != nullptr) {
      profile->radius.push_back(std::min(r, r_surface));
      profile->density.push_back(std::max(rho, static_cast<Real>(0.0)));
      profile->pressure.push_back(std::max(p, static_cast<Real>(0.0)));
    }

    if (!(rho > 0.0) || !(p > 0.0)) {
      return finish_surface(r - dr,
          mass - (dr / 6.0) * (k1_m + 2.0 * k2_m + 2.0 * k3_m + k4_m),
                            rho - (dr / 6.0) * (k1_rho + 2.0 * k2_rho + 2.0 * k3_rho + k4_rho),
                            p - (dr / 6.0) * (k1_p + 2.0 * k2_p + 2.0 * k3_p + k4_p),
                            r, mass, rho, p);
    }
  }

  mass_at_surface = mass;
  pressure_at_surface = p;
  if (surface_radius_out != nullptr) {
    *surface_radius_out = r_surface;
  }
  if (reached_target_surface_out != nullptr) {
    *reached_target_surface_out = true;
  }
  if (profile != nullptr) {
    profile->rho_central = rho_central;
    profile->p_central = p_central;
    profile->mass_at_surface = mass_at_surface;
    profile->pressure_at_surface = pressure_at_surface;
    profile->surface_radius = r_surface;
    profile->reached_target_surface = true;
  }
  return true;
}

//----------------------------------------------------------------------------------------
//! \fn bool Renormalize
//! \brief Scales the density to the exact target mass and rebuilds the pressure by
//! hydrostatic quadrature from p(r_surface) = 0 inwards, on the profile's own nodes.
//! The mass and the inertia depend on p in the GR model, so the mass/pressure pair is
//! iterated to a fixed point; the Newtonian model reaches it in one pass (the second pass
//! reproduces the first bit for bit and stops the loop).

template <class Model>
bool Renormalize(const Model &model, const Real target_mass, const Real r_surface,
                 RadialProfile &profile) {
  constexpr int kMaxPasses = 16;
  constexpr Real kPassTol = 1.0e-14;
  const std::size_t npts = profile.radius.size();
  if (!(target_mass > 0.0) || npts < 2 || profile.density.size() != npts ||
      profile.pressure.size() != npts) {
    return false;
  }

  if (profile.radius.back() < r_surface) {
    profile.radius.push_back(r_surface);
    profile.density.push_back(0.0);
    profile.pressure.push_back(0.0);
  } else if (profile.radius.back() > r_surface) {
    profile.radius.back() = r_surface;
  }

  const std::size_t n = profile.radius.size();
  profile.density[n - 1] = 0.0;
  profile.pressure[n - 1] = 0.0;
  std::vector<Real> enclosed_mass(n, 0.0);
  Real current_mass = 0.0;
  for (std::size_t i = 0; i + 1 < n; ++i) {
    const Real r0 = profile.radius[i];
    const Real r1 = profile.radius[i + 1];
    const Real dr = r1 - r0;
    if (!(dr >= 0.0)) return false;
    const Real r_mid = static_cast<Real>(0.5) * (r0 + r1);
    Real inertia0 = 0.0, active0 = 0.0, inertia1 = 0.0, active1 = 0.0;
    model.Densities(std::max(profile.density[i], static_cast<Real>(0.0)),
                    profile.pressure[i], inertia0, active0);
    model.Densities(std::max(profile.density[i + 1], static_cast<Real>(0.0)),
                    profile.pressure[i + 1], inertia1, active1);
    const Real active_mid = static_cast<Real>(0.5) * (active0 + active1);
    current_mass += 4.0 * M_PI * r_mid * r_mid * active_mid * dr;
    enclosed_mass[i + 1] = current_mass;
  }
  if (!(current_mass > 0.0) || !std::isfinite(current_mass)) {
    return false;
  }

  const Real rho_scale = target_mass / current_mass;
  for (Real &rho : profile.density) {
    rho = std::max(rho * rho_scale, static_cast<Real>(0.0));
  }

  std::vector<Real> pressure_new(n, 0.0);
  for (int pass = 0; pass < kMaxPasses; ++pass) {
    enclosed_mass.assign(n, 0.0);
    current_mass = 0.0;
    for (std::size_t i = 0; i + 1 < n; ++i) {
      const Real r0 = profile.radius[i];
      const Real r1 = profile.radius[i + 1];
      const Real dr = r1 - r0;
      const Real r_mid = static_cast<Real>(0.5) * (r0 + r1);
      Real inertia0 = 0.0, active0 = 0.0, inertia1 = 0.0, active1 = 0.0;
      model.Densities(profile.density[i], profile.pressure[i], inertia0, active0);
      model.Densities(profile.density[i + 1], profile.pressure[i + 1], inertia1,
                      active1);
      const Real active_mid = static_cast<Real>(0.5) * (active0 + active1);
      current_mass += 4.0 * M_PI * r_mid * r_mid * active_mid * dr;
      enclosed_mass[i + 1] = current_mass;
    }

    pressure_new.assign(n, 0.0);
    for (std::size_t ii = n - 1; ii > 0; --ii) {
      const std::size_t i = ii - 1;
      const Real r0 = profile.radius[i];
      const Real r1 = profile.radius[i + 1];
      const Real dr = r1 - r0;
      const Real r_mid = std::max(static_cast<Real>(0.5) * (r0 + r1),
                                  static_cast<Real>(1.0e-30));
      Real inertia0 = 0.0, active0 = 0.0, inertia1 = 0.0, active1 = 0.0;
      model.Densities(profile.density[i], profile.pressure[i], inertia0, active0);
      model.Densities(profile.density[i + 1], profile.pressure[i + 1], inertia1,
                      active1);
      const Real inertia_mid = static_cast<Real>(0.5) * (inertia0 + inertia1);
      const Real mass_mid =
          static_cast<Real>(0.5) * (enclosed_mass[i] + enclosed_mass[i + 1]);
      const Real dp = model.G() * mass_mid * inertia_mid * dr / (r_mid * r_mid);
      pressure_new[i] = pressure_new[i + 1] + dp;
      if (!std::isfinite(pressure_new[i]) || !(pressure_new[i] >= 0.0)) {
        return false;
      }
    }

    Real change = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
      change = std::max(change, std::abs(pressure_new[i] - profile.pressure[i]));
    }
    profile.pressure = pressure_new;
    if (change <= kPassTol * pressure_new.front()) break;
    if (pass == kMaxPasses - 1) return false;
  }

  profile.rho_central = profile.density.front();
  profile.p_central = profile.pressure.front();
  profile.mass_at_surface = enclosed_mass.back();
  profile.pressure_at_surface = 0.0;
  profile.surface_radius = profile.radius.back();
  profile.reached_target_surface =
      std::abs(profile.surface_radius - r_surface) <=
      static_cast<Real>(1.0e-12) * std::max(r_surface, static_cast<Real>(1.0));
  return std::isfinite(profile.rho_central) && std::isfinite(profile.p_central) &&
         std::isfinite(profile.mass_at_surface) &&
         std::abs(profile.mass_at_surface - target_mass) <=
             static_cast<Real>(1.0e-12) * MassScale(model, target_mass);
}

//----------------------------------------------------------------------------------------
//! \fn bool SolveCentralPressure
//! \brief For a given central density, the central pressure (bisection in ln p to
//! 1e-10) whose structure keeps p > 0 exactly up to r_surface.

template <class Model>
bool SolveCentralPressure(const Model &model, const Real r_surface,
                          const Real rho_central, const Real p_guess, Real &p_central,
                          Real &mass_at_surface, Real &pressure_at_surface) {
  constexpr int kStructureSteps = 4096;
  constexpr int kMaxBracketIters = 80;
  constexpr int kMaxBisectIters = 96;
  constexpr Real kLogPressureTol = 1.0e-10;

  auto evaluate = [&](const Real trial_p, Real &trial_mass, Real &trial_psurf,
                      Real &trial_rsurf, bool &trial_reached) {
    return IntegrateStructure(model, r_surface, rho_central, trial_p, kStructureSteps,
                              nullptr, trial_mass, trial_psurf, &trial_rsurf,
                              &trial_reached);
  };

  Real p_lo = std::max(p_guess * static_cast<Real>(1.0e-3), static_cast<Real>(1.0e-16));
  Real p_hi = std::max(p_guess, p_lo * static_cast<Real>(10.0));
  if (model.HasPressureBounds()) {
    const Real p_floor = model.PressureFloor(rho_central);
    const Real p_ceil = model.PressureCeiling(rho_central);
    p_lo = std::max(p_lo, p_floor);
    p_hi = std::min(std::max(p_hi, p_lo * static_cast<Real>(10.0)), p_ceil);
  }
  if (!(p_hi >= p_lo) || !std::isfinite(p_lo) || !std::isfinite(p_hi)) return false;

  Real m_lo = 0.0, psurf_lo = 0.0, rsurf_lo = 0.0;
  bool reached_lo = false;
  if (!evaluate(p_lo, m_lo, psurf_lo, rsurf_lo, reached_lo)) return false;
  for (int n = 0; n < kMaxBracketIters && reached_lo; ++n) {
    const Real old_p = p_lo;
    p_lo *= static_cast<Real>(0.1);
    if (model.HasPressureBounds()) {
      p_lo = std::max(p_lo, model.PressureFloor(rho_central));
    }
    if (!(p_lo < old_p)) break;
    if (!evaluate(p_lo, m_lo, psurf_lo, rsurf_lo, reached_lo)) return false;
  }

  if (reached_lo) {
    p_central = p_lo;
    mass_at_surface = m_lo;
    pressure_at_surface = psurf_lo;
    return true;
  }

  Real m_hi = 0.0, psurf_hi = 0.0, rsurf_hi = 0.0;
  bool reached_hi = false;
  if (!evaluate(p_hi, m_hi, psurf_hi, rsurf_hi, reached_hi)) return false;
  for (int n = 0; n < kMaxBracketIters && !reached_hi; ++n) {
    const Real old_p = p_hi;
    p_hi *= static_cast<Real>(10.0);
    if (model.HasPressureBounds()) {
      p_hi = std::min(p_hi, model.PressureCeiling(rho_central));
    }
    if (!(p_hi > old_p) || !std::isfinite(p_hi)) break;
    if (!evaluate(p_hi, m_hi, psurf_hi, rsurf_hi, reached_hi)) return false;
  }
  if (!reached_hi) return false;

  for (int n = 0; n < kMaxBisectIters; ++n) {
    const Real log_p_lo = std::log(std::max(p_lo, static_cast<Real>(1.0e-300)));
    const Real log_p_hi = std::log(std::max(p_hi, static_cast<Real>(1.0e-300)));
    if (std::abs(log_p_hi - log_p_lo) <= kLogPressureTol) break;

    const Real p_mid = std::exp(static_cast<Real>(0.5)*(log_p_lo + log_p_hi));
    Real m_mid = 0.0, psurf_mid = 0.0, rsurf_mid = 0.0;
    bool reached_mid = false;
    if (!evaluate(p_mid, m_mid, psurf_mid, rsurf_mid, reached_mid)) return false;
    if (reached_mid) {
      p_hi = p_mid;
      m_hi = m_mid;
      psurf_hi = psurf_mid;
      rsurf_hi = rsurf_mid;
    } else {
      p_lo = p_mid;
      m_lo = m_mid;
      psurf_lo = psurf_mid;
      rsurf_lo = rsurf_mid;
    }
  }

  p_central = p_hi;
  mass_at_surface = m_hi;
  pressure_at_surface = psurf_hi;
  return true;
}

//----------------------------------------------------------------------------------------
//! \fn bool SolveProfile
//! \brief The star of mass star_mass (the model's surface m) and radius r_surface: the
//! central density by regula falsi in ln rho to a relative mass error of 1e-10, each
//! trial with its own central pressure; then the final structure and the
//! renormalisation.  (rho_guess, p_guess, guess_gamma) seed the brackets only.

template <class Model>
bool SolveProfile(const Model &model, const Real star_mass, const Real r_surface,
                  const Real rho_guess, const Real p_guess, const Real guess_gamma,
                  RadialProfile &profile) {
  constexpr int kStructureSteps = 4096;
  constexpr int kMaxBracketIters = 80;
  constexpr int kMaxBisectIters = 96;
  constexpr Real kMassTolFrac = 1.0e-10;

  auto evaluate = [&](const Real trial_rho, Real &trial_p_c, Real &trial_mass,
                      Real &trial_psurf) {
    const Real scaled_p_guess =
        p_guess * std::pow(std::max(trial_rho, static_cast<Real>(1.0e-30)) /
                               std::max(rho_guess, static_cast<Real>(1.0e-30)),
                           guess_gamma);
    return SolveCentralPressure(model, r_surface, trial_rho, scaled_p_guess,
                                trial_p_c, trial_mass, trial_psurf);
  };

  Real rho_active_max = std::numeric_limits<Real>::max();
  if (model.HasPressureBounds()) {
    rho_active_max = model.DensityMax();
  }
  Real rho_lo = std::max(rho_guess * static_cast<Real>(0.5), static_cast<Real>(1.0e-12));
  Real rho_hi = std::min(std::max(rho_guess, rho_lo * static_cast<Real>(2.0)),
      rho_active_max);
  Real p_lo = 0.0, mass_lo = 0.0, psurf_lo = 0.0;
  Real p_hi = 0.0, mass_hi = 0.0, psurf_hi = 0.0;

  if (!evaluate(rho_lo, p_lo, mass_lo, psurf_lo)) return false;
  for (int n = 0; n < kMaxBracketIters && mass_lo > star_mass; ++n) {
    rho_lo *= static_cast<Real>(0.5);
    if (!(rho_lo > 0.0) || !evaluate(rho_lo, p_lo, mass_lo, psurf_lo)) return false;
  }

  if (!evaluate(rho_hi, p_hi, mass_hi, psurf_hi)) return false;
  for (int n = 0; n < kMaxBracketIters && mass_hi < star_mass; ++n) {
    if (!(rho_hi < rho_active_max)) break;
    rho_hi = std::min(rho_hi * static_cast<Real>(2.0), rho_active_max);
    if (!evaluate(rho_hi, p_hi, mass_hi, psurf_hi)) return false;
  }

  if (!(mass_lo <= star_mass && mass_hi >= star_mass)) return false;

  Real rho_mid = rho_guess;
  Real p_mid = 0.0;
  Real mass_mid = 0.0;
  Real psurf_mid = 0.0;
  for (int n = 0; n < kMaxBisectIters; ++n) {
    const Real log_rho_lo =
        std::log(std::max(rho_lo, static_cast<Real>(1.0e-300)));
    const Real log_rho_hi =
        std::log(std::max(rho_hi, static_cast<Real>(1.0e-300)));
    Real frac = static_cast<Real>(0.5);
    const Real dmass = mass_hi - mass_lo;
    if (std::isfinite(dmass) && dmass > 0.0) {
      const Real trial_frac = (star_mass - mass_lo) / dmass;
      if (trial_frac > 0.0 && trial_frac < 1.0 && std::isfinite(trial_frac)) {
        frac = trial_frac;
      }
    }
    rho_mid = std::exp(log_rho_lo + frac * (log_rho_hi - log_rho_lo));
    if (!evaluate(rho_mid, p_mid, mass_mid, psurf_mid)) return false;
    if (std::abs(mass_mid - star_mass) <=
        kMassTolFrac * MassScale(model, star_mass)) {
      break;
    }
    if (mass_mid > star_mass) {
      rho_hi = rho_mid;
      p_hi = p_mid;
      mass_hi = mass_mid;
      psurf_hi = psurf_mid;
    } else {
      rho_lo = rho_mid;
      p_lo = p_mid;
      mass_lo = mass_mid;
      psurf_lo = psurf_mid;
    }
  }

  if (std::abs(mass_mid - star_mass) >
      kMassTolFrac * MassScale(model, star_mass)) {
    const Real log_rho_lo =
        std::log(std::max(rho_lo, static_cast<Real>(1.0e-300)));
    const Real log_rho_hi =
        std::log(std::max(rho_hi, static_cast<Real>(1.0e-300)));
    Real frac = static_cast<Real>(0.5);
    const Real dmass = mass_hi - mass_lo;
    if (std::isfinite(dmass) && dmass > 0.0) {
      const Real trial_frac = (star_mass - mass_lo) / dmass;
      if (trial_frac > 0.0 && trial_frac < 1.0 && std::isfinite(trial_frac)) {
        frac = trial_frac;
      }
    }
    rho_mid = std::exp(log_rho_lo + frac * (log_rho_hi - log_rho_lo));
    if (!evaluate(rho_mid, p_mid, mass_mid, psurf_mid)) return false;
  }

  Real rho_profile = rho_mid;
  Real p_profile = p_mid;
  if (psurf_mid < 0.0 && psurf_hi >= 0.0) {
    rho_profile = rho_hi;
    p_profile = p_hi;
  }
  Real final_mass = 0.0;
  Real final_psurf = 0.0;
  if (!IntegrateStructure(model, r_surface, rho_profile, p_profile, kStructureSteps,
                          &profile, final_mass, final_psurf)) {
    return false;
  }
  if (!Renormalize(model, star_mass, r_surface, profile)) {
    return false;
  }
  return true;
}

}  // namespace stellar_profile

#endif  // PGEN_STELLAR_PROFILE_HPP_
