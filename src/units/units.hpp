#ifndef UNITS_UNITS_HPP_
#define UNITS_UNITS_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file units.hpp
//! \brief definitions for Units class

// Athena++ headers
#include "athena.hpp"
#include "parameter_input.hpp"

namespace units {

//----------------------------------------------------------------------------------------
//! \class Units
//! \brief data and definitions of functions used to store and access units
//  Functions are implemented in units.cpp

class Units {
 public:
  explicit Units(ParameterInput *pin);
  ~Units();

  // data
  // CGS unit per X
  static constexpr Real cm_cgs = 1.0;                           // cm
  static constexpr Real pc_cgs = 3.0856775809623245e+18;        // cm
  static constexpr Real kpc_cgs = 3.0856775809623245e+21;       // cm
  static constexpr Real g_cgs = 1.0;                            // g
  static constexpr Real msun_cgs = 1.98841586e+33;              // g
  static constexpr Real atomic_mass_unit_cgs = 1.660538921e-24; // g
  static constexpr Real s_cgs = 1.0;                            // s
  static constexpr Real yr_cgs = 3.15576e+7;                    // s
  static constexpr Real myr_cgs = 3.15576e+13;                  // s
  static constexpr Real cm_s_cgs = 1.0;                         // cm/s
  static constexpr Real km_s_cgs = 1.0e5;                       // cm/s
  static constexpr Real g_cm3_cgs = 1.0;                        // g/cm^3
  static constexpr Real erg_cgs = 1.0;                          // erg
  static constexpr Real dyne_cm2_cgs = 1.0;                     // dyne/cm^2
  static constexpr Real kelvin_cgs = 1.0;                       // k

  // PHYSICAL CONSTANTS
  static constexpr Real k_boltzmann_cgs = 1.3806488e-16;          // erg/k
  static constexpr Real grav_constant_cgs = 6.67408e-8;           // cm^3/(g*s^2)
  static constexpr Real speed_of_light_cgs = 2.99792458e10;       // cm/s
  static constexpr Real rad_constant_cgs = 7.56573325e-15;        // erg/(cm^3*K^4)
  static constexpr Real electron_rest_mass_energy_cgs = 5.93e9;   // k

  // Kramers power-law opacity coefficients (cgs): kappa = coef * rho * T^-3.5.
  // These replaced the pure free-free Rosseland value 8.0e22 (Rybicki & Lightman)
  // in upstream commit 811c2de4, and carry the magnitude of a Kramers BOUND-FREE +
  // FREE-FREE Rosseland mean at roughly solar composition
  // (4.34e25 Z(1+X)/t + 3.68e22 (1-Z)(1+X) with X ~ 0.7, Z ~ 0.02-0.025, guillotine
  // t ~ 1), NOT pure free-free (which is 6e22 Rosseland / 2.2e24 Planck for a
  // solar H/He mix).  The Planck coefficient is 37x the Rosseland one; the bare
  // nu^-3 (1 - e^{-h nu/kT}) law gives 30.26 (radiation/opacity_ff_means.hpp computes
  // it), so the pair carries a Planck/Rosseland Gaunt-factor ratio of ~1.22 from its
  // source; for bound-free the same law is an approximation (edges lower the Planck
  // mean).  For an AGN disk at T ~ 1e5-1e7 K, where bound-free
  // dominates absorption, this is the physically intended choice; pure free-free
  // would be ~25x too transparent.  The ratio is what sets the Rosseland-mean flux
  // damping.  KORAL's default
  // free-free kappa (6.6e-24/m_p^2 ~ 2.4e24, opacities.c) is a PLANCK mean and is
  // not comparable to the Rosseland value here.
  static constexpr Real rosseland_coef_cgs = 1.95705e24;
  static constexpr Real planck_minus_rosseland_coef_cgs = 7.04536e25;

  // functions
  // Code scales in cgs
  Real length_cgs() const;
  Real mass_cgs() const;
  Real time_cgs() const;

  // mean molecular weight
  Real mu() const;

  // Derived code scales in cgs
  Real velocity_cgs() const;
  Real density_cgs() const;
  Real energy_cgs() const;
  Real pressure_cgs() const;
  //! The IDEAL-GAS temperature unit, v_cgs^2*mu*m_u/k_B with the deck's <units>/mu.
  //! It is NOT necessarily the unit of the number the run calls temperature: the code
  //! temperature unit is the EOS's, and Units::temperature_cgs() equals it only for
  //! the ideal EOS (a tabulated EOS builds its own scale).
  //! Consumers must use MeshBlockPack::TemperatureUnitCGS() instead; this stays public
  //! because kelvin(), k_boltzmann() and the EOS itself define that ideal-gas unit.
  Real temperature_cgs() const;

  // Code unit per X, or X in code units
  Real cm() const;
  Real pc() const;
  Real kpc() const;
  Real g() const;
  Real msun() const;
  Real atomic_mass_unit() const;
  Real s() const;
  Real yr() const;
  Real myr() const;
  Real cm_s() const;
  Real km_s() const;
  Real g_cm3() const;
  Real erg() const;
  Real dyne_cm2() const;
  Real kelvin() const;

  // Physical Constants in code units
  Real k_boltzmann() const;
  Real grav_constant() const;
  Real speed_of_light() const;

 private:
  // Specified code scales in cgs units
  // (Multiply code units to get quantities in cgs units)
  // (cgs unit per code unit)
  Real length_cgs_, mass_cgs_, time_cgs_;

  // mean molecular weight
  const Real mu_;
};

} // namespace units

#endif // UNITS_UNITS_HPP_
