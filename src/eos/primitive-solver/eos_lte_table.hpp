#ifndef EOS_PRIMITIVE_SOLVER_EOS_LTE_TABLE_HPP_
#define EOS_PRIMITIVE_SOLVER_EOS_LTE_TABLE_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file eos_lte_table.hpp
//! \brief PrimitiveSolver EOS policy for the tabulated H+He LTE gas (<mhd>/dyn_eos =
//! lte_table).
//!
//! The policy works directly in code units: the table was shifted into them when it was
//! read (see eos/lte_table_core.hpp), the baryon mass is the code mass unit (so n = rho),
//! and the EOS unit system equals the code unit system.  The total energy density is
//! e = n (1 + eps(n, T)) and the enthalpy per baryon h = 1 + eps + p/n.  Temperature
//! inversion from e or from p is the exact inverse of the table's interpolant, so that
//! Energy(n, TemperatureFromE(n, e)) = e to round-off.  Temperatures are code
//! temperatures, in units of c^2 m_H/k_B.
//!
//! The gas entropy of the GR entropy fallback is the table's adiabat label L = ln eps at
//! the last density row on the adiabat through the state (lte_table_core.hpp), built at
//! load.  The table carries no thermodynamic entropy or chemical potentials, so entropy
//! output and neutrino physics are not available.

#include <limits>
#include <string>

#include <Kokkos_Core.hpp>

#include "../../athena.hpp"
#include "../lte_table_core.hpp"
#include "eos_policy_interface.hpp"
#include "ps_types.hpp"
#include "unit_system.hpp"

namespace Primitive {

class EOSLTETable : public EOSPolicyInterface, public SupportsGasEntropy {
 protected:
  EOSLTETable() : m_data("lte table", 1) {
    n_species = 0;
    mb = 1.0;
    eos_units = MakeNuclear();   // equal to the code units set by Primitive::EOS
    min_n = 0.0;
    max_n = std::numeric_limits<Real>::max();
    min_T = 0.0;
    max_T = std::numeric_limits<Real>::max();
    for (int i = 0; i < MAX_SPECIES; i++) {
      min_Y[i] = 0.0;
      max_Y[i] = 1.0;
    }
  }

  /// Temperature from the total energy density.
  KOKKOS_INLINE_FUNCTION Real TemperatureFromE(Real n, Real e, Real *Y) const {
    if (!(n > 0.0)) {
      return min_T;
    }
    const Real eps = e/n - 1.0;
    if (!(eps > 0.0)) {
      return min_T;
    }
    const lte_eos::TableView v = View();
    const Real lnT = lte_eos::LnTFromLnEps(v, Kokkos::log(n), Kokkos::log(eps));
    return (lnT <= v.lt_min) ? min_T : Kokkos::exp(lnT);
  }

  /// Temperature from the pressure.
  KOKKOS_INLINE_FUNCTION Real TemperatureFromP(Real n, Real p, Real *Y) const {
    if (!(n > 0.0) || !(p > 0.0)) {
      return min_T;
    }
    const lte_eos::TableView v = View();
    const Real lnT = lte_eos::LnTFromLnP(v, Kokkos::log(n), Kokkos::log(p));
    return (lnT <= v.lt_min) ? min_T : Kokkos::exp(lnT);
  }

  /// Total energy density, including the rest mass.
  KOKKOS_INLINE_FUNCTION Real Energy(Real n, Real T, const Real *Y) const {
    const lte_eos::TableView v = View();
    const lte_eos::Cell c = lte_eos::Locate(v, Kokkos::log(n), Kokkos::log(T));
    return n*(1.0 + Kokkos::exp(lte_eos::LnAt(v, c, lte_eos::kLnEps)));
  }

  /// Pressure.
  KOKKOS_INLINE_FUNCTION Real Pressure(Real n, Real T, Real *Y) const {
    const lte_eos::TableView v = View();
    const lte_eos::Cell c = lte_eos::Locate(v, Kokkos::log(n), Kokkos::log(T));
    return Kokkos::exp(lte_eos::LnAt(v, c, lte_eos::kLnP));
  }

  /// Enthalpy per baryon, h = 1 + eps + p/n.
  KOKKOS_INLINE_FUNCTION Real Enthalpy(Real n, Real T, Real *Y) const {
    const lte_eos::TableView v = View();
    const lte_eos::Cell c = lte_eos::Locate(v, Kokkos::log(n), Kokkos::log(T));
    return 1.0 + Kokkos::exp(lte_eos::LnAt(v, c, lte_eos::kLnEps)) +
           Kokkos::exp(lte_eos::LnAt(v, c, lte_eos::kLnP))/n;
  }

  /// Sound speed in units of c.  The table's cs2N is Gamma1 p/rho, so
  /// cs^2 = Gamma1 p/(rho h) = cs2N/h.
  KOKKOS_INLINE_FUNCTION Real SoundSpeed(Real n, Real T, Real *Y) const {
    const lte_eos::TableView v = View();
    const lte_eos::Cell c = lte_eos::Locate(v, Kokkos::log(n), Kokkos::log(T));
    const Real h = 1.0 + Kokkos::exp(lte_eos::LnAt(v, c, lte_eos::kLnEps)) +
                   Kokkos::exp(lte_eos::LnAt(v, c, lte_eos::kLnP))/n;
    return Kokkos::sqrt(Kokkos::exp(lte_eos::LnAt(v, c, lte_eos::kLnCs2N))/h);
  }

  /// Specific internal energy eps (without the rest mass).
  KOKKOS_INLINE_FUNCTION Real SpecificInternalEnergy(Real n, Real T, Real *Y) const {
    const lte_eos::TableView v = View();
    const lte_eos::Cell c = lte_eos::Locate(v, Kokkos::log(n), Kokkos::log(T));
    return Kokkos::exp(lte_eos::LnAt(v, c, lte_eos::kLnEps));
  }

  /// The adiabat label L(n, T) (lte_eos::AdiabatLabel).
  KOKKOS_INLINE_FUNCTION Real GasEntropy(Real n, Real T, Real *Y) const {
    return lte_eos::AdiabatLabel(View(), Kokkos::log(n), Kokkos::log(T));
  }

  /// Its exact inverse at fixed n: min_T where L is at or below the label of the first
  /// T node (the EOS lower bound), NaN above the last density row (no label there).  The
  /// last row is max_n_gas_entropy = exp(m_ln_n_label_top), where the entropy root's
  /// bracket ends (PrimitiveSolver::ConToPrim); ln n is held at m_ln_n_label_top, which
  /// locates on that row, so the rounding of exp and log there does not read as above it.
  KOKKOS_INLINE_FUNCTION Real TemperatureFromGasEntropy(Real n, Real s, Real *Y) const {
    if (!(n > 0.0)) {
      return min_T;
    }
    if (!(n <= max_n_gas_entropy)) {
      return std::numeric_limits<Real>::quiet_NaN();
    }
    const lte_eos::TableView v = View();
    const Real lnT = lte_eos::LnTFromAdiabatLabel(
        v, Kokkos::fmin(Kokkos::log(n), m_ln_n_label_top), s);
    if (!(lnT == lnT)) {
      return lnT;
    }
    return (lnT <= v.lt_min) ? min_T : Kokkos::exp(lnT);
  }

  /// eps >= 0 and p > 0 everywhere in the table and in its continuations, so h >= 1.
  KOKKOS_INLINE_FUNCTION Real MinimumEnthalpy() const {
    return 1.0;
  }

  KOKKOS_INLINE_FUNCTION Real MinimumPressure(Real n, Real *Y) const {
    return Pressure(n, min_T, Y);
  }

  KOKKOS_INLINE_FUNCTION Real MaximumPressure(Real n, Real *Y) const {
    return std::numeric_limits<Real>::max();
  }

  KOKKOS_INLINE_FUNCTION Real MinimumEnergy(Real n, Real *Y) const {
    return Energy(n, min_T, Y);
  }

  KOKKOS_INLINE_FUNCTION Real MaximumEnergy(Real n, Real *Y) const {
    return std::numeric_limits<Real>::max();
  }

 public:
  /// Read the table file, shift it into code units, upload it and set the bounds of the
  /// policy.  Returns what the table says about itself.
  lte_eos::LTETableMeta ReadTableFromFile(const std::string &fname,
                                          const lte_eos::LTECodeUnits &units);

  /// The device view of the table (code units).  M1 and the host builders read this.
  KOKKOS_INLINE_FUNCTION lte_eos::TableView View() const {
    lte_eos::TableView v = m_geom;
    v.f = m_data.data();
    return v;
  }

  KOKKOS_INLINE_FUNCTION bool IsInitialized() const {
    return m_initialized;
  }

  /// Number of passive scalars; they do not enter the thermodynamics.
  KOKKOS_INLINE_FUNCTION void SetNSpecies(int n) {
    if (n > MAX_SPECIES || n < 0) {
      return;
    }
    n_species = n;
  }

 private:
  lte_eos::TableView m_geom;        // geometry of the table; the data pointer is unused
  Real m_ln_n_label_top = 0.0;      // ln rho of the last density row, as located on it
  DvceArray1D<Real> m_data;         // lte_eos::TableSize(nr, nt), code units
  bool m_initialized = false;
};

}  // namespace Primitive

#endif  // EOS_PRIMITIVE_SOLVER_EOS_LTE_TABLE_HPP_
