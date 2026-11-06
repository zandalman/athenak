#ifndef EOS_PRIMITIVE_SOLVER_EOS_POLICY_INTERFACE_HPP_
#define EOS_PRIMITIVE_SOLVER_EOS_POLICY_INTERFACE_HPP_
//========================================================================================
// PrimitiveSolver equation-of-state framework
// Copyright(C) 2023 Jacob M. Fields <jmf6719@psu.edu>
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file eos_policy_interface.hpp
//  \brief Defines a class that provides all the common class
//         member variables needed by an EOSPolicy.

#include <limits>

#include "ps_types.hpp"
#include "unit_system.hpp"

namespace Primitive {

struct UnitSystem;

class EOSPolicyInterface {
 protected:
  EOSPolicyInterface() = default;
  ~EOSPolicyInterface() = default;

  /// Number of particle species
  int n_species;
  /// Baryon mass
  Real mb;
  /// maximum number density
  Real max_n;
  /// minimum number density
  Real min_n;
  /// maximum number density at which the gas entropy (SupportsGasEntropy) is defined.
  /// A policy whose label covers less density than its EOS lowers it: the LTE table has
  /// no adiabat label above its last density row, where its energy still extrapolates.
  Real max_n_gas_entropy = std::numeric_limits<Real>::max();
  /// maximum temperature
  Real max_T;
  /// minimum temperature
  Real min_T;
  /// maximum Y
  Real min_Y[MAX_SPECIES];
  /// minimum Y
  Real max_Y[MAX_SPECIES];
  /// Code unit system
  UnitSystem code_units;
  /// EOS unit system
  UnitSystem eos_units;
};

} // namespace Primitive

#endif  // EOS_PRIMITIVE_SOLVER_EOS_POLICY_INTERFACE_HPP_
