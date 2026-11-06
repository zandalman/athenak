//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file noop_dyngrmhd.cpp
//! \brief derived class for DynGRMHD that acts as a no-op

//----------------------------------------------------------------------------------------
// ctor: also calls EOS base class constructor

#include <iostream>
#include <string>

#include "athena.hpp"
#include "mhd/mhd.hpp"
#include "eos.hpp"
#include "eos/lte_table_core.hpp"
#include "units/units.hpp"

#include "coordinates/coordinates.hpp"
#include "coordinates/cartesian_ks.hpp"
#include "coordinates/cell_locations.hpp"

NoOpDynGRMHD::NoOpDynGRMHD(MeshBlockPack *pp, ParameterInput *pin,
                           const std::string &block) :
    EquationOfState(block, pp, pin) {
  eos_data.is_ideal = true;
  // DynGRMHD recovers primitives with PrimitiveSolver, whose own EOS policy owns the
  // adiabatic index (PrimitiveSolverHydro::SetPolicyParams reads <mhd>/gamma for
  // dyn_eos = ideal and the pwp_* family for dyn_eos = piecewise_poly).  Nothing on the
  // recovery path reads eos_data.gamma; only shared helpers written for the fixed-metric
  // fluids do (srcterms disk cooling converts P to eps with it).  So a deck whose dyn_eos
  // is not ideal need not carry <mhd>/gamma, and the default here is the same 5/3 the
  // ideal policy defaults to, so the two agree whenever the key is absent.
  eos_data.gamma = pin->GetOrAddReal(block, "gamma", 5.0/3.0);
  eos_data.iso_cs = 0.0;
  eos_data.use_e = true;  // ideal gas EOS always uses internal energy
  eos_data.use_t = false;
  eos_data.gamma_max = pin->GetOrAddReal(block, "gamma_max", (FLT_MAX));  // ceiling

  // dyn_eos = lte_table: the gas is the tabulated LTE table in code units, so the code
  // temperature unit is the table's, c^2 m_H/k_B, not the ideal-gas unit the base class
  // stored.  It is published here, from <units> alone, because the radiation modules
  // read MeshBlockPack::TemperatureUnitCGS() in their constructors and they are built
  // before the DynGRMHD policy (which loads the table) exists.
  if (pin->DoesParameterExist(block, "dyn_eos") &&
      pin->GetString(block, "dyn_eos") == "lte_table") {
    if (pp->punit == nullptr || !pin->GetOrAddBoolean("coord", "general_rel", false)) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "<" << block << ">/dyn_eos = lte_table needs a <units> "
                << "block with "
                << "the GR scales (<units>/bhmass_msun and <units>/density_cgs) and "
                << "<coord>/general_rel = true: the table is converted to code units "
                << "G M_BH/c^2, c, <units>/density_cgs." << std::endl;
      std::exit(EXIT_FAILURE);
    }
    eos_data.temp_unit_cgs = lte_eos::LTECodeUnits::FromUnits(*pp->punit).T_u;
    eos_data.dyn_lte_table = true;
  }
}
