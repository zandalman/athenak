#ifndef HYDRO_REMOVED_GR_DUAL_ENERGY_HPP_
#define HYDRO_REMOVED_GR_DUAL_ENERGY_HPP_
//========================================================================================
// AthenaK astrophysical fluid dynamics and numerical relativity code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file removed_gr_dual_energy.hpp
//! \brief The refusal of the removed general-relativistic dual-energy keys.  GR fluids
//! (fixed or dynamical metric, <mhd> or <hydro>) carry the gas entropy instead, always,
//! with no key.  The Newtonian dual energy is unaffected.

#include <cstdlib>
#include <iostream>
#include <string>

#include "parameter_input.hpp"

//----------------------------------------------------------------------------------------
//! \fn void RejectRemovedGRDualEnergy()
//! \brief FATAL on <block>/dual_energy = true, dual_energy_eta1 or dual_energy_eta2 in a
//! GR fluid block, and on a checkpoint whose header says <block>/dual_energy = true
//! (main.cpp records the header value under <saha_runtime>, ahead of any deck override):
//! such a file holds D*kappa in the auxiliary column, not D*s.  dual_energy = false is
//! accepted silently: every GR checkpoint written before the removal stores it.

inline void RejectRemovedGRDualEnergy(ParameterInput *pin, const std::string &block) {
  const std::string restart_key = block + "_dual_energy_from_restart";
  if (pin->DoesParameterExist("saha_runtime", restart_key) &&
      pin->GetBoolean("saha_runtime", restart_key)) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << "This checkpoint was written with the removed GR "
              << "dual-energy channel (<" << block << ">/dual_energy = true in its "
              << "header): its auxiliary column holds D*kappa, not the gas entropy, and "
              << "cannot be restarted." << std::endl;
    std::exit(EXIT_FAILURE);
  }
  const bool on = pin->DoesParameterExist(block, "dual_energy") &&
                  pin->GetBoolean(block, "dual_energy");
  if (on || pin->DoesParameterExist(block, "dual_energy_eta1") ||
      pin->DoesParameterExist(block, "dual_energy_eta2")) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << "<" << block << ">/dual_energy was removed for general "
              << "relativity: the GR adiabat channel and its eta1/eta2 thresholds no "
              << "longer exist. The gas entropy fallback (KORAL U2P_ENTROPY) is always "
              << "on and has no key. Delete the key(s)." << std::endl;
    std::exit(EXIT_FAILURE);
  }
}

#endif  // HYDRO_REMOVED_GR_DUAL_ENERGY_HPP_
