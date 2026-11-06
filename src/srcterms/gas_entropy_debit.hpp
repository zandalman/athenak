#ifndef SRCTERMS_GAS_ENTROPY_DEBIT_HPP_
#define SRCTERMS_GAS_ENTROPY_DEBIT_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file gas_entropy_debit.hpp
//! \brief The conserved gas entropy D s of a general-relativistic run under an operator
//! that changes the internal energy inside a stage, outside the conservative transport
//! (cooling, heating).  s = ln(p/rho^Gamma)/(Gamma - 1) per unit rest mass (KORAL
//! physics.c:1457-1475), so at fixed rest-mass density and fixed D a change of the
//! internal energy from eint to eint - de moves the column by exactly
//!   D [s(rho, eint - de) - s(rho, eint)] = D ln(1 - de/eint)/(Gamma - 1),
//! evaluated on the state the source was evaluated on (w0).

#include <cmath>

#include "athena.hpp"

//! The ideal-gas law and the energy floors of a cell, the scalars of IdealGRMHDEintFloor
//! (eos/ideal_c2p_mhd.hpp), so that a kernel captures four numbers and not an EOS_Data.
struct CoolingGasLaw {
  Real gamma, pfloor, tfloor, sfloor;
};

//! \brief The change D ln(eint_new/eint)/(Gamma - 1) of the column, eint_new = eint - de,
//! D the conserved density.  de is the COMOVING internal energy removed per unit proper
//! volume (negative for a source).  A sink that removes all of the internal energy or
//! more (de >= eint) drives the cell to its energy floor: eint_new is then
//! IdealGRMHDEintFloor(rho), the state the recovery floor publishes there, derived and
//! not a clamp.  Returns false (no change) where eint, de or Gamma are not usable.
KOKKOS_INLINE_FUNCTION
bool GasEntropyDebitIncrement(const Real dens, const Real de, const Real eint,
                              const Real rho, const CoolingGasLaw &gas, Real &increment) {
  if (!(eint > 0.0) || !isfinite(eint) || !isfinite(de) || !(gas.gamma > 1.0)) {
    return false;
  }
  const Real gm1 = gas.gamma - 1.0;
  Real eint_new = eint - de;
  if (!(eint_new > 0.0)) {
    const Real s_floor = (gas.sfloor > 0.0) ? gas.sfloor*pow(rho, gas.gamma)/gm1 : 0.0;
    eint_new = fmax(fmax(gas.pfloor/gm1, s_floor), rho*gas.tfloor/gm1);
    if (!(eint_new > 0.0)) return false;
  }
  increment = dens*log(eint_new/eint)/gm1;
  return true;
}

//! \brief u0(entropy_idx) += the GasEntropyDebitIncrement of u0(IDN).
KOKKOS_INLINE_FUNCTION
void GasEntropyDebit(const DvceArray5D<Real> &u0, const int entropy_idx, const Real de,
                     const Real eint, const Real rho, const CoolingGasLaw &gas,
                     const int m, const int k, const int j, const int i) {
  Real increment = 0.0;
  if (GasEntropyDebitIncrement(u0(m, IDN, k, j, i), de, eint, rho, gas, increment)) {
    u0(m, entropy_idx, k, j, i) += increment;
  }
}

#endif  // SRCTERMS_GAS_ENTROPY_DEBIT_HPP_
