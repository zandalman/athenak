#ifndef RECONSTRUCT_THERMAL_FLOORS_HPP_
#define RECONSTRUCT_THERMAL_FLOORS_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file thermal_floors.hpp
//! \brief shared helpers for applying hydro thermal floors/ceilings to reconstructed
//! face states before they are consumed by EOS closures.

#include <math.h>

#include "athena.hpp"
#include "eos/eos.hpp"

//----------------------------------------------------------------------------------------
//! \fn ApplyReconstructedHydroThermalFloors()
//! \brief Apply the same thermal floors used by cell-centered hydro states to a
//! reconstructed internal-energy face state before it is consumed by an EOS closure.

KOKKOS_INLINE_FUNCTION
Real ApplyReconstructedHydroThermalFloors(const EOS_Data &eos, const Real dens_in,
                                          Real eint) {
  const Real dens = fmax(dens_in, eos.dfloor);
  Real eint_floor, eint_ceil;
  eos.HydroInternalEnergyDensityBounds(dens, eint_floor, eint_ceil);
  if (eint < eint_floor) {
    eint = eint_floor;
  }

  // Tabulated LTE EOS floors already enforce tfloor through the cached table closure.
  if ((eos.tfloor > 0.0) && !eos.UsesTabulatedLTE()) {
    const auto thermo = eos.EvalThermoStateFromRhoEint(dens, eint);
    if (thermo.temperature < eos.tfloor) {
      eint = dens*eos.SpecificEintFromRhoT(dens, eos.tfloor);
    }
  }

  return eos.ClampHydroInternalEnergyDensityTo(eint, eint_ceil);
}

//----------------------------------------------------------------------------------------
//! \fn ApplyReconstructedIdealGasThermalFloorsAtGamma()
//! \brief ApplyReconstructedHydroThermalFloors() for an ideal gas whose adiabatic index
//! is not eos.gamma: the pressure, entropy and temperature floors and the sound-speed
//! ceiling all convert through (gamma - 1), so a variable-gamma (two-temperature) face
//! evaluates them at its own effective gamma.  The arithmetic is the gamma-law branch of
//! the EOS_Data methods with gamma replaced.  Anything but a plain gamma-law EOS has no
//! gamma to replace and takes the ordinary path.

KOKKOS_INLINE_FUNCTION
Real ApplyReconstructedIdealGasThermalFloorsAtGamma(const EOS_Data &eos, const Real gamma,
                                                    const Real dens_in, Real eint) {
  if (eos.UsesTabulatedLTE() || eos.hydro_eos != HydroEOSModel::gamma_law) {
    return ApplyReconstructedHydroThermalFloors(eos, dens_in, eint);
  }
  const Real dens = fmax(dens_in, eos.dfloor);
  const Real gm1 = gamma - 1.0;
  const Real dens_safe = fmax(dens, static_cast<Real>(1.0e-300));
  const Real eint_floor = fmax(eos.pfloor/gm1,
      (eos.sfloor > 0.0) ? (eos.sfloor*pow(dens, gamma)/gm1) : static_cast<Real>(0.0));
  if (eint < eint_floor) {
    eint = eint_floor;
  }
  if (eos.tfloor > 0.0) {
    const Real temperature = gm1*eint/dens_safe;
    if (temperature < eos.tfloor) {
      eint = dens*(eos.tfloor/gm1);
    }
  }
  if (!(eos.use_e) || !(eos.cs_ceil > 0.0)) {
    return eint;
  }
  return fmin(eint, fmax(dens_safe*SQR(eos.cs_ceil)/(gamma*gm1), eint_floor));
}

//----------------------------------------------------------------------------------------
//! \fn FloorReconstructedInternalEnergyPair()
//! \brief Apply primitive-state thermal floors/ceilings to a reconstructed internal-energy
//! pair so face states stay consistent with cell-centered thermal repairs.

KOKKOS_INLINE_FUNCTION
void FloorReconstructedInternalEnergyPair(const EOS_Data &eos, Real &ql, Real &qr,
                                          const Real ql_rho, const Real qr_rho) {
  ql = ApplyReconstructedHydroThermalFloors(eos, ql_rho, ql);
  qr = ApplyReconstructedHydroThermalFloors(eos, qr_rho, qr);
}

#endif // RECONSTRUCT_THERMAL_FLOORS_HPP_
