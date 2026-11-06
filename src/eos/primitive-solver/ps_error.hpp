#ifndef EOS_PRIMITIVE_SOLVER_PS_ERROR_HPP_
#define EOS_PRIMITIVE_SOLVER_PS_ERROR_HPP_
//========================================================================================
// PrimitiveSolver equation-of-state framework
// Copyright(C) 2023 Jacob M. Fields <jmf6719@psu.edu>
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file ps_error.hpp
//  \brief defines an enumerator struct for error types.

namespace Primitive {
enum struct Error {
  SUCCESS,
  RHO_TOO_BIG,
  RHO_TOO_SMALL,
  NANS_IN_CONS,
  MAG_TOO_BIG,
  BRACKETING_FAILED,
  NO_SOLUTION,
  CONS_FLOOR,
  PRIM_FLOOR,
  CONS_ADJUSTED,
};

struct SolverResult {
  Error error;
  int  iterations;
  bool cons_floor;
  bool prim_floor;
  bool cons_adjusted;
  //! rho*h as the raw inversion recovered it, before any floor touched the state.  The
  //! GRMHD floors downstream re-inject whatever mass and heat they add in the drift frame
  //! (eos/drift_frame_floor.hpp), and that map needs the enthalpy density it started
  //! from.  Zero means "no usable value" -- a failed or excised inversion -- and the
  //! re-injection is then skipped, which leaves the velocity alone.
  Real w_prefloor = 0.0;
  //! The conserved floor raised D, by any amount.  The temperature recovered afterwards is
  //! the kept tau spread over the raised mass: for D << D_floor that mass is the floor's
  //! and the temperature is no gas temperature.  Every further mass the recovery adds to
  //! the cell then comes cold (see PrimitiveSolver::ConToPrim), which is also the
  //! conservative choice where D was just under D_floor and the mass is mostly the cell's.
  bool cons_mass_floor = false;
  //! The root's internal energy was not positive: the root function's temperature limits
  //! held T at the EOS minimum, so the solve converged on a state the energy equation
  //! does not have.  KORAL reports such an energy-equation root as a failure
  //! (koral_lite u2p.c, u2p_solver_W: uint < 0 returns -104) and inverts the cell again
  //! from its entropy (u2p: U2P_ENTROPY).  Set before the primitive floor, which raises
  //! T to T_atm and so hides it.
  bool energy_at_minimum = false;
  //! The root's specific internal energy is not above what the rounding of the energy
  //! equation leaves of it: eps is the difference of sums of magnitude W S (the root
  //! function's terms, in units of D) with the Lorentz factor W, so kEnergyRoundingOps
  //! eps_mach W S bounds the rounding alone, and a root below it holds no significant
  //! digit of eps however well it reproduces U (see PrimitiveSolver::ConToPrim).  Set for
  //! the energy root only; it is a hot failure for the entropy inversion
  //! (PSHydro::KoralEntropyFallback) and changes nothing else.  KORAL's U2P_HOT accepts
  //! such a root (u2p.c:1728, 1781-1830: it fails on u < 0 or on a 10 % error of UU,
  //! and UU is reproduced to rounding whatever eps is).
  bool energy_unresolved = false;
};

} // namespace Primitive

#endif  // EOS_PRIMITIVE_SOLVER_PS_ERROR_HPP_
