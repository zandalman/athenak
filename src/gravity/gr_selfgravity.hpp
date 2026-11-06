#ifndef GRAVITY_GR_SELFGRAVITY_HPP_
#define GRAVITY_GR_SELFGRAVITY_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file gr_selfgravity.hpp
//! \brief Potential service of the weak-field self-gravity of the GR paths (FUGUE model,
//! Plonka & Janiuk, arXiv:2610.20869; the physics is gr_selfgravity_core.hpp's).
//!
//! The multigrid Poisson solver of gravity/mg_gravity.cpp is reused unchanged.  Its GR
//! branch loads the active mass rho_eff = E + gamma_ij S^ij (paper Eq. 19) of every
//! component that gravitates, assembled here from the per-module hooks, in place of the
//! Newtonian rest-mass density.  This facade
//!   - decides whether a pack runs the GR branch (Enabled) and validates the deck,
//!   - assembles the source: zero, the module hooks (fixed-KS gas or dyn-GR gas), then
//!     the gravity weight of the rest-mass density and the excision mask,
//!   - reduces the l=1 reflex acceleration of the pinned hole from that source,
//!   - hands every force adapter the frozen potential as a PhiView.
//! The Poisson source is 4 pi G rho_eff, G the gravitational constant in code units
//! (GravConstant).

#include "athena.hpp"
#include "gravity/gr_selfgravity_core.hpp"

class Driver;
class MeshBlockPack;
class ParameterInput;

namespace gravity {
class Gravity;
}

namespace grsg {

constexpr Real kFourPi = 12.566370614359172;

//! The gravitational constant in code units, derived from <units> and nowhere else.  The
//! metric is geometric (length G M_BH/c^2, time G M_BH/c^3, c = 1, hole mass 1); the code
//! mass unit is rho_u L^3 with rho_u = <units>/density_cgs, so G = G_cgs rho_u t_u^2
//! (Units::grav_constant) = rho_u/rho_geom, rho_geom = M_BH/L^3: exactly 1 for the
//! geometric density (the code mass unit is M_BH) and without <units>.  The hole's mass
//! is 1/G in code mass.  Phi stays dimensionless, and the forces, which carry no G, are
//! unchanged.
Real GravConstant(const MeshBlockPack *pmbp);

//! True when the pack evolves a general-relativistic fluid, fixed Kerr-Schild or dyn-GR
//! (the metric the Newtonian SourceTerms::Gravity cannot be applied on).  Valid once the
//! coordinates exist, so before the Gravity module does.
bool RelativisticPack(const MeshBlockPack *pmbp);

//! True when the GR self-gravity runs: a Gravity module on a relativistic pack.
bool Enabled(const MeshBlockPack *pmbp);

//! True when the hole of the fixed metric is evolved and pinned (Enabled, fixed
//! Kerr-Schild, a hole rather than Minkowski space): the reflex acceleration is on and
//! the face sums into the excised region are accumulated.
bool HoleEvolves(const MeshBlockPack *pmbp);

//! Deck guards of the GR branch (design section 10).  Fatal: gravity/lat_time_centered_work,
//! a four_pi_G other than 4 pi GravConstant, MeshBlocks that are not cubic with a
//! power-of-two size, a hole without excision.  Supplies four_pi_G = 4 pi GravConstant
//! when the deck does not set it.  (Special-relativistic self-gravity and the legacy Newtonian
//! switch are refused by the SourceTerms constructor; the multigrid boundary and
//! mean-subtraction guards sit in the MGGravityDriver constructor.)
void ValidateDeck(const MeshBlockPack *pmbp, ParameterInput *pin);

//! The frozen potential of the last solve and the frame acceleration, for the force
//! adapters.  Inactive when self-gravity is off.  Fatal when self-gravity is on and the
//! potential is not valid: a force from a potential that is not there is never applied.
PhiView MakePhiView(const MeshBlockPack *pmbp);

//! Fills rho_eff(m, 0, k, j, i) over the active cells with the Poisson source density:
//! the gas and radiation hooks, multiplied by the gravity weight of the rest-mass density
//! (rho_min, rho_floor: gravity/rho_grav_min and the density floor), zero in the excised
//! cells and, with mask_radius > 0, outside it.
void AssembleSource(MeshBlockPack *pmbp, DvceArray5D<Real> &rho_eff, Real rho_min,
                    Real rho_floor, Real mask_radius, const Real mask_origin[3]);

//! a_frame = G sum rho_eff dV x / r^3: minus the gravitational field of the source at the
//! hole (the origin), the acceleration the hole has to be given to stay at rest in the
//! frame of the gas.  Per-block partials folded in block order, then one reduction.
//! Zero without a pinned hole (Minkowski, dyn-GR).
void ReflexAcceleration(const MeshBlockPack *pmbp, const DvceArray5D<Real> &rho_eff,
                        Real a_frame[3]);

//! Prints once per run when the potential leaves the weak-field range |Phi| << 1 the
//! model is built for.  Collective.
void WarnIfStrongField(gravity::Gravity *pgrav);

}  // namespace grsg

#endif  // GRAVITY_GR_SELFGRAVITY_HPP_
