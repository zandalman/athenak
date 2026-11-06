//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file gr_bondi.cpp
//! \brief Problem generator for spherically symmetric black hole accretion (Bondi)
//! solution in GR hydrodynamics. Sets up the analytic solution initially, and then code
//! is run to see if solution is preserved.  Automatically outputs errors (difference
//! between initial and final times).

#include <cmath>   // abs(), NAN, pow(), sqrt()
#include <cstring> // strcmp()
#include <iostream>
#include <sstream>
#include <string> // string
#include <cstdio> // fclose

#include <Kokkos_StdAlgorithms.hpp>

#include "athena.hpp"
#include "globals.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "coordinates/coordinates.hpp"
#include "coordinates/cartesian_ks.hpp"
#include "coordinates/cell_locations.hpp"
#include "eos/eos.hpp"
#include "hydro/hydro.hpp"
#include "mhd/mhd.hpp"
#include "dyn_grmhd/dyn_grmhd.hpp"
#include "pgen/pgen.hpp"

namespace {

KOKKOS_INLINE_FUNCTION
static void ComputePrimitiveSingle(Real x1v, Real x2v, Real x3v, CoordData coord,
                                   struct bondi_pgen pgen,
                                   Real& rho, Real& pgas,
                                   Real& uu1, Real& uu2, Real& uu3,
                                   Real *deficit = nullptr);

KOKKOS_INLINE_FUNCTION
static void GetBoyerLindquistCoordinates(struct bondi_pgen pgen,
                                         Real x1, Real x2, Real x3,
                                         Real *pr, Real *ptheta, Real *pphi);

KOKKOS_INLINE_FUNCTION
static Real SolveU0(const Real glower[4][4], Real u1, Real u2, Real u3);

KOKKOS_INLINE_FUNCTION
static Real U0Deficit(const Real glower[4][4], Real u1, Real u2, Real u3);

KOKKOS_INLINE_FUNCTION
static Real CovariantUPhi(const Real glower[4][4], Real x1, Real x2,
                          Real u1, Real u2, Real u3);

KOKKOS_INLINE_FUNCTION
static void TransformVector(struct bondi_pgen pgen,
                            Real a1_bl, Real a2_bl, Real a3_bl,
                            Real x1, Real x2, Real x3, bool bl_azimuth,
                            Real *pa1, Real *pa2, Real *pa3);

KOKKOS_INLINE_FUNCTION
static void CalculatePrimitives(struct bondi_pgen pgen, Real r,
                                Real *prho, Real *ppgas, Real *pur);

KOKKOS_INLINE_FUNCTION
static Real TemperatureMin(struct bondi_pgen pgen, Real r, Real t_min, Real t_max);

KOKKOS_INLINE_FUNCTION
static Real TemperatureBisect(struct bondi_pgen pgen, Real r, Real t_min, Real t_max);

KOKKOS_INLINE_FUNCTION
static Real TemperatureResidual(struct bondi_pgen pgen, Real t, Real r);

struct bondi_pgen {
  Real spin;                // black hole spin
  Real dexcise, pexcise;    // excision parameters
  Real n_adi, k_adi, gm;    // hydro EOS parameters
  Real r_crit;              // sonic point radius
  Real c1, c2;              // useful constants
  Real temp_min, temp_max;  // bounds for temperature root find
  bool reset_ic = false;    // reset initial conditions after run
  bool zero_angmom = false; // radial inflow with u_phi = 0 instead of u^phi_BL = 0
};

  bondi_pgen bondi;

} // namespace

// prototypes for user-defined BCs and error functions
void FixedBondiInflow(Mesh *pm);
void BondiErrors(ParameterInput *pin, Mesh *pm);

//----------------------------------------------------------------------------------------
//! \fn void ProblemGenerator::BondiAccretion_()
//! \brief set initial conditions for Bondi accretion test
//  Compile with '-D PROBLEM=gr_bondi' to enroll as user-specific problem generator
//    reference: Hawley, Smarr, & Wilson 1984, ApJ 277 296 (HSW)

void ProblemGenerator::BondiAccretion(ParameterInput *pin, const bool restart) {
  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;

  if (pmbp->pdyngr == nullptr) {
    if (!pmbp->pcoord->is_general_relativistic) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl
                << "GR bondi problem can only be run when GR defined in <coord> block"
                << std::endl;
      exit(EXIT_FAILURE);
    }
  }

  // set user-defined BCs and error function pointers
  pgen_final_func = BondiErrors;
  user_bcs_func = FixedBondiInflow;

  // Read problem-specific parameters from input file
  // global parameters
  bondi.k_adi = pin->GetReal("problem", "k_adi");
  bondi.zero_angmom = pin->GetOrAddBoolean("problem", "zero_angmom", false);
  bondi.r_crit = pin->GetReal("problem", "r_crit");

  // Get ideal gas EOS data
  if (pmbp->pdyngr == nullptr) {
    bondi.gm = pmbp->phydro->peos->eos_data.gamma;
  } else {
    bondi.gm = pmbp->pmhd->peos->eos_data.gamma;
  }
  Real gm1 = bondi.gm - 1.0;
  if (pmbp->pdyngr != nullptr) {
    gm1 = 1.;
  }

  // Parameters
  bondi.temp_min = 1.0e-2;  // lesser temperature root must be greater than this
  bondi.temp_max = 1.0e1;   // greater temperature root must be less than this

  // Spin of the deck's hole.  The HSW solution below is in units of its mass M = 1, so its
  // Boyer-Lindquist map takes <coord>/a.  CoordData::bh_spin is not that on a restart of
  // an evolving hole (kerr_hole::RestoreHole has put the live spin length J/M of the
  // grown hole there), and the boundary states would differ from the uninterrupted run's.
  bondi.spin = pmbp->pcoord->coord_data.is_minkowski ? 0.0 : pin->GetReal("coord", "a");

  // Get excision parameters
  bondi.dexcise = pmbp->pcoord->coord_data.dexcise;
  bondi.pexcise = pmbp->pcoord->coord_data.pexcise;

  // Get ratio of specific heats
  bondi.n_adi = 1.0/(bondi.gm - 1.0);

  // Prepare various constants for determining primitives
  Real u_crit_sq = 1.0/(2.0*bondi.r_crit);                           // (HSW 71)
  Real u_crit = -sqrt(u_crit_sq);
  Real t_crit = (bondi.n_adi/(bondi.n_adi+1.0)
                 * u_crit_sq/(1.0-(bondi.n_adi+3.0)*u_crit_sq));     // (HSW 74)
  bondi.c1 = pow(t_crit, bondi.n_adi) * u_crit * SQR(bondi.r_crit);  // (HSW 68)
  bondi.c2 = (SQR(1.0 + (bondi.n_adi+1.0) * t_crit)
              * (1.0 - 3.0/(2.0*bondi.r_crit)));                     // (HSW 69)

  // A restart needs the solution constants for the boundary states, not the initial data.
  if (restart) return;

  // capture variables for the kernel
  auto &indcs = pmy_mesh_->mb_indcs;
  auto &size = pmbp->pmb->mb_size;
  auto &coord = pmbp->pcoord->coord_data;
  auto bondi_ = bondi;
  int &ng = indcs.ng;
  int n1 = indcs.nx1 + 2*ng;
  int n2 = (indcs.nx2 > 1)? (indcs.nx2 + 2*ng) : 1;
  int n3 = (indcs.nx3 > 1)? (indcs.nx3 + 2*ng) : 1;
  int is = indcs.is;
  int js = indcs.js;
  int ks = indcs.ks;
  int nmb = pmbp->nmb_thispack;
  DvceArray5D<Real> w0_;
  if (pmbp->pdyngr != nullptr) {
    w0_ = pmbp->pmhd->w0;
  } else {
    w0_ = pmbp->phydro->w0;
  }

  // A spatial 4-velocity needs a timelike completion u^0 in every cell outside the
  // horizon.  The Boyer-Lindquist start u^phi_BL = 0 (zero_angmom = false) has none inside
  // the ergosphere of a spinning hole, where no static observer exists and a flow without
  // azimuthal motion in these coordinates is not physical; the u_phi = 0 inflow
  // (zero_angmom = true) is regular there.  Refuse rather than clamp.
  {
    Real maxdef = 0.0;
    int ndef = 0;
    int ncell = nmb*n3*n2*n1;
    Kokkos::parallel_reduce("pgen_bondi_u0", Kokkos::RangePolicy<>(DevExeSpace(), 0, ncell),
    KOKKOS_LAMBDA(const int idx, Real &dmax, int &cnt) {
      int i = idx % n1;
      int j = (idx / n1) % n2;
      int k = (idx / (n1*n2)) % n3;
      int m = idx / (n1*n2*n3);
      Real x1v = CellCenterX(i-is, indcs.nx1, size.d_view(m).x1min, size.d_view(m).x1max);
      Real x2v = CellCenterX(j-js, indcs.nx2, size.d_view(m).x2min, size.d_view(m).x2max);
      Real x3v = CellCenterX(k-ks, indcs.nx3, size.d_view(m).x3min, size.d_view(m).x3max);
      Real rho, pgas, uu1, uu2, uu3, def;
      ComputePrimitiveSingle(x1v,x2v,x3v,coord,bondi_,rho,pgas,uu1,uu2,uu3,&def);
      if (def > 0.0) {
        cnt += 1;
        if (def > dmax) dmax = def;
      }
    }, Kokkos::Max<Real>(maxdef), Kokkos::Sum<int>(ndef));
    if (ndef > 0) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
                << "<problem>/zero_angmom = " << (bondi.zero_angmom ? "true" : "false")
                << " with <coord>/a = " << bondi.spin << ": " << ndef
                << " cells outside the horizon have no timelike completion of the radial "
                << "inflow (discriminant of the u^0 equation negative by up to "
                << maxdef << " of its terms)." << std::endl
                << "The mesh reaches into the ergosphere, where u^phi_BL = 0 is not a "
                << "possible flow; set <problem>/zero_angmom = true (the u_phi = 0 "
                << "inflow) or use a = 0." << std::endl;
      exit(EXIT_FAILURE);
    }
  }

  // Initialize primitive values (HYDRO ONLY)
  par_for("pgen_bondi", DevExeSpace(), 0,(nmb-1),0,(n3-1),0,(n2-1),0,(n1-1),
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    Real rho, pgas, uu1, uu2, uu3;
    Real &x1min = size.d_view(m).x1min;
    Real &x1max = size.d_view(m).x1max;
    Real x1v = CellCenterX(i-is, indcs.nx1, x1min, x1max);

    Real &x2min = size.d_view(m).x2min;
    Real &x2max = size.d_view(m).x2max;
    Real x2v = CellCenterX(j-js, indcs.nx2, x2min, x2max);

    Real &x3min = size.d_view(m).x3min;
    Real &x3max = size.d_view(m).x3max;
    Real x3v = CellCenterX(k-ks, indcs.nx3, x3min, x3max);

    ComputePrimitiveSingle(x1v,x2v,x3v,coord,bondi_,rho,pgas,uu1,uu2,uu3);
    w0_(m,IDN,k,j,i) = rho;
    w0_(m,IEN,k,j,i) = pgas/gm1;
    w0_(m,IM1,k,j,i) = uu1;
    w0_(m,IM2,k,j,i) = uu2;
    w0_(m,IM3,k,j,i) = uu3;
  });

  // Initialize ADM variables
  if (pmbp->padm != nullptr) {
    pmbp->padm->SetADMVariables(pmbp);

    auto &b0 = pmbp->pmhd->b0;
    auto &bcc_ = pmbp->pmhd->bcc0;
    int ie = indcs.ie;
    int je = indcs.je;
    int ke = indcs.ke;
    par_for("pgen_bvars", DevExeSpace(), 0,nmb-1,ks,ke,js,je,is,ie,
    KOKKOS_LAMBDA(int m, int k, int j, int i) {
      b0.x1f(m,k,j,i) = b0.x2f(m,k,j,i) = b0.x3f(m,k,j,i) = 0.0;
      bcc_(m,IBX,k,j,i) = bcc_(m,IBY,k,j,i) = bcc_(m,IBZ,k,j,i) = 0.0;
      if (i==ie) {
        b0.x1f(m,k,j,i+1) = 0.0;
      }
      if (j==je) {
        b0.x2f(m,k,j+1,i) = 0.0;
      }
      if (k==ke) {
        b0.x3f(m,k+1,j,i) = 0.0;
      }
    });
  }

  // Convert primitives to conserved
  if (pmbp->padm == nullptr) {
    auto &u0_ = pmbp->phydro->u0;
    auto &u1_ = pmbp->phydro->u1.data;
    if (bondi.reset_ic) {
      pmbp->phydro->peos->PrimToCons(w0_, u1_, 0, (n1-1), 0, (n2-1), 0, (n3-1));
    } else {
      pmbp->phydro->peos->PrimToCons(w0_, u0_, 0, (n1-1), 0, (n2-1), 0, (n3-1));
    }
  } else {
    if (bondi.reset_ic) {
      // TODO(JMF): need to call PrimToCon in a way that operates on u1, not u0.
      auto &u0_ = pmbp->pmhd->u0;
      auto &u1_ = pmbp->pmhd->u1;
      Kokkos::deep_copy(DevExeSpace(), u1_, u0_);
      pmbp->pdyngr->PrimToConInit(0, n1-1, 0, n2-1, 0, n3-1);
    } else {
      pmbp->pdyngr->PrimToConInit(0, n1-1, 0, n2-1, 0, n3-1);
    }
  }

  return;
}

//----------------------------------------------------------------------------------------
//! \fn void ProblemGenerator::BondiErrors_()
//! \brief Computes errors in Bondi solution and outputs to file, using generic error
//! output function.

void BondiErrors(ParameterInput *pin, Mesh *pm) {
  // calculate reference solution by calling pgen again.  Solution stored in second
  // register u1/b1 when flag is false.
  // The reference goes into u1 over the whole block (Hydro and MHD::ExpandStartRegister).
  if (pm->pmb_pack->phydro != nullptr) pm->pmb_pack->phydro->ExpandStartRegister();
  bondi.reset_ic=true;
  pm->pgen->BondiAccretion(pin, false);
  pm->pgen->OutputErrors(pin, pm);
  return;
}

namespace {

//----------------------------------------------------------------------------------------
// Function for returning corresponding Boyer-Lindquist coordinates of point
// Inputs:
//   x1,x2,x3: global coordinates to be converted
// Outputs:
//   pr,ptheta,pphi: variables pointed to set to Boyer-Lindquist coordinates

KOKKOS_INLINE_FUNCTION
static void ComputePrimitiveSingle(Real x1v, Real x2v, Real x3v, CoordData coord,
                                   struct bondi_pgen pgen,
                                   Real& rho, Real& pgas,
                                   Real& uu1, Real& uu2, Real& uu3,
                                   Real *deficit) {
  // Calculate Boyer-Lindquist coordinates of cell
  Real r, theta, phi;
  GetBoyerLindquistCoordinates(pgen, x1v, x2v, x3v, &r, &theta, &phi);

  // Compute primitive in BL coordinates, transform to Cartesian KS
  Real my_rho, my_pgas, my_ur;
  CalculatePrimitives(pgen, r, &my_rho, &my_pgas, &my_ur);
  Real u0(0.0), u1(0.0), u2(0.0), u3(0.0);
  // With zero_angmom the azimuthal velocity is solved for below, so the radial inflow
  // starts from its Kerr-Schild components at fixed phi_KS, not from the Boyer-Lindquist
  // u^phi_BL = 0 (see TransformVector).
  TransformVector(pgen, my_ur, 0.0, 0.0, x1v, x2v, x3v, !pgen.zero_angmom,
                  &u1, &u2, &u3);

  Real glower[4][4], gupper[4][4];
  ComputeMetricAndInverse(x1v,x2v,x3v, coord.Hole(), glower, gupper);

  u0 = SolveU0(glower, u1, u2, u3);

  // The Schwarzschild solution above has u^phi_BL = 0, which for a != 0 carries the
  // angular momentum u_phi = g_{t phi} u^t of a frame-dragged flow.  With zero_angmom the
  // inflow instead gets the rotation Omega (-y, x, 0) of the spatial 4-velocity that
  // makes the covariant u_phi vanish, so the hole's spin is accreted without angular
  // momentum: Newton iteration on the linear-in-Omega u_phi, u^0 renormalised at every
  // trial.  It starts from the radial inflow at fixed phi_KS (Omega = 0 there): the
  // Boyer-Lindquist start, Omega = a u^r/Delta, diverges at the horizon, and in the
  // ergosphere (g_00 > 0) such a spatial vector has no timelike completion (SolveU0's
  // discriminant is negative), so the iteration stalled on a vector of norm up to +3 in
  // a tenth of the r > 1 cells of a = 0.9 (W ~ 300, -u_t ~ 380 where the gas took it).
  // From the radial start it converges to the zero-angular-momentum inflow, whose u_t is
  // the Boyer-Lindquist -sqrt((Delta Sigma + Sigma^2 (u^r)^2)/A), regular at the
  // horizon.
  if (pgen.zero_angmom && (SQR(x1v) + SQR(x2v)) > 1.0e-20) {
    const Real u1r = u1, u2r = u2, u3r = u3;
    const Real dw = 1.0e-4;
    Real omega = 0.0;
    for (int it = 0; it < 8; ++it) {
      const Real f0 = CovariantUPhi(glower, x1v, x2v, u1r - omega*x2v,
                                    u2r + omega*x1v, u3r);
      const Real f1 = CovariantUPhi(glower, x1v, x2v, u1r - (omega+dw)*x2v,
                                    u2r + (omega+dw)*x1v, u3r);
      if (f1 == f0) break;
      omega -= f0*dw/(f1 - f0);
    }
    u1 = u1r - omega*x2v;
    u2 = u2r + omega*x1v;
    u0 = SolveU0(glower, u1, u2, u3);
  }

  // Whether the spatial velocity has a timelike completion at all (checked at startup)
  // (cells inside the horizon r_+ are excised, whatever state is set there)
  if (deficit != nullptr) {
    const Real r_hor = 1.0 + sqrt(fmax(1.0 - SQR(pgen.spin), 0.0));
    *deficit = (r > r_hor) ? U0Deficit(glower, u1, u2, u3) : 0.0;
  }

  if (r > 1.0) {
    rho = my_rho;
    pgas = my_pgas;
    uu1 = u1 - gupper[0][1]/gupper[0][0] * u0;
    uu2 = u2 - gupper[0][2]/gupper[0][0] * u0;
    uu3 = u3 - gupper[0][3]/gupper[0][0] * u0;
  } else {
    rho = pgen.dexcise;
    pgas = pgen.pexcise;
    uu1 = 0.0;
    uu2 = 0.0;
    uu3 = 0.0;
  }

  return;
}


//----------------------------------------------------------------------------------------
// u^0 of the future-directed unit 4-velocity with spatial components u^i
// (the root of g_{mu nu} u^mu u^nu = -1 used by ComputePrimitiveSingle)

KOKKOS_INLINE_FUNCTION
static Real SolveU0(const Real glower[4][4], Real u1, Real u2, Real u3) {
  Real tmp = glower[1][1]*u1*u1 + 2.0*glower[1][2]*u1*u2 + 2.0*glower[1][3]*u1*u3
           + glower[2][2]*u2*u2 + 2.0*glower[2][3]*u2*u3
           + glower[3][3]*u3*u3;
  Real gammasq = 1.0 + tmp;
  Real b = glower[0][1]*u1 + glower[0][2]*u2 + glower[0][3]*u3;
  // The discriminant of g_{mu nu} u^mu u^nu = -1 in u^0 is non-negative for a spatial
  // vector that has a timelike completion.  A negative one beyond round-off is refused by
  // the startup check (BondiAccretion), so this clamp only removes round-off there; the
  // Newton trial vectors of zero_angmom pass through it without being the result.
  return (-b - sqrt(fmax(SQR(b) - glower[0][0]*gammasq, 0.0)))/glower[0][0];
}

//----------------------------------------------------------------------------------------
// How far below zero the discriminant of the u^0 equation is, relative to its terms
// (0 when the spatial 4-velocity has a timelike completion, up to round-off)

KOKKOS_INLINE_FUNCTION
static Real U0Deficit(const Real glower[4][4], Real u1, Real u2, Real u3) {
  Real tmp = glower[1][1]*u1*u1 + 2.0*glower[1][2]*u1*u2 + 2.0*glower[1][3]*u1*u3
           + glower[2][2]*u2*u2 + 2.0*glower[2][3]*u2*u3
           + glower[3][3]*u3*u3;
  Real gammasq = 1.0 + tmp;
  Real b = glower[0][1]*u1 + glower[0][2]*u2 + glower[0][3]*u3;
  Real t1 = SQR(b), t2 = glower[0][0]*gammasq;
  Real disc = t1 - t2;
  Real scale = fabs(t1) + fabs(t2);
  return (disc < -1.0e-10*scale) ? -disc/scale : 0.0;
}

//----------------------------------------------------------------------------------------
// Covariant angular momentum per unit mass u_phi = u_mu (d_phi)^mu, d_phi = (0, -y, x, 0)
// in Cartesian Kerr-Schild, for the unit 4-velocity with spatial components u^i

KOKKOS_INLINE_FUNCTION
static Real CovariantUPhi(const Real glower[4][4], Real x1, Real x2,
                          Real u1, Real u2, Real u3) {
  const Real u0 = SolveU0(glower, u1, u2, u3);
  const Real ux = glower[1][0]*u0 + glower[1][1]*u1 + glower[1][2]*u2 + glower[1][3]*u3;
  const Real uy = glower[2][0]*u0 + glower[2][1]*u1 + glower[2][2]*u2 + glower[2][3]*u3;
  return -x2*ux + x1*uy;
}

//----------------------------------------------------------------------------------------
// Function for returning corresponding Boyer-Lindquist coordinates of point
// Inputs:
//   x1,x2,x3: global coordinates to be converted
// Outputs:
//   pr,ptheta,pphi: variables pointed to set to Boyer-Lindquist coordinates

KOKKOS_INLINE_FUNCTION
static void GetBoyerLindquistCoordinates(struct bondi_pgen pgen,
                                         Real x1, Real x2, Real x3,
                                         Real *pr, Real *ptheta, Real *pphi) {
    Real rad = sqrt(SQR(x1) + SQR(x2) + SQR(x3));
    Real r = fmax((sqrt( SQR(rad) - SQR(pgen.spin) + sqrt(SQR(SQR(rad)-SQR(pgen.spin))
                        + 4.0*SQR(pgen.spin)*SQR(x3)) ) / sqrt(2.0)), 1.0);
    *pr = r;
    *ptheta = acos(x3/r);
    *pphi = atan2(r*x2-pgen.spin*x1, pgen.spin*x2+r*x1) -
            pgen.spin*r/(SQR(r)-2.0*r+SQR(pgen.spin));
  return;
}

//----------------------------------------------------------------------------------------
// Function for transforming 4-vector from Boyer-Lindquist to desired coordinates
// Inputs:
//   a0_bl,a1_bl,a2_bl,a3_bl: upper 4-vector components in Boyer-Lindquist coordinates
//   x1,x2,x3: Cartesian Kerr-Schild coordinates of point
//   bl_azimuth: a3_bl is u^phi_BL (true), or u^phi_KS (false: no frame-dragging term)
// Outputs:
//   pa0,pa1,pa2,pa3: pointers to upper 4-vector components in desired coordinates
// Notes:
//   Schwarzschild coordinates match Boyer-Lindquist when a = 0

KOKKOS_INLINE_FUNCTION
static void TransformVector(struct bondi_pgen pgen,
                            Real a1_bl, Real a2_bl, Real a3_bl,
                            Real x1, Real x2, Real x3, bool bl_azimuth,
                            Real *pa1, Real *pa2, Real *pa3) {
  Real rad = sqrt( SQR(x1) + SQR(x2) + SQR(x3) );
  Real r = fmax((sqrt( SQR(rad) - SQR(pgen.spin) + sqrt(SQR(SQR(rad)-SQR(pgen.spin))
                      + 4.0*SQR(pgen.spin)*SQR(x3)) ) / sqrt(2.0)), 1.0);
  Real delta = SQR(r) - 2.0*r + SQR(pgen.spin);
  // d phi_KS = d phi_BL + (a/Delta) dr: the frame-dragging term of a vector with the
  // given u^phi_BL; without bl_azimuth the radial component is taken at fixed phi_KS.
  const Real drag1 = bl_azimuth ? x2*pgen.spin/delta : 0.0;
  const Real drag2 = bl_azimuth ? x1*pgen.spin/delta : 0.0;
  *pa1 = a1_bl * ( (r*x1+pgen.spin*x2)/(SQR(r) + SQR(pgen.spin)) - drag1) +
         a2_bl * x1*x3/r * sqrt((SQR(r) + SQR(pgen.spin))/(SQR(x1) + SQR(x2))) -
         a3_bl * x2;
  *pa2 = a1_bl * ( (r*x2-pgen.spin*x1)/(SQR(r) + SQR(pgen.spin)) + drag2) +
         a2_bl * x2*x3/r * sqrt((SQR(r) + SQR(pgen.spin))/(SQR(x1) + SQR(x2))) +
         a3_bl * x1;
  *pa3 = a1_bl * x3/r -
         a2_bl * r * sqrt((SQR(x1) + SQR(x2))/(SQR(r) + SQR(pgen.spin)));
  return;
}

//----------------------------------------------------------------------------------------
// Function for calculating primitives given radius
// Inputs:
//   r: Schwarzschild radius
//   temp_min,temp_max: bounds on temperature
// Outputs:
//   prho: value set to density
//   ppgas: value set to gas pressure
//   put: value set to u^t in Schwarzschild coordinates
//   pur: value set to u^r in Schwarzschild coordinates
// Notes:
//   references Hawley, Smarr, & Wilson 1984, ApJ 277 296 (HSW)

KOKKOS_INLINE_FUNCTION
static void CalculatePrimitives(struct bondi_pgen pgen, Real r,
                                Real *prho, Real *ppgas, Real *pur) {
  // Calculate solution to (HSW 76)
  Real temp_neg_res = TemperatureMin(pgen, r, pgen.temp_min, pgen.temp_max);
  Real temp;
  if (r <= pgen.r_crit) {  // use lesser of two roots
    temp = TemperatureBisect(pgen, r, pgen.temp_min, temp_neg_res);
  } else {  // user greater of two roots
    temp = TemperatureBisect(pgen, r, temp_neg_res, pgen.temp_max);
  }

  // Calculate primitives
  Real rho = pow(temp/pgen.k_adi, pgen.n_adi);             // not same K as HSW
  Real pgas = temp * rho;
  Real ur = pgen.c1 / (SQR(r) * pow(temp, pgen.n_adi));    // (HSW 75)

  // Set primitives
  *prho = rho;
  *ppgas = pgas;
  *pur = ur;
  return;
}

//----------------------------------------------------------------------------------------
// Function for finding temperature at which residual is minimized
// Inputs:
//   r: Schwarzschild radius
//   t_min,t_max: bounds between which minimum must occur
// Outputs:
//   returned value: some temperature for which residual of (HSW 76) is negative
// Notes:
//   references Hawley, Smarr, & Wilson 1984, ApJ 277 296 (HSW)
//   performs golden section search (cf. Numerical Recipes, 3rd ed., 10.2)

KOKKOS_INLINE_FUNCTION
static Real TemperatureMin(struct bondi_pgen pgen, Real r, Real t_min, Real t_max) {
  // Parameters
  const Real ratio = 0.3819660112501051;  // (3+\sqrt{5})/2
  const int max_iterations = 40;          // maximum number of iterations

  // Initialize values
  Real t_mid = t_min + ratio * (t_max - t_min);
  Real res_mid = TemperatureResidual(pgen, t_mid, r);

  // Apply golden section method
  bool larger_to_right = true;  // flag indicating larger subinterval is on right
  for (int n = 0; n < max_iterations; ++n) {
    if (res_mid < 0.0) {
      return t_mid;
    }
    Real t_new;
    if (larger_to_right) {
      t_new = t_mid + ratio * (t_max - t_mid);
      Real res_new = TemperatureResidual(pgen, t_new, r);
      if (res_new < res_mid) {
        t_min = t_mid;
        t_mid = t_new;
        res_mid = res_new;
      } else {
        t_max = t_new;
        larger_to_right = false;
      }
    } else {
      t_new = t_mid - ratio * (t_mid - t_min);
      Real res_new = TemperatureResidual(pgen, t_new, r);
      if (res_new < res_mid) {
        t_max = t_mid;
        t_mid = t_new;
        res_mid = res_new;
      } else {
        t_min = t_new;
        larger_to_right = true;
      }
    }
  }
  return NAN;
}

//----------------------------------------------------------------------------------------
// Bisection root finder
// Inputs:
//   r: Schwarzschild radius
//   t_min,t_max: bounds between which root must occur
// Outputs:
//   returned value: temperature that satisfies (HSW 76)
// Notes:
//   references Hawley, Smarr, & Wilson 1984, ApJ 277 296 (HSW)
//   performs bisection search

KOKKOS_INLINE_FUNCTION
static Real TemperatureBisect(struct bondi_pgen pgen, Real r, Real t_min, Real t_max) {
  // Parameters
  const int max_iterations = 40;
  const Real tol_residual = 1.0e-12;
  const Real tol_temperature = 1.0e-12;

  // Find initial residuals
  Real res_min = TemperatureResidual(pgen, t_min, r);
  Real res_max = TemperatureResidual(pgen, t_max, r);
  if (std::abs(res_min) < tol_residual) {
    return t_min;
  }
  if (std::abs(res_max) < tol_residual) {
    return t_max;
  }
  if ((res_min < 0.0 && res_max < 0.0) || (res_min > 0.0 && res_max > 0.0)) {
    return NAN;
  }

  // Iterate to find root
  Real t_mid;
  for (int i = 0; i < max_iterations; ++i) {
    t_mid = (t_min + t_max) / 2.0;
    if (t_max - t_min < tol_temperature) {
      return t_mid;
    }
    Real res_mid = TemperatureResidual(pgen, t_mid, r);
    if (std::abs(res_mid) < tol_residual) {
      return t_mid;
    }
    if ((res_mid < 0.0 && res_min < 0.0) || (res_mid > 0.0 && res_min > 0.0)) {
      t_min = t_mid;
      res_min = res_mid;
    } else {
      t_max = t_mid;
      res_max = res_mid;
    }
  }
  return t_mid;
}

//----------------------------------------------------------------------------------------
// Function whose value vanishes for correct temperature
// Inputs:
//   t: temperature
//   r: Schwarzschild radius
// Outputs:
//   returned value: residual that should vanish for correct temperature
// Notes:
//   implements (76) from Hawley, Smarr, & Wilson 1984, ApJ 277 296

KOKKOS_INLINE_FUNCTION
static Real TemperatureResidual(struct bondi_pgen pgen, Real t, Real r) {
  return SQR(1.0 + (pgen.n_adi+1.0) * t)
      * (1.0 - 2.0/r + SQR(pgen.c1)
         / (SQR(SQR(r)) * pow(t, 2.0*pgen.n_adi))) - pgen.c2;
}

} // namespace

//----------------------------------------------------------------------------------------
//! \fn FixedBondiInflow
//  \brief Sets boundary condition on surfaces of computational domain
// Note quantities at boundaryies are held fixed to initial condition values

void FixedBondiInflow(Mesh *pm) {
  auto &indcs = pm->mb_indcs;
  auto &size = pm->pmb_pack->pmb->mb_size;
  auto &coord = pm->pmb_pack->pcoord->coord_data;
  int &ng = indcs.ng;
  int n1 = indcs.nx1 + 2*ng;
  int n2 = (indcs.nx2 > 1)? (indcs.nx2 + 2*ng) : 1;
  int n3 = (indcs.nx3 > 1)? (indcs.nx3 + 2*ng) : 1;
  int &is = indcs.is;  int &ie  = indcs.ie;
  int &js = indcs.js;  int &je  = indcs.je;
  int &ks = indcs.ks;  int &ke  = indcs.ke;
  auto &mb_bcs = pm->pmb_pack->pmb->mb_bcs;
  auto bondi_ = bondi;
  bool use_dyngr = pm->pmb_pack->pdyngr != nullptr;

  int nmb = pm->pmb_pack->nmb_thispack;
  DvceArray5D<Real> u0_, w0_;
  if (use_dyngr) {
    u0_ = pm->pmb_pack->pmhd->u0;
    w0_ = pm->pmb_pack->pmhd->w0;
  } else {
    u0_ = pm->pmb_pack->phydro->u0;
    w0_ = pm->pmb_pack->phydro->w0;
  }

  Real gm1 = 1.0;
  if (!use_dyngr) {
    pm->pmb_pack->phydro->peos->SetUserBoundaryFaceFilter(BoundaryFace::inner_x1);
    pm->pmb_pack->phydro->peos->ConsToPrim(u0_,w0_,false,is-ng,is-1,0,(n2-1),0,(n3-1));
    pm->pmb_pack->phydro->peos->ClearUserBoundaryFaceFilter();
    pm->pmb_pack->phydro->peos->SetUserBoundaryFaceFilter(BoundaryFace::outer_x1);
    pm->pmb_pack->phydro->peos->ConsToPrim(u0_,w0_,false,ie+1,ie+ng,0,(n2-1),0,(n3-1));
    pm->pmb_pack->phydro->peos->ClearUserBoundaryFaceFilter();
    gm1 = bondi_.gm - 1.0;
  }
  par_for("fixed_x1", DevExeSpace(),0,(nmb-1),0,(n3-1),0,(n2-1),0,(ng-1),
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    // inner x1 boundary
    Real &x1min = size.d_view(m).x1min;
    Real &x1max = size.d_view(m).x1max;
    Real x1v = CellCenterX(i-is, indcs.nx1, x1min, x1max);

    Real &x2min = size.d_view(m).x2min;
    Real &x2max = size.d_view(m).x2max;
    Real x2v = CellCenterX(j-js, indcs.nx2, x2min, x2max);

    Real &x3min = size.d_view(m).x3min;
    Real &x3max = size.d_view(m).x3max;
    Real x3v = CellCenterX(k-ks, indcs.nx3, x3min, x3max);

    Real rho, pgas, uu1, uu2, uu3;
    if (mb_bcs.d_view(m,BoundaryFace::inner_x1) == BoundaryFlag::user) {
      ComputePrimitiveSingle(x1v,x2v,x3v,coord,bondi_,rho,pgas,uu1,uu2,uu3);
      w0_(m,IDN,k,j,i) = rho;
      w0_(m,IEN,k,j,i) = pgas/gm1;
      w0_(m,IM1,k,j,i) = uu1;
      w0_(m,IM2,k,j,i) = uu2;
      w0_(m,IM3,k,j,i) = uu3;
    }

    // outer x1 boundary
    x1v = CellCenterX((ie+i+1)-is, indcs.nx1, x1min, x1max);

    if (mb_bcs.d_view(m,BoundaryFace::outer_x1) == BoundaryFlag::user) {
      ComputePrimitiveSingle(x1v,x2v,x3v,coord,bondi_, rho,pgas,uu1,uu2,uu3);
      w0_(m,IDN,k,j,(ie+i+1)) = rho;
      w0_(m,IEN,k,j,(ie+i+1)) = pgas/gm1;
      w0_(m,IM1,k,j,(ie+i+1)) = uu1;
      w0_(m,IM2,k,j,(ie+i+1)) = uu2;
      w0_(m,IM3,k,j,(ie+i+1)) = uu3;
    }
  });
  // PrimToCons on X1 physical boundary ghost zones
  if (use_dyngr) {
    pm->pmb_pack->pdyngr->PrimToConInit(is-ng,is-1,0,(n2-1),0,(n3-1));
    pm->pmb_pack->pdyngr->PrimToConInit(ie+1,ie+ng,0,(n2-1),0,(n3-1));
  } else {
    pm->pmb_pack->phydro->peos->SetUserBoundaryFaceFilter(BoundaryFace::inner_x1);
    pm->pmb_pack->phydro->peos->PrimToCons(w0_,u0_,is-ng,is-1,0,(n2-1),0,(n3-1));
    pm->pmb_pack->phydro->peos->ClearUserBoundaryFaceFilter();
    pm->pmb_pack->phydro->peos->SetUserBoundaryFaceFilter(BoundaryFace::outer_x1);
    pm->pmb_pack->phydro->peos->PrimToCons(w0_,u0_,ie+1,ie+ng,0,(n2-1),0,(n3-1));
    pm->pmb_pack->phydro->peos->ClearUserBoundaryFaceFilter();

    pm->pmb_pack->phydro->peos->SetUserBoundaryFaceFilter(BoundaryFace::inner_x2);
    pm->pmb_pack->phydro->peos->ConsToPrim(u0_,w0_,false,0,(n1-1),js-ng,js-1,0,(n3-1));
    pm->pmb_pack->phydro->peos->ClearUserBoundaryFaceFilter();
    pm->pmb_pack->phydro->peos->SetUserBoundaryFaceFilter(BoundaryFace::outer_x2);
    pm->pmb_pack->phydro->peos->ConsToPrim(u0_,w0_,false,0,(n1-1),je+1,je+ng,0,(n3-1));
    pm->pmb_pack->phydro->peos->ClearUserBoundaryFaceFilter();
  }

  par_for("fixed_x2", DevExeSpace(),0,(nmb-1),0,(n3-1),0,(ng-1),0,(n1-1),
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    // inner x2 boundary
    Real &x1min = size.d_view(m).x1min;
    Real &x1max = size.d_view(m).x1max;
    Real x1v = CellCenterX(i-is, indcs.nx1, x1min, x1max);

    Real &x2min = size.d_view(m).x2min;
    Real &x2max = size.d_view(m).x2max;
    Real x2v = CellCenterX(j-js, indcs.nx2, x2min, x2max);

    Real &x3min = size.d_view(m).x3min;
    Real &x3max = size.d_view(m).x3max;
    Real x3v = CellCenterX(k-ks, indcs.nx3, x3min, x3max);

    Real rho, pgas, uu1, uu2, uu3;
    if (mb_bcs.d_view(m,BoundaryFace::inner_x2) == BoundaryFlag::user) {
      ComputePrimitiveSingle(x1v,x2v,x3v,coord,bondi_,rho,pgas,uu1,uu2,uu3);
      w0_(m,IDN,k,j,i) = rho;
      w0_(m,IEN,k,j,i) = pgas/gm1;
      w0_(m,IM1,k,j,i) = uu1;
      w0_(m,IM2,k,j,i) = uu2;
      w0_(m,IM3,k,j,i) = uu3;
    }

    // outer x2 boundary
    x2v = CellCenterX((je+j+1)-js, indcs.nx2, x2min, x2max);

    if (mb_bcs.d_view(m,BoundaryFace::outer_x2) == BoundaryFlag::user) {
      ComputePrimitiveSingle(x1v,x2v,x3v,coord,bondi_,rho,pgas,uu1,uu2,uu3);
      w0_(m,IDN,k,(je+j+1),i) = rho;
      w0_(m,IEN,k,(je+j+1),i) = pgas/gm1;
      w0_(m,IM1,k,(je+j+1),i) = uu1;
      w0_(m,IM2,k,(je+j+1),i) = uu2;
      w0_(m,IM3,k,(je+j+1),i) = uu3;
    }
  });
  if (use_dyngr) {
    pm->pmb_pack->pdyngr->PrimToConInit(0,(n1-1),js-ng,js-1,0,(n3-1));
    pm->pmb_pack->pdyngr->PrimToConInit(0,(n1-1),je+1,je+ng,0,(n3-1));
  } else {
    pm->pmb_pack->phydro->peos->SetUserBoundaryFaceFilter(BoundaryFace::inner_x2);
    pm->pmb_pack->phydro->peos->PrimToCons(w0_,u0_,0,(n1-1),js-ng,js-1,0,(n3-1));
    pm->pmb_pack->phydro->peos->ClearUserBoundaryFaceFilter();
    pm->pmb_pack->phydro->peos->SetUserBoundaryFaceFilter(BoundaryFace::outer_x2);
    pm->pmb_pack->phydro->peos->PrimToCons(w0_,u0_,0,(n1-1),je+1,je+ng,0,(n3-1));
    pm->pmb_pack->phydro->peos->ClearUserBoundaryFaceFilter();

    pm->pmb_pack->phydro->peos->SetUserBoundaryFaceFilter(BoundaryFace::inner_x3);
    pm->pmb_pack->phydro->peos->ConsToPrim(u0_,w0_,false,0,(n1-1),0,(n2-1),ks-ng,ks-1);
    pm->pmb_pack->phydro->peos->ClearUserBoundaryFaceFilter();
    pm->pmb_pack->phydro->peos->SetUserBoundaryFaceFilter(BoundaryFace::outer_x3);
    pm->pmb_pack->phydro->peos->ConsToPrim(u0_,w0_,false,0,(n1-1),0,(n2-1),ke+1,ke+ng);
    pm->pmb_pack->phydro->peos->ClearUserBoundaryFaceFilter();
  }
  par_for("fixed_ix3", DevExeSpace(),0,(nmb-1),0,(ng-1),0,(n2-1),0,(n1-1),
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    // inner x3 boundary
    Real &x1min = size.d_view(m).x1min;
    Real &x1max = size.d_view(m).x1max;
    Real x1v = CellCenterX(i-is, indcs.nx1, x1min, x1max);

    Real &x2min = size.d_view(m).x2min;
    Real &x2max = size.d_view(m).x2max;
    Real x2v = CellCenterX(j-js, indcs.nx2, x2min, x2max);

    Real &x3min = size.d_view(m).x3min;
    Real &x3max = size.d_view(m).x3max;
    Real x3v = CellCenterX(k-ks, indcs.nx3, x3min, x3max);

    Real rho, pgas, uu1, uu2, uu3;
    if (mb_bcs.d_view(m,BoundaryFace::inner_x3) == BoundaryFlag::user) {
      ComputePrimitiveSingle(x1v,x2v,x3v,coord,bondi_,rho,pgas,uu1,uu2,uu3);
      w0_(m,IDN,k,j,i) = rho;
      w0_(m,IEN,k,j,i) = pgas/gm1;
      w0_(m,IM1,k,j,i) = uu1;
      w0_(m,IM2,k,j,i) = uu2;
      w0_(m,IM3,k,j,i) = uu3;
    }

    // outer x3 boundary
    x3v = CellCenterX((ke+k+1)-ks, indcs.nx3, x3min, x3max);

    if (mb_bcs.d_view(m,BoundaryFace::outer_x3) == BoundaryFlag::user) {
      ComputePrimitiveSingle(x1v,x2v,x3v,coord,bondi_,rho,pgas,uu1,uu2,uu3);
      w0_(m,IDN,(ke+k+1),j,i) = rho;
      w0_(m,IEN,(ke+k+1),j,i) = pgas/gm1;
      w0_(m,IM1,(ke+k+1),j,i) = uu1;
      w0_(m,IM2,(ke+k+1),j,i) = uu2;
      w0_(m,IM3,(ke+k+1),j,i) = uu3;
    }
  });
  if (use_dyngr) {
    pm->pmb_pack->pdyngr->PrimToConInit(0,(n1-1),0,(n2-1),ks-ng,ks-1);
    pm->pmb_pack->pdyngr->PrimToConInit(0,(n1-1),0,(n2-1),ke+1,ke+ng);
  } else {
    pm->pmb_pack->phydro->peos->SetUserBoundaryFaceFilter(BoundaryFace::inner_x3);
    pm->pmb_pack->phydro->peos->PrimToCons(w0_,u0_,0,(n1-1),0,(n2-1),ks-ng,ks-1);
    pm->pmb_pack->phydro->peos->ClearUserBoundaryFaceFilter();
    pm->pmb_pack->phydro->peos->SetUserBoundaryFaceFilter(BoundaryFace::outer_x3);
    pm->pmb_pack->phydro->peos->PrimToCons(w0_,u0_,0,(n1-1),0,(n2-1),ke+1,ke+ng);
    pm->pmb_pack->phydro->peos->ClearUserBoundaryFaceFilter();
  }

  return;
}

#if USER_PROBLEM_ENABLED
//----------------------------------------------------------------------------------------
//! \fn void ProblemGenerator::UserProblem()
//! \brief Built as its own executable (-D PROBLEM=tests/gr_bondi, or an
//! ATHENA_EXTRA_PROBLEMS entry), the built-in dispatch is compiled out and this generator
//! is the user entry point.

void ProblemGenerator::UserProblem(ParameterInput *pin, const bool restart) {
  BondiAccretion(pin, restart);
}
#endif
