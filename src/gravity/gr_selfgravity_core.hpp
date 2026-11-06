#ifndef GRAVITY_GR_SELFGRAVITY_CORE_HPP_
#define GRAVITY_GR_SELFGRAVITY_CORE_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file gr_selfgravity_core.hpp
//! \brief Device-side physics of weak-field self-gravity on the GR paths (FUGUE model,
//! Plonka & Janiuk, arXiv:2610.20869).  Every GR adapter (fixed-KS fluid, dyn-GR) takes its source, force and gradient from here and nowhere else.
//!
//! The model.  A Newtonian-like potential solves the flat Poisson equation
//!     nabla^2 Phi = 4 pi varrho,   varrho = E + S = E + gamma_ij S^ij              (1a)
//! (paper Eqs. 18-19; E = T_mn n^m n^n, S^ij the Eulerian stress).  Every component of
//! the stress-energy feels the four-force
//!     G_nu = -(T_nu^j + epsbar delta_nu^j) d_j Phi,                                (1)
//! with epsbar = T_ab u^a u^b its own rest-frame (Landau) energy density and dPhi/dt
//! neglected.  For MHD epsbar = rho(1+eps) + b^2/2 and (1) is the paper's Eqs. 25-26,
//! I^mn = T^mn + epsbar g^mn.  u_nu G^nu = 0 whenever u is the timelike eigenvector of T,
//! so the force does no work in the rest frame.  Conservation reads
//! d_t(sqrt(-g) T^t_nu) + ... = +sqrt(-g) G_nu (from paper Eq. 27; the printed minus
//! of Eq. 30 would make gravity repulsive and is not used).
//!
//! Projected on the Eulerian frame (n^j = -beta^j/alpha, (dPhi)^k = gamma^kj d_j Phi) the
//! densitized sources of sqrt(gamma) E (or tau) and sqrt(gamma) S_i are
//!     s_tau = sqrt(gamma) [ -alpha S_k (dPhi)^k + (E - epsbar) beta^j d_j Phi ]
//!     s_i   = sqrt(gamma) [ -alpha S_ik (dPhi)^k + S_i beta^j d_j Phi
//!                           - alpha epsbar d_i Phi ]                               (2)
//! identically for gas (E, S_i, S_ij, epsbar) and radiation (E, F_i, P_ij, Ebar).
//! Fixed Cartesian KS (sqrt(-g) = 1, C_nu = T^t_nu, IEN = C_t + D, IM_i = C_i):
//!     dC_i = s_i ,   dC_t = -alpha s_tau + beta^i s_i = sqrt(-g) G_t                (3)
//! D is never sourced.  dyn-GR: d tau = s_tau, d S_i = s_i.
//! Radiation (KORAL's radiation fluid R^ij = 4/3 Erf u_R^i u_R^j + 1/3 Erf g^ij with Erf
//! the rest-frame energy, koral_lite rad.c:3098-3126 calc_Rij_M1; Ebar here is Erf):
//!     G_R = -(R_nu^j + Ebar delta_nu^j) d_j Phi,  varrho_R = 2 E_R (traceless R)     (4)
//!     Ebar = 1/2 [ (E - P_par) + sqrt( (E + P_par)^2 - 4 F^2 ) ]                    (5)
//! with P_par = F^i F^j P_ij / F^2, exact for every closure with P = a gamma + b f^f^.
//!
//! The fluid energy as a face work.  With dPhi/dt neglected G_t = -T^j_t d_j Phi
//! exactly, and sqrt(-g) T^j_t is the flux of C_t; in the Valencia form likewise
//!     s_tau = -F^j[tau + D] d_j Phi - sqrt(gamma) epsbar beta^j d_j Phi,
//!     F^j[tau + D] = sqrt(gamma) (alpha S^j - beta^j E).                           (3a)
//! The source is the work of the energy current, and the scheme moves energy and rest
//! mass with its numerical face fluxes, which carry the Riemann solver's dissipation.
//! The fluid adapters therefore form the flux part from those fluxes, each face's flux
//! times its own one-sided potential difference, half to each of its two cells
//! (FaceWorkRate; the Godunov mass-flux work of Mullen, Hanawa & Gammie 2021, as in
//! SourceTerms::Gravity).  Summed over cells it telescopes to sum Phi dE/dt|transport,
//! so E + (1/2) int varrho Phi closes at Newtonian order to the solver tolerance.  The
//! cell value of T^j_t books kinetic energy for rho v while the mass moves with the face
//! flux; the two differ at O(1) wherever the flow is under-resolved, and at the Evrard
//! bounce (64^3, test T4) the difference integrated to 40 % of the binding energy.  The
//! momentum stays the pointwise s_i, so u.G = 0 holds to O(dx^2) in smooth flow; the
//! remainder is the potential energy released by the scheme's own mass diffusion, which
//! becomes heat.  Under LAT the work of the window-end reflux at a coarse/fine face is
//! not booked (the Newtonian refund ledger has no GR counterpart).  GR M1 keeps the
//! cell form of Eq. (2): its share of the work is O(E_R/rho) of the gas's.
//!
//! Symmetric 3x3 arrays (g_dd, g_uu, S_dd) use the ps_types.hpp order
//! S11,S12,S13,S22,S23,S33 = xx,xy,xz,yy,yz,zz.  No templates; no views except phi.

#include <math.h>

#include "../athena.hpp"
#include "../utils/gravity_weight.hpp"

namespace grsg {

//----------------------------------------------------------------------------------------
//! \struct PhiView
//! \brief The frozen potential and frame acceleration, built by the GR self-gravity
//! facade once per task entry and captured by value in every force kernel (never
//! modified there).  a_frame is the l=1 reflex acceleration of the pinned hole (zero on
//! the dyn path and in Minkowski).  Force adapters weight their source by
//! gravity_weight::Weight(rho, dfloor, rho_min) in rest-mass density, the same weight
//! the Poisson source carries.

struct PhiView {
  DvceArray5D<Real> phi;
  Real a_frame[3];
  Real rho_min, dfloor;
  bool active;
};

//----------------------------------------------------------------------------------------
//! \fn int SymIdx
//! \brief Position of (a,b) in a symmetric 3x3 array stored xx,xy,xz,yy,yz,zz.

KOKKOS_INLINE_FUNCTION
int SymIdx(const int a, const int b) {
  return (a <= b) ? (a*(5 - a))/2 + b : (b*(5 - b))/2 + a;
}

//----------------------------------------------------------------------------------------
//! \fn void GradPhi
//! \brief d_j Phi_tot = d_j Phi + a_frame^j at cell (m,k,j,i).  idx1..3 are the inverse
//! cell widths.  The centred difference (Phi_{i+1} - Phi_{i-1})/(2 dx) is the mean of the
//! two face gradients and the discrete adjoint of the 7-point Laplacian the multigrid
//! solves, so sum_cells (L Phi) d Phi telescopes to zero: exact discrete momentum balance
//! in the Newtonian limit.  Needs one ghost layer of phi.

KOKKOS_INLINE_FUNCTION
void GradPhi(const DvceArray5D<Real> &phi, const int m, const int k, const int j,
             const int i, const Real idx1, const Real idx2, const Real idx3,
             const Real a_frame[3], Real dphi[3]) {
  dphi[0] = 0.5*idx1*(phi(m,0,k,j,i+1) - phi(m,0,k,j,i-1)) + a_frame[0];
  dphi[1] = 0.5*idx2*(phi(m,0,k,j+1,i) - phi(m,0,k,j-1,i)) + a_frame[1];
  dphi[2] = 0.5*idx3*(phi(m,0,k+1,j,i) - phi(m,0,k-1,j,i)) + a_frame[2];
}

//----------------------------------------------------------------------------------------
//! \fn Real FaceWorkRate
//! \brief The face form of -F^j d_j Phi_tot (Eq. 3a) at cell (m,k,j,i).  fe holds the
//! numerical fluxes of the energy E (rest mass included) through the cell's six faces,
//! {x1 at i, x1 at i+1, x2 at j, x2 at j+1, x3 at k, x3 at k+1}; each multiplies its own
//! one-sided gradient plus the reflex, half to this cell.  Returns the rate of E, which
//! the caller converts to its conserved energy.  The mean of the face gradients is
//! GradPhi's centred one.

KOKKOS_INLINE_FUNCTION
Real FaceWorkRate(const DvceArray5D<Real> &phi, const int m, const int k, const int j,
                  const int i, const Real idx1, const Real idx2, const Real idx3,
                  const Real a_frame[3], const Real fe[6]) {
  const Real p = phi(m,0,k,j,i);
  return -0.5*(fe[0]*(idx1*(p - phi(m,0,k,j,i-1)) + a_frame[0])
               + fe[1]*(idx1*(phi(m,0,k,j,i+1) - p) + a_frame[0])
               + fe[2]*(idx2*(p - phi(m,0,k,j-1,i)) + a_frame[1])
               + fe[3]*(idx2*(phi(m,0,k,j+1,i) - p) + a_frame[1])
               + fe[4]*(idx3*(p - phi(m,0,k-1,j,i)) + a_frame[2])
               + fe[5]*(idx3*(phi(m,0,k+1,j,i) - p) + a_frame[2]));
}

//----------------------------------------------------------------------------------------
//! \fn void MHDStress3p1
//! \brief Non-densitized Eulerian projections of the ideal-MHD stress-energy, its
//! rest-frame energy density and its active mass varrho = E + S (Eq. 1a).
//!
//! Inputs: H = rho h = rho(1+eps) + p, p, rho_eps = rho(1+eps), util = u~^i = W v^i,
//! Bdens = sqrt(gamma) B^i (the code's bcc), the spatial metric and its inverse.
//! The algebra is that of the dyn-GR coordinate-source kernel
//! (dyn_grmhd_coord_terms_impl.hpp), lowered with gamma_ij:
//!   W^2 = 1 + gamma_ij u~^i u~^j,  B.v = gamma_ij u~^i B^j / W,
//!   b^2 = B^2/W^2 + (B.v)^2,
//!   E    = H W^2 + B^2 - p - b^2/2,
//!   S_i  = (H W^2 + B^2) v_i - (B.v) B_i,
//!   S_ij = (H + B^2/W^2) u_i u_j - B_i B_j/W^2 - (B.v)(B_i u_j + B_j u_i)/W
//!          + (p + b^2/2) gamma_ij,
//!   epsbar = rho(1+eps) + b^2/2 (= H - p + b^2/2, formed without the cancellation),
//!   rho_eff = E + gamma^ij S_ij
//!           = (H + b^2)(2W^2 - 1) + 2p - 2(b.n)^2   (paper Eq. 19).
//! GR hydro passes Bdens = 0.  In force-free cells rho_eff -> E_EM^2 + B^2.

KOKKOS_INLINE_FUNCTION
void MHDStress3p1(const Real H, const Real p, const Real rho_eps, const Real util[3],
                  const Real Bdens[3], const Real g_dd[6], const Real g_uu[6],
                  const Real sqrt_gamma, Real &E, Real S_d[3], Real S_dd[6],
                  Real &epsbar, Real &rho_eff) {
  const Real isg = 1.0/sqrt_gamma;
  const Real B_u[3] = {Bdens[0]*isg, Bdens[1]*isg, Bdens[2]*isg};
  Real u_d[3], B_d[3];
  for (int a = 0; a < 3; ++a) {
    u_d[a] = 0.0;
    B_d[a] = 0.0;
    for (int b = 0; b < 3; ++b) {
      u_d[a] += g_dd[SymIdx(a,b)]*util[b];
      B_d[a] += g_dd[SymIdx(a,b)]*B_u[b];
    }
  }
  const Real usq = u_d[0]*util[0] + u_d[1]*util[1] + u_d[2]*util[2];
  const Real Wsq = 1.0 + usq;
  const Real W = sqrt(Wsq);
  const Real iW = 1.0/W;
  const Real Bsq = B_d[0]*B_u[0] + B_d[1]*B_u[1] + B_d[2]*B_u[2];
  const Real Bv = (u_d[0]*B_u[0] + u_d[1]*B_u[1] + u_d[2]*B_u[2])*iW;
  const Real bsq = Bv*Bv + Bsq/Wsq;
  const Real ptot = p + 0.5*bsq;

  E = H*Wsq + Bsq - ptot;
  const Real mom = (H*Wsq + Bsq)*iW;
  for (int a = 0; a < 3; ++a) {
    S_d[a] = mom*u_d[a] - Bv*B_d[a];
  }
  const Real Huu = H + Bsq/Wsq;
  for (int a = 0; a < 3; ++a) {
    for (int b = a; b < 3; ++b) {
      const int ab = SymIdx(a,b);
      S_dd[ab] = Huu*u_d[a]*u_d[b] - B_d[a]*B_d[b]/Wsq
                 - Bv*(B_d[a]*u_d[b] + B_d[b]*u_d[a])*iW + ptot*g_dd[ab];
    }
  }
  epsbar = rho_eps + 0.5*bsq;

  const Real S_trace = g_uu[0]*S_dd[0] + g_uu[3]*S_dd[3] + g_uu[5]*S_dd[5]
                       + 2.0*(g_uu[1]*S_dd[1] + g_uu[2]*S_dd[2] + g_uu[4]*S_dd[4]);
  rho_eff = E + S_trace;
}

//----------------------------------------------------------------------------------------
//! \fn void Source3p1
//! \brief Densitized Eulerian sources of Eq. (2) from Eq. (1).  vol = sqrt(gamma) for
//! non-densitized inputs; vol = 1 for densitized inputs (GR M1: E~, F~_i, P~_ij, Ebar~).
//! dphi = d_j Phi_tot from GradPhi.  s_tau sources sqrt(gamma) E (dyn-GR tau, M1 E~),
//! s_d sources sqrt(gamma) S_i (dyn-GR S_i, fixed-KS C_i, M1 F~_i).
//! Limits: no shift, static photon gas -> s_i = -alpha sqrt(gamma) (4/3) E d_i Phi;
//! a beam (P = E n n, epsbar = 0) -> s_i = -E n_i (n.dPhi), s_tau = -E (n.dPhi), exact
//! redshift with k_mu G^mu = 0.

KOKKOS_INLINE_FUNCTION
void Source3p1(const Real alpha, const Real beta_u[3], const Real g_uu[6],
               const Real vol, const Real E, const Real S_d[3], const Real S_dd[6],
               const Real epsbar, const Real dphi[3], Real &s_tau, Real s_d[3]) {
  Real dphi_u[3];
  for (int a = 0; a < 3; ++a) {
    dphi_u[a] = g_uu[SymIdx(a,0)]*dphi[0] + g_uu[SymIdx(a,1)]*dphi[1]
                + g_uu[SymIdx(a,2)]*dphi[2];
  }
  const Real beta_dphi = beta_u[0]*dphi[0] + beta_u[1]*dphi[1] + beta_u[2]*dphi[2];
  const Real S_dphi = S_d[0]*dphi_u[0] + S_d[1]*dphi_u[1] + S_d[2]*dphi_u[2];
  s_tau = vol*(-alpha*S_dphi + (E - epsbar)*beta_dphi);
  for (int a = 0; a < 3; ++a) {
    const Real Sdd_dphi = S_dd[SymIdx(a,0)]*dphi_u[0] + S_dd[SymIdx(a,1)]*dphi_u[1]
                          + S_dd[SymIdx(a,2)]*dphi_u[2];
    s_d[a] = vol*(-alpha*Sdd_dphi + S_d[a]*beta_dphi - alpha*epsbar*dphi[a]);
  }
}

//----------------------------------------------------------------------------------------
//! \fn Real LandauEnergyM1
//! \brief Rest-frame (Landau) radiation energy density Ebar of Eq. (5).  E = Eulerian
//! energy, F2 = gamma^ij F_i F_j, P_par = F^i F^j P_ij / F2.  For a closure
//! P = a gamma + b f^f^ (Minerbo, Levermore, Kershaw, Eddington, thin), F^ is an
//! eigenvector of P and the timelike eigenvalue of R^mu_nu solves a 2x2 problem in the
//! (n, F^) plane: lambda^2 - (E - P_par) lambda - (E P_par - F^2) = 0, root Ebar.
//! For Levermore it equals KORAL's Erf (rad.c:3114, = R^ab u_Ra u_Rb as in rad.c:3054;
//! boost beta: E - P_par = 2/3 Erf, root = 4/3 Erf).  Homogeneous of degree 1, so densitized inputs give densitized Ebar.
//! At F2 <= tiny*E^2 the root is E (and P_par is not formed by the caller); the radicand
//! is clamped at 0 against roundoff at f -> 1, where Ebar -> 0.

KOKKOS_INLINE_FUNCTION
Real LandauEnergyM1(const Real E, const Real F2, const Real P_par) {
  if (!(F2 > 1.0e-30*E*E)) return E;
  const Real disc = fmax((E + P_par)*(E + P_par) - 4.0*F2, 0.0);
  return 0.5*((E - P_par) + sqrt(disc));
}

}  // namespace grsg

#endif  // GRAVITY_GR_SELFGRAVITY_CORE_HPP_
