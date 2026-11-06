//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file gr_selfgravity_fluid.cpp
//! \brief Fixed Cartesian Kerr-Schild fluid adapters of the weak-field GR self-gravity
//! (gr_selfgravity_core.hpp has the physics): the active mass varrho = E + S of MHD
//! (ideal, variable gamma, two-temperature) and GR hydro, and their four-force
//! G_nu = -(T_nu^j + epsbar delta_nu^j) d_j Phi in the conserved IM, IEN.
//!
//! Both kernels read the stage primitives w0 (rho, u~^i = W v^i, e_int) and bcc0
//! (= sqrt(gamma) B^i), the same inputs as Coordinates::CoordSrcTerms, and the per-cell
//! adiabatic index through the same two-temperature path; the energy is the face work
//! of the stage's final numerical fluxes (Eq. 3a), here and, for the dynamical metric,
//! in AddFaceWorkDyn.  The Kerr-Schild 3+1 data come from the null form (f, l_i) of the
//! metric, g = eta + f l l with l_0 = 1:
//!   alpha^2 = 1/(1+f),  beta^i = f l_i/(1+f),  gamma_ij = delta_ij + f l_i l_j,
//!   gamma^ij = delta_ij - f l_i l_j/(1+f),  sqrt(gamma) = 1/alpha  (sqrt(-g) = 1),
//! so no derivative of the metric and no 4x4 array is formed and the unit stays light.

#include <math.h>

#include "athena.hpp"
#include "globals.hpp"
#include "mesh/mesh.hpp"
#include "eos/eos.hpp"
#include "hydro/hydro.hpp"
#include "mhd/mhd.hpp"
#include "coordinates/cartesian_ks.hpp"
#include "coordinates/cell_locations.hpp"
#include "coordinates/coordinates.hpp"
#include "utils/gravity_weight.hpp"
#include "gravity/gr_selfgravity_core.hpp"
#include "gravity/gr_selfgravity_fluid.hpp"

namespace grsg {
namespace {

//! The 3+1 data and the stress of one cell, in the arrays the core header takes.
struct CellStress {
  Real alpha, vol;
  Real beta_u[3];
  Real g_dd[6], g_uu[6];
  Real E, S_d[3], S_dd[6], epsbar, rho_eff;
};

//! The Kerr-Schild 3+1 data at (x,y,z) and the Eulerian stress of the cell's gas.
KOKKOS_INLINE_FUNCTION
void FixedCellStress(const KerrHole &hole, const Real x, const Real y, const Real z,
                     const Real rho, const Real util[3], const Real eint,
                     const Real gamma_gas, const Real bdens[3], CellStress &c) {
  KSNullForm nf;
  ComputeKSNullForm(x, y, z, hole, nf);
  const Real l[3] = {nf.l1, nf.l2, nf.l3};
  const Real fa2 = KSGammaUpperFactor(nf);   // f/(1+f) = alpha^2 f
  c.alpha = KSAlpha(nf);
  c.vol = 1.0/c.alpha;
  for (int a = 0; a < 3; ++a) {
    c.beta_u[a] = fa2*l[a];
    for (int b = a; b < 3; ++b) {
      const Real delta = (a == b) ? 1.0 : 0.0;
      c.g_dd[SymIdx(a,b)] = delta + nf.f*l[a]*l[b];
      c.g_uu[SymIdx(a,b)] = delta - fa2*l[a]*l[b];
    }
  }
  const Real pgas = (gamma_gas - 1.0)*eint;
  MHDStress3p1(rho + gamma_gas*eint, pgas, rho + eint, util, bdens, c.g_dd, c.g_uu,
               c.vol, c.E, c.S_d, c.S_dd, c.epsbar, c.rho_eff);
}

}  // namespace

//----------------------------------------------------------------------------------------
//! \fn void AddGravitatingDensityFixed

void AddGravitatingDensityFixed(MeshBlockPack *pack, DvceArray5D<Real> &rho_eff) {
  const bool mhd = (pack->pmhd != nullptr);
  const auto w0 = mhd ? pack->pmhd->w0 : pack->phydro->w0;
  const DvceArray5D<Real> bcc = mhd ? pack->pmhd->bcc0 : DvceArray5D<Real>();
  const EOS_Data eos = mhd ? pack->pmhd->peos->eos_data : pack->phydro->peos->eos_data;

  const auto &indcs = pack->pmesh->mb_indcs;
  const int is = indcs.is, ie = indcs.ie;
  const int js = indcs.js, je = indcs.je;
  const int ks = indcs.ks, ke = indcs.ke;
  const auto &size = pack->pmb->mb_size;
  const KerrHole hole = pack->pcoord->coord_data.Hole();
  const bool excise = pack->pcoord->coord_data.bh_excise;
  const auto floor = pack->pcoord->excision_floor;
  const Real gamma_eos = eos.gamma;
  const int nx1 = indcs.nx1, nx2 = indcs.nx2, nx3 = indcs.nx3;
  const int nmb = pack->nmb_thispack;

  par_for("grsg_varrho_fixed", DevExeSpace(), 0, nmb-1, ks, ke, js, je, is, ie,
  KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
    if (excise && floor(m,k,j,i)) return;
    const Real x = CellCenterX(i-is, nx1, size.d_view(m).x1min, size.d_view(m).x1max);
    const Real y = CellCenterX(j-js, nx2, size.d_view(m).x2min, size.d_view(m).x2max);
    const Real z = CellCenterX(k-ks, nx3, size.d_view(m).x3min, size.d_view(m).x3max);
    const Real util[3] = {w0(m,IVX,k,j,i), w0(m,IVY,k,j,i), w0(m,IVZ,k,j,i)};
    Real bdens[3] = {0.0, 0.0, 0.0};
    if (mhd) {
      bdens[0] = bcc(m,IBX,k,j,i);
      bdens[1] = bcc(m,IBY,k,j,i);
      bdens[2] = bcc(m,IBZ,k,j,i);
    }
    const Real gamma_gas = gamma_eos;
    CellStress c;
    FixedCellStress(hole, x, y, z, w0(m,IDN,k,j,i), util, w0(m,IEN,k,j,i),
                    gamma_gas, bdens, c);
    rho_eff(m,0,k,j,i) += c.rho_eff;
  });
}

//----------------------------------------------------------------------------------------
//! \fn void AddFluidForceFixed

void AddFluidForceFixed(MeshBlockPack *pack, const PhiView &pv, const Real dt,
                        const BandFaceFld5D<Real> &flux, DvceArray5D<Real> &u0) {
  if (!pv.active) return;
  const bool mhd = (pack->pmhd != nullptr);
  const auto w0 = mhd ? pack->pmhd->w0 : pack->phydro->w0;
  const DvceArray5D<Real> bcc = mhd ? pack->pmhd->bcc0 : DvceArray5D<Real>();
  const EOS_Data eos = mhd ? pack->pmhd->peos->eos_data : pack->phydro->peos->eos_data;

  const auto &indcs = pack->pmesh->mb_indcs;
  const int is = indcs.is, ie = indcs.ie;
  const int js = indcs.js, je = indcs.je;
  const int ks = indcs.ks, ke = indcs.ke;
  const auto &size = pack->pmb->mb_size;
  const KerrHole hole = pack->pcoord->coord_data.Hole();
  const bool excise = pack->pcoord->coord_data.bh_excise;
  const auto floor = pack->pcoord->excision_floor;
  const Real gamma_eos = eos.gamma;
  const int nx1 = indcs.nx1, nx2 = indcs.nx2, nx3 = indcs.nx3;
  const PhiView phi_view = pv;
  const auto f1 = flux.x1f, f2 = flux.x2f, f3 = flux.x3f;

  // Per-block step exactly as CoordSrcTerms: the LAT per-block dt over the active list.
  const bool lat_enabled = pack->lat_active_mask_enabled;
  const bool lat_per_block_dt = pack->lat_per_block_timestep;
  const Real mesh_dt = pack->pmesh->dt;
  const Real dt_scale = (lat_per_block_dt && mesh_dt != 0.0) ? dt/mesh_dt : 0.0;
  auto active_indices = pack->lat_active_indices.d_view;
  auto lat_step_dt = pack->lat_step_dt.d_view;
  const int nwork1 = lat_enabled ? (pack->lat_nactive_thispack - 1) :
                                   (pack->nmb_thispack - 1);
  if (nwork1 < 0) return;

  par_for("grsg_force_fixed", DevExeSpace(), 0, nwork1, ks, ke, js, je, is, ie,
  KOKKOS_LAMBDA(const int a, const int k, const int j, const int i) {
    const int m = lat_enabled ? active_indices(a) : a;
    if (excise && floor(m,k,j,i)) return;
    const Real rho = w0(m,IDN,k,j,i);
    const Real weight = gravity_weight::Weight(rho, phi_view.dfloor, phi_view.rho_min);
    if (!(weight > 0.0)) return;
    const Real block_dt = lat_per_block_dt ? dt_scale*lat_step_dt(m) : dt;

    const Real x = CellCenterX(i-is, nx1, size.d_view(m).x1min, size.d_view(m).x1max);
    const Real y = CellCenterX(j-js, nx2, size.d_view(m).x2min, size.d_view(m).x2max);
    const Real z = CellCenterX(k-ks, nx3, size.d_view(m).x3min, size.d_view(m).x3max);
    const Real util[3] = {w0(m,IVX,k,j,i), w0(m,IVY,k,j,i), w0(m,IVZ,k,j,i)};
    Real bdens[3] = {0.0, 0.0, 0.0};
    if (mhd) {
      bdens[0] = bcc(m,IBX,k,j,i);
      bdens[1] = bcc(m,IBY,k,j,i);
      bdens[2] = bcc(m,IBZ,k,j,i);
    }
    const Real gamma_gas = gamma_eos;
    CellStress c;
    FixedCellStress(hole, x, y, z, rho, util, w0(m,IEN,k,j,i), gamma_gas, bdens,
                    c);

    const Real idx1 = 1.0/size.d_view(m).dx1;
    const Real idx2 = 1.0/size.d_view(m).dx2;
    const Real idx3 = 1.0/size.d_view(m).dx3;
    Real dphi[3];
    GradPhi(phi_view.phi, m, k, j, i, idx1, idx2, idx3, phi_view.a_frame, dphi);
    for (int n = 0; n < 3; ++n) dphi[n] *= weight;

    // Momentum: dC_i = s_i of Eq. (2), pointwise.
    Real s_tau, s_d[3];
    Source3p1(c.alpha, c.beta_u, c.g_uu, c.vol, c.E, c.S_d, c.S_dd, c.epsbar, dphi,
              s_tau, s_d);
    u0(m,IM1,k,j,i) += block_dt*s_d[0];
    u0(m,IM2,k,j,i) += block_dt*s_d[1];
    u0(m,IM3,k,j,i) += block_dt*s_d[2];

    // Energy: dC_t = sqrt(-g) G_t = -F^j[C_t] d_j Phi as the face work of Eq. (3a).  The
    // energy E = -C_t has the flux F_IDN - F_IEN (IEN = C_t + D).
    const Real fe[6] = {f1(m,IDN,k,j,i) - f1(m,IEN,k,j,i),
                        f1(m,IDN,k,j,i+1) - f1(m,IEN,k,j,i+1),
                        f2(m,IDN,k,j,i) - f2(m,IEN,k,j,i),
                        f2(m,IDN,k,j+1,i) - f2(m,IEN,k,j+1,i),
                        f3(m,IDN,k,j,i) - f3(m,IEN,k,j,i),
                        f3(m,IDN,k+1,j,i) - f3(m,IEN,k+1,j,i)};
    u0(m,IEN,k,j,i) -= block_dt*weight*FaceWorkRate(phi_view.phi, m, k, j, i, idx1, idx2,
                                                    idx3, phi_view.a_frame, fe);
  });
}

//----------------------------------------------------------------------------------------
//! \fn void AddFaceWorkDyn

void AddFaceWorkDyn(MeshBlockPack *pack, const PhiView &pv, const Real dt,
                    const BandFaceFld5D<Real> &flux, DvceArray5D<Real> &u0) {
  if (!pv.active) return;
  const auto w0 = pack->pmhd->w0;
  const auto &indcs = pack->pmesh->mb_indcs;
  const int is = indcs.is, ie = indcs.ie;
  const int js = indcs.js, je = indcs.je;
  const int ks = indcs.ks, ke = indcs.ke;
  const auto &size = pack->pmb->mb_size;
  const bool excise = pack->pcoord->coord_data.bh_excise;
  const auto floor = pack->pcoord->excision_floor;
  const PhiView phi_view = pv;
  const auto f1 = flux.x1f, f2 = flux.x2f, f3 = flux.x3f;

  // The per-block step of AddCoordTerms: the LAT per-block dt over the active list.
  const bool lat_enabled = pack->lat_active_mask_enabled;
  const bool lat_per_block_dt = pack->lat_per_block_timestep;
  const Real mesh_dt = pack->pmesh->dt;
  const Real dt_scale = (lat_per_block_dt && mesh_dt != 0.0) ? dt/mesh_dt : 0.0;
  auto active_indices = pack->lat_active_indices.d_view;
  auto lat_step_dt = pack->lat_step_dt.d_view;
  const int nwork1 = lat_enabled ? (pack->lat_nactive_thispack - 1) :
                                   (pack->nmb_thispack - 1);
  if (nwork1 < 0) return;

  par_for("grsg_face_work_dyn", DevExeSpace(), 0, nwork1, ks, ke, js, je, is, ie,
  KOKKOS_LAMBDA(const int a, const int k, const int j, const int i) {
    const int m = lat_enabled ? active_indices(a) : a;
    if (excise && floor(m,k,j,i)) return;
    const Real weight =
        gravity_weight::Weight(w0(m,IDN,k,j,i), phi_view.dfloor, phi_view.rho_min);
    if (!(weight > 0.0)) return;
    const Real block_dt = lat_per_block_dt ? dt_scale*lat_step_dt(m) : dt;
    // tau + D = E (rest mass included) has the flux F_IEN + F_IDN.
    const Real fe[6] = {f1(m,IEN,k,j,i) + f1(m,IDN,k,j,i),
                        f1(m,IEN,k,j,i+1) + f1(m,IDN,k,j,i+1),
                        f2(m,IEN,k,j,i) + f2(m,IDN,k,j,i),
                        f2(m,IEN,k,j+1,i) + f2(m,IDN,k,j+1,i),
                        f3(m,IEN,k,j,i) + f3(m,IDN,k,j,i),
                        f3(m,IEN,k+1,j,i) + f3(m,IDN,k+1,j,i)};
    u0(m,IEN,k,j,i) += block_dt*weight*FaceWorkRate(
        phi_view.phi, m, k, j, i, 1.0/size.d_view(m).dx1, 1.0/size.d_view(m).dx2,
        1.0/size.d_view(m).dx3, phi_view.a_frame, fe);
  });
}

}  // namespace grsg
