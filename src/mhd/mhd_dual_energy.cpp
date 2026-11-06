//========================================================================================
// AthenaK astrophysical fluid dynamics and numerical relativity code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file mhd_dual_energy.cpp
//! \brief Dual-energy formalism for Newtonian MHD with EOS-aware thermal support

#include <algorithm>
#include <cmath>

#include "athena.hpp"
#include "driver/driver.hpp"
#include "mesh/mesh.hpp"
#include "coordinates/cell_locations.hpp"
#include "eos/eos.hpp"
#include "eos/dual_energy_shock_rule.hpp"
#include "eos/general_c2p_hyd.hpp"
#include "eos/general_c2p_mhd.hpp"
#include "mhd.hpp"
#include "pgen/pgen.hpp"

namespace {

KOKKOS_INLINE_FUNCTION
Real BccX(const DvceFaceFld4D<Real> &b, const int m, const int k, const int j,
    const int i) {
  return 0.5*(b.x1f(m,k,j,i) + b.x1f(m,k,j,i+1));
}

KOKKOS_INLINE_FUNCTION
Real BccY(const DvceFaceFld4D<Real> &b, const int m, const int k, const int j,
    const int i) {
  return 0.5*(b.x2f(m,k,j,i) + b.x2f(m,k,j+1,i));
}

KOKKOS_INLINE_FUNCTION
Real BccZ(const DvceFaceFld4D<Real> &b, const int m, const int k, const int j,
    const int i) {
  return 0.5*(b.x3f(m,k,j,i) + b.x3f(m,k+1,j,i));
}

//----------------------------------------------------------------------------------------
//! \brief CornerE's EMF on an edge: efld less the EMFs other modules added after it.

KOKKOS_INLINE_FUNCTION
Real CornerEMF(const DvceArray4D<Real> &e, const DvceArray4D<Real> &a,
               const bool with_added, const int m, const int k, const int j,
               const int i) {
  return with_added ? (e(m, k, j, i) - a(m, k, j, i)) : e(m, k, j, i);
}

//----------------------------------------------------------------------------------------
//! \brief The cell-centred field of cell (m,k,j,i) after this stage as the fluxes of E
//! transported it.  E's Poynting flux through a face is the Riemann solver's, consistent
//! with that face's EMFs (e3x1 .. e2x3), whereas CT moves the face fields with the edge
//! EMFs (CornerE).  The two cell fields are both gam0 bcc0 + gam1 <b1> plus the curl of
//! their EMFs and differ by the curl of the face EMFs against the face-averaged curl of
//! the edge EMFs, first order in the field change; their energies differ by B.dB, the
//! first-order magnetic error of E - E_K - |B_CT|^2/2.  So the field E carries is the CT
//! field after the update plus that difference, with CT's signs and stage weight
//! (mhd_ct.cpp) and the face EMFs as the FOFC trial reads them (mhd_fofc.cpp).  In 1D
//! CornerE copies the face EMFs to the edges, so the two are the same field.  The edge
//! EMFs other modules add after CornerE (resistivity, the 2D shearing box; a1..a3 when
//! with_added) have no face counterpart in E's Riemann flux: they move B through CT and E
//! through their own energy terms, as in the conservative scheme, so the difference is
//! taken against CornerE's edge EMFs alone and the field keeps their curl.

KOKKOS_INLINE_FUNCTION
void MHDFieldCarriedByE(const DvceFaceFld4D<Real> &b, const DvceArray4D<Real> &e1,
                        const DvceArray4D<Real> &e2, const DvceArray4D<Real> &e3,
                        const DvceArray4D<Real> &a1, const DvceArray4D<Real> &a2,
                        const DvceArray4D<Real> &a3, const bool with_added,
                        const BandView4D<Real> &e2x1, const BandView4D<Real> &e3x1,
                        const BandView4D<Real> &e1x2, const BandView4D<Real> &e3x2,
                        const BandView4D<Real> &e1x3, const BandView4D<Real> &e2x3,
                        const int m, const int k, const int j, const int i,
                        const Real dt, const Real dx1, const Real dx2, const Real dx3,
                        const bool multi_d, const bool three_d,
                        Real &bx, Real &by, Real &bz) {
  bx = BccX(b, m, k, j, i);
  by = BccY(b, m, k, j, i);
  bz = BccZ(b, m, k, j, i);
  if (!multi_d) return;
  // the stage's EMF update of each component: the face EMFs at the cell (finite volume),
  // and CornerE's edge EMFs on the cell's two faces normal to it (CT)
  const bool wa = with_added;
  Real fv_x = -(dt*(e3x2(m, k, j + 1, i) - e3x2(m, k, j, i))/dx2);
  Real ct_x0 = -(dt*(CornerEMF(e3, a3, wa, m, k, j + 1, i) -
                     CornerEMF(e3, a3, wa, m, k, j, i))/dx2);
  Real ct_x1 = -(dt*(CornerEMF(e3, a3, wa, m, k, j + 1, i + 1) -
                     CornerEMF(e3, a3, wa, m, k, j, i + 1))/dx2);
  Real fv_y = dt*(e3x1(m, k, j, i + 1) - e3x1(m, k, j, i))/dx1;
  Real ct_y0 = dt*(CornerEMF(e3, a3, wa, m, k, j, i + 1) -
                   CornerEMF(e3, a3, wa, m, k, j, i))/dx1;
  Real ct_y1 = dt*(CornerEMF(e3, a3, wa, m, k, j + 1, i + 1) -
                   CornerEMF(e3, a3, wa, m, k, j + 1, i))/dx1;
  Real fv_z = -(dt*(e2x1(m, k, j, i + 1) - e2x1(m, k, j, i))/dx1);
  Real ct_z0 = -(dt*(CornerEMF(e2, a2, wa, m, k, j, i + 1) -
                     CornerEMF(e2, a2, wa, m, k, j, i))/dx1);
  Real ct_z1 = -(dt*(CornerEMF(e2, a2, wa, m, k + 1, j, i + 1) -
                     CornerEMF(e2, a2, wa, m, k + 1, j, i))/dx1);
  fv_z += dt*(e1x2(m, k, j + 1, i) - e1x2(m, k, j, i))/dx2;
  ct_z0 += dt*(CornerEMF(e1, a1, wa, m, k, j + 1, i) -
               CornerEMF(e1, a1, wa, m, k, j, i))/dx2;
  ct_z1 += dt*(CornerEMF(e1, a1, wa, m, k + 1, j + 1, i) -
               CornerEMF(e1, a1, wa, m, k + 1, j, i))/dx2;
  if (three_d) {
    fv_x += dt*(e2x3(m, k + 1, j, i) - e2x3(m, k, j, i))/dx3;
    ct_x0 += dt*(CornerEMF(e2, a2, wa, m, k + 1, j, i) -
                 CornerEMF(e2, a2, wa, m, k, j, i))/dx3;
    ct_x1 += dt*(CornerEMF(e2, a2, wa, m, k + 1, j, i + 1) -
                 CornerEMF(e2, a2, wa, m, k, j, i + 1))/dx3;
    fv_y -= dt*(e1x3(m, k + 1, j, i) - e1x3(m, k, j, i))/dx3;
    ct_y0 -= dt*(CornerEMF(e1, a1, wa, m, k + 1, j, i) -
                 CornerEMF(e1, a1, wa, m, k, j, i))/dx3;
    ct_y1 -= dt*(CornerEMF(e1, a1, wa, m, k + 1, j + 1, i) -
                 CornerEMF(e1, a1, wa, m, k, j + 1, i))/dx3;
  }
  bx += fv_x - 0.5*(ct_x0 + ct_x1);
  by += fv_y - 0.5*(ct_y0 + ct_y1);
  bz += fv_z - 0.5*(ct_z0 + ct_z1);
}

}  // namespace

namespace mhd {

TaskStatus MHD::DualEnergyStep(Driver *pdrive, int stage) {
  if (!use_dual_energy) return TaskStatus::complete;
  // As Hydro::DualEnergyStep: the components of the shock rule's offsets this integrator
  // needs (dual_shock_debt, mhd.hpp).
  const bool two_register = (pdrive->integrator == "rk4");
  if (stage == 1) {
    const int ncomp = two_register ? 4 : ((pdrive->nexp_stages >= 3) ? 2 : 1);
    if (dual_shock_debt.extent_int(1) != ncomp) {
      Kokkos::realloc(dual_shock_debt, dual_shock_debt.extent_int(0), ncomp,
                      dual_shock_debt.extent_int(2), dual_shock_debt.extent_int(3),
                      dual_shock_debt.extent_int(4));
    }
  }
  const Real beta_dt = (pdrive->beta[stage-1])*(pmy_pack->pmesh->dt);
  ApplyDualEnergyFormalism(beta_dt, stage, pdrive->nexp_stages, pdrive->gam0[stage-1],
                           pdrive->gam1[stage-1],
                           two_register ? pdrive->delta[stage-1] : 0.0);
  return TaskStatus::complete;
}

void MHD::InitializeDualEnergyFieldFromTotal() {
  if (!use_dual_energy) return;

  auto &indcs = pmy_pack->pmesh->mb_indcs;
  const int is = indcs.is;
  const int ie = indcs.ie;
  const int js = indcs.js;
  const int je = indcs.je;
  const int ks = indcs.ks;
  const int ke = indcs.ke;
  const int nmb1 = pmy_pack->nmb_thispack - 1;
  auto u0_ = u0;
  auto w0_ = w0;
  auto b0_ = b0;
  const int de_idx = dual_energy_idx;
  auto &eos = peos->eos_data;

  // Seed the auxiliary internal-energy field from the full conserved state, including
  // magnetic energy, so restarts and fresh starts begin synchronized.
  par_for("mhd_dual_energy_init", DevExeSpace(), 0, nmb1, ks, ke, js, je, is, ie,
  KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
    const Real dens = fmax(u0_(m, IDN, k, j, i), eos.dfloor);
    const Real bx = BccX(b0_, m, k, j, i);
    const Real by = BccY(b0_, m, k, j, i);
    const Real bz = BccZ(b0_, m, k, j, i);
    const Real e_k = 0.5*(SQR(u0_(m, IM1, k, j, i)) + SQR(u0_(m, IM2, k, j, i)) +
                          SQR(u0_(m, IM3, k, j, i)))/dens;
    const Real e_m = 0.5*(SQR(bx) + SQR(by) + SQR(bz));
    bool efloor_used = false, tfloor_used = false;
    Real eint = u0_(m, IEN, k, j, i) - e_k - e_m;
    eint = eos_general::ApplyMHDThermalFloors(eos, dens, eint, efloor_used,
                                              tfloor_used);
    u0_(m, de_idx, k, j, i) = eint;
    w0_(m, de_idx, k, j, i) = eint;
  });
}

void MHD::ApplyDualEnergyFormalism(const Real dt, const int stage, const int nstages,
                                   const Real gam0, const Real gam1, const Real delta) {
  if (!dual_energy_pdv) return;

  auto &indcs = pmy_pack->pmesh->mb_indcs;
  const int is = indcs.is;
  const int ie = indcs.ie;
  const int js = indcs.js;
  const int je = indcs.je;
  const int ks = indcs.ks;
  const int ke = indcs.ke;
  const int nmb1 = pmy_pack->nmb_thispack - 1;
  const bool multi_d = pmy_pack->pmesh->multi_d;
  const bool three_d = pmy_pack->pmesh->three_d;
  const int ndim = three_d ? 3 : (multi_d ? 2 : 1);
  auto u0_ = u0;
  auto w0_ = w0;
  auto b0_ = b0;
  auto e1_ = efld.x1e;
  auto e2_ = efld.x2e;
  auto e3_ = efld.x3e;
  // the edge EMFs other modules added to efld after CornerE (MHD::EField)
  const bool with_added =
      (presist != nullptr) || ((psbox_b != nullptr) && pmy_pack->pmesh->two_d);
  auto a1_ = dual_efld_added.x1e;
  auto a2_ = dual_efld_added.x2e;
  auto a3_ = dual_efld_added.x3e;
  auto e2x1_ = EmfBand(e2x1);
  auto e3x1_ = EmfBand(e3x1);
  auto e1x2_ = EmfBand(e1x2);
  auto e3x2_ = EmfBand(e3x2);
  auto e1x3_ = EmfBand(e1x3);
  auto e2x3_ = EmfBand(e2x3);
  auto vf1_ = FluxBand(dual_vf.x1f);
  auto vf2_ = FluxBand(dual_vf.x2f);
  auto vf3_ = FluxBand(dual_vf.x3f);
  auto &mbsize = pmy_pack->pmb->mb_size;
  auto &eos = peos->eos_data;
  const int de_idx = dual_energy_idx;
  const bool lat_enabled = pmy_pack->lat_active_mask_enabled;
  const bool lat_per_block_dt = pmy_pack->lat_per_block_timestep;
  const Real mesh_dt = pmy_pack->pmesh->dt;
  const Real stage_weight =
      (lat_per_block_dt && mesh_dt > 0.0) ? (dt/mesh_dt) : 0.0;
  auto lat_step_dt = pmy_pack->lat_step_dt.d_view;
  auto active_indices = pmy_pack->lat_active_indices.d_view;
  const int nwork1 = lat_enabled ? (pmy_pack->lat_nactive_thispack - 1) : nmb1;
  if (nwork1 < 0) return;
  const int shock_reach = eos_general::DualEnergyShockReach(indcs.ng);
  const bool decide_shock = (stage == 1);
  const bool step_final = (stage >= nstages);
  const bool two_register = (dual_shock_debt.extent_int(1) == 4);
  auto shocked_ = dual_shocked;
  auto debt_ = dual_shock_debt;
  const eos_general::DualEnergyCellField field{bcc0};
  // Hydro::ApplyDualEnergyFormalism with the field: evolve the auxiliary internal energy
  // on its adiabat and with the heat of its faces' fans, and give it the energy channel
  // where the cell is being shocked (the shock rule, eos/dual_energy_shock_rule.hpp; the
  // shock test reads w0 and bcc0 shock_reach + 2 = nghost cells off).  The test runs at
  // the step's first stage on the step-start primitives w0 and field bcc0; dual_shocked
  // holds its flags and dual_shock_debt the offsets of the step-final guard for the later
  // stages.  Everywhere else the conservative-to-auxiliary synchronization follows later
  // in the stage, cell by cell on the owner: MHD::ResolveDualEnergyChannel (RestrictU,
  // before the state leaves the block) and the owned-cell inversion.  Neither reads a
  // neighbouring cell.
  par_for("mhd_dual_energy_formalism", DevExeSpace(), 0, nwork1, ks, ke, js, je, is, ie,
  KOKKOS_LAMBDA(const int a, const int k, const int j, const int i) {
    const int m = lat_enabled ? active_indices(a) : a;
    const Real block_dt = lat_per_block_dt ? stage_weight*lat_step_dt(m) : dt;
    // The cell's volume change and the heat its faces' Riemann fans leave in it
    // (eos/dual_energy_shock_rule.hpp).
    Real divv, heat;
    eos_general::DualEnergyCellPdV(vf1_, vf2_, vf3_, m, k, j, i, mbsize.d_view(m).dx1,
                                   mbsize.d_view(m).dx2, mbsize.d_view(m).dx3, multi_d,
                                   three_d, divv, heat);

    const Real dens = fmax(u0_(m, IDN, k, j, i), eos.dfloor);
    Real eint_floor, eint_ceil, log_rho_cgs;
    eos.HydroInternalEnergyDensityBounds(dens, eint_floor, eint_ceil, log_rho_cgs);
    const Real eint_advected = u0_(m, de_idx, k, j, i);
    const Real divv_dt = divv*block_dt;
    Real eint = eos_general::DualEnergyAuxiliaryPdV(
        eos, dens, eint_advected, divv_dt, heat*block_dt, eint_floor, eint_ceil,
        log_rho_cgs);

    // The shock rule, decided once per step on the step-start state, with the
    // step-final guard, as hydro's.
    const int kb = k - ks, jb = j - js, ib = i - is;
    bool shocked;
    if (decide_shock) {
      shocked = eos_general::DualEnergyCellShocked(eos, w0_, field, m, k, j, i, ndim,
                                                   shock_reach, dens, eint);
      shocked_(m, kb, jb, ib) = shocked;
    } else {
      shocked = shocked_(m, kb, jb, ib);
    }
    if (shocked) {
      // The stage's energy channel against the field E's fluxes carried, so it holds no
      // part of the first-order mismatch between that field and the CT field
      // (MHDFieldCarriedByE).
      Real bx, by, bz;
      MHDFieldCarriedByE(b0_, e1_, e2_, e3_, a1_, a2_, a3_, with_added, e2x1_, e3x1_,
                         e1x2_, e3x2_, e1x3_, e2x3_, m, k, j, i, block_dt,
                         mbsize.d_view(m).dx1, mbsize.d_view(m).dx2,
                         mbsize.d_view(m).dx3, multi_d, three_d, bx, by, bz);
      const Real e_k = 0.5*(SQR(u0_(m, IM1, k, j, i)) + SQR(u0_(m, IM2, k, j, i)) +
                            SQR(u0_(m, IM3, k, j, i)))/dens;
      const Real e_m = 0.5*(SQR(bx) + SQR(by) + SQR(bz));
      const Real eint_cons = u0_(m, IEN, k, j, i) - e_k - e_m;
      eint = eos_general::DualEnergyShockedStageEint(
          eos, debt_, m, kb, jb, ib, stage, step_final, two_register, gam0, gam1, delta,
          dens, eint, eint_advected, divv_dt, eint_floor, eint_ceil, log_rho_cgs,
          eint_cons, u0_(m, IEN, k, j, i) + e_k + e_m);
    }

    u0_(m, de_idx, k, j, i) = eint;
    w0_(m, de_idx, k, j, i) = eint;
  });
}

//----------------------------------------------------------------------------------------
//! \fn void MHD::ResolveDualEnergyChannel()
//! \brief The owner's thermal decision, made on its own cells before their state leaves
//! the block (MHD::RestrictU, ahead of the restriction and the send), as
//! Hydro::ResolveDualEnergyChannel makes it.  Where the inversion accepts the energy
//! channel (eos_general::MHDEnergyChannelResolvesEint, the eta1 test and the magnetic
//! test of the energy split) the auxiliary takes E - E_K - E_M of the CT field; elsewhere
//! E becomes eint_aux + E_K + E_M.  Every state a block hands on then satisfies
//! E - E_K - E_M = eint_aux, so a state assembled from several of them -- a restriction,
//! a LAT time interpolation or extrapolation, a prolongation -- carries in its auxiliary
//! the same combination of the owners' thermal energies, and ghost zones and coarse
//! buffers read the auxiliary (eos_general::SingleC2P_GeneralMHDDual): the kinetic and
//! magnetic energy of a combined state differ from the combined energies by the
//! dispersion of the parts, which is not heat.  The owner's recovery repeats this
//! decision for whatever changed E after the send.

void MHD::ResolveDualEnergyChannel() {
  if (!dual_energy_pdv) return;

  auto &indcs = pmy_pack->pmesh->mb_indcs;
  const int is = indcs.is, ie = indcs.ie;
  const int js = indcs.js, je = indcs.je;
  const int ks = indcs.ks, ke = indcs.ke;
  const int nx1 = indcs.nx1, nx2 = indcs.nx2, nx3 = indcs.nx3;
  const int nmb1 = pmy_pack->nmb_thispack - 1;
  const bool lat_enabled = pmy_pack->lat_active_mask_enabled;
  auto active_indices = pmy_pack->lat_active_indices.d_view;
  const int nwork1 = lat_enabled ? (pmy_pack->lat_nactive_thispack - 1) : nmb1;
  if (nwork1 < 0) return;
  auto u0_ = u0;
  auto b0_ = b0;
  auto &mbsize = pmy_pack->pmb->mb_size;
  auto &eos = peos->eos_data;
  const int de_idx = dual_energy_idx;
  const Real eta1 = dual_energy_eta1;
  bool excise_enabled = false;
  Real excise_radius = 0.0, excise_density = 0.0, excise_eint = 0.0;
  Real sink_x = 0.0, sink_y = 0.0, sink_z = 0.0;
  problem_runtime::GetExcisionState(pmy_pack->pmesh->time, excise_enabled,
      excise_radius, excise_density, excise_eint, sink_x, sink_y, sink_z);
  const Real excise_r2 = excise_radius*excise_radius;

  par_for("mhd_dual_energy_resolve", DevExeSpace(), 0, nwork1, ks, ke, js, je, is, ie,
  KOKKOS_LAMBDA(const int a, const int k, const int j, const int i) {
    const int m = lat_enabled ? active_indices(a) : a;
    MHDCons1D u;
    u.d = u0_(m, IDN, k, j, i);
    u.mx = u0_(m, IM1, k, j, i);
    u.my = u0_(m, IM2, k, j, i);
    u.mz = u0_(m, IM3, k, j, i);
    u.e = u0_(m, IEN, k, j, i);
    // the inversion resets the atmosphere and the excision interior itself
    if (eos_general::NeedsMHDAtmosphereReset(eos, u)) return;
    if (excise_enabled) {
      const Real x = CellCenterX(i - is, nx1, mbsize.d_view(m).x1min,
                                 mbsize.d_view(m).x1max);
      const Real y = CellCenterX(j - js, nx2, mbsize.d_view(m).x2min,
                                 mbsize.d_view(m).x2max);
      const Real z = CellCenterX(k - ks, nx3, mbsize.d_view(m).x3min,
                                 mbsize.d_view(m).x3max);
      if (problem_runtime::InsideExcisionZone(x, y, z, sink_x, sink_y, sink_z,
                                              excise_r2)) return;
    }
    const Real e_k = 0.5*(SQR(u.mx) + SQR(u.my) + SQR(u.mz))/u.d;
    const Real e_m = 0.5*(SQR(BccX(b0_, m, k, j, i)) + SQR(BccY(b0_, m, k, j, i)) +
                          SQR(BccZ(b0_, m, k, j, i)));
    const Real eint_cons = u.e - e_k - e_m;
    if (eos_general::MHDEnergyChannelResolvesEint(eint_cons, u.e, e_m, eta1)) {
      u0_(m, de_idx, k, j, i) = eint_cons;
      return;
    }
    bool efloor_used = false, tfloor_used = false;
    const Real eint_aux = eos_general::ApplyHydroThermalFloors(
        eos, u.d, u0_(m, de_idx, k, j, i), efloor_used, tfloor_used);
    u0_(m, IEN, k, j, i) = eint_aux + e_k + e_m;
    u0_(m, de_idx, k, j, i) = eint_aux;
  });
}

void MHD::SynchronizeRestrictedDualEnergyField(bool lat_active_only) {
  SynchronizeRestrictedDualEnergyField(coarse_u0, coarse_w0, lat_active_only);
}

void MHD::SynchronizeRestrictedDualEnergyField(
    DvceArray5D<Real> &coarse_cons, DvceArray5D<Real> &coarse_prim,
    bool lat_active_only) {
  if (!use_dual_energy) return;

  auto &indcs = pmy_pack->pmesh->mb_indcs;
  const int is = indcs.cis;
  const int ie = indcs.cie;
  const int js = indcs.cjs;
  const int je = indcs.cje;
  const int ks = indcs.cks;
  const int ke = indcs.cke;
  const int nmb1 = pmy_pack->nmb_thispack - 1;
  const bool lat_enabled = pmy_pack->lat_active_mask_enabled;
  auto lat_boundary_send_indices = lat_active_only ?
      pmy_pack->lat_active_indices.d_view : pmy_pack->lat_boundary_send_indices.d_view;
  const int nwork1 = lat_enabled ?
      ((lat_active_only ? pmy_pack->lat_nactive_thispack :
                          pmy_pack->lat_nboundary_send_thispack) - 1) : nmb1;
  if (nwork1 < 0) return;
  auto cu = coarse_cons;
  auto cw = coarse_prim;
  auto &eos = peos->eos_data;
  const int de_idx = dual_energy_idx;

  // Restriction already updated the conserved auxiliary field. Mirror it into the
  // primitive-side storage used by the coarse-boundary prolongation path: the auxiliary
  // is an internal-energy density, the same number in both slots.
  par_for("dual_energy_sync_restricted_mhd", DevExeSpace(), 0, nwork1,
          ks, ke, js, je, is, ie,
  KOKKOS_LAMBDA(const int a, const int k, const int j, const int i) {
    const int m = lat_enabled ? lat_boundary_send_indices(a) : a;
    const Real dens = fmax(cu(m, IDN, k, j, i), eos.dfloor);
    bool efloor_used = false, tfloor_used = false;
    const Real eint_aux = eos_general::ApplyMHDThermalFloors(
        eos, dens, cu(m, de_idx, k, j, i), efloor_used, tfloor_used);
    cu(m, de_idx, k, j, i) = eint_aux;
    cw(m, de_idx, k, j, i) = eint_aux;
  });
}

void MHD::RepairRefinedDualEnergyState(DualArray1D<int> &n2o, DualArray1D<int> &rflag,
                                       const int new_gids, const int new_nmb_local) {
  if (!use_dual_energy) return;
  if (new_nmb_local <= 0) return;

  auto &indcs = pmy_pack->pmesh->mb_indcs;
  const int is = indcs.is;
  const int ie = indcs.ie;
  const int js = indcs.js;
  const int je = indcs.je;
  const int ks = indcs.ks;
  const int ke = indcs.ke;
  const int nmb1 = new_nmb_local - 1;
  auto u0_ = u0;
  auto w0_ = w0;
  auto &eos = peos->eos_data;
  const int de_idx = dual_energy_idx;
  // Step 9 of AMR regrid operates on the new rank-local MeshBlock count, not the old
  // pack size stored in pmy_pack->nmb_thispack. Restrict this kernel to the new local
  // range so it never walks into stale pack slots from the pre-regrid layout.
  par_for("dual_energy_repair_refined_mhd", DevExeSpace(), 0, nmb1, ks, ke, js, je, is,
      ie,
  KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
    if (rflag.d_view(n2o.d_view(m + new_gids)) <= 0) return;
    const Real dens = fmax(u0_(m, IDN, k, j, i), eos.dfloor);
    bool efloor_used = false, tfloor_used = false;
    const Real eint_aux = eos_general::ApplyMHDThermalFloors(
        eos, dens, u0_(m, de_idx, k, j, i), efloor_used, tfloor_used);
    u0_(m, de_idx, k, j, i) = eint_aux;
    w0_(m, de_idx, k, j, i) = eint_aux;
  });
}

}  // namespace mhd
