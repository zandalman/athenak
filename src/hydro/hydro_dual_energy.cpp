//========================================================================================
// AthenaK astrophysical fluid dynamics and numerical relativity code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file hydro_dual_energy.cpp
//! \brief Dual-energy formalism for Newtonian hydro with EOS-aware thermal support

#include <algorithm>
#include <cmath>

#include "athena.hpp"
#include "driver/driver.hpp"
#include "mesh/mesh.hpp"
#include "coordinates/cell_locations.hpp"
#include "eos/eos.hpp"
#include "eos/dual_energy_shock_rule.hpp"
#include "eos/general_c2p_hyd.hpp"
#include "hydro.hpp"
#include "pgen/pgen.hpp"


namespace hydro {

TaskStatus Hydro::DualEnergyStep(Driver *pdrive, int stage) {
  if (!use_dual_energy) return TaskStatus::complete;
  const bool two_register = (pdrive->integrator == "rk4");
  if (stage == 1) {
    // the components of the shock rule's offsets this integrator needs (hydro.hpp)
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

void Hydro::InitializeDualEnergyFieldFromTotal() {
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
  const int de_idx = dual_energy_idx;
  auto &eos = peos->eos_data;

  // Seed the auxiliary internal-energy field from the conserved total energy so restart
  // activation and fresh starts begin from a synchronized state.
  par_for("dual_energy_init", DevExeSpace(), 0, nmb1, ks, ke, js, je, is, ie,
  KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
    const Real dens = fmax(u0_(m, IDN, k, j, i), eos.dfloor);
    const Real e_k = 0.5*(SQR(u0_(m, IM1, k, j, i)) + SQR(u0_(m, IM2, k, j, i)) +
                          SQR(u0_(m, IM3, k, j, i)))/dens;
    bool efloor_used = false, tfloor_used = false;
    Real eint = u0_(m, IEN, k, j, i) - e_k;
    eint = eos_general::ApplyHydroThermalFloors(eos, dens, eint, efloor_used,
                                                tfloor_used);
    u0_(m, de_idx, k, j, i) = eint;
    w0_(m, de_idx, k, j, i) = eint;
  });
}

void Hydro::ApplyDualEnergyFormalism(const Real dt, const int stage, const int nstages,
                                     const Real gam0, const Real gam1,
                                     const Real delta) {
  if (!use_dual_energy) return;

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
  const eos_general::DualEnergyNoField no_field;
  // Evolve the auxiliary internal energy on its adiabat and with the heat of its faces'
  // fans, and give it the energy channel where the cell is being shocked (needs nghost >=
  // 2: the shock test reads w0 shock_reach + 2 = nghost cells off).  The shock test runs
  // at the step's first stage; dual_shocked holds its flags, and dual_shock_debt how far
  // the channel taken at the step's earlier stages moved the auxiliary off its adiabat,
  // and the part of that the first stage's channel left, for the later stages.
  // Everywhere else the conservative-to-auxiliary synchronization follows later in the
  // stage, cell by cell on the owner: Hydro::ResolveDualEnergyChannel (RestrictU, before
  // the state leaves the block, also when the bin-final exchange is deferred) and the
  // owned-cell inversion.  Neither reads a neighbouring cell.
  par_for("dual_energy_formalism", DevExeSpace(), 0, nwork1, ks, ke, js, je, is, ie,
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

    // The shock rule (eos/general_c2p_hyd.hpp, eos/dual_energy_shock_rule.hpp), decided
    // once per step on the step-start state, with the step-final guard.
    const int kb = k - ks, jb = j - js, ib = i - is;
    bool shocked;
    if (decide_shock) {
      shocked = eos_general::DualEnergyCellShocked(eos, w0_, no_field, m, k, j, i, ndim,
                                                   shock_reach, dens, eint);
      shocked_(m, kb, jb, ib) = shocked;
    } else {
      shocked = shocked_(m, kb, jb, ib);
    }
    if (shocked) {
      const Real e_k = 0.5*(SQR(u0_(m, IM1, k, j, i)) + SQR(u0_(m, IM2, k, j, i)) +
                            SQR(u0_(m, IM3, k, j, i)))/dens;
      const Real eint_cons = u0_(m, IEN, k, j, i) - e_k;
      eint = eos_general::DualEnergyShockedStageEint(
          eos, debt_, m, kb, jb, ib, stage, step_final, two_register, gam0, gam1, delta,
          dens, eint, eint_advected, divv_dt, eint_floor, eint_ceil, log_rho_cgs,
          eint_cons, u0_(m, IEN, k, j, i) + e_k);
    }

    u0_(m, de_idx, k, j, i) = eint;
    w0_(m, de_idx, k, j, i) = eint;
  });
}

//----------------------------------------------------------------------------------------
//! \fn void Hydro::ResolveDualEnergyChannel()
//! \brief The owner's thermal decision, made on its own cells before their state leaves
//! the block (Hydro::RestrictU, ahead of the restriction and the send).  Where the
//! inversion accepts the energy channel (eos_general::HydroEnergyChannelResolvesEint) the
//! auxiliary takes E - E_K; elsewhere E becomes eint_aux + E_K, the discarded residual
//! booked like a floor.  Every state a block hands on then satisfies E - E_K = eint_aux,
//! so a state assembled from several of them -- a restriction, a LAT time interpolation
//! or extrapolation, a prolongation -- carries in its auxiliary the same combination of
//! the owners' thermal energies.  Its own E - E_K does not: the kinetic energy of the
//! combined momentum differs from the combined kinetic energy by the velocity dispersion
//! of the parts (a restriction) or by theta (1 - theta) rho |dv|^2 / 2 (a time
//! interpolation), which in a cold hypersonic stream exceeds the thermal energy and is
//! not heat.  Ghost zones and coarse buffers therefore read the auxiliary
//! (eos_general::SingleC2P_GeneralHydDual), and the owner's recovery repeats this
//! decision for whatever changed E after the send.

void Hydro::ResolveDualEnergyChannel(const Real tally_weight) {
  if (!use_dual_energy) return;

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
  const bool tally = floor_energy_ledger;
  auto floor_block = floor_energy_block;
  const int floor_block_count = floor_block.extent_int(0);

  par_for("dual_energy_resolve", DevExeSpace(), 0, nwork1, ks, ke, js, je, is, ie,
  KOKKOS_LAMBDA(const int a, const int k, const int j, const int i) {
    const int m = lat_enabled ? active_indices(a) : a;
    HydCons1D u;
    u.d = u0_(m, IDN, k, j, i);
    u.mx = u0_(m, IM1, k, j, i);
    u.my = u0_(m, IM2, k, j, i);
    u.mz = u0_(m, IM3, k, j, i);
    u.e = u0_(m, IEN, k, j, i);
    // the inversion resets the atmosphere and the excision interior itself
    if (eos_general::NeedsHydroAtmosphereReset(eos, u)) return;
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
    const Real eint_cons = u.e - e_k;
    if (eos_general::HydroEnergyChannelResolvesEint(eint_cons, u.e, eta1)) {
      u0_(m, de_idx, k, j, i) = eint_cons;
      return;
    }
    bool efloor_used = false, tfloor_used = false;
    const Real eint_aux = eos_general::ApplyHydroThermalFloors(
        eos, u.d, u0_(m, de_idx, k, j, i), efloor_used, tfloor_used);
    const Real e_new = eint_aux + e_k;
    if (tally && m < floor_block_count) {
      Kokkos::atomic_add(&floor_block(m), tally_weight*(e_new - u.e)*mbsize.d_view(m).dx1*
                         mbsize.d_view(m).dx2*mbsize.d_view(m).dx3);
    }
    u0_(m, IEN, k, j, i) = e_new;
    u0_(m, de_idx, k, j, i) = eint_aux;
  });
}

void Hydro::SynchronizeRestrictedDualEnergyField(bool lat_active_only) {
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
  auto cu = coarse_u0;
  auto cw = coarse_w0;
  auto &eos = peos->eos_data;
  const int de_idx = dual_energy_idx;

  // Restriction already updated the conserved auxiliary field. Mirror it into the
  // primitive-side storage used by the coarse-boundary prolongation path: the auxiliary
  // is an internal-energy density, the same number in both slots.
  par_for("dual_energy_sync_restricted_hydro", DevExeSpace(), 0, nwork1, ks, ke, js, je,
      is, ie,
  KOKKOS_LAMBDA(const int a, const int k, const int j, const int i) {
    const int m = lat_enabled ? lat_boundary_send_indices(a) : a;
    const Real dens = fmax(cu(m, IDN, k, j, i), eos.dfloor);
    bool efloor_used = false, tfloor_used = false;
    const Real eint_aux = eos_general::ApplyHydroThermalFloors(
        eos, dens, cu(m, de_idx, k, j, i), efloor_used, tfloor_used);
    cu(m, de_idx, k, j, i) = eint_aux;
    cw(m, de_idx, k, j, i) = eint_aux;
  });
}

void Hydro::RepairRefinedDualEnergyState(DualArray1D<int> &n2o, DualArray1D<int> &rflag,
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
  par_for("dual_energy_repair_refined_hydro", DevExeSpace(), 0, nmb1, ks, ke, js, je, is,
      ie,
  KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
    if (rflag.d_view(n2o.d_view(m + new_gids)) <= 0) return;
    const Real dens = fmax(u0_(m, IDN, k, j, i), eos.dfloor);
    bool efloor_used = false, tfloor_used = false;
    const Real eint_aux = eos_general::ApplyHydroThermalFloors(
        eos, dens, u0_(m, de_idx, k, j, i), efloor_used, tfloor_used);
    u0_(m, de_idx, k, j, i) = eint_aux;
    w0_(m, de_idx, k, j, i) = eint_aux;
  });
}

}  // namespace hydro
