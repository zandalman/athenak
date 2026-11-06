//========================================================================================
// Athena++ (Kokkos version) astrophysical MHD code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file ideal_grhyd.cpp
//! \brief derived class that implements ideal gas EOS in general relativistic hydro
//! Uses the same algorithm as implemented for SR hydro.

#include <float.h>

#include <algorithm>

#include "athena.hpp"
#include "mesh/mesh.hpp"
#include "hydro/hydro.hpp"
#include "eos.hpp"
#include "eos/ideal_c2p_hyd.hpp"
#include "eos/gr_gas_entropy.hpp"

#include "coordinates/coordinates.hpp"
#include "coordinates/cartesian_ks.hpp"
#include "coordinates/cell_locations.hpp"

// Occupancy knob for the 256-thread grhyd_c2p launch: blocks/SM ptxas must fit, trading
// registers per thread (2 => <=128 registers) against spilling; 0 removes the constraint.
constexpr int kC2PMinBlocksPerSM = 2;

//----------------------------------------------------------------------------------------
// ctor: also calls EOS base class constructor

IdealGRHydro::IdealGRHydro(MeshBlockPack *pp, ParameterInput *pin) :
    EquationOfState("hydro", pp, pin),
    track_event_counters_(EventLogRequested(pin)),
    event_counters_("grhyd_event_counters", 6) {
  eos_data.hydro_eos = HydroEOSModel::gamma_law;
  eos_data.is_ideal = true;
  eos_data.is_gamma_law = true;
  eos_data.gamma = pin->GetReal("hydro","gamma");
  eos_data.iso_cs = 0.0;
  eos_data.use_e = true;  // ideal gas EOS always uses internal energy
  eos_data.use_t = false;
  eos_data.gamma_max = pin->GetOrAddReal("hydro","gamma_max",(FLT_MAX));  // gamma ceiling
}

//----------------------------------------------------------------------------------------
//! \fn void ConsToPrim()
//! \brief Converts conserved into primitive variables for an ideal gas in GR hydro.
//! Operates over range of cells given in argument list.

void IdealGRHydro::ConsToPrim(DvceArray5D<Real> &cons, DvceArray5D<Real> &prim,
                              const bool only_testfloors,
                              const int il, const int iu, const int jl, const int ju,
                              const int kl, const int ku) {
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int &is = indcs.is, &js = indcs.js, &ks = indcs.ks;
  auto &size = pmy_pack->pmb->mb_size;
  int &nhyd  = pmy_pack->phydro->nhydro;
  int &nscal = pmy_pack->phydro->nscalars;
  int &nmb = pmy_pack->nmb_thispack;
  auto &fofc_ = pmy_pack->phydro->fofc;
  auto eos = eos_data;
  Real gm1 = eos_data.gamma - 1.0;
  const bool lat_enabled = pmy_pack->lat_active_mask_enabled;
  auto active_indices = pmy_pack->lat_active_indices.d_view;
  const int nwork = lat_enabled ? pmy_pack->lat_nactive_thispack : nmb;
  if (nwork <= 0) return;

  const KerrHole hole = pmy_pack->pcoord->coord_data.Hole();
  auto &use_excise = pmy_pack->pcoord->coord_data.bh_excise;
  auto &excision_floor_ = pmy_pack->pcoord->excision_floor;
  auto &excision_flux_ = pmy_pack->pcoord->excision_flux;
  auto &dexcise_ = pmy_pack->pcoord->coord_data.dexcise;
  auto &pexcise_ = pmy_pack->pcoord->coord_data.pexcise;

  // The gas-entropy slot, D s with s = ln(p/rho^Gamma)/(Gamma-1) (eos/gr_gas_entropy.hpp).
  // It is present in every GR run; an array without the column (none at present) is
  // recovered without it.  Before the initial resynchronization the slot holds no
  // entropy yet, so the fallback that reads it waits for that.
  const int ent_idx = pmy_pack->phydro->gas_entropy_idx;
  const bool entropy_slot = (ent_idx >= 0) && (cons.extent_int(1) > ent_idx) &&
      (prim.extent_int(1) > ent_idx);
  const bool entropy_fallback = entropy_slot && !only_testfloors &&
      !pmy_pack->phydro->EntropyInitPending();

  // A failed inversion keeps the primitives the cell last published (below).  They are
  // in w0 when this launch recovers u0 into w0, and they are the cell's own only while
  // the mesh is the one the last recovery of every cell wrote them on.
  const bool recovers_main_state = !only_testfloors &&
      (cons.data() == pmy_pack->phydro->u0.data()) &&
      (prim.data() == pmy_pack->phydro->w0.data());
  const bool keep_previous_primitives = recovers_main_state &&
      (pmy_pack->phydro->w0_topology_version == pmy_pack->pmesh->topology_version);

  // A user-boundary pass converts only the MeshBlocks that carry its face, and skips the
  // cells beyond a user face whose pass comes later (see
  // EquationOfState::SetUserBoundaryFaceFilter).
  const int user_boundary_face = UserBoundaryFaceFilter();
  const int user_boundary_skip = UserBoundarySkipFaces();
  // A plain run's stage inversion converts the cells its fluxes read only
  // (EquationOfState::SetC2PReadSet).
  const int read_set_arm = C2PReadSetArm();
  const int read_set_box = C2PReadSetBox();
  const bool read_set_multi_d = indcs.nx2 > 1;
  const bool read_set_three_d = indcs.nx3 > 1;
  auto &mb_bcs = pmy_pack->pmb->mb_bcs;
  int &ie = indcs.ie, &je = indcs.je, &ke = indcs.ke;

  const int ni   = (iu - il + 1);
  const int nji  = (ju - jl + 1)*ni;
  const int nkji = (ku - kl + 1)*nji;
  const int nmkji = nwork*nkji;

  // Event counters: device-side atomics read back only when an event log exists, so the
  // launch is a plain asynchronous parallel_for rather than a value-returning reduction
  // (the same scheme as IdealGRMHD::ConsToPrimImpl).
  const bool track_events = track_event_counters_;
  auto event_counters = event_counters_.d_view;
  if (track_events) {
    Kokkos::deep_copy(DevExeSpace(), event_counters, 0);
  }

  // The floor ledger, as IdealHydro::ConsToPrim keeps it: every rewrite of U in an owned
  // cell (floors, ceiling, a failure that keeps the previous primitives or falls back to
  // the atmosphere, excision, the entropy root) books its change of D and tau into
  // Mesh::ecounter.eos_floor_mass and eos_floor_energy, in the measure of the mass and
  // tot-E history columns and with the weight of this stage's state in the completed
  // step (Hydro::floor_energy_weight).  S_i has no ledger (as in Newtonian hydro).
  // Resets fire in whole atmosphere regions, so the atomics are spread over slots.
  constexpr int kLedgerSlots = 256;
  const bool book_floors = !only_testfloors;
  auto &ledger_view = pmy_pack->phydro->gr_floor_ledger;
  if (book_floors) {
    if (ledger_view.extent_int(0) != 2*kLedgerSlots) {
      ledger_view = DvceArray1D<Real>("gr_floor_ledger", 2*kLedgerSlots);
    } else {
      Kokkos::deep_copy(DevExeSpace(), ledger_view, 0.0);
    }
  }
  auto ledger = ledger_view;
  const Real floor_weight = pmy_pack->phydro->floor_energy_weight;
  Kokkos::parallel_for("grhyd_c2p", athenak_lw(
  Kokkos::RangePolicy<DevExeSpace, Kokkos::LaunchBounds<256, kC2PMinBlocksPerSM>>
      (DevExeSpace(), 0, nmkji)),
  KOKKOS_LAMBDA(const int &idx) {
    const int a = (idx)/nkji;
    const int m = lat_enabled ? active_indices(a) : a;
    int k = (idx - a*nkji)/nji;
    int j = (idx - a*nkji - k*nji)/ni;
    int i = (idx - a*nkji - k*nji - j*ni) + il;
    j += jl;
    k += kl;
    if (user_boundary_face >= 0 &&
        mb_bcs.d_view(m,user_boundary_face) != BoundaryFlag::user) return;
    if (user_boundary_skip != 0 &&
        BeyondUserBoundaryFaces(mb_bcs.d_view, user_boundary_skip, m, k, j, i,
                                ks, ke, js, je, is, ie)) return;
    if (read_set_arm >= 0 &&
        C2PReadSetSkips(mb_bcs.d_view, m, k, j, i, ks, ke, js, je, is, ie, read_set_arm,
                        read_set_box, read_set_multi_d, read_set_three_d)) return;

    // load single state conserved variables
    HydCons1D u;
    u.d  = cons(m,IDN,k,j,i);
    u.mx = cons(m,IM1,k,j,i);
    u.my = cons(m,IM2,k,j,i);
    u.mz = cons(m,IM3,k,j,i);
    u.e  = cons(m,IEN,k,j,i);
    const Real scalar_cons_density = u.d;

    // Extract components of metric
    Real &x1min = size.d_view(m).x1min;
    Real &x1max = size.d_view(m).x1max;
    Real x1v = CellCenterX(i-is, indcs.nx1, x1min, x1max);

    Real &x2min = size.d_view(m).x2min;
    Real &x2max = size.d_view(m).x2max;
    Real x2v = CellCenterX(j-js, indcs.nx2, x2min, x2max);

    Real &x3min = size.d_view(m).x3min;
    Real &x3max = size.d_view(m).x3max;
    Real x3v = CellCenterX(k-ks, indcs.nx3, x3min, x3max);

    // Four-number Cartesian Kerr-Schild null form (f, l_1, l_2, l_3) in place of the
    // 32 doubles of glower[4][4]/gupper[4][4]; see coordinates/cartesian_ks.hpp.
    KSNullForm nf;
    ComputeKSNullForm(x1v, x2v, x3v, hole, nf);

    HydPrim1D w;
    bool dfloor_used=false, efloor_used=false;
    bool vceiling_used=false, c2p_failure=false;
    int iter_used=0;
    // The state came from KORAL's U2P_ENTROPY: the hot inversion failed.
    bool entropy_inverted = false;

    // Only execute cons2prim if outside excised region
    bool excised = false;
    if (use_excise) {
      if (excision_floor_(m,k,j,i)) {
        w.d = dexcise_;
        w.vx = 0.0;
        w.vy = 0.0;
        w.vz = 0.0;
        w.e = pexcise_/gm1;
        excised = true;
      }
      if (only_testfloors) {
        if (excision_flux_(m,k,j,i)) {
          excised = true;
        }
      }
    }

    if (!(excised)) {
      // calculate SR conserved quantities
      HydCons1D u_sr;
      Real s2;
      TransformToSRHyd(u,nf,s2,u_sr);
      const HydCons1D u_sr_in = u_sr;  // the energy solve floors u_sr in place

      // call c2p function
      // (inline function in ideal_c2p_hyd.hpp file)
      bool eps_unresolved = false;
      SingleC2P_IdealSRHyd(u_sr, eos, s2, w,
                           dfloor_used, efloor_used, c2p_failure, iter_used,
                           &eps_unresolved);

      // KORAL's U2P_ENTROPY where U2P_HOT failed (ApplyEntropyFallback_IdealSRHyd).  The
      // transported D s is brought to the SR frame with the lapse that brings D there
      // (u_sr.d = alpha D).  The FOFC trial (only_testfloors) never takes it: its verdict
      // stays energy-only.
      if (entropy_fallback) {
        const bool hot_suspect = c2p_failure || dfloor_used || efloor_used ||
            eps_unresolved ||
            !(isfinite(w.d) && isfinite(w.e) && isfinite(w.vx) && isfinite(w.vy) &&
              isfinite(w.vz));
        if (hot_suspect) {
          const Real entropy_sr = KSAlpha(nf)*cons(m,ent_idx,k,j,i);
          entropy_inverted = ApplyEntropyFallback_IdealSRHyd(
              u_sr_in, eos, s2, entropy_sr, w, dfloor_used, efloor_used, c2p_failure,
              iter_used, eps_unresolved);
        }
      }

      // A failed inversion keeps the cell's primitives: KORAL's no-fixup branch, as in
      // GRMHD (eos/ideal_grmhd_c2p_impl.hpp).  Where neither the energy nor the entropy
      // inversion solves the cell, KORAL leaves the primitives unchanged (koral_lite
      // u2p.c:342-357, pp = ppbak), applies check_floors_mhd to them like to any
      // recovered state (u2p.c:76-82) and rebuilds U from them at the end of the step
      // (update_entropy, u2p.c:2257-2281).  So the cell publishes the primitives it last
      // published, under the floors and ceilings that follow, and U is rebuilt from
      // them.  The atmosphere at rest stays only where those are not a gas state (rho or
      // eint not positive, anything not finite) or are not this cell's (the mesh changed
      // since they were written: Hydro::w0_topology_version).  The
      // failure flag is untouched, so the failure is counted; the floor flags of the
      // discarded solve are replaced by those of the floors applied to the kept state.
      if (c2p_failure && keep_previous_primitives) {
        KeepPreviousPrimitives_IdealSRHyd(
            eos, prim(m,IDN,k,j,i), prim(m,IEN,k,j,i), prim(m,IVX,k,j,i),
            prim(m,IVY,k,j,i), prim(m,IVZ,k,j,i), w, dfloor_used, efloor_used);
      }

      // apply velocity ceiling if necessary
      Real tmp = KSSpatialNormSq(nf, 1, 2, 3, w.vx, w.vy, w.vz);
      Real lor = sqrt(1.0+tmp);
      if (lor > eos.gamma_max) {
        vceiling_used = true;
        Real factor = sqrt((SQR(eos.gamma_max)-1.0)/(SQR(lor)-1.0));
        w.vx *= factor;
        w.vy *= factor;
        w.vz *= factor;
      }
    }

    // set FOFC flag and quit loop if this function called only to check floors
    if (only_testfloors) {
      if (dfloor_used || efloor_used || vceiling_used || c2p_failure) {
        fofc_(m,k,j,i) = true;
        // dfloor slot counts the cells for which any of the four is true
        if (track_events) Kokkos::atomic_increment(&event_counters(0));
      }
    } else {
      if (track_events) {
        if (dfloor_used) Kokkos::atomic_increment(&event_counters(0));
        if (efloor_used) Kokkos::atomic_increment(&event_counters(1));
        if (vceiling_used) Kokkos::atomic_increment(&event_counters(2));
        if (c2p_failure) Kokkos::atomic_increment(&event_counters(3));
        Kokkos::atomic_max(&event_counters(4), iter_used);
        if (entropy_inverted) Kokkos::atomic_increment(&event_counters(5));
      }

      // store primitive state in 3D array
      prim(m,IDN,k,j,i) = w.d;
      prim(m,IVX,k,j,i) = w.vx;
      prim(m,IVY,k,j,i) = w.vy;
      prim(m,IVZ,k,j,i) = w.vz;
      prim(m,IEN,k,j,i) = w.e;

      // reset conserved variables if floor, ceiling, failure, or excision encountered,
      // and where the state came from the entropy inversion: U is rebuilt from it, the
      // cell's share of KORAL's update_entropy p2u (u2p.c:2257-2281).
      const bool reset_conserved = dfloor_used || efloor_used || vceiling_used ||
          c2p_failure || excised || entropy_inverted;
      // Of that U only tau is new where nothing but the entropy root acted: U2P_ENTROPY
      // replaces the energy equation alone and closes the cell's own D and S_i (its
      // velocity is their exact inverse at the root), so the p2u returns them up to
      // rounding.  D and S_i are kept, and tau and the entropy column are published.  A
      // floor or ceiling after the root changes the state, and U is rebuilt whole.
      const bool entropy_energy_only = entropy_inverted && !dfloor_used &&
          !efloor_used && !vceiling_used && !c2p_failure && !excised;
      if (reset_conserved) {
        HydCons1D u_out;
        SingleP2C_IdealGRHyd(nf, w, eos.gamma, u_out);
        if (book_floors && (i >= is) && (i <= ie) && (j >= js) && (j <= je) &&
            (k >= ks) && (k <= ke)) {
          const Real weight_volume = floor_weight*size.d_view(m).dx1*
                                     size.d_view(m).dx2*size.d_view(m).dx3;
          const int slot = idx & (kLedgerSlots - 1);
          Kokkos::atomic_add(&ledger(slot), weight_volume*(u_out.e - u.e));
          if (!entropy_energy_only) {
            Kokkos::atomic_add(&ledger(kLedgerSlots + slot),
                               weight_volume*(u_out.d - u.d));
          }
        }
        if (entropy_energy_only) {
          u.e = u_out.e;
        } else {
          u = u_out;
          cons(m,IDN,k,j,i) = u.d;
          cons(m,IM1,k,j,i) = u.mx;
          cons(m,IM2,k,j,i) = u.my;
          cons(m,IM3,k,j,i) = u.mz;
        }
        cons(m,IEN,k,j,i) = u.e;
      }

      // The gas entropy.  A state the energy inversion recovered untouched publishes the
      // transported s = (D s)/D.  Any floor, ceiling, failure, excision or entropy root
      // publishes the entropy of the published state, as KORAL resets ENTR wherever
      // check_floors_mhd acted (u2p.c:978-982), and its D s rides the published D.
      if (entropy_slot) {
        const Real s_transported = (scalar_cons_density > 0.0) ?
            cons(m,ent_idx,k,j,i)/scalar_cons_density : 0.0;
        Real s_pub = s_transported;
        if (reset_conserved) {
          s_pub = gr_entropy::SpecificEntropyIdeal(w.d, w.e, eos.gamma);
          // A published state without an entropy (eint = 0, only reachable with every
          // floor at zero) keeps the label it was transported with.
          if (!isfinite(s_pub)) s_pub = s_transported;
          cons(m,ent_idx,k,j,i) = u.d*(isfinite(s_pub) ? s_pub : 0.0);
        }
        prim(m,ent_idx,k,j,i) = isfinite(s_pub) ? s_pub : 0.0;
      }

      // convert scalars (if any)
      const bool preserve_scalars_across_kinematic_repair = vceiling_used &&
          !dfloor_used && !efloor_used && !c2p_failure && !excised;
      for (int n=nhyd; n<(nhyd+nscal); ++n) {
        const Real scalar_den = preserve_scalars_across_kinematic_repair ?
            scalar_cons_density : u.d;
        prim(m,n,k,j,i) = cons(m,n,k,j,i)/scalar_den;
        if (preserve_scalars_across_kinematic_repair) {
          cons(m,n,k,j,i) = u.d*prim(m,n,k,j,i);
        }
      }
    }
  });

  // Every cell of every block now holds primitives a recovery on this mesh wrote, which
  // is what a failed inversion keeps from here on.
  if (recovers_main_state && !lat_enabled && il == 0 && jl == 0 && kl == 0 &&
      iu == prim.extent_int(4) - 1 && ju == prim.extent_int(3) - 1 &&
      ku == prim.extent_int(2) - 1) {
    pmy_pack->phydro->w0_topology_version = pmy_pack->pmesh->topology_version;
  }

  if (book_floors) {
    auto ledger_host = Kokkos::create_mirror_view_and_copy(HostMemSpace(), ledger);
    Real floor_energy = 0.0, floor_mass = 0.0;
    for (int s = 0; s < kLedgerSlots; ++s) {
      floor_energy += ledger_host(s);
      floor_mass += ledger_host(kLedgerSlots + s);
    }
    pmy_pack->pmesh->ecounter.eos_floor_energy += floor_energy;
    pmy_pack->pmesh->ecounter.eos_floor_mass += floor_mass;
  }

  // store appropriate counters
  if (track_events) {
    event_counters_.template modify<DevExeSpace>();
    event_counters_.template sync<HostMemSpace>();
    if (only_testfloors) {
      pmy_pack->pmesh->ecounter.nfofc += event_counters_.h_view(0);
    } else {
      pmy_pack->pmesh->ecounter.neos_dfloor += event_counters_.h_view(0);
      pmy_pack->pmesh->ecounter.neos_efloor += event_counters_.h_view(1);
      pmy_pack->pmesh->ecounter.neos_vceil  += event_counters_.h_view(2);
      pmy_pack->pmesh->ecounter.neos_fail   += event_counters_.h_view(3);
      // the largest count of the step: a second recovery must not overwrite it
      pmy_pack->pmesh->ecounter.maxit_c2p =
          std::max(pmy_pack->pmesh->ecounter.maxit_c2p, event_counters_.h_view(4));
      pmy_pack->pmesh->ecounter.neos_entropy += event_counters_.h_view(5);
    }
  }

  return;
}

//----------------------------------------------------------------------------------------
//! \fn void PrimToCons()
//! \brief Converts primitive into conserved variables for GR hydrodynamics.  Operates
//! over range of cells given in argument list.

void IdealGRHydro::PrimToCons(const DvceArray5D<Real> &prim, DvceArray5D<Real> &cons,
                              const int il, const int iu, const int jl, const int ju,
                              const int kl, const int ku) {
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int &is = indcs.is, &js = indcs.js, &ks = indcs.ks;
  auto &size = pmy_pack->pmb->mb_size;
  const KerrHole hole = pmy_pack->pcoord->coord_data.Hole();
  int &nhyd  = pmy_pack->phydro->nhydro;
  int &nscal = pmy_pack->phydro->nscalars;
  int &nmb = pmy_pack->nmb_thispack;
  Real &gamma = eos_data.gamma;
  const bool lat_enabled = pmy_pack->lat_active_mask_enabled;
  auto active_indices = pmy_pack->lat_active_indices.d_view;
  const int nwork = lat_enabled ? pmy_pack->lat_nactive_thispack : nmb;
  if (nwork <= 0) return;
  // A user-boundary pass writes the ghosts of the MeshBlocks that carry its face only;
  // the ghosts of every other face hold the state just exchanged.
  const int user_boundary_face = UserBoundaryFaceFilter();
  auto &mb_bcs = pmy_pack->pmb->mb_bcs;
  const int ent_idx = pmy_pack->phydro->gas_entropy_idx;
  const bool entropy_slot = (ent_idx >= 0) && (cons.extent_int(1) > ent_idx) &&
      (prim.extent_int(1) > ent_idx);

  par_for("grhyd_p2c", DevExeSpace(), 0, (nwork-1), kl, ku, jl, ju, il, iu,
  KOKKOS_LAMBDA(int a, int k, int j, int i) {
    const int m = lat_enabled ? active_indices(a) : a;
    if (user_boundary_face >= 0 &&
        mb_bcs.d_view(m,user_boundary_face) != BoundaryFlag::user) return;
    // Extract components of metric
    Real &x1min = size.d_view(m).x1min;
    Real &x1max = size.d_view(m).x1max;
    Real x1v = CellCenterX(i-is, indcs.nx1, x1min, x1max);

    Real &x2min = size.d_view(m).x2min;
    Real &x2max = size.d_view(m).x2max;
    Real x2v = CellCenterX(j-js, indcs.nx2, x2min, x2max);

    Real &x3min = size.d_view(m).x3min;
    Real &x3max = size.d_view(m).x3max;
    Real x3v = CellCenterX(k-ks, indcs.nx3, x3min, x3max);

    KSNullForm nf;
    ComputeKSNullForm(x1v, x2v, x3v, hole, nf);

    // Load single state of primitive variables
    HydPrim1D w;
    w.d  = prim(m,IDN,k,j,i);
    w.vx = prim(m,IVX,k,j,i);
    w.vy = prim(m,IVY,k,j,i);
    w.vz = prim(m,IVZ,k,j,i);
    w.e  = prim(m,IEN,k,j,i);

    // call p2c function
    HydCons1D u;
    SingleP2C_IdealGRHyd(nf, w, gamma, u);

    // Set conserved quantities
    cons(m,IDN,k,j,i) = u.d;
    cons(m,IM1,k,j,i) = u.mx;
    cons(m,IM2,k,j,i) = u.my;
    cons(m,IM3,k,j,i) = u.mz;
    cons(m,IEN,k,j,i) = u.e;

    // convert scalars (if any)
    for (int n=nhyd; n<(nhyd+nscal); ++n) {
      cons(m,n,k,j,i) = u.d*prim(m,n,k,j,i);
    }

    // The gas-entropy slot sits past the scalars and rides D like them, but s is
    // signed, so it is not clamped.  A problem generator's first conversion writes
    // D*0 here; nothing reads it before the initial ResyncGasEntropy.
    if (entropy_slot) {
      const Real s = prim(m,ent_idx,k,j,i);
      cons(m,ent_idx,k,j,i) = u.d*(isfinite(s) ? s : 0.0);
    }
  });

  return;
}

//----------------------------------------------------------------------------------------
//! \fn void IdealGRHydro::ResyncGasEntropy()
//! \brief KORAL's update_entropy (u2p.c:2257-2281): s = s(rho, eint) of the primitive
//! state in every cell of the range, w0(slot) = s and u0(slot) = u0(IDN) s.  Only the slot
//! is written: KORAL's p2u_mhd of the same primitives rebuilds the other conserved
//! variables to what they already are, except in cells whose C2P republished them itself.

void IdealGRHydro::ResyncGasEntropy(const DvceArray5D<Real> &w0,
                                    const DvceArray5D<Real> &u0, const bool ghosts) {
  const int ent_idx = pmy_pack->phydro->gas_entropy_idx;
  if (ent_idx < 0) return;
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  const int il = ghosts ? 0 : indcs.is, iu = ghosts ? (u0.extent_int(4) - 1) : indcs.ie;
  const int jl = ghosts ? 0 : indcs.js, ju = ghosts ? (u0.extent_int(3) - 1) : indcs.je;
  const int kl = ghosts ? 0 : indcs.ks, ku = ghosts ? (u0.extent_int(2) - 1) : indcs.ke;
  const Real gamma = eos_data.gamma;
  const bool lat_enabled = pmy_pack->lat_active_mask_enabled;
  auto active_indices = pmy_pack->lat_active_indices.d_view;
  const int nwork = lat_enabled ? pmy_pack->lat_nactive_thispack :
      pmy_pack->nmb_thispack;
  if (nwork <= 0) return;
  par_for("grhyd_entropy_resync", DevExeSpace(), 0, (nwork-1), kl, ku, jl, ju, il, iu,
  KOKKOS_LAMBDA(int a, int k, int j, int i) {
    const int m = lat_enabled ? active_indices(a) : a;
    const Real dens = u0(m,IDN,k,j,i);
    Real s = gr_entropy::SpecificEntropyIdeal(w0(m,IDN,k,j,i), w0(m,IEN,k,j,i), gamma);
    // A state without an entropy (eint = 0) keeps the label it carries.
    if (!isfinite(s)) s = (dens > 0.0) ? u0(m,ent_idx,k,j,i)/dens : 0.0;
    if (!isfinite(s)) s = 0.0;
    w0(m,ent_idx,k,j,i) = s;
    u0(m,ent_idx,k,j,i) = dens*s;
  });
}
