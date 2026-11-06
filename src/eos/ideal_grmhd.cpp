//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file ideal_grmhd.cpp
//! \brief derived class that implements ideal gas EOS in general relativistic mhd.  The
//! grmhd_c2p kernel, IdealGRMHD::ConsToPrimImpl, is in ideal_grmhd_c2p_impl.hpp and is
//! compiled by the ideal_grmhd_c2p_*.cpp units.

#include <float.h>

#include <algorithm>

#include "athena.hpp"
#include "mhd/hybrid_forcefree_algebra.hpp"
#include "mhd/mhd.hpp"
#include "parameter_input.hpp"
#include "eos.hpp"
#include "eos/gr_gas_entropy.hpp"
#include "eos/ideal_c2p_mhd.hpp"

#include "coordinates/coordinates.hpp"
#include "coordinates/cartesian_ks.hpp"
#include "coordinates/cell_locations.hpp"
#include "pgen/pgen.hpp"

namespace {

// The Kastaun warm-start cache of a constant-index run, sized like one component of u0
// with nmb_maxperrank so AMR/load balancing never overruns it, and empty (0 = "no
// guess").  Every accessor allocates it on first use, so its presence is a property of
// the policy alone and the same on every rank, as the load-balance message layout needs
// (CarriedC2PHistory).
void AllocateC2PWarmStartCache(DvceArray5D<Real> &cache, std::uint64_t &version,
                               MeshBlockPack *pp) {
  auto &indcs = pp->pmesh->mb_indcs;
  int ncells1 = indcs.nx1 + 2*(indcs.ng);
  int ncells2 = (indcs.nx2 > 1)? (indcs.nx2 + 2*(indcs.ng)) : 1;
  int ncells3 = (indcs.nx3 > 1)? (indcs.nx3 + 2*(indcs.ng)) : 1;
  int nmb = std::max((pp->nmb_thispack), (pp->pmesh->nmb_maxperrank));
  cache = DvceArray5D<Real>("c2p_mu_cache", nmb, 1, ncells3, ncells2, ncells1);
  Kokkos::deep_copy(cache, 0.0);
  version = pp->pmesh->topology_version;
}

}  // namespace

//----------------------------------------------------------------------------------------
// ctor: also calls EOS base class constructor

IdealGRMHD::IdealGRMHD(MeshBlockPack *pp, ParameterInput *pin) :
    EquationOfState("mhd", pp, pin),
    track_event_counters_(EventLogRequested(pin)),
    event_counters_("grmhd_event_counters", 6) {
  eos_data.is_ideal = true;
  eos_data.is_gamma_law = true;
  eos_data.gamma = pin->GetReal("mhd","gamma");
  eos_data.iso_cs = 0.0;
  eos_data.use_e = true;  // ideal gas EOS always uses internal energy
  eos_data.use_t = false;
  eos_data.gamma_max = pin->GetOrAddReal("mhd","gamma_max",(FLT_MAX));  // gamma ceiling
  if (pin->DoesParameterExist("mhd", "sigma_ceiling")) {
    eos_data.sigma_max = pin->GetReal("mhd", "sigma_ceiling");
  } else if (pin->DoesParameterExist("mhd", "sigma_max")) {
    eos_data.sigma_max = pin->GetReal("mhd", "sigma_max");
  }
  if (pin->DoesParameterExist("mhd", "sigma_ceiling") ||
      pin->DoesParameterExist("mhd", "sigma_max")) {
    if (!isfinite(eos_data.sigma_max) || (eos_data.sigma_max <= 0.0)) {
      eos_data.sigma_max = 0.0;
    }
  }
  // KORAL's B2UURATIOMAX (choices.h:367-368), the partner of sigma_ceiling, read as the
  // dynamical-GR inversion reads it (primitive_solver_hyd.hpp): absent or <= 0 is off.
  if (pin->DoesParameterExist("mhd", "bsq_over_u_ceiling")) {
    bsq_over_u_max = pin->GetReal("mhd", "bsq_over_u_ceiling");
    if (!isfinite(bsq_over_u_max) || (bsq_over_u_max <= 0.0)) {
      bsq_over_u_max = 0.0;
    }
  }

  // The warm-start cache for the Kastaun root of the c2p inversion is allocated on first
  // use (AllocateC2PWarmStartCache), and only by a constant-index run: the
  // variable-index inversion solves every cell cold, so it would never read it.

  floor_ledger_ = DvceArray1D<Real>("grmhd_floor_ledger", 2*kFloorLedgerSlots);
}

//----------------------------------------------------------------------------------------
//! \fn void IdealGRMHD::MarkStepWeight
//! \brief Moves the blocks the stage advances (the LAT active set, or all) to the point of
//! the step MHD::RKUpdate has just brought them to, weight w.  Between that update and the
//! next no recovery that books runs on a block before its own next update (the FOFC trial
//! only tests), so the weight of the update just made is the weight of every booking up
//! to the next one.  A block not yet seen is at the end of a step.

void IdealGRMHD::MarkStepWeight(const Real w) {
  const int nmb = pmy_pack->nmb_thispack;
  if (static_cast<int>(step_weight_.size()) < nmb) step_weight_.resize(nmb, 1.0);
  if (pmy_pack->lat_active_mask_enabled) {
    const auto &active = pmy_pack->lat_active_indices.h_view;
    for (int a = 0; a < pmy_pack->lat_nactive_thispack; ++a) {
      step_weight_[active(a)] = w;
    }
  } else {
    std::fill(step_weight_.begin(), step_weight_.begin() + nmb, w);
  }
  step_weight_seen_ = true;
  step_weight_dirty_ = true;
}

//! \brief The device copy of the weight of every block, sized to the pack and 1 for a
//! block not seen.  The recovery reads the weight of the block it works on: a launch may
//! hold blocks at different points of the step (the LAT receivers' recovery at a bin end
//! visits blocks the last stage update did not advance), so there is no single weight.

DvceArray1D<Real> IdealGRMHD::StepWeightView() {
  const int nmb = pmy_pack->nmb_thispack;
  if (static_cast<int>(step_weight_.size()) < nmb) step_weight_.resize(nmb, 1.0);
  if (static_cast<int>(step_weight_dev_.extent(0)) != nmb || step_weight_dirty_) {
    if (static_cast<int>(step_weight_dev_.extent(0)) != nmb) {
      step_weight_dev_ = DvceArray1D<Real>("grmhd_step_weight", nmb);
    }
    Kokkos::View<Real *, HostMemSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>
        host(step_weight_.data(), nmb);
    Kokkos::deep_copy(step_weight_dev_, host);
    step_weight_dirty_ = false;
  }
  return step_weight_dev_;
}

//----------------------------------------------------------------------------------------
//! \fn void IdealGRMHD::FlushFloorLedger
//! \brief Moves what the recoveries booked since the last flush into
//! Mesh::ecounter.eos_floor_energy and eos_floor_mass and empties the device slots.  The
//! only synchronization of the ledger.

void IdealGRMHD::FlushFloorLedger() {
  auto host = Kokkos::create_mirror_view_and_copy(HostMemSpace(), floor_ledger_);
  Real floor_energy = 0.0, floor_mass = 0.0;
  for (int s = 0; s < kFloorLedgerSlots; ++s) {
    floor_energy += host(s);
    floor_mass += host(kFloorLedgerSlots + s);
  }
  pmy_pack->pmesh->ecounter.eos_floor_energy += floor_energy;
  pmy_pack->pmesh->ecounter.eos_floor_mass += floor_mass;
  DiscardFloorLedger();
}

//! \fn void IdealGRMHD::DiscardFloorLedger
//! \brief Empties the device slots: the checkpoint writer's replay recovery is put back
//! with u0, so what it booked never reached the evolved state.

void IdealGRMHD::DiscardFloorLedger() {
  Kokkos::deep_copy(DevExeSpace(), floor_ledger_, 0.0);
}

//----------------------------------------------------------------------------------------
//! \fn DvceArray4D<Real> IdealGRMHD::C2PWarmStartRoots()
//! \brief The Kastaun warm-start cache as the next inversion reads it.  A cache filled
//! under an earlier mesh topology is emptied here, exactly as ConsToPrimImpl empties it
//! before its first read, so the view holds the roots that inversion would start from.
//!
//! A constant-index run allocates it on first use (AllocateC2PWarmStartCache).  It starts
//! empty (0 = "no guess"), and is emptied again on every topology change that does not
//! carry it (a pure rebalance does), so the guess a cell warm starts from is always that
//! cell's own previous root.  See the declaration in eos.hpp for what the cache does and
//! does not preserve: it changes the converged root at the ~1e-12 level, so it is run
//! history.

DvceArray4D<Real> IdealGRMHD::C2PWarmStartRoots() {
  if (c2p_mu_cache.size() == 0) {
    AllocateC2PWarmStartCache(c2p_mu_cache, c2p_mu_cache_topology_version, pmy_pack);
  }
  if (c2p_mu_cache.size() > 0 &&
      c2p_mu_cache_topology_version != pmy_pack->pmesh->topology_version) {
    Kokkos::deep_copy(c2p_mu_cache, 0.0);
    c2p_mu_cache_topology_version = pmy_pack->pmesh->topology_version;
  }
  if (c2p_mu_cache.size() == 0) return DvceArray4D<Real>();
  // The cache keeps a unit component axis so a rebalance can carry it like u0; with one
  // component the (m,k,j,i) slice is contiguous and assigns to a LayoutRight 4D view.
  return Kokkos::subview(c2p_mu_cache, Kokkos::ALL, 0, Kokkos::ALL, Kokkos::ALL,
                         Kokkos::ALL);
}

//----------------------------------------------------------------------------------------
//! \fn bool IdealGRMHD::ResetC2PWarmStart()
//! \brief Empty the Kastaun warm-start cache.  Every cell then takes the cold bracket on
//! its next inversion, which is the only starting point a restart can reproduce from an
//! ordinary checkpoint: it carries u0, not the cache, so the restart recovers cold.  True
//! without a cache too: the recovery is not idempotent either (floors written back to
//! U), so a checkpoint repeats the recovery as the restart will.

bool IdealGRMHD::ResetC2PWarmStart() {
  if (c2p_mu_cache.size() > 0) {
    Kokkos::deep_copy(c2p_mu_cache, 0.0);
    c2p_mu_cache_topology_version = pmy_pack->pmesh->topology_version;
  }
  return true;
}

//----------------------------------------------------------------------------------------
//! \fn IdealGRMHD::CarriedC2PHistory(), PrepareC2PHistoryCarry(), C2PHistoryCarried()
//! \brief The warm-start cache travels with its blocks in a pure rebalance, like the
//! primitives, so a repartitioned run warm starts every cell from the same root as the
//! run that never repartitioned.

DvceArray5D<Real> *IdealGRMHD::CarriedC2PHistory() {
  if (c2p_mu_cache.size() == 0) {
    AllocateC2PWarmStartCache(c2p_mu_cache, c2p_mu_cache_topology_version, pmy_pack);
  }
  return (c2p_mu_cache.size() == 0) ? nullptr : &c2p_mu_cache;
}

void IdealGRMHD::PrepareC2PHistoryCarry(const int nslots) {
  if (c2p_mu_cache.size() == 0) {
    AllocateC2PWarmStartCache(c2p_mu_cache, c2p_mu_cache_topology_version, pmy_pack);
  }
  if (c2p_mu_cache.size() == 0) return;
  if (c2p_mu_cache_topology_version != pmy_pack->pmesh->topology_version) {
    Kokkos::deep_copy(DevExeSpace(), c2p_mu_cache, 0.0);
    c2p_mu_cache_topology_version = pmy_pack->pmesh->topology_version;
  }
  if (c2p_mu_cache.extent_int(0) < nslots) {
    // Kokkos::resize keeps the existing entries and zero-fills the new blocks.
    Kokkos::resize(c2p_mu_cache, nslots, c2p_mu_cache.extent(1), c2p_mu_cache.extent(2),
                   c2p_mu_cache.extent(3), c2p_mu_cache.extent(4));
  }
}

void IdealGRMHD::C2PHistoryCarried(const std::uint64_t old_version,
                                   const std::uint64_t new_version) {
  if (c2p_mu_cache.size() != 0 && c2p_mu_cache_topology_version == old_version) {
    c2p_mu_cache_topology_version = new_version;
  }
}

//----------------------------------------------------------------------------------------
//! \fn void ConsToPrim()
//! \brief Converts conserved into primitive variables.
//! Operates over range of cells given in argument list.

void IdealGRMHD::ConsToPrim(DvceArray5D<Real> &cons, const DvceFaceFld4D<Real> &b,
                            DvceArray5D<Real> &prim, DvceArray5D<Real> &bcc,
                            const bool only_testfloors,
                            const int il, const int iu, const int jl, const int ju,
                            const int kl, const int ku) {
  const bool apply_magnetization_ceilings =
      MagnetizationCeilingsActive_IdealGRMHD(eos_data, UOverBsqMin());
  if (only_testfloors) {
    if (apply_magnetization_ceilings) {
      ConsToPrimImpl<true, true>(cons, b, prim, bcc, il, iu, jl, ju, kl, ku,
                                 track_event_counters_);
    } else {
      ConsToPrimImpl<true, false>(cons, b, prim, bcc, il, iu, jl, ju, kl, ku,
                                  track_event_counters_);
    }
  } else {
    if (apply_magnetization_ceilings) {
      ConsToPrimImpl<false, true>(cons, b, prim, bcc, il, iu, jl, ju, kl, ku,
                                  track_event_counters_);
    } else {
      ConsToPrimImpl<false, false>(cons, b, prim, bcc, il, iu, jl, ju, kl, ku,
                                   track_event_counters_);
    }
  }
}

//----------------------------------------------------------------------------------------
//! \fn void PrimToCons()
//! \brief Converts primitive into conserved variables.  Operates over range of cells
//! given in argument list.

void IdealGRMHD::PrimToCons(const DvceArray5D<Real> &prim, const DvceArray5D<Real> &bcc,
                            DvceArray5D<Real> &cons, const int il, const int iu,
                            const int jl, const int ju, const int kl, const int ku) {
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int &is = indcs.is, &js = indcs.js, &ks = indcs.ks;
  auto &size = pmy_pack->pmb->mb_size;
  const KerrHole hole = pmy_pack->pcoord->coord_data.Hole();
  int &nmhd  = pmy_pack->pmhd->nmhd;
  int &nscal = pmy_pack->pmhd->nscalars;
  int &nmb = pmy_pack->nmb_thispack;
  Real gamma = eos_data.gamma;
  Real gamma_max = eos_data.gamma_max;
  const bool lat_enabled = pmy_pack->lat_active_mask_enabled;
  auto active_indices = pmy_pack->lat_active_indices.d_view;
  const int nwork = lat_enabled ? pmy_pack->lat_nactive_thispack : nmb;
  if (nwork <= 0) return;
  const int user_boundary_face = UserBoundaryFaceFilter();
  auto &mb_bcs = pmy_pack->pmb->mb_bcs;

  const int ent_idx = pmy_pack->pmhd->gas_entropy_idx;
  const bool entropy_slot = (ent_idx >= 0) && (cons.extent_int(1) > ent_idx) &&
      (prim.extent_int(1) > ent_idx);

  par_for("grmhd_p2c", DevExeSpace(), 0, (nwork-1), kl, ku, jl, ju, il, iu,
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

    // Same four-number Kerr-Schild null form as ConsToPrim above (see the comment
    // there); the two 4x4 metric matrices are never materialized in this kernel either.
    KSNullForm nf;
    ComputeKSNullForm(x1v, x2v, x3v, hole, nf);

    // Load single state of primitive variables
    MHDPrim1D w;
    w.d  = prim(m,IDN,k,j,i);
    w.vx = prim(m,IVX,k,j,i);
    w.vy = prim(m,IVY,k,j,i);
    w.vz = prim(m,IVZ,k,j,i);
    w.e  = prim(m,IEN,k,j,i);

    // load cell-centered fields into primitive state
    w.bx = bcc(m,IBX,k,j,i);
    w.by = bcc(m,IBY,k,j,i);
    w.bz = bcc(m,IBZ,k,j,i);

    const Real local_gamma = gamma;

    // call p2c function
    HydCons1D u;
    SingleP2C_IdealGRMHD(nf, w, local_gamma, u);

    // store conserved quantities in 3D array
    cons(m,IDN,k,j,i) = u.d;
    cons(m,IM1,k,j,i) = u.mx;
    cons(m,IM2,k,j,i) = u.my;
    cons(m,IM3,k,j,i) = u.mz;
    cons(m,IEN,k,j,i) = u.e;

    // convert scalars (if any)
    for (int n=nmhd; n<(nmhd+nscal); ++n) {
      Real scalar = prim(m,n,k,j,i);
      scalar = (isfinite(scalar) && scalar > 0.0) ? scalar : 0.0;
      cons(m,n,k,j,i) = u.d*scalar;
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
//! \fn void IdealGRMHD::ResyncGasEntropy()
//! \brief KORAL's update_entropy (u2p.c:2257-2281): s = s(rho, eint) of the primitive
//! state in every cell of the range, w0(slot) = s and u0(slot) = u0(IDN) s.  Only the slot
//! is written: KORAL's p2u_mhd of the same primitives rebuilds the other conserved
//! variables to what they already are, except in cells whose C2P republished them itself.

void IdealGRMHD::ResyncGasEntropy(const DvceArray5D<Real> &w0,
                                  const DvceArray5D<Real> &u0, const bool ghosts) {
  const int ent_idx = pmy_pack->pmhd->gas_entropy_idx;
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

  par_for("grmhd_entropy_resync", DevExeSpace(), 0, (nwork-1), kl, ku, jl, ju, il, iu,
  KOKKOS_LAMBDA(int a, int k, int j, int i) {
    const int m = lat_enabled ? active_indices(a) : a;
    const Real rho = w0(m,IDN,k,j,i);
    const Real eint = w0(m,IEN,k,j,i);
    Real local_gamma = gamma;
    const Real dens = u0(m,IDN,k,j,i);
    Real s = gr_entropy::SpecificEntropyIdeal(rho, eint, local_gamma);
    // A state without an entropy (eint = 0) keeps the label it carries.
    if (!isfinite(s)) s = (dens > 0.0) ? u0(m,ent_idx,k,j,i)/dens : 0.0;
    if (!isfinite(s)) s = 0.0;
    w0(m,ent_idx,k,j,i) = s;
    u0(m,ent_idx,k,j,i) = dens*s;
  });
}
