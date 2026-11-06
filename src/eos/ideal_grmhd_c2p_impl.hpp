#ifndef EOS_IDEAL_GRMHD_C2P_IMPL_HPP_
#define EOS_IDEAL_GRMHD_C2P_IMPL_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file ideal_grmhd_c2p_impl.hpp
//! \brief IdealGRMHD::ConsToPrimImpl, the grmhd_c2p kernel, and the helpers only it
//! reads.  Included only by the ideal_grmhd_c2p_*.cpp units, each of which explicitly
//! instantiates half of the specializations IdealGRMHD::ConsToPrim (ideal_grmhd.cpp)
//! calls, so each gets its own ptxas pass and every specialization is compiled in
//! exactly one unit.

#include <float.h>

#include <algorithm>

#include "athena.hpp"
#include "mhd/hybrid_forcefree_algebra.hpp"
#include "mhd/mhd.hpp"
#include "parameter_input.hpp"
#include "eos.hpp"
#include "eos/ideal_c2p_mhd.hpp"

#include "coordinates/coordinates.hpp"
#include "coordinates/cartesian_ks.hpp"
#include "coordinates/cell_locations.hpp"
#include "pgen/pgen.hpp"

// Occupancy knob for the 256-thread grmhd_c2p launch: blocks/SM ptxas must fit, trading
// registers per thread (2 => <=128, i.e. 25% occupancy on a V100) against spilling; 0
// removes the constraint and restores the unbounded ~236-register, 12.5%-occupancy build.
constexpr int kC2PMinBlocksPerSM = 2;

namespace {

// The grmhd_c2p launch policy.  A light-weight launch passes the ~2 kB functor as a
// kernel argument; the default puts it in constant memory, and every such launch first
// blocks the host until the previous constant-memory kernel has finished.  An argument
// functor is only cheap while the lambda body is inlined into the kernel: a body nvcc
// keeps out of line receives `this` as a pointer into the parameters, and nvcc then
// copies the whole functor into local memory in every thread.  The bodies of the
// constant-gamma recovery (at most ~22k SASS instructions) are inlined.
template <bool light_weight, typename Policy>
inline auto C2PLaunchPolicy(const Policy &policy) {
  if constexpr (light_weight) {
    return athenak_lw(policy);
  } else {
    return policy;
  }
}

// A test-mode conversion reads the FOFC trial arrays, which sit on the owned cells at the
// band origin of the MHD module; any other conversion reads the register as it is.
template <bool only_testfloors>
auto TrialOrPlain(DvceArray5D<Real> &a, const mhd::MHD &mhd) {
  if constexpr (only_testfloors) {
    return mhd.FofcTrialBand(a);
  } else {
    return a;
  }
}

}  // namespace

template <bool only_testfloors, bool apply_magnetization_ceilings>
void IdealGRMHD::ConsToPrimImpl(DvceArray5D<Real> &cons_in,
                                const DvceFaceFld4D<Real> &b,
                                DvceArray5D<Real> &prim, DvceArray5D<Real> &bcc_in,
                                const int il, const int iu, const int jl, const int ju,
                                const int kl, const int ku, const bool track_event_counters) {
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int &is = indcs.is, &ie = indcs.ie;
  int &js = indcs.js, &je = indcs.je;
  int &ks = indcs.ks, &ke = indcs.ke;
  auto &size = pmy_pack->pmb->mb_size;
  int &nmhd  = pmy_pack->pmhd->nmhd;
  int &nscal = pmy_pack->pmhd->nscalars;
  int &nmb = pmy_pack->nmb_thispack;
  auto &fofc_ = pmy_pack->pmhd->fofc;
  auto eos = eos_data;
  // The FOFC test reads the trial state, which sits on the owned cells (MHD::utest): its
  // conserved state and field are indexed by the global cell through the band origin.
  const auto cons = TrialOrPlain<only_testfloors>(cons_in, *pmy_pack->pmhd);
  const auto bcc = TrialOrPlain<only_testfloors>(bcc_in, *pmy_pack->pmhd);
  const Real u_over_bsq_min = UOverBsqMin();
  const bool lat_enabled = pmy_pack->lat_active_mask_enabled;
  auto active_indices = pmy_pack->lat_active_indices.d_view;
  const int nwork = lat_enabled ? pmy_pack->lat_nactive_thispack : nmb;
  if (nwork <= 0) return;
  const int user_boundary_face = UserBoundaryFaceFilter();
  const int user_boundary_skip = UserBoundarySkipFaces();
  // A plain run's stage inversion converts the cells its fluxes read only
  // (EquationOfState::SetC2PReadSet).
  const int read_set_arm = C2PReadSetArm();
  const int read_set_box = C2PReadSetBox();
  const bool read_set_multi_d = indcs.nx2 > 1;
  const bool read_set_three_d = indcs.nx3 > 1;
  auto &mb_bcs = pmy_pack->pmb->mb_bcs;
  const KerrHole hole = pmy_pack->pcoord->coord_data.Hole();
  auto &use_excise = pmy_pack->pcoord->coord_data.bh_excise;
  auto &excision_floor_ = pmy_pack->pcoord->excision_floor;
  auto &excision_flux_ = pmy_pack->pcoord->excision_flux;
  auto &dexcise_ = pmy_pack->pcoord->coord_data.dexcise;
  // The excised state at rest at <coord>/pexcise: its internal energy at the deck index,
  // or, where the species closure is the gas, that of the pair at pexcise
  // (SpeciesStateAtPressure), which the recovery's species reset re-splits at the same
  // temperatures, so the index it publishes is 1 + pexcise/eint and the pressure pexcise.
  Real eint_excise = 0.0;
  if (use_excise) {
    eint_excise = pmy_pack->pcoord->coord_data.pexcise/(eos.gamma - 1.0);

  }

  // The gas-entropy slot, D s with s = ln(p/rho^Gamma)/(Gamma-1) (eos/gr_gas_entropy.hpp).
  // It is present in every GR run; an array without the column is recovered without it.
  // The FOFC trial never reads it (its verdict stays energy-only), and before the
  // initial resynchronization it holds no entropy yet, so the fallback that reads it
  // waits for that.
  const int ent_idx = pmy_pack->pmhd->gas_entropy_idx;
  const bool entropy_slot = !only_testfloors && (ent_idx >= 0) &&
      (cons_in.extent_int(1) > ent_idx) && (prim.extent_int(1) > ent_idx);
  const bool entropy_fallback = entropy_slot && !pmy_pack->pmhd->EntropyInitPending();

  const int ni   = (iu - il + 1);
  const int nji  = (ju - jl + 1)*ni;
  const int nkji = (ku - kl + 1)*nji;
  const int nmkji = nwork*nkji;
  // The cells the call leaves out (c2p_skip_box), relative to the call's box.  The launch
  // enumerates the others, CellOutsideBox; an empty box leaves the k-j-i order of a full
  // box.  Only the ordinary inversion honours it.
  const CellBox skip_abs = c2p_skip_box;
  const bool skip_cells = !CellBoxEmpty(skip_abs);
  if (skip_cells) {
    if (only_testfloors || skip_abs.k0 < kl || skip_abs.k0 + skip_abs.nk > ku + 1 || skip_abs.j0 < jl || skip_abs.j0 + skip_abs.nj > ju + 1 || skip_abs.i0 < il || skip_abs.i0 + skip_abs.ni > iu + 1) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "An inversion skip box outside the call's box or on a "
                << "test-floors or selected-cell launch" << std::endl;
      std::exit(EXIT_FAILURE);
    }
  }
  const int snk = skip_cells ? skip_abs.nk : 0;
  const int snj = skip_cells ? skip_abs.nj : 0;
  const int sni = skip_cells ? skip_abs.ni : 0;
  const int sk0 = skip_cells ? skip_abs.k0 - kl : 0;
  const int sj0 = skip_cells ? skip_abs.j0 - jl : 0;
  const int si0 = skip_cells ? skip_abs.i0 - il : 0;
  const int nvisit = nkji - snk*snj*sni;

  auto event_counters = event_counters_.d_view;
  if (track_event_counters) {
    Kokkos::deep_copy(DevExeSpace(), event_counters, 0);
  }

  // Warm-start cache for the Kastaun root.  It is shaped like one component of the fine
  // u0, so it is only addressable when this launch works on arrays of that shape (never
  // for the coarse arrays).  Any pass may READ it -- a good guess stays a good guess --
  // but only the authoritative u0/w0 inversion may WRITE it, so that the FOFC test pass
  // and the hybrid/selected-cell repairs cannot publish a root for a state that is not
  // the one being evolved.
  // A remesh (AMR or load balance) renumbers the local blocks, so an entry left over
  // from before it is a DIFFERENT cell's root -- the extent test below cannot see that.
  // Empty the cache instead; the cells then take the cold bracket for one call, which is
  // exactly what they did before the cache existed.  Costs one fill on remesh steps only.
  // A pure rebalance carries the cache with its blocks and re-stamps its version
  // (C2PHistoryCarried), so it is not emptied here.  Only a constant-index run keeps it,
  // allocated on first use.
  if (c2p_mu_cache.size() == 0) (void) C2PWarmStartRoots();

  if (c2p_mu_cache_topology_version != pmy_pack->pmesh->topology_version) {
    Kokkos::deep_copy(DevExeSpace(), c2p_mu_cache, 0.0);
    c2p_mu_cache_topology_version = pmy_pack->pmesh->topology_version;
  }
  auto mu_cache = c2p_mu_cache;
  const bool mu_cache_owned_only = c2p_mu_cache_owned_only;
  const bool mu_cache_addressable =
      (mu_cache.extent_int(0) >= nmb) &&
      (mu_cache.extent_int(2) == prim.extent_int(2)) &&
      (mu_cache.extent_int(3) == prim.extent_int(3)) &&
      (mu_cache.extent_int(4) == prim.extent_int(4));
  const bool read_mu_cache = mu_cache_addressable;
  const bool write_mu_cache = mu_cache_addressable && !only_testfloors && (cons_in.data() == pmy_pack->pmhd->u0.data()) && (prim.data() == pmy_pack->pmhd->w0.data());
  // A failed inversion keeps the primitives the cell last published (below).  They are
  // in w0 when this launch recovers u0 into w0, and they are the cell's own only while
  // the mesh is the one the last recovery of every cell wrote them on.
  const bool recovers_main_state = !only_testfloors &&
      (cons_in.data() == pmy_pack->pmhd->u0.data()) &&
      (prim.data() == pmy_pack->pmhd->w0.data());
  const bool keep_previous_primitives = recovers_main_state &&
      (pmy_pack->pmhd->w0_topology_version == pmy_pack->pmesh->topology_version);
  // The floor ledger (eos.hpp): every rewrite of U in an owned cell of the main state
  // (floors, ceilings, a failure, excision, the entropy root) books its change of tau and
  // D at the weight of the point of the step its block is at, as IdealGRHydro::ConsToPrim does.  Device
  // atomics, no readback here.  The ledger view, the weight and the flag are captured
  // at the top level of the lambda, outside every if constexpr.
  constexpr int kLedgerSlots = kFloorLedgerSlots;
  const bool book_floors = recovers_main_state;
  auto ledger = floor_ledger_;
  auto step_weight = StepWeightView();
  // The entropy inversion of a variable-index cell solves on the adiabat of its species
  // scalars where they carry the gas (ideal_c2p_variable_gamma.hpp).

  auto c2p_cell = KOKKOS_LAMBDA(const int &idx) {
    // NVCC cannot first-capture an extended-lambda variable from inside an
    // `if constexpr` body.  Anchor every value first used by the hybrid-only
    // branch here; the false template instantiation still compiles that branch
    // away and retains the ordinary dev kernel arithmetic.
    (void) step_weight;
    (void) use_excise;
    (void) excision_floor_;
    (void) is;
    (void) ie;
    (void) js;
    (void) je;
    (void) ks;
    (void) ke;
    (void) nmhd;
    (void) nscal;
    (void) prim;
    // NVCC extended lambdas require captures to be established outside if constexpr.
    (void) mu_cache;
    (void) read_mu_cache;
    (void) write_mu_cache;
    (void) fofc_;
    (void) excision_flux_;
    (void) keep_previous_primitives;
    (void) ent_idx;
    (void) entropy_slot;
    (void) entropy_fallback;
    (void) b;
    (void) bcc;

    int a = (idx)/nvisit;
    int m = lat_enabled ? active_indices(a) : a;
    if (user_boundary_face >= 0 &&
        mb_bcs.d_view(m,user_boundary_face) != BoundaryFlag::user) return;
    const int local = idx - a*nvisit;
    int k, j, i;
    if (snk > 0) {
      CellOutsideBox(local, ni, nji, sk0, snk, sj0, snj, si0, sni, k, j, i);
    } else {
      k = local/nji;
      j = (local - k*nji)/ni;
      i = local - k*nji - j*ni;
    }
    i += il;
    j += jl;
    k += kl;
    // A user-boundary pass inverts the first active layer of its face (NoInflowTorus).
    // A cell of that layer beyond a user face whose pass comes later is a ghost that
    // pass rebuilds from its own first active layer: its U is left over from before, and
    // an inversion here would only count events of a state nobody publishes.
    if (user_boundary_skip != 0 &&
        BeyondUserBoundaryFaces(mb_bcs.d_view, user_boundary_skip, m, k, j, i,
                                ks, ke, js, je, is, ie)) return;
    if (read_set_arm >= 0 &&
        C2PReadSetSkips(mb_bcs.d_view, m, k, j, i, ks, ke, js, je, is, ie, read_set_arm,
                        read_set_box, read_set_multi_d, read_set_three_d)) return;

    const bool owned_cell =
        i >= is && i <= ie &&
        j >= js && j <= je &&
        k >= ks && k <= ke;
    const bool derived_communication_ghost =
        !owned_cell;

    // load single state conserved variables
    MHDCons1D u;
    u.d  = cons(m,IDN,k,j,i);
    u.mx = cons(m,IM1,k,j,i);
    u.my = cons(m,IM2,k,j,i);
    u.mz = cons(m,IM3,k,j,i);
    u.e  = cons(m,IEN,k,j,i);
    Real scalar_cons_density = u.d;

    // load cell-centered fields into conserved state
    // use input CC fields if only testing floors with FOFC
    if (only_testfloors) {
      u.bx = bcc(m,IBX,k,j,i);
      u.by = bcc(m,IBY,k,j,i);
      u.bz = bcc(m,IBZ,k,j,i);
    // else use simple linear average of face-centered fields
    } else {
      u.bx = 0.5*(b.x1f(m,k,j,i) + b.x1f(m,k,j,i+1));
      u.by = 0.5*(b.x2f(m,k,j,i) + b.x2f(m,k,j+1,i));
      u.bz = 0.5*(b.x3f(m,k,j,i) + b.x3f(m,k+1,j,i));
    }

    // Per-cell verdict on the magnetization ceilings (see the host-side comment above).
    // `apply_magnetization_ceilings` is a template parameter, so with neither ceiling
    // configured the weight load below is compiled out entirely.
    bool apply_magnetization_ceilings_here = apply_magnetization_ceilings;

    // Communication ghosts are derived primitive views of an owner-exchanged conserved
    // state.  This includes ghosts selected by a committed hybrid transition: local
    // floors may repair W/raw branch data for the face solve, but must never republish U.
    // An owned cell that a pass after an exchange recovers publishes as its owner would.
    const bool publish_conserved_repairs =
        true;

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

    // Cartesian Kerr-Schild is a rank-1 update of Minkowski, g = eta + f l l with
    // l_0 = 1, so the four numbers (f, l_1, l_2, l_3) carry the whole geometry.  Holding
    // those instead of glower[4][4]/gupper[4][4] keeps 4 live doubles per thread instead
    // of 32 and removes the ~40 multiplies that materialize the two matrices.  Every
    // accessor below is an exact rank-1 identity; none of them assumes l.l = 1, which the
    // r < 1e-6 floor inside ComputeKSNullForm genuinely breaks (at the origin f = 4e6 and
    // l.l - 1 = -1), so the reduction stays valid down to the coordinate singularity.
    KSNullForm nf;
    ComputeKSNullForm(x1v, x2v, x3v, hole, nf);

    bool coordinate_excised = use_excise && excision_floor_(m,k,j,i);
    EOS_Data cell_eos = eos;

    HydPrim1D w;
    bool dfloor_used=false, efloor_used=false;
    // The published state came from KORAL's U2P_ENTROPY: the hot inversion failed.
    bool entropy_inverted = false;
    // rho*h before the floors, kept so the velocity of whatever they inject is decided
    // once, after the magnetization ceilings too.
    Real w_prefloor = 0.0;
    bool vceiling_used=false, invalid_velocity_state=false;
    bool magceiling_used=false, c2p_failure=false;
    // A failed cell that publishes its previous primitives instead of the atmosphere.
    bool kept_previous = false;
    // Remembered past the entropy fallback, which may clear c2p_failure: the warm-start
    // cache must not learn the root of a solve that failed.
    bool energy_solve_failed = false;
    // The energy root holds no significant digit of eps (SingleC2P_IdealSRMHD): a hot
    // failure for the entropy inversion, not a failure of the cell.
    bool eps_unresolved = false;
    int iter_used=0;
    // Cached Kastaun root of the previous inversion of this cell (0 means "no guess"),
    // and the root this inversion converges to.  The variable-EOS inversion solves cold:
    // the root find accepts any mu within its tolerance, which at W = 10 is ~1e-9 in the
    // density, and which mu it accepts depends on the bracket it starts from.  A
    // warm start would tie the published state to the cell's inversion history -- an
    // owner and its ghost, or a run and its restart, have different histories -- where
    // the variable-index inversion makes it a function of U alone.
    const Real mu_guess = read_mu_cache ? mu_cache(m,0,k,j,i) : 0.0;
    Real mu_root = 0.0;

    // Only execute cons2prim if outside excised region
    bool excised = false;
    if (use_excise) {
      if (coordinate_excised) {
        w.d = dexcise_;
        w.vx = 0.0;
        w.vy = 0.0;
        w.vz = 0.0;
        w.e = eint_excise;
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
      MHDCons1D u_sr;
      Real s2, b2, rpar;
      TransformToSRMHD(u,nf,s2,b2,rpar,u_sr);
      MHDCons1D u_sr_initial = u_sr;

      // call c2p function
      // (inline function in ideal_c2p_mhd.hpp file)
      SingleC2P_IdealSRMHD(u_sr, cell_eos, s2, b2, rpar, w,
                           dfloor_used, efloor_used, c2p_failure, iter_used,
                           mu_guess, mu_root, &w_prefloor, ConstantGammaC2P(), nullptr,
                           &eps_unresolved);

      energy_solve_failed = c2p_failure;

      // KORAL's U2P_ENTROPY where U2P_HOT failed (ApplyEntropyFallback_IdealSRMHD): no
      // root, a non-finite one, the wrong-root residual flag, a root whose eps holds no
      // significant digit (not KORAL's test, which the round trip of UU cannot fail), or
      // -- only where a floor acted -- a failure of the zero-floor solve of the cell's
      // own U.  The transported
      // D s is brought to the SR frame with the lapse that brings D there (u_sr.d =
      // alpha D).  Never in the FOFC trial, whose verdict stays energy-only.
      if constexpr (!only_testfloors) {
        if (entropy_fallback &&
            (c2p_failure || dfloor_used || efloor_used || eps_unresolved)) {
          entropy_inverted = ApplyEntropyFallback_IdealSRMHD(
              u_sr_initial, cell_eos, s2, b2, rpar, mu_root,
              KSAlpha(nf)*cons(m,ent_idx,k,j,i), w,
              dfloor_used, efloor_used,
              c2p_failure, w_prefloor, iter_used, eps_unresolved);

        }
      }

      // A failed inversion keeps the cell's primitives: KORAL's no-fixup branch, as on
      // the dynamical metric (eos/primitive_solver_hyd.hpp, kept_previous).  Where no
      // channel inverts the cell, KORAL leaves the primitives unchanged (koral_lite
      // u2p.c:342-357, pp = ppbak), requests a fixup that its default build does not run
      // (choices.h:310-311, DOU2PMHDFIXUPS 0; finite.c:5014-5018), applies
      // check_floors_mhd to them like to any recovered state (u2p.c:76-82), and rebuilds
      // U from them at the end of the step (update_entropy, u2p.c:2257-2281, p2u_mhd).
      // So the cell publishes the primitives it last published, under the floors and
      // ceilings that follow, and U is rebuilt from them.  The atmosphere stays only
      // where those are not a gas state (rho or u not positive, anything not finite) or
      // are not this cell's (the mesh changed since they were written:
      // MHD::w0_topology_version).  The failure flag is untouched, so the failure is
      // counted and every reader of it still sees one.  The floor flags of the discarded
      // solve describe a state nobody publishes, so they are replaced by those of the
      // floors applied to the kept one.
      if constexpr (!only_testfloors) {
        if (c2p_failure && keep_previous_primitives) {
          const Real rho_prev = prim(m,IDN,k,j,i);
          const Real eint_prev = prim(m,IEN,k,j,i);
          const Real vx_prev = prim(m,IVX,k,j,i);
          const Real vy_prev = prim(m,IVY,k,j,i);
          const Real vz_prev = prim(m,IVZ,k,j,i);
          if (isfinite(rho_prev) && rho_prev > 0.0 &&
              isfinite(eint_prev) && eint_prev > 0.0 &&
              isfinite(vx_prev) && isfinite(vy_prev) && isfinite(vz_prev)) {
            w.d = rho_prev;
            w.e = eint_prev;
            w.vx = vx_prev;
            w.vy = vy_prev;
            w.vz = vz_prev;
            // The enthalpy the floors and ceilings below measure what they add against.
            w_prefloor = rho_prev + cell_eos.gamma*eint_prev;
            dfloor_used = false;
            efloor_used = false;
            if (w.d < cell_eos.dfloor) {
              w.d = cell_eos.dfloor;
              dfloor_used = true;
            }
            const Real eint_floor = IdealGRMHDEintFloor(cell_eos, w.d);
            if (w.e < eint_floor) {
              w.e = eint_floor;
              efloor_used = true;
            }
            kept_previous = true;
          }
        }
      }

      // Apply the common KORAL GAMMAMAXHD ceiling with a scaled metric norm.  The
      // traditional direct quadratic form can overflow or become NaN before its
      // comparison, which lets an exact-zero hybrid cell publish an invalid velocity.
      Real glower_lorentz[4][4];
      KSFillLower(nf, glower_lorentz);
      const auto lorentz_limit = mhd::forcefree::ApplyIsotropicLorentzLimit(
          glower_lorentz, cell_eos.gamma_max, w.vx, w.vy, w.vz);
      if (!lorentz_limit.valid) {
        w.vx = 0.0;
        w.vy = 0.0;
        w.vz = 0.0;
        vceiling_used = true;
        invalid_velocity_state = true;
      } else if (lorentz_limit.limited) {
        vceiling_used = true;
      }
      Real rho_before_sigceiling = w.d;
      if (apply_magnetization_ceilings_here) {
        magceiling_used = ApplyMagnetizationCeilings_IdealGRMHD(
            cell_eos, u_over_bsq_min, nf, u, w);
      }
      // Everything above has raised rho or u somewhere; this is the single place the
      // velocity that comes with the added material is decided.  A failed cell that
      // kept its primitives is floored like a recovered one, against their enthalpy.  A
      // cell reset to the atmosphere, or with an invalid velocity, is a different animal
      // -- the state was reset, not floored -- and has no pre-floor state worth
      // re-injecting into.
      if ((dfloor_used || efloor_used || magceiling_used) &&
          (!c2p_failure || kept_previous) && !invalid_velocity_state) {
        ReinjectFlooredMass_IdealGRMHD(cell_eos.gamma, nf, u, w_prefloor, w);
      }

    }

    // Store the converged Kastaun root so the next inversion of this cell can warm
    // start from it.  Only a successful, non-excised, authoritative inversion writes.
    if (write_mu_cache && !excised && !energy_solve_failed && (mu_root > 0.0) && (owned_cell || !mu_cache_owned_only)) {
      mu_cache(m,0,k,j,i) = mu_root;
    }

    // set FOFC flag and quit loop if this function called only to check floors
    if (only_testfloors) {
      if (dfloor_used || efloor_used || vceiling_used || magceiling_used || c2p_failure) {
        fofc_(m,k,j,i) = true;
        if (track_event_counters) {
          Kokkos::atomic_increment(&event_counters(0));
        }
      }
    } else {
      if (track_event_counters) {
        if (dfloor_used) Kokkos::atomic_increment(&event_counters(0));
        if (efloor_used) Kokkos::atomic_increment(&event_counters(1));
        if (vceiling_used || magceiling_used) {
          Kokkos::atomic_increment(&event_counters(2));
        }
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

      // store cell-centered fields in 3D array
      bcc(m,IBX,k,j,i) = u.bx;
      bcc(m,IBY,k,j,i) = u.by;
      bcc(m,IBZ,k,j,i) = u.bz;

      // A failed cell that kept its primitives keeps their species too, as on the
      // dynamical metric; only one reset to the atmosphere is re-split from scratch.
      // A Lorentz ceiling changes only the normal-frame velocity.  It requires a new
      // conservative core (and hence a new D), but it must not consume the transported
      // total-vs-species residual by repartitioning se/si onto the unchanged internal
      // energy.  Preserve every passive primitive ratio across a purely kinematic repair.
      const bool preserve_scalars_across_kinematic_repair = vceiling_used &&
          !dfloor_used && !efloor_used && !magceiling_used &&
          !c2p_failure && !excised;
      // A state from the entropy inversion rebuilds U, the cell's share of KORAL's
      // update_entropy p2u (u2p.c:2257-2281): the energy equation had already failed
      // there, and a cell that kept its tau would fail again every step while exporting
      // fluxes built from the entropy state.
      const bool reset_conserved = dfloor_used || efloor_used || vceiling_used || magceiling_used || c2p_failure || excised || entropy_inverted;
      // Of that U only tau is new where nothing but the entropy root acted: U2P_ENTROPY
      // replaces the energy equation alone and closes the cell's own D, S_i and B (its
      // velocity is the exact inverse of S_i at the root), so the p2u returns D and S_i
      // up to rounding, rounding that the B^2 terms amplify by ~sigma Gamma^2 where
      // B.v != 0 (1e-4 of S at sigma 1e8, Gamma 50).  D and S_i are kept, and tau and the
      // entropy column are published.  A floor, ceiling or species repair after the root
      // changes the state, and U is rebuilt whole as for any repaired cell.
      const bool entropy_energy_only = entropy_inverted && !dfloor_used && !efloor_used && !vceiling_used && !magceiling_used && !c2p_failure && !excised;

      // reset conserved variables if floor, ceiling, failure, or excision encountered.
      if (reset_conserved) {
        MHDPrim1D w_in;
        w_in.d  = w.d;
        w_in.vx = w.vx;
        w_in.vy = w.vy;
        w_in.vz = w.vz;
        w_in.e  = w.e;
        w_in.bx = u.bx;
        w_in.by = u.by;
        w_in.bz = u.bz;

        HydCons1D u_out;
        SingleP2C_IdealGRMHD(nf, w_in, cell_eos.gamma, u_out);
        if (book_floors && publish_conserved_repairs &&
            (i >= is) && (i <= ie) && (j >= js) && (j <= je) && (k >= ks) && (k <= ke)) {
          const Real weight_volume = step_weight(m)*size.d_view(m).dx1*
                                     size.d_view(m).dx2*size.d_view(m).dx3;
          const int slot = idx & (kLedgerSlots - 1);
          Kokkos::atomic_add(&ledger(slot), weight_volume*(u_out.e - u.e));
          if (!entropy_energy_only) {
            Kokkos::atomic_add(&ledger(kLedgerSlots + slot),
                               weight_volume*(u_out.d - u.d));
          }
        }
        if (publish_conserved_repairs) {
          if (!entropy_energy_only) {
            cons(m,IDN,k,j,i) = u_out.d;
            cons(m,IM1,k,j,i) = u_out.mx;
            cons(m,IM2,k,j,i) = u_out.my;
            cons(m,IM3,k,j,i) = u_out.mz;
          }
          cons(m,IEN,k,j,i) = u_out.e;
        }
        // (needed if there are scalars below; the kept D where only tau is new)
        if (!entropy_energy_only) u.d = u_out.d;
      }

      // The gas entropy.  A state the energy inversion recovered untouched publishes the
      // transported s = (D s)/D.  Any floor, ceiling, failure, excision or entropy root
      // publishes the entropy of the published state with the index U was rebuilt with,
      // as KORAL resets ENTR wherever check_floors_mhd acted (u2p.c:978-982), and where
      // U is republished its D s rides the published D.  A hybrid-deferred cell leaves U
      // to the hybrid recovery, and a communication ghost never writes it.
      if (entropy_slot) {
        const Real s_transported = (scalar_cons_density > 0.0) ?
            cons(m,ent_idx,k,j,i)/scalar_cons_density : 0.0;
        Real s_pub = s_transported;
        if (reset_conserved) {
          s_pub = gr_entropy::SpecificEntropyIdeal(w.d, w.e, cell_eos.gamma);
          // A published state without an entropy (eint = 0, reachable only with every
          // floor at zero) keeps the label it was transported with.
          if (!isfinite(s_pub)) s_pub = s_transported;
          if (!isfinite(s_pub)) s_pub = 0.0;
          // A repair defines the state anew, and the column takes it whole.
          if (publish_conserved_repairs) {
            cons(m,ent_idx,k,j,i) = u.d*s_pub;
          }
        }
        prim(m,ent_idx,k,j,i) = isfinite(s_pub) ? s_pub : 0.0;
      }

      // convert scalars (if any)
      for (int n=nmhd; n<(nmhd+nscal); ++n) {
        Real scalar_cons = cons(m,n,k,j,i);
        if (!isfinite(scalar_cons) || scalar_cons < 0.0) {
          scalar_cons = 0.0;
          if (publish_conserved_repairs) {
            cons(m,n,k,j,i) = 0.0;
          }
        }
        const Real scalar_den =
            preserve_scalars_across_kinematic_repair ? scalar_cons_density : u.d;
        Real scalar = (scalar_den > 0.0) ? (scalar_cons/scalar_den) : 0.0;
        // A failed cell that kept its primitives keeps these ratios with them (KORAL's
        // pp = ppbak is every primitive), and its U is rebuilt from them.
        if (kept_previous) {
          scalar = prim(m,n,k,j,i);
        }
        prim(m,n,k,j,i) = isfinite(scalar) ? scalar : 0.0;
        if (publish_conserved_repairs && (preserve_scalars_across_kinematic_repair || kept_previous)) {
          cons(m,n,k,j,i) = (u.d > 0.0) ? u.d*prim(m,n,k,j,i) : 0.0;
        }
      }

    }
  };
  // The light-weight launch passes the functor as a kernel argument.
  static_assert(sizeof(c2p_cell) < 4096,
                "grmhd_c2p functor exceeds the light-weight argument limit");
  Kokkos::parallel_for("grmhd_c2p", C2PLaunchPolicy<true>(
  Kokkos::RangePolicy<DevExeSpace, Kokkos::LaunchBounds<256, kC2PMinBlocksPerSM>>
      (DevExeSpace(), 0, nwork*nvisit)), c2p_cell);

  // Every cell of every block now holds primitives a recovery on this mesh wrote, which
  // is what a failed inversion keeps from here on.
  if (recovers_main_state && !lat_enabled && !skip_cells && user_boundary_face < 0 && il == 0 && jl == 0 && kl == 0 && iu == prim.extent_int(4) - 1 && ju == prim.extent_int(3) - 1 && ku == prim.extent_int(2) - 1) {
    pmy_pack->pmhd->w0_topology_version = pmy_pack->pmesh->topology_version;
  }

  // The transitional cells grmhd_c2p passed over.  Same cells and blocks as that launch.

  // Before this process's first RKUpdate (the setup and restart recoveries) there is no
  // step weight and the caller puts the ecounter back: book at once.  After it the
  // bookings stay on the device until the history row or the checkpoint asks.
  if (book_floors && !step_weight_seen_) FlushFloorLedger();

  // These counters are consumed only by event-log output. Avoid a device-to-host
  // synchronization after every C2P launch when no event log is configured.
  if (track_event_counters) {
    event_counters_.template modify<DevExeSpace>();
    event_counters_.template sync<HostMemSpace>();
    if constexpr (only_testfloors) {
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

// One specialization of ConsToPrimImpl, in the order of its template parameters:
// only_testfloors, apply_magnetization_ceilings.
#define INSTANTIATE_IDEAL_GRMHD_C2P(A, B) \
template void IdealGRMHD::ConsToPrimImpl<A, B>( \
    DvceArray5D<Real> &cons_in, const DvceFaceFld4D<Real> &b, DvceArray5D<Real> &prim, \
    DvceArray5D<Real> &bcc_in, const int il, const int iu, const int jl, const int ju, \
    const int kl, const int ku, const bool track_event_counters);

#endif  // EOS_IDEAL_GRMHD_C2P_IMPL_HPP_
