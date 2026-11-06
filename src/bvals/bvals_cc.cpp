//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file bvals_cc.cpp
//! \brief functions to pack/send and recv/unpack boundary values for cell-centered (CC)
//! Mesh variables.
//! Prolongation of CC variables  occurs in ProlongateCC() function called from task list

#include <cstdlib>
#include <algorithm>
#include <iostream>
#include <limits>
#include <utility>

#include "athena.hpp"
#include "globals.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "bvals.hpp"
#include "bvals/narrow_blocks.hpp"

namespace {
//! \brief SSPRK2 dense-output polynomial.  For RK2, ye is the stage-1 forward-Euler
//! endpoint over the same interval; the weights (1-th, th(1-th), th^2) are non-negative
//! and sum to one, so the value is a convex combination of the three states.
KOKKOS_INLINE_FUNCTION
Real LATDensePolyCC(Real y0, Real y1, Real ye, Real theta) {
  return (static_cast<Real>(1.0) - theta)*y0 +
         theta*(static_cast<Real>(1.0) - theta)*ye +
         theta*theta*y1;
}

//! \brief One component of a LAT history-interpolated ghost state: the dense polynomial
//! where it is admissible, the linear chord where it is not.
//!
//! The chord and the dense polynomial are each a convex combination of the same three
//! states, so each of them separately lies in every convex set the three states lie in
//! -- including M1's closure cone E >= |F|.  Their COMPONENT-WISE mixture does not: it
//! lies in the axis-aligned box spanned by the endpoints, and that box is strictly
//! larger than the cone (E0=E1=1, F0=(1,0,0), F1=(0,1,0) with only Fy demoted gives
//! |F|=1.118 > E=1 at theta=0.5).  The fallback therefore has to be decided once per
//! CELL, over every component, and not per component: a neighbour handed E < |F| is
//! rescaled in place by RadiationM1::CalcClosure's apply_floor, i.e. a non-conservative
//! clip on exactly the mixed-cadence coarse/fine faces.  With more than one species this
//! couples the species groups, which is stricter than the per-cone requirement but never
//! wrong.  MHD passes limit_dense_to_endpoints = false and returns before the scan, so
//! it keeps the unbounded polynomial that U shares with face-centered B.
//!
//! The stage-1 snapshot yev may be stored on the owned cells only
//! (MeshBoundaryValuesCC::LATStage1OwnedOnly).  (ek,ej,ei) is its origin: (ks,js,is)
//! for an owned-cell snapshot, (0,0,0) for a ghost-extended one.  The start register
//! y0v may be owned-only on the same condition; (sk,sj,si) is its origin.
KOKKOS_INLINE_FUNCTION
Real LATInterpolateCC(const DvceArray5D<Real> &y0v, const DvceArray5D<Real> &y1v,
                      const DvceArray5D<Real> &yev, const int sk, const int sj,
                      const int si, const int ek, const int ej, const int ei,
                      int m, int v, int k, int j, int i, int nvar, Real theta,
                      bool use_dense, bool limit_dense_to_endpoints, Real tol) {
  const Real y0 = y0v(m,v,k-sk,j-sj,i-si);
  const Real y1 = y1v(m,v,k,j,i);
  const Real linear = y0 + theta*(y1 - y0);
  if (!use_dense) return linear;

  const Real dense = LATDensePolyCC(y0, y1, yev(m,v,k-ek,j-ej,i-ei), theta);
  if (!limit_dense_to_endpoints) return dense;

  for (int w=0; w<nvar; ++w) {
    const Real a0 = y0v(m,w,k-sk,j-sj,i-si);
    const Real a1 = y1v(m,w,k,j,i);
    const Real ad = LATDensePolyCC(a0, a1, yev(m,w,k-ek,j-ej,i-ei), theta);
    const Real lo = fmin(a0, a1);
    const Real hi = fmax(a0, a1);
    const Real scale = fmax(static_cast<Real>(1.0), fmax(fabs(lo), fabs(hi)));
    if (!(ad == ad) || ad < lo - tol*scale || ad > hi + tol*scale) {
      return linear;
    }
  }
  return dense;
}
//! \brief State a pending faster sender publishes into a slower bin's corrector refresh.
//!
//! Under the union predictor the slower bin's second stage runs before the faster bin
//! has moved past its own first substep, so the only registers a faster neighbour holds
//! are its window start y0 and its forward-Euler endpoint y1 at t0 + f*dt.  Publishing
//! y0 at the slower stage's time t0 + F*dt hands a second-order stage an argument that
//! is O(F*dt) stale, which caps the coupled scheme at first order in time on every
//! inflow face of the slower block (1D linear wave: coarse-cell error ~ dt^2 per
//! window while fine cells stay ~ dt^3).  The forward extrapolation
//! y0 + theta*(y1 - y0) with theta = F/f is the Euler predictor over the slower step
//! and has the O(dt^2) accuracy Heun's second stage needs.  It is applied to every
//! transported system alike (conserved fluid, radiation moments, force-free tails,
//! whose leading component is signed).
//!
//! It is not a convex combination.  y0 + theta*(y1 - y0) = y0 + F*dt*L(y0) is the
//! sender's forward-Euler predictor stretched from its own step to the slower one, and no
//! FOFC ever sees it.  Each cell therefore takes LATPendingBound: its positive components
//! must stay positive, and its per-mass columns cross as the extrapolated D times the
//! endpoint's ratio, as a prolongation carries them, so the ghost's specific entropy and
//! abundances are ones the sender has (their own extrapolation, (D s)_ext/D_ext across a
//! contact, is a value neither side has).  A cell whose positive component would not stay
//! positive publishes the sender's own Euler endpoint y1 in every component, first order
//! in time there.  The force-free drift image is limited by its sender instead, which
//! stores a pre-image that keeps the extrapolation inside the drift ball
//! (MHD::LimitHybridForceFreePendingExtrapolation), because a receiver could only peg it.
KOKKOS_INLINE_FUNCTION
Real LATExtrapolateCC(const DvceArray5D<Real> &y0v, const DvceArray5D<Real> &y1v,
                      const int sk, const int sj, const int si,
                      int m, int v, int k, int j, int i, Real theta,
                      const LATPendingBound &bound) {
  if (theta > 1.0) {
    for (int q = 0; q < bound.nvar; ++q) {
      const int w = bound.var[q];
      const Real w0 = y0v(m,w,k-sk,j-sj,i-si);
      if (!(w0 + theta*(y1v(m,w,k,j,i) - w0) > 0.0)) return y1v(m,v,k,j,i);
    }
    if (v >= bound.ratio_begin && v < bound.ratio_end) {
      const Real d0 = y0v(m,bound.idn,k-sk,j-sj,i-si);
      const Real d1 = y1v(m,bound.idn,k,j,i);
      const Real d_ext = d0 + theta*(d1 - d0);
      return (d1 > 0.0 && d_ext > 0.0) ? d_ext*(y1v(m,v,k,j,i)/d1) : y1v(m,v,k,j,i);
    }
  }
  const Real y0 = y0v(m,v,k-sk,j-sj,i-si);
  return y0 + theta*(y1v(m,v,k,j,i) - y0);
}

} // namespace

//----------------------------------------------------------------------------------------
// BValCC constructor:

MeshBoundaryValuesCC::MeshBoundaryValuesCC(MeshBlockPack *pp, ParameterInput *pin,
                                           bool z4c) :
  MeshBoundaryValues(pp, pin, z4c) {
#if MPI_PARALLEL_ENABLED
  rank_packed_lat_capable_ = true;
#endif
}

//----------------------------------------------------------------------------------------
//! \fn void MeshBoundaryValuesCC::PackAndSendCC()
//! \brief Pack cell-centered variables into boundary buffers and send to neighbors.
//!
//! This routine packs ALL the buffers on ALL the faces, edges, and corners simultaneously
//! for ALL the MeshBlocks. This reduces the number of kernel launches when there are a
//! large number of MeshBlocks per MPI rank. Buffer data are then sent (via MPI) or copied
//! directly for periodic or block boundaries.
//!
//! Input arrays must be 5D Kokkos View dimensioned (nmb, nvar, nx3, nx2, nx1)
//! 5D Kokkos View of coarsened (restricted) array data also required with SMR/AMR

TaskStatus MeshBoundaryValuesCC::PackAndSendCC(DvceArray5D<Real> &a,
                                               DvceArray5D<Real> &ca,
                                               DvceArray5D<Real> *a_start,
                                               Real target_time,
                                               DvceArray5D<Real> *ca_start,
                                               DvceArray5D<Real> *a_stage1,
                                               DvceArray5D<Real> *ca_stage1,
                                               DvceArray1D<int> *a_stage1_valid,
                                               bool limit_dense_to_endpoints,
                                               const LATPendingBound &pending_bound) {
  // A send-clear latched by LatchSendClear completes before the pack below rewrites the
  // aggregate send buffer those sends read, and before their requests are restarted.
  DrainLatchedSendClear();
  // create local references for variables in kernel
  int nmb = pmy_pack->nmb_thispack;
  int nnghbr = pmy_pack->pmb->nnghbr;
  int nvar = a.extent_int(1);  // TODO(@user): 2nd index from L of in array must be NVAR
  DevExeSpace pack_exec;
  const bool lat_enabled = pmy_pack->lat_active_mask_enabled;
  const bool lat_time_interp = lat_enabled && (a_start != nullptr);
  const bool lat_coarse_time_interp = lat_enabled && (ca_start != nullptr);
  const bool lat_dense_interp = lat_time_interp && (a_stage1 != nullptr);
  const bool lat_coarse_dense_interp =
      lat_coarse_time_interp && (ca_stage1 != nullptr);
  const bool lat_cross_level_dense_interp = (ca_stage1 != nullptr);
  const bool lat_dense_validity_required = (a_stage1_valid != nullptr);
  auto a_start_view = (a_start != nullptr) ? *a_start : a;
  auto ca_start_view = (ca_start != nullptr) ? *ca_start : ca;
  auto a_stage1_view = (a_stage1 != nullptr) ? *a_stage1 : a;
  auto ca_stage1_view = (ca_stage1 != nullptr) ? *ca_stage1 : ca;
  DvceArray1D<int> a_stage1_valid_view;
  if (a_stage1_valid != nullptr) {
    a_stage1_valid_view = *a_stage1_valid;
  }
  const Real dense_tol = static_cast<Real>(64.0)*std::numeric_limits<Real>::epsilon();
  // A fine snapshot shorter than its state array holds the owned cells only
  // (LATStage1OwnedOnly); its origin is then the first owned cell of each extended axis.
  // The coarse snapshot is always ghost-extended, origin (0,0,0).
  auto &snap_indcs = pmy_pack->pmesh->mb_indcs;
  const int s1_ko = (a_stage1_view.extent_int(2) == a.extent_int(2)) ? 0 : snap_indcs.ks;
  const int s1_jo = (a_stage1_view.extent_int(3) == a.extent_int(3)) ? 0 : snap_indcs.js;
  const int s1_io = (a_stage1_view.extent_int(4) == a.extent_int(4)) ? 0 : snap_indcs.is;
  // The fine start register may hold a centred band of the block on the same condition,
  // the owned cells (the start registers u1, the force-free sidecar's ff_xfer1) or the
  // owned cells plus g ghost layers (u1 of the dynamical-GR FOFC); it is read on the same
  // send ranges, and its origin ng - g is half the extent it lacks (CentredBand).
  const int st_ko = (a.extent_int(2) - a_start_view.extent_int(2))/2;
  const int st_jo = (a.extent_int(3) - a_start_view.extent_int(3))/2;
  const int st_io = (a.extent_int(4) - a_start_view.extent_int(4))/2;
  const LATPendingBound pbound = pending_bound;

#if MPI_PARALLEL_ENABLED
  // InitRecv normally selects the mode, but LAT can have sender-only ranks that do not
  // post receives.  Select it again here so ClearSend waits on the requests posted below.
  rank_packed_bvals_active_ = UseRankPackedVars();
  rank_packed_lat_active_ = rank_packed_bvals_active_ && lat_enabled &&
                            rank_packed_lat_capable_;
  if (rank_packed_bvals_active_) {
    EnsureRankPackedVarMetadata(nvar);
    if (rank_packed_lat_active_) PrepareLATRankPackedVarLayout();

    int my_rank = global_variable::my_rank;
    auto &nghbr = pmy_pack->pmb->nghbr;
    auto &mbgid = pmy_pack->pmb->mb_gid;
    auto &mblev = pmy_pack->pmb->mb_lev;
    auto sbuf = sendbuf_device;
    auto rbuf = recvbuf_device;
    auto &is_z4c = is_z4c_;
    auto &multilevel = pmy_pack->pmesh->multilevel;
    auto aggsbuf = rank_sendbuf_vars_;
    auto sendoff = rank_packed_lat_active_ ?
        lat_var_layouts_[lat_var_layout_index_].send_offsets : send_agg_offset_;
    const bool same_level_only = same_level_vars_only;
    const bool local_xlev = stub_recv_same_vars && !same_level_vars_only;
    auto xlevbuf = local_xlev_vars_;
    auto xlevoff = local_xlev_offset_;
    auto lat_active = pmy_pack->lat_active_mb.d_view;
    auto lat_send = pmy_pack->lat_send_nghbr.d_view;
    auto lat_boundary_send_edges = pmy_pack->lat_boundary_send_edges.d_view;
    auto lat_t0 = pmy_pack->lat_time_start.d_view;
    auto lat_t1 = pmy_pack->lat_time_end.d_view;
    auto lat_step_factor = pmy_pack->lat_step_factor.d_view;
    auto lat_nghbr_factor = pmy_pack->lat_nghbr_factor.d_view;
    const int lat_pending_below_factor = pmy_pack->lat_union_pending_below_factor;
    const bool lat_union_refresh = lat_enabled && pmy_pack->lat_union_refresh;
    const Real lat_union_start = pmy_pack->lat_union_start;
    const Real lat_union_fine_dt = pmy_pack->lat_union_fine_dt;

    const int nmn = lat_enabled ? pmy_pack->lat_nboundary_send_edges_thispack :
                                  nmb*nnghbr;
    // The data for a same-rank same-level neighbour is copied straight into its ghost
    // cells.  A narrow MeshBlock's pack for a finer neighbour reads its own ghost cells
    // (NarrowMeshBlocks), so within one launch it would read them before or after such a
    // copy as the teams happen to run -- an order the rank layout sets and a GPU does not
    // fix -- whereas cells from another rank always arrive after the pack.  On such a
    // mesh the copies run in a second launch, once every pack has read its cells.
    const int nlaunch = NarrowMeshBlocks(pmy_pack->pmesh) ? 2 : 1;
    if (nmn > 0) {
      for (int launch=0; launch<nlaunch; ++launch) {
        const bool packs = (launch == 0);
        const bool copies = (launch == nlaunch - 1);
        Kokkos::TeamPolicy<> policy(pack_exec, nmn, Kokkos::AUTO);
        Kokkos::parallel_for("SendBuffRankPackedCC", athenak_lw(policy),
        KOKKOS_LAMBDA(TeamMember_t tmember) {
          const int edge = lat_enabled ? lat_boundary_send_edges(tmember.league_rank()) :
                                         tmember.league_rank();
          const int m = edge/nnghbr;
          const int n = edge - m*nnghbr;

          if (lat_enabled && lat_send(m,n) == 0) return;

          // Both ratios exist only for a pack that interpolates in time.  A pack with no
          // start register (the FOFC and LAT-correction masks, the force-free weight and
          // owner-primitive images) has no time axis: its target_time is the 0.0
          // default, and a ratio formed from it is negative for every sender whose span
          // starts after t = 0.
          const bool lat_interpolates = lat_time_interp || lat_coarse_time_interp;
          // A pending sender's span is its own predictor step [t0, t0 + f*dt] and the
          // refresh target is the slower bin's endpoint, so its ratio is F/f >= 1; in the
          // union refresh the target is the receiver's own endpoint (LATSendEdgeSpan).
          const LATSendSpan lat_span = LATSendEdgeSpan(
              lat_enabled, lat_interpolates, lat_enabled ? lat_active(m) : 1,
              lat_enabled ? lat_step_factor(m) : 1, lat_pending_below_factor,
              lat_enabled ? lat_t0(m) : 0.0, lat_enabled ? lat_t1(m) : 0.0,
              target_time, lat_union_refresh, lat_union_start, lat_union_fine_dt,
              (lat_enabled && lat_union_refresh) ? lat_nghbr_factor(m,n) : 0);
          const bool lat_pending_start = lat_span.pending;
          const bool lat_history_interp = lat_span.history;
          const Real lat_extrap = lat_span.extrap;
          Real lat_theta = lat_span.theta_raw;
          if (lat_interpolates && lat_history_interp) {
            // theta > 1 is REACHABLE and the clamp below is load-bearing, not a
            // diagnostic: on the factor-by-factor path only the active bin's block
            // times are rearmed, so a faster neighbour (factor f < F) still carries its
            // previous span [t-f, t] while the factor-F bin sends for t + F*dt_fine,
            // i.e. theta = (F+f)/f >= 3.  Clamping then publishes that neighbour's
            // end-of-span state, which is the most recent state that exists for it.
            // Only theta < 0 -- a target_time preceding a sender's own span start --
            // means broken block times.  Every target and every span start is a value
            // of the one accumulated mesh time (t^n, or t^n plus a nonnegative stage
            // offset) and a start is never later than the target it serves, so
            // theta >= 0 holds exactly; the tolerance is slack, not a rounding bound.
            KOKKOS_ASSERT(lat_theta > -1.0e-10);
            lat_theta = fmax(static_cast<Real>(0.0),
                             fmin(static_cast<Real>(1.0), lat_theta));
          }

          if (nghbr.d_view(m,n).gid >= 0) {
            const int lev = nghbr.d_view(m,n).lev;
            const int mlev = mblev.d_view(m);
            if (same_level_only && lev != mlev) return;
            int il, iu, jl, ju, kl, ku;
            if (lev < mlev) {
              il = sbuf[n].icoar[0].bis; iu = sbuf[n].icoar[0].bie;
              jl = sbuf[n].icoar[0].bjs; ju = sbuf[n].icoar[0].bje;
              kl = sbuf[n].icoar[0].bks; ku = sbuf[n].icoar[0].bke;
            } else if (lev == mlev) {
              il = sbuf[n].isame[0].bis; iu = sbuf[n].isame[0].bie;
              jl = sbuf[n].isame[0].bjs; ju = sbuf[n].isame[0].bje;
              kl = sbuf[n].isame[0].bks; ku = sbuf[n].isame[0].bke;
            } else {
              il = sbuf[n].ifine[0].bis; iu = sbuf[n].ifine[0].bie;
              jl = sbuf[n].ifine[0].bjs; ju = sbuf[n].ifine[0].bje;
              kl = sbuf[n].ifine[0].bks; ku = sbuf[n].ifine[0].bke;
            }
            const int ni = iu - il + 1;
            const int nj = ju - jl + 1;
            const int nk = ku - kl + 1;
            const int ncells = ni*nj*nk;
            const int ntot = nvar*ncells;
            const int dm = nghbr.d_view(m,n).gid - mbgid.d_view(0);
            const int dn = nghbr.d_view(m,n).dest;
            const bool same_rank = (nghbr.d_view(m,n).rank == my_rank);

            if (same_rank && lev == mlev) {
              if (!copies) return;
              const int dil = rbuf[dn].isame[0].bis;
              const int djl = rbuf[dn].isame[0].bjs;
              const int dkl = rbuf[dn].isame[0].bks;
              Kokkos::parallel_for(Kokkos::TeamVectorRange(tmember, ntot),
              [&](const int idx) {
                const int v = idx / ncells;
                const int r = idx - v*ncells;
                const int kk = r / (ni*nj);
                const int r2 = r - kk*(ni*nj);
                const int jj = r2 / ni;
                const int ii = r2 - jj*ni;
                const int k = kl + kk;
                const int j = jl + jj;
                const int i = il + ii;
                const bool pending_start = lat_time_interp && lat_pending_start;
                Real val = pending_start ?
                    LATExtrapolateCC(a_start_view, a, st_ko, st_jo, st_io, m, v, k, j, i,
                                     lat_extrap, pbound) :
                    a(m,v,k,j,i);
                if (lat_time_interp && lat_history_interp) {
                  const bool use_dense = lat_dense_interp &&
                      (!lat_dense_validity_required || a_stage1_valid_view(m) != 0);
                  val = LATInterpolateCC(a_start_view, a, a_stage1_view,
                                         st_ko, st_jo, st_io, s1_ko, s1_jo, s1_io,
                                         m, v, k, j, i, nvar,
                                         lat_theta, use_dense, limit_dense_to_endpoints,
                                         dense_tol);
                }
                a(dm, v, dkl+kk, djl+jj, dil+ii) = val;
              });
            } else if (packs) {
              // A same-rank payload here is a cross-level one.
              const int base = !same_rank ? sendoff(m*nnghbr + n) :
                               local_xlev ? xlevoff(dm*nnghbr + dn) : 0;
              Kokkos::parallel_for(Kokkos::TeamVectorRange(tmember, ntot),
              [&](const int idx) {
                const int v = idx / ncells;
                const int r = idx - v*ncells;
                const int kk = r / (ni*nj);
                const int r2 = r - kk*(ni*nj);
                const int jj = r2 / ni;
                const int ii = r2 - jj*ni;
                const int k = kl + kk;
                const int j = jl + jj;
                const int i = il + ii;
                Real val;
                if (lev < mlev) {
                  const bool pending_start = lat_coarse_time_interp && lat_pending_start;
                  val = pending_start ?
                      LATExtrapolateCC(ca_start_view, ca, 0, 0, 0, m, v, k, j, i,
                                       lat_extrap, pbound) :
                      ca(m,v,k,j,i);
                  if (lat_coarse_time_interp && lat_history_interp) {
                    const bool use_dense = lat_coarse_dense_interp &&
                        (!lat_dense_validity_required || a_stage1_valid_view(m) != 0);
                    val = LATInterpolateCC(ca_start_view, ca, ca_stage1_view,
                                           0, 0, 0, 0, 0, 0, m, v, k, j, i, nvar,
                                           lat_theta, use_dense, limit_dense_to_endpoints,
                                           dense_tol);
                  }
                } else {
                  const bool pending_start = lat_time_interp && lat_pending_start;
                  val = pending_start ?
                      LATExtrapolateCC(a_start_view, a, st_ko, st_jo, st_io, m, v, k, j,
                                       i, lat_extrap, pbound) :
                      a(m,v,k,j,i);
                  if (lat_time_interp && lat_history_interp) {
                    const bool use_dense = lat_dense_interp &&
                        (lev == mlev || lat_cross_level_dense_interp) &&
                        (!lat_dense_validity_required || a_stage1_valid_view(m) != 0);
                    val = LATInterpolateCC(a_start_view, a, a_stage1_view,
                                           st_ko, st_jo, st_io, s1_ko, s1_jo, s1_io,
                                           m, v, k, j, i, nvar,
                                           lat_theta, use_dense, limit_dense_to_endpoints,
                                           dense_tol);
                  }
                }
                if (!same_rank) {
                  aggsbuf(base + idx) = val;
                } else if (local_xlev) {
                  xlevbuf(base + idx) = val;
                } else {
                  rbuf[dn].vars(dm, idx) = val;
                }
              });
            }
          }
          tmember.team_barrier();
        });
      }

      if (is_z4c && multilevel) {
        Kokkos::TeamPolicy<> z4c_policy(pack_exec, nmn*nvar, Kokkos::AUTO);
        Kokkos::parallel_for("SendBuffRankPackedCCZ4c", z4c_policy,
        KOKKOS_LAMBDA(TeamMember_t tmember) {
          const int mn = tmember.league_rank()/nvar;
          const int edge = lat_enabled ? lat_boundary_send_edges(mn) : mn;
          const int m = edge/nnghbr;
          const int n = edge - m*nnghbr;
          const int v = tmember.league_rank() - mn*nvar;

          if (lat_enabled && lat_send(m,n) == 0) return;
          const bool lat_pending_start = lat_enabled && lat_active(m) == 0 &&
              lat_pending_below_factor > 1 &&
              lat_step_factor(m) < lat_pending_below_factor;
          if (nghbr.d_view(m,n).gid >= 0 &&
              nghbr.d_view(m,n).lev == mblev.d_view(m)) {
            const int il = sbuf[n].isame_z4c.bis;
            const int iu = sbuf[n].isame_z4c.bie;
            const int jl = sbuf[n].isame_z4c.bjs;
            const int ju = sbuf[n].isame_z4c.bje;
            const int kl = sbuf[n].isame_z4c.bks;
            const int ku = sbuf[n].isame_z4c.bke;
            const int ni = iu - il + 1;
            const int nj = ju - jl + 1;
            const int nk = ku - kl + 1;
            const int ncells = ni*nj*nk;
            const int ndat = nvar*sbuf[n].isame_ndat;
            const int dm = nghbr.d_view(m,n).gid - mbgid.d_view(0);
            const int dn = nghbr.d_view(m,n).dest;
            const bool same_rank = (nghbr.d_view(m,n).rank == my_rank);
            const int base = same_rank ? 0 : sendoff(m*nnghbr + n);
            Kokkos::parallel_for(Kokkos::TeamVectorRange(tmember, ncells),
            [&](const int idx) {
              const int kk = idx / (ni*nj);
              const int r2 = idx - kk*(ni*nj);
              const int jj = r2 / ni;
              const int ii = r2 - jj*ni;
              const int bi = ndat + idx + ncells*v;
              const bool pending_start = lat_coarse_time_interp && lat_pending_start;
              const Real val = pending_start ?
                  ca_start_view(m,v,kl+kk,jl+jj,il+ii) : ca(m,v,kl+kk,jl+jj,il+ii);
              if (same_rank) {
                rbuf[dn].vars(dm, bi) = val;
              } else {
                aggsbuf(base + bi) = val;
              }
            });
          }
          tmember.team_barrier();
        });
      }
    }

    if (global_variable::nranks == 1) return TaskStatus::complete;
    pack_exec.fence();
    // Persistent sends: bound once to this layout's (peer, offset, count) tuples, then
    // restarted each step.  See EnsurePersistentVarReqs in bvals.cpp for why this
    // matters with UCX_RCACHE_ENABLE=n.
    const int lay = rank_packed_lat_active_ ? lat_var_layout_index_ : -1;
    const auto &send_msgs = (lay >= 0) ?
        lat_var_layouts_[lay].send_msgs : send_var_msgs_;
    auto &send_reqs = VarReqs(true, lay);
    EnsurePersistentVarReqs(send_msgs, rank_sendbuf_vars_.data(), true, &send_reqs);
    bool no_errors = true;
    if (!send_reqs.empty()) {
      no_errors = (MPI_Startall(static_cast<int>(send_reqs.size()),
                                send_reqs.data()) == MPI_SUCCESS);
    }
    if (!no_errors) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
         << std::endl << "MPI error in starting rank-packed sends" << std::endl;
      std::exit(EXIT_FAILURE);
    }
    lat_send_reqs_layout_ = lay;
    send_var_reqs_started_ = true;
    return TaskStatus::complete;
  }
#endif

  {int my_rank = global_variable::my_rank;
  auto &nghbr = pmy_pack->pmb->nghbr;
  auto &mbgid = pmy_pack->pmb->mb_gid;
  auto &mblev = pmy_pack->pmb->mb_lev;
  auto lat_active = pmy_pack->lat_active_mb.d_view;
  auto lat_send = pmy_pack->lat_send_nghbr.d_view;
  auto lat_boundary_send_edges = pmy_pack->lat_boundary_send_edges.d_view;
  auto lat_t0 = pmy_pack->lat_time_start.d_view;
  auto lat_t1 = pmy_pack->lat_time_end.d_view;
  auto lat_step_factor = pmy_pack->lat_step_factor.d_view;
  auto lat_nghbr_factor = pmy_pack->lat_nghbr_factor.d_view;
  const int lat_pending_below_factor = pmy_pack->lat_union_pending_below_factor;
  const bool lat_union_refresh = lat_enabled && pmy_pack->lat_union_refresh;
  const Real lat_union_start = pmy_pack->lat_union_start;
  const Real lat_union_fine_dt = pmy_pack->lat_union_fine_dt;
  auto sbuf = sendbuf_device;
  auto rbuf = recvbuf_device;
  auto &is_z4c = is_z4c_;
  auto &multilevel = pmy_pack->pmesh->multilevel;
  const bool same_level_only = same_level_vars_only;
  // One team handles one (MeshBlock,neighbor) buffer and flattens all
  // (variable,k,j,i) elements across the team.  This keeps the same staged-buffer
  // behavior as the old path while giving small edge/corner buffers more useful work.
  const int nmn = lat_enabled ? pmy_pack->lat_nboundary_send_edges_thispack :
                                nmb*nnghbr;
  if (nmn <= 0) return TaskStatus::complete;
  Kokkos::TeamPolicy<> policy(pack_exec, nmn, Kokkos::AUTO);
  Kokkos::parallel_for("SendBuff", athenak_lw(policy),
      KOKKOS_LAMBDA(TeamMember_t tmember) {
    const int edge = lat_enabled ? lat_boundary_send_edges(tmember.league_rank()) :
                                   tmember.league_rank();
    const int m = edge/nnghbr;
    const int n = edge - m*nnghbr;
    if (lat_enabled && lat_send(m,n) == 0) return;

    // A pending sender's span is its own predictor step [t0, t0 + f*dt] and the
    // refresh target is the slower bin's endpoint, so its ratio is F/f >= 1; in the
    // union refresh the target is the receiver's own endpoint (LATSendEdgeSpan).
    const LATSendSpan lat_span = LATSendEdgeSpan(
        lat_enabled, true, lat_enabled ? lat_active(m) : 1,
        lat_enabled ? lat_step_factor(m) : 1, lat_pending_below_factor,
        lat_enabled ? lat_t0(m) : 0.0, lat_enabled ? lat_t1(m) : 0.0, target_time,
        lat_union_refresh, lat_union_start, lat_union_fine_dt,
        (lat_enabled && lat_union_refresh) ? lat_nghbr_factor(m,n) : 0);
    const bool lat_pending_start = lat_span.pending;
    const bool lat_history_interp = lat_span.history;
    const Real lat_extrap = lat_span.extrap;
    const Real lat_theta = lat_span.theta;

    // only load buffers when neighbor exists
    if (nghbr.d_view(m,n).gid >= 0) {
      const int lev = nghbr.d_view(m,n).lev;
      const int mlev = mblev.d_view(m);
      if (same_level_only && lev != mlev) return;
      // if neighbor is at coarser level, use coar indices to pack buffer
      int il, iu, jl, ju, kl, ku;
      if (lev < mlev) {
        il = sbuf[n].icoar[0].bis;
        iu = sbuf[n].icoar[0].bie;
        jl = sbuf[n].icoar[0].bjs;
        ju = sbuf[n].icoar[0].bje;
        kl = sbuf[n].icoar[0].bks;
        ku = sbuf[n].icoar[0].bke;
      // if neighbor is at same level, use same indices to pack buffer
      } else if (lev == mlev) {
        il = sbuf[n].isame[0].bis;
        iu = sbuf[n].isame[0].bie;
        jl = sbuf[n].isame[0].bjs;
        ju = sbuf[n].isame[0].bje;
        kl = sbuf[n].isame[0].bks;
        ku = sbuf[n].isame[0].bke;
      // if neighbor is at finer level, use fine indices to pack buffer
      } else {
        il = sbuf[n].ifine[0].bis;
        iu = sbuf[n].ifine[0].bie;
        jl = sbuf[n].ifine[0].bjs;
        ju = sbuf[n].ifine[0].bje;
        kl = sbuf[n].ifine[0].bks;
        ku = sbuf[n].ifine[0].bke;
      }
      int ni = iu - il + 1;
      int nj = ju - jl + 1;
      int nk = ku - kl + 1;
      const int ncells = ni*nj*nk;
      const int ntot = nvar*ncells;

      // indices of recv'ing (destination) MB and buffer: MB IDs are stored sequentially
      // in MeshBlockPacks, so array index equals (target_id - first_id)
      int dm = nghbr.d_view(m,n).gid - mbgid.d_view(0);
      int dn = nghbr.d_view(m,n).dest;
      const bool same_rank = (nghbr.d_view(m,n).rank == my_rank);
      const bool coarser = (lev < mlev);

      Kokkos::parallel_for(Kokkos::TeamVectorRange(tmember, ntot), [&](const int idx) {
        const int v = idx / ncells;
        const int r = idx - v*ncells;
        const int kk = r / (ni*nj);
        const int r2 = r - kk*(ni*nj);
        const int jj = r2 / ni;
        const int ii = r2 - jj*ni;
        const int k = kl + kk;
        const int j = jl + jj;
        const int i = il + ii;
        Real val;
        if (coarser) {
          const bool pending_start = lat_coarse_time_interp && lat_pending_start;
          val = pending_start ?
                    LATExtrapolateCC(ca_start_view, ca, 0, 0, 0, m, v, k, j, i,
                                     lat_extrap, pbound) :
                    ca(m,v,k,j,i);
          if (lat_coarse_time_interp && lat_history_interp) {
            const bool use_dense = lat_coarse_dense_interp &&
                (!lat_dense_validity_required || a_stage1_valid_view(m) != 0);
            val = LATInterpolateCC(ca_start_view, ca, ca_stage1_view,
                                   0, 0, 0, 0, 0, 0, m, v, k, j, i, nvar,
                                   lat_theta, use_dense, limit_dense_to_endpoints,
                                   dense_tol);
          }
        } else {
          const bool pending_start = lat_time_interp && lat_pending_start;
          val = pending_start ?
                    LATExtrapolateCC(a_start_view, a, st_ko, st_jo, st_io, m, v, k, j, i,
                                     lat_extrap, pbound) :
                    a(m,v,k,j,i);
          if (lat_time_interp && lat_history_interp) {
            const bool use_dense = lat_dense_interp &&
                (lev == mlev || lat_cross_level_dense_interp) &&
                (!lat_dense_validity_required || a_stage1_valid_view(m) != 0);
            val = LATInterpolateCC(a_start_view, a, a_stage1_view,
                                   st_ko, st_jo, st_io, s1_ko, s1_jo, s1_io,
                                   m, v, k, j, i, nvar,
                                   lat_theta, use_dense, limit_dense_to_endpoints,
                                   dense_tol);
          }
        }
        if (same_rank) {
          rbuf[dn].vars(dm, idx) = val;
        } else {
          sbuf[n].vars(m, idx) = val;
        }
      });
    } // end if-neighbor-exists block
    tmember.team_barrier();
  }); // end par_for_outer

  if (is_z4c && multilevel) {
    Kokkos::TeamPolicy<> z4c_policy(pack_exec, nmn*nvar, Kokkos::AUTO);
    Kokkos::parallel_for("SendBuff", z4c_policy, KOKKOS_LAMBDA(TeamMember_t tmember) {
      const int mn = tmember.league_rank()/nvar;
      const int edge = lat_enabled ? lat_boundary_send_edges(mn) : mn;
      const int m = edge/nnghbr;
      const int n = edge - m*nnghbr;
      const int v = tmember.league_rank() - mn*nvar;
      if (lat_enabled && lat_send(m,n) == 0) return;
      const bool lat_pending_start = lat_enabled && lat_active(m) == 0 &&
          lat_pending_below_factor > 1 &&
          lat_step_factor(m) < lat_pending_below_factor;

      // only load buffers when neighbor exists
      if (nghbr.d_view(m,n).gid >= 0) {
        int il, iu, jl, ju, kl, ku;
        // If neighbor is at same level and data is for Z4c module, append data from
        // coarse array for higher-order prolongation
        if (nghbr.d_view(m,n).lev == mblev.d_view(m)) {
          il = sbuf[n].isame_z4c.bis;
          iu = sbuf[n].isame_z4c.bie;
          jl = sbuf[n].isame_z4c.bjs;
          ju = sbuf[n].isame_z4c.bje;
          kl = sbuf[n].isame_z4c.bks;
          ku = sbuf[n].isame_z4c.bke;
          int ni = iu - il + 1;
          int nj = ju - jl + 1;
          int nk = ku - kl + 1;
          int nkj  = nk*nj;
          int ndat = nvar*sbuf[n].isame_ndat; // size of same level data already in buff

          // indices of recv'ing (destination) MB and buffer: MB IDs are stored
          // sequentially in MeshBlockPacks, so array index equals (target_id - first_id)
          int dm = nghbr.d_view(m,n).gid - mbgid.d_view(0);
          int dn = nghbr.d_view(m,n).dest;

          // Middle loop over k,j
          Kokkos::parallel_for(Kokkos::TeamThreadRange<>(tmember, nkj),
              [&](const int idx) {
            int k = idx / nj;
            int j = (idx - k * nj) + jl;
            k += kl;

            // Inner (vector) loop over i
            // copy directly into recv buffer if MeshBlocks on same rank
            if (nghbr.d_view(m,n).rank == my_rank) {
              // load data from coarse_u0
              Kokkos::parallel_for(Kokkos::ThreadVectorRange(tmember,il,iu+1),
              [&](const int i) {
                const bool pending_start = lat_coarse_time_interp && lat_pending_start;
                rbuf[dn].vars(dm,ndat + (i-il + ni*(j-jl + nj*(k-kl + nk*v)))) =
                    pending_start ? ca_start_view(m,v,k,j,i) : ca(m,v,k,j,i);
              });

            // else copy into send buffer for MPI communication below
            } else {
              // load data from coarse_u0
              Kokkos::parallel_for(Kokkos::ThreadVectorRange(tmember,il,iu+1),
              [&](const int i) {
                const bool pending_start = lat_coarse_time_interp && lat_pending_start;
                sbuf[n].vars(m,ndat + (i-il + ni*(j-jl + nj*(k-kl + nk*v)))) =
                    pending_start ? ca_start_view(m,v,k,j,i) : ca(m,v,k,j,i);
              });
            }
          });
        }
      } // end if-neighbor-exists block
      tmember.team_barrier();
    }); // end par_for_outer
  }
  }

#if MPI_PARALLEL_ENABLED
  // Send boundary buffer to neighboring MeshBlocks using MPI
  if (global_variable::nranks == 1) return TaskStatus::complete;
  pack_exec.fence();
  auto &is_z4c = is_z4c_;
  int my_rank = global_variable::my_rank;
  auto &nghbr = pmy_pack->pmb->nghbr;
  auto &lat_send = pmy_pack->lat_send_nghbr;
  auto &lat_boundary_send_indices_h = pmy_pack->lat_boundary_send_indices;
  bool no_errors=true;
  const int nwork_send_h = lat_enabled ? pmy_pack->lat_nboundary_send_thispack : nmb;
  for (int a=0; a<nwork_send_h; ++a) {
    const int m = lat_enabled ? lat_boundary_send_indices_h.h_view(a) : a;
    for (int n=0; n<nnghbr; ++n) {
      if (lat_enabled && lat_send.h_view(m,n) == 0) continue;
      if (nghbr.h_view(m,n).gid >= 0) {  // neighbor exists and not a physical boundary
        // index and rank of destination Neighbor
        int dn = nghbr.h_view(m,n).dest;
        int drank = nghbr.h_view(m,n).rank;
        if (drank != my_rank) {
          // create tag using local ID and buffer index of *receiving* MeshBlock
          int lid = nghbr.h_view(m,n).gid - pmy_pack->pmesh->gids_eachrank[drank];
          int tag = CreateBvals_MPI_Tag(lid, dn);

          // get ptr to send buffer when neighbor is at coarser/same/fine level
          int data_size = nvar;
          if ( nghbr.h_view(m,n).lev < pmy_pack->pmb->mb_lev.h_view(m) ) {
            data_size *= sendbuf[n].icoar_ndat;
          } else if ( nghbr.h_view(m,n).lev == pmy_pack->pmb->mb_lev.h_view(m) ) {
            if (is_z4c) {
              data_size *= sendbuf[n].isame_z4c_ndat;
            } else {
              data_size *= sendbuf[n].isame_ndat;
            }
          } else {
            data_size *= sendbuf[n].ifine_ndat;
          }
          auto send_ptr = Kokkos::subview(sendbuf[n].vars, m, Kokkos::ALL);

          int ierr = MPI_Isend(send_ptr.data(), data_size, MPI_ATHENA_REAL, drank, tag,
                               comm_vars, &(sendbuf[n].vars_req[m]));
          if (ierr != MPI_SUCCESS) {no_errors=false;}
        }
      }
    }
  }
  // Quit if MPI error detected
  if (!(no_errors)) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
       << std::endl << "MPI error in posting sends" << std::endl;
    std::exit(EXIT_FAILURE);
  }
#endif
  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
// \!fn void RecvBuffers()
// \brief Unpack boundary buffers

TaskStatus MeshBoundaryValuesCC::RecvAndUnpackCC(DvceArray5D<Real> &a,
                                                 DvceArray5D<Real> &ca) {
  // create local references for variables in kernel
  int nmb = pmy_pack->nmb_thispack;
  int nnghbr = pmy_pack->pmb->nnghbr;
  auto &nghbr = pmy_pack->pmb->nghbr;
  const bool lat_enabled = pmy_pack->lat_active_mask_enabled;
  auto lat_active_indices = pmy_pack->lat_active_indices.d_view;
  auto &lat_active_indices_h = pmy_pack->lat_active_indices;
  auto &rbuf_h = recvbuf;
  auto rbuf = recvbuf_device;
  auto &is_z4c = is_z4c_;
  auto &multilevel = pmy_pack->pmesh->multilevel;
#if MPI_PARALLEL_ENABLED
  if (rank_packed_bvals_active_) {
    // Test the requests that were STARTED (the layout InitRecv selected), with one
    // MPI_Testall rather than a poll per peer: that is a single progress pass, and
    // because Testall modifies no request unless the whole set completed, a persistent
    // request -- which goes INACTIVE rather than to MPI_REQUEST_NULL -- can never be
    // re-tested into an EMPTY status whose count would fail the size check below.
    bool no_errors=true;
    const bool all_recvd = TestVarRecvComplete(&no_errors);
    if (!(no_errors)) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "MPI error or size mismatch in rank-packed receives"
                << std::endl;
      std::exit(EXIT_FAILURE);
    }
    if (!all_recvd) return TaskStatus::incomplete;

    const int nvar_rank = a.extent_int(1);
    auto &mblev = pmy_pack->pmb->mb_lev;
    auto aggrbuf = rank_recvbuf_vars_;
    auto recvoff = rank_packed_lat_active_ ?
        lat_var_layouts_[lat_var_layout_index_].recv_offsets : recv_agg_offset_;
    const bool same_level_only = same_level_vars_only;
    const bool local_xlev = stub_recv_same_vars && !same_level_vars_only;
    auto xlevbuf = local_xlev_vars_;
    auto xlevoff = local_xlev_offset_;
    const int my_rank = global_variable::my_rank;
    const int nwork_recv_rank = lat_enabled ? pmy_pack->lat_nactive_thispack : nmb;
    const int nmn = nwork_recv_rank*nnghbr;
    if (nmn > 0) {
      Kokkos::TeamPolicy<> policy(DevExeSpace(), nmn, Kokkos::AUTO);
      Kokkos::parallel_for("RecvBuffRankPackedCC", athenak_lw(policy),
      KOKKOS_LAMBDA(TeamMember_t tmember) {
        const int im = tmember.league_rank()/nnghbr;
        const int m = lat_enabled ? lat_active_indices(im) : im;
        const int n = tmember.league_rank() - im*nnghbr;

        if (nghbr.d_view(m,n).gid >= 0) {
          const int lev = nghbr.d_view(m,n).lev;
          const int mlev = mblev.d_view(m);
          if ((nghbr.d_view(m,n).rank == my_rank && lev == mlev) ||
              (same_level_only && lev != mlev)) {
            tmember.team_barrier();
            return;
          }
          int il, iu, jl, ju, kl, ku;
          if (lev < mlev) {
            il = rbuf[n].icoar[0].bis; iu = rbuf[n].icoar[0].bie;
            jl = rbuf[n].icoar[0].bjs; ju = rbuf[n].icoar[0].bje;
            kl = rbuf[n].icoar[0].bks; ku = rbuf[n].icoar[0].bke;
          } else if (lev == mlev) {
            il = rbuf[n].isame[0].bis; iu = rbuf[n].isame[0].bie;
            jl = rbuf[n].isame[0].bjs; ju = rbuf[n].isame[0].bje;
            kl = rbuf[n].isame[0].bks; ku = rbuf[n].isame[0].bke;
          } else {
            il = rbuf[n].ifine[0].bis; iu = rbuf[n].ifine[0].bie;
            jl = rbuf[n].ifine[0].bjs; ju = rbuf[n].ifine[0].bje;
            kl = rbuf[n].ifine[0].bks; ku = rbuf[n].ifine[0].bke;
          }
          const int ni = iu - il + 1;
          const int nj = ju - jl + 1;
          const int nk = ku - kl + 1;
          const int ncells = ni*nj*nk;
          const int ntot = nvar_rank*ncells;
          // Every off-rank neighbour has an aggregate offset, so base < 0 is a same-rank
          // cross-level payload.
          const int base = recvoff(m*nnghbr + n);
          const int xbase = (base < 0 && local_xlev) ? xlevoff(m*nnghbr + n) : 0;
          const bool coarser = (lev < mlev);
          Kokkos::parallel_for(Kokkos::TeamVectorRange(tmember, ntot),
          [&](const int idx) {
            const int v = idx / ncells;
            const int r = idx - v*ncells;
            const int kk = r / (ni*nj);
            const int r2 = r - kk*(ni*nj);
            const int jj = r2 / ni;
            const int ii = r2 - jj*ni;
            const Real val = (base >= 0) ? aggrbuf(base + idx) :
                             local_xlev ? xlevbuf(xbase + idx) : rbuf[n].vars(m, idx);
            if (coarser) {
              ca(m, v, kl+kk, jl+jj, il+ii) = val;
            } else {
              a(m, v, kl+kk, jl+jj, il+ii) = val;
            }
          });
        }
        tmember.team_barrier();
      });

      if (is_z4c && multilevel) {
        Kokkos::TeamPolicy<> z4c_policy(DevExeSpace(), nmn*nvar_rank, Kokkos::AUTO);
        Kokkos::parallel_for("RecvBuffRankPackedCCZ4c", z4c_policy,
        KOKKOS_LAMBDA(TeamMember_t tmember) {
          const int mn = tmember.league_rank()/nvar_rank;
          const int im = mn/nnghbr;
          const int m = lat_enabled ? lat_active_indices(im) : im;
          const int n = mn - im*nnghbr;
          const int v = tmember.league_rank() - mn*nvar_rank;
          if (nghbr.d_view(m,n).gid >= 0 &&
              nghbr.d_view(m,n).lev == mblev.d_view(m)) {
            const int il = rbuf[n].isame_z4c.bis;
            const int iu = rbuf[n].isame_z4c.bie;
            const int jl = rbuf[n].isame_z4c.bjs;
            const int ju = rbuf[n].isame_z4c.bje;
            const int kl = rbuf[n].isame_z4c.bks;
            const int ku = rbuf[n].isame_z4c.bke;
            const int ni = iu - il + 1;
            const int nj = ju - jl + 1;
            const int nk = ku - kl + 1;
            const int ncells = ni*nj*nk;
            const int ndat = nvar_rank*rbuf[n].isame_ndat;
            const int base = recvoff(m*nnghbr + n);
            Kokkos::parallel_for(Kokkos::TeamVectorRange(tmember, ncells),
            [&](const int idx) {
              const int kk = idx / (ni*nj);
              const int r2 = idx - kk*(ni*nj);
              const int jj = r2 / ni;
              const int ii = r2 - jj*ni;
              const int bi = ndat + idx + ncells*v;
              ca(m, v, kl+kk, jl+jj, il+ii) =
                  (base >= 0) ? aggrbuf(base + bi) : rbuf[n].vars(m, bi);
            });
          }
          tmember.team_barrier();
        });
      }
    }
    if (nmn > 0) MarkRecvUnpackPending();
    return TaskStatus::complete;
  }

  //----- STEP 1: check that recv boundary buffer communications have all completed
  if (global_variable::nranks > 1) {
    bool bflag = false;
    bool no_errors=true;
    const int nwork_recv_h = lat_enabled ? pmy_pack->lat_nactive_thispack : nmb;
    for (int a=0; a<nwork_recv_h; ++a) {
      const int m = lat_enabled ? lat_active_indices_h.h_view(a) : a;
      for (int n=0; n<nnghbr; ++n) {
        if (nghbr.h_view(m,n).gid >= 0) { // neighbor exists and not a physical boundary
          if (nghbr.h_view(m,n).rank != global_variable::my_rank) {
            int test;
            int ierr = MPI_Test(&(rbuf_h[n].vars_req[m]), &test, MPI_STATUS_IGNORE);
            if (ierr != MPI_SUCCESS) {no_errors=false;}
            if (!(static_cast<bool>(test))) {
              bflag = true;
            }
          }
        }
      }
    }
    // Quit if MPI error detected
    if (!(no_errors)) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "MPI error in testing non-blocking receives"
                << std::endl;
      std::exit(EXIT_FAILURE);
    }
    // exit if recv boundary buffer communications have not completed
    if (bflag) {return TaskStatus::incomplete;}
  }
#endif

  //----- STEP 2: buffers have all completed, so unpack

  int nvar = a.extent_int(1);  // TODO(@user): 2nd index from L of in array must be NVAR
  auto &mblev = pmy_pack->pmb->mb_lev;
  const bool same_level_only = same_level_vars_only;

  // One team unpacks one (MeshBlock,neighbor) buffer and flattens all variables/cells.
  const int nwork_recv = lat_enabled ? pmy_pack->lat_nactive_thispack : nmb;
  if (nwork_recv <= 0) return TaskStatus::complete;
  Kokkos::TeamPolicy<> policy(DevExeSpace(), (nwork_recv*nnghbr), Kokkos::AUTO);
  Kokkos::parallel_for("RecvBuff", athenak_lw(policy),
      KOKKOS_LAMBDA(TeamMember_t tmember) {
    const int im = tmember.league_rank()/nnghbr;
    const int m = lat_enabled ? lat_active_indices(im) : im;
    const int n = tmember.league_rank() - im*nnghbr;

    // only unpack buffers when neighbor exists
    if (nghbr.d_view(m,n).gid >= 0) {
      const int lev = nghbr.d_view(m,n).lev;
      const int mlev = mblev.d_view(m);
      if (same_level_only && lev != mlev) return;
      int il, iu, jl, ju, kl, ku;
      // if neighbor is at coarser level, use coar indices to unpack buffer
      if (lev < mlev) {
        il = rbuf[n].icoar[0].bis;
        iu = rbuf[n].icoar[0].bie;
        jl = rbuf[n].icoar[0].bjs;
        ju = rbuf[n].icoar[0].bje;
        kl = rbuf[n].icoar[0].bks;
        ku = rbuf[n].icoar[0].bke;
      // if neighbor is at same level, use same indices to unpack buffer
      } else if (lev == mlev) {
        il = rbuf[n].isame[0].bis;
        iu = rbuf[n].isame[0].bie;
        jl = rbuf[n].isame[0].bjs;
        ju = rbuf[n].isame[0].bje;
        kl = rbuf[n].isame[0].bks;
        ku = rbuf[n].isame[0].bke;
      // if neighbor is at finer level, use fine indices to unpack buffer
      } else {
        il = rbuf[n].ifine[0].bis;
        iu = rbuf[n].ifine[0].bie;
        jl = rbuf[n].ifine[0].bjs;
        ju = rbuf[n].ifine[0].bje;
        kl = rbuf[n].ifine[0].bks;
        ku = rbuf[n].ifine[0].bke;
      }
      int ni = iu - il + 1;
      int nj = ju - jl + 1;
      int nk = ku - kl + 1;
      const int ncells = ni*nj*nk;
      const int ntot = nvar*ncells;
      const bool coarser = (lev < mlev);

      Kokkos::parallel_for(Kokkos::TeamVectorRange(tmember, ntot), [&](const int idx) {
        const int v = idx / ncells;
        const int r = idx - v*ncells;
        const int kk = r / (ni*nj);
        const int r2 = r - kk*(ni*nj);
        const int jj = r2 / ni;
        const int ii = r2 - jj*ni;
        const Real val = rbuf[n].vars(m, idx);
        if (coarser) {
          ca(m, v, kl+kk, jl+jj, il+ii) = val;
        } else {
          a(m, v, kl+kk, jl+jj, il+ii) = val;
        }
      });
    }  // end if-neighbor-exists block
    tmember.team_barrier();
  });  // end par_for_outer

  if (is_z4c && multilevel) {
    Kokkos::TeamPolicy<> z4c_policy(DevExeSpace(), (nwork_recv*nnghbr*nvar),
        Kokkos::AUTO);
    Kokkos::parallel_for("RecvBuff", z4c_policy, KOKKOS_LAMBDA(TeamMember_t tmember) {
      const int im = (tmember.league_rank())/(nnghbr*nvar);
      const int m = lat_enabled ? lat_active_indices(im) : im;
      const int n = (tmember.league_rank() - im*(nnghbr*nvar))/nvar;
      const int v = (tmember.league_rank() - im*(nnghbr*nvar) - n*nvar);
      // only unpack buffers when neighbor exists
      if (nghbr.d_view(m,n).gid >= 0) {
        int il, iu, jl, ju, kl, ku;
        // If neighbor is at same level and data is for Z4c module, unpack data from
        // coarse array for higher-order prolongation
        if (nghbr.d_view(m,n).lev == mblev.d_view(m)) {
          il = rbuf[n].isame_z4c.bis;
          iu = rbuf[n].isame_z4c.bie;
          jl = rbuf[n].isame_z4c.bjs;
          ju = rbuf[n].isame_z4c.bje;
          kl = rbuf[n].isame_z4c.bks;
          ku = rbuf[n].isame_z4c.bke;
          int ni = iu - il + 1;
          int nj = ju - jl + 1;
          int nk = ku - kl + 1;
          int nkj  = nk*nj;
          int ndat = nvar*rbuf[n].isame_ndat; // size of same level data packed in buff

          // Middle loop over k,j
          Kokkos::parallel_for(Kokkos::TeamThreadRange<>(tmember, nkj),
              [&](const int idx) {
            int k = idx / nj;
            int j = (idx - k * nj) + jl;
            k += kl;

            // load data into coarse_u0
            Kokkos::parallel_for(Kokkos::ThreadVectorRange(tmember,il,iu+1),
            [&](const int i) {
              ca(m,v,k,j,i) = rbuf[n].vars(m,
                  ndat + (i-il + ni*(j-jl + nj*(k-kl + nk*v))) );
            });
          });
        }
      }  // end if-neighbor-exists block
      tmember.team_barrier();
    });  // end par_for_outer
  }

#if MPI_PARALLEL_ENABLED
  if (global_variable::nranks > 1) MarkRecvUnpackPending();
#endif

  return TaskStatus::complete;
}
