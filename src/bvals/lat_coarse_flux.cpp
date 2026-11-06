//========================================================================================
// AthenaK astrophysical fluid dynamics & numerical relativity code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file lat_coarse_flux.cpp
//! \brief see lat_coarse_flux.hpp

#include "bvals/lat_coarse_flux.hpp"

#include "athena.hpp"
#include "bvals/bvals.hpp"
#include "driver/driver.hpp"
#include "driver/lat_weights.hpp"
#include "mesh/mesh.hpp"

namespace lat {
namespace {

//! One face cell's contribution, in the expression tree each module had before the
//! kernel was shared: GR (kLedger) forms the product once and adds it to the accumulator
//! and, under lat_diagnostics, to the ledger copy; the Newtonian module adds the product
//! directly.  The two trees are kept apart because FMA contraction may treat them
//! differently, and each module's reflux stays bit for bit what it was.
template <bool kLedger>
KOKKOS_FORCEINLINE_FUNCTION void AddCoarseFlux(Real &acc, Real &own, const bool has_own,
                                               const Real scale, const Real flx) {
  if constexpr (kLedger) {
    const Real contrib = scale*flx;
    acc += contrib;
    if (has_own) own += contrib;
  } else {
    acc += scale*flx;
  }
}

template <bool kLedger>
void AccumulateImpl(MeshBlockPack *pmbp, MeshBoundaryValuesCC *pbval, Driver *pdrive,
                    const int stage, const BandView5D<Real> &flx1,
                    const BandView5D<Real> &flx2, const BandView5D<Real> &flx3,
                    const DvceFaceFld5D<Real> &acc, const DvceFaceFld5D<Real> *own,
                    const int nvar, const char *label) {
  if (!(pmbp->lat_active_mask_enabled)) return;
  const int nactive = pmbp->lat_nactive_thispack;
  if (nactive <= 0) return;
  const Real flux_weight = FinalFluxWeight(pdrive, stage);
  if (flux_weight == 0.0) return;

  const int nnghbr = pmbp->pmb->nnghbr;
  auto &indcs = pmbp->pmesh->mb_indcs;
  const int is = indcs.is, js = indcs.js, ks = indcs.ks;
  auto &nghbr = pmbp->pmb->nghbr;
  auto &mblev = pmbp->pmb->mb_lev;
  auto active_indices = pmbp->lat_active_indices.d_view;
  // FALSE on the factor path, where lat_step_dt is identically zero: the pack ships the
  // RAW instantaneous flux there and the receiver applies weight*pmesh->dt (the modules'
  // SendFlux/RecvFlux).  The union stage-1 predictor sets it, and then each block
  // integrates its own flux over lat_step_dt(m) = fine_dt*factor(m).
  const bool lat_per_block_dt = pmbp->lat_per_block_timestep;
  auto lat_step_dt = pmbp->lat_step_dt.d_view;
  const Real mesh_dt = pmbp->pmesh->dt;
  auto lat_flux_accum = pmbp->lat_flux_accum_nghbr.d_view;
  auto lat_step_factor = pmbp->lat_step_factor.d_view;
  auto lat_nghbr_factor = pmbp->lat_nghbr_factor.d_view;
  auto rbuf = pbval->recvbuf_device;
  auto acc1 = acc.x1f;
  auto acc2 = acc.x2f;
  auto acc3 = acc.x3f;
  const bool has_own = (own != nullptr);
  auto own1 = has_own ? own->x1f : acc.x1f;
  auto own2 = has_own ? own->x2f : acc.x2f;
  auto own3 = has_own ? own->x3f : acc.x3f;

  const int nflux_nghbr = FluxFaceCount(nnghbr);
  if (nflux_nghbr <= 0) return;
  Kokkos::TeamPolicy<> policy(DevExeSpace(), nactive*nflux_nghbr*nvar, Kokkos::AUTO);
  Kokkos::parallel_for(label, athenak_lw(policy),
  KOKKOS_LAMBDA(TeamMember_t tmember) {
    const int a = tmember.league_rank()/(nflux_nghbr*nvar);
    const int slot = (tmember.league_rank() - a*nflux_nghbr*nvar)/nvar;
    const int nb = FluxFaceNeighborIndex(slot, nnghbr);
    const int v = tmember.league_rank() - a*nflux_nghbr*nvar - slot*nvar;
    const int m = active_indices(a);
    if (lat_flux_accum(m,nb) == 0) return;
    if (nghbr.d_view(m,nb).lev > mblev.d_view(m) &&
        lat_step_factor(m) == lat_nghbr_factor(m,nb)) return;
    const Real scale = flux_weight*(lat_per_block_dt ? lat_step_dt(m) : mesh_dt);

    const bool same_level = (nghbr.d_view(m,nb).lev == mblev.d_view(m));
    const auto &iflux = same_level ? rbuf(nb).iflux_same[0] : rbuf(nb).iflux_coar[0];
    const int il = iflux.bis, iu = iflux.bie;
    const int jl = iflux.bjs, ju = iflux.bje;
    const int kl = iflux.bks, ku = iflux.bke;
    const int ni = iu - il + 1;
    const int nj = ju - jl + 1;
    const int nk = ku - kl + 1;

    if (nb < 8) {
      const int ia = (il == is) ? 0 : 1;
      Kokkos::parallel_for(Kokkos::TeamThreadRange<>(tmember, nk*nj),
      [&](const int idx) {
        int k = idx/nj;
        const int j = idx - k*nj + jl;
        k += kl;
        AddCoarseFlux<kLedger>(acc1(m,v,k,j,ia), own1(m,v,k,j,ia), has_own, scale,
                               flx1(m,v,k,j,il));
      });
    } else if (nb < 16) {
      const int ja = (jl == js) ? 0 : 1;
      Kokkos::parallel_for(Kokkos::TeamThreadRange<>(tmember, nk*ni),
      [&](const int idx) {
        int k = idx/ni;
        const int i = idx - k*ni + il;
        k += kl;
        AddCoarseFlux<kLedger>(acc2(m,v,k,ja,i), own2(m,v,k,ja,i), has_own, scale,
                               flx2(m,v,k,jl,i));
      });
    } else if (nb >= 24 && nb < 32) {
      const int ka = (kl == ks) ? 0 : 1;
      Kokkos::parallel_for(Kokkos::TeamThreadRange<>(tmember, nj*ni),
      [&](const int idx) {
        int j = idx/ni;
        const int i = idx - j*ni + il;
        j += jl;
        AddCoarseFlux<kLedger>(acc3(m,v,ka,j,i), own3(m,v,ka,j,i), has_own, scale,
                               flx3(m,v,kl,j,i));
      });
    }
    tmember.team_barrier();
  });
}

}  // namespace

void AllocateRefluxBands(DvceFaceFld5D<Real> &acc, const int nmb, const int nvar,
                         const int n3, const int n2, const int n1) {
  Kokkos::realloc(acc.x1f, nmb, nvar, n3, n2, 2);
  Kokkos::realloc(acc.x2f, nmb, nvar, n3, 2, n1);
  Kokkos::realloc(acc.x3f, nmb, nvar, 2, n2, n1);
  Kokkos::deep_copy(acc.x1f, 0.0);
  Kokkos::deep_copy(acc.x2f, 0.0);
  Kokkos::deep_copy(acc.x3f, 0.0);
}

void AccumulateCoarseFluxes(MeshBlockPack *pmbp, MeshBoundaryValuesCC *pbval,
                            Driver *pdrive, const int stage, const BandView5D<Real> &flx1,
                            const BandView5D<Real> &flx2, const BandView5D<Real> &flx3,
                            const DvceFaceFld5D<Real> &acc,
                            const DvceFaceFld5D<Real> *own, const bool ledger,
                            const int nvar, const char *label) {
  if (ledger) {
    AccumulateImpl<true>(pmbp, pbval, pdrive, stage, flx1, flx2, flx3, acc, own, nvar,
                         label);
  } else {
    AccumulateImpl<false>(pmbp, pbval, pdrive, stage, flx1, flx2, flx3, acc, nullptr,
                          nvar, label);
  }
}

}  // namespace lat
