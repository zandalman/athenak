//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file srcterms_newdt.cpp
//! \brief function to compute timestep for source terms across all MeshBlock(s) in a
//! MeshBlockPack

#include <float.h>

#include <limits>

#include "athena.hpp"
#include "coordinates/cell_locations.hpp"
#include "mesh/mesh.hpp"
#include "eos/eos.hpp"
#include "pgen/pgen.hpp"
#include "ismcooling.hpp"
#include "srcterms.hpp"
#include "units/units.hpp"
#include "utils/gravity_weight.hpp"

//----------------------------------------------------------------------------------------
//! \fn void SourceTerms::NewTimeStep()
//! \brief Compute new timestep for source terms.

void SourceTerms::NewTimeStep(const DvceArray5D<Real> &w0, const EOS_Data &eos_data) {
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int is = indcs.is, nx1 = indcs.nx1;
  int js = indcs.js, nx2 = indcs.nx2;
  int ks = indcs.ks, nx3 = indcs.nx3;
  const int nkji = nx3*nx2*nx1;
  const int nji  = nx2*nx1;
  const int nmb = pmy_pack->nmb_thispack;
  const Real big_dt = std::numeric_limits<Real>::max();
  dtnew = static_cast<Real>(std::numeric_limits<float>::max());
  ResizeMeshBlockStorage(nmb);
  Kokkos::deep_copy(DevExeSpace(), dtnew_eachmb.d_view, big_dt);
  dtnew_eachmb.template modify<DevExeSpace>();

  if (ism_cooling) {
    Real gamma = eos_data.gamma;
    Real gm1 = gamma - 1.0;
    Real heating_rate = hrate;
    Real temp_unit = pmy_pack->TemperatureUnitCGS();
    Real n_unit = pmy_pack->punit->density_cgs()/pmy_pack->punit->mu()
                  / pmy_pack->punit->atomic_mass_unit_cgs;
    Real cooling_unit = pmy_pack->punit->pressure_cgs()/pmy_pack->punit->time_cgs()
                        / n_unit/n_unit;
    Real heating_unit = pmy_pack->punit->pressure_cgs()/pmy_pack->punit->time_cgs()
                        / n_unit;

    // find smallest (e/cooling_rate) in each cell
    Real cooling_dt = big_dt;
    auto dt_eachmb = dtnew_eachmb.d_view;
    Kokkos::parallel_reduce("srcterms_ism_cooling_newdt",
                            Kokkos::TeamPolicy<>(DevExeSpace(), nmb, Kokkos::AUTO),
    KOKKOS_LAMBDA(TeamMember_t tmember, Real &pack_min) {
      const int m = tmember.league_rank();
      Real mb_dt = big_dt;
      Kokkos::parallel_reduce(Kokkos::TeamThreadRange(tmember, nkji),
      [=](const int idx, Real &local_min) {
        // compute k,j,i indices of thread and call function
        int k = idx/nji;
        int j = (idx - k*nji)/nx1;
        int i = (idx - k*nji - j*nx1) + is;
        k += ks;
        j += js;

        // temperature in cgs unit
        Real temp = temp_unit*w0(m,IEN,k,j,i)/w0(m,IDN,k,j,i)*gm1;
        Real eint = w0(m,IEN,k,j,i);

        Real lambda_cooling = ISMCoolFn(temp)/cooling_unit;
        Real gamma_heating = heating_rate/heating_unit;

        // add a tiny number
        Real cooling_heating = FLT_MIN + fabs(w0(m,IDN,k,j,i) *
                               (w0(m,IDN,k,j,i) * lambda_cooling - gamma_heating));

        local_min = fmin((eint/cooling_heating), local_min);
      }, Kokkos::Min<Real>(mb_dt));
      dt_eachmb(m) = fmin(dt_eachmb(m), mb_dt);
      pack_min = fmin(pack_min, mb_dt);
    }, Kokkos::Min<Real>(cooling_dt));
    dtnew = fmin(dtnew, cooling_dt);
    dtnew_eachmb.template modify<DevExeSpace>();
  }

  if (rel_cooling) {
    Real gamma = eos_data.gamma;
    Real gm1 = gamma - 1.0;
    Real cooling_rate = crate_rel;
    Real cooling_power = cpower_rel;

    // find smallest (e/cooling_rate) in each cell
    Real cooling_dt = big_dt;
    auto dt_eachmb = dtnew_eachmb.d_view;
    Kokkos::parallel_reduce("srcterms_rel_cooling_newdt",
                            Kokkos::TeamPolicy<>(DevExeSpace(), nmb, Kokkos::AUTO),
    KOKKOS_LAMBDA(TeamMember_t tmember, Real &pack_min) {
      const int m = tmember.league_rank();
      Real mb_dt = big_dt;
      Kokkos::parallel_reduce(Kokkos::TeamThreadRange(tmember, nkji),
      [=](const int idx, Real &local_min) {
        // compute k,j,i indices of thread and call function
        int k = idx/nji;
        int j = (idx - k*nji)/nx1;
        int i = (idx - k*nji - j*nx1) + is;
        k += ks;
        j += js;

        // temperature in cgs unit
        Real temp = w0(m,IEN,k,j,i)/w0(m,IDN,k,j,i)*gm1;
        Real eint = w0(m,IEN,k,j,i);

        auto &ux = w0(m, IVX, k, j, i);
        auto &uy = w0(m, IVY, k, j, i);
        auto &uz = w0(m, IVZ, k, j, i);

        auto ut = 1. + ux * ux + uy * uy + uz * uz;
        ut = sqrt(ut);

        // The following should be approximately correct
        // add a tiny number
        Real cooling_heating = FLT_MIN + fabs(w0(m,IDN,k,j,i) * ut *
                               pow(temp*cooling_rate, cooling_power));

        local_min = fmin((eint/cooling_heating), local_min);
      }, Kokkos::Min<Real>(mb_dt));
      dt_eachmb(m) = fmin(dt_eachmb(m), mb_dt);
      pack_min = fmin(pack_min, mb_dt);
    }, Kokkos::Min<Real>(cooling_dt));
    dtnew = fmin(dtnew, cooling_dt);
    dtnew_eachmb.template modify<DevExeSpace>();
  }

  if (external_bh_gravity && external_bh_dt_factor > 0.0 &&
      problem_runtime::ExternalBHGravitySourceCouplingEnabled()) {
    bool external_bh_enabled = false;
    Real bhx = 0.0, bhy = 0.0, bhz = 0.0;
    Real bh_mass = 0.0, bh_softening = 0.0, newton_g = 0.0, bh_rg = 0.0;
    problem_runtime::GetExternalBHPotential(
        pmy_pack->pmesh->time, external_bh_enabled, bhx, bhy, bhz,
        bh_mass, bh_softening, newton_g, bh_rg);
    if (external_bh_enabled && bh_mass > 0.0 && newton_g > 0.0) {
      bool bh_sink_mask_enabled = false;
      Real bh_sink_mask_radius = 0.0;
      Real bh_sink_mask_x = 0.0;
      Real bh_sink_mask_y = 0.0;
      Real bh_sink_mask_z = 0.0;
      problem_runtime::GetBHSinkGravityMask(
          pmy_pack->pmesh->time, bh_sink_mask_enabled, bh_sink_mask_radius,
          bh_sink_mask_x, bh_sink_mask_y, bh_sink_mask_z);
      // the excision interior is reset every step and SourceTerms::Gravity skips it, so it
      // does not limit the step either (with a dense bh_excise_density it would, near r=0)
      bool excise_enabled = false;
      Real excise_radius = 0.0, excise_density = 0.0, excise_eint = 0.0;
      Real sink_x = 0.0, sink_y = 0.0, sink_z = 0.0;
      problem_runtime::GetExcisionState(pmy_pack->pmesh->time, excise_enabled,
          excise_radius, excise_density, excise_eint, sink_x, sink_y, sink_z);
      const Real excise_r2 = excise_enabled ? excise_radius*excise_radius : -1.0;
      auto &mbsize = pmy_pack->pmb->mb_size;
      auto dt_eachmb = dtnew_eachmb.d_view;
      const Real rho_min = rho_external_bh_min;
      const Real rho_floor = eos_data.dfloor;
      const Real dt_factor = external_bh_dt_factor;
      const Real eps2 = bh_softening*bh_softening;
      Real bh_dt = big_dt;
      Kokkos::parallel_reduce("srcterms_external_bh_newdt",
                              Kokkos::TeamPolicy<>(DevExeSpace(), nmb, Kokkos::AUTO),
      KOKKOS_LAMBDA(TeamMember_t tmember, Real &pack_min) {
        const int m = tmember.league_rank();
        const Real dx_min = fmin(mbsize.d_view(m).dx1,
                            fmin(mbsize.d_view(m).dx2, mbsize.d_view(m).dx3));
        Real mb_dt = big_dt;
        Kokkos::parallel_reduce(Kokkos::TeamThreadRange(tmember, nkji),
        [=](const int idx, Real &local_min) {
          int k = idx/nji;
          int j = (idx - k*nji)/nx1;
          int i = (idx - k*nji - j*nx1) + is;
          k += ks;
          j += js;

          const Real rho = w0(m,IDN,k,j,i);
          if (!(rho > 0.0)) return;
          // SourceTerms::Gravity accelerates this cell by w(rho) of the BH pull, so the
          // limit is 0.5 sqrt(dx/(w a)) over every cell it pulls at all.
          const Real w_bh = gravity_weight::Weight(rho, rho_floor, rho_min);
          if (!(w_bh > 0.0)) return;
          const Real x = CellCenterX(i - is, nx1, mbsize.d_view(m).x1min,
                                     mbsize.d_view(m).x1max);
          const Real y = CellCenterX(j - js, nx2, mbsize.d_view(m).x2min,
                                     mbsize.d_view(m).x2max);
          const Real z = CellCenterX(k - ks, nx3, mbsize.d_view(m).x3min,
                                     mbsize.d_view(m).x3max);
          const Real rx = x - bhx;
          const Real ry = y - bhy;
          const Real rz = z - bhz;
          const Real r2 = rx*rx + ry*ry + rz*rz;
          if (excise_r2 > 0.0 &&
              problem_runtime::InsideExcisionZone(x, y, z, sink_x, sink_y, sink_z,
                                                  excise_r2)) {
            return;
          }
          if (bh_sink_mask_enabled &&
              problem_runtime::InsideBHSinkGravityMask(x, y, z, bh_sink_mask_x,
                                                       bh_sink_mask_y,
                                                       bh_sink_mask_z,
                                                       bh_sink_mask_radius)) {
            return;
          }
          const Real effective_r2 = bh_sink_mask_enabled ?
              fmax(r2, bh_sink_mask_radius*bh_sink_mask_radius) : r2;
          const Real denom = effective_r2 + eps2;
          if (!(denom > 0.0)) return;
          // |dPhi/dr_s| of the applied potential (G M / r_s^2 when Newtonian)
          const Real accel = w_bh*fabs(bh_force_pair::PotentialSlope(sqrt(denom), bh_mass,
                                                                     newton_g, bh_rg));
          if (!(accel > 0.0)) return;
          local_min = fmin(local_min, dt_factor*sqrt(dx_min/accel));
        }, Kokkos::Min<Real>(mb_dt));
        dt_eachmb(m) = fmin(dt_eachmb(m), mb_dt);
        pack_min = fmin(pack_min, mb_dt);
      }, Kokkos::Min<Real>(bh_dt));
      dtnew = fmin(dtnew, bh_dt);
      dtnew_eachmb.template modify<DevExeSpace>();
    }
  }

  return;
}
