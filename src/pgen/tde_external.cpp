//========================================================================================
// AthenaK astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file tde_external.cpp
//! \brief Problem generator for a star in an external SMBH potential.
//!
//! The legacy stellar profile is a Lane-Emden polytrope with fixed total stellar mass
//! M*=1. A selectable EOS-balanced mode also constructs a 1D hydrostatic profile with
//! the active EOS closure before mapping it into 3D.
//! An external Newtonian BH gravity is added through the gravity source path:
//!   Phi_BH = -G M_BH / sqrt(|r-r_BH|^2 + eps^2).
//! Here G is inferred from gravity/four_pi_G as G = four_pi_G/(4*pi), consistent with
//! the unit system used by self-gravity setups in this code.

#include <algorithm>
#include <cctype>
#include <array>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <memory>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "athena.hpp"
#include "globals.hpp"
#include "parameter_input.hpp"
#include "coordinates/cell_locations.hpp"
#include "mesh/mesh.hpp"
#include "mesh/nghbr_index.hpp"
#include "eos/eos.hpp"
#include "eos/general_c2p_mhd.hpp"
#include "hydro/hydro.hpp"
#include "gravity/gravity.hpp"
#include "driver/driver.hpp"
#include "outputs/io_wrapper.hpp"
#include "srcterms/srcterms.hpp"
#include "utils/gravity_weight.hpp"
#include "units/units.hpp"
#include "remap/remap.hpp"
#include "pgen.hpp"
#include "pgen/bh_dynamics.hpp"
#include "pgen/star_bh_orbit.hpp"
#include "pgen/stellar_profile.hpp"
#include "pgen/tde_amr.hpp"
#include "tde_external.hpp"

namespace tde_external {
void StoreRuntimeMetadata(ParameterInput *pin);

}  // namespace tde_external

namespace {

// External BH gravity source-term parameters
Real bh_mass_global;
Real bh_inertial_x_global, bh_inertial_y_global, bh_inertial_z_global;
Real bh_x_global, bh_y_global, bh_z_global;
Real bh_vx_global, bh_vy_global, bh_vz_global;
Real bh_soft_global;
Real bh_rg_global = 0.0;  // G M / c^2 with bh_potential = wegg, 0 for the Newtonian mass
Real bh_grav_rho_min_global;  // minimum density for BH gravity source term
Real frame_rho_min_global;    // minimum density used to measure frame acceleration
bool use_translating_frame_global = false;
bool hydro_lat_enabled_global = false;
Real bh_excise_radius_global;
Real bh_excise_density_global;
Real bh_excise_eint_global;
Real newton_g_global;
bool bh_inner_boundary_global = true;

// Translating, non-rotating debris-frame state. When the translating frame is
// disabled, these are reset to zero and the BH is frozen in the current mesh
// coordinates.
Real orbit_center_x_global, orbit_center_y_global, orbit_center_z_global;
Real frame_vx_global, frame_vy_global, frame_vz_global;
Real frame_ax_global, frame_ay_global, frame_az_global;
Real frame_state_time_global;
int frame_step_cycle_global = -1;
Real frame_step_time_global = 0.0;
Real frame_step_dt_global = 0.0;
Real frame_step_x0_global, frame_step_y0_global, frame_step_z0_global;
Real frame_step_vx0_global, frame_step_vy0_global, frame_step_vz0_global;
Real frame_step_ax0_global, frame_step_ay0_global, frame_step_az0_global;
Real frame_step_x1_global, frame_step_y1_global, frame_step_z1_global;
Real frame_step_vx1_global, frame_step_vy1_global, frame_step_vz1_global;
Real frame_step_ax1_global, frame_step_ay1_global, frame_step_az1_global;
bool frame_step_valid_global = false;
bool relax_enable_global;
Real relax_tau_global, relax_t_end_global, relax_radius_global;
Real hydro_dfloor_global = 0.0;

// <tde_amr> scheme, active only when the deck has a <tde_amr> block.  It then sets every
// refinement flag itself; without the block the legacy rules above are untouched.
bool tde_amr_enable_global = false;
tde_amr::Params tde_amr_params_global{};

int bh_step_cycle_global = -1;
Real bh_step_time_global = 0.0;
Real bh_step_dt_global = 0.0;
Real bh_step_x0_global, bh_step_y0_global, bh_step_z0_global;
Real bh_step_vx0_global, bh_step_vy0_global, bh_step_vz0_global;
Real bh_step_x1_global, bh_step_y1_global, bh_step_z1_global;
Real bh_step_vx1_global, bh_step_vy1_global, bh_step_vz1_global;
bool bh_step_valid_global = false;

// Live BH (problem/bh_live = true).  The BH inertial state is advanced by the gas gravity
// with the kick-drift-kick leapfrog at rank-synchronized points exactly as in
// star_bh_collision (shared helpers in pgen/bh_dynamics.hpp); the simulation-frame BH is
// that state minus the translating frame.  By default the BH stays at rest in the
// inertial frame.
bool bh_live_global = false;
Real bh_inertial_vx_global = 0.0, bh_inertial_vy_global = 0.0;
Real bh_inertial_vz_global = 0.0;
Real bh_gas_ax_global = 0.0, bh_gas_ay_global = 0.0, bh_gas_az_global = 0.0;
bool bh_reciprocal_force_global = false;
bool bh_analytic_pair_global = false;
Real bh_local_gas_impulse[3] = {0.0, 0.0, 0.0};
Real bh_total_gas_impulse[3] = {0.0, 0.0, 0.0};
Real bh_last_impulse_residual = 0.0;
bool bh_impulse_window_open = false;
Real bh_window_start_v[3] = {0.0, 0.0, 0.0};   // inertial BH velocity at the step start
Real bh_window_frame_dv[3] = {0.0, 0.0, 0.0};  // frame velocity change over the step
Real bh_window_frame_v1[3] = {0.0, 0.0, 0.0};  // frame velocity at the step end
// Energy ledger (gravity/lat_time_centered_work): cumulative work of the translating
// frame's fictitious force (and relaxation damping) on the gas and on a live BH.  The
// gas part is accumulated rank-locally in the source pass and reduced only at
// synchronized points.
bool energy_ledger_global = false;
Real frame_work_local = 0.0;
Real frame_work_total = 0.0;

struct LaneEmdenProfile {
  std::vector<Real> xi;
  std::vector<Real> theta;
  std::vector<Real> dtheta;
  Real xi1;
  Real dtheta_xi1;
};

enum class StellarStructureMode {
  legacy_polytrope,
  eos_balanced,
};

using StellarRadialProfile = stellar_profile::RadialProfile;

void RefineTDEAMR(MeshBlockPack *pmbp);
void CapHydroLATFactorsNearBH(Mesh *pm, int max_factor, int *lat_factor_eachmb);
void SyncProblemRuntimeState();
void InvalidateFrameBHStepState();
void RestoreOptionalFrameBHStepState(ParameterInput *pin);
void TDEExternalHydroStateFixup(MeshBlockPack *pmbp, const Real time);
void SwitchToFrozenBHInertialFrame(Mesh *pm, MeshBlockPack *pmbp);
bool RestoreFrameBHStateFromMetadata(ParameterInput *pin);
void RestoreLiveBHMetadata(ParameterInput *pin);

inline Real ClampUnitInterval(const Real x);
KOKKOS_INLINE_FUNCTION
Real CubicHermiteValue(const Real y0, const Real m0, const Real y1, const Real m1,
                       const Real h, const Real t);
KOKKOS_INLINE_FUNCTION
Real CubicHermiteSlope(const Real y0, const Real m0, const Real y1, const Real m1,
                       const Real h, const Real t);

inline Real ThetaPow(const Real theta, const Real n) {
  return (theta > 0.0) ? std::pow(theta, n) : 0.0;
}

inline void EvaluateBHAccelerationAtPoint(const Real x, const Real y, const Real z,
                                          const Real bhx, const Real bhy, const Real bhz,
                                          Real &ax, Real &ay, Real &az) {
  Real dx = x - bhx;
  Real dy = y - bhy;
  Real dz = z - bhz;
  Real eps2 = bh_soft_global * bh_soft_global;
  Real r2 = dx*dx + dy*dy + dz*dz + eps2;
  r2 = std::fmax(r2, static_cast<Real>(1.0e-30));
  const Real rs = std::sqrt(r2);
  // -(dPhi/dr_s)/r_s of the applied potential (bh_force_pair.hpp)
  const Real factor = -bh_force_pair::PotentialSlope(rs, bh_mass_global, newton_g_global,
                                                     bh_rg_global)/rs;
  ax = factor * dx;
  ay = factor * dy;
  az = factor * dz;
}

// Apply a uniform Galilean boost to the loaded hydro state.  The internal energy is
// frame-invariant, but the dual-energy inversion's choice of channel is not: its eta1
// test compares E - K with the total E, which the boost raises.  A cell whose auxiliary
// an eta2 resynchronisation left behind (a steep edge in a checkpoint written before the
// hydro inversion published its thermal state, see DualEnergyInversionEint)
// would then be read off that stale auxiliary instead of E - K, so the auxiliary is first
// set to the internal energy the inversion takes in the old frame.  The boost itself is
// unchanged: rest mass, momentum and total energy transform exactly as before.
void BoostHydroState(MeshBlockPack *pmbp, const Real dvx, const Real dvy, const Real dvz,
                     const Real rho_boost_min) {
  if (pmbp == nullptr || pmbp->phydro == nullptr) return;
  if (dvx == 0.0 && dvy == 0.0 && dvz == 0.0) return;

  auto &u0 = pmbp->phydro->u0;
  auto &indcs = pmbp->pmesh->mb_indcs;
  int is = indcs.is, ie = indcs.ie;
  int js = indcs.js, je = indcs.je;
  int ks = indcs.ks, ke = indcs.ke;
  int nmb = pmbp->nmb_thispack;
  const bool eos_has_energy = pmbp->phydro->peos->eos_data.use_e;
  const bool dual = eos_has_energy && pmbp->phydro->use_dual_energy;
  const int de_idx = pmbp->phydro->dual_energy_idx;
  const Real dual_eta1 = pmbp->phydro->dual_energy_eta1;
  Real dv2 = dvx*dvx + dvy*dvy + dvz*dvz;
  // Restriction/prolongation can leave a uniform atmosphere a few ulps above
  // dfloor.  A strict rho > dfloor test then boosts whole background blocks while
  // leaving their exactly-at-floor neighbours stationary.  Ignore only roundoff
  // around the atmosphere floor; resolved gas still receives the full boost.
  const Real rho_boost_cut = rho_boost_min *
      (1.0 + 64.0*std::numeric_limits<Real>::epsilon());

  par_for("tde_boost_hydro_state", DevExeSpace(), 0, nmb - 1, ks, ke, js, je, is, ie,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    Real rho = u0(m, IDN, k, j, i);
    if (!(rho > rho_boost_cut)) return;

    Real mx = u0(m, IM1, k, j, i);
    Real my = u0(m, IM2, k, j, i);
    Real mz = u0(m, IM3, k, j, i);
    if (dual) {
      // E - K as SingleC2P_GeneralHydDual forms it
      const Real e = u0(m, IEN, k, j, i);
      const Real di = 1.0/rho;
      const Real eint_cons = e - 0.5*di*(SQR(mx) + SQR(my) + SQR(mz));
      u0(m, de_idx, k, j, i) = eos_general::DualEnergyInversionEint(
          eint_cons, e, 0.0, u0(m, de_idx, k, j, i), dual_eta1);
    }
    if (eos_has_energy) {
      u0(m, IEN, k, j, i) += mx*dvx + my*dvy + mz*dvz + 0.5*rho*dv2;
    }
    u0(m, IM1, k, j, i) = mx + rho*dvx;
    u0(m, IM2, k, j, i) = my + rho*dvy;
    u0(m, IM3, k, j, i) = mz + rho*dvz;
  });
}

// Convert a loaded translating-frame restart/remap state into the fixed BH
// inertial-frame mode without changing mesh coordinates.
void SwitchToFrozenBHInertialFrame(Mesh *pm, MeshBlockPack *pmbp) {
  const Real rho_boost_min =
      (pmbp != nullptr &&
          pmbp->phydro != nullptr) ? pmbp->phydro->peos->eos_data.dfloor : 0.0;
  BoostHydroState(pmbp, frame_vx_global, frame_vy_global, frame_vz_global, rho_boost_min);

  // Keep the current mesh coordinates and freeze the BH at its current location.
  bh_inertial_x_global = bh_x_global;
  bh_inertial_y_global = bh_y_global;
  bh_inertial_z_global = bh_z_global;
  bh_vx_global = bh_live_global ? bh_inertial_vx_global : 0.0;
  bh_vy_global = bh_live_global ? bh_inertial_vy_global : 0.0;
  bh_vz_global = bh_live_global ? bh_inertial_vz_global : 0.0;

  orbit_center_x_global = 0.0;
  orbit_center_y_global = 0.0;
  orbit_center_z_global = 0.0;
  frame_vx_global = 0.0;
  frame_vy_global = 0.0;
  frame_vz_global = 0.0;
  frame_ax_global = 0.0;
  frame_ay_global = 0.0;
  frame_az_global = 0.0;
  frame_state_time_global = pm->time;

  InvalidateFrameBHStepState();

  SyncProblemRuntimeState();
}

inline Real DegToRad(const Real deg) {
  return deg * (M_PI/180.0);
}

inline void InterpolateFrameStepState(const Real t, Real &cx, Real &cy, Real &cz,
                                      Real &cvx, Real &cvy, Real &cvz) {
  if (!frame_step_valid_global || frame_step_dt_global <= 0.0) {
    Real dt = t - frame_state_time_global;
    cx = orbit_center_x_global + frame_vx_global*dt + 0.5*frame_ax_global*dt*dt;
    cy = orbit_center_y_global + frame_vy_global*dt + 0.5*frame_ay_global*dt*dt;
    cz = orbit_center_z_global + frame_vz_global*dt + 0.5*frame_az_global*dt*dt;
    cvx = frame_vx_global + frame_ax_global*dt;
    cvy = frame_vy_global + frame_ay_global*dt;
    cvz = frame_vz_global + frame_az_global*dt;
    return;
  }

  Real s = ClampUnitInterval((t - frame_step_time_global) / frame_step_dt_global);
  cx = CubicHermiteValue(frame_step_x0_global, frame_step_vx0_global,
                         frame_step_x1_global, frame_step_vx1_global,
                         frame_step_dt_global, s);
  cy = CubicHermiteValue(frame_step_y0_global, frame_step_vy0_global,
                         frame_step_y1_global, frame_step_vy1_global,
                         frame_step_dt_global, s);
  cz = CubicHermiteValue(frame_step_z0_global, frame_step_vz0_global,
                         frame_step_z1_global, frame_step_vz1_global,
                         frame_step_dt_global, s);
  cvx = CubicHermiteSlope(frame_step_x0_global, frame_step_vx0_global,
                          frame_step_x1_global, frame_step_vx1_global,
                          frame_step_dt_global, s);
  cvy = CubicHermiteSlope(frame_step_y0_global, frame_step_vy0_global,
                          frame_step_y1_global, frame_step_vy1_global,
                          frame_step_dt_global, s);
  cvz = CubicHermiteSlope(frame_step_z0_global, frame_step_vz0_global,
                          frame_step_z1_global, frame_step_vz1_global,
                          frame_step_dt_global, s);
}

inline void FrameCenterAtTime(const Real t, Real &cx, Real &cy, Real &cz) {
  Real cvx = 0.0, cvy = 0.0, cvz = 0.0;
  InterpolateFrameStepState(t, cx, cy, cz, cvx, cvy, cvz);
}

bool MeasureSelectedFrameAcceleration(Mesh *pm,
                                      Real &frame_ax, Real &frame_ay, Real &frame_az) {
  MeshBlockPack *pmbp = pm->pmb_pack;
  if (pmbp == nullptr || pmbp->phydro == nullptr) return false;

  auto &indcs = pm->mb_indcs;
  int is = indcs.is, ie = indcs.ie;
  int js = indcs.js, je = indcs.je;
  int ks = indcs.ks, ke = indcs.ke;
  int ni = ie - is + 1;
  int nj = je - js + 1;
  int nk = ke - ks + 1;
  int nmb = pmbp->nmb_thispack;
  int nmkji = nmb * nk * nj * ni;
  if (nmkji <= 0) return false;

  auto &u0 = pmbp->phydro->u0;
  auto size = pmbp->pmb->mb_size.d_view;
  Real bhx = bh_x_global;
  Real bhy = bh_y_global;
  Real bhz = bh_z_global;
  const Real excise_r2 = bh_inner_boundary_global ?
      bh_excise_radius_global * bh_excise_radius_global : -1.0;
  Real rho_min = frame_rho_min_global;
  Real soft2 = bh_soft_global * bh_soft_global;
  const Real bh_mass = bh_mass_global, newton_g = newton_g_global, rg = bh_rg_global;

  // A reduction, not atomic adds into one slot: the order the cells of a floating-point
  // sum arrive in must not change from run to run, since the frame acceleration moves
  // every cell.
  const int nkji = nk*nj*ni;
  const int nji = nj*ni;
  array_sum::GlobalSum sums;
  Kokkos::parallel_reduce("tde_frame_moments", Kokkos::RangePolicy<>(DevExeSpace(), 0,
                                                                     nmkji),
  KOKKOS_LAMBDA(const int &idx, array_sum::GlobalSum &acc) {
    const int m = idx/nkji;
    int k = (idx - m*nkji)/nji;
    int j = (idx - m*nkji - k*nji)/ni;
    int i = (idx - m*nkji - k*nji - j*ni) + is;
    k += ks;
    j += js;
    Real rho = u0(m, IDN, k, j, i);
    if (rho <= 0.0 || rho < rho_min) return;

    Real x = CellCenterX(i - is, ni, size(m).x1min, size(m).x1max);
    Real y = CellCenterX(j - js, nj, size(m).x2min, size(m).x2max);
    Real z = CellCenterX(k - ks, nk, size(m).x3min, size(m).x3max);
    Real dx = x - bhx;
    Real dy = y - bhy;
    Real dz = z - bhz;
    if ((dx*dx + dy*dy + dz*dz) <= excise_r2) return;

    Real vol = size(m).dx1 * size(m).dx2 * size(m).dx3;
    Real mass = rho * vol;
    Real r2 = dx*dx + dy*dy + dz*dz + soft2;
    r2 = Kokkos::fmax(r2, static_cast<Real>(1.0e-30));
    const Real rs = Kokkos::sqrt(r2);
    const Real factor = -bh_force_pair::PotentialSlope(rs, bh_mass, newton_g, rg)/rs;
    Real ax = factor * dx;
    Real ay = factor * dy;
    Real az = factor * dz;

    acc.the_array[0] += mass;
    acc.the_array[1] += mass * ax;
    acc.the_array[2] += mass * ay;
    acc.the_array[3] += mass * az;
  }, Kokkos::Sum<array_sum::GlobalSum>(sums));

  Real local_buf[4];
  for (int n = 0; n < 4; ++n) local_buf[n] = sums.the_array[n];
  Real global_buf[4];
#if MPI_PARALLEL_ENABLED
  MPI_Allreduce(local_buf, global_buf, 4, MPI_ATHENA_REAL, MPI_SUM, MPI_COMM_WORLD);
#else
  for (int n = 0; n < 4; ++n) global_buf[n] = local_buf[n];
#endif

  Real total_mass = global_buf[0];
  if (!(total_mass > 0.0) || !std::isfinite(total_mass)) return false;

  Real cm_ax = global_buf[1] / total_mass;
  Real cm_ay = global_buf[2] / total_mass;
  Real cm_az = global_buf[3] / total_mass;
  if (!std::isfinite(cm_ax) || !std::isfinite(cm_ay) || !std::isfinite(cm_az)) {
    return false;
  }

  frame_ax = cm_ax;
  frame_ay = cm_ay;
  frame_az = cm_az;
  return true;
}

void RotateAboutY(Real &x, Real &y, Real &z, const Real angle) {
  Real c = std::cos(angle);
  Real s = std::sin(angle);
  Real x_new = c*x + s*z;
  Real z_new = -s*x + c*z;
  x = x_new;
  z = z_new;
}

//----------------------------------------------------------------------------------------
//! \fn void RefineTDEAMR()
//! \brief The <tde_amr> hook: the shared passes of tde_amr.hpp with the Newtonian orbit
//! model, then the block budget about the BH.

void RefineTDEAMR(MeshBlockPack *pmbp) {
  Mesh *pm = pmbp->pmesh;
  tde_amr::Params p = tde_amr_params_global;
  p.bhx = bh_x_global;
  p.bhy = bh_y_global;
  p.bhz = bh_z_global;
  p.bhvx = bh_vx_global;
  p.bhvy = bh_vy_global;
  p.bhvz = bh_vz_global;
  tde_amr::FinishParams(pm, p);
  const tde_amr::NewtonianOrbit orbit{p, pmbp->phydro->w0};
  tde_amr::SetTargetFlags(pmbp, p, pmbp->phydro->w0, orbit);
  tde_amr::LimitToBudget(pm, pmbp, p, bh_x_global, bh_y_global, bh_z_global);
  pm->pmr->refine_flag.template sync<DevExeSpace>();
}

//----------------------------------------------------------------------------------------
//! \fn void ConfigureTDEAMR()
//! \brief Refuses the retired AMR keys, reads the <tde_amr> block if present and prints
//! the derived quantities.  r_star and mass_ratio give r_t; (sx,sy,sz) and (svx,svy,svz)
//! are the initial star position and velocity relative to the BH, whose cross product is
//! the orbit normal.

void ConfigureTDEAMR(ParameterInput *pin, Mesh *pm, const bool restart, const Real r_star,
                     const Real mass_ratio, const Real r_peri, const Real theta_bh,
                     const Real sx,
                     const Real sy, const Real sz, const Real svx, const Real svy,
                     const Real svz) {
  auto fatal = [](const std::string &msg) {
    std::cout << "### FATAL ERROR in ProblemGenerator::TDEExternal" << std::endl
              << msg << std::endl;
    std::exit(EXIT_FAILURE);
  };
  std::string keys = "cells_per_radian, cells_per_thickness, spine_frac, "
                     "midplane_finest, bound_only, rho_a";
  for (const auto &k : tde_amr::RuleKeys()) keys += ", " + k;

  // The <problem> rules bh_max_amr*, unbound_amr*, stream_shell_* and every <tde_amr> key
  // not listed above are retired, and the ambient cut is <tde_amr>/rho_a, never
  // <problem>/rho_a (which nothing would read).  A restart file carries the keys of the
  // run that wrote it, so a restart only warns.
  std::string bad;
  for (auto &blk : pin->block) {
    for (auto &ln : blk.line) {
      const std::string &k = ln.param_name;
      const bool retired = blk.block_name == "problem" &&
          (k.compare(0, 10, "bh_max_amr") == 0 || k.compare(0, 11, "unbound_amr") == 0 ||
           k.compare(0, 13, "stream_shell_") == 0 || k == "rho_a");
      const bool unknown = blk.block_name == "tde_amr" &&
          !tde_amr::IsSchemeKey(k) && !tde_amr::IsRuleKey(k);
      if (retired || unknown) bad += (bad.empty() ? "" : ", ") + blk.block_name + "/" + k;
    }
  }
  if (!bad.empty()) {
    const std::string msg = "retired or unknown TDE AMR keys: " + bad + ".  The TDE "
        "refinement is the <tde_amr> block with an <amr_criterionN> method = user, and "
        "its only keys are " + keys + ".";
    if (!restart) fatal(msg + "  Remove these keys from the input.");
    if (global_variable::my_rank == 0) {
      std::cout << "WARNING (tde_external): " << msg << "  They come from the restart "
                << "file and are ignored." << std::endl;
    }
  }

  tde_amr_enable_global = pin->DoesBlockExist("tde_amr");
  if (tde_amr_enable_global) tde_amr::InitHold(pin, restart);
  bool has_user = false;
  for (auto &blk : pin->block) {
    if (blk.block_name.compare(0, 13, "amr_criterion") != 0) continue;
    if (pin->DoesParameterExist(blk.block_name, "method") &&
        pin->GetString(blk.block_name, "method") == "user") {
      has_user = true;
    }
  }
  if (!tde_amr_enable_global) {
    if (has_user && pm->adaptive) {
      fatal("an <amr_criterionN> with method = user needs the <tde_amr> block (keys " +
            keys + ").");
    }
    return;
  }
  if (!pm->adaptive || pm->pmr == nullptr) {
    fatal("<tde_amr> requires <mesh_refinement>/refinement = adaptive.");
  }
  if (!has_user) fatal("<tde_amr> needs an <amr_criterionN> block with method = user.");

  tde_amr::Params p{};
  {
    std::string bad_value = tde_amr::ReadSchemeKeys(pin, p);
    if (bad_value.empty()) bad_value = tde_amr::ReadRuleKeys(pin, p);
    if (!bad_value.empty()) fatal(bad_value);
  }
  const bool rho_a_set = p.rho_a > 0.0;
  if (!rho_a_set) p.rho_a = 100.0*std::max(hydro_dfloor_global, static_cast<Real>(0.0));
  const Real r_t = r_star*std::cbrt(mass_ratio);
  const Real a_mb = r_t*r_t/(2.0*r_star);
  p.r_c = 2.0*a_mb;
  p.r_floor = r_peri;
  p.r_sink = bh_inner_boundary_global ? bh_excise_radius_global : 0.0;
  p.root_level = pm->root_level;
  p.max_level = pm->max_level;
  p.gm = newton_g_global*bh_mass_global;
  p.soft2 = bh_soft_global*bh_soft_global;
  p.rg = bh_rg_global;

  tde_amr::SetOrbitalPlane(p, theta_bh, sx, sy, sz, svx, svy, svz);

  // Innermost ring edge: the excision radius (the softening without excision), never
  // below one finest cell.
  const Real finest =
      pm->mesh_size.dx1/static_cast<Real>(1 << (p.max_level - p.root_level));
  p.rbin_min = std::max(bh_inner_boundary_global ? bh_excise_radius_global :
                        bh_soft_global, finest);
  p.log_rbin_min = std::log(p.rbin_min);
  p.band_floor = (pm->three_d ? pm->mesh_size.dx3 : pm->mesh_size.dx1)/
                 static_cast<Real>(1 << (p.max_level - p.root_level));
  tde_amr_params_global = p;

  if (global_variable::my_rank == 0) {
    tde_amr::PrintBanner(p, pm, r_t, {rho_a_set ? "<tde_amr>/rho_a" : "100 dfloor",
                                      "m_c = -div(v) dx/c_s, c_s^2 = (10/9) e/rho",
                                      "the infalling stream's pericentre", ""});
  }
}

void CapHydroLATFactorsNearBH(Mesh *pm, int max_factor, int *lat_factor_eachmb) {
  if (!bh_inner_boundary_global || bh_excise_radius_global <= 0.0 ||
      pm == nullptr || pm->pmb_pack == nullptr || pm->pmb_pack->pmb == nullptr ||
      lat_factor_eachmb == nullptr || max_factor <= 1 || pm->nmb_total <= 0) {
    return;
  }

  MeshBlockPack *pmbp = pm->pmb_pack;
  auto &size = pmbp->pmb->mb_size;
  auto &mb_gid = pmbp->pmb->mb_gid;
  const int nmb = pmbp->nmb_thispack;

  auto cap_gid_to_one = [&](const int gid) {
    if (gid < 0 || gid >= pm->nmb_total) return;
    lat_factor_eachmb[gid] = 1;
  };

  auto &nghbr = pmbp->pmb->nghbr;
  const int nnghbr = pmbp->pmb->nnghbr;

  // In the translating frame the BH drifts through the mesh during a window, so protect
  // the neighbourhood it sweeps before the next factor rebuild.
  Real frame_travel = 0.0;
  if (use_translating_frame_global || bh_live_global) {
    const Real window_dt = std::max(static_cast<Real>(0.0), pm->dt)*
        static_cast<Real>(std::max(1, max_factor));
    const Real speed = std::sqrt(SQR(bh_vx_global) + SQR(bh_vy_global) +
                                 SQR(bh_vz_global));
    const Real accel = std::sqrt(SQR(frame_ax_global) + SQR(frame_ay_global) +
                                 SQR(frame_az_global)) +
        (bh_live_global ? std::sqrt(SQR(bh_gas_ax_global) + SQR(bh_gas_ay_global) +
                                    SQR(bh_gas_az_global)) : static_cast<Real>(0.0));
    frame_travel = speed*window_dt + static_cast<Real>(0.5)*accel*window_dt*window_dt;
  }

  for (int m = 0; m < nmb; ++m) {
    const int gid = mb_gid.h_view(m);
    if (gid < 0 || gid >= pm->nmb_total) continue;

    const Real dx =
        (bh_x_global < size.h_view(m).x1min) ? (size.h_view(m).x1min - bh_x_global) :
        ((bh_x_global > size.h_view(m).x1max) ? (bh_x_global - size.h_view(m).x1max) :
         static_cast<Real>(0.0));
    const Real dy =
        (bh_y_global < size.h_view(m).x2min) ? (size.h_view(m).x2min - bh_y_global) :
        ((bh_y_global > size.h_view(m).x2max) ? (bh_y_global - size.h_view(m).x2max) :
         static_cast<Real>(0.0));
    const Real dz =
        (bh_z_global < size.h_view(m).x3min) ? (size.h_view(m).x3min - bh_z_global) :
        ((bh_z_global > size.h_view(m).x3max) ? (bh_z_global - size.h_view(m).x3max) :
         static_cast<Real>(0.0));

    const Real cell_diag = std::sqrt(size.h_view(m).dx1*size.h_view(m).dx1 +
                                     size.h_view(m).dx2*size.h_view(m).dx2 +
                                     size.h_view(m).dx3*size.h_view(m).dx3);
    Real guard_radius = bh_excise_radius_global +
        static_cast<Real>(std::max(1, pm->mb_indcs.ng))*cell_diag;
    if (use_translating_frame_global || bh_live_global) guard_radius += frame_travel;
    if (dx*dx + dy*dy + dz*dz <= guard_radius*guard_radius) {
      cap_gid_to_one(gid);
      for (int n = 0; n < nnghbr; ++n) {
        cap_gid_to_one(nghbr.h_view(m, n).gid);
      }
    }
  }

#if MPI_PARALLEL_ENABLED
  if (global_variable::nranks > 1) {
    MPI_Allreduce(MPI_IN_PLACE, lat_factor_eachmb, pm->nmb_total, MPI_INT, MPI_MIN,
                  MPI_COMM_WORLD);
  }
#endif
}

void SyncProblemRuntimeState() {
  // Export the evolving TDE state through the generic runtime interface so core
  // modules do not need to include this pgen directly.
  problem_runtime::SetExcisionState(bh_inner_boundary_global, bh_excise_radius_global,
                                    bh_excise_density_global, bh_excise_eint_global);
  problem_runtime::SetLiveBHState(use_translating_frame_global,
                                  bh_x_global, bh_y_global, bh_z_global,
                                  bh_vx_global, bh_vy_global, bh_vz_global,
                                  orbit_center_x_global, orbit_center_y_global,
                                  orbit_center_z_global,
                                  frame_vx_global, frame_vy_global, frame_vz_global);
  problem_runtime::SetExternalBHPotential(true, bh_mass_global, bh_soft_global,
                                          newton_g_global, bh_rg_global);
  problem_runtime::SetExternalBHGravitySourceCoupling(true);
  problem_runtime::SetFrameStepState(frame_step_valid_global,
                                     frame_step_time_global, frame_step_dt_global,
                                     frame_step_x0_global, frame_step_y0_global,
                                     frame_step_z0_global,
                                     frame_step_vx0_global, frame_step_vy0_global,
                                     frame_step_vz0_global,
                                     frame_step_ax0_global, frame_step_ay0_global,
                                     frame_step_az0_global,
                                     frame_step_x1_global, frame_step_y1_global,
                                     frame_step_z1_global,
                                     frame_step_vx1_global, frame_step_vy1_global,
                                     frame_step_vz1_global,
                                     frame_step_ax1_global, frame_step_ay1_global,
                                     frame_step_az1_global);
  problem_runtime::SetBHStepState(bh_step_valid_global, bh_step_time_global,
      bh_step_dt_global,
                                  bh_step_x0_global, bh_step_y0_global, bh_step_z0_global,
                                  bh_step_vx0_global, bh_step_vy0_global, bh_step_vz0_global,
                                  bh_step_x1_global, bh_step_y1_global, bh_step_z1_global,
                                  bh_step_vx1_global, bh_step_vy1_global, bh_step_vz1_global);
}

void InvalidateFrameBHStepState() {
  frame_step_valid_global = false;
  frame_step_cycle_global = -1;
  frame_step_time_global = 0.0;
  frame_step_dt_global = 0.0;
  bh_step_valid_global = false;
  bh_step_cycle_global = -1;
  bh_step_time_global = 0.0;
  bh_step_dt_global = 0.0;
}

void RestoreOptionalFrameBHStepState(ParameterInput *pin) {
  InvalidateFrameBHStepState();
  if (pin == nullptr) return;
  if (!pin->DoesParameterExist("problem", "frame_step_state_valid") ||
      !pin->DoesParameterExist("problem", "bh_step_state_valid")) {
    return;
  }
  if (!pin->GetBoolean("problem", "frame_step_state_valid") ||
      !pin->GetBoolean("problem", "bh_step_state_valid")) {
    return;
  }

  const char *frame_keys[] = {
      "frame_step_time", "frame_step_dt",
      "frame_step_x0", "frame_step_y0", "frame_step_z0",
      "frame_step_vx0", "frame_step_vy0", "frame_step_vz0",
      "frame_step_ax0", "frame_step_ay0", "frame_step_az0",
      "frame_step_x1", "frame_step_y1", "frame_step_z1",
      "frame_step_vx1", "frame_step_vy1", "frame_step_vz1",
      "frame_step_ax1", "frame_step_ay1", "frame_step_az1"};
  const char *bh_keys[] = {
      "bh_step_time", "bh_step_dt",
      "bh_step_x0", "bh_step_y0", "bh_step_z0",
      "bh_step_vx0", "bh_step_vy0", "bh_step_vz0",
      "bh_step_x1", "bh_step_y1", "bh_step_z1",
      "bh_step_vx1", "bh_step_vy1", "bh_step_vz1"};
  for (const char *key : frame_keys) {
    if (!pin->DoesParameterExist("problem", key)) return;
  }
  for (const char *key : bh_keys) {
    if (!pin->DoesParameterExist("problem", key)) return;
  }

  frame_step_time_global = pin->GetReal("problem", "frame_step_time");
  frame_step_dt_global = pin->GetReal("problem", "frame_step_dt");
  bh_step_time_global = pin->GetReal("problem", "bh_step_time");
  bh_step_dt_global = pin->GetReal("problem", "bh_step_dt");
  if (frame_step_dt_global <= 0.0 || bh_step_dt_global <= 0.0) {
    InvalidateFrameBHStepState();
    return;
  }

  frame_step_x0_global = pin->GetReal("problem", "frame_step_x0");
  frame_step_y0_global = pin->GetReal("problem", "frame_step_y0");
  frame_step_z0_global = pin->GetReal("problem", "frame_step_z0");
  frame_step_vx0_global = pin->GetReal("problem", "frame_step_vx0");
  frame_step_vy0_global = pin->GetReal("problem", "frame_step_vy0");
  frame_step_vz0_global = pin->GetReal("problem", "frame_step_vz0");
  frame_step_ax0_global = pin->GetReal("problem", "frame_step_ax0");
  frame_step_ay0_global = pin->GetReal("problem", "frame_step_ay0");
  frame_step_az0_global = pin->GetReal("problem", "frame_step_az0");
  frame_step_x1_global = pin->GetReal("problem", "frame_step_x1");
  frame_step_y1_global = pin->GetReal("problem", "frame_step_y1");
  frame_step_z1_global = pin->GetReal("problem", "frame_step_z1");
  frame_step_vx1_global = pin->GetReal("problem", "frame_step_vx1");
  frame_step_vy1_global = pin->GetReal("problem", "frame_step_vy1");
  frame_step_vz1_global = pin->GetReal("problem", "frame_step_vz1");
  frame_step_ax1_global = pin->GetReal("problem", "frame_step_ax1");
  frame_step_ay1_global = pin->GetReal("problem", "frame_step_ay1");
  frame_step_az1_global = pin->GetReal("problem", "frame_step_az1");

  bh_step_x0_global = pin->GetReal("problem", "bh_step_x0");
  bh_step_y0_global = pin->GetReal("problem", "bh_step_y0");
  bh_step_z0_global = pin->GetReal("problem", "bh_step_z0");
  bh_step_vx0_global = pin->GetReal("problem", "bh_step_vx0");
  bh_step_vy0_global = pin->GetReal("problem", "bh_step_vy0");
  bh_step_vz0_global = pin->GetReal("problem", "bh_step_vz0");
  bh_step_x1_global = pin->GetReal("problem", "bh_step_x1");
  bh_step_y1_global = pin->GetReal("problem", "bh_step_y1");
  bh_step_z1_global = pin->GetReal("problem", "bh_step_z1");
  bh_step_vx1_global = pin->GetReal("problem", "bh_step_vx1");
  bh_step_vy1_global = pin->GetReal("problem", "bh_step_vy1");
  bh_step_vz1_global = pin->GetReal("problem", "bh_step_vz1");

  frame_step_valid_global = true;
  bh_step_valid_global = true;
}

inline Real ClampUnitInterval(const Real x) {
  return std::fmax(0.0, std::fmin(1.0, x));
}

inline void LaneEmdenRHS(const Real xi, const Real theta, const Real dtheta,
                         const Real n, Real &dtheta_dxi, Real &ddtheta_dxi) {
  dtheta_dxi = dtheta;
  if (xi <= 0.0) {
    ddtheta_dxi = -1.0/3.0;
  } else {
    ddtheta_dxi = -2.0*dtheta/xi - ThetaPow(theta, n);
  }
}

KOKKOS_INLINE_FUNCTION
Real CubicHermiteValue(const Real y0, const Real m0, const Real y1, const Real m1,
                       const Real h, const Real t) {
  Real t2 = t*t;
  Real t3 = t2*t;
  Real h00 = (2.0*t3 - 3.0*t2 + 1.0);
  Real h10 = (t3 - 2.0*t2 + t);
  Real h01 = (-2.0*t3 + 3.0*t2);
  Real h11 = (t3 - t2);
  return h00*y0 + h10*h*m0 + h01*y1 + h11*h*m1;
}

KOKKOS_INLINE_FUNCTION
Real CubicHermiteSlope(const Real y0, const Real m0, const Real y1, const Real m1,
                       const Real h, const Real t) {
  if (h <= 0.0) return m0;
  Real t2 = t*t;
  Real dh00 = (6.0*t2 - 6.0*t);
  Real dh10 = (3.0*t2 - 4.0*t + 1.0);
  Real dh01 = (-6.0*t2 + 6.0*t);
  Real dh11 = (3.0*t2 - 2.0*t);
  Real dtheta_dt = dh00*y0 + dh10*h*m0 + dh01*y1 + dh11*h*m1;
  return dtheta_dt / h;
}

void InterpolateBHStepState(const Real t, Real &x, Real &y, Real &z,
                            Real &vx, Real &vy, Real &vz) {
  if (!bh_step_valid_global || bh_step_dt_global <= 0.0) {
    x = bh_x_global;
    y = bh_y_global;
    z = bh_z_global;
    vx = bh_vx_global;
    vy = bh_vy_global;
    vz = bh_vz_global;
    return;
  }

  Real s = ClampUnitInterval((t - bh_step_time_global) / bh_step_dt_global);
  x = CubicHermiteValue(bh_step_x0_global, bh_step_vx0_global,
                        bh_step_x1_global, bh_step_vx1_global, bh_step_dt_global, s);
  y = CubicHermiteValue(bh_step_y0_global, bh_step_vy0_global,
                        bh_step_y1_global, bh_step_vy1_global, bh_step_dt_global, s);
  z = CubicHermiteValue(bh_step_z0_global, bh_step_vz0_global,
                        bh_step_z1_global, bh_step_vz1_global, bh_step_dt_global, s);
  vx = CubicHermiteSlope(bh_step_x0_global, bh_step_vx0_global,
                         bh_step_x1_global, bh_step_vx1_global, bh_step_dt_global, s);
  vy = CubicHermiteSlope(bh_step_y0_global, bh_step_vy0_global,
                         bh_step_y1_global, bh_step_vy1_global, bh_step_dt_global, s);
  vz = CubicHermiteSlope(bh_step_z0_global, bh_step_vz0_global,
                         bh_step_z1_global, bh_step_vz1_global, bh_step_dt_global, s);
}

inline void FrameAccelerationAtTime(const Real t, Real &ax, Real &ay, Real &az) {
  if (!frame_step_valid_global || frame_step_dt_global <= 0.0) {
    ax = frame_ax_global;
    ay = frame_ay_global;
    az = frame_az_global;
    return;
  }

  Real s = ClampUnitInterval((t - frame_step_time_global) / frame_step_dt_global);
  ax = (1.0 - s) * frame_step_ax0_global + s * frame_step_ax1_global;
  ay = (1.0 - s) * frame_step_ay0_global + s * frame_step_ay1_global;
  az = (1.0 - s) * frame_step_az0_global + s * frame_step_az1_global;
}

//----------------------------------------------------------------------------------------
//! \brief Prepare the frame and BH state for the current timestep.
//!
//! In translating-frame mode, the frame position/velocity are integrated
//! continuously in time, while the frame acceleration is taken to be the
//! mass-weighted BH acceleration of density-selected debris. This removes the
//! bulk COM acceleration without introducing per-cycle coordinate resets.
//!
//! In fixed-frame mode, the frame state is identically zero and the BH remains
//! fixed at its current mesh position.
//!
//! The frame state for the current timestep is:
//!   1. Use the existing continuous frame center/velocity as the step start state
//!   2. Measure the selected-gas mean BH acceleration
//!   3. Advance the frame center/velocity with that acceleration over dt
//!   4. Derive BH position in simulation frame:  bh_sim = bh_inertial - frame_center
//!
//! The start (_x0) and end (_x1) state pairs are stored so that
//! InterpolateBHStepState / InterpolateFrameStepState can provide sub-step
//! values via cubic Hermite interpolation during RK stages.
//!
//! Without LAT, called once per cycle from TDEExternalGravitySource (guarded by
//! cycle/time match).  With LAT the translating frame is advanced per window by
//! PrepareTranslatingFrameLATWindow instead.

void AdvanceTranslatingFrameStep(Mesh *pm, const Real step_dt,
                                 const bool lat_window = false);
void AdvanceFrozenFrameLiveBHStep(Mesh *pm, const Real step_dt);

// Gas acceleration of the BH at a simulation-frame position: the reciprocal
// (volume-weighted opposite gas) force or the self-gravity sample.  MPI collectives.
bool SampleBHGasAcceleration(Mesh *pm, const Real x, const Real y, const Real z,
                             Real &ax, Real &ay, Real &az) {
  if (bh_reciprocal_force_global) {
    return bh_dynamics::SampleReciprocalAcceleration(pm, x, y, z, bh_mass_global,
        bh_soft_global, newton_g_global, bh_rg_global, bh_analytic_pair_global, ax, ay,
        az);
  }
  return bh_dynamics::SampleSelfGravityAcceleration(pm, x, y, z, ax, ay, az);
}

//! \brief Advance the live BH inertial state over [pm->time, pm->time + step_dt] and
//! store its simulation-frame step pair.  f0/fv0 and f1/fv1 are the frame center and
//! velocity at the step ends (zero without the translating frame).  The orbit step is
//! star_bh_collision's leapfrog: acceleration sampled at the start, then kick-drift-kick
//! with the endpoint force sampled at the drifted position against the current gas.
//! Collectives: synchronized points only.
void AdvanceLiveBHStep(Mesh *pm, const Real step_dt, const Real f0[3], const Real fv0[3],
                       const Real f1[3], const Real fv1[3]) {
  const std::array<Real, 3> x0 = {bh_inertial_x_global, bh_inertial_y_global,
                                  bh_inertial_z_global};
  const std::array<Real, 3> v0 = {bh_inertial_vx_global, bh_inertial_vy_global,
                                  bh_inertial_vz_global};
  std::array<Real, 3> a0 = {bh_gas_ax_global, bh_gas_ay_global, bh_gas_az_global};
  const std::array<Real, 3> previous_a = a0;
  // The gas potential is not solved yet on a new mesh (the first step after a remap).
  // The step then keeps the gas acceleration carried by the source restart.
  const bool phi_ready = pm->pmb_pack == nullptr || pm->pmb_pack->pgrav == nullptr ||
                         pm->pmb_pack->pgrav->phi_valid;
  if (!phi_ready) {
    a0 = previous_a;
  } else if (SampleBHGasAcceleration(pm, x0[0] - f0[0], x0[1] - f0[1], x0[2] - f0[2],
                                     a0[0], a0[1], a0[2])) {
    bh_gas_ax_global = a0[0];
    bh_gas_ay_global = a0[1];
    bh_gas_az_global = a0[2];
  } else if (pm->pmb_pack != nullptr && pm->pmb_pack->pgrav != nullptr) {
    std::cerr << "### FATAL ERROR in AdvanceLiveBHStep\n"
              << "No valid gas-potential acceleration sample at the BH position.\n";
    std::exit(EXIT_FAILURE);
  } else {
    a0 = previous_a;
  }
  std::array<Real, 3> x1, v1;
  {
    const Real end_frame[3] = {f1[0], f1[1], f1[2]};
    auto force = [pm, end_frame, phi_ready, previous_a](
                     const std::array<Real, 3> &position, std::array<Real, 3> &a) {
      if (pm->pmb_pack->pgrav == nullptr) {
        a = {0.0, 0.0, 0.0};
        return true;
      }
      if (!phi_ready) {
        a = previous_a;
        return true;
      }
      return SampleBHGasAcceleration(pm, position[0] - end_frame[0],
          position[1] - end_frame[1], position[2] - end_frame[2], a[0], a[1], a[2]);
    };
    if (!star_bh_orbit::LeapfrogKDK(x0, v0, a0, step_dt, force, x1, v1)) {
      std::cerr << "### FATAL ERROR: invalid leapfrog BH endpoint force or state\n";
      std::exit(EXIT_FAILURE);
    }
  }
  bh_step_x0_global = x0[0] - f0[0];
  bh_step_y0_global = x0[1] - f0[1];
  bh_step_z0_global = x0[2] - f0[2];
  bh_step_vx0_global = v0[0] - fv0[0];
  bh_step_vy0_global = v0[1] - fv0[1];
  bh_step_vz0_global = v0[2] - fv0[2];
  bh_step_x1_global = x1[0] - f1[0];
  bh_step_y1_global = x1[1] - f1[1];
  bh_step_z1_global = x1[2] - f1[2];
  bh_step_vx1_global = v1[0] - fv1[0];
  bh_step_vy1_global = v1[1] - fv1[1];
  bh_step_vz1_global = v1[2] - fv1[2];
  for (int d = 0; d < 3; ++d) {
    bh_window_start_v[d] = v0[d];
    bh_window_frame_dv[d] = fv1[d] - fv0[d];
    bh_window_frame_v1[d] = fv1[d];
  }
  bh_inertial_x_global = x1[0];
  bh_inertial_y_global = x1[1];
  bh_inertial_z_global = x1[2];
  bh_inertial_vx_global = v1[0];
  bh_inertial_vy_global = v1[1];
  bh_inertial_vz_global = v1[2];
  bh_x_global = bh_step_x1_global;
  bh_y_global = bh_step_y1_global;
  bh_z_global = bh_step_z1_global;
  bh_vx_global = bh_step_vx1_global;
  bh_vy_global = bh_step_vy1_global;
  bh_vz_global = bh_step_vz1_global;
}

// Work of the frame's fictitious force on the live BH over the step just taken:
// Delta K_sim = M v_avg . (Delta v_inertial - Delta v_frame), and the frame part is
// -M v_avg . Delta v_frame exactly (v_avg the mean of the step-end simulation-frame
// velocities).  Call once the step-end velocity is final.
void AccountLiveBHFrameWork() {
  if (!energy_ledger_global || !bh_live_global) return;
  const Real vs0[3] = {bh_step_vx0_global, bh_step_vy0_global, bh_step_vz0_global};
  const Real vs1[3] = {bh_vx_global, bh_vy_global, bh_vz_global};
  Real work = 0.0;
  for (int d = 0; d < 3; ++d) {
    work -= bh_mass_global*0.5*(vs0[d] + vs1[d])*bh_window_frame_dv[d];
  }
  frame_work_total += work;
}

void PrepareFrameBHStepState(Mesh *pm) {
  if (pm->dt <= 0.0) return;
  if (!use_translating_frame_global && bh_live_global) {
    // Live BH in the frozen (non-translating) frame.  With LAT the window hook owns it.
    if (hydro_lat_enabled_global) return;
    if (frame_step_valid_global && bh_step_valid_global &&
        bh_step_cycle_global == pm->ncycle && bh_step_time_global == pm->time &&
        bh_step_dt_global == pm->dt) {
      return;
    }
    AdvanceFrozenFrameLiveBHStep(pm, pm->dt);
    return;
  }
  if (!use_translating_frame_global) {
    if (frame_step_valid_global && bh_step_valid_global &&
        frame_step_cycle_global == pm->ncycle && bh_step_cycle_global == pm->ncycle &&
        frame_step_time_global == pm->time && bh_step_time_global == pm->time &&
        frame_step_dt_global == pm->dt && bh_step_dt_global == pm->dt) {
      return;
    }

    frame_step_cycle_global = pm->ncycle;
    frame_step_time_global = pm->time;
    frame_step_dt_global = pm->dt;
    bh_step_cycle_global = pm->ncycle;
    bh_step_time_global = pm->time;
    bh_step_dt_global = pm->dt;

    frame_step_x0_global = 0.0;
    frame_step_y0_global = 0.0;
    frame_step_z0_global = 0.0;
    frame_step_vx0_global = 0.0;
    frame_step_vy0_global = 0.0;
    frame_step_vz0_global = 0.0;
    frame_step_ax0_global = 0.0;
    frame_step_ay0_global = 0.0;
    frame_step_az0_global = 0.0;
    frame_step_x1_global = 0.0;
    frame_step_y1_global = 0.0;
    frame_step_z1_global = 0.0;
    frame_step_vx1_global = 0.0;
    frame_step_vy1_global = 0.0;
    frame_step_vz1_global = 0.0;
    frame_step_ax1_global = 0.0;
    frame_step_ay1_global = 0.0;
    frame_step_az1_global = 0.0;

    bh_step_x0_global = bh_x_global;
    bh_step_y0_global = bh_y_global;
    bh_step_z0_global = bh_z_global;
    bh_step_vx0_global = 0.0;
    bh_step_vy0_global = 0.0;
    bh_step_vz0_global = 0.0;
    bh_step_x1_global = bh_x_global;
    bh_step_y1_global = bh_y_global;
    bh_step_z1_global = bh_z_global;
    bh_step_vx1_global = 0.0;
    bh_step_vy1_global = 0.0;
    bh_step_vz1_global = 0.0;

    frame_step_valid_global = true;
    bh_step_valid_global = true;
    SyncProblemRuntimeState();
    return;
  }

  // With LAT the frame is advanced once per synchronization window by
  // PrepareTranslatingFrameLATWindow; LAT source and fixup passes only read it.
  if (hydro_lat_enabled_global) return;

  if (frame_step_valid_global && bh_step_valid_global &&
      frame_step_cycle_global == pm->ncycle && bh_step_cycle_global == pm->ncycle &&
      frame_step_time_global == pm->time && bh_step_time_global == pm->time &&
      frame_step_dt_global == pm->dt && bh_step_dt_global == pm->dt) {
    return;
  }

  AdvanceTranslatingFrameStep(pm, pm->dt);
}

//----------------------------------------------------------------------------------------
//! \brief Advance the translating frame over [pm->time, pm->time + step_dt].
//!
//! Measures the selected-gas mean BH acceleration with an MPI collective, so every rank
//! must call it at the same mesh time: once per cycle without LAT, once per LAT window
//! from the driver's common-time hook with LAT.  The acceleration is held constant over
//! the step, so the stored Hermite pair reproduces the quadratic frame trajectory exactly
//! at every block's own stage time inside it.

void AdvanceTranslatingFrameStep(Mesh *pm, const Real step_dt, const bool lat_window) {
  // A LAT window holds the frame acceleration for many fine steps, so there it varies
  // linearly, extrapolated from the previous window's sample (bootstrapped after a
  // restart or a window more than twice the previous one); the frame trajectory is then
  // second order in the window length.  The per-cycle path keeps its constant sample.
  const Real previous_time = frame_step_time_global;
  const bool previous_valid = lat_window && frame_step_valid_global;
  const Real previous_ax = frame_step_ax0_global;
  const Real previous_ay = frame_step_ay0_global;
  const Real previous_az = frame_step_az0_global;
  frame_step_cycle_global = pm->ncycle;
  frame_step_time_global = pm->time;
  frame_step_dt_global = step_dt;
  bh_step_cycle_global = pm->ncycle;
  bh_step_time_global = pm->time;
  bh_step_dt_global = step_dt;

  frame_step_x0_global = orbit_center_x_global;
  frame_step_y0_global = orbit_center_y_global;
  frame_step_z0_global = orbit_center_z_global;
  frame_step_vx0_global = frame_vx_global;
  frame_step_vy0_global = frame_vy_global;
  frame_step_vz0_global = frame_vz_global;
  frame_step_ax0_global = frame_ax_global;
  frame_step_ay0_global = frame_ay_global;
  frame_step_az0_global = frame_az_global;

  Real measured_ax = 0.0;
  Real measured_ay = 0.0;
  Real measured_az = 0.0;
  if (MeasureSelectedFrameAcceleration(pm, measured_ax, measured_ay, measured_az)) {
    frame_step_ax0_global = measured_ax;
    frame_step_ay0_global = measured_ay;
    frame_step_az0_global = measured_az;
  }

  if (!bh_live_global) {
    bh_step_x0_global = bh_inertial_x_global - frame_step_x0_global;
    bh_step_y0_global = bh_inertial_y_global - frame_step_y0_global;
    bh_step_z0_global = bh_inertial_z_global - frame_step_z0_global;
    bh_step_vx0_global = -frame_step_vx0_global;
    bh_step_vy0_global = -frame_step_vy0_global;
    bh_step_vz0_global = -frame_step_vz0_global;
  }

  // Advance the continuous translating frame over this timestep using the
  // selected-gas mean BH acceleration. This removes the selected debris COM
  // acceleration without resetting the coordinates to the instantaneous COM.
  const Real previous_dt = pm->time - previous_time;
  const Real time_tol = static_cast<Real>(1024.0)*std::numeric_limits<Real>::epsilon()*
      std::max(static_cast<Real>(1.0), std::abs(pm->time));
  if (previous_valid && previous_dt > time_tol && step_dt <= 2.0*previous_dt) {
    const Real dt = step_dt;
    const Real jx = (frame_step_ax0_global - previous_ax)/previous_dt;
    const Real jy = (frame_step_ay0_global - previous_ay)/previous_dt;
    const Real jz = (frame_step_az0_global - previous_az)/previous_dt;
    frame_step_ax1_global = frame_step_ax0_global + jx*dt;
    frame_step_ay1_global = frame_step_ay0_global + jy*dt;
    frame_step_az1_global = frame_step_az0_global + jz*dt;
    frame_step_vx1_global = frame_step_vx0_global + dt*frame_step_ax0_global + 0.5*dt*dt*jx;
    frame_step_vy1_global = frame_step_vy0_global + dt*frame_step_ay0_global + 0.5*dt*dt*jy;
    frame_step_vz1_global = frame_step_vz0_global + dt*frame_step_az0_global + 0.5*dt*dt*jz;
    frame_step_x1_global = frame_step_x0_global + dt*frame_step_vx0_global +
                           0.5*dt*dt*frame_step_ax0_global + dt*dt*dt*jx/6.0;
    frame_step_y1_global = frame_step_y0_global + dt*frame_step_vy0_global +
                           0.5*dt*dt*frame_step_ay0_global + dt*dt*dt*jy/6.0;
    frame_step_z1_global = frame_step_z0_global + dt*frame_step_vz0_global +
                           0.5*dt*dt*frame_step_az0_global + dt*dt*dt*jz/6.0;
  } else {
    frame_step_ax1_global = frame_step_ax0_global;
    frame_step_ay1_global = frame_step_ay0_global;
    frame_step_az1_global = frame_step_az0_global;
    frame_step_vx1_global = frame_step_vx0_global + frame_step_ax0_global*step_dt;
    frame_step_vy1_global = frame_step_vy0_global + frame_step_ay0_global*step_dt;
    frame_step_vz1_global = frame_step_vz0_global + frame_step_az0_global*step_dt;
    frame_step_x1_global = frame_step_x0_global +
                           0.5*(frame_step_vx0_global + frame_step_vx1_global)*step_dt;
    frame_step_y1_global = frame_step_y0_global +
                           0.5*(frame_step_vy0_global + frame_step_vy1_global)*step_dt;
    frame_step_z1_global = frame_step_z0_global +
                           0.5*(frame_step_vz0_global + frame_step_vz1_global)*step_dt;
  }

  // End-of-step BH in simulation frame = bh_inertial - frame_center
  if (bh_live_global) {
    const Real f0[3] = {frame_step_x0_global, frame_step_y0_global, frame_step_z0_global};
    const Real fv0[3] = {frame_step_vx0_global, frame_step_vy0_global,
                         frame_step_vz0_global};
    const Real f1[3] = {frame_step_x1_global, frame_step_y1_global, frame_step_z1_global};
    const Real fv1[3] = {frame_step_vx1_global, frame_step_vy1_global,
                         frame_step_vz1_global};
    AdvanceLiveBHStep(pm, step_dt, f0, fv0, f1, fv1);
  } else {
    bh_step_vx1_global = -frame_step_vx1_global;
    bh_step_vy1_global = -frame_step_vy1_global;
    bh_step_vz1_global = -frame_step_vz1_global;
    bh_step_x1_global = bh_inertial_x_global - frame_step_x1_global;
    bh_step_y1_global = bh_inertial_y_global - frame_step_y1_global;
    bh_step_z1_global = bh_inertial_z_global - frame_step_z1_global;
  }

  frame_step_valid_global = true;
  bh_step_valid_global = true;

  // Commit end-of-step frame state for the next cycle's start-of-step
  orbit_center_x_global = frame_step_x1_global;
  orbit_center_y_global = frame_step_y1_global;
  orbit_center_z_global = frame_step_z1_global;
  frame_vx_global = frame_step_vx1_global;
  frame_vy_global = frame_step_vy1_global;
  frame_vz_global = frame_step_vz1_global;
  frame_ax_global = frame_step_ax1_global;
  frame_ay_global = frame_step_ay1_global;
  frame_az_global = frame_step_az1_global;
  frame_state_time_global = pm->time + step_dt;

  // Commit end-of-step BH state (used by metadata, AMR, and excision)
  bh_x_global = bh_step_x1_global;
  bh_y_global = bh_step_y1_global;
  bh_z_global = bh_step_z1_global;
  bh_vx_global = bh_step_vx1_global;
  bh_vy_global = bh_step_vy1_global;
  bh_vz_global = bh_step_vz1_global;
  if (!bh_reciprocal_force_global) AccountLiveBHFrameWork();
  SyncProblemRuntimeState();
}

//! \brief Live BH step without a translating frame (frozen BH-inertial coordinates).
void AdvanceFrozenFrameLiveBHStep(Mesh *pm, const Real step_dt) {
  frame_step_cycle_global = pm->ncycle;
  frame_step_time_global = pm->time;
  frame_step_dt_global = step_dt;
  bh_step_cycle_global = pm->ncycle;
  bh_step_time_global = pm->time;
  bh_step_dt_global = step_dt;
  frame_step_x0_global = frame_step_y0_global = frame_step_z0_global = 0.0;
  frame_step_vx0_global = frame_step_vy0_global = frame_step_vz0_global = 0.0;
  frame_step_ax0_global = frame_step_ay0_global = frame_step_az0_global = 0.0;
  frame_step_x1_global = frame_step_y1_global = frame_step_z1_global = 0.0;
  frame_step_vx1_global = frame_step_vy1_global = frame_step_vz1_global = 0.0;
  frame_step_ax1_global = frame_step_ay1_global = frame_step_az1_global = 0.0;
  const Real zero[3] = {0.0, 0.0, 0.0};
  AdvanceLiveBHStep(pm, step_dt, zero, zero, zero, zero);
  frame_step_valid_global = true;
  bh_step_valid_global = true;
  SyncProblemRuntimeState();
}

//----------------------------------------------------------------------------------------
//! \brief LAT window hook: advance the translating frame over the whole window.
//!
//! The driver calls this on every rank at the window's common start time, after the
//! window length is final and before any rank-local bin runs, so the collective in
//! MeasureSelectedFrameAcceleration is safe here and nowhere inside the window.

void PrepareTranslatingFrameLATWindow(Mesh *pm, const Real window_dt) {
  if ((!use_translating_frame_global && !bh_live_global) || pm == nullptr ||
      window_dt <= 0.0) {
    return;
  }
  const Real t0 = pm->time;
  const Real t1 = t0 + window_dt;
  const Real time_tol = static_cast<Real>(1024.0)*std::numeric_limits<Real>::epsilon()*
      std::max(static_cast<Real>(1.0), std::max(std::abs(t0), std::abs(t1)));
  if (frame_step_valid_global && bh_step_valid_global &&
      frame_step_time_global <= t0 + time_tol &&
      frame_step_time_global + frame_step_dt_global >= t1 - time_tol) {
    return;
  }
  if (use_translating_frame_global) {
    AdvanceTranslatingFrameStep(pm, window_dt, true);
  } else {
    AdvanceFrozenFrameLiveBHStep(pm, window_dt);
  }
  if (bh_reciprocal_force_global) {
    if (bh_impulse_window_open) {
      std::cerr << "Previous BH impulse window was not reconciled\n";
      std::exit(EXIT_FAILURE);
    }
    for (Real &value : bh_local_gas_impulse) value = 0.0;
    bh_impulse_window_open = true;
  }
}

// Rank-local stage impulse of the BH force on the gas (no MPI).
void AccumulateTDEBHGasImpulse(Mesh *pm, const Real final_dt) {
  if (!bh_impulse_window_open) {
    std::cerr << "BH impulse ledger used outside its synchronization window\n";
    std::exit(EXIT_FAILURE);
  }
  bh_dynamics::AccumulateStageGasImpulse(pm, final_dt, bh_analytic_pair_global,
                                         bh_local_gas_impulse);
}

// LAT window end, after every bin and reflux and before output/AMR: reconcile the BH
// velocity with the gas impulse (as star_bh_collision) and reduce the frame work.
void CompleteTDELATWindow(Mesh *pm) {
  if (bh_reciprocal_force_global) {
    const Real scale = std::max(static_cast<Real>(1.0), std::abs(pm->time));
    const Real tolerance = 2048.0*std::numeric_limits<Real>::epsilon()*scale;
    if (!bh_impulse_window_open ||
        std::abs(pm->time-bh_step_time_global-bh_step_dt_global) > tolerance) {
      std::cerr << "BH impulse reconciliation requires a complete synchronized LAT "
                << "window\n";
      std::exit(EXIT_FAILURE);
    }
    Real impulse[3] = {bh_local_gas_impulse[0], bh_local_gas_impulse[1],
                       bh_local_gas_impulse[2]};
#if MPI_PARALLEL_ENABLED
    MPI_Allreduce(bh_local_gas_impulse, impulse, 3, MPI_ATHENA_REAL, MPI_SUM,
                  MPI_COMM_WORLD);
#endif
    Real end_v[3];
    bh_last_impulse_residual = 0.0;
    for (int d = 0; d < 3; ++d) {
      end_v[d] = bh_window_start_v[d]-impulse[d]/bh_mass_global;
      bh_total_gas_impulse[d] += impulse[d];
      const Real residual = bh_mass_global*(end_v[d]-bh_window_start_v[d])+impulse[d];
      bh_last_impulse_residual = std::max(bh_last_impulse_residual, std::abs(residual));
      bh_local_gas_impulse[d] = 0.0;
    }
    // Keep the already-used position predictor; the next window starts from the
    // corrected velocity.
    bh_inertial_vx_global = end_v[0];
    bh_inertial_vy_global = end_v[1];
    bh_inertial_vz_global = end_v[2];
    bh_vx_global = end_v[0] - bh_window_frame_v1[0];
    bh_vy_global = end_v[1] - bh_window_frame_v1[1];
    bh_vz_global = end_v[2] - bh_window_frame_v1[2];
    bh_impulse_window_open = false;
    AccountLiveBHFrameWork();
  }
  if (energy_ledger_global) {
    // The frame's share of the window-end refund joins its stage work.
    frame_work_local += problem_runtime::TakeFrameRefundWork();
    Real work = frame_work_local;
#if MPI_PARALLEL_ENABLED
    MPI_Allreduce(&frame_work_local, &work, 1, MPI_ATHENA_REAL, MPI_SUM, MPI_COMM_WORLD);
#endif
    frame_work_total += work;
    frame_work_local = 0.0;
  }
  SyncProblemRuntimeState();
}

// Sum of u0(IEN) dV over the blocks this source pass updates.
Real ActiveSourceGasEnergy(MeshBlockPack *pmbp, const int nwork1) {
  auto &indcs = pmbp->pmesh->mb_indcs;
  const int is = indcs.is, js = indcs.js, ks = indcs.ks;
  const int nx1 = indcs.nx1, nx2 = indcs.nx2, nx3 = indcs.nx3;
  const int per_block = nx1*nx2*nx3;
  const bool lat_enabled = pmbp->lat_active_mask_enabled;
  const auto active = pmbp->lat_active_indices.d_view;
  const auto size = pmbp->pmb->mb_size.d_view;
  const auto u0 = pmbp->phydro->u0;
  Real total = 0.0;
  Kokkos::parallel_reduce("tde_source_energy",
      Kokkos::RangePolicy<>(DevExeSpace(), 0, (nwork1 + 1)*per_block),
  KOKKOS_LAMBDA(const int index, Real &sum) {
    const int a = index/per_block;
    const int m = lat_enabled ? active(a) : a;
    const int cell = index - a*per_block;
    const int k = cell/(nx1*nx2) + ks;
    const int j = (cell/nx1)%nx2 + js;
    const int i = cell%nx1 + is;
    sum += u0(m, IEN, k, j, i)*size(m).dx1*size(m).dx2*size(m).dx3;
  }, Kokkos::Sum<Real>(total));
  return total;
}

//! \brief History: the shared energy ledger columns (simulation-frame K_BH and BH
//! state) followed by W_frame, the cumulative fictitious-force work on gas and BH.
//! Closure: tot-E + K_BH + U_bhgas + U_self + B_E + B_Wself + B_Wbh - E_remap - E_floor
//! - E_asym - W_frame is constant for a live BH with the reciprocal force.  With
void TDEExternalHistory(HistoryData *pdata, Mesh *pm) {
  bh_dynamics::LedgerHistory(pdata, pm, 0, bh_mass_global, bh_x_global, bh_y_global,
                             bh_z_global, bh_vx_global, bh_vy_global, bh_vz_global);
  const int n = pdata->nhist;
  pdata->nhist = n + 1;
  pdata->label[n] = "W_frame";
  const Real share = 1.0/static_cast<Real>(global_variable::nranks);
  pdata->hdata[n] = share*frame_work_total + frame_work_local;
}

void StoreTDEExternalMetadata(ParameterInput *pin, Mesh *pm) {
  if (pin == nullptr) return;
  if (pm != nullptr && pm->pmb_pack != nullptr && pm->pmb_pack->pgrav != nullptr) {
    pm->pmb_pack->pgrav->StoreBoundaryFluxMetadata(pin);
  }
  tde_external::StoreRuntimeMetadata(pin);
  if (bh_live_global) {
    pin->SetReal("problem", "bh_live_inertial_x", bh_inertial_x_global);
    pin->SetReal("problem", "bh_live_inertial_y", bh_inertial_y_global);
    pin->SetReal("problem", "bh_live_inertial_z", bh_inertial_z_global);
    pin->SetReal("problem", "bh_live_inertial_vx", bh_inertial_vx_global);
    pin->SetReal("problem", "bh_live_inertial_vy", bh_inertial_vy_global);
    pin->SetReal("problem", "bh_live_inertial_vz", bh_inertial_vz_global);
    pin->SetReal("problem", "bh_live_ax", bh_gas_ax_global);
    pin->SetReal("problem", "bh_live_ay", bh_gas_ay_global);
    pin->SetReal("problem", "bh_live_az", bh_gas_az_global);
    if (bh_reciprocal_force_global) {
      pin->SetReal("problem", "bh_pair_gas_impulse_x", bh_total_gas_impulse[0]);
      pin->SetReal("problem", "bh_pair_gas_impulse_y", bh_total_gas_impulse[1]);
      pin->SetReal("problem", "bh_pair_gas_impulse_z", bh_total_gas_impulse[2]);
      pin->SetReal("problem", "bh_pair_impulse_residual", bh_last_impulse_residual);
    }
  }
  if (energy_ledger_global) {
    pin->SetReal("problem", "frame_work_total", frame_work_total + frame_work_local);
  }
}

//! \brief <remap>/auto_expand hook (remap::ExpandNeed).  The needed gas is the gas above
//! rho_need (<remap>/expand_rho_min, 100 dfloor by default).  With bound_only
//! (<remap>/expand_bound_only) it is only the gas that is bound to the BH and reaches its
//! next pericentre before tlim on its Kepler orbit about the BH (velocities relative to
//! the BH, so a translating frame gives the same orbit).  A needed cell within buf of a
//! face marks that face.  A full scan in the BH frame also returns the union of the exact
//! bounding boxes of the ellipses of the needed bound gas; needed gas on an open orbit
//! has no such reach, so a face it comes near is pushed out by its axis length, as
//! without a prediction.  In a translating frame the orbit is no fixed curve of the grid,
//! so it returns no box.  The cheap scan skips the blocks that reach no buffer slab.
void TDEExternalExpandNeed(Mesh *pm, const Real *buf, const bool full, const Real tlim,
                           const Real rho_need, const bool bound_only,
                           remap::ExpandNeed &need) {
  need = remap::ExpandNeed();
  need.center[0] = bh_x_global;
  need.center[1] = bh_y_global;
  need.center[2] = bh_z_global;
  MeshBlockPack *pmbp = pm->pmb_pack;
  if (pmbp == nullptr || pmbp->phydro == nullptr) return;
  auto &indcs = pm->mb_indcs;
  const int is = indcs.is, js = indcs.js, ks = indcs.ks;
  const int nx1 = indcs.nx1, nx2 = indcs.nx2, nx3 = indcs.nx3;
  const int nji = nx2*nx1;
  const int nkji = nx3*nji;
  const int nmb = pmbp->nmb_thispack;
  const bool act2 = pm->multi_d, act3 = pm->three_d;
  const Real lo1 = pm->mesh_size.x1min, hi1 = pm->mesh_size.x1max;
  const Real lo2 = pm->mesh_size.x2min, hi2 = pm->mesh_size.x2max;
  const Real lo3 = pm->mesh_size.x3min, hi3 = pm->mesh_size.x3max;
  const Real len1 = hi1 - lo1, len2 = hi2 - lo2, len3 = hi3 - lo3;
  const Real b1 = buf[0], b2 = act2 ? buf[1] : 0.0, b3 = act3 ? buf[2] : 0.0;
  const Real gm = newton_g_global*bh_mass_global;
  const Real bh_mass = bh_mass_global, newton_g = newton_g_global, rg = bh_rg_global;
  const Real soft = bh_soft_global;
  const Real t_left = tlim - pm->time;
  const Real bhx = bh_x_global, bhy = bh_y_global, bhz = bh_z_global;
  const Real bhvx = bh_vx_global, bhvy = bh_vy_global, bhvz = bh_vz_global;
  const bool scan_all = full;
  const bool want_box = full && !use_translating_frame_global;
  const Real two_pi = 2.0*M_PI;
  auto &size = pmbp->pmb->mb_size;
  auto &w0_ = pmbp->phydro->w0;
  // bits 0-5: near faces, bit 6: any needed gas; ext: -min and max of the box per axis
  DvceArray1D<int> bits_d("tde_expand_bits", 1);
  DvceArray1D<Real> ext_d("tde_expand_ext", 6);
  Kokkos::deep_copy(ext_d, -std::numeric_limits<Real>::max());
  Kokkos::parallel_for("tde_expand_need",
  Kokkos::TeamPolicy<>(DevExeSpace(), nmb, Kokkos::AUTO),
  KOKKOS_LAMBDA(TeamMember_t tmember) {
    const int m = tmember.league_rank();
    const Real x1l = size.d_view(m).x1min, x1u = size.d_view(m).x1max;
    const Real x2l = size.d_view(m).x2min, x2u = size.d_view(m).x2max;
    const Real x3l = size.d_view(m).x3min, x3u = size.d_view(m).x3max;
    if (!scan_all && !(x1l < lo1 + b1 || x1u > hi1 - b1 ||
                       (act2 && (x2l < lo2 + b2 || x2u > hi2 - b2)) ||
                       (act3 && (x3l < lo3 + b3 || x3u > hi3 - b3)))) return;
    int tbits = 0;
    Real t0, t1, t2, t3, t4, t5;
    Kokkos::parallel_reduce(Kokkos::TeamThreadRange(tmember, nkji),
    [=](const int idx, int &lbits, Real &n1, Real &n2, Real &n3,
        Real &p1, Real &p2, Real &p3) {
      const int kk = idx/nji;
      const int jj = (idx - kk*nji)/nx1;
      const int ii = idx - kk*nji - jj*nx1;
      const Real x = CellCenterX(ii, nx1, x1l, x1u);
      const Real y = CellCenterX(jj, nx2, x2l, x2u);
      const Real z = CellCenterX(kk, nx3, x3l, x3u);
      int fb = 0;
      if (x - lo1 < b1) fb |= 1;
      if (hi1 - x < b1) fb |= 2;
      if (act2 && y - lo2 < b2) fb |= 4;
      if (act2 && hi2 - y < b2) fb |= 8;
      if (act3 && z - lo3 < b3) fb |= 16;
      if (act3 && hi3 - z < b3) fb |= 32;
      if (!scan_all && fb == 0) return;
      const int k = kk + ks, j = jj + js, i = ii + is;
      if (!(w0_(m, IDN, k, j, i) > rho_need)) return;
      const Real dx = x - bhx, dy = y - bhy, dz = z - bhz;
      const Real r = Kokkos::sqrt(dx*dx + dy*dy + dz*dz);
      if (!(r > 0.0)) return;
      const Real ux = w0_(m, IVX, k, j, i) - bhvx;
      const Real uy = w0_(m, IVY, k, j, i) - bhvy;
      const Real uz = w0_(m, IVZ, k, j, i) - bhvz;
      // Bound in the applied potential; the Kepler elements below use the Newtonian
      // energy, which a Wegg potential matches far from the hole where this gas lies.
      const Real ekin = 0.5*(ux*ux + uy*uy + uz*uz);
      const Real eps_bh = ekin + bh_force_pair::PotentialOfRadius(
          Kokkos::sqrt(r*r + soft*soft), bh_mass, newton_g, rg);
      const Real eps = ekin - gm/r;
      if (!(eps_bh < 0.0) || !(eps < 0.0)) {
        if (bound_only) return;
        lbits |= (fb | 64);
        if (!want_box) return;
        if (fb & 1) n1 = Kokkos::fmax(n1, len1 - lo1);
        if (fb & 2) p1 = Kokkos::fmax(p1, hi1 + len1);
        if (fb & 4) n2 = Kokkos::fmax(n2, len2 - lo2);
        if (fb & 8) p2 = Kokkos::fmax(p2, hi2 + len2);
        if (fb & 16) n3 = Kokkos::fmax(n3, len3 - lo3);
        if (fb & 32) p3 = Kokkos::fmax(p3, hi3 + len3);
        return;
      }
      const Real a = -gm/(2.0*eps);
      const Real hx = dy*uz - dz*uy, hy = dz*ux - dx*uz, hz = dx*uy - dy*ux;
      // eccentricity vector (u x h)/GM - d/|d|; its direction is the pericentre
      Real ex = (uy*hz - uz*hy)/gm - dx/r;
      Real ey = (uz*hx - ux*hz)/gm - dy/r;
      Real ez = (ux*hy - uy*hx)/gm - dz/r;
      Real ecc = Kokkos::sqrt(ex*ex + ey*ey + ez*ez);
      if (ecc >= 1.0e-8) {
        // Kepler time to the next pericentre
        const Real cos_ea = Kokkos::fmin(Kokkos::fmax((1.0 - r/a)/ecc, -1.0), 1.0);
        Real ea = Kokkos::acos(cos_ea);
        if (dx*ux + dy*uy + dz*uz < 0.0) ea = two_pi - ea;
        const Real mean = ea - ecc*Kokkos::sin(ea);
        const Real t_peri = (two_pi - mean)/Kokkos::sqrt(gm/(a*a*a));
        if (bound_only && t_peri > t_left) return;
        ex /= ecc;
        ey /= ecc;
        ez /= ecc;
      } else {
        // circular: needed; any in-plane unit vector serves as e_hat
        ecc = 0.0;
        ex = dx/r;
        ey = dy/r;
        ez = dz/r;
      }
      lbits |= (fb | 64);
      if (!want_box) return;
      // exact bounding box of the ellipse: centre c = BH - a e e_hat, half width along
      // axis k sqrt((a e_hat_k)^2 + (b q_hat_k)^2), q_hat = j_hat x e_hat
      Real qx = 0.0, qy = 0.0, qz = 0.0;
      const Real hmag = Kokkos::sqrt(hx*hx + hy*hy + hz*hz);
      if (hmag > 0.0) {
        const Real jx = hx/hmag, jy = hy/hmag, jz = hz/hmag;
        qx = jy*ez - jz*ey;
        qy = jz*ex - jx*ez;
        qz = jx*ey - jy*ex;
      }
      const Real bsemi = a*Kokkos::sqrt(Kokkos::fmax(1.0 - ecc*ecc, 0.0));
      const Real cx = bhx - a*ecc*ex, cy = bhy - a*ecc*ey, cz = bhz - a*ecc*ez;
      const Real wx = Kokkos::sqrt(a*a*ex*ex + bsemi*bsemi*qx*qx);
      const Real wy = Kokkos::sqrt(a*a*ey*ey + bsemi*bsemi*qy*qy);
      const Real wz = Kokkos::sqrt(a*a*ez*ez + bsemi*bsemi*qz*qz);
      n1 = Kokkos::fmax(n1, wx - cx);
      n2 = Kokkos::fmax(n2, wy - cy);
      n3 = Kokkos::fmax(n3, wz - cz);
      p1 = Kokkos::fmax(p1, cx + wx);
      p2 = Kokkos::fmax(p2, cy + wy);
      p3 = Kokkos::fmax(p3, cz + wz);
    }, Kokkos::BOr<int>(tbits), Kokkos::Max<Real>(t0), Kokkos::Max<Real>(t1),
       Kokkos::Max<Real>(t2), Kokkos::Max<Real>(t3), Kokkos::Max<Real>(t4),
       Kokkos::Max<Real>(t5));
    Kokkos::single(Kokkos::PerTeam(tmember), [&]() {
      if (tbits == 0) return;
      Kokkos::atomic_or(&bits_d(0), tbits);
      if (!want_box || (tbits & 64) == 0) return;
      Kokkos::atomic_max(&ext_d(0), t0);
      Kokkos::atomic_max(&ext_d(1), t1);
      Kokkos::atomic_max(&ext_d(2), t2);
      Kokkos::atomic_max(&ext_d(3), t3);
      Kokkos::atomic_max(&ext_d(4), t4);
      Kokkos::atomic_max(&ext_d(5), t5);
    });
  });
  auto bits_h = Kokkos::create_mirror_view_and_copy(HostMemSpace(), bits_d);
  auto ext_h = Kokkos::create_mirror_view_and_copy(HostMemSpace(), ext_d);
  // one reduction: every entry is a maximum (box minima negated, flags 0/1)
  Real red[13];
  for (int n = 0; n < 6; ++n) red[n] = ext_h(n);
  for (int n = 0; n < 7; ++n) red[6 + n] = ((bits_h(0) >> n) & 1) ? 1.0 : 0.0;
#if MPI_PARALLEL_ENABLED
  MPI_Allreduce(MPI_IN_PLACE, red, 13, MPI_ATHENA_REAL, MPI_MAX, MPI_COMM_WORLD);
#endif
  for (int f = 0; f < 6; ++f) need.near_face[f] = red[6 + f] > 0.0;
  need.have_box = want_box && red[12] > 0.0;
  if (need.have_box) {
    for (int d = 0; d < 3; ++d) {
      need.box_min[d] = -red[d];
      need.box_max[d] = red[3 + d];
    }
  }
}

void RestoreLiveBHMetadata(ParameterInput *pin) {
  auto get = [pin](const char *key, const Real fallback) {
    return pin->DoesParameterExist("problem", key) ? pin->GetReal("problem", key) :
                                                     fallback;
  };
  if (bh_live_global && pin->DoesParameterExist("problem", "bh_live_inertial_x")) {
    bh_inertial_x_global = get("bh_live_inertial_x", bh_inertial_x_global);
    bh_inertial_y_global = get("bh_live_inertial_y", bh_inertial_y_global);
    bh_inertial_z_global = get("bh_live_inertial_z", bh_inertial_z_global);
    bh_inertial_vx_global = get("bh_live_inertial_vx", 0.0);
    bh_inertial_vy_global = get("bh_live_inertial_vy", 0.0);
    bh_inertial_vz_global = get("bh_live_inertial_vz", 0.0);
    bh_gas_ax_global = get("bh_live_ax", 0.0);
    bh_gas_ay_global = get("bh_live_ay", 0.0);
    bh_gas_az_global = get("bh_live_az", 0.0);
  } else if (bh_live_global) {
    // The source was written without a live BH (it followed the two-body orbit), so the
    // BH starts from rest in the inertial frame with no stored gas pull.  This also keeps
    // a remap settle pass from carrying the velocity gathered during the settle steps.
    bh_inertial_vx_global = 0.0;
    bh_inertial_vy_global = 0.0;
    bh_inertial_vz_global = 0.0;
    bh_gas_ax_global = 0.0;
    bh_gas_ay_global = 0.0;
    bh_gas_az_global = 0.0;
  }
  if (bh_reciprocal_force_global) {
    bh_total_gas_impulse[0] = get("bh_pair_gas_impulse_x", 0.0);
    bh_total_gas_impulse[1] = get("bh_pair_gas_impulse_y", 0.0);
    bh_total_gas_impulse[2] = get("bh_pair_gas_impulse_z", 0.0);
    for (Real &value : bh_local_gas_impulse) value = 0.0;
    bh_impulse_window_open = false;
  }
  if (energy_ledger_global) {
    frame_work_total = get("frame_work_total", 0.0);
    frame_work_local = 0.0;
  }
}

bool RestoreFrameBHStateFromMetadata(ParameterInput *pin) {
  if (pin == nullptr) return false;
  if (!pin->DoesParameterExist("problem", "bh_live_state_valid")) return false;
  if (!pin->GetBoolean("problem", "bh_live_state_valid")) return false;

  const char *required_keys[] = {
      "bh_live_x", "bh_live_y", "bh_live_z",
      "bh_live_vx", "bh_live_vy", "bh_live_vz",
      "frame_live_x", "frame_live_y", "frame_live_z",
      "frame_live_vx", "frame_live_vy", "frame_live_vz"};
  for (const char *key : required_keys) {
    if (!pin->DoesParameterExist("problem", key)) return false;
  }

  bh_x_global = pin->GetReal("problem", "bh_live_x");
  bh_y_global = pin->GetReal("problem", "bh_live_y");
  bh_z_global = pin->GetReal("problem", "bh_live_z");
  bh_vx_global = pin->GetReal("problem", "bh_live_vx");
  bh_vy_global = pin->GetReal("problem", "bh_live_vy");
  bh_vz_global = pin->GetReal("problem", "bh_live_vz");
  orbit_center_x_global = pin->GetReal("problem", "frame_live_x");
  orbit_center_y_global = pin->GetReal("problem", "frame_live_y");
  orbit_center_z_global = pin->GetReal("problem", "frame_live_z");
  frame_vx_global = pin->GetReal("problem", "frame_live_vx");
  frame_vy_global = pin->GetReal("problem", "frame_live_vy");
  frame_vz_global = pin->GetReal("problem", "frame_live_vz");
  bh_inertial_x_global = orbit_center_x_global + bh_x_global;
  bh_inertial_y_global = orbit_center_y_global + bh_y_global;
  bh_inertial_z_global = orbit_center_z_global + bh_z_global;
  RestoreOptionalFrameBHStepState(pin);
  SyncProblemRuntimeState();
  return true;
}

inline void RK4Step(const Real xi, const Real h, const Real theta, const Real dtheta,
                    const Real n, Real &theta_out, Real &dtheta_out) {
  Real k1_t, k1_dt;
  LaneEmdenRHS(xi, theta, dtheta, n, k1_t, k1_dt);

  Real t2 = theta + 0.5*h*k1_t;
  Real dt2 = dtheta + 0.5*h*k1_dt;
  Real k2_t, k2_dt;
  LaneEmdenRHS(xi + 0.5*h, t2, dt2, n, k2_t, k2_dt);

  Real t3 = theta + 0.5*h*k2_t;
  Real dt3 = dtheta + 0.5*h*k2_dt;
  Real k3_t, k3_dt;
  LaneEmdenRHS(xi + 0.5*h, t3, dt3, n, k3_t, k3_dt);

  Real t4 = theta + h*k3_t;
  Real dt4 = dtheta + h*k3_dt;
  Real k4_t, k4_dt;
  LaneEmdenRHS(xi + h, t4, dt4, n, k4_t, k4_dt);

  theta_out = theta + (h/6.0)*(k1_t + 2.0*k2_t + 2.0*k3_t + k4_t);
  dtheta_out = dtheta + (h/6.0)*(k1_dt + 2.0*k2_dt + 2.0*k3_dt + k4_dt);
}

bool SolveLaneEmden(const Real n, LaneEmdenProfile &profile) {
  constexpr Real xi_max = 2.0e3;
  constexpr int max_steps = 5000000;
  constexpr Real atol = 1.0e-12;
  constexpr Real rtol = 1.0e-10;
  constexpr Real safety = 0.9;
  constexpr Real fac_min = 0.2;
  constexpr Real fac_max = 2.0;
  constexpr Real h_min = 1.0e-10;
  constexpr Real h_max = 5.0e-2;
  constexpr Real xi0 = 1.0e-6;

  profile.xi.clear();
  profile.theta.clear();
  profile.dtheta.clear();
  profile.xi.push_back(0.0);
  profile.theta.push_back(1.0);
  profile.dtheta.push_back(0.0);

  Real xi = xi0;
  Real theta = 1.0 - SQR(xi)/6.0 + n*SQR(SQR(xi))/120.0;
  Real dtheta = -xi/3.0 + n*xi*SQR(xi)/30.0;
  profile.xi.push_back(xi);
  profile.theta.push_back(theta);
  profile.dtheta.push_back(dtheta);

  Real h = 1.0e-4;

  for (int step = 0; step < max_steps; ++step) {
    if (xi >= xi_max) break;
    if (h < h_min) h = h_min;
    if (h > h_max) h = h_max;
    if (xi + h > xi_max) h = xi_max - xi;
    if (h <= 0.0) break;

    Real theta_big, dtheta_big;
    RK4Step(xi, h, theta, dtheta, n, theta_big, dtheta_big);

    Real theta_half, dtheta_half;
    RK4Step(xi, 0.5*h, theta, dtheta, n, theta_half, dtheta_half);
    Real theta_half2, dtheta_half2;
    RK4Step(xi + 0.5*h, 0.5*h, theta_half, dtheta_half, n, theta_half2, dtheta_half2);

    if (!std::isfinite(theta_big) || !std::isfinite(dtheta_big) ||
        !std::isfinite(theta_half2) || !std::isfinite(dtheta_half2)) {
      return false;
    }

    Real sc_t = atol + rtol*std::fmax(std::abs(theta_half2), std::abs(theta_big));
    Real sc_dt = atol + rtol*std::fmax(std::abs(dtheta_half2), std::abs(dtheta_big));
    Real err_t = std::abs(theta_half2 - theta_big) / (15.0*sc_t);
    Real err_dt = std::abs(dtheta_half2 - dtheta_big) / (15.0*sc_dt);
    Real err = std::fmax(err_t, err_dt);

    if (err <= 1.0) {
      Real xi_next = xi + h;
      Real theta_next = theta_half2;
      Real dtheta_next = dtheta_half2;

      if (theta_next <= 0.0) {
        Real tlo = 0.0;
        Real thi = 1.0;
        for (int it = 0; it < 80; ++it) {
          Real tm = 0.5*(tlo + thi);
          Real thm = CubicHermiteValue(theta, dtheta, theta_next, dtheta_next, h, tm);
          if (thm > 0.0) {
            tlo = tm;
          } else {
            thi = tm;
          }
        }
        Real troot = 0.5*(tlo + thi);
        profile.xi1 = xi + troot*h;
        profile.dtheta_xi1 =
            CubicHermiteSlope(theta, dtheta, theta_next, dtheta_next, h, troot);
        profile.xi.push_back(profile.xi1);
        profile.theta.push_back(0.0);
        profile.dtheta.push_back(profile.dtheta_xi1);
        return true;
      }

      profile.xi.push_back(xi_next);
      profile.theta.push_back(theta_next);
      profile.dtheta.push_back(dtheta_next);
      xi = xi_next;
      theta = theta_next;
      dtheta = dtheta_next;
    }

    Real fac = fac_max;
    if (err > 1.0e-30) {
      fac = safety*std::pow(1.0/err, 0.2);
      fac = std::fmax(fac_min, std::fmin(fac_max, fac));
    }
    h = std::fmax(h_min, std::fmin(h_max, h*fac));
  }

  return false;
}

KOKKOS_INLINE_FUNCTION
Real LaneEmdenTheta(const Real xi, const DvceArray1D<Real> xi_tab,
                    const DvceArray1D<Real> theta_tab,
                    const DvceArray1D<Real> dtheta_tab, const int npts) {
  if (xi <= 0.0) return 1.0;
  if (xi >= xi_tab(npts - 1)) return 0.0;

  int lo = 0;
  int hi = npts - 1;
  while (hi - lo > 1) {
    int mid = (lo + hi) / 2;
    if (xi_tab(mid) <= xi) {
      lo = mid;
    } else {
      hi = mid;
    }
  }

  Real x0 = xi_tab(lo);
  Real x1 = xi_tab(hi);
  if (x1 <= x0) return Kokkos::fmax(theta_tab(lo), 0.0);

  Real frac = (xi - x0) / (x1 - x0);
  Real theta = CubicHermiteValue(theta_tab(lo), dtheta_tab(lo),
                                 theta_tab(hi), dtheta_tab(hi), x1 - x0, frac);
  return Kokkos::fmax(theta, 0.0);
}

const char *StellarStructureModeName(const StellarStructureMode mode) {
  switch (mode) {
    case StellarStructureMode::legacy_polytrope:
      return "legacy_polytrope";
    case StellarStructureMode::eos_balanced:
      return "eos_balanced";
  }
  return "unknown";
}

bool ParseStellarStructureMode(const std::string &mode_name, StellarStructureMode &mode) {
  if (mode_name == "legacy_polytrope") {
    mode = StellarStructureMode::legacy_polytrope;
    return true;
  }
  if (mode_name == "eos_balanced") {
    mode = StellarStructureMode::eos_balanced;
    return true;
  }
  return false;
}

bool EOSGamma1FromRhoP(const EOS_Data &eos, const Real rho, const Real p, Real &gamma1) {
  if (!(rho > 0.0) || !(p > 0.0)) return false;
  if (eos.UsesTabulatedLTE()) {
    gamma1 = eos.HostGamma1FromRhoP(rho, p);
  } else if (eos.is_gamma_law) {
    gamma1 = eos.gamma;
  } else {
    return false;
  }
  return std::isfinite(gamma1) && (gamma1 > 0.0);
}

bool EOSPressureInActiveDomain(const EOS_Data &eos, const Real rho, const Real p) {
  if (!(rho > 0.0) || !(p > 0.0) || !std::isfinite(rho) || !std::isfinite(p)) {
    return false;
  }
  if (!(eos.UsesTabulatedLTE())) {
    return true;
  }
  if (!(eos.HostDensityInActiveDomain(rho))) {
    return false;
  }
  const Real p_floor = eos.HostTabulatedPressureFloor(rho);
  const Real p_ceil = eos.HostTabulatedPressureCeiling(rho);
  return (p >= p_floor) && (p <= p_ceil);
}

//! The Newtonian star model of stellar_profile.hpp: Inertia = ActiveDensity = rho, the
//! adiabatic index of the deck's EOS, and the table's (rho, T) box when it is tabulated.
struct NewtonianStarModel {
  const EOS_Data &eos;
  Real grav_const;
  Real G() const { return grav_const; }
  Real MassUnit() const { return 1.0; }
  bool InDomain(const Real rho, const Real p) const {
    return EOSPressureInActiveDomain(eos, rho, p);
  }
  bool Coefficients(const Real rho, const Real p, Real &gamma1, Real &inertia,
                    Real &active) const {
    if (!EOSGamma1FromRhoP(eos, rho, p, gamma1)) return false;
    inertia = rho;
    active = rho;
    return true;
  }
  void Densities(const Real rho, const Real, Real &inertia, Real &active) const {
    inertia = rho;
    active = rho;
  }
  bool HasPressureBounds() const { return eos.UsesTabulatedLTE(); }
  Real PressureFloor(const Real rho) const {
    return eos.HostTabulatedPressureFloor(rho);
  }
  Real PressureCeiling(const Real rho) const {
    return eos.HostTabulatedPressureCeiling(rho);
  }
  Real DensityMax() const {
    return std::exp(eos.saha_logrho_max) / std::max(eos.density_unit_cgs,
        static_cast<Real>(1.0e-300));
  }
};

KOKKOS_INLINE_FUNCTION
Real RadialProfileValue(const Real r, const DvceArray1D<Real> radius_tab,
                        const DvceArray1D<Real> value_tab, const int npts) {
  if (npts <= 0) return 0.0;
  if (r <= radius_tab(0)) return value_tab(0);
  if (r >= radius_tab(npts - 1)) return 0.0;

  int lo = 0;
  int hi = npts - 1;
  while (hi - lo > 1) {
    const int mid = (lo + hi) / 2;
    if (radius_tab(mid) <= r) {
      lo = mid;
    } else {
      hi = mid;
    }
  }

  const Real r0 = radius_tab(lo);
  const Real r1 = radius_tab(hi);
  if (!(r1 > r0)) return value_tab(lo);
  const Real frac = (r - r0) / (r1 - r0);
  return value_tab(lo) + frac * (value_tab(hi) - value_tab(lo));
}

//----------------------------------------------------------------------------------------
//! \fn Real AtmosphereColumn()
//! \brief I(r) = int_r^inf exp(-(r' - r)/L) G M/r'^2 dr', the pressure per unit base
//! density of the atmosphere rho exp(-(r' - r)/L) in hydrostatic equilibrium in the field
//! of the mass M inside it (trapezoid rule in u = (r' - r)/L out to u = 50).

Real AtmosphereColumn(const Real r, const Real length, const Real gm) {
  constexpr int kN = 2000;
  const Real du = 50.0/kN;
  Real sum = 0.0;
  for (int n = 0; n <= kN; ++n) {
    const Real rr = r + length*n*du;
    const Real f = std::exp(-n*du)/(rr*rr);
    sum += (n == 0 || n == kN) ? 0.5*f : f;
  }
  return gm*length*du*sum;
}

//----------------------------------------------------------------------------------------
//! \fn bool AttachAtmosphere()
//! \brief Replaces the EOS-balanced profile outward of r_s by the atmosphere
//! rho_s exp(-(r - r_s)/L) in hydrostatic equilibrium in the star's field,
//! P(r) = rho(r) I(r) (AtmosphereColumn), whose pressure falls to zero with its density.
//! r_s is the outermost node where the profile's pressure still exceeds rho I at the
//! node's density, so the pressure is continuous there (about where the profile's
//! pressure scale height is L).  The atmosphere is tabulated in steps of L/16 down to
//! rho_end and ends on a zero node, as the profile did.  m_old and m_new are the masses
//! outward of r_s before and after.  False if no node qualifies.

bool AttachAtmosphere(StellarRadialProfile &prof, const Real length, const Real gm,
                      const Real rho_end, Real &r_s, Real &rho_s, Real &p_s,
                      Real &m_old, Real &m_new) {
  const std::size_t n = prof.radius.size();
  std::size_t i_s = n;
  for (std::size_t i = n; i-- > 1;) {
    if (!(prof.density[i] > 0.0)) continue;
    if (prof.pressure[i] > prof.density[i]*
                           AtmosphereColumn(prof.radius[i], length, gm)) break;
    i_s = i;
  }
  if (i_s >= n) return false;
  auto shell_mass = [&](const std::size_t a) {
    return 2.0*M_PI*(prof.radius[a + 1] - prof.radius[a])*
           (SQR(prof.radius[a])*prof.density[a] +
            SQR(prof.radius[a + 1])*prof.density[a + 1]);
  };
  m_old = 0.0;
  for (std::size_t a = i_s; a + 1 < n; ++a) m_old += shell_mass(a);
  r_s = prof.radius[i_s];
  rho_s = prof.density[i_s];
  prof.radius.resize(i_s);
  prof.density.resize(i_s);
  prof.pressure.resize(i_s);
  const Real dr = length/16.0;
  for (int k = 0;; ++k) {
    const Real r = r_s + k*dr;
    const Real rho = rho_s*std::exp(-k*dr/length);
    prof.radius.push_back(r);
    prof.density.push_back(rho);
    prof.pressure.push_back(rho*AtmosphereColumn(r, length, gm));
    if (!(rho > rho_end)) break;
  }
  p_s = prof.pressure[i_s];
  prof.radius.push_back(prof.radius.back() + dr);
  prof.density.push_back(0.0);
  prof.pressure.push_back(0.0);
  m_new = 0.0;
  for (std::size_t a = i_s; a + 1 < prof.radius.size(); ++a) m_new += shell_mass(a);
  return true;
}

void TDEExternalGravitySource(Mesh *pm, const Real bdt) {
  MeshBlockPack *pmbp = pm->pmb_pack;
  if (pmbp->phydro == nullptr) return;

  PrepareFrameBHStepState(pm);

  auto &indcs = pm->mb_indcs;
  int is = indcs.is, ie = indcs.ie;
  int js = indcs.js, je = indcs.je;
  int ks = indcs.ks, ke = indcs.ke;
  int nmb = pmbp->nmb_thispack;

  auto &size = pmbp->pmb->mb_size;
  auto &u0 = pmbp->phydro->u0;
  const bool external_bh_gravity_source =
      pmbp->phydro->psrc != nullptr &&
      pmbp->phydro->psrc->external_bh_gravity &&
      problem_runtime::ExternalBHGravitySourceCouplingEnabled();
  const bool lat_enabled = pmbp->lat_active_mask_enabled;
  const bool lat_per_block_dt = pmbp->lat_per_block_timestep;
  const Real mesh_dt = pm->dt;
  const Real stage_weight =
      (lat_per_block_dt && mesh_dt > 0.0) ? (bdt/mesh_dt) : 0.0;
  auto lat_step_dt = pmbp->lat_step_dt.d_view;
  auto active_indices = pmbp->lat_active_indices.d_view;
  const int nwork1 = lat_enabled ? (pmbp->lat_nactive_thispack - 1) : (nmb - 1);
  if (nwork1 < 0) return;
  const bool eos_has_energy = pmbp->phydro->peos->eos_data.use_e;

  const Real t_src = (pm->pgen != nullptr && pm->pgen->user_srcs_time_valid) ?
      pm->pgen->user_srcs_time : problem_runtime::HydroStageTimeOr(pm->time);
  Real bhvx = 0.0, bhvy = 0.0, bhvz = 0.0;
  Real bhx, bhy, bhz;
  InterpolateBHStepState(t_src, bhx, bhy, bhz, bhvx, bhvy, bhvz);
  Real cx, cy, cz;
  FrameCenterAtTime(t_src, cx, cy, cz);

  Real gm = external_bh_gravity_source ? static_cast<Real>(0.0) :
      newton_g_global * bh_mass_global;
  const Real bh_mass = bh_mass_global, newton_g = newton_g_global, rg = bh_rg_global;
  const Real excise_r2 = bh_inner_boundary_global ?
      bh_excise_radius_global * bh_excise_radius_global : -1.0;
  Real eps2 = bh_soft_global * bh_soft_global;

  // Translating-frame correction using the measured selected-gas mean BH
  // acceleration. This removes the bulk selected-debris acceleration from the frame.
  Real ax_frame = 0.0;
  Real ay_frame = 0.0;
  Real az_frame = 0.0;
  FrameAccelerationAtTime(t_src, ax_frame, ay_frame, az_frame);
  bool apply_relax = relax_enable_global && (t_src <= relax_t_end_global);
  const Real relax_tau = relax_tau_global;
  const Real relax_fac = apply_relax ? std::exp(-bdt/relax_tau) : 1.0;
  if (gm == 0.0 && ax_frame == 0.0 && ay_frame == 0.0 && az_frame == 0.0 &&
      !apply_relax) {
    // Core external-BH source active, fixed frame, relaxation over: the kernel below
    // would only add exact zeros to every active cell.
    return;
  }
  const bool ledger_frame_work = energy_ledger_global && eos_has_energy &&
      bdt != 0.0 && pm->pgen != nullptr && pm->pgen->user_srcs_stage_valid;
  const Real frame_energy_before =
      ledger_frame_work ? ActiveSourceGasEnergy(pmbp, nwork1) : 0.0;
  Kokkos::Array<Real, 21> lat_relax_fac;
  for (int level = 0; level <= 20; ++level) {
    lat_relax_fac[level] = 1.0;
  }
  if (apply_relax && lat_per_block_dt) {
    for (int level = 0; level <= 20; ++level) {
      const int factor = 1 << level;
      const Real block_bdt = stage_weight*mesh_dt*static_cast<Real>(factor);
      lat_relax_fac[level] = std::exp(-block_bdt/relax_tau);
    }
  }
  auto lat_step_level = pmbp->lat_step_level.d_view;
  Real relax_r2 = relax_radius_global * relax_radius_global;
  Real rho_floor = bh_grav_rho_min_global;
  const Real rho_weight_floor = pmbp->phydro->peos->eos_data.dfloor;
  auto w0 = pmbp->phydro->w0;
  const auto flx1 = pmbp->phydro->FluxBand(pmbp->phydro->uflx.x1f);
  const auto flx2 = pmbp->phydro->FluxBand(pmbp->phydro->uflx.x2f);
  const auto flx3 = pmbp->phydro->FluxBand(pmbp->phydro->uflx.x3f);
  const bool multi_d = pm->multi_d;
  const bool three_d = pm->three_d;

  par_for("tde_bh_src", DevExeSpace(), 0, nwork1, ks, ke, js, je, is, ie,
  KOKKOS_LAMBDA(int a, int k, int j, int i) {
    const int m = lat_enabled ? active_indices(a) : a;
    const Real block_bdt = lat_per_block_dt ? stage_weight*lat_step_dt(m) : bdt;
    Real &x1min = size.d_view(m).x1min;
    Real &x1max = size.d_view(m).x1max;
    Real &x2min = size.d_view(m).x2min;
    Real &x2max = size.d_view(m).x2max;
    Real &x3min = size.d_view(m).x3min;
    Real &x3max = size.d_view(m).x3max;

    Real x = CellCenterX(i - is, indcs.nx1, x1min, x1max);
    Real y = CellCenterX(j - js, indcs.nx2, x2min, x2max);
    Real z = CellCenterX(k - ks, indcs.nx3, x3min, x3max);

    const Real rx = x - bhx;
    const Real ry = y - bhy;
    const Real rz = z - bhz;

    if (excise_r2 > 0.0 &&
        tde_external::InsideExcisionZone(x, y, z, bhx, bhy, bhz, excise_r2)) return;

    Real r2 = rx*rx + ry*ry + rz*rz + eps2;
    r2 = Kokkos::fmax(r2, static_cast<Real>(1.0e-30));
    const Real rs = Kokkos::sqrt(r2);
    // the pgen applies the BH pull itself only without external_bh_gravity_source
    const Real factor = (gm == 0.0) ? static_cast<Real>(0.0) :
        -bh_force_pair::PotentialSlope(rs, bh_mass, newton_g, rg)/rs;

    Real ax = factor * rx - ax_frame;
    Real ay = factor * ry - ay_frame;
    Real az = factor * rz - az_frame;

    // The stage source rho a and its work are taken on the stage state, as every RK
    // source is, and in the form SourceTerms::Gravity gives the BH pull this frame term
    // cancels: momentum from the stage primitives w0, work from the stage Godunov mass
    // fluxes (Mullen, Hanawa & Gammie 2021).  For a uniform acceleration (linear
    // potential) that work is the face-averaged mass flux times a, so the uniform part
    // of BH pull and frame correction now cancels to round-off cell by cell and stage
    // by stage.  The former impulse form E += 1/2 rho (v_new^2 - v_old^2) on the updated
    // u0 also counted the BH kick already in u0: it took 1/2 rho a^2 (beta dt)^2 per
    // stage, 3/8 rho a^2 dt^2 per rk2 step, a first-order energy sink (|a| ~ G M/r^2 is
    // the star's own surface gravity at the start of the TDE deck) that made the star
    // contract.
    const Real rho_s = w0(m, IDN, k, j, i);
    if (rho_s > 0.0) {
      // The frame correction removes the BH acceleration of the gas, so it is applied
      // with the coupling weight the BH force itself carries (utils/gravity_weight.hpp),
      // read off the same stage density.
      const Real w_bh = gravity_weight::Weight(rho_s, rho_weight_floor, rho_floor);
      if (w_bh > 0.0) {
        const Real coupled_dt = block_bdt * w_bh;
        u0(m, IM1, k, j, i) += coupled_dt * rho_s * ax;
        u0(m, IM2, k, j, i) += coupled_dt * rho_s * ay;
        u0(m, IM3, k, j, i) += coupled_dt * rho_s * az;
        if (eos_has_energy) {
          // Work only, so the dual-energy auxiliary thermal field stays untouched.  A
          // collapsed dimension has no flux; its stage momentum stands in.
          const Real fx = 0.5*(flx1(m, IDN, k, j, i) + flx1(m, IDN, k, j, i+1));
          const Real fy = multi_d ?
              0.5*(flx2(m, IDN, k, j, i) + flx2(m, IDN, k, j+1, i)) :
              rho_s*w0(m, IVY, k, j, i);
          const Real fz = three_d ?
              0.5*(flx3(m, IDN, k, j, i) + flx3(m, IDN, k+1, j, i)) :
              rho_s*w0(m, IVZ, k, j, i);
          u0(m, IEN, k, j, i) += coupled_dt * (fx*ax + fy*ay + fz*az);
        }
      }
    }
    // The relaxation damping keeps its own two-valued density cut.
    const Real rho = u0(m, IDN, k, j, i);
    if (rho <= 0.0 || rho < rho_floor) return;

    if (apply_relax) {
      Real rs2 = SQR(x - cx) + SQR(y - cy) + SQR(z - cz);
      if (rs2 <= relax_r2) {
        const int lat_level = lat_per_block_dt ? lat_step_level(m) : 0;
        const Real block_relax_fac =
            lat_per_block_dt ? lat_relax_fac[lat_level] : relax_fac;
        Real mom1 = u0(m, IM1, k, j, i);
        Real mom2 = u0(m, IM2, k, j, i);
        Real mom3 = u0(m, IM3, k, j, i);
        Real ke_rel_old = 0.0;
        if (eos_has_energy) {
          ke_rel_old = 0.5 * (mom1*mom1 + mom2*mom2 + mom3*mom3) / rho;
        }

        mom1 *= block_relax_fac;
        mom2 *= block_relax_fac;
        mom3 *= block_relax_fac;
        u0(m, IM1, k, j, i) = mom1;
        u0(m, IM2, k, j, i) = mom2;
        u0(m, IM3, k, j, i) = mom3;

        if (eos_has_energy) {
          Real ke_rel_new = 0.5 * (mom1*mom1 + mom2*mom2 + mom3*mom3) / rho;
          u0(m, IEN, k, j, i) += (ke_rel_new - ke_rel_old);
        }
      }
    }
  });
  // Coarse/fine faces: the frame's linear potential a_f . x enters the coarse/fine work
  // correction with the black hole's (SourceTerms::CoarseFineGravityWorkFlux), paid to
  // the coarse side through the restricted energy flux, so the pull and the frame still
  // cancel across a refinement boundary.
  if (ledger_frame_work) {
    // This stage's energy change enters the completed step with weight
    // FinalFluxWeight*dt/(beta*dt).  Without LAT every rank runs every stage, so the
    // reduction can happen here; with LAT it waits for the window end.
    const Real weight =
        pm->pgen->user_srcs_final_weight*pm->pgen->user_srcs_step_dt/bdt;
    frame_work_local +=
        (ActiveSourceGasEnergy(pmbp, nwork1) - frame_energy_before)*weight;
    if (!hydro_lat_enabled_global) {
      Real work = frame_work_local;
#if MPI_PARALLEL_ENABLED
      MPI_Allreduce(&frame_work_local, &work, 1, MPI_ATHENA_REAL, MPI_SUM,
                    MPI_COMM_WORLD);
#endif
      frame_work_total += work;
      frame_work_local = 0.0;
    }
  }
}

// MeshBlocks whose cells can reach the excision shell rewritten by
// TDEExternalHydroStateFixup.  Only the block geometry is cached; the LAT active mask
// changes every substep and is re-applied on every call.
struct ExcisionShellBlocks {
  bool valid = false;
  int nmb = -1;
  std::uint64_t topology_version = 0;
  Real bhx = 0.0, bhy = 0.0, bhz = 0.0;
  Real radius = -1.0;
  std::vector<char> candidate;
};
ExcisionShellBlocks excision_shell_blocks;

void TDEExternalHydroStateFixup(MeshBlockPack *pmbp, const Real time) {
  if (pmbp == nullptr || pmbp->pmesh == nullptr || pmbp->phydro == nullptr) return;
  // A stage-0 call outside a step (InitBoundaryValuesAndPrimitives) has no valid
  // pmesh->dt -- on a fresh start it runs before the first Mesh::NewTimeStep, and after a
  // remap it holds the source's last dt -- so it must not advance the translating frame
  // or a live BH; it reads the current frame and BH state instead.
  if ((!use_translating_frame_global && !bh_live_global) ||
      pmbp->pmesh->pgen->user_srcs_time_valid) {
    PrepareFrameBHStepState(pmbp->pmesh);
  }
  if (!bh_inner_boundary_global || bh_excise_radius_global <= 0.0) return;

  auto &indcs = pmbp->pmesh->mb_indcs;
  const int is = indcs.is, ie = indcs.ie;
  const int js = indcs.js, je = indcs.je;
  const int ks = indcs.ks, ke = indcs.ke;
  const int nmb = pmbp->nmb_thispack;
  const bool lat_enabled = pmbp->lat_active_mask_enabled;
  const int nscan = lat_enabled ? pmbp->lat_nactive_thispack : nmb;
  if (nscan <= 0) return;

  Real bhvx = 0.0, bhvy = 0.0, bhvz = 0.0;
  Real bhx, bhy, bhz;
  InterpolateBHStepState(time, bhx, bhy, bhz, bhvx, bhvy, bhvz);
  auto &size = pmbp->pmb->mb_size;
  auto &u0 = pmbp->phydro->u0;
  const bool eos_has_energy = pmbp->phydro->peos->eos_data.use_e;
  const bool dual_enabled = pmbp->phydro->use_dual_energy;
  const int dual_energy_idx = pmbp->phydro->dual_energy_idx;
  const Real bh_excise_radius = bh_excise_radius_global;
  const Real exact_floor_rho_max = hydro_dfloor_global * static_cast<Real>(10.0);
  const Real exact_floor_eint = bh_excise_eint_global;

  // Every cell of a block whose bounding box, grown by the excision radius plus the
  // cell diagonal (the kernel's own reach) plus one cell of slack, cannot reach the BH
  // returns at the distance test below, so skipping such blocks is a no-op.
  auto &cache = excision_shell_blocks;
  if (!cache.valid || cache.nmb != nmb ||
      cache.topology_version != pmbp->pmesh->topology_version ||
      cache.bhx != bhx || cache.bhy != bhy || cache.bhz != bhz ||
      cache.radius != bh_excise_radius) {
    cache.candidate.assign(nmb, 0);
    for (int m = 0; m < nmb; ++m) {
      const Real d1 = size.h_view(m).dx1;
      const Real d2 = size.h_view(m).dx2;
      const Real d3 = size.h_view(m).dx3;
      const Real cell_diag = std::sqrt(d1*d1 + d2*d2 + d3*d3);
      const Real guard = bh_excise_radius + cell_diag + std::max(d1, std::max(d2, d3));
      const Real ox =
          (bhx < size.h_view(m).x1min) ? (size.h_view(m).x1min - bhx) :
          ((bhx > size.h_view(m).x1max) ? (bhx - size.h_view(m).x1max) :
           static_cast<Real>(0.0));
      const Real oy =
          (bhy < size.h_view(m).x2min) ? (size.h_view(m).x2min - bhy) :
          ((bhy > size.h_view(m).x2max) ? (bhy - size.h_view(m).x2max) :
           static_cast<Real>(0.0));
      const Real oz =
          (bhz < size.h_view(m).x3min) ? (size.h_view(m).x3min - bhz) :
          ((bhz > size.h_view(m).x3max) ? (bhz - size.h_view(m).x3max) :
           static_cast<Real>(0.0));
      cache.candidate[m] = (ox*ox + oy*oy + oz*oz <= guard*guard) ? 1 : 0;
    }
    cache.valid = true;
    cache.nmb = nmb;
    cache.topology_version = pmbp->pmesh->topology_version;
    cache.bhx = bhx;
    cache.bhy = bhy;
    cache.bhz = bhz;
    cache.radius = bh_excise_radius;
  }

  // Hydro's scratch list of active blocks around the sink: it is refilled before each
  // use and every consumer runs in stream order behind this upload.
  auto &block_list = pmbp->phydro->sink_block_indices;
  auto &lat_indices = pmbp->lat_active_indices;
  if (lat_enabled) lat_indices.template sync<HostMemSpace>();
  int nshell_blocks = 0;
  for (int a = 0; a < nscan; ++a) {
    const int m = lat_enabled ? lat_indices.h_view(a) : a;
    if (cache.candidate[m]) block_list.h_view(nshell_blocks++) = m;
  }
  if (nshell_blocks == 0) return;

  // Stream-ordered upload: DualView::sync would fence the whole device queue here.
  Kokkos::deep_copy(DevExeSpace(), block_list.d_view, block_list.h_view);
  block_list.clear_sync_state();
  auto shell_blocks = block_list.d_view;

  par_for("tde_bh_floor_shell_fixup", DevExeSpace(), 0, nshell_blocks-1, ks, ke, js, je,
  is, ie, KOKKOS_LAMBDA(int a, int k, int j, int i) {
    const int m = shell_blocks(a);
    const Real rho = u0(m, IDN, k, j, i);
    if (rho >= exact_floor_rho_max) return;

    const Real &x1min = size.d_view(m).x1min;
    const Real &x1max = size.d_view(m).x1max;
    const Real &x2min = size.d_view(m).x2min;
    const Real &x2max = size.d_view(m).x2max;
    const Real &x3min = size.d_view(m).x3min;
    const Real &x3max = size.d_view(m).x3max;

    const Real x = CellCenterX(i - is, indcs.nx1, x1min, x1max);
    const Real y = CellCenterX(j - js, indcs.nx2, x2min, x2max);
    const Real z = CellCenterX(k - ks, indcs.nx3, x3min, x3max);
    const Real dx = (x1max - x1min) / static_cast<Real>(indcs.nx1);
    const Real dy = (x2max - x2min) / static_cast<Real>(indcs.nx2);
    const Real dz = (x3max - x3min) / static_cast<Real>(indcs.nx3);
    const Real shell = Kokkos::sqrt(dx*dx + dy*dy + dz*dz);
    const Real dx_bh = x - bhx;
    const Real dy_bh = y - bhy;
    const Real dz_bh = z - bhz;
    const Real r = Kokkos::sqrt(dx_bh*dx_bh + dy_bh*dy_bh + dz_bh*dz_bh);
    if (r > bh_excise_radius + shell) return;

    u0(m, IM1, k, j, i) = 0.0;
    u0(m, IM2, k, j, i) = 0.0;
    u0(m, IM3, k, j, i) = 0.0;
    if (eos_has_energy) {
      u0(m, IEN, k, j, i) = exact_floor_eint;
    }
    if (dual_enabled && dual_energy_idx >= 0) {
      u0(m, dual_energy_idx, k, j, i) = exact_floor_eint;
    }
  });
}

}  // namespace

//----------------------------------------------------------------------------------------
//! \fn void ProblemGenerator::UserProblem()
//! \brief Lane-Emden star (M*=1) with external Newtonian BH gravity source term.

void ProblemGenerator::UserProblem(ParameterInput *pin, const bool restart) {
  // Follow disk.cpp behavior: always enable and enroll user source terms here.
  user_srcs = true;
  user_srcs_lat_safe = false;
  user_srcs_lat_union_safe = false;
  user_hydro_state_fixup_lat_union_safe = false;
  user_srcs_func = TDEExternalGravitySource;
  user_hydro_state_fixup_func = TDEExternalHydroStateFixup;
  user_hydro_lat_factor_cap_func = CapHydroLATFactorsNearBH;

  Real four_pi_G = pin->GetOrAddReal("gravity", "four_pi_G", 1.0);
  if (four_pi_G <= 0.0) {
    std::cout << "### FATAL ERROR in ProblemGenerator::UserProblem" << std::endl
              << "gravity/four_pi_G must be > 0 (set it in <gravity>, e.g. 1.0)."
              << std::endl;
    std::exit(EXIT_FAILURE);
  }
  newton_g_global = four_pi_G / (4.0*M_PI);

  // Star parameters
  constexpr Real star_mass = 1.0;
  Real r_star = pin->GetOrAddReal("problem", "star_radius", 0.5);
  Real rho_floor = pin->GetOrAddReal("problem", "rho_floor", 1.0e-8);
  Real poly_n = pin->GetOrAddReal("problem", "poly_n", 1.5);
  std::string stellar_structure_mode_name =
      pin->GetOrAddString("problem", "stellar_structure_mode", "legacy_polytrope");
  StellarStructureMode stellar_structure_mode = StellarStructureMode::legacy_polytrope;
  Real amp = pin->GetOrAddReal("problem", "amp", 0.0);
  Real x_center = pin->GetOrAddReal("problem", "x_center", -1.0);
  Real y_center = pin->GetOrAddReal("problem", "y_center", 0.0);
  Real z_center = pin->GetOrAddReal("problem", "z_center", 0.0);
  Real vx_star = pin->GetOrAddReal("problem", "vx_star", 0.0);
  Real vy_star = pin->GetOrAddReal("problem", "vy_star", 0.0);
  Real vz_star = pin->GetOrAddReal("problem", "vz_star", 0.0);
  bool relax_damp = pin->GetOrAddBoolean("problem", "relax_damp", false);
  Real relax_tau = pin->GetOrAddReal("problem", "relax_tau", 0.0);
  Real relax_t_end = pin->GetOrAddReal("problem", "relax_t_end", 0.0);
  Real relax_radius = pin->GetOrAddReal("problem", "relax_radius", r_star);
  // Remap is configured through the module-level <remap> block (src/remap/remap.hpp).
  // The TDE settle-pass cadence keys also live there.  Only touch the block when it
  // exists: GetOrAdd would otherwise create it and arm auto-remap unintentionally.
  bool remap_enable = remap::IsAutoRemapEnabled(pin);
  std::string remap_restart_source;
  if (pin->DoesBlockExist("remap")) {
    remap_restart_source = pin->GetOrAddString("remap", "source", "");
    // The settle steps are run by the remap module (remap.hpp).  tde_external only seeds
    // its own default of 20 steps, which the module then reads from the <remap> block.
    (void) pin->GetOrAddInteger("remap", "settle_steps", 20);
  }
  use_translating_frame_global =
      pin->GetOrAddBoolean("problem", "use_translating_frame", false);
  hydro_lat_enabled_global = pin->IsLATEnabled();
  user_srcs_lat_safe = true;
  user_srcs_lat_union_safe = true;
  user_hydro_state_fixup_lat_union_safe = true;
  {
    // Read without GetOrAdd so a deck that does not use these keys dumps an unchanged
    // parameter set.
    bh_live_global = pin->DoesParameterExist("problem", "bh_live") &&
        pin->GetBoolean("problem", "bh_live");
    bh_reciprocal_force_global =
        pin->DoesParameterExist("problem", "bh_reciprocal_force") &&
        pin->GetBoolean("problem", "bh_reciprocal_force");
    const std::string pair_force =
        pin->DoesParameterExist("problem", "bh_pair_force") ?
        pin->GetString("problem", "bh_pair_force") : std::string("finite_difference");
    if (pair_force != "finite_difference" && pair_force != "analytic") {
      std::cout << "### FATAL ERROR in ProblemGenerator::UserProblem" << std::endl
                << "problem/bh_pair_force must be finite_difference or analytic."
                << std::endl;
      std::exit(EXIT_FAILURE);
    }
    bh_analytic_pair_global = pair_force == "analytic";
    MeshBlockPack *pack = pmy_mesh_->pmb_pack;
    const bool external_source = pack->phydro != nullptr &&
        pack->phydro->psrc != nullptr && pack->phydro->psrc->external_bh_gravity;
    if (bh_analytic_pair_global && !bh_reciprocal_force_global) {
      std::cout << "### FATAL ERROR in ProblemGenerator::UserProblem" << std::endl
                << "problem/bh_pair_force = analytic requires bh_reciprocal_force = true."
                << std::endl;
      std::exit(EXIT_FAILURE);
    }
    if (bh_reciprocal_force_global &&
        (!bh_live_global || !hydro_lat_enabled_global || pack->pgrav == nullptr ||
         !external_source)) {
      std::cout << "### FATAL ERROR in ProblemGenerator::UserProblem" << std::endl
                << "problem/bh_reciprocal_force requires a live BH "
                << "(bh_live = true), time/lat, self gravity, and "
                << "external_bh_gravity_source = true." << std::endl;
      std::exit(EXIT_FAILURE);
    }
    if (bh_live_global && !external_source) {
      std::cout << "### FATAL ERROR in ProblemGenerator::UserProblem" << std::endl
                << "a live BH needs external_bh_gravity_source = true so the gas force "
                << "and the BH force use the same potential stencil." << std::endl;
      std::exit(EXIT_FAILURE);
    }
    if (hydro_lat_enabled_global && !external_source &&
        pack->phydro != nullptr && pack->phydro->peos->eos_data.use_e) {
      // The window-end refund of the pull's mass-flux work on refluxed faces
      // (Hydro::ApplyLATFluxCorrection) exists for the core external-BH source only.
      std::cout << "### FATAL ERROR in ProblemGenerator::UserProblem" << std::endl
                << "tde_external with time/lat and an energy equation needs "
                << "problem/external_bh_gravity_source = true: the manual BH pull has no "
                << "window-end refund of its work on refluxed coarse/fine faces."
                << std::endl;
      std::exit(EXIT_FAILURE);
    }
    analytic_external_bh_momentum = bh_analytic_pair_global;
    energy_ledger_global = pack->pgrav != nullptr && pack->pgrav->lat_time_centered_work;
    problem_runtime::BookFrameRefundWork(
        energy_ledger_global && hydro_lat_enabled_global && use_translating_frame_global);
    if (hydro_lat_enabled_global && (use_translating_frame_global || bh_live_global)) {
      user_hydro_lat_window_func = PrepareTranslatingFrameLATWindow;
    }
    if (hydro_lat_enabled_global &&
        (bh_reciprocal_force_global || energy_ledger_global)) {
      user_hydro_lat_window_end_func = CompleteTDELATWindow;
    }
    if (bh_reciprocal_force_global) {
      user_hydro_gravity_ledger_func = AccumulateTDEBHGasImpulse;
    }
    if (energy_ledger_global) {
      user_hist = true;
      user_hist_func = TDEExternalHistory;
    }
    if (energy_ledger_global || bh_live_global) {
      user_metadata_func = StoreTDEExternalMetadata;
    }
  }
  bool has_loaded_restart_state = restart || pmy_mesh_->ncycle > 0
      || pmy_mesh_->time != 0.0;

  if (poly_n <= 0.0 || poly_n >= 5.0) {
    std::cout << "### FATAL ERROR in ProblemGenerator::UserProblem" << std::endl
              << "problem/poly_n must satisfy 0 < n < 5." << std::endl;
    std::exit(EXIT_FAILURE);
  }
  if (r_star <= 0.0) {
    std::cout << "### FATAL ERROR in ProblemGenerator::UserProblem" << std::endl
              << "problem/star_radius must be > 0." << std::endl;
    std::exit(EXIT_FAILURE);
  }
  if (rho_floor < 0.0) {
    std::cout << "### FATAL ERROR in ProblemGenerator::UserProblem" << std::endl
              << "problem/rho_floor must be >= 0." << std::endl;
    std::exit(EXIT_FAILURE);
  }
  if (!ParseStellarStructureMode(stellar_structure_mode_name, stellar_structure_mode)) {
    std::cout << "### FATAL ERROR in ProblemGenerator::UserProblem" << std::endl
              << "problem/stellar_structure_mode must be one of "
              << "'legacy_polytrope' or 'eos_balanced'." << std::endl;
    std::exit(EXIT_FAILURE);
  }
  if (relax_damp) {
    if (relax_tau <= 0.0) {
      std::cout << "### FATAL ERROR in ProblemGenerator::UserProblem" << std::endl
                << "problem/relax_tau must be > 0 when problem/relax_damp = true."
                << std::endl;
      std::exit(EXIT_FAILURE);
    }
    if (relax_t_end <= 0.0) {
      std::cout << "### FATAL ERROR in ProblemGenerator::UserProblem" << std::endl
                << "problem/relax_t_end must be > 0 when problem/relax_damp = true."
                << std::endl;
      std::exit(EXIT_FAILURE);
    }
    if (relax_radius <= 0.0) {
      std::cout << "### FATAL ERROR in ProblemGenerator::UserProblem" << std::endl
                << "problem/relax_radius must be > 0 when problem/relax_damp = true."
                << std::endl;
      std::exit(EXIT_FAILURE);
    }
  }
  if (remap_enable) {
    // The <remap> + time/lat refusal moved into the remap module itself
    // (remap.cpp), so it now fires for every pgen instead of this one.
    if (remap_restart_source.empty()) {
      std::cout << "### FATAL ERROR in ProblemGenerator::UserProblem" << std::endl
                << "remap/source must be set when the <remap> block is enabled."
                << std::endl;
      std::exit(EXIT_FAILURE);
    }
  }
  if (!use_translating_frame_global && !has_loaded_restart_state && !remap_enable) {
    std::cout << "### FATAL ERROR in ProblemGenerator::UserProblem" << std::endl
              << "problem/use_translating_frame = false is only supported for "
              << "restart/remap-based TDE conversion." << std::endl;
    std::exit(EXIT_FAILURE);
  }
  relax_enable_global = relax_damp;
  relax_tau_global = relax_tau;
  relax_t_end_global = relax_t_end;
  relax_radius_global = relax_radius;

  // External BH setup: Phantom-style orbit controls, but in star-centered coordinates.
  bh_soft_global = pin->GetOrAddReal("problem", "bh_softening", 1.0e-2);
  if (bh_soft_global < 0.0) {
    std::cout << "### FATAL ERROR in ProblemGenerator::TDEExternal" << std::endl
              << "problem/bh_softening must be >= 0." << std::endl;
    std::exit(EXIT_FAILURE);
  }
  bh_grav_rho_min_global = pin->GetOrAddReal("problem", "bh_grav_rho_min", 0.0);
  if (bh_grav_rho_min_global < 0.0) {
    std::cout << "### FATAL ERROR in ProblemGenerator::TDEExternal" << std::endl
              << "problem/bh_grav_rho_min must be >= 0." << std::endl;
    std::exit(EXIT_FAILURE);
  }
  frame_rho_min_global = pin->GetOrAddReal("problem", "frame_rho_min", 0.0);
  if (frame_rho_min_global < 0.0) {
    std::cout << "### FATAL ERROR in ProblemGenerator::TDEExternal" << std::endl
              << "problem/frame_rho_min must be >= 0." << std::endl;
    std::exit(EXIT_FAILURE);
  }
  bh_excise_radius_global = pin->GetOrAddReal("problem", "bh_excise_radius", 0.5);
  bh_inner_boundary_global = pin->GetOrAddBoolean("problem", "bh_inner_boundary", true);
  if (bh_inner_boundary_global && bh_excise_radius_global <= 0.0) {
    std::cout << "### FATAL ERROR in ProblemGenerator::TDEExternal" << std::endl
              << "problem/bh_excise_radius must be > 0." << std::endl;
    std::exit(EXIT_FAILURE);
  }

  bool provide_params = pin->GetOrAddBoolean("problem", "provide_params", false);
  Real mass_ratio = pin->GetOrAddReal("problem", "mass_ratio", 1000.0);
  Real beta = pin->GetOrAddReal("problem", "beta", 1.0);
  Real ecc_bh = pin->GetOrAddReal("problem", "ecc_bh", 1.0);
  Real theta_bh_deg = pin->GetOrAddReal("problem", "theta_bh", 0.0);
  Real sep_initial = pin->GetOrAddReal("problem", "sep_initial", 10.0);

  Real x1 = pin->GetOrAddReal("problem", "x1", 0.0);
  Real y1 = pin->GetOrAddReal("problem", "y1", 0.0);
  Real z1 = pin->GetOrAddReal("problem", "z1", 0.0);
  Real vx1 = pin->GetOrAddReal("problem", "vx1", 0.0);
  Real vy1 = pin->GetOrAddReal("problem", "vy1", 0.0);
  Real vz1 = pin->GetOrAddReal("problem", "vz1", 0.0);

  if (mass_ratio <= 0.0) {
    std::cout << "### FATAL ERROR in ProblemGenerator::UserProblem" << std::endl
              << "problem/mass_ratio must be > 0." << std::endl;
    std::exit(EXIT_FAILURE);
  }
  if (beta <= 0.0) {
    std::cout << "### FATAL ERROR in ProblemGenerator::UserProblem" << std::endl
              << "problem/beta must be > 0." << std::endl;
    std::exit(EXIT_FAILURE);
  }
  if (ecc_bh <= 0.0 || ecc_bh > 1.0) {
    std::cout << "### FATAL ERROR in ProblemGenerator::UserProblem" << std::endl
              << "problem/ecc_bh must satisfy 0 < ecc_bh <= 1." << std::endl;
    std::exit(EXIT_FAILURE);
  }
  if (sep_initial <= 0.0) {
    std::cout << "### FATAL ERROR in ProblemGenerator::UserProblem" << std::endl
              << "problem/sep_initial must be > 0." << std::endl;
    std::exit(EXIT_FAILURE);
  }
  bh_mass_global = mass_ratio * star_mass;
  // External BH potential: newtonian, the softened point mass, or wegg, the Wegg (2012)
  // pseudo-Newtonian potential (bh_force_pair.hpp), which carries the Schwarzschild
  // apsidal precession and needs r_g = G M / c^2 in code units, hence <units>.
  {
    std::string bh_potential =
        pin->GetOrAddString("problem", "bh_potential", "newtonian");
    std::transform(bh_potential.begin(), bh_potential.end(), bh_potential.begin(),
                   [](unsigned char c) { return std::tolower(c); });
    if (bh_potential == "newtonian") {
      bh_rg_global = 0.0;
    } else if (bh_potential == "wegg") {
      units::Units *punit = pmy_mesh_->pmb_pack->punit;
      if (punit == nullptr) {
        std::cout << "### FATAL ERROR in ProblemGenerator::TDEExternal" << std::endl
                  << "problem/bh_potential = wegg needs a <units> block (the speed of "
                  << "light in code units sets r_g = G M / c^2)." << std::endl;
        std::exit(EXIT_FAILURE);
      }
      const Real c_code = punit->speed_of_light();
      bh_rg_global = newton_g_global*bh_mass_global/(c_code*c_code);
      if (!bh_inner_boundary_global || bh_excise_radius_global < 2.0*bh_rg_global) {
        std::cout << "### FATAL ERROR in ProblemGenerator::TDEExternal" << std::endl
                  << "problem/bh_potential = wegg needs bh_inner_boundary = true and "
                  << "bh_excise_radius >= 2 r_g = " << 2.0*bh_rg_global << " (the "
                  << "potential is held inside 1.6 r_g and diverges at 0.8 r_g)."
                  << std::endl;
        std::exit(EXIT_FAILURE);
      }
    } else {
      std::cout << "### FATAL ERROR in ProblemGenerator::TDEExternal" << std::endl
                << "Unknown problem/bh_potential = '" << bh_potential
                << "'; expected newtonian or wegg." << std::endl;
      std::exit(EXIT_FAILURE);
    }
  }
  orbit_center_x_global = x_center;
  orbit_center_y_global = y_center;
  orbit_center_z_global = z_center;
  frame_vx_global = 0.0;
  frame_vy_global = 0.0;
  frame_vz_global = 0.0;

  Real r_tidal = r_star * std::cbrt(mass_ratio);
  Real r_peri = r_tidal / beta;
  Real mu_orbit = newton_g_global * (star_mass + bh_mass_global);
  // the two-body orbit's potential at r (the applied one, Newtonian or Wegg)
  auto orbit_phi = [&](const Real r) {
    return bh_force_pair::PotentialOfRadius(std::sqrt(r*r + bh_soft_global*bh_soft_global),
                                            star_mass + bh_mass_global, newton_g_global,
                                            bh_rg_global);
  };

  Real star_x = 0.0, star_y = 0.0, star_z = 0.0;
  Real star_vx = 0.0, star_vy = 0.0, star_vz = 0.0;

  if (provide_params) {
    // Same parameter names as Phantom setup: user gives the star state relative to BH.
    // Star-centered frame here uses the opposite sign for the BH state.
    star_x = x1;
    star_y = y1;
    star_z = z1;
    star_vx = vx1;
    star_vy = vy1;
    star_vz = vz1;
  } else {
    Real theta_bh = DegToRad(theta_bh_deg);
    if (ecc_bh < (1.0 - 1.0e-12)) {
      Real semia = r_peri/(1.0 - ecc_bh);
      Real r_apo = semia*(1.0 + ecc_bh);
      Real v_apo = std::sqrt(mu_orbit*(2.0/r_apo - 1.0/semia));
      if (bh_rg_global > 0.0) {
        // turning points at r_apo and r_peri of the applied potential:
        // h^2 = 2 (Phi(r_apo) - Phi(r_peri)) / (1/r_peri^2 - 1/r_apo^2)
        const Real h2 = 2.0*(orbit_phi(r_apo) - orbit_phi(r_peri))/
                        (1.0/(r_peri*r_peri) - 1.0/(r_apo*r_apo));
        v_apo = std::sqrt(h2)/r_apo;
      }

      // Start at apoapsis with pericenter along +y, then apply inclination.
      star_x = 0.0;
      star_y = -r_apo;
      star_z = 0.0;
      star_vx = v_apo;
      star_vy = 0.0;
      star_vz = 0.0;
      RotateAboutY(star_x, star_y, star_z, theta_bh);
      RotateAboutY(star_vx, star_vy, star_vz, theta_bh);
    } else {
      // Parabolic branch follows the same construction used in Phantom's TDE setup.
      Real r0 = sep_initial * r_tidal;
      Real y0 = -2.0*r_peri + r0;
      Real x2 = r0*r0 - y0*y0;
      if (x2 < -1.0e-12*std::fmax(1.0, r0*r0)) {
        std::cout << "### FATAL ERROR in ProblemGenerator::UserProblem" << std::endl
                  << "problem/sep_initial is too small for the chosen beta." << std::endl;
        std::exit(EXIT_FAILURE);
      }
      Real x0 = std::sqrt(std::fmax(x2, 0.0));
      Real denom = std::sqrt(4.0*r_peri*r_peri + x0*x0);
      if (denom <= 0.0 || r0 <= 0.0) {
        std::cout << "### FATAL ERROR in ProblemGenerator::UserProblem" << std::endl
                  << "Invalid parabolic initial condition geometry." << std::endl;
        std::exit(EXIT_FAILURE);
      }
      Real vel = std::sqrt(2.0*mu_orbit/r0);

      star_x = -x0;
      star_y = y0;
      star_z = 0.0;
      star_vx = vel*(2.0*r_peri/denom);
      star_vy = vel*(-x0/denom);
      star_vz = 0.0;
      if (bh_rg_global > 0.0) {
        // Parabolic in the applied potential: zero energy, and the angular momentum of
        // a turning point at r_peri, h = r_peri sqrt(-2 Phi(r_peri)).  The position and
        // the sense of motion are the Newtonian parabola's; the pericentre then lies a
        // few degrees from -y, the precession the potential accumulates on the way in.
        const Real v2 = -2.0*orbit_phi(r0);
        const Real vt = r_peri*std::sqrt(-2.0*orbit_phi(r_peri))/r0;
        const Real vr = -std::sqrt(std::fmax(v2 - vt*vt, 0.0));
        const Real rhx = star_x/r0, rhy = star_y/r0;
        Real tx = star_vx - (star_vx*rhx + star_vy*rhy)*rhx;
        Real ty = star_vy - (star_vx*rhx + star_vy*rhy)*rhy;
        const Real tn = std::sqrt(tx*tx + ty*ty);
        tx /= tn;
        ty /= tn;
        star_vx = vr*rhx + vt*tx;
        star_vy = vr*rhy + vt*ty;
      }
      RotateAboutY(star_x, star_y, star_z, theta_bh);
      RotateAboutY(star_vx, star_vy, star_vz, theta_bh);
    }
  }

  Real bh_init_x = -star_x;
  Real bh_init_y = -star_y;
  Real bh_init_z = -star_z;
  Real bh_init_vx = -star_vx;
  Real bh_init_vy = -star_vy;
  Real bh_init_vz = -star_vz;

  if (!std::isfinite(bh_init_x) || !std::isfinite(bh_init_y) ||
      !std::isfinite(bh_init_z) || !std::isfinite(bh_init_vx) ||
      !std::isfinite(bh_init_vy) || !std::isfinite(bh_init_vz)) {
    std::cout << "### FATAL ERROR in ProblemGenerator::UserProblem" << std::endl
              << "Non-finite BH orbit initial state." << std::endl;
    std::exit(EXIT_FAILURE);
  }

  // The BH is fixed in the inertial frame. The simulation coordinates use a
  // translating, non-rotating frame, so the BH moves only through frame motion.
  bh_inertial_x_global = orbit_center_x_global + bh_init_x;
  bh_inertial_y_global = orbit_center_y_global + bh_init_y;
  bh_inertial_z_global = orbit_center_z_global + bh_init_z;
  bh_x_global = bh_init_x;
  bh_y_global = bh_init_y;
  bh_z_global = bh_init_z;
  frame_vx_global = star_vx;
  frame_vy_global = star_vy;
  frame_vz_global = star_vz;
  bh_vx_global = -frame_vx_global;
  bh_vy_global = -frame_vy_global;
  bh_vz_global = -frame_vz_global;
  InvalidateFrameBHStepState();

  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;
  if (pmbp->phydro == nullptr) {
    std::cout << "### FATAL ERROR in ProblemGenerator::UserProblem" << std::endl
              << "This problem requires a <hydro> block." << std::endl;
    std::exit(EXIT_FAILURE);
  }
  if (pmbp->pmhd != nullptr || pmbp->prad != nullptr || pmbp->pturb != nullptr ||
      pmbp->pz4c != nullptr || pmbp->padm != nullptr) {
    std::cout << "### FATAL ERROR in ProblemGenerator::UserProblem" << std::endl
              << "tde_external remap currently supports hydro-only runs."
              << std::endl;
    std::exit(EXIT_FAILURE);
  }
  auto eos = pmbp->phydro->peos->eos_data;
  hydro_dfloor_global = eos.dfloor;
  ConfigureTDEAMR(pin, pmy_mesh_, restart, r_star, mass_ratio, r_peri,
                  DegToRad(theta_bh_deg), star_x, star_y, star_z, star_vx, star_vy, star_vz);
  if (tde_amr_enable_global) {
    user_ref_func = RefineTDEAMR;
  }
  if (!(eos.use_e)) {
    std::cout << "### FATAL ERROR in ProblemGenerator::UserProblem" << std::endl
              << "This problem requires a hydro EOS with an energy variable."
              << std::endl;
    std::exit(EXIT_FAILURE);
  }
  bool self_gravity_enabled =
      (pmbp->phydro->psrc != nullptr && pmbp->phydro->psrc->self_gravity);
  Real gamma_eos = eos.gamma;
  if (eos.is_gamma_law && gamma_eos <= 1.0) {
    std::cout << "### FATAL ERROR in ProblemGenerator::UserProblem" << std::endl
              << "hydro/gamma must be > 1 for ideal EOS." << std::endl;
    std::exit(EXIT_FAILURE);
  }
  // The excision interior: problem/bh_excise_density (the density floor by default) at
  // rest and at the thermal floor of that density, so it is not a vacuum the infall falls
  // into when the floor is far below the debris.  It stays cold: the floors' pressure
  // ratio pfloor/dfloor taken to 1e-10 is radiation-dominated in the LTE table, with a
  // sound speed near c/sqrt(3) that would set the time step of the whole run.
  bh_excise_density_global = std::max(
      pin->GetOrAddReal("problem", "bh_excise_density", eos.dfloor), eos.dfloor);
  // Host-side startup uses the host Saha table mirror for consistent EOS floors.
  bh_excise_eint_global = eos.HostClampHydroInternalEnergyDensity(
      bh_excise_density_global,
      eos.HostHydroInternalEnergyDensityFloor(bh_excise_density_global));
  frame_state_time_global = 0.0;
  InvalidateFrameBHStepState();
  excision_shell_blocks = ExcisionShellBlocks();  // the blocks of an earlier pass are gone
  EvaluateBHAccelerationAtPoint(orbit_center_x_global, orbit_center_y_global,
                                orbit_center_z_global,
                                bh_inertial_x_global, bh_inertial_y_global,
                                bh_inertial_z_global,
                                frame_ax_global, frame_ay_global, frame_az_global);
  SyncProblemRuntimeState();

  // <remap>/auto_expand hook, on every path (fresh, remap, restart)
  {
    const Real expand_tlim = pin->GetReal("time", "tlim");
    // Read without GetOrAdd: adding a key would create the <remap> block, and
    // remap::IsAutoRemapEnabled treats an existing block as a remap request.
    const Real rho_need = pin->DoesParameterExist("remap", "expand_rho_min") ?
        pin->GetReal("remap", "expand_rho_min") :
        100.0*pmy_mesh_->pmb_pack->phydro->peos->eos_data.dfloor;
    const bool bound_only = pin->DoesParameterExist("remap", "expand_bound_only") ?
        pin->GetBoolean("remap", "expand_bound_only") : false;
    user_expand_need_func = [expand_tlim, rho_need, bound_only](
        Mesh *pm, const Real *buf, const bool full, remap::ExpandNeed &need) {
      TDEExternalExpandNeed(pm, buf, full, expand_tlim, rho_need, bound_only, need);
    };
  }

  if (remap_enable && !has_loaded_restart_state) {
    // The remap itself runs from remap::MaybeAutoRemap right after this function
    // returns (fresh-start ProblemGenerator constructor).  Enroll the TDE hooks and
    // skip the fresh-star initialization below.
    user_remap_skip_func = [](Real x, Real y, Real z) {
      const Real excise_r2 = bh_inner_boundary_global ?
          bh_excise_radius_global * bh_excise_radius_global : -1.0;
      return tde_external::InsideExcisionZone(x, y, z, bh_x_global, bh_y_global,
                                              bh_z_global, excise_r2);
    };
    user_remap_loaded_func = [](ParameterInput *src_pin) {
      // Restore the BH/frame record BEFORE the state is applied: the excision skip
      // hook above reads the restored BH position during the apply.
      if (!RestoreFrameBHStateFromMetadata(src_pin)) {
        std::cout << "### FATAL ERROR in ProblemGenerator::UserProblem" << std::endl
                  << "remap source restart is missing the BH/frame metadata."
                  << std::endl;
        std::exit(EXIT_FAILURE);
      }
      // A live BH also keeps its inertial velocity and gas acceleration across the
      // remap.  Without them the leapfrog restarts from rest after every remap and
      // SwitchToFrozenBHInertialFrame places the BH at rest in the new frame.
      RestoreLiveBHMetadata(src_pin);
    };
    Mesh *pm_cap = pmy_mesh_;
    MeshBlockPack *pmbp_cap = pmbp;
    ParameterInput *pin_cap = pin;
    user_remap_post_func = [pm_cap, pmbp_cap, pin_cap](
        const remap::RemapSummary &) {
      if (use_translating_frame_global) {
        frame_state_time_global = pm_cap->time;
        // The source state moved the frame center and the BH, so the stored frame
        // acceleration is re-evaluated at the new state, as on a restart.
        EvaluateBHAccelerationAtPoint(orbit_center_x_global, orbit_center_y_global,
                                      orbit_center_z_global,
                                      bh_inertial_x_global, bh_inertial_y_global,
                                      bh_inertial_z_global,
                                      frame_ax_global, frame_ay_global, frame_az_global);
        InvalidateFrameBHStepState();
        SyncProblemRuntimeState();
      } else {
        SwitchToFrozenBHInertialFrame(pm_cap, pmbp_cap);
      }
      tde_external::StoreRuntimeMetadata(pin_cap);

      if (global_variable::my_rank == 0) {
        std::cout << "use translating frame  = "
                  << (use_translating_frame_global ? "true" : "false") << std::endl
                  << std::endl;
      }
    };
    return;
  }

  if (has_loaded_restart_state) {
    if (!RestoreFrameBHStateFromMetadata(pin)) {
      std::cout << "### FATAL ERROR in ProblemGenerator::UserProblem" << std::endl
                << "tde_external restart requires stored BH/frame metadata in the "
                << "restart file." << std::endl;
      std::exit(EXIT_FAILURE);
    }
    RestoreLiveBHMetadata(pin);
    if (use_translating_frame_global) {
      frame_state_time_global = pmy_mesh_->time;
      EvaluateBHAccelerationAtPoint(orbit_center_x_global, orbit_center_y_global,
                                    orbit_center_z_global,
                                    bh_inertial_x_global, bh_inertial_y_global,
                                    bh_inertial_z_global,
                                    frame_ax_global, frame_ay_global, frame_az_global);
      SyncProblemRuntimeState();
    } else {
      // Boosting into the frozen BH frame rewrites the interior cells, so the
      // checkpointed ghost zones no longer describe the state and must be rebuilt.
      if (frame_vx_global != 0.0 || frame_vy_global != 0.0 || frame_vz_global != 0.0) {
        restart_state_verbatim = false;
      }
      SwitchToFrozenBHInertialFrame(pmy_mesh_, pmbp);
    }
    if (global_variable::my_rank == 0) {
      std::cout << "Restored BH/frame state from restart metadata: "
                << "BH=(" << bh_x_global << ", " << bh_y_global << ", " << bh_z_global
                << "), frame=(" << orbit_center_x_global << ", "
                << orbit_center_y_global << ", " << orbit_center_z_global << ")"
                << std::endl;
    }
    return;
  }

  LaneEmdenProfile lane_profile;
  if (!SolveLaneEmden(poly_n, lane_profile)) {
    std::cout << "### FATAL ERROR in ProblemGenerator::UserProblem" << std::endl
              << "Failed to solve Lane-Emden equation for n=" << poly_n << std::endl;
    std::exit(EXIT_FAILURE);
  }

  Real xi1 = lane_profile.xi1;
  Real qn = -SQR(xi1) * lane_profile.dtheta_xi1;
  if (qn <= 0.0) {
    std::cout << "### FATAL ERROR in ProblemGenerator::UserProblem" << std::endl
              << "Lane-Emden mass factor is non-positive." << std::endl;
    std::exit(EXIT_FAILURE);
  }

  Real a_scale = r_star / xi1;
  Real rho_central = star_mass / (4.0*M_PI*std::pow(a_scale, 3.0)*qn);
  Real poly_k = (four_pi_G / (poly_n + 1.0)) * SQR(a_scale)
              * std::pow(rho_central, 1.0 - 1.0/poly_n);
  Real gamma_poly = 1.0 + 1.0/poly_n;

  int lane_npts = static_cast<int>(lane_profile.xi.size());
  DvceArray1D<Real> lane_xi_d("lane_xi", lane_npts);
  DvceArray1D<Real> lane_theta_d("lane_theta", lane_npts);
  DvceArray1D<Real> lane_dtheta_d("lane_dtheta", lane_npts);
  HostArray1D<Real> lane_xi_h = Kokkos::create_mirror_view(lane_xi_d);
  HostArray1D<Real> lane_theta_h = Kokkos::create_mirror_view(lane_theta_d);
  HostArray1D<Real> lane_dtheta_h = Kokkos::create_mirror_view(lane_dtheta_d);
  for (int n = 0; n < lane_npts; ++n) {
    lane_xi_h(n) = lane_profile.xi[n];
    lane_theta_h(n) = lane_profile.theta[n];
    lane_dtheta_h(n) = lane_profile.dtheta[n];
  }
  Kokkos::deep_copy(lane_xi_d, lane_xi_h);
  Kokkos::deep_copy(lane_theta_d, lane_theta_h);
  Kokkos::deep_copy(lane_dtheta_d, lane_dtheta_h);

  auto &indcs = pmy_mesh_->mb_indcs;
  int is = indcs.is, ie = indcs.ie;
  int js = indcs.js, je = indcs.je;
  int ks = indcs.ks, ke = indcs.ke;
  auto &size = pmbp->pmb->mb_size;
  auto &u0 = pmbp->phydro->u0;
  const bool eos_has_energy = eos.use_e;
  bool eos_is_gamma_law = eos.is_gamma_law;
  int nhyd = pmbp->phydro->nhydro;
  int nscalars = pmbp->phydro->nscalars;
  Real xi_scale = xi1 / r_star;
  Real excise_bhx = bh_x_global;
  Real excise_bhy = bh_y_global;
  Real excise_bhz = bh_z_global;
  const Real excise_r2 = bh_inner_boundary_global ?
      bh_excise_radius_global * bh_excise_radius_global : -1.0;
  Real excise_density = bh_excise_density_global;
  Real excise_eint = bh_excise_eint_global;
  const Real floor_eint =
      eos.HostClampHydroInternalEnergyDensity(rho_floor,
                                              eos.HostHydroInternalEnergyDensityFloor(rho_floor));
  const Real p_floor = eos.HostPressureFromRhoEint(rho_floor, floor_eint);
  int nmb = pmbp->nmb_thispack;
  const bool use_eos_balanced_profile =
      (stellar_structure_mode == StellarStructureMode::eos_balanced);

  if (use_eos_balanced_profile) {
    if (!eos_has_energy) {
      std::cout << "### FATAL ERROR in ProblemGenerator::UserProblem" << std::endl
                << "problem/stellar_structure_mode = eos_balanced requires an EOS "
                << "with an energy variable." << std::endl;
      std::exit(EXIT_FAILURE);
    }
    if (amp != 0.0) {
      std::cout << "### FATAL ERROR in ProblemGenerator::UserProblem" << std::endl
                << "problem/amp must be 0 when problem/stellar_structure_mode = "
                << "eos_balanced. The legacy density perturbation is only defined for "
                << "the pressure-mapped polytropic initializer." << std::endl;
      std::exit(EXIT_FAILURE);
    }
  }

  // An atmosphere outside the star (problem/atmosphere_length, a fraction of r_star; 0
  // leaves the floor there): rho_s exp(-(r - r_s)/L) above the floor, in hydrostatic
  // equilibrium in the star's field (AttachAtmosphere), replacing the profile outward of
  // r_s, where the profile's pressure scale height is about L.  It replaces the step from
  // the star's edge to the floor, which the mesh cannot resolve, by a decay it resolves
  // when L spans a few of its finest cells.  Its pressure, int_r^inf rho G M/r'^2 dr',
  // vanishes with its density, so it meets the floor without a pressure step, and its
  // temperature is what the EOS gives that pressure (a hot corona).  An atmosphere without
  // that pressure falls onto the star, and the shock it drives accelerates down the
  // density decay and throws gas out over the whole mesh.
  const Real atm_length = pin->GetOrAddReal("problem", "atmosphere_length", 0.0)*r_star;
  Real atm_r_s = r_star, atm_rho_s = 0.0, atm_p_s = 0.0, atm_m_old = 0.0,
       atm_m_new = 0.0;
  if (atm_length < 0.0) {
    std::cout << "### FATAL ERROR in ProblemGenerator::UserProblem" << std::endl
              << "problem/atmosphere_length must be >= 0." << std::endl;
    std::exit(EXIT_FAILURE);
  }
  if (atm_length > 0.0 && !(use_eos_balanced_profile && rho_floor > 0.0)) {
    std::cout << "### FATAL ERROR in ProblemGenerator::UserProblem" << std::endl
              << "problem/atmosphere_length needs problem/stellar_structure_mode = "
              << "eos_balanced and problem/rho_floor > 0." << std::endl;
    std::exit(EXIT_FAILURE);
  }

  StellarRadialProfile eos_profile;
  int radial_profile_npts = 1;
  DvceArray1D<Real> radial_r_d("stellar_profile_r", 1);
  DvceArray1D<Real> radial_rho_d("stellar_profile_rho", 1);
  DvceArray1D<Real> radial_p_d("stellar_profile_p", 1);
  {
    auto radial_r_h = Kokkos::create_mirror_view(radial_r_d);
    auto radial_rho_h = Kokkos::create_mirror_view(radial_rho_d);
    auto radial_p_h = Kokkos::create_mirror_view(radial_p_d);
    radial_r_h(0) = 0.0;
    radial_rho_h(0) = 0.0;
    radial_p_h(0) = 0.0;
    Kokkos::deep_copy(radial_r_d, radial_r_h);
    Kokkos::deep_copy(radial_rho_d, radial_rho_h);
    Kokkos::deep_copy(radial_p_d, radial_p_h);
  }

  if (use_eos_balanced_profile) {
    const Real p_central_guess = poly_k * std::pow(rho_central, gamma_poly);
    EOS_Data profile_eos = eos;
    const NewtonianStarModel star_model{profile_eos, newton_g_global};
    if (!stellar_profile::SolveProfile(star_model, star_mass, r_star, rho_central,
                                       p_central_guess, gamma_poly, eos_profile)) {
      std::cout << "### FATAL ERROR in ProblemGenerator::UserProblem" << std::endl
                << "Failed to construct an EOS-balanced stellar profile for "
                << "problem/stellar_structure_mode = eos_balanced." << std::endl;
      if (profile_eos.UsesTabulatedLTE()) {
        std::cout << "The requested stellar setup is not consistent with the loaded "
                  << "tabulated LTE EOS over its active rho-T domain. "
                  << "Do not rely on host-side clamping here; use a table that "
                  << "covers the intended stellar regime or revert to a lower-density "
                  << "validation scaling." << std::endl;
      }
      std::exit(EXIT_FAILURE);
    }
    rho_central = eos_profile.rho_central;
    if (atm_length > 0.0 &&
        !AttachAtmosphere(eos_profile, atm_length, newton_g_global*star_mass,
                          1.0e-3*rho_floor, atm_r_s, atm_rho_s, atm_p_s, atm_m_old,
                          atm_m_new)) {
      std::cout << "### FATAL ERROR in ProblemGenerator::UserProblem" << std::endl
                << "problem/atmosphere_length: no profile node where the hydrostatic "
                << "atmosphere's pressure reaches the profile's." << std::endl;
      std::exit(EXIT_FAILURE);
    }
    radial_profile_npts = static_cast<int>(eos_profile.radius.size());
    radial_r_d = DvceArray1D<Real>("stellar_profile_r", radial_profile_npts);
    radial_rho_d = DvceArray1D<Real>("stellar_profile_rho", radial_profile_npts);
    radial_p_d = DvceArray1D<Real>("stellar_profile_p", radial_profile_npts);
    HostArray1D<Real> radial_r_h = Kokkos::create_mirror_view(radial_r_d);
    HostArray1D<Real> radial_rho_h = Kokkos::create_mirror_view(radial_rho_d);
    HostArray1D<Real> radial_p_h = Kokkos::create_mirror_view(radial_p_d);
    for (int n = 0; n < radial_profile_npts; ++n) {
      radial_r_h(n) = eos_profile.radius[n];
      radial_rho_h(n) = eos_profile.density[n];
      radial_p_h(n) = eos_profile.pressure[n];
    }
    Kokkos::deep_copy(radial_r_d, radial_r_h);
    Kokkos::deep_copy(radial_rho_d, radial_rho_h);
    Kokkos::deep_copy(radial_p_d, radial_p_h);
  }

  if (!use_eos_balanced_profile && eos_is_gamma_law && global_variable::my_rank == 0) {
    if (std::abs(gamma_eos - gamma_poly) > 1.0e-8) {
      std::cout << "### WARNING in ProblemGenerator::UserProblem" << std::endl
                << "For consistent polytropic initialization, set hydro/gamma = "
                << gamma_poly << " (1 + 1/n)." << std::endl;
    }
  }

  par_for("tde_external_init", DevExeSpace(), 0, nmb - 1, ks, ke, js, je, is, ie,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    Real &x1min = size.d_view(m).x1min;
    Real &x1max = size.d_view(m).x1max;
    Real &x2min = size.d_view(m).x2min;
    Real &x2max = size.d_view(m).x2max;
    Real &x3min = size.d_view(m).x3min;
    Real &x3max = size.d_view(m).x3max;

    Real x = CellCenterX(i - is, indcs.nx1, x1min, x1max);
    Real y = CellCenterX(j - js, indcs.nx2, x2min, x2max);
    Real z = CellCenterX(k - ks, indcs.nx3, x3min, x3max);

    if (excise_r2 > 0.0 && tde_external::InsideExcisionZone(x, y, z, excise_bhx,
                                                           excise_bhy, excise_bhz,
                                                           excise_r2)) {
      u0(m, IDN, k, j, i) = excise_density;
      u0(m, IM1, k, j, i) = 0.0;
      u0(m, IM2, k, j, i) = 0.0;
      u0(m, IM3, k, j, i) = 0.0;
      if (eos_has_energy) {
        u0(m, IEN, k, j, i) = excise_eint;
      }
      for (int n = nhyd; n < (nhyd + nscalars); ++n) {
        u0(m, n, k, j, i) = 0.0;
      }
      return;
    }

    Real r = Kokkos::sqrt(SQR(x - x_center) + SQR(y - y_center) + SQR(z - z_center));
    Real rho_star = 0.0;
    Real p_star = 0.0;
    if (use_eos_balanced_profile) {
      // the table ends on a zero node (beyond r_star, or the atmosphere's end)
      rho_star = RadialProfileValue(r, radial_r_d, radial_rho_d, radial_profile_npts);
      p_star = RadialProfileValue(r, radial_r_d, radial_p_d, radial_profile_npts);
    } else {
      Real theta = 0.0;
      if (r < r_star) {
        Real xi = r * xi_scale;
        theta = LaneEmdenTheta(xi, lane_xi_d, lane_theta_d, lane_dtheta_d, lane_npts);
      }

      rho_star = rho_central * Kokkos::pow(theta, poly_n);
      if (amp > 0.0 && r < r_star) {
        rho_star *= (1.0 + amp * (r * r) / (r_star * r_star)
                     * Kokkos::cos(2.0 * Kokkos::atan2(y, x)));
      }
      p_star = poly_k * Kokkos::pow(Kokkos::fmax(rho_star, 0.0), gamma_poly);
    }

    Real rho = rho_floor + rho_star;
    u0(m, IDN, k, j, i) = rho;
    // Hydro state is initialized in the comoving frame, so the star starts at
    // rest apart from any explicitly requested frame-relative bulk velocity.
    u0(m, IM1, k, j, i) = rho * vx_star;
    u0(m, IM2, k, j, i) = rho * vy_star;
    u0(m, IM3, k, j, i) = rho * vz_star;

    if (eos_has_energy) {
      // The star is stored as a star-over-floor decomposition. In legacy mode the
      // pressure comes from the Lane-Emden polytrope and is only mapped through the
      // active EOS. In eos_balanced mode p_star itself comes from the active-EOS
      // hydrostatic structure solve.
      Real pgas = p_floor + p_star;
      Real ekin = 0.5 * rho * (vx_star*vx_star + vy_star*vy_star + vz_star*vz_star);
      Real eint = eos.InternalEnergyDensityFromRhoP(rho, pgas);
      eint = eos.ClampHydroInternalEnergyDensity(rho, eint);
      u0(m, IEN, k, j, i) = eint + ekin;
    }
    for (int n = nhyd; n < (nhyd + nscalars); ++n) {
      u0(m, n, k, j, i) = 0.0;
    }
  });

  if (global_variable::my_rank == 0) {
    std::cout << std::endl
      << "--- TDE External Potential ---" << std::endl
      << "star_mass (fixed)      = " << star_mass << std::endl
      << "star_radius            = " << r_star << std::endl
      << "stellar_structure_mode = "
      << StellarStructureModeName(stellar_structure_mode) << std::endl
      << "rho_central (derived)  = " << rho_central << std::endl
      << "rho_floor              = " << rho_floor << std::endl;
    if (atm_length > 0.0) {
      // The base of the atmosphere, its decay length in the finest cells of the mesh,
      // the radius where it falls to the floor, and the mass outward of r_s before and
      // after it replaced the profile there.
      const Real dx_finest = pmy_mesh_->mesh_size.dx1/
          std::pow(2.0, pmy_mesh_->max_level - pmy_mesh_->root_level);
      const Real atm_r_floor = (atm_rho_s > rho_floor) ?
          atm_r_s + atm_length*std::log(atm_rho_s/rho_floor) : atm_r_s;
      std::cout
        << "atmosphere             = rho_s exp(-(r - r_s)/L), hydrostatic, "
        << "P = int_r^inf rho G M/r'^2 dr'" << std::endl
        << "                         r_s = " << atm_r_s << ", rho_s = " << atm_rho_s
        << ", P_s = " << atm_p_s << ", L = " << atm_length << " ("
        << atm_length/dx_finest << " finest cells), down to rho_floor at r = "
        << atm_r_floor << std::endl
        << "                         mass outward of r_s " << atm_m_old/star_mass
        << " -> " << atm_m_new/star_mass << " M_*" << std::endl;
    }
    std::cout
      << "poly_n                 = " << poly_n << std::endl
      << "poly_gamma             = " << gamma_poly << std::endl
      << "relax_damp             = " << (relax_enable_global ? "true" : "false")
      << std::endl
      << "relax_tau              = " << relax_tau_global << std::endl
      << "relax_t_end            = " << relax_t_end_global << std::endl
      << "relax_radius           = " << relax_radius_global << std::endl
      << "BH mass                = " << bh_mass_global << std::endl
      << "BH position            = (" << bh_x_global << ", "
      << bh_y_global << ", " << bh_z_global << ")" << std::endl
      << "BH velocity            = (" << bh_vx_global << ", "
      << bh_vy_global << ", " << bh_vz_global << ")" << std::endl
      << "Frame center           = (" << orbit_center_x_global << ", "
      << orbit_center_y_global << ", " << orbit_center_z_global << ")" << std::endl
      << "Frame velocity         = (" << frame_vx_global << ", "
      << frame_vy_global << ", " << frame_vz_global << ")" << std::endl
      << "Frame acceleration     = (" << frame_ax_global << ", "
      << frame_ay_global << ", " << frame_az_global << ")" << std::endl
      << "Use translating frame  = "
      << (use_translating_frame_global ? "true" : "false") << std::endl
      << "Self gravity enabled   = " << (self_gravity_enabled ? "true" : "false")
      << std::endl
      << "Frame rho min          = " << frame_rho_min_global << std::endl
      << "BH softening           = " << bh_soft_global << std::endl
      << "BH potential           = "
      << ((bh_rg_global > 0.0) ? "wegg (Wegg 2012 pseudo-Newtonian)" : "newtonian")
      << std::endl;
    if (bh_rg_global > 0.0) {
      std::cout << "BH r_g = G M / c^2      = " << bh_rg_global << "  (r_p/r_g = "
                << r_peri/bh_rg_global << ", sink/r_g = "
                << bh_excise_radius_global/bh_rg_global << ")" << std::endl;
    }
    std::cout
      << "BH grav rho min        = " << bh_grav_rho_min_global << std::endl
      << "BH inner boundary      = " << (bh_inner_boundary_global ? "true" : "false")
      << std::endl
      << "BH excise radius       = " << bh_excise_radius_global << std::endl
      << "TDE AMR                = "
      << (tde_amr_enable_global ? "<tde_amr> (see its banner)" : "off") << std::endl
      << "BH frame note          = "
      << (use_translating_frame_global
              ? "fixed inertial BH, continuous translating frame"
              : "fixed BH in the simulation frame after restart/remap conversion")
      << std::endl
      << "Newtonian G            = " << newton_g_global << std::endl
      << "four_pi_G              = " << four_pi_G << std::endl
      << std::endl;
    if (use_eos_balanced_profile) {
      std::cout
        << "EOS-balanced p_c       = " << eos_profile.p_central << std::endl
        << "EOS-balanced M(r_*)    = " << eos_profile.mass_at_surface << std::endl
        << "EOS-balanced P(r_*)    = " << eos_profile.pressure_at_surface << std::endl
        << "EOS profile samples    = " << radial_profile_npts << std::endl
        << std::endl;
    } else {
      std::cout
        << "poly_K (derived)       = " << poly_k << std::endl
        << "Lane-Emden xi1         = " << xi1 << std::endl
        << std::endl;
    }
    std::cout
      << "provide_params         = "
      << (provide_params ? "true" : "false") << std::endl
      << "mass_ratio             = " << mass_ratio << std::endl
      << "beta                   = " << beta << std::endl
      << "ecc_bh                 = " << ecc_bh << std::endl
      << "theta_bh (deg)         = " << theta_bh_deg << std::endl
      << "sep_initial            = " << sep_initial << std::endl
      << "r_tidal                = " << r_tidal << std::endl
      << "r_peri                 = " << r_peri << std::endl
      << "BH init rel pos        = (" << bh_init_x << ", "
      << bh_init_y << ", " << bh_init_z << ")" << std::endl
      << "BH init rel vel        = (" << bh_init_vx << ", "
      << bh_init_vy << ", " << bh_init_vz << ")" << std::endl
      << std::endl;
  }
}

namespace tde_external {

void StoreRuntimeMetadata(ParameterInput *pin) {
  SyncProblemRuntimeState();
  problem_runtime::StoreRuntimeMetadata(pin);
}

}  // namespace tde_external
