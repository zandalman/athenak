//========================================================================================
// AthenaK astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file remap.cpp
//! \brief orchestration entry point of the universal restart-remap module.

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <string>

#include "athena.hpp"
#include "globals.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "mesh/mesh_refinement.hpp"
#include "mesh/meshblock_pack.hpp"
#include "coordinates/adm.hpp"
#include "eos/eos.hpp"
#include "hydro/hydro.hpp"
#include "mhd/mhd.hpp"
#include "dyn_grmhd/dyn_grmhd.hpp"
#include "driver/driver.hpp"
#include "pgen/pgen.hpp"
#include "remap/remap.hpp"
#include "remap/remap_impl.hpp"

namespace remap {

namespace {

[[noreturn]] void FatalRemap(const std::string &msg) {
  std::cout << "### FATAL ERROR in remap module" << std::endl << msg << std::endl;
  std::exit(EXIT_FAILURE);
}

// set by a startup remap whose floors did not match (<remap>/floor_reset)
bool floor_reset_pass = false;

// The density thresholds a floor reset scales with <hydro>/dfloor (remap_impl.hpp): the
// pressure floor, tde_external's ambient, BH-gravity and frame gates, the self-gravity
// gate and the expansion trigger.  Absent keys are skipped.
const char *const kDensityThresholdKeys[][2] = {
    {"hydro", "pfloor"}, {"problem", "rho_floor"}, {"problem", "bh_grav_rho_min"},
    {"problem", "frame_rho_min"}, {"gravity", "rho_grav_min"},
    {"remap", "expand_rho_min"}};

// carry <outputN> file_number/last_time forward so the remapped run continues the
// source run's output sequence (matched by block name)
void CopyRemapOutputState(ParameterInput *dst, ParameterInput *src) {
  if (dst == nullptr) return;
  if (src == nullptr) return;
  for (const auto &block : src->block) {
    if (block.block_name.compare(0, 6, "output") != 0) continue;
    if (!dst->DoesBlockExist(block.block_name)) continue;
    if (src->DoesParameterExist(block.block_name, "file_number")) {
      dst->SetInteger(block.block_name, "file_number",
                      src->GetInteger(block.block_name, "file_number"));
    }
    if (src->DoesParameterExist(block.block_name, "last_time")) {
      dst->SetReal(block.block_name, "last_time",
                   src->GetReal(block.block_name, "last_time"));
    }
  }
}

// Settle steps: see remap.hpp.  One remap per process, so the state is file-scope.
struct SettleState {
  bool active = false;
  int steps = 0;
  int passes_remaining = 0;
  int target_cycle = -1;
  bool copy_output_state = true;
  int saved_ncycle_check = 1;
  int saved_refinement_interval = 1;
  RemapOptions opts;
  std::function<void(ParameterInput *)> on_loaded;
  std::function<void(const RemapSummary &)> post;
};
SettleState settle;

void BeginSettle(ProblemGenerator *pgen, ParameterInput *pin, Mesh *pm,
                 const RemapOptions &opts, const bool copy_out, const int steps,
                 const int passes) {
  if (steps <= 0 || passes <= 0) return;
  settle.active = true;
  settle.steps = steps;
  settle.passes_remaining = passes;
  settle.target_cycle = pm->ncycle + steps;
  settle.copy_output_state = copy_out;
  settle.opts = opts;
  // A pass runs in the middle of the run: it is allowed under LAT only because LAT is
  // paused for it, so it must not inherit the startup remap's lat_idle.
  settle.opts.lat_idle = false;
  settle.on_loaded = pgen->user_remap_loaded_func;
  settle.post = pgen->user_remap_post_func;
  if (pm->pmr != nullptr) {
    settle.saved_ncycle_check = pm->pmr->ncyc_check_amr;
    settle.saved_refinement_interval = pm->pmr->refinement_interval;
    pm->pmr->ncyc_check_amr = 1;
    pm->pmr->refinement_interval = 1;
  }
  pm->hydro_lat_suspended = pin->IsLATEnabled();
  if (global_variable::my_rank == 0) {
    std::cout << "remap settle steps     = " << steps << std::endl
              << "remap settle passes    = " << passes << std::endl
              << "first settle pass      = cycle " << settle.target_cycle << std::endl
              << "AMR cadence            = every cycle until the settle ends "
              << "(then ncycle_check=" << settle.saved_ncycle_check
              << ", refinement_interval=" << settle.saved_refinement_interval << ")"
              << std::endl
              << "outputs suppressed     = true until the settle ends" << std::endl
              << "LAT                    = "
              << (pm->hydro_lat_suspended ? "paused until the settle ends" : "off")
              << std::endl << std::endl;
  }
}

}  // namespace

RemapSummary LoadAndApplyRemap(Mesh *pm, MeshBlockPack *pmbp, ParameterInput *dst_pin,
                               bool copy_output_state, const char *banner_label,
                               const RemapOptions &opts,
                               const std::function<void(ParameterInput *)> &on_loaded) {
  RemapSummary summary;
  if (pm == nullptr || pmbp == nullptr) return summary;
  // Remap and LAT (<time>/lat) can share a run, but not a LAT window.  The remap replaces
  // the state on every block at once, while a window carries per-block time levels, stage
  // snapshots and reflux accumulators built against the state that was there before.  The
  // startup remap qualifies (no window exists yet), and so does a mid-run remap made
  // while the pgen has paused LAT (Mesh::hydro_lat_suspended).  Anything else is refused
  // here, on the one path every caller (auto <remap> block, direct call) takes.
  const bool lat_on = dst_pin != nullptr && dst_pin->IsLATEnabled();
  if (lat_on && !opts.lat_idle && !pm->hydro_lat_suspended) {
    FatalRemap("A remap cannot run inside a LAT window (time/lat = true).  Remap at "
               "startup, or pause LAT first by setting Mesh::hydro_lat_suspended at a "
               "synchronized point, as the remap settle steps do.");
  }
  if (opts.source_path.empty()) {
    std::cout << "### FATAL ERROR in remap::LoadAndApplyRemap" << std::endl
              << "Remap requested without a valid source restart path." << std::endl;
    std::exit(EXIT_FAILURE);
  }

  ParameterInput src_pin;
  impl::RemapSourceData src;
  impl::LoadRemapSourceData(opts.source_path, pm, pmbp, opts, src, src_pin);

  // ---- relativity class: resolve the band mode, then audit the two parameter sets -----
  RemapOptions eff_opts = opts;
  if (eff_opts.band_mode_auto) {
    eff_opts.band_mode = src.gr_mode ? RemapBandMode::kKeepTarget
                                     : RemapBandMode::kFloorFade;
  } else if (src.gr_mode && eff_opts.band_mode == RemapBandMode::kFloorFade) {
    FatalRemap(std::string("<remap>/band_mode = floor is invalid for a ") +
               impl::RemapRelClassName(src.rel_class) + " remap: the floor-fade band "
               "reconstructs a Newtonian ambient state (rho/p floors, v = p/rho, "
               "E = eint + 0.5*rho*v^2), which has no meaning for densitized GR "
               "conserved variables.  Use band_mode = keep (or auto).");
  }
  impl::CheckRemapGRConsistency(src_pin, dst_pin, src, pmbp);

  const bool is_mhd = (pmbp->pmhd != nullptr);
  const bool use_dual = is_mhd ? pmbp->pmhd->use_dual_energy
                               : pmbp->phydro->use_dual_energy;

  // <remap>/floor_reset: the target was built on kFloorResetRatio times the largest
  // source density, or the input gets those floors and this pass ends (remap_impl.hpp)
  const bool floor_reset = dst_pin != nullptr &&
                           dst_pin->GetOrAddBoolean("remap", "floor_reset", false);
  Real reset_rho_max = 0.0, floor_replace_below = 0.0;
  if (floor_reset) {
    if (src.gr_mode || is_mhd) {
      FatalRemap("<remap>/floor_reset is implemented for Newtonian hydro only.");
    }
    reset_rho_max = impl::SourceDensityMax(src);
    if (MatchFloorsToDensity(dst_pin, reset_rho_max,
                             pmbp->phydro->peos->eos_data.dfloor)) {
      floor_reset_pass = true;
      return summary;
    }
    // the source's floor gas is replaced only where the floor has fallen since
    const Real src_dfloor = src_pin.GetOrAddReal("hydro", "dfloor", 0.0);
    if (src_dfloor > (1.0 + 1.0e-6)*pmbp->phydro->peos->eos_data.dfloor) {
      floor_replace_below = kFloorReplaceFactor*src_dfloor;
    }
  }
  // The Newtonian floor prep (outer source ghost zones floored, dual-energy aux column
  // rebuilt from the remapped state) is meaningless in GR: there is no ambient floor
  // state, and the keep-mode pass never writes one.
  if (!src.gr_mode) {
    auto eos = is_mhd ? pmbp->pmhd->peos->eos_data : pmbp->phydro->peos->eos_data;
    const Real floor_rho = eos.dfloor;
    const Real floor_eint = eos.HostClampHydroInternalEnergyDensity(
        floor_rho, eos.HostHydroInternalEnergyDensityFloor(floor_rho));
    const int dual_idx = use_dual ? (is_mhd ? pmbp->pmhd->dual_energy_idx
                                            : pmbp->phydro->dual_energy_idx) : -1;
    impl::FloorOuterSourceGhostZones(src, dual_idx, floor_rho, floor_eint);
    if (floor_replace_below > 0.0) {
      impl::ReplaceSourceFloorCells(src, dual_idx, floor_replace_below, floor_rho,
                                    floor_eint);
    }
    // The CC engine interpolates the source's THERMAL energy and rebuilds the total from
    // it (see sample_full_state in remap_cc.cpp), so that column is built here: after the
    // outer ghost zones are floored, whose thermal state it has to agree with.
    const int base_nvars = is_mhd ? (pmbp->pmhd->nmhd + pmbp->pmhd->nscalars)
                                  : (pmbp->phydro->nhydro + pmbp->phydro->nscalars);
    impl::BuildSourceThermalEnergy(src, base_nvars, floor_rho, floor_eint);
  }

  if (on_loaded) {
    on_loaded(&src_pin);
  }

  // The GR gas entropy (D s, MHD/Hydro::gas_entropy_idx) is a conserved column like the
  // passive scalars, and a source that stores it hands it over with them: the remapped
  // D s is the mass-weighted entropy of the source cells, which the first recovery's
  // entropy root (KORAL's U2P_ENTROPY) needs where the remapped energy has no root.  The
  // keep pass blends it with the target's own column outside the source box, so that
  // column is first made the entropy of the target's state (generator or previous run).
  const bool tgt_entropy = is_mhd ? (pmbp->pmhd->gas_entropy_idx >= 0)
                                  : (pmbp->phydro->gas_entropy_idx >= 0);
  if (tgt_entropy && src.gas_entropy_column) {
    if (is_mhd) {
      pmbp->pmhd->ResyncGasEntropy(true);
    } else {
      pmbp->phydro->ResyncGasEntropy(true);
    }
  }

  impl::ApplyRemapCC(pm, pmbp, eff_opts, src);
  if (src.load_fc) {
    impl::BuildCoveringPotential(src, eff_opts);
    impl::ApplyRemapFC(pm, pmbp, eff_opts, src);
  }
  // A floor-fade pass writes the dual-energy column of every target cell (the source's
  // own auxiliary in a copied cell, else the remapped e_int, which is the internal energy
  // the source's inversion read), so the startup re-seed from E - KE, which would discard
  // the auxiliary the source trusted in its cold cells, is skipped.  A keep pass leaves
  // the target's pgen cells to be seeded.
  if (!src.gr_mode && use_dual) {
    const bool seed = (eff_opts.band_mode == RemapBandMode::kKeepTarget);
    if (is_mhd) {
      pmbp->pmhd->dual_energy_needs_init = seed;
    } else {
      pmbp->phydro->dual_energy_needs_init = seed;
    }
  }

  // A source without the column (a GR checkpoint written before it existed) left the
  // target's column as the generator wrote it: it is taken from the remapped state after
  // the first recovery (Driver::Initialize), as a restart of such a checkpoint takes it.
  if (tgt_entropy) {
    const bool pending = !src.gas_entropy_column;
    if (is_mhd) {
      pmbp->pmhd->gas_entropy_init_pending = pending;
    } else {
      pmbp->phydro->gas_entropy_init_pending = pending;
    }
  }

  pm->time = src.time;
  pm->dt = src.dt;
  pm->dtold = src.dt;
  pm->ncycle = src.ncycle;

  // Under LAT, the AMR and rebalance gates compare ncycle with the cycle of the last AMR
  // call, topology change and rebalance attempt.  A mid-run remap rewinds ncycle to the
  // source's, so a later anchor would hold those gates shut until ncycle caught up.  The
  // per-block factors were derived from the old state; the driver rebuilds them from the
  // new one at the next time step.
  if (lat_on) {
    if (pm->pmr != nullptr) {
      pm->pmr->last_amr_call_cycle =
          std::min(pm->pmr->last_amr_call_cycle, pm->ncycle);
    }
    pm->topology_last_change_cycle =
        std::min(pm->topology_last_change_cycle, pm->ncycle);
    pm->hydro_lat_lb_last_attempt_cycle =
        std::min(pm->hydro_lat_lb_last_attempt_cycle, pm->ncycle);
    pm->InvalidateHydroLATMetadata();
  }

  // Refresh the analytic ADM metric (and, for BBH, the excision + M1 radiation masks) at
  // the SOURCE time now that pm->time carries it.  Dyn-GR pgens whose metric callback is
  // registered therefore need no remap-specific code at all: the state the pgen built at
  // t = 0 has been replaced by state at t = src.time, and this puts the spacetime the
  // rest of the initialization (C2P, boundary values) sees back in sync with it.
  if (pmbp->padm != nullptr && pmbp->padm->SetADMVariables != nullptr) {
    pmbp->padm->SetADMVariables(pmbp);
  }

  summary.time = src.time;
  summary.ncycle = src.ncycle;
  summary.fc_applied = src.load_fc;
  summary.gr_mode = src.gr_mode;
  summary.i0_applied = src.load_i0;

  if (dst_pin != nullptr) {
    dst_pin->SetReal("time", "start_time", src.time);
    if (copy_output_state) {
      CopyRemapOutputState(dst_pin, &src_pin);
    }
  }

  if (global_variable::my_rank == 0) {
    auto group_state = [](const bool applied, const bool discarded,
                          const bool target_only, const bool present_source) {
      if (applied) return "yes";
      if (discarded) return "no (source group discarded: target lacks the module)";
      if (target_only) return "no (source has none; target pgen state kept)";
      if (present_source) return "no (disabled by option)";
      return "no";
    };
    std::cout << std::endl
              << ((banner_label != nullptr) ? banner_label : "--- Remap ---") << std::endl
              << "source restart         = " << eff_opts.source_path << std::endl
              << "copy output state      = "
              << (copy_output_state ? "true" : "false") << std::endl
              << "relativity class       = "
              << impl::RemapRelClassName(src.rel_class) << std::endl
              << "band mode              = "
              << ((eff_opts.band_mode == RemapBandMode::kKeepTarget)
                      ? "keep (target pgen state is the ambient)"
                      : "floor (ambient hydro floor outside the source box)")
              << (eff_opts.band_mode_auto ? " [auto]" : " [explicit]") << std::endl
              << "transition band        = "
              << (eff_opts.use_transition_band ? "on" : "off (pure interpolation)")
              << std::endl
              << "source time            = " << src.time << std::endl;
    if (floor_reset) {
      std::cout << "floor reset            = dfloor "
                << pmbp->phydro->peos->eos_data.dfloor << " (" << kFloorResetRatio
                << " x source max density " << reset_rho_max << "); source cells below "
                << floor_replace_below << " set to the floor" << std::endl;
    }
    std::cout
              << "source cycle           = " << src.ncycle << std::endl
              << "loaded source blocks   = " << src.blocks.size() << std::endl
              << "gas CC group           = " << (src.source_has_mhd ? "mhd" : "hydro")
              << " (nvars_file = " << src.nvars
              << (src.gr_mode ? (src.gas_entropy_column ? ", gas entropy D s remapped"
                                                        : ", no gas entropy column: "
                                                          "set from the remapped state")
                              : "")
              << ")" << std::endl
              << "face-centered B remap  = " << (src.load_fc ? "yes" : "no") << std::endl
              << "radiation i0 remap     = "
              << group_state(src.load_i0, src.i0_discarded, src.i0_target_only,
                             src.source_has_rad) << std::endl;
    if (src.nadm_vars > 0) {
      std::cout << "skipped source <adm>   = " << src.nadm_vars << " variables"
                << std::endl;
    }
    std::cout << std::endl;
  }
  return summary;
}

bool IsAutoRemapEnabled(ParameterInput *pin) {
  if (pin == nullptr || !pin->DoesBlockExist("remap")) return false;
  return pin->GetOrAddBoolean("remap", "enable", true);
}

RemapOptions OptionsFromInput(ParameterInput *pin) {
  RemapOptions opts;
  opts.source_path = pin->GetOrAddString("remap", "source", "");
  opts.use_transition_band = pin->GetOrAddBoolean("remap", "transition_band", true);
  // band_mode: auto (default) | floor | keep.  "auto" is resolved from the relativity
  // class once the source has been classified, in LoadAndApplyRemap.
  std::string band = pin->GetOrAddString("remap", "band_mode", "auto");
  std::transform(band.begin(), band.end(), band.begin(),
                 [](unsigned char c) { return std::tolower(c); });
  if (band == "auto") {
    opts.band_mode_auto = true;
    opts.band_mode = RemapBandMode::kFloorFade;
  } else if (band == "floor") {
    opts.band_mode_auto = false;
    opts.band_mode = RemapBandMode::kFloorFade;
  } else if (band == "keep") {
    opts.band_mode_auto = false;
    opts.band_mode = RemapBandMode::kKeepTarget;
  } else {
    FatalRemap("Unknown <remap>/band_mode = '" + band +
               "'; expected auto, floor or keep.");
  }
  opts.b_coarsen_ok = pin->GetOrAddBoolean("remap", "b_coarsen_ok", false);
  opts.b_taper_root_cells = pin->GetOrAddInteger("remap", "b_taper_root_cells", 4);
  opts.b_report = pin->GetOrAddBoolean("remap", "b_report", true);
  opts.remap_radiation_i0 = pin->GetOrAddBoolean("remap", "radiation_i0", true);
  return opts;
}

bool MatchFloorsToDensity(ParameterInput *pin, const Real rho_max, const Real dfloor) {
  if (!(rho_max > 0.0) || !(dfloor > 0.0)) {
    FatalRemap("<remap>/floor_reset needs a positive source density and <hydro>/dfloor.");
  }
  const bool root = (global_variable::my_rank == 0);
  bool changed = false;
  auto header = [&]() {
    if (root && !changed) {
      std::cout << std::endl << "--- Remap floor reset ---" << std::endl
                << "largest density        = " << rho_max << std::endl;
    }
    changed = true;
  };
  // the floor only falls: a target at or above the present floor leaves it and its
  // thresholds
  const Real target = kFloorResetRatio*rho_max;
  const Real s = target/dfloor;
  if (s < 1.0 - 1.0e-6) {
    header();
    if (root) {
      std::cout << "hydro/dfloor = " << dfloor << " -> " << target << " (x " << s << ")"
                << std::endl;
    }
    pin->SetReal("hydro", "dfloor", target);
    for (const auto &key : kDensityThresholdKeys) {
      if (!pin->DoesParameterExist(key[0], key[1])) continue;
      const Real v = pin->GetReal(key[0], key[1]);
      if (!(v > 0.0)) continue;
      pin->SetReal(key[0], key[1], v*s);
      if (root) {
        std::cout << key[0] << "/" << key[1] << " = " << v << " -> " << v*s << std::endl;
      }
    }
  }
  // the <tde_amr> ambient cut follows the largest density itself, both ways
  if (pin->DoesParameterExist("tde_amr", "rho_a")) {
    const Real rho_a = pin->GetReal("tde_amr", "rho_a");
    const Real rho_a_new = kAmbientCutRatio*rho_max;
    if (rho_a > 0.0 && std::abs(rho_a_new/rho_a - 1.0) > 1.0e-6) {
      header();
      if (root) {
        std::cout << "tde_amr/rho_a = " << rho_a << " -> " << rho_a_new << std::endl;
      }
      pin->SetReal("tde_amr", "rho_a", rho_a_new);
    }
  }
  if (root && changed) std::cout << std::endl;
  return changed;
}

bool TakeFloorResetPass() {
  const bool pending = floor_reset_pass;
  floor_reset_pass = false;
  return pending;
}

void MaybeAutoRemap(ProblemGenerator *pgen, ParameterInput *pin, Mesh *pm) {
  if (!IsAutoRemapEnabled(pin)) return;
  const int settle_steps = pin->GetOrAddInteger("remap", "settle_steps", 0);
  const int settle_passes = pin->GetOrAddInteger("remap", "settle_passes", 1);
  if (settle_steps < 0 || settle_passes < 0) {
    FatalRemap("<remap>/settle_steps and <remap>/settle_passes must be >= 0.");
  }
  RemapOptions opts = OptionsFromInput(pin);
  opts.skip_cell = pgen->user_remap_skip_func;
  opts.lat_idle = true;  // startup: no LAT window exists yet
  const bool copy_out = pin->GetOrAddBoolean("remap", "copy_output_state", true);
  const RemapSummary summary =
      LoadAndApplyRemap(pm, pm->pmb_pack, pin, copy_out, "--- Remap (<remap> block) ---",
                        opts, pgen->user_remap_loaded_func);
  if (floor_reset_pass) return;  // main builds the next pass on the new floors
  if (pgen->user_remap_post_func) {
    pgen->user_remap_post_func(summary);
  }
  BeginSettle(pgen, pin, pm, opts, copy_out, settle_steps, settle_passes);
}

bool SettleActive() { return settle.active; }

bool AfterCycleSettle(Driver *driver, ParameterInput *pin, Mesh *pm) {
  if (!settle.active || pm->ncycle < settle.target_cycle) return false;
  if (driver == nullptr || pin == nullptr || pm->pmb_pack == nullptr) return false;

  if (global_variable::my_rank == 0) {
    std::cout << std::endl
              << "--- Remap settle pass ---" << std::endl
              << "current cycle           = " << pm->ncycle << std::endl
              << "current time            = " << pm->time << std::endl
              << "target cycle            = " << settle.target_cycle << std::endl
              << "remaining passes before = " << settle.passes_remaining << std::endl
              << std::endl;
  }
  // LAT is still paused here (Mesh::hydro_lat_suspended), which is what lets the remap
  // replace the state.
  const RemapSummary summary =
      LoadAndApplyRemap(pm, pm->pmb_pack, pin, settle.copy_output_state,
                        "--- Remap settle pass (<remap> block) ---", settle.opts,
                        settle.on_loaded);
  if (settle.post) settle.post(summary);
  --settle.passes_remaining;
  if (settle.passes_remaining > 0) {
    settle.target_cycle = pm->ncycle + settle.steps;
  } else {
    settle.active = false;
    settle.target_cycle = -1;
    if (pm->pmr != nullptr) {
      pm->pmr->ncyc_check_amr = settle.saved_ncycle_check;
      pm->pmr->refinement_interval = settle.saved_refinement_interval;
    }
    pm->hydro_lat_suspended = false;
  }
  driver->InitBoundaryValuesAndPrimitives(pm);
  if (global_variable::my_rank == 0) {
    std::cout << "remap settle passes remaining = " << settle.passes_remaining
              << std::endl;
    if (settle.active) {
      std::cout << "next settle pass              = cycle " << settle.target_cycle
                << std::endl;
    } else {
      std::cout << "settle passes complete; AMR cadence restored (ncycle_check="
                << settle.saved_ncycle_check << ", refinement_interval="
                << settle.saved_refinement_interval << ")" << std::endl;
    }
    std::cout << std::endl;
  }
  return true;
}

}  // namespace remap
