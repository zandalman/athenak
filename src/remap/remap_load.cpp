//========================================================================================
// AthenaK astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file remap_load.cpp
//! \brief restart-source parsing and loading for the universal remap module.
//!
//! Restart layout consumed here (see docs/design/remap_module_design.md section 3):
//! ASCII parameter dump to <par_end> | mesh binary header | LogicalLocation list |
//! float cost list | (refused step-3 internal state) | int MeshBlock refinement ages
//! (adaptive writer, skipped) | IOWrapperSizeT data_size |
//! block-major payload.  Per-block section order (mirrors src/outputs/restart.cpp):
//! hydro u0 | mhd u0 | b0.x1f | b0.x2f | b0.x3f | radiation i0 |
//! turbulence force (refused) | z4c u0 (refused) XOR adm u_adm
//! (skipped) | carried mhd w0 | carried hydro w0
//! (skipped). Arrays include ghosts; faces are +1 in their own direction.  A GR fluid's
//! u0 ends with the gas-entropy column D s unless the checkpoint predates it.

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "athena.hpp"
#include "globals.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "mesh/meshblock_pack.hpp"
#include "coordinates/adm.hpp"
#include "eos/eos.hpp"
#include "geodesic-grid/geodesic_grid.hpp"
#include "hydro/hydro.hpp"
#include "mhd/mhd.hpp"
#include "radiation/radiation.hpp"
#include "remap/remap.hpp"
#include "remap/remap_impl.hpp"

namespace remap {
namespace impl {

namespace {

[[noreturn]] void FatalLoad(const std::string &msg) {
  std::cout << "### FATAL ERROR in remap module (source load)" << std::endl
            << msg << std::endl;
  std::exit(EXIT_FAILURE);
}

void WarnRemap(const std::string &msg) {
  if (global_variable::my_rank != 0) return;
  std::cout << "WARNING (remap): " << msg << std::endl;
}

// Read-only accessors: NEVER GetOrAdd (that would inject the key into the pin and, for
// the target pin, into every restart parameter dump this run writes from now on).
bool PinBool(ParameterInput *pin, const char *block, const char *key, const bool def) {
  if (pin == nullptr || !pin->DoesParameterExist(block, key)) return def;
  return pin->GetBoolean(block, key);
}

std::string PinString(ParameterInput *pin, const char *block, const char *key,
                      const char *def) {
  if (pin == nullptr || !pin->DoesParameterExist(block, key)) return std::string(def);
  return pin->GetString(block, key);
}

// Arm the un-densitization sampler for this remap.  It is built at the SOURCE time on
// purpose: while the remap runs, padm still holds the metric the pgen installed at t = 0,
// and for a binary the punctures have since moved -- dividing the source's densitized
// state by the wrong metric would be worse than not dividing at all.
void BuildRemapMetricSampler(MeshBlockPack *pmbp, RemapSourceData &src) {
  // Only the dyn-GR (<adm>) path densitizes its conserved variables; see the struct
  // comment for why fixed-GR is deliberately left alone.
  if (!src.gr_mode || pmbp->padm == nullptr) return;
  if (!pmbp->padm->IsAnalyticBBH()) {
    WarnRemap("the target's ADM backend stores its metric on the target grid only, so "
              "there is no way to evaluate gamma_ij at a source cell center; the "
              "densitized conserved variables are interpolated as stored, and the "
              "recovered primitives carry the metric's curvature as an O(h^2) error.");
    return;
  }
  src.metric.kind = RemapMetricSampler::Kind::kAnalyticBBH;
  src.metric.bbh_view = pmbp->padm->GetMetricView(src.time);
}

// Source blocks that a remap can read for THIS rank's target blocks.
//
// RULE.  A source block is selected iff it overlaps (closed boxes) at least one target
// block of this rank, grown by a stencil margin.  Per target block T, let C be T clamped
// to the source mesh and dq the largest source cell width (per axis) among the leaves that
// overlap C.  The margin is
//     2 dq                        if T lies deeper than (g + 1) root cells inside the source
//                                 mesh on every axis, g = max(ngh + 1, 3);
//     (g + 3) root cells          otherwise (T within the transition band of a mesh face,
//                                 or outside the mesh), when the transition band is on.
//
// PROOF SKETCH (every claim refers to remap_cc.cpp).
//  1. Every sub-sample point p of a target cell of T maps to p_src = clamp(p) in C, and the
//     leaf B that FindContainingSourceBlock returns for it overlaps C, so its cell width
//     dq_B <= dq.  The target-cell center test (TargetCellMatchesSourceGrid) is also in C.
//  2. The trilinear stencil (SampleSourceCellCenteredArr) uses the cells whose centers
//     bracket p_src, so its node centers lie within one cell width dq_B of p_src.  A node
//     outside B's active cells is not read from B's ghost zones: it is re-sampled
//     (SampleSourceActiveCellContainingArr) from the leaf that contains the node center,
//     which is the finest loaded leaf there whatever its level, and it overlaps the box
//     C + dq_B.  So at coarse/fine interfaces the neighbour (finer or coarser) is reached
//     iff it overlaps C grown by dq <= 2 dq.  Ghost zones are never read at all.
//  3. Keep mode, the plain sampler and the band-off paths sample only at p_src: the taper
//     decays the WEIGHT, not the sample position, so step 2 is the whole reach.
//  4. Floor-fade with the transition band can move the sample: the reference point is
//     clamped to at most g cells (of dq_B <= root cell) inside p_src, the deeper point one
//     cell further, the ambient inward shift one more, and the stencil one more, so every
//     read lies within (g + 3) root cells of C.  That path needs a weight < 1 or a point
//     outside the mesh, which requires T within g root cells of a face (dq_B <= root cell).
//  5. The face-centered B never uses this set: the covering grid is streamed from every
//     block, and the per-block magnetic-energy swap uses the block's own faces.
// The caller also keeps the old per-rank bounding-box test (a superset of this one), so
// the loaded set is a subset of the old set, and the result is bitwise identical: each
// block the samplers can reach is present in both, every other block is never read.
//
// Cost: source blocks are binned on a uniform lattice (about one block per bin), each
// target block is two bin queries, so O(Ntarget * blocks per queried bin), not
// O(Ntarget * Nsource).
std::vector<char> SelectSourceBlocksByTarget(const RemapSourceData &src,
                                             const std::vector<RegionSize> &bsize,
                                             Mesh *pm, const bool use_band,
                                             const Real root_dx[3]) {
  const int nb = std::clamp(
      static_cast<int>(std::ceil(std::cbrt(static_cast<double>(src.nmb_total)))), 1, 64);
  const Real lo[3] = {src.mesh_size.x1min, src.mesh_size.x2min, src.mesh_size.x3min};
  const Real hi[3] = {src.mesh_size.x1max, src.mesh_size.x2max, src.mesh_size.x3max};
  auto bin_of = [&](const int a, const Real q) {
    const Real f = std::floor((q - lo[a]) / (hi[a] - lo[a]) * static_cast<Real>(nb));
    return std::clamp(static_cast<int>(std::max(f, static_cast<Real>(-1.0))), 0, nb - 1);
  };
  auto bmin = [](const RegionSize &r, const int a) {
    return (a == 0) ? r.x1min : ((a == 1) ? r.x2min : r.x3min);
  };
  auto bmax = [](const RegionSize &r, const int a) {
    return (a == 0) ? r.x1max : ((a == 1) ? r.x2max : r.x3max);
  };
  auto bdx = [](const RegionSize &r, const int a) {
    return (a == 0) ? r.dx1 : ((a == 1) ? r.dx2 : r.dx3);
  };

  std::vector<std::vector<int>> bins(static_cast<std::size_t>(nb) * nb * nb);
  for (int n = 0; n < src.nmb_total; ++n) {
    int b0[3], b1[3];
    for (int a = 0; a < 3; ++a) {
      b0[a] = bin_of(a, bmin(bsize[n], a));
      b1[a] = bin_of(a, bmax(bsize[n], a));
    }
    for (int k = b0[2]; k <= b1[2]; ++k) {
      for (int j = b0[1]; j <= b1[1]; ++j) {
        for (int i = b0[0]; i <= b1[0]; ++i) {
          bins[(static_cast<std::size_t>(k) * nb + j) * nb + i].push_back(n);
        }
      }
    }
  }

  // visit every source block whose closed box meets [qlo, qhi]
  auto query = [&](const Real qlo[3], const Real qhi[3], auto &&visit) {
    int b0[3], b1[3];
    for (int a = 0; a < 3; ++a) {
      b0[a] = bin_of(a, qlo[a]);
      b1[a] = bin_of(a, qhi[a]);
    }
    for (int k = b0[2]; k <= b1[2]; ++k) {
      for (int j = b0[1]; j <= b1[1]; ++j) {
        for (int i = b0[0]; i <= b1[0]; ++i) {
          for (const int n : bins[(static_cast<std::size_t>(k) * nb + j) * nb + i]) {
            const RegionSize &r = bsize[n];
            if (bmin(r, 0) <= qhi[0] && bmax(r, 0) >= qlo[0] &&
                bmin(r, 1) <= qhi[1] && bmax(r, 1) >= qlo[1] &&
                bmin(r, 2) <= qhi[2] && bmax(r, 2) >= qlo[2]) visit(n);
          }
        }
      }
    }
  };

  const Real g = static_cast<Real>(std::max(src.ngh + 1, 3));
  std::vector<char> needed(src.nmb_total, 0);
  auto &mb_size = pm->pmb_pack->pmb->mb_size;
  for (int m = 0; m < pm->pmb_pack->nmb_thispack; ++m) {
    Real tlo[3] = {mb_size.h_view(m).x1min, mb_size.h_view(m).x2min,
                   mb_size.h_view(m).x3min};
    Real thi[3] = {mb_size.h_view(m).x1max, mb_size.h_view(m).x2max,
                   mb_size.h_view(m).x3max};
    Real clo[3], chi[3], dq[3] = {0.0, 0.0, 0.0};
    bool near_face = false;
    for (int a = 0; a < 3; ++a) {
      clo[a] = std::clamp(tlo[a], lo[a], hi[a]);
      chi[a] = std::clamp(thi[a], lo[a], hi[a]);
      if (tlo[a] < lo[a] + (g + 1.0) * root_dx[a] ||
          thi[a] > hi[a] - (g + 1.0) * root_dx[a]) near_face = true;
    }
    query(clo, chi, [&](const int n) {
      for (int a = 0; a < 3; ++a) dq[a] = std::max(dq[a], bdx(bsize[n], a));
    });
    Real qlo[3], qhi[3];
    for (int a = 0; a < 3; ++a) {
      if (!(dq[a] > 0.0)) dq[a] = root_dx[a];   // no leaf found: be conservative
      const Real margin = (use_band && near_face) ? (g + 3.0) * root_dx[a]
                                                  : 2.0 * dq[a];
      qlo[a] = clo[a] - margin;
      qhi[a] = chi[a] + margin;
    }
    query(qlo, qhi, [&](const int n) { needed[n] = 1; });
  }
  return needed;
}

}  // namespace

//----------------------------------------------------------------------------------------
//! \fn RemapMetricSampler::SpatialMetric
//! \brief gamma_ij of the prescribed metric at an arbitrary point.  ADMMetricView's
//! evaluator is a KOKKOS_INLINE_FUNCTION, i.e. callable from this host-side setup code,
//! and it is exactly the evaluation the target's own kernels perform at their cell
//! centers -- so a source node that lands on a target cell center agrees bitwise.

void RemapMetricSampler::SpatialMetric(const Real x, const Real y, const Real z,
                                       Real g_dd[NSPMETRIC]) const {
  if (kind == Kind::kAnalyticBBH) {
    // The BASE evaluation is the right one: it defines gamma_ij, alpha and beta^i, and
    // differs from the full one only in K_ij and psi4, which no densitization uses.
    adm::ADMMetricPoint point{};
    bbh_view.CartesianMetric(x, y, z, point);
    for (int n = 0; n < NSPMETRIC; ++n) g_dd[n] = point.g_dd[n];
    return;
  }
  g_dd[S11] = 1.0; g_dd[S12] = 0.0; g_dd[S13] = 0.0;
  g_dd[S22] = 1.0; g_dd[S23] = 0.0; g_dd[S33] = 1.0;
}

Real RemapMetricSampler::SqrtGamma(const Real x, const Real y, const Real z) const {
  if (!Active()) return 1.0;
  Real g_dd[NSPMETRIC];
  SpatialMetric(x, y, z, g_dd);
  const Real det = adm::SpatialDet(g_dd[S11], g_dd[S12], g_dd[S13],
                                   g_dd[S22], g_dd[S23], g_dd[S33]);
  // A non-positive determinant means the point sits on a coordinate pathology (a
  // puncture centre).  Those cells are excised by the target and their state is reset
  // every stage, so returning the flat value there keeps the sampler total instead of
  // propagating a NaN into every neighbouring interpolation.
  return (det > 0.0) ? std::sqrt(det) : 1.0;
}

const char *RemapRelClassName(RemapRelClass c) {
  switch (c) {
    case RemapRelClass::kFixedGR:       return "fixed-GR (<coord>/general_rel)";
    case RemapRelClass::kDynGRAnalytic: return "dynamical-GR (<adm> analytic metric)";
    default:                            return "Newtonian";
  }
}

bool LoadRemapSourceData(const std::string &path, Mesh *pm, MeshBlockPack *pmbp,
                         const RemapOptions &opts, RemapSourceData &src,
                         ParameterInput &src_pin) {
  IOWrapper input;
  input.Open(path.c_str(), IOWrapper::FileMode::read, false);
  src_pin.LoadFromFile(input, false);

  // ---- source physics whitelist ------------------------------------------------------
  // Step-3 internal state (z4c trackers, turbulence RNG, sink list) carries no length
  // markers in the restart, so any source with those modules cannot be parsed here.
  if (src_pin.DoesBlockExist("z4c") || src_pin.DoesBlockExist("cce")) {
    FatalLoad("Remap of z4c restarts is unsupported.");
  }
  if (src_pin.DoesBlockExist("turbulence") || src_pin.DoesBlockExist("turb_driving")) {
    FatalLoad("Remap sources with turbulence driving (RNG step-3 state + force section) "
              "are unsupported.");
  }
  if (src_pin.DoesBlockExist("sink_particles")) {
    FatalLoad("Remap sources with <sink_particles> are unsupported.");
  }
  if (src_pin.DoesBlockExist("shearing_box")) {
    FatalLoad("Remap sources with <shearing_box> are unsupported.");
  }

  // ---- source relativity class -------------------------------------------------------
  // <adm> wins over the <coord> flags: Coordinates reports is_general_relativistic ==
  // false for the dyn_grmhd path, so a Coordinates-only test misclassifies BBH restarts.
  const bool src_has_hydro = src_pin.DoesBlockExist("hydro");
  src.source_has_mhd = src_pin.DoesBlockExist("mhd");
  if (src_pin.DoesBlockExist("adm")) {
    src.rel_class = RemapRelClass::kDynGRAnalytic;
    if (!src.source_has_mhd && !src_has_hydro) {
      FatalLoad("Remap source carries <adm> but neither <mhd> nor <hydro>: the "
                "dyn_grmhd remap path requires the GRMHD conserved group.");
    }
    // <adm> with <hydro> and no <mhd> is the field-free dynamical-GR fluid (GRHD,
    // MHD::HasField() false): the same conserved group as GRMHD (checkpointed as the
    // "mhd u0" section), no face field.
    src.source_field_free = !src.source_has_mhd;
  } else if (PinBool(&src_pin, "coord", "special_rel", false)) {
    FatalLoad("Remap sources must not be special relativistic (<coord>/special_rel).");
  } else if (PinBool(&src_pin, "coord", "general_rel", false)) {
    src.rel_class = RemapRelClass::kFixedGR;
  } else {
    src.rel_class = RemapRelClass::kNewtonian;
  }
  src.gr_mode = (src.rel_class != RemapRelClass::kNewtonian);

  src.source_has_rad = src_pin.DoesBlockExist("radiation");
  if (src_has_hydro == src.source_has_mhd) {
    FatalLoad("Remap source must carry exactly one of <hydro> or <mhd>.");
  }

  // ---- target compatibility ----------------------------------------------------------
  hydro::Hydro *phyd = pmbp->phydro;
  mhd::MHD *pmhd = pmbp->pmhd;
  if (phyd == nullptr && pmhd == nullptr) {
    FatalLoad("Remap target has neither hydro nor MHD.");
  }
  if (phyd != nullptr && pmhd != nullptr) {
    FatalLoad("Remap target with both hydro and MHD (two-fluid) is unsupported.");
  }
  if (src.source_has_mhd && pmhd == nullptr) {
    FatalLoad("MHD remap source but the target is hydro-only.");
  }
  if (src.source_has_mhd && !pmhd->HasField()) {
    FatalLoad("MHD remap source (it stores a face field) but the target is the "
              "field-free dynamical-GR fluid (<hydro> with <adm>).");
  }
  if (src.source_field_free && (pmhd == nullptr || pmhd->HasField())) {
    FatalLoad("Remap source is the field-free dynamical-GR fluid (<hydro> with <adm>) "
              "but the target is not.");
  }
  if (src_has_hydro && !src.source_field_free && pmhd != nullptr) {
    FatalLoad("Hydro remap source but the target is MHD.");
  }

  // ---- target relativity class -------------------------------------------------------
  RemapRelClass tgt_class = RemapRelClass::kNewtonian;
  if (pmbp->pz4c != nullptr) {
    FatalLoad("Remap into a z4c target is unsupported.");
  } else if (pmbp->padm != nullptr) {
    // dyn_grmhd: pcoord->is_general_relativistic is FALSE here, so this branch MUST come
    // before the Coordinates tests below.
    tgt_class = RemapRelClass::kDynGRAnalytic;
  } else if (pmbp->pcoord->is_special_relativistic) {
    FatalLoad("Remap into a special-relativistic target is unsupported.");
  } else if (pmbp->pcoord->is_general_relativistic) {
    tgt_class = RemapRelClass::kFixedGR;
  }
  if (tgt_class != src.rel_class) {
    FatalLoad(std::string("Remap relativity-class mismatch: the source is ") +
              RemapRelClassName(src.rel_class) + " but the target is " +
              RemapRelClassName(tgt_class) + ".  Remap only transfers state between two "
              "runs of the same relativity class.");
  }

  const int target_nvars = (pmhd != nullptr) ? pmhd->nvars : phyd->nvars;
  const bool target_dual = (pmhd != nullptr) ? pmhd->use_dual_energy
                                             : phyd->use_dual_energy;
  const int target_nscalars = (pmhd != nullptr) ? pmhd->nscalars : phyd->nscalars;
  // The one hidden column after the scalars: the Newtonian dual-energy auxiliary, or the
  // gas entropy D s (MHD/Hydro::gas_entropy_idx) that every GR fluid carries.
  const int target_naux = (pmhd != nullptr) ? pmhd->naux : phyd->naux;
  const bool target_entropy =
      ((pmhd != nullptr) ? pmhd->gas_entropy_idx : phyd->gas_entropy_idx) >= 0;
  const int target_nhyd = target_nvars - target_nscalars - target_naux;
  if (target_entropy != src.gr_mode) {
    FatalLoad(std::string("Remap target ") + (target_entropy ? "carries" : "lacks") +
              " the GR gas-entropy column but the source is " +
              RemapRelClassName(src.rel_class) + ".");
  }
  // The source layout is derived from the source's OWN parameter dump, not inferred
  // from the payload byte size alone: a dual-energy aux column and an extra passive
  // scalar have the same width but different meanings.  The one exception is the GR
  // gas-entropy column, which no header key announces; it is matched by width below.
  const char *gas_block = src.source_has_mhd ? "mhd" : "hydro";
  const bool src_dual = src_pin.GetOrAddBoolean(gas_block, "dual_energy", false);
  const int src_nscalars = src_pin.GetOrAddInteger(gas_block, "nscalars", 0);
  if (src_nscalars != target_nscalars) {
    FatalLoad("Remap source has " + std::to_string(src_nscalars) +
              " passive scalars but the target has " + std::to_string(target_nscalars) +
              "; the CC layout would misalign.");
  }
  if (src.gr_mode && src_dual) {
    FatalLoad(std::string("Remap source <") + gas_block + ">/dual_energy = true: it was "
              "written with the removed GR dual-energy channel, whose auxiliary column "
              "holds D*kappa, not the gas entropy D s.");
  }
  if (src_dual && !target_dual) {
    FatalLoad("Remap source carries a dual-energy aux column but the target has "
              "<hydro/mhd> dual_energy = false.");
  }
  // the source inversion's channel threshold (default as in Hydro and MHD)
  if (src_dual) {
    src.dual_eta1 = src_pin.GetOrAddReal(gas_block, "dual_energy_eta1", 1.0e-3);
  }

  // ---- fluid EOS audit ---------------------------------------------------------------
  // The EOS fixes BOTH the gas column count and the conserved -> primitive decode, and
  // neither is recoverable from the payload bytes.  v1 derived the source width from the
  // TARGET's nhyd, so an ideal source read into an isothermal target came out exactly one
  // cell-centered array too narrow -- and the <adm> residual test below then took that
  // missing array for a stored metric section and remapped every column off by one.
  // Derive the source's nhyd from the source's own EOS name (the same dispatch hydro and
  // mhd.cpp use: only "isothermal" drops the energy column) and refuse any disagreement.
  if (!src_pin.DoesParameterExist(gas_block, "eos")) {
    FatalLoad(std::string("Remap source has no <") + gas_block + ">/eos key, so its gas "
              "column layout cannot be reconstructed from its parameter dump.");
  }
  const std::string src_eos = src_pin.GetString(gas_block, "eos");
  const bool src_is_ideal = (src_eos != "isothermal");
  const bool src_tabulated = (src_eos != "ideal" && src_eos != "isothermal");
  const int src_nhyd = src_is_ideal ? 5 : 4;
  auto gas_eos = (pmhd != nullptr) ? pmhd->peos->eos_data : phyd->peos->eos_data;
  // The remap engine floors, carries and rebuilds the energy column (IEN), which an
  // isothermal gas does not have, so isothermal sources and targets are refused.
  if (!src_is_ideal || !gas_eos.is_ideal) {
    FatalLoad("Remap requires a fluid EOS with an energy equation on both ends; "
              "isothermal sources and targets are not supported.");
  }
  if (src_nhyd != target_nhyd) {
    FatalLoad(std::string("Remap source <") + gas_block + ">/eos = '" + src_eos +
              "' stores " + std::to_string(src_nhyd) + " conserved gas columns but the "
              "target EOS has " + std::to_string(target_nhyd) + "; every column past the "
              "first would be misread.");
  }
  // The tabulated dynamical-GR gas decodes its conserved energy through its table, so the
  // remap is valid only between two ends that use the SAME closure: both dyn_eos =
  // lte_table (the table itself is audited in CheckRemapGRConsistency, which sees the
  // target's input).  The GR engine moves the conserved columns (D, S_i, tau, D s) and
  // never forms a thermal energy, so the closure enters only through the recovery the
  // target runs after the remap.
  const bool src_dyn_lte =
      src_pin.GetOrAddString(gas_block, "dyn_eos", "ideal") == "lte_table";
  if (gas_eos.dyn_lte_table != src_dyn_lte) {
    FatalLoad(std::string("Remap with <mhd>/dyn_eos = lte_table on exactly one end is not "
              "supported (source ") + (src_dyn_lte ? "lte_table" : "not lte_table") +
              ", target " + (gas_eos.dyn_lte_table ? "lte_table" : "not lte_table") +
              "): the conserved energy would be decoded against a different closure.");
  }
  if (src_is_ideal != gas_eos.is_ideal || src_tabulated != gas_eos.UsesTabulatedLTE()) {
    FatalLoad(std::string("Remap source and target disagree on the fluid EOS (source <") +
              gas_block + ">/eos = '" + src_eos + "'); the conserved variables would be "
              "decoded against a different thermodynamic closure.");
  }
  if (src_is_ideal && !src_tabulated && !src_dyn_lte) {
    // gamma enters the decode linearly (p = (gamma-1)*eint), so a 5/3-vs-4/3 mismatch is
    // a factor-of-two pressure error in every cell -- silent, and invisible in the dumps
    // because rho and the momenta are untouched.
    const Real src_gamma = src_pin.DoesParameterExist(gas_block, "gamma") ?
        src_pin.GetReal(gas_block, "gamma") : 0.0;
    const Real gamma_scale = std::max(std::abs(gas_eos.gamma), static_cast<Real>(1.0));
    if (std::abs(src_gamma - gas_eos.gamma) > 1.0e-12 * gamma_scale) {
      FatalLoad(std::string("Remap source <") + gas_block + ">/gamma = " +
                std::to_string(src_gamma) + " but the target EOS has gamma = " +
                std::to_string(gas_eos.gamma) + "; the remapped conserved energy would "
                "decode to a different pressure everywhere.");
    }
  }
  // Floors are a legitimate per-run choice (the target re-applies its own every stage),
  // so they only warn -- but they warn, because a source whose state sits on a floor the
  // target does not share will be reprocessed on the first cycle.
  const char *floor_keys[] = {"dfloor", "pfloor", "tfloor"};
  const Real target_floors[] = {gas_eos.dfloor, gas_eos.pfloor, gas_eos.tfloor};
  for (int n = 0; n < 3; ++n) {
    if (!src_pin.DoesParameterExist(gas_block, floor_keys[n])) continue;
    const Real src_floor = src_pin.GetReal(gas_block, floor_keys[n]);
    if (src_floor == target_floors[n]) continue;
    WarnRemap(std::string("<") + gas_block + ">/" + floor_keys[n] + " differs (source " +
              std::to_string(src_floor) + " vs target " +
              std::to_string(target_floors[n]) + "); the target's value is used.");
  }
  // The gas columns every source of this class stores, without the GR gas-entropy column.
  const int src_nvars_base = src_nhyd + src_nscalars + (src_dual ? 1 : 0);

  // ---- mesh binary header ------------------------------------------------------------
  IOWrapperSizeT headersize = 3*sizeof(int) + 2*sizeof(Real)
                            + sizeof(RegionSize) + 2*sizeof(RegionIndcs);
  std::vector<char> header(headersize);
  if (global_variable::my_rank == 0) {
    if (input.Read_bytes(header.data(), 1, headersize, false) != headersize) {
      FatalLoad("Failed to read restart mesh header from remap source.");
    }
  }
#if MPI_PARALLEL_ENABLED
  MPI_Bcast(header.data(), headersize, MPI_CHAR, 0, MPI_COMM_WORLD);
#endif

  IOWrapperSizeT os = 0;
  std::memcpy(&src.nmb_total, &header[os], sizeof(int));
  os += sizeof(int);
  std::memcpy(&src.root_level, &header[os], sizeof(int));
  os += sizeof(int);
  std::memcpy(&src.mesh_size, &header[os], sizeof(RegionSize));
  os += sizeof(RegionSize);
  std::memcpy(&src.mesh_indcs, &header[os], sizeof(RegionIndcs));
  os += sizeof(RegionIndcs);
  std::memcpy(&src.mb_indcs, &header[os], sizeof(RegionIndcs));
  os += sizeof(RegionIndcs);
  std::memcpy(&src.time, &header[os], sizeof(Real));
  os += sizeof(Real);
  std::memcpy(&src.dt, &header[os], sizeof(Real));
  os += sizeof(Real);
  std::memcpy(&src.ncycle, &header[os], sizeof(int));

  src.nx1 = src.mb_indcs.nx1;
  src.nx2 = (src.mb_indcs.nx2 > 1) ? src.mb_indcs.nx2 : 1;
  src.nx3 = (src.mb_indcs.nx3 > 1) ? src.mb_indcs.nx3 : 1;
  src.ngh = src.mb_indcs.ng;
  src.nout1 = src.nx1 + 2*src.ngh;
  src.nout2 = (src.mb_indcs.nx2 > 1) ? (src.nx2 + 2*src.ngh) : 1;
  src.nout3 = (src.mb_indcs.nx3 > 1) ? (src.nx3 + 2*src.ngh) : 1;
  src.nmb_rootx1 = src.mesh_indcs.nx1 / src.mb_indcs.nx1;
  src.nmb_rootx2 = (src.mesh_indcs.nx2 > 1) ? (src.mesh_indcs.nx2 / src.mb_indcs.nx2) : 1;
  src.nmb_rootx3 = (src.mesh_indcs.nx3 > 1) ? (src.mesh_indcs.nx3 / src.mb_indcs.nx3) : 1;

  // ---- logical-location + cost lists -------------------------------------------------
  std::size_t listsize = sizeof(LogicalLocation) + sizeof(float);
  std::vector<char> idlist(listsize * src.nmb_total);
  if (global_variable::my_rank == 0) {
    if (input.Read_bytes(idlist.data(), listsize, src.nmb_total, false)
        != static_cast<std::size_t>(src.nmb_total)) {
      FatalLoad("Failed to read remap-source MeshBlock location list.");
    }
  }
#if MPI_PARALLEL_ENABLED
  MPI_Bcast(idlist.data(), idlist.size(), MPI_CHAR, 0, MPI_COMM_WORLD);
#endif

  std::vector<LogicalLocation> lloc(src.nmb_total);
  os = 0;
  src.max_level = src.root_level;
  for (int n = 0; n < src.nmb_total; ++n) {
    std::memcpy(&lloc[n], &idlist[os], sizeof(LogicalLocation));
    os += sizeof(LogicalLocation);
    if (lloc[n].level > src.max_level) src.max_level = lloc[n].level;
  }

  // ---- MeshBlock refinement ages ---------------------------------------------------
  // An adaptive writer stores each block's cycles since its last refinement after the
  // step-3 state (src/outputs/restart.cpp); the remapped mesh is new, so they are skipped.
  const int nncyc_since_ref =
      src_pin.DoesParameterExist("mesh_refinement", "restart_ncyc_since_ref_count") ?
      src_pin.GetInteger("mesh_refinement", "restart_ncyc_since_ref_count") : 0;
  if (nncyc_since_ref > 0 && global_variable::my_rank == 0) {
    std::vector<int> ages(nncyc_since_ref);
    if (input.Read_bytes(ages.data(), sizeof(int), nncyc_since_ref, false)
        != static_cast<std::size_t>(nncyc_since_ref)) {
      FatalLoad("Failed to read remap-source MeshBlock refinement ages.");
    }
  }

  // ---- data_size + payload layout ----------------------------------------------------
  if (global_variable::my_rank == 0) {
    if (input.Read_bytes(&src.data_size, 1, sizeof(IOWrapperSizeT), false)
        != sizeof(IOWrapperSizeT)) {
      FatalLoad("Failed to read remap-source payload size.");
    }
    src.data_offset = input.GetPosition(false);
  }
#if MPI_PARALLEL_ENABLED
  MPI_Bcast(&src.data_size, sizeof(IOWrapperSizeT), MPI_BYTE, 0, MPI_COMM_WORLD);
  MPI_Bcast(&src.data_offset, sizeof(IOWrapperSizeT), MPI_BYTE, 0, MPI_COMM_WORLD);
#endif

  const IOWrapperSizeT ncc =
      static_cast<IOWrapperSizeT>(src.nout1) * src.nout2 * src.nout3;
  const IOWrapperSizeT fc_size =
      (static_cast<IOWrapperSizeT>(src.nout1 + 1) * src.nout2 * src.nout3 +
       static_cast<IOWrapperSizeT>(src.nout1) * (src.nout2 + 1) * src.nout3 +
       static_cast<IOWrapperSizeT>(src.nout1) * src.nout2 * (src.nout3 + 1)) *
      sizeof(Real);

  // ---- radiation (i0 intensities) section width --------------------------------------
  // nangles is a pure function of <radiation>/nlevel (geodesic_grid.cpp: 5*2*nlevel^2+2),
  // so the source's own width is reconstructable from its parameter dump alone.
  radiation::Radiation *prad = pmbp->prad;
  if (src.source_has_rad) {
    const int src_nlevel = src_pin.GetInteger("radiation", "nlevel");
    if (src_nlevel <= 0) {
      FatalLoad("Remap source <radiation>/nlevel = " + std::to_string(src_nlevel) +
                " is not a usable geodesic grid.");
    }
    src.nvars_i0 = 5*2*src_nlevel*src_nlevel + 2;
    if (prad != nullptr) {
      const bool src_rotate = PinBool(&src_pin, "radiation", "rotate_geo", true);
      const bool src_afluxes = PinBool(&src_pin, "radiation", "angular_fluxes", true);
      if (prad->prgeo == nullptr || prad->prgeo->nangles != src.nvars_i0) {
        FatalLoad("Remap source and target disagree on <radiation>/nlevel (source "
                  "nangles = " + std::to_string(src.nvars_i0) + ", target nangles = " +
                  std::to_string((prad->prgeo != nullptr) ? prad->prgeo->nangles : 0) +
                  "); the intensity layout would misalign.");
      }
      if (src_rotate != prad->rotate_geo) {
        FatalLoad("Remap source and target disagree on <radiation>/rotate_geo; the "
                  "angular grids do not correspond.");
      }
      if (src_afluxes != prad->angular_fluxes) {
        FatalLoad("Remap source and target disagree on <radiation>/angular_fluxes; the "
                  "angular grids do not correspond.");
      }
    }
  }
  const IOWrapperSizeT rad_size =
      src.source_has_rad ? ncc * src.nvars_i0 * sizeof(Real) : 0;

  // The carried primitives (outputs/restart_primitives.hpp): w0 of the writer's
  // recovery, sections after those sized by their own header keys.  They are
  // skipped: the remapped mesh is new, and its first recovery keeps no previous
  // primitives.
  auto carried_w0_count = [&src_pin](const char *block) {
    if (!src_pin.DoesParameterExist(block, "w0_count")) return 0;
    const int n = src_pin.GetInteger(block, "w0_count");
    if (n < 0 || (n > 0 && !src_pin.DoesParameterExist(block, "w0_kept"))) {
      FatalLoad(std::string("Remap source <") + block + ">/w0_count = " +
                std::to_string(n) + " without a matching w0_kept: the restart header "
                "is broken.");
    }
    return n;
  };
  const int src_nw0 = carried_w0_count("mhd_restart");
  const int src_nhw0 = carried_w0_count("hydro_restart");
  // the field-free dyn-GR fluid is the MHD object too: its primitives are <mhd_restart>'s
  const bool src_mhd_object = src.source_has_mhd || src.source_field_free;
  if ((src_nw0 > 0 && !src_mhd_object) || (src_nhw0 > 0 && src_mhd_object)) {
    FatalLoad("Remap source carries the primitives of a fluid it does not have "
              "(<mhd_restart>/w0_count = " + std::to_string(src_nw0) +
              ", <hydro_restart>/w0_count = " + std::to_string(src_nhw0) + ").");
  }

  {
    const IOWrapperSizeT cc_bytes = ncc * sizeof(Real);
    IOWrapperSizeT known_rest = 0;
    if (src.source_has_mhd) known_rest += fc_size;
    known_rest += rad_size;
    known_rest += cc_bytes * static_cast<IOWrapperSizeT>(src_nw0 + src_nhw0);
    // Anything left over can only be the <adm> u_adm section: z4c, turbulence forcing and
    // sinks were refused; the carried primitives were sized above.
    // The <adm> section is written whole or not at all: restart.cpp emits
    // padm->RestartVariableCount() arrays, which is either 0 (the analytic backends)
    // or the compile-time adm::ADM::nadm.  Requiring EXACTLY that width is what stops this
    // test from quietly absorbing a mis-sized earlier section.
    const bool src_has_adm = src_pin.DoesBlockExist("adm");
    const IOWrapperSizeT adm_bytes =
        static_cast<IOWrapperSizeT>(adm::ADM::nadm) * cc_bytes;
    // The GR gas-entropy column (D s after the scalars, MHD/Hydro::gas_entropy_idx):
    // every GR checkpoint written since the entropy fallback stores it, one written
    // before does not, and no header key says which.  The restart reader tells the two
    // apart by the payload width (ProblemGenerator, pgen.cpp), and so does this loader:
    // the column is there iff the gas width with it closes the payload, the <adm> section
    // counted where the source has one.  Both widths cannot close it (they differ by one
    // array, the <adm> section is nadm = 17).  The carried primitives, where present,
    // have the same columns and must agree (below).
    auto known_for = [&](const int nvars) {
      return cc_bytes * static_cast<IOWrapperSizeT>(nvars) + known_rest;
    };
    auto closes = [&](const int nvars) {
      const IOWrapperSizeT known = known_for(nvars);
      if (src.data_size < known) return false;
      const IOWrapperSizeT residual = src.data_size - known;
      return residual == 0 || (src_has_adm && residual == adm_bytes);
    };
    const int nvars_written = src_nvars_base + (src.gr_mode ? 1 : 0);
    int src_nvars = -1;
    if (closes(nvars_written)) {
      src_nvars = nvars_written;
    } else if (src.gr_mode && closes(src_nvars_base)) {
      src_nvars = src_nvars_base;
    }
    if (src_nvars < 0) {
      // Neither closes it: report against the layout this binary writes.
      const IOWrapperSizeT known = known_for(nvars_written);
      const std::string both = src.gr_mode ? "  The layout of a GR checkpoint written "
          "before the gas-entropy column existed (" + std::to_string(src_nvars_base) +
          " gas columns) does not match either." : "";
      if (src.data_size < known) {
        FatalLoad("Remap source payload size (" + std::to_string(src.data_size) +
                  ") is smaller than the layout derived from its parameter dump (gas "
                  "nvars " + std::to_string(nvars_written) + (src.source_has_mhd ?
                  ", plus MHD faces" : "") + (src.source_has_rad ?
                  ", plus radiation i0" : "") + " = " + std::to_string(known) + ")." +
                  both);
      }
      const IOWrapperSizeT residual = src.data_size - known;
      if (!src_has_adm) {
        FatalLoad("Remap source payload has an unexpected payload residual of " +
                  std::to_string(residual) + " bytes beyond the layout derived from its "
                  "parameter dump (" + std::to_string(known) + " of " +
                  std::to_string(src.data_size) + "); the restart carries a section this "
                  "module does not know how to skip." + both);
      }
      FatalLoad("Remap source <adm> payload residual (" + std::to_string(residual) +
                " bytes = " + std::to_string(static_cast<double>(residual)/
                static_cast<double>(cc_bytes)) + " cell-centered arrays) is not the "
                "stored ADM section this module knows how to skip (" +
                std::to_string(adm::ADM::nadm) + " arrays = " +
                std::to_string(adm_bytes) + " bytes).  A residual off by a whole "
                "number of arrays means an earlier section was sized wrong." + both);
    }
    const IOWrapperSizeT residual = src.data_size - known_for(src_nvars);
    if (residual > 0) {
      src.nadm_vars = adm::ADM::nadm;
      src.adm_skip_bytes = residual;
    }
    src.nvars = src_nvars;
    src.gas_entropy_column = src.gr_mode && (src_nvars == nvars_written);
    const int src_nw0_fluid = src_mhd_object ? src_nw0 : src_nhw0;
    if (src_nw0_fluid > 0 && src_nw0_fluid != src.nvars) {
      FatalLoad(std::string("Remap source <") + gas_block + "_restart>/w0_count = " +
                std::to_string(src_nw0_fluid) + " carried primitives, but its payload "
                "closes with " + std::to_string(src.nvars) + " conserved gas columns" +
                (src.gr_mode ? (src.gas_entropy_column ? " (the gas-entropy column "
                "included)" : " (no gas-entropy column)") : "") + ".");
    }
  }
  src.gas_cc_offset = 0;
  src.fc_offset = ncc * src.nvars * sizeof(Real);
  src.i0_offset = src.fc_offset + (src.source_has_mhd ? fc_size : 0);

  src.load_i0 = src.source_has_rad && (prad != nullptr) && opts.remap_radiation_i0;
  src.load_fc = src.source_has_mhd;  // target-is-MHD already enforced above

  // groups present on exactly one side: the source's bytes are skipped, or the target's
  // pgen initialization is left standing.  Both are reported, never silently dropped.
  src.i0_discarded = src.source_has_rad && (prad == nullptr);
  src.i0_target_only = (prad != nullptr) && !src.source_has_rad;
  if (src.i0_discarded) {
    WarnRemap("source radiation i0 discarded (the target carries no <radiation>).");
  }
  if (src.i0_target_only) {
    WarnRemap("the target carries <radiation> but the source does not; the pgen's own "
              "intensity initialization is kept (summary.i0_applied = false).");
  }
  if (src.nadm_vars > 0) {
    WarnRemap("skipping the source's stored <adm> metric (" +
              std::to_string(src.nadm_vars) + " variables); the target pgen recomputes "
              "its own metric.");
  }

  // ---- dimensionality ----------------------------------------------------------------
  // 3D on BOTH ends, whether or not there is a B field to remap.  Two reasons, each
  // sufficient: a 2D source into a 3D target is an extrusion, not a remap (v1 took it
  // because the only guard was gated on load_fc), and the CC sampler addresses source
  // blocks as (ngh+k, ngh+j, ngh+i) unconditionally -- on a 2D source, where nout3 == 1,
  // that walks a full ghost stride into the next variable's slice.
  const bool three_d_src = (src.mb_indcs.nx3 > 1) && (src.mb_indcs.nx2 > 1);
  if (!three_d_src || !pm->three_d) {
    FatalLoad(std::string("Remap requires 3D on both ends (source nx2 x nx3 = ") +
              std::to_string(src.mb_indcs.nx2) + " x " +
              std::to_string(src.mb_indcs.nx3) + " per block, target " +
              (pm->three_d ? "3D" : (pm->two_d ? "2D" : "1D")) + ").");
  }
  if (src.load_fc && src.max_level > src.root_level && !opts.b_coarsen_ok) {
    FatalLoad("MHD remap source has mesh refinement: the B field would be restricted to "
              "the source root grid (CC fields keep full detail).  Set the consumer's "
              "remap_b_coarsen_ok option to accept this.");
  }

  // ---- covering-grid setup (FC) ------------------------------------------------------
  RemapCoveringField &cov = src.cov;
  if (src.load_fc) {
    cov.n1 = src.mesh_indcs.nx1;
    cov.n2 = src.mesh_indcs.nx2;
    cov.n3 = src.mesh_indcs.nx3;
    cov.x1min = src.mesh_size.x1min;
    cov.x2min = src.mesh_size.x2min;
    cov.x3min = src.mesh_size.x3min;
    cov.x1max = src.mesh_size.x1max;
    cov.x2max = src.mesh_size.x2max;
    cov.x3max = src.mesh_size.x3max;
    cov.dx1 = (cov.x1max - cov.x1min) / static_cast<Real>(cov.n1);
    cov.dx2 = (cov.x2max - cov.x2min) / static_cast<Real>(cov.n2);
    cov.dx3 = (cov.x3max - cov.x3min) / static_cast<Real>(cov.n3);
    cov.b1f.assign(static_cast<std::size_t>(cov.n3) * cov.n2 * (cov.n1 + 1), 0.0);
    cov.b2f.assign(static_cast<std::size_t>(cov.n3) * (cov.n2 + 1) * cov.n1, 0.0);
    cov.b3f.assign(static_cast<std::size_t>(cov.n3 + 1) * cov.n2 * cov.n1, 0.0);
    cov.b1w.assign(cov.b1f.size(), 0.0);
    cov.b2w.assign(cov.b2f.size(), 0.0);
    cov.b3w.assign(cov.b3f.size(), 0.0);
  }

  // ---- un-densitization metric (GR, prescribed backends only) ------------------------
  BuildRemapMetricSampler(pmbp, src);

  // ---- per-block reads ---------------------------------------------------------------
  {
    IOWrapperSizeT file_size = 0;
    if (global_variable::my_rank == 0) {
      std::ifstream fcheck(path, std::ios::binary | std::ios::ate);
      file_size = fcheck.good() ?
          static_cast<IOWrapperSizeT>(fcheck.tellg()) : 0;
    }
#if MPI_PARALLEL_ENABLED
    MPI_Bcast(&file_size, sizeof(IOWrapperSizeT), MPI_BYTE, 0, MPI_COMM_WORLD);
#endif
    const IOWrapperSizeT need = src.data_offset +
        static_cast<IOWrapperSizeT>(src.nmb_total) * src.data_size;
    if (file_size < need) {
      FatalLoad("Remap source file is smaller than header + nmb_total blocks ("
                + std::to_string(file_size) + " < " + std::to_string(need) +
                " bytes): truncated file, or a per-rank restart "
                "(single_file_per_rank) that holds only one rank's blocks.  "
                "Point remap/source at a single global restart.");
    }
  }

  const Real root_dx1 = (src.mesh_indcs.nx1 > 0) ?
      (src.mesh_size.x1max - src.mesh_size.x1min) /
          static_cast<Real>(src.mesh_indcs.nx1) : 0.0;
  const Real root_dx2 = (src.mesh_indcs.nx2 > 1) ?
      (src.mesh_size.x2max - src.mesh_size.x2min) /
          static_cast<Real>(src.mesh_indcs.nx2) : 0.0;
  const Real root_dx3 = (src.mesh_indcs.nx3 > 1) ?
      (src.mesh_size.x3max - src.mesh_size.x3min) /
          static_cast<Real>(src.mesh_indcs.nx3) : 0.0;
  RegionSize local_bounds =
      ExpandRegionSize(LocalPackBounds(pm),
                       static_cast<Real>(2.0) * root_dx1,
                       static_cast<Real>(2.0) * root_dx2,
                       static_cast<Real>(2.0) * root_dx3);
  // Per-target-block selection (see SelectSourceBlocksByTarget).  A rank's bounding box
  // can span the whole domain when its blocks are scattered, so the box alone would load
  // nearly the whole source on every rank; it is kept only as a cheap superset test.
  std::vector<RegionSize> bsize(src.nmb_total);
  for (int n = 0; n < src.nmb_total; ++n) {
    bsize[n] = LogicalLocationToRegionSize(src.mesh_size, src.mesh_indcs, src.mb_indcs,
                                           src.root_level, lloc[n]);
  }
  const Real root_dx3v[3] = {root_dx1, root_dx2, root_dx3};
  const std::vector<char> needed = SelectSourceBlocksByTarget(
      src, bsize, pm, opts.use_transition_band, root_dx3v);
  src.blocks.clear();
  src.block_map.clear();

  const std::size_t n_x1f = static_cast<std::size_t>(src.nout3) * src.nout2
                            * (src.nout1 + 1);
  const std::size_t n_x2f = static_cast<std::size_t>(src.nout3) * (src.nout2 + 1)
                            * src.nout1;
  const std::size_t n_x3f = static_cast<std::size_t>(src.nout3 + 1) * src.nout2
                            * src.nout1;
  std::vector<Real> x1f, x2f, x3f;
  if (src.load_fc) {
    x1f.resize(n_x1f);
    x2f.resize(n_x2f);
    x3f.resize(n_x3f);
  }

  for (int n = 0; n < src.nmb_total; ++n) {
    const RegionSize &block_size = bsize[n];
    const bool overlaps = needed[n] && RegionOverlaps(block_size, local_bounds);
    const IOWrapperSizeT block_base = src.data_offset +
                                      static_cast<IOWrapperSizeT>(n) * src.data_size;
    int kept_idx = -1;
    if (overlaps) {
      RemapSourceBlock block;
      block.lloc = lloc[n];
      block.size = block_size;
      block.hydro.resize(static_cast<std::size_t>(ncc) * src.nvars);
      if (input.Read_Reals_at(block.hydro.data(), block.hydro.size(),
                              block_base + src.gas_cc_offset, false)
          != block.hydro.size()) {
        FatalLoad("Failed to read remap-source gas block data.");
      }
      if (src.load_i0) {
        block.i0.resize(static_cast<std::size_t>(ncc) * src.nvars_i0);
        if (input.Read_Reals_at(block.i0.data(), block.i0.size(),
                                block_base + src.i0_offset, false)
            != block.i0.size()) {
          FatalLoad("Failed to read remap-source radiation i0 block data.");
        }
      }
      if (src.metric.Active()) {
        // Un-densitization divisor at every cell center of this block, ghosts included.
        // The samplers only ever read active cells, but filling the whole array costs one
        // metric evaluation per ghost cell and leaves no index that could silently divide
        // by an uninitialized zero.
        block.sqrt_gamma.resize(static_cast<std::size_t>(ncc));
        for (int k = 0; k < src.nout3; ++k) {
          const Real zc = block.size.x3min +
              (static_cast<Real>(k - src.ngh) + 0.5) * block.size.dx3;
          for (int j = 0; j < src.nout2; ++j) {
            const Real yc = block.size.x2min +
                (static_cast<Real>(j - src.ngh) + 0.5) * block.size.dx2;
            for (int i = 0; i < src.nout1; ++i) {
              const Real xc = block.size.x1min +
                  (static_cast<Real>(i - src.ngh) + 0.5) * block.size.dx1;
              const std::size_t cell =
                  (static_cast<std::size_t>(k)*src.nout2 + j)*src.nout1 + i;
              block.sqrt_gamma[cell] = src.metric.SqrtGamma(xc, yc, zc);
            }
          }
        }
      }
      kept_idx = static_cast<int>(src.blocks.size());
      src.blocks.emplace_back(std::move(block));
      RemapBlockKey key{lloc[n].lx1, lloc[n].lx2, lloc[n].lx3, lloc[n].level};
      src.block_map.emplace(key, kept_idx);
    }

    if (!src.load_fc) continue;

    // stream this block's face fields into the root covering grid
    if (input.Read_Reals_at(x1f.data(), n_x1f, block_base + src.fc_offset, false)
        != n_x1f ||
        input.Read_Reals_at(x2f.data(), n_x2f,
                            block_base + src.fc_offset + n_x1f*sizeof(Real), false)
        != n_x2f ||
        input.Read_Reals_at(x3f.data(), n_x3f,
                            block_base + src.fc_offset + (n_x1f+n_x2f)*sizeof(Real),
                            false)
        != n_x3f) {
      FatalLoad("Failed to read remap-source face-field block data.");
    }

    const int dl = lloc[n].level - src.root_level;
    const std::int64_t stride = static_cast<std::int64_t>(1) << dl;
    const Real areafrac = 1.0 / static_cast<Real>(stride * stride);
    // A face on the contributing block's own boundary is stored by BOTH abutting blocks
    // (same-level twins or a coarse/fine pair), a block-interior face by exactly one.
    // Half-weighting boundary faces makes every fully covered root face normalize with
    // total weight 1 (0.5 per side at interfaces, 0.5 single-sided at the domain
    // boundary is handled by the weight division), so root faces with MIXED
    // interior/interface coverage still get the true area-weighted mean.
    const Real bnd = 0.5;
    const std::int64_t gx0 = static_cast<std::int64_t>(lloc[n].lx1) * src.nx1;
    const std::int64_t gy0 = static_cast<std::int64_t>(lloc[n].lx2) * src.nx2;
    const std::int64_t gz0 = static_cast<std::int64_t>(lloc[n].lx3) * src.nx3;
    auto idx_x1f = [&](int k, int j, int i) {
      return (static_cast<std::size_t>(k) * src.nout2 + j) * (src.nout1 + 1) + i;
    };
    auto idx_x2f = [&](int k, int j, int i) {
      return (static_cast<std::size_t>(k) * (src.nout2 + 1) + j) * src.nout1 + i;
    };
    auto idx_x3f = [&](int k, int j, int i) {
      return (static_cast<std::size_t>(k) * src.nout2 + j) * src.nout1 + i;
    };
    // active x1-faces aligned with root planes
    for (int k = 0; k < src.nx3; ++k) {
      const std::int64_t K = (gz0 + k) >> dl;
      for (int j = 0; j < src.nx2; ++j) {
        const std::int64_t J = (gy0 + j) >> dl;
        for (int i = 0; i <= src.nx1; ++i) {
          const std::int64_t gxf = gx0 + i;
          if ((gxf & (stride - 1)) != 0) continue;
          const std::int64_t I = gxf >> dl;
          const std::size_t c = (static_cast<std::size_t>(K) * cov.n2 + J)
                                * (cov.n1 + 1) + I;
          const Real w = ((i == 0 || i == src.nx1) ? bnd : 1.0) * areafrac;
          cov.b1f[c] += w * x1f[idx_x1f(src.ngh + k, src.ngh + j, src.ngh + i)];
          cov.b1w[c] += w;
        }
      }
    }
    // active x2-faces aligned with root planes
    for (int k = 0; k < src.nx3; ++k) {
      const std::int64_t K = (gz0 + k) >> dl;
      for (int j = 0; j <= src.nx2; ++j) {
        const std::int64_t gyf = gy0 + j;
        if ((gyf & (stride - 1)) != 0) continue;
        const std::int64_t J = gyf >> dl;
        const Real w2 = ((j == 0 || j == src.nx2) ? bnd : 1.0) * areafrac;
        for (int i = 0; i < src.nx1; ++i) {
          const std::int64_t I = (gx0 + i) >> dl;
          const std::size_t c = (static_cast<std::size_t>(K) * (cov.n2 + 1) + J)
                                * cov.n1 + I;
          cov.b2f[c] += w2 * x2f[idx_x2f(src.ngh + k, src.ngh + j, src.ngh + i)];
          cov.b2w[c] += w2;
        }
      }
    }
    // active x3-faces aligned with root planes
    for (int k = 0; k <= src.nx3; ++k) {
      const std::int64_t gzf = gz0 + k;
      if ((gzf & (stride - 1)) != 0) continue;
      const std::int64_t K = gzf >> dl;
      const Real w3 = ((k == 0 || k == src.nx3) ? bnd : 1.0) * areafrac;
      for (int j = 0; j < src.nx2; ++j) {
        const std::int64_t J = (gy0 + j) >> dl;
        for (int i = 0; i < src.nx1; ++i) {
          const std::int64_t I = (gx0 + i) >> dl;
          const std::size_t c = (static_cast<std::size_t>(K) * cov.n2 + J)
                                * cov.n1 + I;
          cov.b3f[c] += w3 * x3f[idx_x3f(src.ngh + k, src.ngh + j, src.ngh + i)];
          cov.b3w[c] += w3;
        }
      }
    }

    // magnetic-energy swap: the CC engine must remap GAS total energy, so subtract the
    // source cell's face-averaged magnetic energy from the stored IEN column.
    //
    // NEWTONIAN ONLY.  In GR the mhd u0 column is the densitized conserved tau (or E),
    // not a Newtonian total energy, and there is no local "subtract 0.5*B^2" identity to
    // undo: the conserved variables are smooth functions of the analytic metric and are
    // remapped verbatim, with the O(h^2) E-vs-B inconsistency absorbed downstream by the
    // C2P inversion, FOFC and excision.  remap_fc.cpp gates the matching add-back.
    if (kept_idx >= 0 && !src.gr_mode) {
      RemapSourceBlock &block = src.blocks[kept_idx];
      block.emag.assign(static_cast<std::size_t>(src.nout3) * src.nout2 * src.nout1, 0.0);
      for (int k = 0; k < src.nx3; ++k) {
        for (int j = 0; j < src.nx2; ++j) {
          for (int i = 0; i < src.nx1; ++i) {
            const int kk = src.ngh + k, jj = src.ngh + j, ii = src.ngh + i;
            const Real bx = 0.5 * (x1f[idx_x1f(kk, jj, ii)] +
                                   x1f[idx_x1f(kk, jj, ii + 1)]);
            const Real by = 0.5 * (x2f[idx_x2f(kk, jj, ii)] +
                                   x2f[idx_x2f(kk, jj + 1, ii)]);
            const Real bz = 0.5 * (x3f[idx_x3f(kk, jj, ii)] +
                                   x3f[idx_x3f(kk + 1, jj, ii)]);
            const Real emag = 0.5 * (bx*bx + by*by + bz*bz);
            const std::size_t cell =
                (static_cast<std::size_t>(kk) * src.nout2 + jj) * src.nout1 + ii;
            const std::size_t idx =
                ((((static_cast<std::size_t>(IEN) * src.nout3 + kk) * src.nout2 + jj)
                  * src.nout1) + ii);
            block.hydro[idx] -= emag;
            block.emag[cell] = emag;
          }
        }
      }
    }
  }

  input.Close(false);

  // ---- normalize the covering faces --------------------------------------------------
  if (src.load_fc) {
    auto normalize = [&](std::vector<Real> &b, std::vector<Real> &w, const char *name) {
      for (std::size_t c = 0; c < b.size(); ++c) {
        if (!(w[c] > 0.25)) {
          FatalLoad(std::string("Remap covering grid has an uncovered ") + name +
                    " face: the source block list does not tile its own root grid.");
        }
        b[c] /= w[c];
      }
      w.clear();
      w.shrink_to_fit();
    };
    normalize(cov.b1f, cov.b1w, "x1");
    normalize(cov.b2f, cov.b2w, "x2");
    normalize(cov.b3f, cov.b3w, "x3");
  }

  return true;
}

//----------------------------------------------------------------------------------------
//! \fn CheckRemapGRConsistency
//! \brief GR-only audit of the SOURCE parameter dump against the TARGET input.
//!
//! Split into fatals (keys whose disagreement makes the remapped state physically wrong —
//! a different black-hole spin means the source's densitized conserved variables were
//! built on a different metric) and rank-0 warnings (keys the target legitimately
//! overrides, e.g. excision radii, orbit files or unit scalings).  Only keys present
//! on at least one side are examined, and the target pin is never written to.

void CheckRemapGRConsistency(ParameterInput &src_pin, ParameterInput *dst_pin,
                             const RemapSourceData &src, const MeshBlockPack *pmbp) {
  if (!src.gr_mode || dst_pin == nullptr) return;

  // ---- generic key comparators -------------------------------------------------------
  enum class KeyState { kAbsent, kSourceOnly, kTargetOnly, kBoth };
  auto key_state = [&](const char *block, const char *key) {
    const bool in_src = src_pin.DoesParameterExist(block, key);
    const bool in_dst = dst_pin->DoesParameterExist(block, key);
    if (in_src && in_dst) return KeyState::kBoth;
    if (in_src) return KeyState::kSourceOnly;
    if (in_dst) return KeyState::kTargetOnly;
    return KeyState::kAbsent;
  };
  auto one_sided_msg = [](const char *block, const char *key, const KeyState st) {
    return std::string("<") + block + ">/" + key +
           ((st == KeyState::kSourceOnly) ? " is set in the remap source but not in this "
                                            "run's input"
                                          : " is set in this run's input but not in the "
                                            "remap source");
  };
  // An audit must never be able to abort the run on its own: ParameterInput's typed
  // getters std::exit() on a value they cannot parse, so every comparison goes through
  // GetString (which always succeeds) and falls back to an exact textual comparison when
  // the value does not have the type this audit expected.
  auto raw_pair = [&](const char *block, const char *key, std::string &a,
                      std::string &b) {
    a = src_pin.GetString(block, key);
    b = dst_pin->GetString(block, key);
  };
  auto differs_msg = [](const char *block, const char *key, const std::string &a,
                        const std::string &b) {
    return std::string("<") + block + ">/" + key + " differs (source '" + a +
           "' vs target '" + b + "')";
  };
  auto as_real = [](const std::string &s, Real &out) {
    if (s.empty()) return false;
    const char *begin = s.c_str();
    char *end = nullptr;
    const double v = std::strtod(begin, &end);
    if (end == begin) return false;
    while (*end != '\0' && std::isspace(static_cast<unsigned char>(*end))) ++end;
    if (*end != '\0') return false;
    out = static_cast<Real>(v);
    return true;
  };
  auto as_bool = [](std::string s, bool &out) {
    std::transform(s.begin(), s.end(), s.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    if (s == "true" || s == "1") { out = true; return true; }
    if (s == "false" || s == "0") { out = false; return true; }
    return false;
  };
  // returns a human-readable difference description, or an empty string when the key
  // agrees (or is absent on both sides)
  auto diff_real = [&](const char *block, const char *key, const Real rtol) {
    const KeyState st = key_state(block, key);
    if (st == KeyState::kAbsent) return std::string();
    if (st != KeyState::kBoth) return one_sided_msg(block, key, st);
    std::string sa, sb;
    raw_pair(block, key, sa, sb);
    Real a = 0.0, b = 0.0;
    if (!as_real(sa, a) || !as_real(sb, b)) {
      return (sa == sb) ? std::string() : differs_msg(block, key, sa, sb);
    }
    const Real scale = std::max(std::max(std::abs(a), std::abs(b)),
                                static_cast<Real>(1.0));
    if (std::abs(a - b) <= rtol * scale) return std::string();
    return differs_msg(block, key, sa, sb);
  };
  auto diff_bool = [&](const char *block, const char *key) {
    const KeyState st = key_state(block, key);
    if (st == KeyState::kAbsent) return std::string();
    if (st != KeyState::kBoth) return one_sided_msg(block, key, st);
    std::string sa, sb;
    raw_pair(block, key, sa, sb);
    bool a = false, b = false;
    if (!as_bool(sa, a) || !as_bool(sb, b)) {
      return (sa == sb) ? std::string() : differs_msg(block, key, sa, sb);
    }
    if (a == b) return std::string();
    return differs_msg(block, key, sa, sb);
  };
  auto fatal_on = [&](const std::string &d) {
    if (d.empty()) return;
    FatalLoad("Remap spacetime mismatch: " + d +
              ".  The source's conserved state was built on a different metric and "
              "cannot be reinterpreted on this one.");
  };
  auto warn_on = [&](const std::string &d) {
    if (d.empty()) return;
    WarnRemap(d + "; the target's value is used.");
  };

  constexpr Real kSpinRelTol = 1.0e-14;
  if (src.rel_class == RemapRelClass::kFixedGR) {
    fatal_on(diff_real("coord", "a", kSpinRelTol));
    fatal_on(diff_bool("coord", "minkowski"));
    warn_on(diff_bool("coord", "excise"));
    warn_on(diff_real("coord", "dexcise", kSpinRelTol));
    warn_on(diff_real("coord", "pexcise", kSpinRelTol));
    warn_on(diff_real("coord", "flux_excise_r", kSpinRelTol));
  } else {
    // kDynGRAnalytic: the BBH pgen owns its own excision masks and rebuilds them from its
    // own input, so a disagreement is informational rather than fatal.
    warn_on(diff_real("coord", "a", kSpinRelTol));
    warn_on(diff_bool("coord", "excise"));
    warn_on(diff_real("coord", "dexcise", kSpinRelTol));
    warn_on(diff_real("coord", "texcise", kSpinRelTol));
  }
  // ---- tabulated LTE closure (dyn_eos = lte_table on both ends, checked by the loader)
  // The conserved energy of such a gas is decoded through its table in code units, so a
  // different table, composition, radiation content, hole spin or unit system would
  // reinterpret every cell: all are fatal.
  const char *src_gas = src.source_field_free || !src.source_has_mhd ? "hydro" : "mhd";
  const bool lte = src.rel_class == RemapRelClass::kDynGRAnalytic && pmbp != nullptr &&
                   pmbp->pmhd != nullptr && pmbp->pmhd->peos->eos_data.dyn_lte_table &&
                   PinString(&src_pin, src_gas, "dyn_eos", "ideal") == "lte_table";
  if (lte) {
    const std::string kb = pmbp->pmhd->KeyBlock();
    const auto &eos = pmbp->pmhd->peos->eos_data;
    auto base = [](const std::string &p) {
      const std::size_t n = p.find_last_of('/');
      return (n == std::string::npos) ? p : p.substr(n + 1);
    };
    const std::string t_src = PinString(&src_pin, src_gas, "table", "");
    const std::string t_dst = PinString(dst_pin, kb.c_str(), "table", "");
    if (base(t_src) != base(t_dst)) {
      FatalLoad("Remap EOS table mismatch: the source ran on '" + t_src +
                "' but this run uses '" + t_dst + "'.  The tau of the source decodes "
                "against its own table only.");
    }
    const std::string ty_src = PinString(&src_pin, "saha_runtime",
                                         (std::string(src_gas) + "_table_type_runtime").c_str(),
                                         "");
    const std::string ty_dst = PinString(dst_pin, "saha_runtime",
                                         (kb + "_dyn_table_type").c_str(), "");
    if (!ty_src.empty() && !ty_dst.empty() && ty_src != ty_dst) {
      FatalLoad("Remap EOS table type mismatch: source '" + ty_src + "', target '" +
                ty_dst + "'.");
    }
    struct Meta { const char *key; Real tgt; };
    const Meta meta[] = {
        {"_lte_x_runtime", eos.lte_h_mass_fraction},
        {"_lte_y_runtime", eos.lte_he_mass_fraction},
        {"_lte_prad_runtime", eos.lte_has_radiation ? 1.0 : 0.0},
        {"_lte_h2_runtime", eos.lte_has_h2 ? 1.0 : 0.0},
        {"_lte_zpe_runtime", eos.lte_zpe_subtracted ? 1.0 : 0.0}};
    for (const Meta &mt : meta) {
      const std::string key = std::string(src_gas) + mt.key;
      if (!src_pin.DoesParameterExist("saha_runtime", key.c_str())) continue;
      const Real v = src_pin.GetReal("saha_runtime", key.c_str());
      if (std::abs(v - mt.tgt) > 1.0e-12) {
        FatalLoad("Remap EOS table mismatch: source <saha_runtime>/" + key + " = " +
                  std::to_string(v) + " but the target table has " +
                  std::to_string(mt.tgt) + " (composition, radiation content, H2 or "
                  "zero-point energy differ).");
      }
    }
    fatal_on(diff_real("coord", "a", kSpinRelTol));
    fatal_on(diff_real("units", "bhmass_msun", kSpinRelTol));
    fatal_on(diff_real("units", "density_cgs", kSpinRelTol));
    fatal_on(diff_real("units", "mu", kSpinRelTol));
  }
  const char *unit_reals[] = {"bhmass_msun", "density_cgs", "mu"};
  for (const char *key : unit_reals) {
    warn_on(diff_real("units", key, kSpinRelTol));
  }
}

}  // namespace impl
}  // namespace remap
