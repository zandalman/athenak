#ifndef PGEN_TDE_AMR_HPP_
#define PGEN_TDE_AMR_HPP_
//========================================================================================
// AthenaK astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file tde_amr.hpp
//! \brief Per-cell math of the <tde_amr> refinement scheme of tde_external.
//!
//! Two rules set the target level of a cell, both for gas bound to the BH and denser than
//! the ambient cut rho_a.  A spine cell has rho >= f rho_bin, rho_bin being the densest
//! bound cell of its (ln r, phi) bin.  Rule A, the angular ladder: inside R_c = 2 a_mb
//! every such cell, and beyond it a spine cell (the tail), has a cell width of at most
//! max(r, r_p)/N_theta; inside pericentre the scale is r_p, not r.  Rule B resolves the
//! stream thickness for r_p/2 <= r < R_c: a spine cell has a vertical width of at most
//! T/N_T, T
//! being the extent normal to the orbital plane of the spine cells of its bin.  The
//! relaxed target (T_down) uses the same rules with rho_a/2, sqrt(2) R_c, the ladder over
//! sqrt(2) N_theta, the spine at f/2 and its extent over sqrt(2) N_T, so a block refines
//! at the nominal and derefines only past the relaxed target.  Two caps follow.  Inside
//! r_p/2, the sink neighbourhood, no cell asks for more than max_level - 2.  With
//! midplane_finest the finest level is reserved for cells on the orbital plane through
//! the BH -- in a block the plane crosses, or inside the measured spine layer of their
//! bin: a stream thin enough to need that level lies in that layer, so the cells above
//! and below it stop one level coarser.  A block that reaches into the sink (excision)
//! sphere is held at max_level - 2 at least.
//!
//! Optional keys split rule A and add two rules; each defaults to the behaviour above.
//! With dense_frac f_d a spine cell must also reach f_d rho_ring, rho_ring being the
//! densest bound cell of its ln r ring, so the hot envelope that fills a bin without a
//! stream is not spine.  Inside R_c the spine keeps N_theta and the other bound gas (the
//! envelope) takes N_env; beyond R_c the spine (the tail) takes N_tail and, with
//! N_T,tail, rule B over N_T,tail.  The shock rule gives N_s to bound gas inside R_c
//! whose compression number m_c = -div(v) dx/c_s exceeds m_s (the orbit model gives
//! div(v) and c_s, see below); a captured shock spans a few cells at any level, so m_c
//! does not fall with refinement and the rule needs its ladder as a cap.  The nozzle rule
//! puts the spine at max_level on the cells on the plane (as above) within d_n of the
//! nozzle point, the pericentre of the infalling stream (with nozzle_mode = thickness at
//! its uncapped level instead, rule B counting min(N_T, N_zone) cells, N_zone about
//! T_min/dx_fin, T_min the thinnest spine of the zone, so only the thinnest part reaches
//! max_level even when the finest level cannot resolve it); every other cell stops at
//! max_level - cap_levels.  The sink cap and floor
//! sit at max_level - sink_levels.  With plane_only every rule acts only on
//! the cells on the plane (as above); the stream, the ring and their shocks lie in that
//! layer, the gas above and below is the dilute envelope.  Where the plane falls on or
//! near a block face the spine layer takes both blocks, so the thin stream is never split
//! by a refinement boundary; with the mesh shifted along the normal by a third of a root
//! block the plane sits at 1/3 or 2/3 of the block height at every level and the layer
//! stays one block thick, which is cheaper.
//!
//! The passes that turn these rules into refinement flags (SetTargetFlags) read every key
//! (ReadSchemeKeys, ReadRuleKeys).  The passes are templated on the orbit model, which
//! gives the radius r of a cell from the BH, whether its gas is bound, the compression
//! number of the shock rule and the cells that place the nozzle point.  The Newtonian
//! model (NewtonianOrbit below, tde_external) uses the Euclidean r; bound when the specific
//! orbital energy in the applied potential is negative; c_s^2 = (10/9) e/rho and
//! div(v) of w0; the nozzle point from the mass-weighted eccentricity vector and
//! pericentre distance h^2/(GM (1 + e)) of the dense bound gas falling in between 2 r_p
//! and R_c, a prediction from the Kepler orbit of each cell.
//! The nozzle point lies along the mass-weighted mean of a vector (eccentricity,
//! or position) projected onto the orbital plane, at the mass-weighted mean distance.
//! The plane is that of the initial orbit.

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "athena.hpp"
#include "globals.hpp"
#include "coordinates/cell_locations.hpp"
#include "mesh/mesh.hpp"
#include "parameter_input.hpp"
#include "pgen/bh_force_pair.hpp"

#if MPI_PARALLEL_ENABLED
#include <mpi.h>
#endif

namespace tde_amr {

constexpr int kNumBinsPhi = 128;
constexpr Real kDLogR = 0.05;
// A sheet thinner than a cell spans one cell or two (when it straddles a face), so a
// measured thickness is known only to two cells: N_zone never counts fewer.
constexpr Real kSheetFloorCells = 2.0;

//----------------------------------------------------------------------------------------
//! \fn Real ZoneCells()
//! \brief N_zone of the nozzle zone for its thinnest spine extent tmin and the finest
//! width dx_fin: the largest power of two not above tmin/dx_fin, at least
//! kSheetFloorCells (0 for no spine).  The octree ladder steps by factors of two, and a
//! power of two holds N_zone through the one-cell jitter of a measured tmin, so the zone
//! does not refine and derefine on alternate checks.

inline Real ZoneCells(const Real tmin, const Real dx_fin) {
  if (!(tmin > 0.0) || !(dx_fin > 0.0)) return 0.0;
  const Real c = std::max(tmin/dx_fin, kSheetFloorCells);
  Real n = kSheetFloorCells;
  while (2.0*n <= c) n *= 2.0;
  return n;
}

struct Params {
  // Keys and derived thresholds.
  Real cells_per_radian;     // N_theta
  Real cells_per_thickness;  // N_T
  Real spine_frac;           // f
  Real rho_a;                // ambient cut: <tde_amr>/rho_a, else 100 x the floor
  Real r_c;                  // 2 a_mb
  Real r_floor;              // r_p: the ladder radius never drops below it
  Real r_sink;               // excision radius (0 without an inner boundary)
  bool midplane_finest;      // finest level only on cells on the plane (block or layer)
  bool bound_only;           // only gas bound to the BH counts (false: all gas, e.g. the
                             // star and the fresh debris before and after pericentre)
  Real band_floor;           // relaxed plane test: one finest cell along the normal
  // Optional rules (defaults reproduce the scheme without them).
  Real cells_per_radian_env;     // N_env, bound non-spine gas inside R_c (N_theta)
  Real cells_per_radian_tail;    // N_tail, spine beyond R_c (N_theta)
  Real cells_per_thickness_tail; // N_T,tail, rule B on the spine beyond R_c (0: off)
  Real dense_frac;               // f_d, spine also needs rho >= f_d rho_ring (0: off)
  Real shock_mach;               // m_s, shock rule threshold (0: off)
  Real cells_per_radian_shock;   // N_s
  int cap_levels;                // all but the nozzle stop at max_level - cap_levels (0)
  int sink_levels;               // sink cap and floor at max_level - sink_levels (2)
  Real nozzle_radius;            // d_n (0: off)
  bool nozzle_thickness;         // nozzle_mode = thickness: rule B uncapped in the zone
                                 // (false: the zone's spine goes to max_level)
  bool plane_only;          // only cells on the plane (block or layer) refine (false)
  long long block_budget;   // <mesh_refinement>/max_nmb_total: MeshBlock budget of the mesh
  // Nozzle point relative to the BH, set every check from the infalling stream.
  bool nozzle_on;
  Real nzx, nzy, nzz;
  // nozzle_mode = thickness: N_zone (ZoneCells) of the nominal and the relaxed rules, set
  // every check from the thinnest spine extent of the zone (0: none, rule B keeps N_T)
  Real nozzle_cells_up, nozzle_cells_down;
  // Logical levels and root-level cell widths (in-plane rule A, vertical rule B).
  int root_level;
  int max_level;
  Real dx_root;
  Real dv_root;
  // BH state and its softened potential.
  Real bhx, bhy, bhz;
  Real bhvx, bhvy, bhvz;
  Real gm;
  Real soft2;
  Real rg;  // G M / c^2 with the Wegg potential, 0 for the Newtonian point mass
  // Orbital plane through the BH: unit normal and an in-plane orthonormal pair.
  Real nx, ny, nz;
  Real e1x, e1y, e1z;
  Real e2x, e2y, e2z;
  // (ln r, phi) bins: ring ir holds ln(r/rbin_min) in [ir, ir+1) kDLogR, r < rbin_min in
  // ring 0.
  Real rbin_min;
  Real log_rbin_min;
  int nbins_r;
};

//----------------------------------------------------------------------------------------
//! \fn int BinIndex()
//! \brief Flat index ir*kNumBinsPhi + iphi of the bin of an offset (dx,dy,dz) at distance
//! r from the BH, phi being the azimuth in the orbital plane, so a bin is a shell sector
//! of every height.

KOKKOS_INLINE_FUNCTION
int BinIndex(const Params &p, const Real dx, const Real dy, const Real dz, const Real r) {
  int ir = 0;
  if (r > p.rbin_min) {
    ir = static_cast<int>(Kokkos::floor((Kokkos::log(r) - p.log_rbin_min)/kDLogR));
    ir = (ir < 0) ? 0 : ((ir > p.nbins_r - 1) ? (p.nbins_r - 1) : ir);
  }
  const Real x = dx*p.e1x + dy*p.e1y + dz*p.e1z;
  const Real y = dx*p.e2x + dy*p.e2y + dz*p.e2z;
  const Real phi = Kokkos::atan2(y, x) + static_cast<Real>(M_PI);  // [0, 2 pi]
  int iphi = static_cast<int>(Kokkos::floor(phi*static_cast<Real>(kNumBinsPhi)/
                                            static_cast<Real>(2.0*M_PI)));
  iphi = (iphi < 0) ? 0 : ((iphi > kNumBinsPhi - 1) ? (kNumBinsPhi - 1) : iphi);
  return ir*kNumBinsPhi + iphi;
}

//----------------------------------------------------------------------------------------
//! \fn Real OrbitalEnergy()
//! \brief Specific orbital energy relative to the BH in the applied (softened Newtonian
//! or Wegg) potential.

KOKKOS_INLINE_FUNCTION
Real OrbitalEnergy(const Params &p, const Real r, const Real vx, const Real vy,
                   const Real vz) {
  const Real dvx = vx - p.bhvx, dvy = vy - p.bhvy, dvz = vz - p.bhvz;
  return 0.5*(dvx*dvx + dvy*dvy + dvz*dvz) +
         bh_force_pair::PotentialOfRadius(Kokkos::sqrt(r*r + p.soft2), p.gm, 1.0, p.rg);
}

//----------------------------------------------------------------------------------------
//! \fn bool IsSpine()
//! \brief Spine membership of a bound cell for the density cut rho_cut and fraction frac
//! of the bin maximum rho_bin.  Shared by the extent and the target passes so both see
//! the same set.

KOKKOS_INLINE_FUNCTION
bool IsSpine(const Real rho, const Real rho_bin, const Real rho_cut, const Real frac) {
  return rho > rho_cut && rho_bin > 0.0 && rho >= frac*rho_bin;
}

//----------------------------------------------------------------------------------------
//! \fn int LevelForWidth()
//! \brief Smallest logical level whose cell width dx_root/2^(L - root_level) is at most
//! width, capped at max_level (a width of zero or NaN gives max_level).

KOKKOS_INLINE_FUNCTION
int LevelForWidth(const Params &p, const Real dx_root, const Real width) {
  int level = p.root_level;
  Real d = dx_root;
  while (level < p.max_level && !(d <= width)) {
    d *= 0.5;
    ++level;
  }
  return level;
}

//----------------------------------------------------------------------------------------
//! \fn bool IsDense()
//! \brief Spine membership with the optional ring cut: rho >= f_d rho_ring as well (no
//! ring cut when dense_frac is 0).

KOKKOS_INLINE_FUNCTION
bool IsDense(const Params &p, const Real rho, const Real rho_bin, const Real rho_ring,
             const Real relax) {
  if (!IsSpine(rho, rho_bin, p.rho_a/relax, p.spine_frac/relax)) return false;
  return !(p.dense_frac > 0.0) || rho >= (p.dense_frac/relax)*rho_ring;
}

//----------------------------------------------------------------------------------------
//! \fn int CellTarget()
//! \brief Target level of one bound cell (root_level if it matches no rule).  relax is 1
//! for T_up and 2 for T_down; thick is the spine extent of the cell's bin for the same
//! evaluation (<= 0 if the bin has no spine); m_c is the cell's compression number (0
//! when the shock rule is off); nozzle says the block is in the nozzle zone for this
//! evaluation.

KOKKOS_INLINE_FUNCTION
int CellTarget(const Params &p, const Real rho, const Real rho_bin, const Real rho_ring,
               const Real r, const Real thick, const Real m_c, const Real relax,
               const bool on_plane, const bool nozzle) {
  // With plane_only a block off the orbital plane asks for nothing; the 2:1 rule alone
  // sets its level, one coarser per block away from the refined layer.
  if (p.plane_only && !on_plane) return p.root_level;
  const Real sr = Kokkos::sqrt(relax);
  const Real cut = p.rho_a/relax;
  const bool circle = r < p.r_c*sr;
  const bool spine = IsDense(p, rho, rho_bin, rho_ring, relax);
  const Real rl = (r > p.r_floor) ? r : p.r_floor;
  int level = p.root_level;
  if (rho > cut && circle) {
    const Real n = spine ? p.cells_per_radian : p.cells_per_radian_env;
    const int la = LevelForWidth(p, p.dx_root, rl/(n*sr));
    level = (la > level) ? la : level;
  } else if (spine) {
    const int la = LevelForWidth(p, p.dx_root, rl/(p.cells_per_radian_tail*sr));
    level = (la > level) ? la : level;
  }
  if (p.shock_mach > 0.0 && rho > cut && circle && m_c > p.shock_mach/relax) {
    const int ls = LevelForWidth(p, p.dx_root, rl/(p.cells_per_radian_shock*sr));
    level = (ls > level) ? ls : level;
  }
  // No thin stream lives inside half the pericentre distance (the nozzle sits at r_p);
  // there the bins shrink to a few cells, a lone dense cell reads as a one-cell-thick
  // spine and the rule would drive the sink inflow to the finest level.
  // In the nozzle zone (nozzle_mode = thickness) rule B counts min(N_T, N_zone) cells
  // across the thickness instead of N_T, N_zone = ZoneCells(T_min, dx_fin), T_min the
  // thinnest spine of the zone (Pass 2b of SetTargetFlags).  Where the finest level
  // resolves T_min with N_T cells this is rule B itself.  Where it cannot, N_T would send
  // every stretch thinner than 2 N_T dx_fin to max_level, most of the zone; N_zone puts
  // max_level on the stretch within about a factor two of the thinnest and gives the rest
  // of the zone the cells across its thickness that the thinnest stretch has.
  int level_zone = level;  // rule A and the shock rule, uncapped
  if (spine && thick > 0.0 && r*sr >= 0.5*p.r_floor) {
    const Real nt = circle ? p.cells_per_thickness : p.cells_per_thickness_tail;
    if (nt > 0.0) {
      Real nz = nt;
      const Real ncz = (relax > 1.0) ? p.nozzle_cells_down : p.nozzle_cells_up;
      if (nozzle && p.nozzle_thickness && ncz > 0.0) nz = Kokkos::fmin(nt, ncz);
      const int lb = LevelForWidth(p, p.dv_root, thick/(nt*sr));
      const int lz = LevelForWidth(p, p.dv_root, thick/(nz*sr));
      level = (lb > level) ? lb : level;
      level_zone = (lz > level_zone) ? lz : level_zone;
    }
  }
  const int sink_cap = (p.max_level - p.sink_levels > p.root_level) ?
                       p.max_level - p.sink_levels : p.root_level;
  if (r*sr < 0.5*p.r_floor) return (level > sink_cap) ? sink_cap : level;
  // In the nozzle zone the spine either goes to max_level, or (nozzle_mode = thickness)
  // takes its uncapped level with rule B over N_zone (above), never coarser than outside
  // the zone, so the finest cells appear only on the thinnest stretch of the stream.
  if (nozzle && spine && !p.nozzle_thickness) return p.max_level;
  if (p.cap_levels > 0 && level > p.max_level - p.cap_levels) {
    level = (p.max_level - p.cap_levels > p.root_level) ? p.max_level - p.cap_levels
                                                         : p.root_level;
  }
  if (p.midplane_finest && !on_plane && level > p.max_level - 1) level = p.max_level - 1;
  if (nozzle && spine) return (level_zone > level) ? level_zone : level;
  return level;
}

//----------------------------------------------------------------------------------------
//! \fn bool BlockNearPoint()
//! \brief Whether the block (extent relative to the BH) comes within d of the point
//! (px,py,pz), by its nearest point.

KOKKOS_INLINE_FUNCTION
bool BlockNearPoint(const Real x0, const Real x1, const Real y0, const Real y1,
                    const Real z0, const Real z1, const Real px, const Real py,
                    const Real pz, const Real d) {
  const Real cx = (px < x0) ? x0 - px : ((px > x1) ? px - x1 : 0.0);
  const Real cy = (py < y0) ? y0 - py : ((py > y1) ? py - y1 : 0.0);
  const Real cz = (pz < z0) ? z0 - pz : ((pz > z1) ? pz - z1 : 0.0);
  return cx*cx + cy*cy + cz*cz <= d*d;
}

//----------------------------------------------------------------------------------------
//! \fn bool BlockOnPlane()
//! \brief Whether the orbital plane through the BH crosses a block, given by its extent
//! relative to the BH; band widens the plane on both sides (the relaxed test).  The
//! signed distance to the plane is linear, so its range over the block is spanned by the
//! corners.

KOKKOS_INLINE_FUNCTION
bool BlockOnPlane(const Params &p, const Real x0, const Real x1, const Real y0,
                  const Real y1, const Real z0, const Real z1, const Real band) {
  const Real hlo = Kokkos::fmin(p.nx*x0, p.nx*x1) + Kokkos::fmin(p.ny*y0, p.ny*y1) +
                   Kokkos::fmin(p.nz*z0, p.nz*z1);
  const Real hhi = Kokkos::fmax(p.nx*x0, p.nx*x1) + Kokkos::fmax(p.ny*y0, p.ny*y1) +
                   Kokkos::fmax(p.nz*z0, p.nz*z1);
  return hlo <= band && hhi >= -band;
}

//----------------------------------------------------------------------------------------
//! \struct TargetPair
//! \brief Per-block result: the max over the cells of T_up and of T_down, reduced in one
//! team pass (the join of this "sum" is the componentwise max).

struct TargetPair {
  int up;
  int down;
  int zone;   // a cell of the block lies in the nominal nozzle zone (1) or none does (0)
  KOKKOS_INLINE_FUNCTION TargetPair() : up(-1), down(-1), zone(0) {}
  KOKKOS_INLINE_FUNCTION
  TargetPair &operator+=(const TargetPair &src) {
    up = (src.up > up) ? src.up : up;
    down = (src.down > down) ? src.down : down;
    zone = (src.zone > zone) ? src.zone : zone;
    return *this;
  }
};

//----------------------------------------------------------------------------------------
//! \struct WeightedSums
//! \brief A weight and four weighted quantities summed over cells in one reduction: the
//! nozzle sums (mass, mass times a vector, mass times a distance).

struct WeightedSums {
  static constexpr int kN = 5;
  Real v[kN];
  KOKKOS_INLINE_FUNCTION WeightedSums() {
    for (int n = 0; n < kN; ++n) v[n] = 0.0;
  }
  KOKKOS_INLINE_FUNCTION
  WeightedSums &operator+=(const WeightedSums &src) {
    for (int n = 0; n < kN; ++n) v[n] += src.v[n];
    return *this;
  }
};

//----------------------------------------------------------------------------------------
//! \struct ZoneThinnest
//! \brief The thinnest spine extent of the nozzle zone, nominal and relaxed, reduced in
//! one pass (the join of this "sum" is the componentwise min).

struct ZoneThinnest {
  Real up;
  Real down;
  KOKKOS_INLINE_FUNCTION ZoneThinnest()
      : up(Kokkos::Experimental::finite_max_v<Real>),
        down(Kokkos::Experimental::finite_max_v<Real>) {}
  KOKKOS_INLINE_FUNCTION
  ZoneThinnest &operator+=(const ZoneThinnest &src) {
    up = (src.up < up) ? src.up : up;
    down = (src.down < down) ? src.down : down;
    return *this;
  }
};

}  // namespace tde_amr

namespace Kokkos {
template <>
struct reduction_identity<tde_amr::ZoneThinnest> {
  KOKKOS_FORCEINLINE_FUNCTION static tde_amr::ZoneThinnest sum() {
    return tde_amr::ZoneThinnest();
  }
};
template <>
struct reduction_identity<tde_amr::TargetPair> {
  KOKKOS_FORCEINLINE_FUNCTION static tde_amr::TargetPair sum() {
    return tde_amr::TargetPair();
  }
};
template <>
struct reduction_identity<tde_amr::WeightedSums> {
  KOKKOS_FORCEINLINE_FUNCTION static tde_amr::WeightedSums sum() {
    return tde_amr::WeightedSums();
  }
};
}  // namespace Kokkos

namespace tde_amr {

//----------------------------------------------------------------------------------------
//! \struct Workspace
//! \brief Bin tables and per-block targets, grown on demand and reused by every check.
//! Freed by a Kokkos finalize hook, since a namespace-scope View would outlive Kokkos.

struct Workspace {
  DvceArray1D<Real> rho_bin_d;  // densest bound cell of each bin
  HostArray1D<Real> rho_bin_h;
  DvceArray1D<Real> ext_d;      // spine top and negated bottom, nominal then relaxed
  HostArray1D<Real> ext_h;
  DvceArray1D<int> target_d;    // T_up, T_down of each local block
  HostArray1D<int> target_h;
  DvceArray1D<Real> rho_ring_d; // densest bound cell of each ln r ring
  HostArray1D<Real> rho_ring_h;
  int ncheck = 0;               // checks so far, to print the nozzle point now and
                                // then
  double hold_radius = std::numeric_limits<double>::infinity();  // LimitToBudget's
                                // eviction hold, mirrored in the checkpointed pin
  ParameterInput *pin = nullptr;  // the run's input, whose <tde_amr_state> block carries
                                // the hold into the restart file
};

inline Workspace &SharedWorkspace() {
  static std::unique_ptr<Workspace> ws;
  if (ws == nullptr) {
    ws = std::make_unique<Workspace>();
    Kokkos::push_finalize_hook([]() { ws.reset(); });
  }
  return *ws;
}

//! Binds the workspace to the run's input and, on a restart, resumes the hold the
//! checkpoint carries in <tde_amr_state>/hold_radius (absent, or 0 once released: none).
inline void InitHold(ParameterInput *pin, const bool restart) {
  Workspace &ws = SharedWorkspace();
  ws = Workspace();  // a run starts clean, whatever an earlier pass of this process left
  ws.pin = pin;
  if (restart && pin->DoesParameterExist("tde_amr_state", "hold_radius")) {
    const double r = std::stod(pin->GetString("tde_amr_state", "hold_radius"));
    if (r > 0.0) ws.hold_radius = r;
  } else if (pin->DoesParameterExist("tde_amr_state", "hold_radius")) {
    // a pass started by an auto-expansion keeps the input of the previous pass, whose
    // hold would otherwise reach the next checkpoint
    pin->SetString("tde_amr_state", "hold_radius", "0");
  }
}

//! Sets the hold and mirrors it in the pin, at full double precision, 0 for none.
inline void SetHold(Workspace &ws, const double r) {
  ws.hold_radius = r;
  if (ws.pin == nullptr) return;
  std::ostringstream ss;
  ss << std::setprecision(std::numeric_limits<double>::max_digits10)
     << (std::isinf(r) ? 0.0 : r);
  ws.pin->SetString("tde_amr_state", "hold_radius", ss.str());
}

//----------------------------------------------------------------------------------------
//! \fn bool IsSchemeKey(), std::string ReadSchemeKeys()
//! \brief The six keys of the <tde_amr> block, read into p with their validated
//! defaults; ReadSchemeKeys returns an empty string, or the message of the first invalid
//! value.  It also sets every optional rule off (the values ReadRuleKeys defaults to), so
//! a caller that reads only these six keys runs the six-key scheme.  rho_a = 0 (the
//! default) leaves the ambient cut to the caller, 100 x its floor or ambient density.

inline bool IsSchemeKey(const std::string &k) {
  return k == "cells_per_radian" || k == "cells_per_thickness" || k == "spine_frac" ||
         k == "midplane_finest" || k == "bound_only" || k == "rho_a";
}

inline std::string ReadSchemeKeys(ParameterInput *pin, Params &p) {
  p.cells_per_radian = pin->GetOrAddReal("tde_amr", "cells_per_radian", 64.0);
  p.cells_per_thickness = pin->GetOrAddReal("tde_amr", "cells_per_thickness", 16.0);
  p.spine_frac = pin->GetOrAddReal("tde_amr", "spine_frac", 0.1);
  if (!(p.cells_per_radian > 0.0) || !std::isfinite(p.cells_per_radian)) {
    return "tde_amr/cells_per_radian must be > 0.";
  }
  if (!(p.cells_per_thickness > 0.0) || !std::isfinite(p.cells_per_thickness)) {
    return "tde_amr/cells_per_thickness must be > 0.";
  }
  if (!(p.spine_frac > 0.0 && p.spine_frac < 1.0)) {
    return "tde_amr/spine_frac must satisfy 0 < value < 1.";
  }
  {
    const long long per_rank = pin->DoesParameterExist("mesh_refinement",
        "max_nmb_per_rank") ? static_cast<long long>(pin->GetReal("mesh_refinement",
        "max_nmb_per_rank")) : 0LL;
    if (per_rank <= 0) {
      return "<tde_amr> needs <mesh_refinement>/max_nmb_per_rank (the per-rank memory "
             "cap) and max_nmb_total (the MeshBlock budget of the mesh).";
    }
    const std::string bad = ReadBlockBudget(pin, per_rank, p.block_budget);
    if (!bad.empty()) return bad;
  }
  p.rho_a = pin->GetOrAddReal("tde_amr", "rho_a", 0.0);
  if (!(p.rho_a >= 0.0) || !std::isfinite(p.rho_a)) {
    return "tde_amr/rho_a must be >= 0 (0: 100 x the floor or ambient density).";
  }
  p.midplane_finest = pin->GetOrAddBoolean("tde_amr", "midplane_finest", true);
  p.bound_only = pin->GetOrAddBoolean("tde_amr", "bound_only", true);
  p.plane_only = false;
  p.cells_per_radian_env = p.cells_per_radian;
  p.cells_per_radian_tail = p.cells_per_radian;
  p.cells_per_thickness_tail = 0.0;
  p.dense_frac = 0.0;
  p.shock_mach = 0.0;
  p.cells_per_radian_shock = p.cells_per_radian;
  p.cap_levels = 0;
  p.sink_levels = 2;
  p.nozzle_radius = 0.0;
  p.nozzle_thickness = false;
  p.nozzle_on = false;
  p.nzx = p.nzy = p.nzz = 0.0;
  p.nozzle_cells_up = p.nozzle_cells_down = 0.0;
  return "";
}

//----------------------------------------------------------------------------------------
//! \fn RuleKeys(), bool IsRuleKey(), std::string ReadRuleKeys()
//! \brief The optional rule keys of the <tde_amr> block, read into p after
//! ReadSchemeKeys, each defaulting to the off value ReadSchemeKeys left in p;
//! ReadRuleKeys returns an empty string, or the message of the first invalid value.

inline const std::vector<std::string> &RuleKeys() {
  static const std::vector<std::string> keys = {
      "cells_per_radian_envelope", "cells_per_radian_tail", "cells_per_thickness_tail",
      "dense_frac", "shock_mach", "cells_per_radian_shock", "cap_levels", "sink_levels",
      "nozzle_radius", "plane_only", "nozzle_mode"};
  return keys;
}

inline bool IsRuleKey(const std::string &k) {
  const auto &keys = RuleKeys();
  return std::find(keys.begin(), keys.end(), k) != keys.end();
}

inline std::string ReadRuleKeys(ParameterInput *pin, Params &p) {
  p.plane_only = pin->GetOrAddBoolean("tde_amr", "plane_only", p.plane_only);
  p.cells_per_radian_env = pin->GetOrAddReal("tde_amr", "cells_per_radian_envelope",
                                             p.cells_per_radian_env);
  p.cells_per_radian_tail = pin->GetOrAddReal("tde_amr", "cells_per_radian_tail",
                                              p.cells_per_radian_tail);
  p.cells_per_thickness_tail = pin->GetOrAddReal("tde_amr", "cells_per_thickness_tail",
                                                 p.cells_per_thickness_tail);
  p.dense_frac = pin->GetOrAddReal("tde_amr", "dense_frac", p.dense_frac);
  p.shock_mach = pin->GetOrAddReal("tde_amr", "shock_mach", p.shock_mach);
  p.cells_per_radian_shock = pin->GetOrAddReal("tde_amr", "cells_per_radian_shock",
                                               p.cells_per_radian_shock);
  p.cap_levels = pin->GetOrAddInteger("tde_amr", "cap_levels", p.cap_levels);
  p.sink_levels = pin->GetOrAddInteger("tde_amr", "sink_levels", p.sink_levels);
  p.nozzle_radius = pin->GetOrAddReal("tde_amr", "nozzle_radius", p.nozzle_radius);
  const std::string nozzle_mode = pin->GetOrAddString("tde_amr", "nozzle_mode", "max");
  if (nozzle_mode != "max" && nozzle_mode != "thickness") {
    return "tde_amr/nozzle_mode must be max or thickness.";
  }
  p.nozzle_thickness = (nozzle_mode == "thickness");
  if (!(p.cells_per_radian_env > 0.0) || !(p.cells_per_radian_tail > 0.0) ||
      !(p.cells_per_radian_shock > 0.0)) {
    return "tde_amr/cells_per_radian_envelope, _tail and _shock must be > 0.";
  }
  if (p.cells_per_thickness_tail < 0.0 || !(p.dense_frac >= 0.0 && p.dense_frac < 1.0) ||
      p.shock_mach < 0.0 || p.nozzle_radius < 0.0) {
    return "tde_amr: cells_per_thickness_tail, shock_mach and nozzle_radius must be "
           ">= 0, and 0 <= dense_frac < 1.";
  }
  if (p.cap_levels < 0 || p.sink_levels < 1) {
    return "tde_amr: cap_levels must be >= 0 and sink_levels >= 1.";
  }
  return "";
}

//----------------------------------------------------------------------------------------
//! \fn void SetOrbitalPlane()
//! \brief Orbital plane through the BH.  The normal is the initial orbit's r x v (star
//! position and velocity relative to the BH), which is the rotated z axis
//! (sin theta_bh, 0, cos theta_bh) for the built-in orbits; e1 is the rotated x axis
//! projected into the plane, e2 = n x e1.

inline void SetOrbitalPlane(Params &p, const Real theta_bh, const Real sx, const Real sy,
                            const Real sz, const Real svx, const Real svy,
                            const Real svz) {
  Real nx = sy*svz - sz*svy, ny = sz*svx - sx*svz, nz = sx*svy - sy*svx;
  Real nn = std::sqrt(nx*nx + ny*ny + nz*nz);
  const Real rv = std::sqrt(sx*sx + sy*sy + sz*sz)*std::sqrt(svx*svx + svy*svy + svz*svz);
  if (!(nn > 1.0e-12*rv) || !(nn > 0.0)) {
    nx = std::sin(theta_bh);
    ny = 0.0;
    nz = std::cos(theta_bh);
    nn = 1.0;
  }
  nx /= nn; ny /= nn; nz /= nn;
  Real ax = std::cos(theta_bh), ay = 0.0, az = -std::sin(theta_bh);
  Real an = ax*nx + ay*ny + az*nz;
  ax -= an*nx; ay -= an*ny; az -= an*nz;
  Real al = std::sqrt(ax*ax + ay*ay + az*az);
  if (!(al > 1.0e-6)) {
    ax = 0.0; ay = 1.0; az = 0.0;
    an = ny;
    ax -= an*nx; ay -= an*ny; az -= an*nz;
    al = std::sqrt(ax*ax + ay*ay + az*az);
  }
  p.nx = nx; p.ny = ny; p.nz = nz;
  p.e1x = ax/al; p.e1y = ay/al; p.e1z = az/al;
  p.e2x = p.ny*p.e1z - p.nz*p.e1y;
  p.e2y = p.nz*p.e1x - p.nx*p.e1z;
  p.e2z = p.nx*p.e1y - p.ny*p.e1x;
}

//----------------------------------------------------------------------------------------
//! \fn void SetPlaneNormal()
//! \brief Turns the orbital plane of p to the unit normal (nx, ny, nz): e1 is the current
//! e1 projected into the new plane (e2 when e1 is nearly along the new normal; of the two
//! orthonormal in-plane axes at most one can be), e2 = n x e1.

inline void SetPlaneNormal(Params &p, const Real nx, const Real ny, const Real nz) {
  Real ax = p.e1x, ay = p.e1y, az = p.e1z;
  Real an = ax*nx + ay*ny + az*nz;
  ax -= an*nx; ay -= an*ny; az -= an*nz;
  Real al = std::sqrt(ax*ax + ay*ay + az*az);
  if (!(al > 0.5)) {
    ax = p.e2x; ay = p.e2y; az = p.e2z;
    an = ax*nx + ay*ny + az*nz;
    ax -= an*nx; ay -= an*ny; az -= an*nz;
    al = std::sqrt(ax*ax + ay*ay + az*az);
  }
  p.nx = nx; p.ny = ny; p.nz = nz;
  p.e1x = ax/al; p.e1y = ay/al; p.e1z = az/al;
  p.e2x = p.ny*p.e1z - p.nz*p.e1y;
  p.e2y = p.nz*p.e1x - p.nx*p.e1z;
  p.e2z = p.nx*p.e1y - p.ny*p.e1x;
}

//----------------------------------------------------------------------------------------
//! \struct BannerNotes
//! \brief The words of the startup banner that depend on the orbit model.

struct BannerNotes {
  std::string rho_a;    // where rho_a comes from
  std::string shock;    // m_c and c_s of the shock rule
  std::string nozzle;   // what the nozzle point is
  std::string plane;    // appended to the orbital plane line
};

//----------------------------------------------------------------------------------------
//! \fn void PrintBanner()
//! \brief The keys and derived quantities of the scheme, printed by the caller's rank 0.
//! r_t is the tidal radius; R_c = 2 a_mb gives a_mb.

inline void PrintBanner(const Params &p, const Mesh *pm, const Real r_t,
                        const BannerNotes &t) {
  const long long budget = static_cast<long long>(
      std::floor(0.9*static_cast<double>(p.block_budget)));
  const Real a_mb = 0.5*p.r_c;
  std::cout << std::endl << "--- TDE AMR <tde_amr> ---" << std::endl
    << "cells_per_radian    = " << p.cells_per_radian
    << "  (A: dx1 <= max(r, r_p)/N_theta for bound gas with rho > rho_a inside R_c and"
    << " for spine cells beyond it; r_p = " << p.r_floor << ")" << std::endl
    << "cells_per_thickness = " << p.cells_per_thickness
    << "  (B: dx3 <= T/N_T on spine cells at r_p/2 <= r < R_c, T the spine extent of "
       "their bin)" << std::endl
    << "spine_frac          = " << p.spine_frac
    << "  (spine: rho > rho_a, rho >= f rho_bin)" << std::endl
    << "bound_only          = " << (p.bound_only ? "true (gas bound to the BH)"
                                                 : "false (all gas)") << std::endl
    << "midplane_finest     = " << (p.midplane_finest ?
           "true (finest level only on cells on the plane or in the spine layer)" :
           "false")
    << std::endl
    << "sink                = r < r_p/2 capped at max_level - " << p.sink_levels
    << "; blocks reaching r < " << p.r_sink << " held at max_level - " << p.sink_levels
    << std::endl
    << "plane_only          = " << (p.plane_only ?
           "true (only cells on the plane or in the spine layer refine; the rest 2:1)" :
           "false") << std::endl
    << "cells_per_radian_envelope = " << p.cells_per_radian_env
    << "  (bound non-spine gas inside R_c)" << std::endl
    << "cells_per_radian_tail     = " << p.cells_per_radian_tail
    << "  (spine beyond R_c);  cells_per_thickness_tail = "
    << p.cells_per_thickness_tail
    << (p.cells_per_thickness_tail > 0.0 ? "" : " (off)") << std::endl
    << "dense_frac          = " << p.dense_frac
    << (p.dense_frac > 0.0 ? "  (spine also rho >= f_d rho_ring)" : "  (off)")
    << std::endl
    << "shock_mach          = " << p.shock_mach
    << (p.shock_mach > 0.0 ? "  (" + t.shock + ", inside R_c, ladder N_s = " +
                             std::to_string(p.cells_per_radian_shock)
                           : std::string("  (off"))
    << ")" << std::endl
    << "cap_levels          = " << p.cap_levels
    << (p.cap_levels > 0 ? "  (all but the nozzle stop at max_level - cap_levels)" : "")
    << std::endl
    << "nozzle_radius       = " << p.nozzle_radius
    << (p.nozzle_radius > 0.0 ? (p.nozzle_thickness ?
           "  (nozzle_mode = thickness: spine at its uncapped level, rule B over "
           "min(N_T, N_zone ~ T_min/dx_fin), on plane blocks within d_n of " + t.nozzle +
           ")" :
           "  (nozzle_mode = max: spine at max_level on plane blocks within d_n of " +
           t.nozzle + ")") : std::string("  (off)"))
    << std::endl
    << "rho_a               = " << p.rho_a << "  (" << t.rho_a << ")" << std::endl
    << "R_c                 = " << p.r_c << "  (2 a_mb, a_mb = r_t^2/(2 R_*) = " << a_mb
    << ", r_t = " << r_t << ")" << std::endl
    << "bins                = " << kNumBinsPhi << " in phi x d ln r = "
    << kDLogR << " from r = " << p.rbin_min << " to the farthest mesh corner"
    << std::endl
    << "orbital plane       = normal (" << p.nx << ", " << p.ny << ", " << p.nz << ")"
    << t.plane << std::endl
    << "derefinement        = rho_a/2, sqrt(2) R_c, max(r, r_p)/(sqrt(2) N_theta), "
       "spine at f/2 with T/(sqrt(2) N_T)" << std::endl
    << "block budget        = " << budget << "  (0.9 max_nmb_total)"
    << std::endl << std::endl;
}

//----------------------------------------------------------------------------------------
//! \fn void FinishParams()
//! \brief The per-check fields of p from the mesh: root cell widths and the number of
//! rings, which reach the farthest mesh corner from the BH (p.bhx, p.bhy, p.bhz).

inline void FinishParams(const Mesh *pm, Params &p) {
  const auto &ms = pm->mesh_size;
  p.dx_root = ms.dx1;
  p.dv_root = pm->three_d ? ms.dx3 : (pm->multi_d ? ms.dx2 : ms.dx1);
  // The rings reach the farthest mesh corner from the BH.
  const Real ex = std::max(std::fabs(ms.x1min - p.bhx), std::fabs(ms.x1max - p.bhx));
  const Real ey = std::max(std::fabs(ms.x2min - p.bhy), std::fabs(ms.x2max - p.bhy));
  const Real ez = std::max(std::fabs(ms.x3min - p.bhz), std::fabs(ms.x3max - p.bhz));
  const Real rbin_max = std::sqrt(ex*ex + ey*ey + ez*ez);
  p.nbins_r = (rbin_max > p.rbin_min) ?
      static_cast<int>(std::ceil(std::log(rbin_max/p.rbin_min)/kDLogR)) : 1;
  p.nbins_r = std::max(1, p.nbins_r);
}

//----------------------------------------------------------------------------------------
//! \struct NewtonianOrbit
//! \brief Orbit model of tde_external: Euclidean distance from the BH, gas bound when its
//! specific orbital energy in the applied potential is negative, and the Newtonian forms
//! of the shock and nozzle rules.  The interface the passes use:
//!   Radius(dx, dy, dz), Bound(m, k, j, i, r)
//!   CompressionNumber(m, k, j, i, rho, dx1, dx2, dx3, two_d, three_d): m_c >= 0
//!   NozzleTerm(p, m, k, j, i, dx, dy, dz, r, rho, rho_bin, rho_ring, dx1, dx2, dx3, t):
//!     whether a dense cell (IsDense, checked by the caller) places the nozzle point, and
//!     its terms (mass, mass times a vector, mass times a distance).

struct NewtonianOrbit {
  Params p;
  DvceArray5D<Real> w0;
  KOKKOS_INLINE_FUNCTION
  Real Radius(const Real dx, const Real dy, const Real dz) const {
    return Kokkos::sqrt(dx*dx + dy*dy + dz*dz);
  }
  KOKKOS_INLINE_FUNCTION
  bool Bound(const int m, const int k, const int j, const int i, const Real r) const {
    return OrbitalEnergy(p, r, w0(m, IVX, k, j, i), w0(m, IVY, k, j, i),
                         w0(m, IVZ, k, j, i)) < 0.0;
  }
  //! m_c = -div(v) dx1/c_s, c_s^2 = (10/9) e/rho; the ghost zones give the neighbours of
  //! the edge cells
  KOKKOS_INLINE_FUNCTION
  Real CompressionNumber(const int m, const int k, const int j, const int i,
                         const Real rho, const Real cdx1, const Real cdx2, const Real cdx3,
                         const bool two_d, const bool three_d) const {
    Real div = (w0(m, IVX, k, j, i+1) - w0(m, IVX, k, j, i-1))/(2.0*cdx1);
    if (two_d || three_d) {
      div += (w0(m, IVY, k, j+1, i) - w0(m, IVY, k, j-1, i))/(2.0*cdx2);
    }
    if (three_d) {
      div += (w0(m, IVZ, k+1, j, i) - w0(m, IVZ, k-1, j, i))/(2.0*cdx3);
    }
    const Real cs2 = (10.0/9.0)*w0(m, IEN, k, j, i)/rho;
    return (div < 0.0 && cs2 > 0.0) ? -div*cdx1/Kokkos::sqrt(cs2) : 0.0;
  }
  //! The dense bound gas falling in (v_r < 0) between 2 r_p and R_c, the stream that will
  //! collapse vertically at its pericentre: its eccentricity vector and pericentre
  //! distance, h = r x v, e = v x h/GM - r/|r|, r_peri = h^2/(GM (1 + e)), the velocity
  //! of w0 relative to the BH.
  KOKKOS_INLINE_FUNCTION
  bool NozzleTerm(const Params &q, const int m, const int k, const int j, const int i,
                  const Real dx, const Real dy, const Real dz, const Real r,
                  const Real rho, const Real, const Real, const Real cdx1, const Real cdx2,
                  const Real cdx3, WeightedSums &t) const {
    if (r < 2.0*q.r_floor || r > q.r_c) return false;
    const Real vx = w0(m, IVX, k, j, i) - q.bhvx;
    const Real vy = w0(m, IVY, k, j, i) - q.bhvy;
    const Real vz = w0(m, IVZ, k, j, i) - q.bhvz;
    if (!(dx*vx + dy*vy + dz*vz < 0.0)) return false;
    if (!(OrbitalEnergy(q, r, w0(m, IVX, k, j, i), w0(m, IVY, k, j, i),
                        w0(m, IVZ, k, j, i)) < 0.0)) return false;
    const Real hx = dy*vz - dz*vy, hy = dz*vx - dx*vz, hz = dx*vy - dy*vx;
    const Real ex = (vy*hz - vz*hy)/q.gm - dx/r;
    const Real ey = (vz*hx - vx*hz)/q.gm - dy/r;
    const Real ez = (vx*hy - vy*hx)/q.gm - dz/r;
    const Real e = Kokkos::sqrt(ex*ex + ey*ey + ez*ez);
    const Real rpe = (hx*hx + hy*hy + hz*hz)/(q.gm*(1.0 + e));
    const Real mass = rho*cdx1*cdx2*cdx3;
    t.v[0] = mass;
    t.v[1] = mass*ex;
    t.v[2] = mass*ey;
    t.v[3] = mass*ez;
    t.v[4] = mass*rpe;
    return true;
  }
};

//----------------------------------------------------------------------------------------
//! \fn void SetTargetFlags()
//! \brief Three device passes over the active cells build the densest bound cell of
//! every (ln r, phi) bin, the extent normal to the orbital plane of the spine of every
//! bin (nominal and relaxed) and the max over the cells of each block of T_up and
//! T_down; an MPI max makes each table global.  The ring maxima follow from the global
//! bin table, and with nozzle_radius a fourth pass (an MPI sum) finds the nozzle point.
//! The flags of the other criteria start at 0 every check, so the TDE request joins them
//! as: below T_up the block refines, between T_up and T_down it is kept (another
//! criterion may still refine it, not derefine it), above T_down it derefines unless
//! another criterion flagged it.  The flags are left modified on the host; the caller
//! syncs them to the device.  p must be complete (FinishParams), with the orbital plane
//! of this check; the check sets its nozzle point (nozzle_on, nzx, nzy, nzz), which
//! LimitToBudget reads.  rho is w0(IDN) of the fluid; the orbit model gives everything
//! else a cell needs (NewtonianOrbit lists the interface).

template <class Orbit>
void SetTargetFlags(MeshBlockPack *pmbp, Params &p, const DvceArray5D<Real> &w0,
                    const Orbit &orbit) {
  Mesh *pm = pmbp->pmesh;
  auto &size = pmbp->pmb->mb_size;
  const int nmb = pmbp->nmb_thispack;
  const int mbs = pm->gids_eachrank[global_variable::my_rank];
  auto &indcs = pm->mb_indcs;
  const int is = indcs.is, ie = indcs.ie;
  const int js = indcs.js, je = indcs.je;
  const int ks = indcs.ks, ke = indcs.ke;
  const int nx1 = indcs.nx1, nx2 = indcs.nx2, nx3 = indcs.nx3;
  const int nji = nx2*nx1;
  const int nkji = nx3*nji;
  const int nbin = p.nbins_r*kNumBinsPhi;

  auto &ws = SharedWorkspace();
  if (static_cast<int>(ws.rho_bin_d.extent(0)) < nbin) {
    ws.rho_bin_d = DvceArray1D<Real>("tde_amr_rho_bin", nbin);
    ws.rho_bin_h = Kokkos::create_mirror_view(ws.rho_bin_d);
    ws.ext_d = DvceArray1D<Real>("tde_amr_extent", 4*nbin);
    ws.ext_h = Kokkos::create_mirror_view(ws.ext_d);
    ws.rho_ring_d = DvceArray1D<Real>("tde_amr_rho_ring", p.nbins_r);
    ws.rho_ring_h = Kokkos::create_mirror_view(ws.rho_ring_d);
  }
  if (static_cast<int>(ws.target_d.extent(0)) < 3*std::max(1, nmb)) {
    // Grow-only, with headroom so a slowly growing mesh does not reallocate every check.
    ws.target_d = DvceArray1D<int>("tde_amr_block_target", 3*std::max(1, nmb + nmb/4));
    ws.target_h = Kokkos::create_mirror_view(ws.target_d);
  }
  auto w0_ = w0;
  auto rho_bin_ = ws.rho_bin_d;
  auto ext_ = ws.ext_d;
  auto target_ = ws.target_d;
  auto rho_ring_ = ws.rho_ring_d;

  // Pass 1: densest bound cell above rho_a of every bin.
  Kokkos::deep_copy(rho_bin_, static_cast<Real>(0.0));
  if (nmb > 0) {
    par_for("tde_amr_bin_max", DevExeSpace(), 0, nmb - 1, ks, ke, js, je, is, ie,
    KOKKOS_LAMBDA(int m, int k, int j, int i) {
      const Real rho = w0_(m, IDN, k, j, i);
      if (!(rho > p.rho_a)) return;
      const Real dx = CellCenterX(i-is, nx1, size.d_view(m).x1min, size.d_view(m).x1max) -
                      p.bhx;
      const Real dy = CellCenterX(j-js, nx2, size.d_view(m).x2min, size.d_view(m).x2max) -
                      p.bhy;
      const Real dz = CellCenterX(k-ks, nx3, size.d_view(m).x3min, size.d_view(m).x3max) -
                      p.bhz;
      const Real r = orbit.Radius(dx, dy, dz);
      if (p.bound_only && !orbit.Bound(m, k, j, i, r)) return;
      const int ib = BinIndex(p, dx, dy, dz, r);
      // The plain read only skips an atomic that could not raise the entry.
      if (rho > rho_bin_(ib)) Kokkos::atomic_max(&rho_bin_(ib), rho);
    });
  }
  Kokkos::deep_copy(ws.rho_bin_h, rho_bin_);
#if MPI_PARALLEL_ENABLED
  MPI_Allreduce(MPI_IN_PLACE, ws.rho_bin_h.data(), nbin, MPI_ATHENA_REAL, MPI_MAX,
                MPI_COMM_WORLD);
#endif
  Kokkos::deep_copy(rho_bin_, ws.rho_bin_h);
  // Densest bound cell of each ln r ring, from the global bin table.
  for (int ir = 0; ir < p.nbins_r; ++ir) {
    Real mx = 0.0;
    for (int iphi = 0; iphi < kNumBinsPhi; ++iphi) {
      mx = std::max(mx, ws.rho_bin_h(ir*kNumBinsPhi + iphi));
    }
    ws.rho_ring_h(ir) = mx;
  }
  Kokkos::deep_copy(rho_ring_, ws.rho_ring_h);

  // Nozzle point: the cells the orbit model picks from the dense gas (NozzleTerm) give
  // a mass-weighted vector and distance.  The vector, projected onto the orbital plane,
  // sets the direction of the point from the BH, the distance its distance; without such
  // gas the nozzle rule is off for this check.  One reduction and one MPI sum.
  p.nozzle_on = false;
  p.nzx = p.nzy = p.nzz = 0.0;
  p.nozzle_cells_up = p.nozzle_cells_down = 0.0;
  if (p.nozzle_radius > 0.0) {
    WeightedSums noz;
    if (nmb > 0) {
      Kokkos::parallel_reduce("tde_amr_nozzle_point",
      Kokkos::RangePolicy<>(DevExeSpace(), 0, nmb*nkji),
      KOKKOS_LAMBDA(const int idx, WeightedSums &acc) {
        const int m = idx/nkji;
        const int kk = (idx - m*nkji)/nji;
        const int jj = (idx - m*nkji - kk*nji)/nx1;
        const int ii = idx - m*nkji - kk*nji - jj*nx1;
        const int k = kk + ks, j = jj + js, i = ii + is;
        const Real rho = w0_(m, IDN, k, j, i);
        if (!(rho > p.rho_a)) return;
        const Real dx = CellCenterX(ii, nx1, size.d_view(m).x1min, size.d_view(m).x1max) -
                        p.bhx;
        const Real dy = CellCenterX(jj, nx2, size.d_view(m).x2min, size.d_view(m).x2max) -
                        p.bhy;
        const Real dz = CellCenterX(kk, nx3, size.d_view(m).x3min, size.d_view(m).x3max) -
                        p.bhz;
        const Real r = orbit.Radius(dx, dy, dz);
        const int ib = BinIndex(p, dx, dy, dz, r);
        const Real rb = rho_bin_(ib);
        const Real rr = rho_ring_(ib/kNumBinsPhi);
        if (!IsDense(p, rho, rb, rr, 1.0)) return;
        WeightedSums t;
        if (orbit.NozzleTerm(p, m, k, j, i, dx, dy, dz, r, rho, rb, rr, size.d_view(m).dx1,
                             size.d_view(m).dx2, size.d_view(m).dx3, t)) {
          acc += t;
        }
      }, Kokkos::Sum<WeightedSums>(noz));
    }
#if MPI_PARALLEL_ENABLED
    MPI_Allreduce(MPI_IN_PLACE, noz.v, WeightedSums::kN, MPI_ATHENA_REAL, MPI_SUM,
                  MPI_COMM_WORLD);
#endif
    const Real msum = noz.v[0];
    if (msum > 0.0) {
      Real ex = noz.v[1]/msum, ey = noz.v[2]/msum, ez = noz.v[3]/msum;
      const Real en = ex*p.nx + ey*p.ny + ez*p.nz;
      ex -= en*p.nx; ey -= en*p.ny; ez -= en*p.nz;
      const Real el = std::sqrt(ex*ex + ey*ey + ez*ez);
      if (el > 0.0) {
        const Real rn = noz.v[4]/msum;
        p.nozzle_on = true;
        p.nzx = rn*ex/el; p.nzy = rn*ey/el; p.nzz = rn*ez/el;
      }
    }
    if (global_variable::my_rank == 0 && ws.ncheck % 50 == 0) {
      std::cout << "TDE AMR nozzle: "
                << (p.nozzle_on ? "on" : "off (no infalling stream)")
                << "  point - BH = (" << p.nzx << ", " << p.nzy << ", " << p.nzz
                << ")  R = " << std::sqrt(p.nzx*p.nzx + p.nzy*p.nzy + p.nzz*p.nzz)
                << "  stream mass " << msum << std::endl;
    }
  }
  ++ws.ncheck;

  // Pass 2: highest top edge and lowest bottom edge (stored negated) along the plane
  // normal of the spine cells of every bin, at f above rho_a (entries 0, 1) and relaxed
  // at f/2 above rho_a/2 (entries 2, 3).  The edges are those of the cell projected onto
  // the normal.
  Kokkos::deep_copy(ext_, std::numeric_limits<Real>::lowest());
  if (nmb > 0) {
    par_for("tde_amr_spine_extent", DevExeSpace(), 0, nmb - 1, ks, ke, js, je, is, ie,
    KOKKOS_LAMBDA(int m, int k, int j, int i) {
      const Real rho = w0_(m, IDN, k, j, i);
      if (!(rho > p.rho_a/2.0)) return;
      const Real dx = CellCenterX(i-is, nx1, size.d_view(m).x1min, size.d_view(m).x1max) -
                      p.bhx;
      const Real dy = CellCenterX(j-js, nx2, size.d_view(m).x2min, size.d_view(m).x2max) -
                      p.bhy;
      const Real dz = CellCenterX(k-ks, nx3, size.d_view(m).x3min, size.d_view(m).x3max) -
                      p.bhz;
      const Real r = orbit.Radius(dx, dy, dz);
      if (p.bound_only && !orbit.Bound(m, k, j, i, r)) return;
      const int ib = BinIndex(p, dx, dy, dz, r);
      const Real rb = rho_bin_(ib);
      const Real h = dx*p.nx + dy*p.ny + dz*p.nz;
      const Real half = 0.5*(Kokkos::fabs(p.nx)*size.d_view(m).dx1 +
                             Kokkos::fabs(p.ny)*size.d_view(m).dx2 +
                             Kokkos::fabs(p.nz)*size.d_view(m).dx3);
      const Real top = h + half;
      const Real bottom = half - h;
      const Real rr = rho_ring_(ib/kNumBinsPhi);
      for (int s = 0; s < 2; ++s) {
        const Real relax = (s == 0) ? 1.0 : 2.0;
        if (!IsDense(p, rho, rb, rr, relax)) continue;
        const int itop = (2*s)*nbin + ib;
        const int ibot = (2*s + 1)*nbin + ib;
        if (top > ext_(itop)) Kokkos::atomic_max(&ext_(itop), top);
        if (bottom > ext_(ibot)) Kokkos::atomic_max(&ext_(ibot), bottom);
      }
    });
  }
  Kokkos::deep_copy(ws.ext_h, ext_);
#if MPI_PARALLEL_ENABLED
  MPI_Allreduce(MPI_IN_PLACE, ws.ext_h.data(), 4*nbin, MPI_ATHENA_REAL, MPI_MAX,
                MPI_COMM_WORLD);
#endif
  Kokkos::deep_copy(ext_, ws.ext_h);

  // Pass 2b (nozzle_mode = thickness): the thinnest spine extent T_min of the nozzle
  // zone, nominal and relaxed, over the cells that take rule B in the zone in Pass 3 (the
  // same zone, plane, spine and radius tests), for N_zone (ZoneCells) in CellTarget.
  // Every rank joins the MPI min (a rank without zone cells gives +inf).  One reduction
  // over the blocks near the nozzle point and one MPI min.
  const Real noz_relaxed = std::sqrt(2.0)*p.nozzle_radius;
  p.nozzle_cells_up = p.nozzle_cells_down = 0.0;
  if (p.nozzle_on && p.nozzle_thickness) {
    ZoneThinnest thin;
    if (nmb > 0) {
      Kokkos::parallel_reduce("tde_amr_nozzle_thinnest",
      Kokkos::TeamPolicy<>(DevExeSpace(), nmb, Kokkos::AUTO),
      KOKKOS_LAMBDA(TeamMember_t tmember, ZoneThinnest &tacc) {
        const int m = tmember.league_rank();
        const Real bx0 = size.d_view(m).x1min - p.bhx, bx1 = size.d_view(m).x1max - p.bhx;
        const Real by0 = size.d_view(m).x2min - p.bhy, by1 = size.d_view(m).x2max - p.bhy;
        const Real bz0 = size.d_view(m).x3min - p.bhz, bz1 = size.d_view(m).x3max - p.bhz;
        const bool near_up = BlockNearPoint(bx0, bx1, by0, by1, bz0, bz1, p.nzx, p.nzy,
                                            p.nzz, p.nozzle_radius);
        const bool near_down = BlockNearPoint(bx0, bx1, by0, by1, bz0, bz1, p.nzx, p.nzy,
                                              p.nzz, noz_relaxed);
        const bool plane_up = BlockOnPlane(p, bx0, bx1, by0, by1, bz0, bz1, 0.0);
        const bool plane_down = BlockOnPlane(p, bx0, bx1, by0, by1, bz0, bz1,
                                             p.band_floor);
        const Real cdx1 = size.d_view(m).dx1, cdx2 = size.d_view(m).dx2;
        const Real cdx3 = size.d_view(m).dx3;
        const int ncell = near_down ? nkji : 0;
        ZoneThinnest bt;
        Kokkos::parallel_reduce(Kokkos::TeamThreadRange(tmember, ncell),
        [=](const int idx, ZoneThinnest &acc) {
          const int kk = idx/nji;
          const int jj = (idx - kk*nji)/nx1;
          const int ii = idx - kk*nji - jj*nx1;
          const int k = kk + ks, j = jj + js, i = ii + is;
          const Real rho = w0_(m, IDN, k, j, i);
          if (!(rho > p.rho_a/2.0)) return;
          const Real dx = CellCenterX(ii, nx1, size.d_view(m).x1min,
                                      size.d_view(m).x1max) - p.bhx;
          const Real dy = CellCenterX(jj, nx2, size.d_view(m).x2min,
                                      size.d_view(m).x2max) - p.bhy;
          const Real dz = CellCenterX(kk, nx3, size.d_view(m).x3min,
                                      size.d_view(m).x3max) - p.bhz;
          const Real r = orbit.Radius(dx, dy, dz);
          if (p.bound_only && !orbit.Bound(m, k, j, i, r)) return;
          const int ib = BinIndex(p, dx, dy, dz, r);
          const Real rb = rho_bin_(ib);
          const Real rr = rho_ring_(ib/kNumBinsPhi);
          const Real h = dx*p.nx + dy*p.ny + dz*p.nz;
          const Real half = 0.5*(Kokkos::fabs(p.nx)*cdx1 + Kokkos::fabs(p.ny)*cdx2 +
                                 Kokkos::fabs(p.nz)*cdx3);
          ZoneThinnest v;
          const Real t_up = ext_(ib) + ext_(nbin + ib);
          if (near_up && IsDense(p, rho, rb, rr, 1.0) && r >= 0.5*p.r_floor &&
              t_up > 0.0 &&
              (plane_up || (h + half >= -ext_(nbin + ib) && h - half <= ext_(ib)))) {
            v.up = t_up;
          }
          const Real t_down = ext_(2*nbin + ib) + ext_(3*nbin + ib);
          if (IsDense(p, rho, rb, rr, 2.0) &&
              r*Kokkos::sqrt(static_cast<Real>(2.0)) >= 0.5*p.r_floor && t_down > 0.0 &&
              (plane_down ||
               (h + half >= -ext_(3*nbin + ib) && h - half <= ext_(2*nbin + ib)))) {
            v.down = t_down;
          }
          acc += v;
        }, Kokkos::Sum<ZoneThinnest>(bt));
        Kokkos::single(Kokkos::PerTeam(tmember), [&]() { tacc += bt; });
      }, Kokkos::Sum<ZoneThinnest>(thin));
    }
#if MPI_PARALLEL_ENABLED
    Real tv[2] = {thin.up, thin.down};
    MPI_Allreduce(MPI_IN_PLACE, tv, 2, MPI_ATHENA_REAL, MPI_MIN, MPI_COMM_WORLD);
    thin.up = tv[0];
    thin.down = tv[1];
#endif
    // The relaxed N_zone is never below the nominal one (the relaxed zone is wider and
    // may hold a thinner bin), so the relaxed rule never derefines what the nominal one
    // refined.
    const Real none = Kokkos::Experimental::finite_max_v<Real>;
    const Real tmin_up = (thin.up < none) ? thin.up : 0.0;
    const Real tmin_down = (thin.down < none) ? std::max(thin.down, tmin_up) : tmin_up;
    p.nozzle_cells_up = ZoneCells(tmin_up, p.band_floor);
    p.nozzle_cells_down = ZoneCells(tmin_down, p.band_floor);
    if (global_variable::my_rank == 0 && (ws.ncheck - 1) % 50 == 0) {
      std::cout << "TDE AMR nozzle: thinnest spine T_min = " << tmin_up << " ("
                << tmin_up/p.band_floor << " finest cells), cells across in the zone "
                << ((p.nozzle_cells_up > 0.0) ?
                    std::min(p.cells_per_thickness, p.nozzle_cells_up) :
                    p.cells_per_thickness) << std::endl;
    }
  }

  // Pass 3: T_up and T_down of every block, the max over its cells.
  //
  // "On the plane" is a property of the stream, not of the block.  A cell counts as on
  // the plane if the plane crosses its block or if it lies inside the measured spine
  // layer of its bin (Pass 2: from the lowest bottom to the highest top edge along the
  // normal).  Where the plane falls on (or near) a block face the thin stream straddles
  // the face, and a block-only test kept the finest level, the nozzle rule and the
  // plane_only layer on one side: a refinement boundary then ran through the middle of
  // the nozzle sheet, one or two cells thick, and split it.  With the layer test both
  // halves of the sheet take the same level, and the cells above and below the layer
  // are still held one level coarser by midplane_finest.
  const bool three_d = pm->three_d, two_d = pm->two_d;
  // plane_only skips a block that is off the plane by more than any spine layer
  Real layer_up = 0.0, layer_down = 0.0;
  for (int ib = 0; ib < nbin; ++ib) {
    layer_up = std::max(layer_up, std::max(ws.ext_h(ib), ws.ext_h(nbin + ib)));
    layer_down = std::max(layer_down,
                          std::max(ws.ext_h(2*nbin + ib), ws.ext_h(3*nbin + ib)));
  }
  if (nmb > 0) {
    Kokkos::parallel_for("tde_amr_block_targets",
    Kokkos::TeamPolicy<>(DevExeSpace(), nmb, Kokkos::AUTO),
    KOKKOS_LAMBDA(TeamMember_t tmember) {
      const int m = tmember.league_rank();
      // block extent relative to the BH: the plane test and the sink floor
      const Real bx0 = size.d_view(m).x1min - p.bhx, bx1 = size.d_view(m).x1max - p.bhx;
      const Real by0 = size.d_view(m).x2min - p.bhy, by1 = size.d_view(m).x2max - p.bhy;
      const Real bz0 = size.d_view(m).x3min - p.bhz, bz1 = size.d_view(m).x3max - p.bhz;
      const bool plane_up = BlockOnPlane(p, bx0, bx1, by0, by1, bz0, bz1, 0.0);
      const bool plane_down = BlockOnPlane(p, bx0, bx1, by0, by1, bz0, bz1,
                                           p.band_floor);
      // nozzle zone: within d_n (relaxed: sqrt(2) d_n) of the nozzle point, for the cells
      // on the plane (the block test or the spine layer, below)
      const bool near_up = p.nozzle_on &&
          BlockNearPoint(bx0, bx1, by0, by1, bz0, bz1, p.nzx, p.nzy, p.nzz,
                         p.nozzle_radius);
      const bool near_down = p.nozzle_on &&
          BlockNearPoint(bx0, bx1, by0, by1, bz0, bz1, p.nzx, p.nzy, p.nzz, noz_relaxed);
      const bool layer_block_up = BlockOnPlane(p, bx0, bx1, by0, by1, bz0, bz1, layer_up);
      const bool layer_block_down = BlockOnPlane(p, bx0, bx1, by0, by1, bz0, bz1,
                                                 layer_down);
      const Real cdx1 = size.d_view(m).dx1, cdx2 = size.d_view(m).dx2;
      const Real cdx3 = size.d_view(m).dx3;
      // With plane_only every cell of a block off the plane (both tests) asks for the
      // root level, which the empty pair below stands for.
      const int ncell = (p.plane_only && !layer_block_up && !layer_block_down) ? 0 : nkji;
      TargetPair bt;
      Kokkos::parallel_reduce(Kokkos::TeamThreadRange(tmember, ncell),
      [=](const int idx, TargetPair &acc) {
        const int kk = idx/nji;
        const int jj = (idx - kk*nji)/nx1;
        const int ii = idx - kk*nji - jj*nx1;
        const int k = kk + ks, j = jj + js, i = ii + is;
        const Real rho = w0_(m, IDN, k, j, i);
        if (!(rho > p.rho_a/2.0)) return;
        const Real dx = CellCenterX(ii, nx1, size.d_view(m).x1min, size.d_view(m).x1max) -
                        p.bhx;
        const Real dy = CellCenterX(jj, nx2, size.d_view(m).x2min, size.d_view(m).x2max) -
                        p.bhy;
        const Real dz = CellCenterX(kk, nx3, size.d_view(m).x3min, size.d_view(m).x3max) -
                        p.bhz;
        const Real r = orbit.Radius(dx, dy, dz);
        if (p.bound_only && !orbit.Bound(m, k, j, i, r)) return;
        const int ib = BinIndex(p, dx, dy, dz, r);
        const Real rb = rho_bin_(ib);
        const Real rr = rho_ring_(ib/kNumBinsPhi);
        // compression number of the shock rule, which acts inside the relaxed R_c only
        Real m_c = 0.0;
        if (p.shock_mach > 0.0 && r < p.r_c*Kokkos::sqrt(static_cast<Real>(2.0))) {
          m_c = orbit.CompressionNumber(m, k, j, i, rho, cdx1, cdx2, cdx3, two_d, three_d);
        }
        // the cell's extent along the normal against its bin's spine layer [-bottom, top]
        const Real h = dx*p.nx + dy*p.ny + dz*p.nz;
        const Real half = 0.5*(Kokkos::fabs(p.nx)*cdx1 + Kokkos::fabs(p.ny)*cdx2 +
                               Kokkos::fabs(p.nz)*cdx3);
        const bool on_up = plane_up ||
            (h + half >= -ext_(nbin + ib) && h - half <= ext_(ib));
        const bool on_down = plane_down ||
            (h + half >= -ext_(3*nbin + ib) && h - half <= ext_(2*nbin + ib));
        TargetPair v;
        v.up = CellTarget(p, rho, rb, rr, r, ext_(ib) + ext_(nbin + ib), m_c, 1.0,
                          on_up, near_up && on_up);
        v.down = CellTarget(p, rho, rb, rr, r, ext_(2*nbin + ib) + ext_(3*nbin + ib), m_c,
                            2.0, on_down, near_down && on_down);
        v.zone = (near_up && on_up) ? 1 : 0;
        acc += v;
      }, Kokkos::Sum<TargetPair>(bt));
      Kokkos::single(Kokkos::PerTeam(tmember), [&]() {
        int up = bt.up, down = bt.down;
        if (p.r_sink > 0.0) {
          // nearest point of the block to the BH
          const Real cx = (bx0 > 0.0) ? bx0 : ((bx1 < 0.0) ? bx1 : 0.0);
          const Real cy = (by0 > 0.0) ? by0 : ((by1 < 0.0) ? by1 : 0.0);
          const Real cz = (bz0 > 0.0) ? bz0 : ((bz1 < 0.0) ? bz1 : 0.0);
          if (cx*cx + cy*cy + cz*cz <= p.r_sink*p.r_sink) {
            const int floor_level = (p.max_level - p.sink_levels > p.root_level) ?
                                    p.max_level - p.sink_levels : p.root_level;
            up = (up > floor_level) ? up : floor_level;
            down = (down > floor_level) ? down : floor_level;
          }
        }
        target_(3*m) = up;
        target_(3*m + 1) = down;
        target_(3*m + 2) = (bt.zone > 0 || (near_up && plane_up)) ? 1 : 0;
      });
    });
    Kokkos::deep_copy(ws.target_h, target_);
  }

  // The relaxed rules reach every cell the nominal ones do, at least as fine, except
  // where the relaxed spine of a bin is more than sqrt(2) times thicker than the nominal
  // one; T_down >= T_up is imposed so that a block at T_up never derefines.
  auto &refine_flag = pm->pmr->refine_flag;
  refine_flag.template sync<HostMemSpace>();
  for (int m = 0; m < nmb; ++m) {
    const int gid = mbs + m;
    const int level = pm->lloc_eachmb[gid].level;
    const int t_up = std::max(p.root_level, ws.target_h(3*m));
    const int t_down = std::max(t_up, ws.target_h(3*m + 1));
    int &flag = refine_flag.h_view(gid);
    if (level < t_up) {
      flag = 1;
    } else if (level <= t_down) {
      flag = std::max(flag, 0);
    } else if (flag == 0) {
      flag = -1;
    }
  }
  refine_flag.template modify<HostMemSpace>();
}

//----------------------------------------------------------------------------------------
//! \struct EvictionCut
//! \brief the two eviction shares of LimitToBudget over the spare blocks sorted by
//! (distance, tree order): the nzone farthest (the zone share), and, of the spare blocks
//! beyond r_far_cut that the zone share did not take, the nfar farthest (the far share).
struct EvictionCut {
  bool has_zone = false, has_far = false;
  std::pair<double, int> lim_zone, lim_far;  // the nearest block of each share
  double r_far_cut = 0.0;
  long long nspare = 0;                      // blocks in both shares
  double r_nearest = 0.0;                    // distance of the nearest evicted block

  template <typename Precedes>
  bool InFarCandidates(const std::pair<double, int> &key, const Precedes &precedes) const {
    return key.first > r_far_cut && (!has_zone || precedes(key, lim_zone));
  }
  template <typename Precedes>
  bool Flags(const std::pair<double, int> &key, const Precedes &precedes) const {
    if (has_zone && !precedes(key, lim_zone)) return true;
    return has_far && InFarCandidates(key, precedes) && !precedes(key, lim_far);
  }
};

template <typename Precedes>
inline EvictionCut SelectEvictions(const std::vector<std::pair<double, int>> &sorted,
                                   const long long nzone_want, const long long nfar_want,
                                   const double r_far_cut, const Precedes &precedes) {
  EvictionCut ev;
  ev.r_far_cut = r_far_cut;
  const long long n = static_cast<long long>(sorted.size());
  const long long nz = std::min(n, std::max(0LL, nzone_want));
  if (nz > 0) {
    ev.has_zone = true;
    ev.lim_zone = sorted[n - nz];
    ev.nspare += nz;
    ev.r_nearest = ev.lim_zone.first;
  }
  std::vector<std::pair<double, int>> far;
  for (const auto &key : sorted) {
    if (ev.InFarCandidates(key, precedes)) far.push_back(key);
  }
  const long long nf = std::min(static_cast<long long>(far.size()),
                                std::max(0LL, nfar_want));
  if (nf > 0) {
    ev.has_far = true;
    ev.lim_far = far[far.size() - nf];
    ev.nspare += nf;
    ev.r_nearest = (ev.nspare == nf) ? ev.lim_far.first
                                     : std::min(ev.r_nearest, ev.lim_far.first);
  }
  return ev;
}

//----------------------------------------------------------------------------------------
//! \fn void LimitToBudget()
//! \brief Keeps the block count expected after this AMR pass, N + (nleaf-1) n_ref,
//! within 0.9 <mesh_refinement>/max_nmb_total.  Derefinements are not credited: they happen only
//! for complete sibling groups, so counting them let a pass overshoot the budget.  Only
//! refinements CheckForRefinement keeps are counted (it clears every flag of a block in
//! its refinement cooldown).  Over the budget the refinements of nozzle-zone blocks are
//! kept first, then the others nearest the BH at (bhx, bhy, bhz) first, in distance and
//! then tree order, until the estimate fits; the 10% headroom absorbs the blocks the 2:1
//! rule adds.
//! A full budget would otherwise freeze the mesh at the levels it had when the budget
//! filled, since blocks refined earlier never give way.  So when refinements are denied,
//! the farthest blocks that are above the root level, out of their cooldown, not flagged
//! to refine and at r >= R_c are flagged to derefine: 2 nleaf per denied zone refinement
//! (a complete sibling group frees nleaf - 1 blocks), and nleaf per other denied
//! refinement, those only beyond the nearest denied request that is not a zone refinement
//! (and not among the blocks the zone share took) and at most 2% of the budget per pass,
//! so the far debris gives way gradually to the gas nearer the BH.  The R_c
//! floor keeps evictions out of the circularizing gas: in a remap settle, where AMR runs
//! every cycle, few blocks are neither refining nor derefining, and the farthest ones can
//! reach in to r of a few.
//! Their rules still ask for the evicted blocks, and the room the eviction freed fitted
//! them, so they were refined again in the very next pass, which put the mesh back over
//! the budget (~1000 blocks deleted and recreated every two passes in run_nozzle).  An
//! eviction therefore also sets a hold radius, the distance of the nearest evicted
//! block: refinement requests from outside the nozzle zone at or beyond it are denied in
//! every later pass.  The radius only moves outward, by 20% per pass, while the held
//! requests together with the others would fit within 95% of the budget.  It lives in
//! the SharedWorkspace and is mirrored in the pin as <tde_amr_state>/hold_radius, which
//! the restart file stores, so a restart resumes with the hold of the checkpoint.
//! The hold is released (set to infinity) once it reaches the farthest mesh corner from
//! the BH, beyond which it denies nothing.

inline void LimitToBudget(Mesh *pm, MeshBlockPack *pmbp, const Params &p, const Real bhx,
                          const Real bhy, const Real bhz) {
  MeshRefinement *pmr = pm->pmr;
  if (pm->nmb_maxperrank <= 0) return;
  const int nranks = std::max(1, global_variable::nranks);
  const long long budget = static_cast<long long>(
      std::floor(0.9*static_cast<double>(p.block_budget)));
  const long long nleaf = pm->three_d ? 8 : (pm->two_d ? 4 : 2);
  const int nmb = pmbp->nmb_thispack;
  const int mbs = pm->gids_eachrank[global_variable::my_rank];
  auto &refine_flag = pmr->refine_flag;
  auto &size = pmbp->pmb->mb_size;
  refine_flag.template sync<HostMemSpace>();
  // r_far, the farthest mesh corner from the BH, is the longest distance a block can
  // have in the code units of the run (2 R_sun in tde_external).  Zone keys
  // are shifted below every distance, by twice r_far, so that they sort first, still in
  // distance order; a hold at or beyond r_far denies nothing and is released.
  const auto &ms = pm->mesh_size;
  const double fx = std::max(std::fabs(ms.x1min - bhx), std::fabs(ms.x1max - bhx));
  const double fy = std::max(std::fabs(ms.x2min - bhy), std::fabs(ms.x2max - bhy));
  const double fz = std::max(std::fabs(ms.x3min - bhz), std::fabs(ms.x3max - bhz));
  const double r_far = std::sqrt(fx*fx + fy*fy + fz*fz);
  const double zone_shift = 2.0*r_far;
  Workspace &ws = SharedWorkspace();
  const double r_hold = ws.hold_radius;
  long long nref = 0, nzone = 0, nheld = 0;
  std::vector<double> local;   // (key, gid) of each local refining block
  std::vector<double> spare;   // (distance, gid) of each local block that may derefine
  for (int m = 0; m < nmb; ++m) {
    const int gid = mbs + m;
    if (pmr->ncyc_since_ref(gid) < pmr->refinement_interval) continue;
    const int flag = refine_flag.h_view(gid);
    const auto &bs = size.h_view(m);
    const double bx0 = bs.x1min - bhx, bx1 = bs.x1max - bhx;
    const double by0 = bs.x2min - bhy, by1 = bs.x2max - bhy;
    const double bz0 = bs.x3min - bhz, bz1 = bs.x3max - bhz;
    const double ddx = 0.5*(bx0 + bx1), ddy = 0.5*(by0 + by1), ddz = 0.5*(bz0 + bz1);
    const double r = std::sqrt(ddx*ddx + ddy*ddy + ddz*ddz);
    if (flag > 0) {
      // the nominal nozzle zone of SetTargetFlags (a cell on the plane or in the spine
      // layer within d_n of the nozzle point), checked just before this pass
      const bool zone = p.nozzle_on && ws.target_h(3*m + 2) > 0;
      if (!zone && r >= r_hold) {
        refine_flag.h_view(gid) = 0;
        ++nheld;
        continue;
      }
      ++nref;
      if (zone) ++nzone;
      local.push_back(zone ? r - zone_shift : r);
      local.push_back(static_cast<double>(gid));
    } else if (flag == 0 && pm->lloc_eachmb[gid].level > pm->root_level &&
               r >= p.r_c) {
      spare.push_back(r);
      spare.push_back(static_cast<double>(gid));
    }
  }
  // One Allgather gives the three sums and, for the gather of the refining blocks below,
  // how many doubles every rank holds.
  std::vector<int> local_sizes(nranks, 0);
  local_sizes[global_variable::my_rank] = static_cast<int>(local.size());
#if MPI_PARALLEL_ENABLED
  {
    std::vector<long long> each(4*static_cast<std::size_t>(nranks), 0);
    const std::size_t me = static_cast<std::size_t>(global_variable::my_rank);
    each[4*me] = nref;
    each[4*me + 1] = nzone;
    each[4*me + 2] = nheld;
    each[4*me + 3] = static_cast<long long>(local.size());
    MPI_Allgather(MPI_IN_PLACE, 4, MPI_LONG_LONG, each.data(), 4, MPI_LONG_LONG,
                  MPI_COMM_WORLD);
    nref = nzone = nheld = 0;
    for (int r = 0; r < nranks; ++r) {
      nref += each[4*r];
      nzone += each[4*r + 1];
      nheld += each[4*r + 2];
      local_sizes[r] = static_cast<int>(each[4*r + 3]);
    }
  }
#endif
  if (nheld > 0) {
    refine_flag.template modify<HostMemSpace>();
    if (pm->nmb_total + (nleaf - 1)*(nref + nheld) <= 0.95*static_cast<double>(budget)) {
      SetHold(ws, (r_hold >= r_far) ? std::numeric_limits<double>::infinity()
                                    : 1.2*r_hold);
      if (global_variable::my_rank == 0) {
        std::cout << "TDE AMR budget: the " << nheld << " requests held at r >= " << r_hold
                  << " would fit; hold radius -> " << ws.hold_radius << std::endl;
      }
    }
  }
  if (nref == 0 || pm->nmb_total + (nleaf - 1)*nref <= budget) return;

  // (key, gid) pairs in key order, equal keys in the tree's depth-first order: the gids
  // of a LAT-permuted or a restarted mesh are not that order, so the ties are broken in
  // the tree's order and a restart ranks the same blocks the same way (ZOrderPrecedes)
  const LogicalLocation *lloc = pm->lloc_eachmb;
  auto precedes = [lloc](const std::pair<double, int> &a,
                         const std::pair<double, int> &b) {
    if (a.first != b.first) return a.first < b.first;
    return ZOrderPrecedes(lloc[a.second], lloc[b.second]);
  };
  // all (key, gid) pairs of one kind, from every rank, sorted; sizes are the doubles of
  // every rank when they are known already (null: gathered here)
  auto gather = [nranks, &precedes](const std::vector<double> &mine,
                                    const std::vector<int> *sizes) {
    (void) sizes;
    std::vector<double> all = mine;
#if MPI_PARALLEL_ENABLED
    const int nmine = static_cast<int>(mine.size());
    std::vector<int> recvcounts(nranks, 0), displs(nranks, 0);
    if (sizes != nullptr) {
      recvcounts = *sizes;
    } else {
      MPI_Allgather(&nmine, 1, MPI_INT, recvcounts.data(), 1, MPI_INT, MPI_COMM_WORLD);
    }
    for (int r = 1; r < nranks; ++r) displs[r] = displs[r - 1] + recvcounts[r - 1];
    all.assign(displs[nranks - 1] + recvcounts[nranks - 1], 0.0);
    MPI_Allgatherv(mine.data(), nmine, MPI_DOUBLE, all.data(), recvcounts.data(),
                   displs.data(), MPI_DOUBLE, MPI_COMM_WORLD);
#endif
    std::vector<std::pair<double, int>> pairs(all.size()/2);
    for (std::size_t n = 0; n < pairs.size(); ++n) {
      pairs[n] = {all[2*n], static_cast<int>(all[2*n + 1])};
    }
    std::sort(pairs.begin(), pairs.end(), precedes);
    return pairs;
  };
  const std::vector<std::pair<double, int>> cand = gather(local, &local_sizes);
  // Over the budget, (nleaf - 1) nref > room, so keep < nref and cand[keep] exists.
  const long long room = budget - pm->nmb_total;
  const long long keep = (room > 0) ? std::min(nref, room/(nleaf - 1)) : 0;
  const std::pair<double, int> cut = cand[keep];
  for (std::size_t n = 0; n < local.size(); n += 2) {
    const std::pair<double, int> key = {local[n], static_cast<int>(local[n + 1])};
    if (!precedes(key, cut)) refine_flag.h_view(key.second) = 0;
  }
  const long long zone_denied = std::max(0LL, nzone - keep);
  const long long far_denied = (nref - keep) - zone_denied;
  const double r_cut = (cut.first < 0.0) ? cut.first + zone_shift : cut.first;
  // The nearest denied request that is not a zone refinement: the first denied key in
  // order past the zone keys (keys are sorted, zone keys first).  The far share serves
  // those requests and takes only blocks beyond it.
  const double r_far_cut = (far_denied > 0) ? cand[std::max(keep, nzone)].first : r_cut;
  // A full mesh gives way from its far end: 2 nleaf blocks per denied zone refinement,
  // taken from the farthest spare blocks, and nleaf per other denied refinement but at
  // most 2% of the budget per pass, so the far debris coarsens gradually.  The share of
  // the other refinements is taken from the spare blocks beyond the nearest of them, and
  // not from those the zone share took, so it never makes room by coarsening nearer gas.
  const long long nzone_want = 2*nleaf*zone_denied;
  const long long nfar_want = std::min(nleaf*far_denied, budget/50);
  EvictionCut ev;
  if (nzone_want + nfar_want > 0) {
    ev = SelectEvictions(gather(spare, nullptr), nzone_want, nfar_want, r_far_cut, precedes);
    for (std::size_t n = 0; n < spare.size(); n += 2) {
      const std::pair<double, int> key = {spare[n], static_cast<int>(spare[n + 1])};
      if (ev.Flags(key, precedes)) refine_flag.h_view(key.second) = -1;
    }
    if (ev.nspare > 0) SetHold(ws, std::min(ws.hold_radius, ev.r_nearest));
  }
  const long long nspare = ev.nspare;
  const double r_spare = ev.r_nearest;
  refine_flag.template modify<HostMemSpace>();
  if (global_variable::my_rank == 0) {
    std::cout << "TDE AMR budget: denied " << (nref - keep) << " refinements beyond r = "
              << r_cut;
    if (zone_denied > 0) {
      std::cout << ", " << zone_denied << " of " << nzone << " in the nozzle zone";
    }
    if (nspare > 0) {
      std::cout << ", " << nspare << " blocks beyond r = " << r_spare
                << " flagged to derefine";
    }
    if (nheld > 0) std::cout << ", " << nheld << " held at r >= " << r_hold;
    if (ws.hold_radius < std::numeric_limits<double>::infinity()) {
      std::cout << ", hold radius " << ws.hold_radius;
    }
    std::cout << " (blocks after AMR ~ " << (pm->nmb_total + (nleaf - 1)*keep)
              << " of budget " << budget << ")" << std::endl;
  }
}

}  // namespace tde_amr

#endif  // PGEN_TDE_AMR_HPP_
