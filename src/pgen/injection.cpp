//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file injection.cpp
//! \brief Problem generator for a global TDE debris-stream injection simulation.
//!
//! Streams are injected near the self-intersection point and the collision-induced
//! outflow falls back onto the black hole.  Newtonian, adiabatic hydro in Cartesian
//! coordinates, with the black hole represented by the Tejeda & Rosswog (2013, TR13)
//! generalized Newtonian potential applied as a user source term.
//!
//! Units: G = M_BH = c = 1 (lengths in r_g, times in r_g/c).
//!
//! Current contents:
//!   - hydrostatic polytropic background atmosphere in the static TR13 potential, with
//!     Bernoulli constant <problem>/bern (default 0), normalized by <problem>/rho_atm at
//!     <problem>/r_atm
//!   - TR13 gravity source term (LAT-safe)
//!   - excision for r < <problem>/r_in, held at the atmosphere state at r_in
//!   - AMR: every <refined_regionN> is kept as a minimum level (they would otherwise be
//!     derefined under adaptive refinement) unless it sets initial_only = true, blocks
//!     anywhere with stream gas (density > <problem>/amr_rho_min and entropy
//!     P/rho^gamma < <problem>/amr_entropy_fac x the stream entropy, so the hot CIO is
//!     excluded) are refined to <problem>/amr_level, and a box around each injection cylinder (padded by
//!     <problem>/inj_pad) is kept at amr_level while the table is on now or at
//!     t + <problem>/inj_lead (look-ahead, so the site is refined before each on phase);
//!     likewise a box around the self-intersection point (padded by <problem>/si_pad)
//!   - stream injection: two time-dependent streams (inbound, outbound) read from the
//!     table <problem>/inj_file (t, state, xc, yc, x_in, y_in, vx_in, vy_in, x_out, ...,
//!     mdot_in, mdot_out; columns found by header name).  Each is a mass/momentum/energy
//!     source uniform over a cylinder (radius H, length H, axis along the stream velocity)
//!     centred on (x, y, 0), injecting mdot with the stream velocity and radiation
//!     temperature temp [K] (e = a_rad T^4; the gas is radiation dominated).  The table
//!     is interpolated linearly in t (state: the row at or before t) once per LAT window,
//!     and the streams are held fixed for that window.
//!   - LAT factor cap: blocks at the injection cylinders and their neighbours step at
//!     factor 1
//!
//! References:
//!    Tejeda & Rosswog 2013, MNRAS 433, 1930 (TR13)

#include <math.h>

#if MPI_PARALLEL_ENABLED
#include <mpi.h>
#endif

#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include "athena.hpp"
#include "globals.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "mesh/mesh_refinement.hpp"
#include "coordinates/cell_locations.hpp"
#include "outputs/outputs.hpp"
#include "eos/eos.hpp"
#include "hydro/hydro.hpp"
#include "pgen/pgen.hpp"
#include "units/units.hpp"
#include "injection.hpp"

namespace {

// Problem parameters, read in UserProblem (also on restart) and used by the source terms.
struct InjectionParams {
  Real r_in;      // inner (excision) radius in r_g; must be > 2
  Real rho_atm;   // atmosphere density at r_atm
  Real r_atm;     // normalization radius of the atmosphere
  Real bern;      // atmosphere Bernoulli constant (0 = zero-Bernoulli)
  Real rho_exc;   // excision density   (atmosphere at r_in)
  Real eint_exc;  // excision internal energy density (atmosphere at r_in)
};
InjectionParams params;

// Axis-aligned box with a (physical, root = 0) refinement level.
struct RefBox {
  Real x1min, x1max, x2min, x2max, x3min, x3max;
  int level;
};
std::vector<RefBox> smr_boxes;  // <refined_regionN> blocks, kept as minimum levels
int amr_level;                  // AMR target level for dense blocks and injection boxes
Real amr_rho_min;               // density threshold for AMR
Real amr_entropy_fac;           // entropy cut for AMR, in units of the stream entropy

// Injected streams.  Plain struct with a fixed-size array so it is captured by value.
constexpr int kNStreams = 2;    // 0 = inbound, 1 = outbound
struct Stream {
  Real x, y;      // centre of the injection cylinder (z = 0)
  Real vx, vy;    // injection velocity
  Real nx, ny;    // unit vector along the velocity (cylinder axis)
  Real h;         // stream radius = cylinder radius = cylinder length
  Real q;         // mass injection rate per unit volume, mdot / V_cyl
  Real espec;     // specific internal energy of injected gas, a_rad T^4 / rho_s
  Real mdot;      // mass injection rate
};
struct Streams {
  Stream s[kNStreams];
  bool on;        // table state for the current window
  Real xc, yc;    // self-intersection point
};
Streams streams;
Real stream_h;      // stream radius H (both streams)
Real stream_erad;   // injected internal energy density a_rad T^4 (code units)
Real dx_fine;       // cell size on amr_level, where the injection cylinders sit

// Injection boxes: one per cylinder, kept at amr_level while the table is on now or at
// t + inj_lead (so the coarse injection site is refined, one level per AMR check, before
// each on phase starts).  Each covers the cylinder's bounding sphere at t and at
// t + inj_lead, padded by inj_pad.  Downstream, the density criterion keeps the streams
// refined.
Real inj_lead;                  // look-ahead time [r_g/c]
Real inj_pad;                   // padding [r_g] in x and y
bool inj_box_on;                // boxes enforced for the current window
RefBox inj_box[kNStreams];
// Collision box: the SI point (xc, yc) at t and at t + inj_lead, padded by si_pad in x, y
// and z (|z| <= si_pad), enforced at amr_level while the table is on at t - tinj, t or
// t + inj_lead (gas injected before an off switch still collides for tinj after it).
// Keeps the stream collision away from refinement boundaries: without it the site
// derefines before the streams first arrive and density AMR rebuilds it only one level per
// check.  The endpoint test suffices because on/off phases are much longer than
// tinj + inj_lead.
Real si_pad;                    // padding [r_g]
Real tinj;                      // stream travel time injection -> SI point [r_g/c]
bool si_box_on;                 // collision box enforced for the current window
RefBox si_box;

// Injection table (host only), columns by name, rows in increasing t.
std::map<std::string, std::vector<Real>> inj_table;

// Injected mass ledger: total up to t_ledger (the current window start), carried through
// restarts as <problem>/m_inj.  Exact: the streams are constant within a window.
Real m_inj_total = 0.0;
Real t_ledger = 0.0;

// Mass removed by the excision (net: negative if the reset adds mass).  Rank-local part
// since the last LAT window end, and the global total reduced at window ends (identical on
// every rank, carried through restarts as <problem>/m_exc).
Real exc_mass_local = 0.0;
Real exc_mass_total = 0.0;

KOKKOS_INLINE_FUNCTION
bool InStream(const Stream &s, Real x, Real y, Real z) {
  const Real dx = x - s.x, dy = y - s.y;
  const Real along = dx*s.nx + dy*s.ny;
  const Real perp2 = dx*dx + dy*dy + z*z - along*along;
  return fabs(along) <= 0.5*s.h && perp2 <= s.h*s.h;
}

//----------------------------------------------------------------------------------------
//! \fn ReadInjectionTable
//! \brief Reads the whitespace-separated table: '#' lines are comments, the last of them
//! before the data holds the column names.  Every rank reads the file (it is small).

void ReadInjectionTable(const std::string &fname) {
  std::ifstream f(fname);
  if (!f) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "Cannot open injection table '" << fname << "'" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  std::vector<std::string> names;
  inj_table.clear();
  std::string line;
  while (std::getline(f, line)) {
    if (line.find_first_not_of(" \t\r") == std::string::npos) continue;
    std::istringstream ss(line);
    if (line[line.find_first_not_of(" \t")] == '#') {
      names.clear();
      std::string tok;
      ss >> tok;  // '#'
      while (ss >> tok) names.push_back(tok);
      continue;
    }
    for (const auto &nm : names) {
      Real v;
      ss >> v;
      inj_table[nm].push_back(v);
    }
  }
}

//----------------------------------------------------------------------------------------
//! \fn TableRow, TableVal, TableOn
//! \brief Table lookup: row i and weight w for time t (clamped at the table ends), a
//! column linearly interpolated, and the state from the row at or before t.

void TableRow(Real t, int &i, Real &w) {
  const std::vector<Real> &tt = inj_table.at("t");
  const int nrow = static_cast<int>(tt.size());
  i = static_cast<int>(std::upper_bound(tt.begin(), tt.end(), t) - tt.begin()) - 1;
  i = std::min(std::max(i, 0), nrow - 2);
  w = std::min(std::max((t - tt[i])/(tt[i+1] - tt[i]), 0.0), 1.0);
}

Real TableVal(const std::string &name, int i, Real w) {
  const std::vector<Real> &c = inj_table.at(name);
  return (1.0 - w)*c[i] + w*c[i+1];
}

bool TableOn(Real t) {
  int i;
  Real w;
  TableRow(t, i, w);
  return inj_table.at("state")[(t >= inj_table.at("t")[i+1]) ? i+1 : i] > 0.5;
}

//----------------------------------------------------------------------------------------
//! \fn UpdateStreams
//! \brief Sets the streams, injection state and injection boxes for a window starting at
//! t.  q = mdot/V uses the discrete volume of the amr_level cells whose centres lie in
//! the cylinder (the injection boxes keep the cylinders on that level), so exactly mdot
//! is injected; counted on the global fine lattice, so no MPI reduction is needed.

void UpdateStreams(const RegionSize &mesh_size, Real t) {
  int i;
  Real w;
  TableRow(t, i, w);
  auto val = [&](const std::string &name) { return TableVal(name, i, w); };
  streams.on = TableOn(t);
  streams.xc = val("xc");
  streams.yc = val("yc");

  const char *suffix[kNStreams] = {"_in", "_out"};
  for (int ns = 0; ns < kNStreams; ++ns) {
    const std::string sf = suffix[ns];
    Stream &s = streams.s[ns];
    s.x = val("x" + sf);
    s.y = val("y" + sf);
    s.vx = val("vx" + sf);
    s.vy = val("vy" + sf);
    s.mdot = val("mdot" + sf);
    s.h = stream_h;
    const Real v = sqrt(s.vx*s.vx + s.vy*s.vy);
    s.nx = s.vx/v;
    s.ny = s.vy/v;
    // Radiation-dominated gas: internal energy density e = a_rad T^4 (= 3P for gamma=4/3),
    // per unit mass of the stream density rho_s = mdot/(pi H^2 v).
    const Real rho_s = s.mdot/(M_PI*s.h*s.h*v);
    s.espec = stream_erad/rho_s;

    // Discrete cylinder volume on the fine lattice (cell centres x1min + (i+1/2) dx).
    const Real rad = sqrt(1.25)*s.h;
    const Real cx[3] = {s.x, s.y, 0.0};
    const Real xmin[3] = {mesh_size.x1min, mesh_size.x2min, mesh_size.x3min};
    int lo[3], hi[3];
    for (int d = 0; d < 3; ++d) {
      lo[d] = static_cast<int>(floor((cx[d] - rad - xmin[d])/dx_fine));
      hi[d] = static_cast<int>(ceil((cx[d] + rad - xmin[d])/dx_fine));
    }
    int ncell = 0;
    for (int k = lo[2]; k <= hi[2]; ++k) {
      const Real z = xmin[2] + (k + 0.5)*dx_fine;
      for (int j = lo[1]; j <= hi[1]; ++j) {
        const Real y = xmin[1] + (j + 0.5)*dx_fine;
        for (int ii = lo[0]; ii <= hi[0]; ++ii) {
          const Real x = xmin[0] + (ii + 0.5)*dx_fine;
          if (InStream(s, x, y, z)) ++ncell;
        }
      }
    }
    s.q = s.mdot/(ncell*dx_fine*dx_fine*dx_fine);
  }

  // Injection boxes: each cylinder's bounding sphere at t and at t + inj_lead, padded in
  // x and y; enforced while the table is on at t or at t + inj_lead.
  const Real t_ahead = t + inj_lead;
  inj_box_on = streams.on || TableOn(t_ahead);
  int ia;
  Real wa;
  TableRow(t_ahead, ia, wa);
  const Real rad = sqrt(1.25)*stream_h;
  for (int ns = 0; ns < kNStreams; ++ns) {
    const std::string sf = suffix[ns];
    const Stream &s = streams.s[ns];
    const Real xa = TableVal("x" + sf, ia, wa), ya = TableVal("y" + sf, ia, wa);
    RefBox &b = inj_box[ns];
    b.x1min = std::min(s.x, xa) - rad - inj_pad;
    b.x1max = std::max(s.x, xa) + rad + inj_pad;
    b.x2min = std::min(s.y, ya) - rad - inj_pad;
    b.x2max = std::max(s.y, ya) + rad + inj_pad;
    b.x3min = -rad;
    b.x3max = rad;
    b.level = amr_level;
  }
  const Real xca = TableVal("xc", ia, wa), yca = TableVal("yc", ia, wa);
  si_box = {std::min(streams.xc, xca) - si_pad, std::max(streams.xc, xca) + si_pad,
            std::min(streams.yc, yca) - si_pad, std::max(streams.yc, yca) + si_pad,
            -si_pad, si_pad, amr_level};
  si_box_on = inj_box_on || TableOn(t - tinj);
}

void InjectionSources(Mesh *pm, const Real bdt);
void CapLATFactorsAtInjection(Mesh *pm, int max_factor, int *lat_factor_eachmb);
void ExcisionFixup(MeshBlockPack *pmbp, const Real time);
void InjectionRefinement(MeshBlockPack *pmbp);
void InjectionWindowEnd(Mesh *pm);
void InjectionHistory(HistoryData *pdata, Mesh *pm);
void InjectionMetadata(ParameterInput *pin, Mesh *pm);

//----------------------------------------------------------------------------------------
//! \fn AtmosphereDensity
//! \brief Polytrope with Bernoulli constant B in the static TR13 potential.
//! Hydrostatic equilibrium with P = K rho^gamma gives enthalpy
//!   h = gamma/(gamma-1) P/rho = B - Phi_s(r),
//! so rho(r) = rho_atm [h(r)/h(r_atm)]^n with n = 1/(gamma-1).  B = 0 is the
//! zero-Bernoulli profile (rho ~ r^-n); B > 0 flattens it to ~constant where |Phi_s| << B.
//! The profile is frozen at r_in for r < r_in (that region will be excised).

KOKKOS_INLINE_FUNCTION
Real AtmosphereDensity(Real r, Real r_in, Real rho_atm, Real h_atm, Real bern, Real n) {
  const Real r_eff = fmax(r, r_in);
  return rho_atm*pow((bern - TR13StaticPot(r_eff))/h_atm, n);
}

}  // namespace

//----------------------------------------------------------------------------------------
//! \fn ProblemGenerator::UserProblem()
//! \brief Sets up the background atmosphere and enrolls the TR13 gravity source.

void ProblemGenerator::UserProblem(ParameterInput *pin, const bool restart) {
  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;

  // Read parameters (needed on restart too, for the source terms).
  params.r_in = pin->GetReal("problem", "r_in");
  params.rho_atm = pin->GetReal("problem", "rho_atm");
  params.r_atm = pin->GetReal("problem", "r_atm");
  params.bern = pin->GetOrAddReal("problem", "bern", 0.0);

  // AMR settings.
  amr_level = pin->GetInteger("problem", "amr_level");
  amr_rho_min = pin->GetReal("problem", "amr_rho_min");
  amr_entropy_fac = pin->GetReal("problem", "amr_entropy_fac");
  inj_lead = pin->GetOrAddReal("problem", "inj_lead", 1500.0);
  inj_pad = pin->GetOrAddReal("problem", "inj_pad", 2.0);
  si_pad = pin->GetOrAddReal("problem", "si_pad", 16.0);
  tinj = pin->GetReal("problem", "tinj");
  smr_boxes.clear();
  for (auto &b : pin->block) {
    if (b.block_name.compare(0, 14, "refined_region") != 0) continue;
    const std::string &n = b.block_name;
    if (pin->GetOrAddBoolean(n, "initial_only", false)) continue;
    smr_boxes.push_back({pin->GetReal(n, "x1min"), pin->GetReal(n, "x1max"),
                         pin->GetReal(n, "x2min"), pin->GetReal(n, "x2max"),
                         pin->GetReal(n, "x3min"), pin->GetReal(n, "x3max"),
                         pin->GetInteger(n, "level")});
  }
  if (pmy_mesh_->adaptive) user_ref_func = InjectionRefinement;

  // Atmosphere: P = kappa rho^gamma with kappa from h(r_atm) = B - Phi_s(r_atm),
  // i.e. kappa = h(r_atm) (gamma-1)/gamma / rho_atm^(gamma-1).
  const Real gamma = pmbp->phydro->peos->eos_data.gamma;
  const Real gm1 = gamma - 1.0;
  const Real n = 1.0/gm1;
  const Real r_in = params.r_in;
  const Real rho_atm = params.rho_atm;
  const Real bern = params.bern;
  const Real h_atm = bern - TR13StaticPot(params.r_atm);
  const Real kappa = h_atm*gm1/gamma/pow(rho_atm, gm1);
  params.rho_exc = AtmosphereDensity(r_in, r_in, rho_atm, h_atm, bern, n);
  params.eint_exc = kappa*pow(params.rho_exc, gamma)/gm1;

  // Streams: time-dependent, from the injection table; set here for the first window and
  // refreshed at every LAT window end (InjectionWindowEnd).
  auto &indcs = pmy_mesh_->mb_indcs;
  const int is = indcs.is, ie = indcs.ie;
  const int js = indcs.js, je = indcs.je;
  const int ks = indcs.ks, ke = indcs.ke;
  auto &size = pmbp->pmb->mb_size;
  stream_h = pin->GetReal("problem", "H");
  stream_erad = units::Units::rad_constant_cgs*pow(pin->GetReal("problem", "temp"), 4)/
                pmbp->punit->pressure_cgs();
  dx_fine = (pmy_mesh_->mesh_size.x1max - pmy_mesh_->mesh_size.x1min)/
            (pmy_mesh_->mesh_indcs.nx1*(1 << amr_level));
  ReadInjectionTable(pin->GetString("problem", "inj_file"));
  UpdateStreams(pmy_mesh_->mesh_size, pmy_mesh_->time);
  t_ledger = pmy_mesh_->time;
  m_inj_total = restart ? pin->GetOrAddReal("problem", "m_inj", 0.0) : 0.0;
  if (global_variable::my_rank == 0) {
    std::cout << "Injection table: " << inj_table.at("t").size() << " rows, t = "
              << inj_table.at("t").front() << " .. " << inj_table.at("t").back()
              << "; at t = " << pmy_mesh_->time << ": state = " << streams.on
              << ", mdot_in = " << streams.s[0].mdot << ", mdot_out = "
              << streams.s[1].mdot << std::endl;
  }

  // Enroll the source terms (TR13 gravity + stream injection).  They obey the LAT
  // contract (active blocks only, per-block dt, no MPI collectives, time-independent),
  // see docs/notes.md.
  user_srcs = true;
  user_srcs_func = InjectionSources;
  user_srcs_lat_safe = true;
  user_srcs_lat_union_safe = true;
  // LAT fixes a block's step for a whole window from its state at the window start, but
  // the injection source creates fast gas in blocks that held static background: keep the
  // injection blocks and their neighbours at factor 1.
  user_hydro_lat_factor_cap_func = CapLATFactorsAtInjection;
  // Excision: reset r < r_in every stage (idempotent, so safe under the union predictor).
  user_hydro_state_fixup_func = ExcisionFixup;
  user_hydro_state_fixup_lat_union_safe = true;
  // History: cumulative injected mass (M_inj) and mass removed by the excision (M_exc).
  // M_exc is reduced across ranks at LAT window ends (outputs and restarts happen only
  // there) and restored from <problem>/m_exc on restart.
  user_hist = true;
  user_hist_func = InjectionHistory;
  user_hydro_lat_window_end_func = InjectionWindowEnd;
  user_metadata_func = InjectionMetadata;
  exc_mass_local = 0.0;
  exc_mass_total = restart ? pin->GetOrAddReal("problem", "m_exc", 0.0) : 0.0;
  // Same-level LAT ghosts: use the linear time chord instead of the SSPRK2 dense output,
  // as the built-in gravity sources do (docs/wiki/Local-Adaptive-Time-Stepping.md).
  pmbp->phydro->ConfigureLATDenseOutput(false);

  if (restart) return;

  // Initialize the hydrostatic atmosphere ------------------------------------------------
  auto &w0 = pmbp->phydro->w0;

  par_for("pgen_injection_atm", DevExeSpace(), 0, (pmbp->nmb_thispack-1),
          ks, ke, js, je, is, ie,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    const Real x = CellCenterX(i-is, indcs.nx1, size.d_view(m).x1min,
                               size.d_view(m).x1max);
    const Real y = CellCenterX(j-js, indcs.nx2, size.d_view(m).x2min,
                               size.d_view(m).x2max);
    const Real z = CellCenterX(k-ks, indcs.nx3, size.d_view(m).x3min,
                               size.d_view(m).x3max);
    const Real r = sqrt(x*x + y*y + z*z);
    const Real rho = AtmosphereDensity(r, r_in, rho_atm, h_atm, bern, n);
    const Real pres = kappa*pow(rho, gamma);
    w0(m,IDN,k,j,i) = rho;
    w0(m,IVX,k,j,i) = 0.0;
    w0(m,IVY,k,j,i) = 0.0;
    w0(m,IVZ,k,j,i) = 0.0;
    w0(m,IEN,k,j,i) = pres/gm1;
  });

  pmbp->phydro->peos->PrimToCons(w0, pmbp->phydro->u0, is, ie, js, je, ks, ke);
}

namespace {

//----------------------------------------------------------------------------------------
//! \fn InjectionSources
//! \brief User source terms, evaluated on the stage primitives:
//!  - TR13 gravity: d(rho v)/dt = rho a,  dE/dt = rho v.a, with a = TR13Accel(x, v).
//!    Work only, so a dual-energy internal-energy field is untouched.  Skipped for r < r_in.
//!  - stream injection: in each stream's cylinder, d(rho)/dt = q, d(rho v)/dt = q v_s,
//!    dE/dt = q (v_s^2/2 + espec), with q = mdot/V_cyl.  With dual energy, the
//!    internal-energy field gains q espec plus the heat of mixing with the cell's gas.
//!    Streams and on/off state are fixed for the LAT window (UpdateStreams), so the
//!    source is time independent within a window and switches land on window boundaries.
//!
//! LAT: loops over the active (due) blocks only and uses each block's own step, following
//! TDEExternalGravitySource in tde_external.cpp.

void InjectionSources(Mesh *pm, const Real bdt) {
  MeshBlockPack *pmbp = pm->pmb_pack;
  auto &indcs = pm->mb_indcs;
  const int is = indcs.is, ie = indcs.ie;
  const int js = indcs.js, je = indcs.je;
  const int ks = indcs.ks, ke = indcs.ke;
  const int nmb = pmbp->nmb_thispack;
  auto &size = pmbp->pmb->mb_size;

  // LAT bookkeeping: active blocks and per-block step.
  const bool lat_enabled = pmbp->lat_active_mask_enabled;
  const bool lat_per_block_dt = pmbp->lat_per_block_timestep;
  const Real mesh_dt = pm->dt;
  const Real stage_weight = (lat_per_block_dt && mesh_dt > 0.0) ? (bdt/mesh_dt) : 0.0;
  auto lat_step_dt = pmbp->lat_step_dt.d_view;
  auto active_indices = pmbp->lat_active_indices.d_view;
  const int nwork1 = lat_enabled ? (pmbp->lat_nactive_thispack - 1) : (nmb - 1);
  if (nwork1 < 0) return;

  auto &w0 = pmbp->phydro->w0;
  auto &u0 = pmbp->phydro->u0;
  const Real r_in2 = params.r_in*params.r_in;
  const Streams st = streams;
  const bool dual = pmbp->phydro->use_dual_energy;
  const int de_idx = pmbp->phydro->dual_energy_idx;

  par_for("injection_srcs", DevExeSpace(), 0, nwork1, ks, ke, js, je, is, ie,
  KOKKOS_LAMBDA(int a, int k, int j, int i) {
    const int m = lat_enabled ? active_indices(a) : a;
    const Real block_bdt = lat_per_block_dt ? stage_weight*lat_step_dt(m) : bdt;

    const Real x = CellCenterX(i-is, indcs.nx1, size.d_view(m).x1min,
                               size.d_view(m).x1max);
    const Real y = CellCenterX(j-js, indcs.nx2, size.d_view(m).x2min,
                               size.d_view(m).x2max);
    const Real z = CellCenterX(k-ks, indcs.nx3, size.d_view(m).x3min,
                               size.d_view(m).x3max);
    if (x*x + y*y + z*z < r_in2) return;

    const Real rho = w0(m,IDN,k,j,i);
    const Real vx = w0(m,IVX,k,j,i);
    const Real vy = w0(m,IVY,k,j,i);
    const Real vz = w0(m,IVZ,k,j,i);
    Real ax, ay, az;
    TR13Accel(x, y, z, vx, vy, vz, ax, ay, az);

    const Real src = block_bdt*rho;
    u0(m,IM1,k,j,i) += src*ax;
    u0(m,IM2,k,j,i) += src*ay;
    u0(m,IM3,k,j,i) += src*az;
    u0(m,IEN,k,j,i) += src*(vx*ax + vy*ay + vz*az);

    if (!st.on) return;
    for (int n = 0; n < kNStreams; ++n) {
      const Stream &s = st.s[n];
      if (!InStream(s, x, y, z)) continue;
      const Real dm = block_bdt*s.q;
      if (dual) {
        // Internal energy gained: the injected gas's own, plus the kinetic energy lost
        // when it mixes inelastically with the cell's gas (E gets it implicitly as E - K).
        const Real d = u0(m,IDN,k,j,i);
        const Real dvx = u0(m,IM1,k,j,i)/d - s.vx;
        const Real dvy = u0(m,IM2,k,j,i)/d - s.vy;
        const Real dvz = u0(m,IM3,k,j,i)/d;
        u0(m,de_idx,k,j,i) += dm*s.espec +
            0.5*d*dm/(d + dm)*(dvx*dvx + dvy*dvy + dvz*dvz);
      }
      u0(m,IDN,k,j,i) += dm;
      u0(m,IM1,k,j,i) += dm*s.vx;
      u0(m,IM2,k,j,i) += dm*s.vy;
      u0(m,IEN,k,j,i) += dm*(0.5*(s.vx*s.vx + s.vy*s.vy) + s.espec);
    }
  });
}

//----------------------------------------------------------------------------------------
//! \fn CapLATFactorsAtInjection
//! \brief LAT factor cap: every block whose bounding box overlaps an injection cylinder
//! (bounded by its circumscribed sphere, radius sqrt(H^2 + (H/2)^2)), and all of its
//! neighbours, take factor 1.  lat_factor_eachmb is indexed by global ID; each rank caps
//! its own blocks and the result is MIN-reduced (this hook runs on the synchronized path,
//! so the collective is allowed).

void CapLATFactorsAtInjection(Mesh *pm, int max_factor, int *lat_factor_eachmb) {
  MeshBlockPack *pmbp = pm->pmb_pack;
  auto &size = pmbp->pmb->mb_size;
  auto &mb_gid = pmbp->pmb->mb_gid;
  auto &nghbr = pmbp->pmb->nghbr;
  const int nnghbr = pmbp->pmb->nnghbr;
  const int nmb_total = pm->nmb_total;
  auto cap = [&](const int gid) {
    if (gid >= 0 && gid < nmb_total) lat_factor_eachmb[gid] = 1;
  };

  for (int m = 0; m < pmbp->nmb_thispack; ++m) {
    const RegionSize &b = size.h_view(m);
    for (int ns = 0; ns < kNStreams; ++ns) {
      const Stream &s = streams.s[ns];
      const Real rad = sqrt(1.25)*s.h;
      if (b.x1max > s.x - rad && b.x1min < s.x + rad &&
          b.x2max > s.y - rad && b.x2min < s.y + rad &&
          b.x3max > -rad && b.x3min < rad) {
        cap(mb_gid.h_view(m));
        for (int n = 0; n < nnghbr; ++n) cap(nghbr.h_view(m,n).gid);
      }
    }
  }

#if MPI_PARALLEL_ENABLED
  if (global_variable::nranks > 1) {
    MPI_Allreduce(MPI_IN_PLACE, lat_factor_eachmb, nmb_total, MPI_INT, MPI_MIN,
                  MPI_COMM_WORLD);
  }
#endif
}

//----------------------------------------------------------------------------------------
//! \fn ExcisionFixup
//! \brief Excision: owned cells with r < r_in are reset to the background atmosphere
//! state at r_in (rho_exc, eint_exc) with v = 0, so the excision pressure-supports the
//! hydrostatic atmosphere.  Acts on the conserved u0 before boundary exchange and primitive
//! recovery.  Under LAT only the active (due) blocks are touched.
//! The mass the reset removes is booked into exc_mass_local, weighted by how much of this
//! state survives into the completed step (user_hydro_state_fixup_weight: 0.5 after rk2
//! stage 1), so M_exc is the mass actually lost from the grid.

void ExcisionFixup(MeshBlockPack *pmbp, const Real time) {
  auto &indcs = pmbp->pmesh->mb_indcs;
  const int is = indcs.is, js = indcs.js, ks = indcs.ks;
  const int nx1 = indcs.nx1;
  const int nkji = indcs.nx3*indcs.nx2*nx1, nji = indcs.nx2*nx1;
  auto &size = pmbp->pmb->mb_size;

  const bool lat_enabled = pmbp->lat_active_mask_enabled;
  auto active_indices = pmbp->lat_active_indices.d_view;
  const int nwork1 = lat_enabled ? (pmbp->lat_nactive_thispack - 1)
                                 : (pmbp->nmb_thispack - 1);
  if (nwork1 < 0) return;

  auto &u0 = pmbp->phydro->u0;
  const Real rho_exc = params.rho_exc;
  const Real eint_exc = params.eint_exc;
  const Real r_in2 = params.r_in*params.r_in;
  const bool dual = pmbp->phydro->use_dual_energy;
  const int de_idx = pmbp->phydro->dual_energy_idx;

  Real removed = 0.0;
  Kokkos::parallel_reduce("injection_excision",
      Kokkos::RangePolicy<>(DevExeSpace(), 0, (nwork1 + 1)*nkji),
  KOKKOS_LAMBDA(const int idx, Real &sum) {
    const int a = idx/nkji;
    const int k = (idx - a*nkji)/nji;
    const int j = (idx - a*nkji - k*nji)/nx1;
    const int i = idx - a*nkji - k*nji - j*nx1;
    const int m = lat_enabled ? active_indices(a) : a;
    const RegionSize &s = size.d_view(m);
    const Real x = CellCenterX(i, nx1, s.x1min, s.x1max);
    const Real y = CellCenterX(j, indcs.nx2, s.x2min, s.x2max);
    const Real z = CellCenterX(k, indcs.nx3, s.x3min, s.x3max);
    if (x*x + y*y + z*z >= r_in2) return;
    const int kk = k + ks, jj = j + js, ii = i + is;
    sum += (u0(m,IDN,kk,jj,ii) - rho_exc)*s.dx1*s.dx2*s.dx3;
    u0(m,IDN,kk,jj,ii) = rho_exc;
    u0(m,IM1,kk,jj,ii) = 0.0;
    u0(m,IM2,kk,jj,ii) = 0.0;
    u0(m,IM3,kk,jj,ii) = 0.0;
    u0(m,IEN,kk,jj,ii) = eint_exc;
    if (dual) u0(m,de_idx,kk,jj,ii) = eint_exc;
  }, removed);
  exc_mass_local += pmbp->pmesh->pgen->user_hydro_state_fixup_weight*removed;
}

//----------------------------------------------------------------------------------------
//! \fn InjectionWindowEnd
//! \brief LAT window end (all bins and refluxes done, before outputs/AMR; collectives
//! allowed): fold the rank-local excised mass into the global total, book the mass
//! injected over the finished window, and set the streams for the next window (this runs
//! before AMR and the next window's LAT factors, which both see the new streams).

void InjectionWindowEnd(Mesh *pm) {
  Real m = exc_mass_local;
#if MPI_PARALLEL_ENABLED
  MPI_Allreduce(&exc_mass_local, &m, 1, MPI_ATHENA_REAL, MPI_SUM, MPI_COMM_WORLD);
#endif
  exc_mass_total += m;
  exc_mass_local = 0.0;

  if (streams.on) {
    m_inj_total += (streams.s[0].mdot + streams.s[1].mdot)*(pm->time - t_ledger);
  }
  t_ledger = pm->time;
  UpdateStreams(pm->mesh_size, pm->time);
}

//----------------------------------------------------------------------------------------
//! \fn InjectionHistory
//! \brief User history columns (the writer sums hdata over ranks, so global values are
//! divided by nranks):
//!   M_inj: cumulative injected mass, summed window by window (exact, since the streams
//!          are constant within a window and q uses the discrete cylinder volume).
//!   M_exc: cumulative mass removed by the excision (mass flowing into r < r_in).
//!   mdot_in, mdot_out, inj_on: the streams of the current window.

void InjectionHistory(HistoryData *pdata, Mesh *pm) {
  const Real share = 1.0/static_cast<Real>(global_variable::nranks);
  pdata->nhist = 5;
  pdata->label[0] = "M_inj";
  pdata->label[1] = "M_exc";
  pdata->label[2] = "mdot_in";
  pdata->label[3] = "mdot_out";
  pdata->label[4] = "inj_on";
  const Real dm_window = streams.on ?
      (streams.s[0].mdot + streams.s[1].mdot)*(pm->time - t_ledger) : 0.0;
  pdata->hdata[0] = share*(m_inj_total + dm_window);
  pdata->hdata[1] = share*exc_mass_total + exc_mass_local;
  pdata->hdata[2] = share*streams.s[0].mdot;
  pdata->hdata[3] = share*streams.s[1].mdot;
  pdata->hdata[4] = share*(streams.on ? 1.0 : 0.0);
}

//----------------------------------------------------------------------------------------
//! \fn InjectionMetadata
//! \brief Before restart/binary dumps: store the excised- and injected-mass totals for
//! restarts.  Dumps happen at LAT window ends, after InjectionWindowEnd, so
//! exc_mass_local is zero and t_ledger = time.

void InjectionMetadata(ParameterInput *pin, Mesh *pm) {
  pin->SetReal("problem", "m_exc", exc_mass_total);
  pin->SetReal("problem", "m_inj", m_inj_total);
}

//----------------------------------------------------------------------------------------
//! \fn InjectionRefinement
//! \brief User AMR criterion (<amr_criterion> method = user).  The target level of a
//! block is the highest of
//!   - the level of every <refined_regionN> box it overlaps (static minimum levels), and
//!   - amr_level, if it holds stream gas anywhere in the domain: a cell with
//!     rho > amr_rho_min and P/rho^gamma < amr_entropy_fac x K_s, where K_s is the larger
//!     of the two streams' injected entropies (current window).  The entropy cut excludes
//!     the shock-heated CIO, whose entropy survives its expansion and fallback, and
//!   - amr_level, if it overlaps an injection box or the collision box while those are
//!     enforced.
//! Blocks below the target are flagged to refine, above it to derefine (the mesh only
//! derefines when all siblings agree, and enforces 2:1 level balance itself).

void InjectionRefinement(MeshBlockPack *pmbp) {
  Mesh *pm = pmbp->pmesh;
  auto &refine_flag = pm->pmr->refine_flag;
  const int mbs = pm->gids_eachrank[global_variable::my_rank];
  const int nmb = pmbp->nmb_thispack;
  auto &indcs = pm->mb_indcs;
  const int is = indcs.is, js = indcs.js, ks = indcs.ks;
  const int nx1 = indcs.nx1, nx2 = indcs.nx2, nx3 = indcs.nx3;
  const int nkji = nx3*nx2*nx1, nji = nx2*nx1;

  // Entropy cut: amr_entropy_fac x the larger injected stream entropy
  // K_s = P_s/rho_s^gamma, P_s = (gamma-1) a_rad T^4, rho_s = mdot/(pi H^2 v).
  const Real gamma = pmbp->phydro->peos->eos_data.gamma;
  const Real gm1 = gamma - 1.0;
  Real k_stream = 0.0;
  for (int ns = 0; ns < kNStreams; ++ns) {
    const Stream &st = streams.s[ns];
    const Real rho_s = st.mdot/(M_PI*st.h*st.h*sqrt(st.vx*st.vx + st.vy*st.vy));
    k_stream = fmax(k_stream, gm1*stream_erad/pow(rho_s, gamma));
  }
  const Real k_max = amr_entropy_fac*k_stream;

  // max density over the owned cells of each block that pass the entropy cut (stream gas)
  DualArray1D<Real> rho_max("inj_amr_rho_max", nmb);
  auto rmax = rho_max.d_view;
  auto &w0 = pmbp->phydro->w0;
  par_for_outer("inj_amr_rho_max", DevExeSpace(), 0, 0, 0, (nmb-1),
  KOKKOS_LAMBDA(TeamMember_t tmember, const int m) {
    Real team_max = 0.0;
    Kokkos::parallel_reduce(Kokkos::TeamThreadRange(tmember, nkji),
    [&](const int idx, Real &rmx) {
      const int k = idx/nji;
      const int j = (idx - k*nji)/nx1;
      const int i = idx - k*nji - j*nx1;
      const Real rho = w0(m,IDN,k+ks,j+js,i+is);
      const Real p = gm1*w0(m,IEN,k+ks,j+js,i+is);
      if (p < k_max*pow(rho, gamma)) rmx = fmax(rmx, rho);
    }, Kokkos::Max<Real>(team_max));
    Kokkos::single(Kokkos::PerTeam(tmember), [&]() { rmax(m) = team_max; });
  });
  rho_max.template modify<DevExeSpace>();
  rho_max.template sync<HostMemSpace>();

  auto &size = pmbp->pmb->mb_size;
  auto &lev = pmbp->pmb->mb_lev;
  auto overlaps = [](const RegionSize &s, const RefBox &b) {
    return s.x1max > b.x1min && s.x1min < b.x1max &&
           s.x2max > b.x2min && s.x2min < b.x2max &&
           s.x3max > b.x3min && s.x3min < b.x3max;
  };
  for (int m = 0; m < nmb; ++m) {
    const RegionSize &s = size.h_view(m);
    int target = 0;
    for (const auto &b : smr_boxes) {
      if (b.level > target && overlaps(s, b)) target = b.level;
    }
    if (inj_box_on) {
      for (const auto &b : inj_box) {
        if (b.level > target && overlaps(s, b)) target = b.level;
      }
    }
    if (si_box_on && si_box.level > target && overlaps(s, si_box)) target = si_box.level;
    if (amr_level > target && rho_max.h_view(m) > amr_rho_min) target = amr_level;
    const int level = lev.h_view(m) - pm->root_level;
    refine_flag.h_view(m + mbs) = (level < target) ? 1 : ((level > target) ? -1 : 0);
  }
  refine_flag.template modify<HostMemSpace>();
  refine_flag.template sync<DevExeSpace>();
}

}  // namespace
