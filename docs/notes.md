# Simulation overview

I want to simulation the stream self-intersection
and resulting collision-induced outflow (CIO)
in the global context.

I'm thinking of injecting streams close to the self-intersection site
and letting the CIO fall back to the black hole 
and build up an envelope.

For now, I'd like to run an adiabatic simulation with gamma=4/3.

# Hongxuan's fork

I am starting with Hongxuan Jiang's AthenaK fork, 
which includes extra features relevant for TDE simulations.
I am most interested in the local adaptive timestepping (LAT) feature,
since I plan to use many levels of adaptive mesh refinement (AMR).
I will need custom source terms 
for the stream injection and gravitational potential,
which the documentation says is incompatible with LAT,
so we might need to modify the code to make the source terms work.

# Design decisions

- **Code style:** keep the pgen minimal. The user is the only one running it and supplies
  sensible parameters, so skip input-validation / configuration checks.

- **Stream injection:** interior region, via source terms.
  Per-stream parameters from the input file: mdot, x, y, xdot, ydot, rho, pres, H (stream radius).
- **Background:** low-density hydrostatic atmosphere.
- **BH region:** excised inside r_in (input parameter), possibly also via source terms.
- **Gravity:** Tejeda & Rosswog 2013 generalized Newtonian potential (helpers in `src/pgen/injection.hpp`, units G=M=c=1).
- **Geometry:** Cartesian box with nested SMR around the intersection point and the BH.
  Refinement around the intersection point still TBD.
- **BH SMR (implemented in deck):** 4096^3 r_g box (±2048), 128^3 root (dx0 = 32),
  nested cubes ±2^(11-l) at level l = 1..7 (AthenaK root = level 0), i.e.
  level = 11 - log2(r), dx ≈ |x|_inf/64. Finest dx = 0.25 inside ±16; no finer level
  because r < r_in = 8 is excised (r_in = 32 finest cells in radius).
  32^3 MeshBlocks: cube edges coincide with parent-block edges (build_tree.cpp pads
  regions outward to whole parent blocks, so this alignment matters).
  456 MeshBlocks, ~15M cells.
- **Performance change (2026-11-07):** first test ran at ~1.3 s wall per r_g/c (tick 0.39,
  set by c_s ≈ 0.18 of the hydrostatic atmosphere at r_in = 8 with dx = 0.25; BH cubes
  ~2/3 of block-steps). Now r_in = 16 and the dx = 0.25 BH cube (old refined_region7) is
  removed: finest BH level 6 (dx = 0.5 inside ±32), r_in still 32 cells in radius, c_s(16)
  ≈ 0.13. r_in = 16 is well inside the pericentre (~47 r_g for a solar star, beta = 1,
  1e6 Msun). cfl_number = 0.5 (as the tde production decks with rk2/PLM). User moving to GPUs.
  Collision-site dx = 0.25 SMR box (old refined_region8) also removed: finest level is now 6
  (dx = 0.5) everywhere, num_levels = 7. Collision region is resolved only by the density
  AMR (amr_level = 6).
- **Injection-region refinement (implemented):** refinement = adaptive with a user criterion
  (`InjectionRefinement`). Injection box from `<problem>/inj_x{1,2,3}{min,max}`
  (currently -1169..-919, -629..-379, -125..125; centre (-1044, -504, 0)).
  Inside it, blocks with max density > `amr_rho_min` go to `amr_level` = 6 (dx = 0.5).
  (A dx = 0.25 collision-site SMR box was tried and later removed for cost.)
  Under AMR, `<refined_regionN>` only builds the initial tree and nothing protects it from
  derefinement, so the user criterion reads every refined_region back and enforces it as a
  minimum level. Target level = max(overlapped refined_region levels, amr_level if dense
  and in box); flag +1 below target, -1 above (core handles sibling/2:1 rules).
  Single threshold, no hysteresis: add a derefine threshold if blocks flap.
- **LAT:** `lat_same_level = true` (ratio 2), so injection-box blocks are not slaved to the
  hot BH shell on the same level.
- **Units:** `<units>` from ref/athinput.tde_si (M_BH = 1e6 Msun with Msun = 1.9891e33;
  r_g = 1.4770934e11 cm; r_g/c = 4.9270606 s). mdot = 1.5623606789395503e-13 = 1 Msun/yr
  (365-day yr) in these units. No mu: temperatures are radiation temperatures.
- **Radiation dominated:** gamma = 4/3 represents radiation-pressure-dominated gas, so a
  temperature T [K] maps to internal energy density e = a_rad T^4 (P = e/3), not k T/(mu m_u).
- **Stream injection (implemented):** per stream (`nstreams`, keys x/y/xdot/ydot/mdot/temp/H
  + index): uniform mass, momentum, energy source over a cylinder of radius H and length H,
  axis along v, centred on (x, y, 0). q = mdot / V_cyl with V_cyl the discrete volume of cells
  whose centres are inside (MPI-summed once in UserProblem), so exactly mdot is injected.
  Injected gas has velocity v_s and specific internal energy a_rad T^4 / rho_s with
  rho_s = mdot/(pi H^2 v), so the stream reaches e = a_rad T^4. Implied stream density
  ≈ 5.6e-14 (stream 1), 6.6e-14 (stream 2); T = 1e5 K gives P = a T^4/3 ≈ 4.5e-22
  (code; pressure unit ≈ 5.5e26 dyn/cm^2). Note: at these densities (~3.5e-8 g/cm^3) gas
  pressure at 1e5 K would be ~2x this, so the injected stream itself is not radiation
  dominated; irrelevant dynamically (Mach ~ 300). Same kernel as the TR13 gravity
  (`InjectionSources`).
  Current streams: x1 = -919 (inj box x max, moving -x), x2 = -1169 (x min, moving +x),
  y/xdot/ydot from ref; mdot = 1 Msun/yr, temp = 1e5 K, H = 5 for both.
  dx = 0.5 SMR boxes (`refined_region9/10`) of ±16 x ±16 x ±8 around the injection points,
  hand-written (keep consistent with x/y). V_cyl assumes this mesh is static.
- **LAT blow-up at injection (2026-11-07):** with cfl = 0.5 the run died in the first LAT
  window (t ≈ 236 = 128 ticks; dt → 1e-42). Diagnosis (not yet confirmed from the dump):
  LAT fixes each block's factor for a whole window from the window-start state; at t = 0
  the injection blocks held static background (c_s ≈ 0.017) and got a step ~7.4, but the
  source immediately creates 0.036c gas there (stable step ≲ 6.9), so they ran over their
  CFL limit all window. Also, the 2:1 same-level ratio lets a block receiving fast gas step
  at up to 2x its own CFL step. Fixes: `CapLATFactorsAtInjection` (user_hydro_lat_factor_cap_func)
  pins blocks overlapping each injection cylinder's bounding sphere + their 26 neighbours
  to f = 1; `<time>/lat_pin_density = 1e-15` pins blocks containing stream gas + neighbours
  to f = 1 (the stream moves < 1 block per window, so its head always enters pinned blocks);
  cfl_number = 0.4 for margin. The general rule: any source that creates fast gas inside a
  window needs its blocks capped.
- **Injection duty cycle (implemented, 2026-11-08):** `<problem>/inj_t_on`, `inj_t_off`
  (deck: 44586 each, r_g/c); on for t_on, off for t_off, repeating, starting on at t = 0;
  t_off <= 0 = always on. Applies to all streams together; gravity is unaffected.
  Under LAT the on/off test reads each block's own step start `lat_time_start(m)` (set on
  device for active blocks in both the ordinary and the fused union-predictor paths), else
  the stage time; so a block step straddling a switch is all on or all off (error <= one
  fine tick of mdot per switch, ~1e-5 of a phase). The injection LAT cap stays on while
  off (harmless). Purely time based, so restarts resume the cycle.
- **Background vs streams:** a hot hydrostatic atmosphere must have P_bg << P_stream at the
  injection site or it confines the cold stream; P/rho_bg ≈ |Phi_s|/4 ≈ 2e-4 there vs ~8e-9
  in the stream, so rho_bg must be << 1e-4 rho_stream. Hence rho_atm = 1e-18 at r = 100
  (rho_bg ≈ 8e-22 near the box, P_bg ≈ 4e-4 P_stream), dfloor = 1e-26, pfloor = 1e-32.
  amr_rho_min was later raised to 1e-14 (~0.1 rho_stream) so AMR fires around the
  streams only.
  **Lowered 100x (2026-11-08):** rho_atm = 1e-20 (rho_bg ≈ 8e-24 near the box, 2e-25 at
  corners, 2e-18 at r_in), dfloor = 1e-28, pfloor = 1e-34. Reason: the background goes as
  r^-3 (n = 3), and a quasi-spherical fallback estimate rho_CIO ≈ mdot_tot/(4 pi r^2 v_ff)
  ≈ 3e-16 at r_in was comparable to the old rho_bg(r_in) ≈ 1.8e-16. The zero-Bernoulli
  profile fixes P/rho at each radius, so c_s and the timestep are unchanged; the costs
  are larger density contrasts (rarefactions toward the floor) and hotter shocked
  background. lat_pin_density (1e-15) and amr_rho_min (1e-14) are tied to the stream,
  so they are unchanged. Not yet checked against the measured CIO density.
- **NaNs once the CIO reaches the injection site (2026-11-08; RESOLVED by 2026-11-09 by
  `vceil` / `cs_ceil` = 0.5; keep them on):** NaNs spread from the
  injection point after the outflow expands back to it. First, least-intrusive try:
  `<hydro>/vceil = 0.5` and `cs_ceil = 0.5` (c units; both applied in the ideal-hydro C2P,
  which also resyncs the conserved state). Restart dumps (`<output3>`, dt = 500 = every 10
  hydro_w outputs) so the run can be restarted just before the failure with dense dumps.
  Event log (`<output4>`, `file_type = log`, dcycle = 1 → `injection.log`) counts
  floor/vceil/c2p-failure hits per cycle, to see whether the caps actually fire.
  First log (no CIO yet): only eos_efloor fires, growing to ~6.5e4 hits/cycle by cycle
  3500, presumably E - K going non-positive in the Mach ~300 stream (unconfirmed; a
  cs_ceil hit also counts as efloor).
- **Dual energy (implemented, 2026-11-08):** `<hydro>/dual_energy = true` (Bryan et al. eta1
  test, default 1e-3; this fork's shock rule sends shocked cells to E - K so shock heating
  is kept). Internal-energy column `u0(dual_energy_idx)`; sources that change internal
  energy write it directly (as the core cooling debit does). Injection adds dm*espec plus
  the inelastic mixing heat 0.5 rho dm/(rho+dm) |v - v_s|^2, so it matches the E - K
  change exactly. Excision sets it to eint_exc. TR13 gravity is work only, so it is
  untouched. LAT supports Newtonian dual energy (wiki). Under AMR the core switches to
  prolong_prims automatically. Floors kept (dfloor still needed for vacuum; pfloor is a
  backstop): check the event log to see whether efloor hits drop.
  Not yet tried: `fofc`, higher floors.
- **Input deck:** `inputs/injection.athinput` (build with `-D PROBLEM=injection`).
  Current mesh/LAT values are placeholders.
- **Atmosphere Bernoulli constant (implemented, 2026-11-08):** `<problem>/bern` (default 0):
  h = B - Phi_s(r), rho = rho_atm (h(r)/h(r_atm))^n, still exactly hydrostatic (any constant
  B is). B > 0 flattens the profile to ~constant for r >> 1/B and rises only inside.
  Motivation: the zero-B profile (rho ~ r^-3) is densest at r_in where the CIO fallback is
  thinnest; a constant-density, non-hydrostatic background was rejected because it would
  free-fall (~0.35c at r_in on the finest level, worse for dt) and pile up as r^-1.5 anyway.
  Deck: bern = 0.01, rho_atm = 1e-22 at r_atm = 100. Gives rho ≈ 1.7e-23 / 1.6e-23 at the
  two injection points (P ≈ 1e-4 P_stream, c_s ≈ 0.06), 1.4e-23 at the corners,
  3.5e-21 at r_in (~1e-5 of the fallback estimate). Costs: hotter background, c_s(r_in)
  0.147 vs 0.135 (~9% on the BH-set fine tick). Bonus: the outer hydrostatic gradient is
  ~20x shallower (dlnP/dr = -4/(r^2 h)), so the zero-gradient outflow boundary is closer
  to equilibrium. Supersedes the rho_atm = 1e-20 / 1e-18 values below; floors unchanged
  (dfloor = 1e-28, pfloor = 1e-34).
- **Atmosphere (implemented):** zero-Bernoulli polytrope, P = K rho^gamma with the gas gamma,
  in the *static* TR13 potential Phi_s = -1/r + 2/r^2 - 4/(3r^3) (`TR13StaticPot`).
  This is the potential whose gradient equals TR13Accel at v = 0, i.e. g = (1-2/r)^2/r^2;
  it is NOT TR13Pot at v = 0 (-1/r), because the TR13 potential is velocity dependent.
  rho = rho_atm (Phi_s(r)/Phi_s(r_atm))^n, n = 1/(gamma-1); profile frozen at r_in for r < r_in.
  Params: `problem/rho_atm`, `problem/r_atm`, `problem/r_in`.
  Only hydrostatic analytically, not to discretization error.
- **TR13 gravity source (implemented):** `TR13GravitySource` in `injection.cpp`.
  Momentum += dt rho a, energy += dt rho v·a (primitives), skipped for r < r_in.
  Marked `user_srcs_lat_safe` and `user_srcs_lat_union_safe`; LAT dense-output ghost
  interpolation disabled (linear chord), as for the built-in gravity sources.
- **Excision (implemented):** `ExcisionFixup` (enrolled as `user_hydro_state_fixup_func`,
  union-safe since it is idempotent) resets owned cells with r < r_in to the atmosphere
  state at r_in (rho_exc, eint_exc; v = 0) on the conserved u0, active LAT blocks only.
  This pressure-supports the hydrostatic atmosphere. A vacuum sink (dfloor/pfloor) was
  tried first and rejected: the background drained into the BH.
  No LAT factor cap near the BH (removed): the BH is static and the excision sits inside
  the finest SMR level, which sets the fine tick anyway; a cap with neighbour pinning
  forced all of level 6 to f=1 (one bin per level, before lat_same_level = true).
  (A cap hook now exists for the injection points; see below.)
  The excision looks square when r_in is only a few cells (r_in=10, dx=4 in the placeholder
  deck); the test is spherical, so this needs resolution (≳10 cells per r_in), not code.

# LAT implementation findings

Reference: `docs/wiki/Local-Adaptive-Time-Stepping.md`.
LAT does **not** refuse user source terms outright; it refuses them unless the pgen
sets `user_srcs_lat_safe = true` (gate in `src/driver/driver.cpp` ~L672).
`tde_external.cpp` (`TDEExternalGravitySource`) is a working LAT-safe user source to copy.
No core-code changes should be needed for injection + gravity source terms.

Contract a LAT-safe user source must satisfy:
- Loop only over active blocks: `m = lat_enabled ? lat_active_indices(a) : a`,
  with `nwork1 = lat_nactive_thispack - 1` (return if < 0).
- Per-block step: `block_bdt = lat_per_block_timestep ? (bdt/pm->dt)*lat_step_dt(m) : bdt`.
  (In the fused union predictor `pm->dt` != each block's step; otherwise all due blocks share `pm->dt`.)
- Stage time from `pgen->user_srcs_time` (valid flag `user_srcs_time_valid`);
  in the union predictor this is not per-block, so the sources should be time-independent
  (constant mdot, static potential) or read per-block `lat_time_start(m)`.
- No MPI collectives inside the source (a rank with no due block would hang).
  Global reductions (e.g. injected-mass ledger) go in `user_hydro_lat_window_end_func`.
- Set `user_srcs_lat_union_safe = true` too, or the fused predictor is disabled (slow).
- Source terms must use primitives `w0`, write conserved `u0`.

Relevant hooks for this problem:
- `user_hydro_lat_factor_cap_func` (example `CapHydroLATFactorsNearBH`): force f=1 near the BH.
  Probably also want it around the injection region.
- `user_hydro_state_fixup_func` (+ `user_hydro_state_fixup_lat_union_safe`, must be idempotent):
  used by tde_external to reset excised cells; candidate for the r_in excision.
- `user_dt_func` needs `user_dt_lat_safe`; injection may need a dt limit (mdot/rho vs cell mass).

Caveats to handle:
- Built-in BH gravity refunds the gravitational work of mass moved by the delayed
  reflux across bin interfaces (`srcterms.cpp` Gravity gate). A user gravity source
  does not get this refund, so energy conservation across bin faces is approximate.
- TR13 acceleration is velocity-dependent; the energy source should be rho v·a
  (not the potential-difference Mullen-Hanawa-Gammie form used for Newtonian gravity).
- Integrator must be rk1, rk2 or imex2; hydro only (no MHD).
- Outputs/restarts/AMR happen only at window ends.
