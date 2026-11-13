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
  Refinement around the intersection point: density AMR plus look-ahead boxes around the
  injection cylinders (see "Injection boxes" under "Time-dependent streams" below).
- **Time-dependent streams (implemented, 2026-11-10):** streams now come from the table
  `data/inj.ascii` (`<problem>/inj_file`, path relative to the run directory). Supersedes
  the fixed per-stream deck keys (x1, xdot1, mdot1, ..., nstreams) and the
  inj_t_on/inj_t_off duty cycle. Table facts (fiducial beta = 2 disruption of a sun-like
  star, since beta = 1 does not fully disrupt it; header rp = 23.56, a_mb = 666.9,
  tinj = 1000; everything in code units, including mdot):
  - columns t, state (1 on / 0 off), r_si/phi (= xc/yc in polar form), a_in/a_out
    (semi-major axes of the inbound/outbound elements, unused), xc/yc (SI point),
    x/y/vx/vy_in and _out (injection point and velocity of the inbound/outbound stream),
    mdot_in/mdot_out. 15974 rows, log spaced (dt/t ~ 3.6e-4), t = 1.2550e5 .. 4.06e7.
  - A row's t is when the gas is injected (user decision; no tinj shift). The injection
    points sit ~30 r_g upstream of (xc, yc): the SI point integrated back along each stream
    for tinj under gravity.
  - All columns are smooth across state switches; on/off phases are ~600 rows each,
    ~3.6e4, 4.0e4, 4.4e4, ... r_g/c and growing.
  - SI point drifts from r = 883 to 2285 (phi ~ 3.36), ~2.7e-3 r_g per r_g/c early.
  - mdot_in 2.5e-14 at t0, peak ~1.6e-13 near t ~ 2e5, ~4.5e-14 at t = 2e6; mdot_out
    starts ~12x lower (2.1e-15) and converges to mdot_in. The last rows of mdot_in are
    repeated exactly (clamp in the generating code, irrelevant before 4e7).
  Implementation: two streams (0 = inbound, 1 = outbound). The table is read on every
  rank (columns by header name), interpolated linearly in t, state from the row at or
  before t, once per LAT window (`InjectionWindowEnd` -> `UpdateStreams`, user decision:
  once per window is fine). The streams are constant within a window, so the source is
  time independent (LAT/union safe) and on/off switches land on window boundaries (error
  <= one window, ~100 r_g/c, vs ~4e4 phases). The per-block lat_time_start duty-cycle test
  is gone. H = 5 and temp = 1e5 K are fixed for both streams (`<problem>/H`, `temp`;
  user decision, for now), so rho_s = mdot/(pi H^2 v) follows mdot.
  q = mdot / V_cyl with V_cyl counted on the host on the global amr_level lattice (dx = 0.5),
  valid because the injection boxes keep the cylinders on that level; no MPI reduction.
  **Injection boxes (2026-11-10, replace the tracking box):** one box per cylinder: its
  bounding sphere at t and at t + `<problem>/inj_lead`, padded by `<problem>/inj_pad`
  (2 r_g) in x/y, z = +-sqrt(1.25) H, enforced at amr_level by `InjectionRefinement`
  only while the table is on at t or at t + inj_lead (look-ahead; user decision over
  delaying each on switch until refined). Reasoning: downstream of the cylinders the
  density criterion already keeps the streams refined (and the SI point / CIO, which are
  dense), so only the cylinders need forcing: for exact V_cyl, because the cylinders creep
  into blocks without dense gas yet, and because gas is still building up in the first
  windows of an on phase. The look-ahead is needed since the site sits on BH-nest levels
  2 -> 0 as the SI point drifts out, AMR adds one level per check, and amr_level = 7, so
  refinement needs up to 7 checks; inj_lead = 1500 r_g/c (~15 windows, ~4% of an off
  phase). Off phases (beyond the look-ahead) derefine the site unless dense. The old box
  also covered (xc, yc) with track_pad = 8 (~36 blocks at t0), held in off phases too.
  The deck's `<refined_region8>` (`initial_only = true`, not enforced by the pgen) still
  builds the old, larger t0 box at start_time (the table is on at t0); its excess
  derefines unless dense. On restart build_tree will warn that this region is ignored;
  harmless. Not yet run: check that the cylinder blocks are at amr_level when each on
  phase starts (AMR check spacing under LAT is assumed, ~1 window).
  **Run:** `<time>/start_time` = first table time (1.255039907173705e5), tlim = 2e6 (user
  decision). Output cadences raised for the long run (hst 100, 2D 1000, 3D 1e5, rst 2e4;
  my choice, revisit for disk use).
  **Thresholds:** fixed, from the lowest stream density in the run (user decision): the
  outbound stream at t0, rho_s = 2.1e-15/(pi 25 x 0.029) = 9.2e-16. amr_rho_min = 1e-16
  and lat_pin_density = 1e-16 (~0.1 of it). Risk: CIO gas above 1e-16 anywhere goes to
  dx = 0.5 (100x lower than before), so watch the block count against
  max_nmb_per_rank = 1500.
  **Ledger:** M_inj is summed per window (exact) and restarted from `<problem>/m_inj`.
  History also has mdot_in, mdot_out, inj_on (current window).
  r_in = 16 kept (user decision) although rp = 23.6 for beta = 2.
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
- **Box doubled (2026-11-09):** first trial run showed the box was too small. Now 8192^3 r_g
  (±4096), still 128^3 root (dx0 = 64), with a new level-1 cube ±2048 (dx = 32 = old root).
  Every existing region keeps its dx but is one level number higher: BH cubes ±2^(12-l)
  at l = 1..7, refined_region9/10 and amr_level = 7, num_levels = 8, lat_levels = 8.
  456 BH MeshBlocks (was 400). Level numbers quoted in older entries below are pre-shift
  (old level L = new level L+1).
- **GPU memory limit (2026-11-09):** one GPU on stellar-m01g4 holds ~1800 MeshBlocks with
  this setup (32^3 blocks, nghost = 2, 6 hydro vars with dual energy, rk2, LAT). The
  doubled-box run (1 rank) died at 1786 blocks, cycle ~2950, t ≈ 3560: Kokkos failed to
  allocate 3.804 GiB for "cons" while growing for the next AMR event
  (= 6 x 36^3 x 8 B x ~1824 blocks). The old max_nmb_per_rank = 2000 was above this, so
  the AMR memory cap never fired; now 1500 (~15% headroom). Stellar GPU nodes: 2 GPUs per
  node, 40 GB each (RC docs), so ~22 MB/block (~10 copies of cons). One full node (2 ranks)
  ≈ 3000 blocks at the cap.
  Block count was ~1600 at cycle ~1000 (vs 456 for the BH nest; the rest is probably the
  two dx = 0.5 injection boxes plus their 2:1 buffers, unconfirmed) and grew ~30 per AMR
  check (every 128 cycles) as the streams advanced. Stream growth is bounded: the streams
  only exist up to the collision (straight-line estimate from the deck x/y/xdot/ydot:
  paths cross near (-1052, -508), r ≈ 1170, with stream 1 arriving at t ≈ 4560 and
  stream 2 at t ≈ 4020; gravity bends this by only ~10 r_g), so the OOM at t ≈ 3560 was
  just before the collision.
  After that the block count is set by how much CIO gas exceeds amr_rho_min (refined to
  dx = 0.5), which is not yet known: watch the block count after t ≈ 4500 and decide
  between more GPUs, a higher amr_rho_min, or a radius-dependent AMR level cap.
- **Density AMR (implemented):** refinement = adaptive with a user criterion
  (`InjectionRefinement`). Blocks anywhere with max density > `amr_rho_min` go to
  `amr_level` (dx = 0.5).
  **Injection box removed (2026-11-09):** AMR used to be restricted to an injection box
  (`<problem>/inj_x{1,2,3}{min,max}`); redundant, since stream-like densities only occur
  near the injection points and near the BH (already SMR-refined), so it now acts
  everywhere. Watch for dense CIO/envelope gas outside ±32 pulling large regions to dx = 0.5
  (raise amr_rho_min if so).
  (A dx = 0.25 collision-site SMR box was tried and later removed for cost.)
  Under AMR, `<refined_regionN>` only builds the initial tree and nothing protects it from
  derefinement, so the user criterion reads every refined_region back and enforces it as a
  minimum level. Target level = max(overlapped refined_region levels, amr_level if dense); flag +1 below target, -1 above (core handles sibling/2:1 rules).
  Single threshold, no hysteresis: add a derefine threshold if blocks flap.
- **LAT:** `lat_same_level = true` (ratio 2), so injection-box blocks are not slaved to the
  hot BH shell on the same level.
- **Units:** `<units>` from ref/athinput.tde_si (M_BH = 1e6 Msun with Msun = 1.9891e33;
  r_g = 1.4770934e11 cm; r_g/c = 4.9270606 s). mdot = 1.5623606789395503e-13 = 1 Msun/yr
  (365-day yr) in these units. No mu: temperatures are radiation temperatures.
- **Radiation dominated:** gamma = 4/3 represents radiation-pressure-dominated gas, so a
  temperature T [K] maps to internal energy density e = a_rad T^4 (P = e/3), not k T/(mu m_u).
- **Stream injection (implemented; geometry, mdot and duty cycle superseded 2026-11-10 by
  the time-dependent streams above):** per stream (`nstreams`, keys x/y/xdot/ydot/mdot/temp/H
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
- **Injection duty cycle (implemented 2026-11-08; REMOVED 2026-11-10, replaced by the
  table's state column):** `<problem>/inj_t_on`, `inj_t_off`
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
  Not yet tried: higher floors.
- **Bad cells at the stream collision (2026-11-11, open):** test runs show an expanding
  sphere of bad cells that starts exactly where the inbound and outbound streams first
  collide (a few hundred r_g/c after start_time, since injection is ~30 r_g upstream of
  the SI point). Leading hypothesis: the Mach ~300 cold-stream collision makes PLM/HLLC
  produce negative pressure/density at the shock, floors patch it, and the damage spreads
  with the CIO. Trial: `<hydro>/fofc = true` (first-order LLF fluxes on cells whose stage
  update would need a floor; supported under LAT and dual energy). If bad cells persist,
  next suspects: the dual-energy shock rule at the collision (compare u_int vs E - K;
  try dual_energy = false), then LAT (lat = false), then rsolver = hlle. Check which
  injection.log counters fire at onset.
  **fofc did not fix it.** User then saw that the collision sits right at the corner of a
  refinement boundary. Likely why: region8 (initial_only) refines the SI point at t0,
  but nothing dense is there until the streams arrive, so it derefines in the first
  windows; when the streams arrive, density AMR rebuilds only one level per check.
  **Collision box (implemented 2026-11-11, user decision to re-add refinement there):**
  `si_box` = SI point (xc, yc) at t and at t + inj_lead, padded by `<problem>/si_pad`
  (16 = one dx = 0.5 block) in x, y and z, enforced at amr_level under the same
  condition as the injection boxes (inj_box_on). Supersedes the reasoning under
  "Injection boxes" that the SI point needs no forcing. Cost: ~3x3x2 blocks plus 2:1
  buffers. fofc = true left on. Did not fix it either.
  **Log + front speed (2026-11-11):** bad cells are NaNs; the front moves ~1 MeshBlock
  per output frame. Under LAT outputs (and the event log) are written only at window
  ends, so a frame is one window (~100 r_g/c): ~0.16c, not necessarily non-physical
  (first read as "per step"; corrected). injection.log (lines every 256 cycles = one LAT window):
  cycle 768 dfloor 4.8e7, vceil 4.5e7, fofc 7.6e7; 1024 vceil 3.4e8, fofc 2.9e8;
  1280 all ~0 (NaN cells fail every floor comparison, so the counters go silent).
  Reading: massive dfloor/vceil hits that FOFC's first-order LLF (positivity preserving
  within CFL) cannot prevent, then NaNs spreading block by block, point to a CFL
  violation. Under LAT the time step is set only at window ends (wiki: "outputs,
  restarts, AMR and the time step run at window ends") and each block's factor is
  frozen for the window, so gas that gets fast mid-window (CIO front leaving the pinned
  blocks; caps allow |v| + c_s up to 1c) overruns its block's step. Hypothesis, to be
  confirmed with lat = false.
  **Confirmed (2026-11-11):** with lat = false the collision produces no NaNs, so LAT is
  the cause.
  **Fix (implemented 2026-11-11, user decision):** LAT back on with
  `<time>/lat_pin_density_contrast = 1e3` (built in: a block whose own max/min density
  >= 1e3, plus its neighbours, gets f = 1; follows the CIO/background front wherever it
  goes) and `lat_levels = 6` (window = max factor = 64 fine ticks, was 256, so the front
  moves a fraction of a block per window and stays inside the pinned ring). Rejected for
  now: lowering lat_pin_density to ~1e-20 (pins the whole CIO volume; fallback if the
  contrast pin is not enough, or lower the contrast to 1e2), a collision-box LAT cap
  (fixes only the onset). Side effects: coarse blocks step 4x more often; AMR checks
  (ncycle_check = 128) now land every 2nd window; streams update 4x more often.
  **Worked (2026-11-11):** test run with LAT on gets through the collision without NaNs.
  Still to check: cost per r_g/c vs the old lat_levels = 8 setup, and whether the
  contrast pin grows expensive as the CIO spreads.
  Lesson (general rule, extends the 2026-11-07 one): LAT freezes the fine tick and each
  block's factor for a whole window, so any gas that becomes fast mid-window outside the
  pinned blocks blows up; pins must follow such fronts, and windows must be short enough
  that a front cannot cross the pinned neighbour ring in one window.
- **tinj = 2000 (2026-11-11, user decision; user regenerates data/inj.ascii):** injection
  moves further upstream of the SI point. Code change: the collision box now has its own
  switch, `si_box_on` = table on at t - tinj, t or t + inj_lead (endpoint test is enough
  since phases >> tinj + inj_lead). Before, it shared inj_box_on and dropped at each off
  switch although gas injected just before the switch collides for another tinj (a latent
  gap at tinj = 1000 too); since the entropy cut no longer refines shocked gas, the box is
  what keeps the collision at amr_level. New deck key `<problem>/tinj` (required, must
  match the table). Deck `<refined_region8>` (initial_only) must be updated by hand to
  cover the new t0 cylinders: at t0 the table is on, so the look-ahead cannot pre-refine
  them, and V_cyl assumes they sit on amr_level.
  Done (2026-11-11): new table has t0 cylinders at (-922.8, -189.1) in / (-810.3, -164.2)
  out, SI at (-861.5, -191.3) (~60 r_g upstream, was ~30). region8 = x [-944, -800],
  y [-208, -144], z [-16, 16]: the union of the pgen's t0 injection boxes (incl. look-ahead)
  and collision box, rounded out to 16 r_g block edges (72 blocks, was 36).
- **CIO fallback cost (2026-11-11):** at t ≈ 1.42e5 the run was at ~2650 blocks (88% of
  2 x 1500) and ~0.84 s wall per r_g/c (about 2x the average since t0; ~18 days to
  tlim = 2e6). The falling-back CIO exceeded amr_rho_min, so it was refined to dx = 0.5, and
  since lat_pin_density = amr_rho_min it was also pinned to f = 1. LAT bins per rank were
  f1/f2/f4/f8 = 1071/175/58/23, so f1 did 91% of the work and LAT saved only ~11%.
  **Stream-only AMR (implemented 2026-11-11, user decision):** the density criterion now
  also requires low entropy: a cell counts if rho > amr_rho_min and
  P/rho^gamma < `<problem>/amr_entropy_fac` (deck: 100) x K_s, where K_s is the larger of
  the two streams' injected entropies (gm1 a_rad T^4 / rho_s^gamma, from the current
  window's mdot). Entropy was chosen over temperature (user choice): rough estimate is
  that the collision shock (~0.06c relative) puts the CIO at ~1e4 K_s, and that survives
  adiabatic expansion/fallback, whereas the radiation T of CIO expanded to rho ~ 1e-16 to
  1e-15 is only ~2x the stream's 1e5 K. The CIO now sits at whatever level the SMR nest /
  SI box give it (dx ~ r/64 in the BH nest). Unverified: that streams stay below the cut
  (numerical heating, efloor hits) and that the CIO is above it; check with a K/K_s slice.
  Not yet changed: lat_pin_density = 1e-16 still pins the (now coarser) CIO blocks at f = 1;
  raising it toward the stream density is the next lever (option 1 in that analysis).
- **3e6 Msun deck (2026-11-11):** `inputs/injection_3e6.athinput` with table
  `data/inj_3e6.ascii` (header rp = 11.33, a_mb = 462.4, tinj = 1154.7 = 2000/sqrt3). The pgen
  is mass-agnostic (G = M = c = 1; only temp -> a_rad T^4 uses `<units>`), so only the deck
  changes. Table: t0 = 6.41e4; SI at r = 204 -> 255 by t = 1.15e6; injection points at
  r ~ 291 (in) and ~86 (out); on/off phases only ~3300 r_g/c (~130 rows), vs ~3.6e4 at 1e6.
  Changes (my choices, flagged to the user): units x3; r_in = 16 -> 8 (rp/r_in = 1.42 as at
  1e6; 16 finest cells in radius, no dx = 0.25 level); thresholds 1e-16 -> 1e-17 (min stream
  density 1.06e-16, outbound at t0); inj_lead 1500 -> 750 (sites on levels 3-5, and phases
  must exceed tinj + inj_lead); tlim = 2e6/sqrt3 (same in fallback-time units); region8 split
  into three initial_only boxes (regions 8-10: in cylinder, SI, out cylinder; 34 blocks vs
  196 for one bounding box). Kept: mesh/nest, rho_atm/bern (P_bg/P_stream ~ 3e-5 to 2e-4),
  H = 5, temp = 1e5 K, vceil (TR13 free fall at r = 8 ~ 0.38). Stream Mach is ~20 at t0,
  ~70-90 later (vs ~300 at 1e6).
  **Revised (2026-11-11, user decisions 1-4):** (1) dx = 0.25 inner cube (|x| <= 16), so
  r_in = 8 is 32 cells; (2) temp lowered to keep the stream cold; (3) H = 1.67 (fixed
  physical width, 5/3); (4) box 4096^3 (±2048, root dx0 = 32; nest shifts one level down,
  level = 11 - log2(r), still 456 BH blocks, num_levels = 8).
  Follow-ons (my choices): amr_level = 7 is now dx = 0.25 (streams ~6.7 cells per H; dx = 0.5
  would give 3.3); blocks are 8 r_g, so si_pad = 8 and regions 8-10 recomputed on 8 r_g edges
  (34 blocks). Smaller H raises rho_s 9x (min 9.5e-16, outbound at t0), so
  amr_rho_min = lat_pin_density = 1e-16. temp = 4e4 K: criterion c_s tinj / H as at 1e6
  (~0.3 for the outbound stream at t0; 1e5 K gave 1.7); Mach ~350 at t0. P_stream drops 39x
  (1.05e-22), so rho_atm 1e-22 -> 1e-24 to keep P_bg/P_stream ~1e-5 to 6e-5. inj_lead = 750
  kept (sites on levels 2-4, <= 5 checks ~115 r_g/c). Cost: fine tick ~0.18 (dx = 0.25 at
  r_in), about half the dx = 0.5 value.
  **tinj = 384.9 (2026-11-11, user regenerated the table):** same t0, SI point, mdot and
  phases; injection points now r ~ 236 (in) / 170 (out), speeds 0.080 / 0.098. Deck: tinj,
  regions 8 and 10 recomputed (SI region 9 unchanged); min stream density 1.4e-15 (outbound
  at t0), thresholds kept at 1e-16. temp = 4e4 K kept: thermal spread over tinj is now only
  ~0.07 H (criterion 0.3 H would allow ~8e4 K); Mach ~300 at t0. P_bg/P_stream ~1e-5 to 2e-5.
- **3e5 Msun deck (2026-11-12):** `inputs/injection_3e5.athinput`, table `data/inj_3e5.ascii`
  (rp = 52.57, a_mb = 996.2, tinj = 20000). Table: t0 = 2.990e5; SI moves out fast
  (r = 1939 at t0, 2862 at 5.1e5, 5192 at 1.29e6, ~9000 at 3.7e6); phases ~2e5-5e5;
  outbound stream near apocentre (|v| = 0.0069 at t0 vs 0.0144 inbound); mdot_out starts
  50x below mdot_in and overtakes it by ~8e5. The injection sites drift at ~4e-3 c, ~0.6x the
  outbound gas speed (mostly sideways), so the outbound stream will be smeared below
  rho_s = mdot/(pi H^2 v) (~10% at 1e6).
  **tinj:** the table generator's M^-3/2 scaling gave 12172, which put the outbound injection
  only ~4 H from the SI point. I recommended ~3e4; user chose 2e4 (in 330 r_g = 19.8 H,
  out 111 r_g = 6.7 H at t0; the outbound distance grows as v_out rises).
  User decisions: box ±16384 (tlim = 2e6 sqrt(10/3) = 3.65e6, r_si/half-width 0.55 at tlim
  as at 1e6), r_in = 32 (rp/r_in = 1.64), amr_level dx = 2 (H = 16.67, 8.3 cells),
  temp = 8e4 K (spread 0.22 H over tinj, outbound Mach ~37 at t0), bern = 0.003.
  Follow-ons (my choices): root dx0 = 256, BH nest ±2^(14-l), l = 1..8 (dx = 1 inside ±64,
  r_in = 32 cells), 512 BH blocks, num_levels = 9, amr_level = 7; rho_atm = 1e-23 so
  P_bg/P_stream ~1e-5 (as the 3e6 deck); thresholds stay 1e-16 (min rho_s 6.4e-16,
  outbound at t0); inj_lead = 1e4 (up to 7 AMR checks as the site moves from level 3 to
  root); inj_pad = 8; si_pad = 64 (one block); initial_only regions 9 (inbound) and 10
  (SI + outbound, merged) written out to the 128 r_g parent blocks build_tree refines
  anyway (16 + 96 blocks); outputs hst 300, 2D 1000, 3D 1e5, rst 2e4.
  **bern caveat:** 0.003 lowers the background c_s at the sites from 0.059 to 0.034, but that
  is still above the stream speeds (0.007-0.014); even bern = 0 gives 0.012-0.013, so no
  hydrostatic gamma = 4/3 atmosphere is slower than the near-apocentre outbound stream.
  Probably harmless: the injection blocks are pinned to f = 1 anyway (fine tick set by the
  BH), and coarse background blocks hit the max LAT factor regardless. Not yet run.
- **Mass history columns (implemented, 2026-11-09):** user history (enrolled by the pgen,
  no deck key) appends `M_inj` and `M_exc` to the .hst.
  M_inj = mdot_tot x duty-cycle on-time, computed analytically (exact because q uses the
  discrete cylinder volume; restart-safe since it is a function of t). (2026-11-10: now a
  per-window ledger, restarted from `<problem>/m_inj`; see time-dependent streams.)
  M_exc = net mass removed by `ExcisionFixup` (sum of (rho - rho_exc) dV over reset cells).
  A reset in RK stage s only survives into the step scaled by the later stages' gam0
  (rk2 stage 1: 0.5), so counting every reset at weight 1 would overcount by ~50%.
  **Core change:** `Hydro::HydroStateFixup` sets `pgen->user_hydro_state_fixup_weight`
  (prod of gam0 over later stages; 1 on all other call paths: LAT refresh, reflux
  correction, init), new member in `pgen.hpp`. Rank-local sum, MPI-reduced in
  `user_hydro_lat_window_end_func` (so this assumes LAT is on), stored as `<problem>/m_exc`
  via `user_metadata_func` for restarts. Pre/post-stage fixup calls see already-reset
  cells and add ~0. Check: mass(t) - mass(0) - M_inj + M_exc should equal minus the
  outer-boundary outflow (plus floor mass).
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
