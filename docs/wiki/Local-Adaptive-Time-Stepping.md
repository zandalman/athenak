# Local Adaptive Time Stepping (LAT)

Local adaptive time stepping lets every MeshBlock advance with its own CFL-limited time step
instead of the global minimum. Each block's step is rounded down to a power-of-two multiple
$f = 1, 2, 4, \dots$ of the finest step, and a block with factor $f$ is updated only on every
$f$-th fine tick. Ghost zones across a time-step interface are filled by time interpolation,
and the flux mismatch across the interface is refluxed when the slower block finishes its
step, so the scheme stays conservative. Use it when the CFL limit varies by orders of
magnitude over a refined mesh (the neighbourhood of a black hole, a stellar core, a dense
stream in a large low-density box). It is switched on with `<time>/lat = true` and is how the
TDE production decks run.

The gain grows with how few blocks sit in the finest bin. The price is a boundary refresh on
every fine tick, a rank layout that must keep every bin busy, and a list of modules that LAT
refuses (see [What works with LAT](#what-works-with-lat)). AMR regrids, outputs, restarts,
gravity solves, the sink-particle operator and remaps happen only when every block is back at
a common time, at the end of a window.

## How it works

### Bins: one time step per block

Each block $m$ reports the largest step it can take on its own,

$$ \Delta t_m = \mathrm{CFL}\;\min\!\left(\Delta t^{\rm fluid}_m,\ \Delta t^{\rm src}_m\right), $$

where the source term is present only when there is one. It is folded in so that
a block in a slow bin never exceeds its own light-crossing limit. Given the fine tick
$\Delta t_{\rm fine}$, a block gets the factor

$$ f_m = \max\left\{\,2^k \le 2^{\texttt{lat\_levels}} \;:\; 2^k\,\Delta t_{\rm fine} \le \Delta t_m \right\}. $$

This raw ladder is then narrowed, in the order below, so that no block steps in a way the
scheme cannot handle. Every step can only lower a factor.

The tick itself is chosen, not inherited. With $D$ the mesh step (normally the smallest
$\Delta t_m$), any tick in $(D/2, D]$ is safe once the factors are taken against it, and a
slightly shorter one can promote a whole level: a level whose smallest $\Delta t_m$ falls a
hair short of $2^k D$ would otherwise drop one bin, and with one bin per level it takes every
block of the level with it. Light-speed transport makes this the common case, because $\hat c$
is the same on every level and a free-streaming or isotropic region gives exactly
$2^{\Delta\ell}$, so the tie is decided by rounding or by a small flux factor somewhere in the
level. The tick is therefore the candidate (a tick at which some level's factor doubles) that
minimises the block steps per unit time $\sum_m 1/(\min(f_m, W)\,\Delta t_{\rm fine})$,
modelled on the level-collapsed ladder with the minimum bin population and the 2:1 rule
below. $W$ is the window the driver really runs at that tick, cut by `gravity/solve_dt` and
by the sink bound (see Windows and ticks): a block whose factor exceeds $W$ steps once per
window, so a promotion that only lifts a level past $W$ buys nothing, and a shorter tick that
lengthens $W$ buys what it lengthens. The fastest
level keeps the factor it has at $D$, so its blocks never step beyond $D$ even where $D$ is
set by a limit the per-block list does not carry, and a gain within the tie tolerance keeps
the longer tick. After the narrowing the tick is recomputed as $\min(D, \min_{f_m > 1} \Delta t_m/f_m)$, so a promotion the narrowing undid gives the step back and no block exceeds its own
limit. Same-level LAT keeps $\Delta t_{\rm fine} = D$, and so do the last window before
`tlim` and the plain steps taken while a generator has paused LAT (the remap settle steps).

| Step | What it does | Controlled by |
| --- | --- | --- |
| Problem cap | The problem generator may force blocks to $f = 1$. The TDE and star-black-hole generators do this for the neighbourhood of the black hole (excision or softening radius, plus the distance the hole travels in one window). | the generator |
| Density pin | A block whose active cells span a large density contrast, or contain very dense gas, and all its neighbours, take $f = 1$. | `lat_pin_density_contrast`, `lat_pin_density` |
| Sink pin | Every block whose bounding box, grown by its own ghost band, touches a sink-particle pin sphere takes $f = 1$. | sink particles |
| One bin per AMR level | Unless `lat_same_level = true`, all blocks on one refinement level take that level's minimum factor. | `lat_same_level` |
| Minimum bin population | A bin with fewer than `lat_min_bin_count` blocks is merged into the next faster bin, repeatedly. | `lat_min_bin_count` |
| Neighbour limiter | Adjacent time levels are kept within a 2:1 staircase: a coarse block steps at most $2^{\Delta\ell}$ times slower than a finer neighbour, and a fine block never steps slower than its coarser neighbour. For same-level neighbours the allowed ratio is 2 next to low-factor blocks and relaxes to `lat_same_level_max_ratio` between blocks that are both in high bins. | `lat_neighbor_limiter`, `lat_same_level_max_ratio` |

Edge and corner neighbours take part in the limiter because the prolongation stencil reads
them. The largest surviving factor sets the length of the next window.

### Windows and ticks

A **window** is $F_{\rm sync} = \max_m f_m$ fine ticks long. The driver shortens it (halving)
until it fits below `tlim`, within the remaining `nlim` cycles, within `gravity/solve_dt` when
self-gravity is on, and within the distance a sink may move (`lat_window_motion_cells`). A
window is never cut short once started.

Inside a window the driver runs $F_{\rm sync}$ **ticks**. On tick $k$ the blocks with
$k \bmod f_m = 0$ are *due* and each advances over its full step $f_m \Delta t_{\rm fine}$.
Before a tick, every due block's ghost zones are brought to the tick time (restriction only
where a due coarser neighbour needs it, prolongation only into due blocks with a coarser
neighbour). The due bins then integrate with the second-order Runge–Kutta (SSPRK2) step: by
default the first stages of all due bins run together as one fused kernel with per-block
$\Delta t$ (the *union predictor*), and the correctors follow from the slowest to the fastest
bin so that fast blocks see completed slow data. The mesh time and the cycle counter advance
by one fine step per tick, so `nlim` counts fine ticks.

After the fused predictor one refresh brings every due block's ghosts to its own predictor endpoint
$t+F_r\,\Delta t_{\rm fine}$. A due sender sends $u_0$ to its own factor and the chord
$u(t)+(F_r/F_s)(u_0-u(t))$ to any other factor. A non-due sender interpolates its history at the
receiver's endpoint. Stage 2, its flux and EMF rounds and the bin-final exchange stay per factor. The
state rounds per union tick fall from one per factor plus one to two. The union refresh is refused, and
each corrector refreshes per factor, with a dynamic metric or Z4c, and a
moving hole. The one-pass union stage 2 is not implemented: its flux and EMF receive rule, the window-end
exchange and the per-block $\Delta t$ audit of stage 2 are open.

### Ghost zones across a bin boundary

A block that is not due holds its state at a different time than its due neighbour. It therefore
hands over not its stored array but a value interpolated to the receiver's time, using its own
bracket: the start-of-step state $u_0$, the end-of-step state $u_1$ and, when a stage-1 snapshot
exists, the stage-1 (forward-Euler) state $`u_*`$,

$$ u(\theta) = (1-\theta)\,u_0 + \theta(1-\theta)\,u_* + \theta^2\,u_1, \qquad \theta\in[0,1], $$

which is the dense output of the SSPRK2 step. Where no stage-1 snapshot exists, or the
polynomial would leave the range spanned by its endpoints, the plain chord
$u_0 + \theta(u_1 - u_0)$ is used instead, decided per cell over all components.

A sender reads its start state and its stage-1 snapshot on owned cells only, so both are stored
without ghost zones wherever the mesh allows it (`nx` at least `nghost`, `2 nghost` with
refinement): the start registers and the snapshots.

The cumulative EMF history of a block holds the owned edges on the surface of the block only: an edge along one
axis lies on a face when one of its two transverse node indices is at its minimum or maximum. The
endpoint exchanges read and replace those edges alone, and the mismatch of every other edge is the
history minus itself, exactly zero, so it is written as zero and constrained transport and the
tail quadrature read what they read with a full history. At $48^3$ cells a block stores 27648
edges in the three directions instead of 345744, about 0.55 GB per card at 108 blocks for the
history and the moment together.

### Exchanges that are deferred

**The bin-final exchange.** At the end of its step a bin publishes its new state to its
neighbours' ghost zones. Unless the step is the bin's last in the window ($k + f_m < F_{\rm
sync}$), that exchange is redundant: before any block reads its ghosts again, the refresh of the
tick on which it is next due rewrites every one of them, and nothing in between reads a ghost
(the tick-end refluxing and recovery and the gravity bookkeeping work on owned cells; sinks,
outputs, restarts, AMR and the time step run at window ends). The state exchange is then
skipped (`time/lat_defer_final_exchange`) and the owned cells are still recovered. The flux
halves of the delayed refluxing are never skipped, and the last step of every bin in the window
keeps its exchange, so a window end sees the full state. The deferral is derived from the run and holds with Newtonian hydro dual energy: the
owner settles the energy channel of its own cells before the state leaves the block, on the
deferred path too, the recovery of an owned cell reads no neighbour, and the auxiliary internal
energy reaches the ghosts with the next refresh like every other variable. It is not used with
MHD dual energy, whose $\eta_2$ test compares a cell's thermal energy with the largest total
energy among its neighbours, ghosts included, so the owned cells next to a block face depend on
the delivered ghosts. It is also not used with a user boundary function on a non-periodic mesh
that may write owned cells from ghosts; a function whose owned writes read owned cells alone
(the torus boundary, which inverts the first active layer of its faces) still runs on the
deferred path, so the owned cells leave the bin as on the full path. On SMR and AMR a bin holding a
block on a user boundary face keeps its bin-final exchange instead: the next refresh's coarse user-boundary
fill restricts the ghosts that the boundary function wrote at the bin end, from the first layer's
transverse ghosts. Uniform grids keep the deferral.

**The wait on the refresh's own sends.** A boundary refresh ends with its last kernel queued.
It does not wait there for the sends it started: hydro latches the wait on its boundary
object, and the next pack, wait, request release or buffer reallocation on that object
completes it first Only the moment the
host blocks moves, so the results are the same bit for bit.

### Conservation: delayed refluxing

Every block accumulates the time integral of its own face flux across each time-step
interface and the faster neighbour sends the integral of its own fluxes. When the slower
block completes its step, the signed difference is applied to its boundary cells. This is the
Berger–Colella refluxing generalised from AMR coarse/fine faces to arbitrary bin boundaries,
including faces between same-level blocks with different factors. The correction is applied
for the gas and for the face-centred field (the constrained-transport EMFs). The gravitational work on the mass that the
correction moves is refunded too, for self-gravity, the analytic black hole and sink
particles.

### Where the ranks fit in

Under LAT the load-balance goal is not "equal cost per rank" but "every rank busy on every
tick". A block's cost is its steps per window times a measured work per step (1). The block order is
interleaved so that each rank owns a slice of every bin, and the partition search minimises the
critical path summed over ticks. On a static mesh no rank may hold more than
$\lceil N/N_{\rm rank}\rceil$ blocks, so only the choice of blocks is free. With AMR,
`mesh_refinement/max_nmb_per_rank` is the hard cap, and the search keeps every rank within
$\max(8, 5\%)$ of the mean block count, since device memory grows with it. The seeds of the search are improved independently, so each distinct seed is improved on one rank and the cuts are gathered, which gives the partition a serial search would. A migration-only
rebalance may run at a window end on a refined mesh (SMR or AMR), once the mesh has been
unchanged for four windows' worth of cycles and has changed since the last attempt, and only
if the current partition is not already balanced. The layout a run starts on is not a LAT
partition (a cold start uses the plain Z-order layout, a restart that does not keep its
checkpoint's partition is cut by checkpoint cost), so it is repartitioned at the first window
end whose ladder has more than one bin, without the wait. A static mesh therefore gets at
most one attempt after a restart.

## Deck parameters

All `<time>` keys are read only when LAT is on.

| Key | Default | Meaning |
| --- | --- | --- |
| `time/lat` | off | Master switch. |
| `time/lat_levels` | 1 (clamped to 1–20) | Number of factor doublings: the largest factor is $2^{\texttt{lat\_levels}}$. This is a ceiling; the factor a window actually reaches is set by the limiters and the minimum bin population. |
| `time/lat_same_level` | `false` | Allow blocks of the same refinement level to have different factors. When `false`, each level collapses to one bin. See [Same-level bins](#same-level-bins) for what each module does with it. |
| `time/lat_same_level_max_ratio` | 2 with `lat_same_level`, else 1 | Largest same-level factor ratio: 1, 2, 4 or 8 (anything else is fatal). Ratio 1 holds every same-level neighbour to one factor, which undoes `lat_same_level`. |
| `time/lat_neighbor_limiter` | `all` | Which same-level neighbours the ratio applies to. `all`: faces, edges and corners, ratio 2 next to factors 1 and 2 and up to `lat_same_level_max_ratio` between higher factors. `hybrid`: the same on edges and corners, but faces always 2:1 (so it is the stricter of the two). `face`: faces only, edges and corners unconstrained; refused with self-gravity or the analytic black hole. AMR neighbours are always limited on all 26 sides. |
| `time/lat_min_bin_count` | −1 | Smallest useful bin population. −1 means $4N_{\rm rank}$; 0 disables the merge. **With the default the trajectory depends on the rank count**; rank 0 prints a warning. Set it explicitly to reproduce a run on a different number of ranks. |
| `time/lat_pin_density_contrast` | 0 (off) | Pin blocks (and neighbours) to factor 1 when max/min of the active-cell conserved density is at least this value. Active only above 1. |
| `time/lat_pin_density` | 0 (off) | Pin blocks (and neighbours) whose maximum conserved density reaches this value, in code units. Active only above 0. Both pin thresholds must be finite. |
| `time/lat_diagnostics` | `false` | Print the bin table after each narrowing step and the same-level / AMR-interface status lines. |
| `time/lat_union_stage1` | `auto` | `auto`, `true` or `false`: fuse the first stage of all due bins into one kernel. `auto` and `true` both enable it only when the integrator and the generator's hooks allow it; `false` forces bin-by-bin stepping. |
| `time/lat_gid_reorder` | `true` | Interleave the bins over ranks (the layout described above). |
| `time/lat_post_amr_rebalance` | value of `lat_gid_reorder` | Allow the migration-only rebalance at a window end. `true` with `gid_reorder = false` is fatal. |
| `time/lat_defer_final_exchange` | derived | Skip the redundant bin-final boundary exchange ([Exchanges that are deferred](#exchanges-that-are-deferred)). Derived from the run's capabilities; an explicit `false` vetoes the derived choice, an explicit `true` never overrides a correctness condition. |
| `mesh_refinement/max_nmb_per_rank` | required with AMR | Per-rank MeshBlock cap and the bound on GPU memory; the LAT partition further keeps every rank within $\max(8, 5\%)$ of the mean count. On a static mesh it is not used. |
| `mesh_refinement/sticky_load_balance` | `false` | Keep block ownership across a regrid when no LAT rebalance is active. Ignored (with a message) when LAT is off. |
| `gravity/solve_dt` | none | **Required and positive** with self-gravity: the potential is frozen inside a window, so this bounds the window length. |
| `sink_particles/lat_window_motion_cells` | 1.0 | How many finest cells (times the CFL number) a sink may move in one window. Must lie in $[1, \min(n_{\rm grow},\,(n_{\rm grow}-\tfrac12)/\mathrm{CFL})]$ with $n_{\rm grow}$ = `accrete_radius_cells`, otherwise the run stops. |

**Legacy spellings.** Every LAT key is read under its `lat_*` name.  The historical `hydro_lat_*`
spellings (`hydro_lat`, `hydro_lat_levels`, `hydro_lat_same_level`,
`hydro_lat_same_level_max_ratio`, `hydro_lat_neighbor_limiter`, `hydro_lat_diagnostics`,
`hydro_lat_union_stage1`, `hydro_lat_min_bin_count`, `hydro_lat_gid_reorder`,
`hydro_lat_post_amr_rebalance`, `hydro_lat_defer_final_exchange`) are accepted as aliases, in a
deck and on the command line, so old decks and checkpoints keep working; new decks use `lat_*`.
`time/hydro_lat_max_nmb_per_rank` is read by no code path and is a fatal error in a deck (use
`mesh_refinement/max_nmb_per_rank`).
Under LAT the driver overwrites `time/hydro_subcycle` and `time/hydro_subcycle_factor`
itself (factor $2^{\texttt{lat\_levels}}$).

### Choosing values

- **`lat_levels`** is a ceiling. Pick it a little above the ratio of the slowest to the fastest
  block you expect ($2^7$–$`2^{10}`$ in the production decks); a larger value costs nothing when
  the limiters do not let the bins reach it. The realised window can be far below the ceiling
  (64 ticks against a ceiling of 512 in one production run).
- **`lat_same_level`** buys extra speed by letting neighbours at the same level differ, at the
  price of more interfaces to interpolate and reflux. Keep
  `lat_same_level_max_ratio = 2` (the default with it) unless a larger ratio has been validated
  for the deck.
- **`lat_min_bin_count`**: set it explicitly (the production decks use 40) whenever you
  want results to be comparable across rank counts.
- **`max_nmb_per_rank`** under AMR: set it to what GPU memory allows; LAT will never put more
  than this on a rank.

## Density pinning at a stellar surface

The delayed reflux across a sharp density transition can briefly expose a dense state to
gravity, floors and radiation before the mass is refunded. Pinning the blocks around such a
transition to factor 1 avoids it:

```ini
<time>
lat = true
lat_pin_density_contrast = 1.0e4
lat_pin_density = 0.0
```

A block is selected when its maximum active-cell conserved density reaches the threshold
multiple of its minimum (a non-positive minimum with a positive maximum also qualifies), or
when it exceeds `lat_pin_density`. Selected blocks and all face, edge and corner neighbours
use factor 1, recomputed at every metadata rebuild; ghost cells are excluded. With
`lat_same_level = false` the level collapse may then pin an entire refinement level and cut the
speedup. The pin does not guarantee the absence of hot dilute gas from other mechanisms.

## A minimal example

```ini
<time>
evolution  = dynamic
integrator = rk2              # rk1, rk2 or imex2 only
cfl_number = 0.3
tlim       = 30000.0
lat        = true
lat_levels = 9
lat_union_stage1 = auto
lat_min_bin_count = 40  # fixed, so the run is reproducible across rank counts

<gravity>                     # only with self-gravity
solve_dt = 0.03               # required with LAT
```

Decks that use LAT: `inputs/TDE_examples/tde_05_fallback_lat.athinput` (remap, settle steps
and fallback in one run) and the small gates in `inputs/LAT/`.

## Outputs and diagnostics

On rank 0, at start-up:

- `Driver: LAT bin-final exchange deferred` or `not deferred (<reason>)`, with `(not in a bin holding a block
  on a user face)` when a bin with a user-face block keeps its exchange, and then either a union corrector
  refresh or a per-factor corrector refresh with its reason.
- `Driver: LAT union stage-1 predictor enabled` or `disabled (<reason>)` – the reason names
  which capability refused the fused predictor; the fallback to bin-by-bin stepping can be a
  large slowdown, so check this line when two decks differ in speed.
- Lines for sink particles, naming the cadence in use.

At every load balance:

```
Mesh: HD LAT per-bin rank distribution f1=[a0,a1,...] f2=[b0,b1,...] ... util%=[u0,u1,...]
```

For each populated bin `fN`, the number of factor-$`N`$ blocks on each rank; `util%` is each
rank's predicted busy fraction over a window. Compare it with measured GPU utilisation.

With `lat_diagnostics = true` the bin populations (`f1=… f2=…`) are printed after each
narrowing step (raw local-CFL, problem-capped, density-pinned, sink-pinned, …), together with
the same-level and AMR-interface status of the neighbour limiter.

## What works with LAT

LAT is available for **hydro and GR hydro only**; MHD and GRMHD decks with LAT are refused at
start-up. LAT needs exactly one fluid and an `rk1`, `rk2` or
`imex2` integrator (the delayed-reflux weights are derived for those; `imex3` is not
analysed). Everything the driver cannot make aware of per-block time levels is refused at
start-up with a message naming the module.

| Module | Status |
| --- | --- |
| Hydro, GRHD | supported; GR hydro runs under LAT with `gr_torus` (its no-inflow boundary is LAT-mask aware) |
| MHD, GRMHD | refused at start-up in this release |
| Dual energy (Newtonian) | supported; the bin-final exchange is deferred as without it ([Exchanges that are deferred](#exchanges-that-are-deferred)) |
| Self-gravity and analytic external black hole | supported, hydro; potential frozen over a window, needs `gravity/solve_dt > 0`; see [Multigrid self-gravity](Multigrid-Self-Gravity) |
| Sink particles | supported; operator runs at window boundaries, nearby blocks pinned to $f=1$ |
| Dynamical GR | only a *prescribed* analytic metric flagged LAT-safe; `prolong_primitives = true` refused (the fluid fills its coarse/fine ghosts from the primitives by itself) |
| Remap | startup remap and settle passes work; see [Remapping](Remapping) |
| FOFC | supported (first-order flux correction runs under the active-block list) |
| Viscosity, conduction, resistivity | refused |
| Orbital advection, shearing box | refused |
| Source terms `const_accel`, `ism_cooling`, `rel_cooling`, `rad_beam`; with hydro also `disk_cooling` | refused |
| Turbulence driving | refused |
| Evolved spacetime (Z4c), legacy `<radiation>`, ion-neutral, particles | refused |
| User source terms, time-step limits, boundary conditions | refused unless the generator marks them LAT-safe; hydro generators with user boundaries need `user_bcs_lat_safe` |

### Design rules behind the refusals

The restrictions follow from a few rules that any new operator must respect:

- AMR, outputs, restarts, gravity solves, the sink operator, remaps and callbacks run only on
  synchronized states, never on a partially advanced mesh; per-block time steps are recomputed
  only there too.
- A state-changing operator must act on the active blocks only and use the per-block step and
  the driver-provided stage time. Nothing that issues an MPI collective may run inside a
  rank-local bin: one rank with no due block would hang the others.
- State that depends on the global gas (the TDE translating frame and live black hole) is
  advanced once per window by a window hook that runs on every rank at the common start time,
  not per bin.
- A generator marks its source terms, time-step limits or boundary conditions LAT-safe only
  when they obey the rules above.
- If same-level LAT is extended, the ghost interpolation and the delayed reflux must change
  together.

## Same-level bins

With `lat_same_level = true` a face between two blocks of one level can separate two factors
$f < F$. The machinery is the coarse/fine one with the restriction removed: the slower block
reads the faster one's ghosts at its own stage times (a pending faster sender is extrapolated
along its Euler predictor), the faster block reads the slower one's time-interpolated ghosts,
and at the slower block's endpoint the time integral of the faster block's face flux replaces
its own (`MeshBlockPack` masks `lat_flux_accum/send/recv_nghbr`, `iflux_same` buffers). At a
common time both sides reconstruct the same states, so there is no spatial flux mismatch and no
pending-flux estimate on same-level faces. Edges and corners exchange no flux; for constrained
transport the edge EMF is reconciled to the fastest of the up to four blocks around it. The
table is the state of the code.

| Module / feature | Same-level status | Where |
| --- | --- | --- |
| Hydro reflux, Newtonian dual-energy $\nabla\cdot v$ reflux, GR gas entropy column | supported (the column takes the corrected mass at the cell's own $s$, [Entropy fallback](Entropy-Fallback#transport)) | `hydro_update.cpp` Reset/Accumulate/ApplyLATFluxCorrection, gates `multilevel \|\| lat_same_level` |
| Ghost time interpolation | supported; the SSPRK2 dense polynomial is used for hydro only without self-gravity or the analytic black hole, otherwise the linear chord (second order in time, as Berger–Oliger) | `hydro.cpp` (`ConfigureLATDenseOutput`, `lat_dense_output_enabled`) |
| Self-gravity and analytic black hole | supported; potential solved at window starts only; the work of the refluxed mass is refunded with the gravity coupling weight of the face cell | `srcterms.cpp` (gate record), `hydro_update.cpp` |
| Sink particles | supported; pin to $f=1$ is per block, before the level collapse | `mesh.cpp`, `hydro_update.cpp` (sink work refund) |
| `tde_external` | supported; blocks around the black hole (and their 26 neighbours) are capped to $f=1$ per block | the generators' factor-cap hooks |
| AMR, load balance | regrids and the post-AMR rebalance at window ends only; the rebalance needs a refined mesh, so a uniform same-level mesh keeps its partition | `driver.cpp` (`RebalanceHydroLATMesh`) |
| Restart | accumulators are drained at every window end and checked at every checkpoint (hydro); bins are recomputed | `restart.cpp` (`CheckLATRefluxDrained`) |
| Outputs, history | written at window ends only | `driver.cpp` |
| `lat_neighbor_limiter = face` with self-gravity or the analytic black hole | refused | `driver.cpp` |

The production configuration of the TDE fallback (hydro, LTE table with radiation pressure,
dual energy, multigrid self-gravity, `tde_external` with the static analytic black hole,
`lat_same_level = true`, ratio 2, limiter `all`, AMR) uses only rows marked supported. Its
same-level ghosts use the linear chord because of the gravity source.

## Interactions

- **AMR/SMR.** A regrid runs only at a window end, and only when at least `ncycle_check` cycles
  have passed since the previous check; there is no constraint that `ncycle_check` be a
  multiple of the window length. After a regrid the factor of a new block is estimated from the
  old mesh (carried over for a surviving block, halved per refinement level relative to the
  nearest old ancestor for a new child, the minimum over children for a derefined parent), and
  on a cold start from the refinement level (finest level $f=1$, doubling per coarser level).
- **Self-gravity.** The Poisson equation is never solved inside a bin. The potential is solved
  at the start of a window and held fixed; if the potential is invalid, a solve is due, or the
  window would cross the next solve time, the solve is brought *forward* instead of shortening
  the window, and the solve times inside that window are consumed after any solve at its
  start, whatever made it due (one solve per
  `solve_dt`). The accuracy cost is a lag of less than `solve_dt` plus one window
  length, at most `2*solve_dt`; see the gravity page.
- **Restart.** LAT needs a single global restart file (`single_file_per_rank` is fatal).
  Checkpoints are written only at window ends (hydro runs check that every reflux
  accumulator is empty and stop otherwise), and they record the AMR cadence anchor so a
  restarted adaptive run reproduces the uninterrupted one. Bin assignment is not stored; it is
  recomputed from the local CFL, so with the default `lat_min_bin_count` a restart on
  another rank count is valid and conservative but a different trajectory. After an adaptive
  restart the block-wise state is bit for bit the uninterrupted run's, but the history file's
  global sums can differ in the last bit or two, because the block IDs are renumbered in Z-order
  and the sums are added in another order. Test a LAT binary by restarting it (`-r`), not only
  by cold starts.
- **Sink particles.** The sink operator (accretion, creation, merging, the gas–sink kicks) runs
  once per window. Blocks within a sink's pin sphere are forced to $f=1$, and the window is
  shortened so that a sink moves at most `lat_window_motion_cells` finest cells.
- **Remap.** The startup remap runs before the first window and the bins are built from the
  remapped state. A remap while LAT windows are running is fatal unless the generator has
  paused LAT; the remap's settle steps do exactly that.
- **FOFC and constrained transport.** FOFC, the face-field EMF refluxing and the order-mask
  exchange are all LAT-aware. Each block tests only its own cells, with its own time step and its
  own pending fine-flux estimate, and the neighbours import those flags in a one-variable exchange
  of the first ghost layer, so both sides of a shared face take the same order and an inactive
  block sends zeros.
- **Outputs.** History, dump and restart files are written at window ends only, so output
  times fall on window boundaries. A `dt` stream writes at the first window end past each
  due time, and a `dcycle` stream at the first window end at or past each multiple of
  `dcycle`.

## Limitations and known issues

- **Rank-dependent bins by default.** `lat_min_bin_count = -1` scales with the rank count,
  deliberately: no rank-independent constant is right at both ends of the range. Set the key.
- **A window cannot be truncated.** Leaving a window half-way would abandon pending reflux
  accumulators, so the driver sizes the window beforehand (`tlim`, `nlim`, wall clock) and
  treats a premature exit as fatal.
- **Rounding of block end times.** The mesh time at a window end is the sum of the fine steps,
  while a block's own end time is a partial sum plus $f\,\Delta t_{\rm fine}$; they differ by
  about $F_{\rm sync}\,\epsilon\,|t|$ (roughly $2\times10^{-7}$ of a fine tick in a
  production shape). Endpoints are selected by integer counts, so no reflux endpoint can be
  mis-selected; the difference would matter only if $\Delta t_{\rm fine}/|t| < 10^{-9}$.
- **Large same-level ratios** are a physics-validation risk: the ghost interpolation is dense
  SSPRK2 output between the start and end states, and the reflux assumes the two sides are
  consistent to that order. Keep `lat_same_level_max_ratio` at 2 unless validated.
- **Time-lagged gravity.** The potential is as old as the last solve (less than `solve_dt` plus
  one window length). In the Jeans-wave
  gate the measured growth rate is off by about 2% with `solve_dt = 0.007` and by 0.14% with
  `solve_dt = 2e-4`; the error follows `solve_dt`, so choose it small compared with the
  dynamical time of the gas that matters.
- **Cost.** What remains is communication and task launch overhead for the per-tick boundary
  refresh at large factors. The design favours correctness and synchronized AMR/gravity over
  asynchronous per-rank progress. A two-bin window runs 2.5 stage passes per tick (the fused
  predictor and one corrector per due bin) against 2 for a synchronous cycle, plus about 2.5
  boundary refreshes and the endpoint corrections, while the tick is the synchronous step. On
  a deck of a few small blocks per GPU, where a pass costs its exchange latency and not its
  arithmetic, LAT is therefore slower than the synchronous run whatever its ladder: it saves
  block steps, which cost little there, and adds passes. LAT pays where blocks are many per GPU and the step is arithmetic.
- **Sparse bins idle ranks.** If a bin has fewer blocks than ranks, some ranks have nothing to
  do on the ticks where only that bin is due. The interleaved layout and the default minimum
  bin population exist to limit this; the `util%` line shows what is left.

## Validation

The regression gates cover: LAT with one bin, or with every block pinned to $f=1$, reproducing
the run without LAT bitwise ; conservation of mass, momentum,
energy and face-field divergence on static-SMR and AMR meshes with mixed factors;
second-order temporal accuracy for the gas; an adaptive LAT run restarted from a checkpoint
matching the uninterrupted run; the density pin on one and two ranks; the frozen-potential Jeans waves for hydro; and rank independence of the hydro LAT gravity refund
(`tst/test_suite/nr/test_nr_hydro_lat_grav_refund_ranks_mpicpu.py`). The analysis harness
is `tst/analysis/lat_density_pin.py`, plus the `*_lat_*`
files in `tst/test_suite/`.

## See also

- [Multigrid self-gravity](Multigrid-Self-Gravity) – the frozen-potential contract and the
  window-end work refund.
- [Remapping](Remapping) – startup remap and settle steps with LAT.
- [TDE](TDE) – the production setup.

## Where in the code

- `src/driver/` – admissibility checks, the window and tick loop, the boundary refresh, the
  rebalance trigger.
- `src/mesh/mesh.cpp` – local-CFL gather, the factor ladder and all narrowing steps.
- `src/mesh/load_balance.cpp`, `src/mesh/mesh_refinement.cpp`, `src/mesh/build_tree.cpp` – bin-aware
  partition, interleaved block order, post-AMR redistribution, restart ordering.
- `src/mesh/meshblock_pack.cpp` – active, boundary-provider and flux-receiver masks and
  per-block time registers.
- `src/bvals/` – time-interpolated ghost exchange and the LAT flux-correction buffers.
- `src/hydro/`, `src/mhd/`, `src/dyn_grmhd/` – active-list kernels and the delayed
  corrections (`*_lat.cpp`, `hydro_update.cpp`).
