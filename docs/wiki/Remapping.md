# Remapping

The remap module resamples the state of an existing AthenaK restart file onto a *new* mesh – a
different domain, resolution, MeshBlock size, refinement pattern or rank count – at the start
of a fresh run, with no problem-generator code. It carries the cell-centred hydro or MHD
variables (with passive scalars, the dual-energy auxiliary and the GR gas entropy), rebuilds the face-centred
magnetic field **divergence-free to machine precision** from a vector potential, and moves the
angular radiation intensities between the two meshes. Time, time
step and cycle number continue from the source. Use it whenever a run must continue on a mesh
the checkpoint was not written for: widen or shrink the domain, change the MeshBlock size for
a different rank count, add or move refinement, or switch from a run without radiation to a
run with it (GR torus regridding, TDE restarts onto a wider box).

The transfer is a host-side, single-pass resampling at start-up. It is second-order accurate
but **not conservative**: see [Accuracy and conservation](#accuracy-and-conservation).

## Quick start

Add a `<remap>` block to the deck of the new (target) run and start it with `athena -i deck`
(never with `-r`: the module runs only on a fresh start):

```ini
<remap>
enable = true                    # keep the key explicit so the command line can flip it
source = rst/OldRun.00042.rst    # checkpoint written by the old mesh configuration
```

The problem generator runs first and builds its own initial state, then the module overwrites it
with the resampled source state. `<time>/nlim` and `tlim` in the target deck are **absolute**: the
run continues from the source's cycle and time, so `nlim` must be the source's final `ncycle`
plus the number of cycles you want.

## How it works

### Cell-centred gas

For each target active cell the module finds the containing source block (walking down the
source's refinement tree) and averages a few point samples of a **trilinear interpolant** of the
source cell centres. A target cell that coincides with a source cell (same width, same centre,
to $10^{-10}$ of a cell) is **copied** bit for bit, so a remap that only changes the MeshBlock
decomposition or widens the domain is the identity on the overlap. (Without this, the two
sub-samples per axis would act as the filter $[1/8, 3/4, 1/8]$ and a same-grid remap would erase
87.5% of the grid-scale gas power.)

**Thermal, not total, energy.** The gas engine interpolates $(\rho, m_i, e_{\rm int})$ and
rebuilds $E = e_{\rm int} + |m|^2/2\rho$ at the destination rather than interpolating $E$. The
reason is Jensen's inequality: the interpolated kinetic energy exceeds the kinetic energy of the
interpolated $(m, \rho)$ by the grid-scale kinetic-energy variance the target cell cannot
represent, and interpolating $E$ would hand exactly that variance to the thermal channel as
heat. On the TDE production dumps ($e_{\rm int}/{\rm KE}\sim10^{-3}$) interpolating $E$ raised the
global internal energy by 32% in one remap and the upper percentiles of $e_{\rm int}/\rho$ in the
low-density envelope by a factor 2.6–14, a spurious luminosity step. The source's thermal energy
is the one its dual-energy inversion reads, cell by cell: $E - |m|^2/2\rho$ where the source's own
$\eta_1$ test accepts that energy channel, and the dual-energy auxiliary where it does not (at steep
stellar edges the $\eta_2$ resynchronisation can leave the auxiliary behind the energy channel). Without
dual energy it is $\max(E - |m|^2/2\rho,\ \text{floor})$. A coincident cell keeps the source's own $E$
verbatim.

### What happens beyond the old domain: the band mode

Where the target extends past the source box, `band_mode` decides what fills the new region.

- **`floor` (the Newtonian default).** Every target cell is overwritten. A transition band just
  inside the old boundary reconstructs the interior profile in $\ln\rho$ and $\ln p$ and tapers
  it to the ambient hydro floor, and cells fully outside the old box become floor state.
- **`keep` (the default for GR).** The target's own problem-generator state *is* the ambient. A
  per-sample weight $w$ equals 1 everywhere the source covers (up to the last active cell) and
  decays to 0 by a quintic smoothstep over `b_taper_root_cells` source root cells outside the
  box; $w=0$ writes nothing, $0<w<1$ blends the conserved variables linearly, $w=1$ overwrites.
  The fade lives entirely *outside* the source box: a band inside a fully covered domain
  measurably resurrects initial-condition material (+11% total mass on a same-domain test).

`transition_band = false` turns the band off and gives plain interpolation (useful for
convergence tests). Floor mode is invalid for a GR source (there is no ambient floor state to
relax to) and is fatal.

### Face-centred magnetic field

The source $B$ faces are restricted exactly onto a uniform covering grid at the source **root**
level (area-weighted mean of the fine faces). The discrete curl is then inverted exactly in the
gauge $A_1 \equiv 0$ by prefix sums along $x$,

```math
 A_2(\cdot,i{+}1) = A_2(\cdot,i) + \Delta x\,B_3(\cdot,i), \qquad
   A_3(\cdot,i{+}1) = A_3(\cdot,i) - \Delta x\,B_2(\cdot,i),
```

(with the redundant $A_3$ seeded along the $i=0$ plane from $B_1$). By construction
$\nabla\times A$ reproduces $B_2$ and $B_3$ exactly and $B_1$ exactly if the source was discretely
divergence-free; the residual is measured, warned about above $10^{-8}$ of $\max|B|$ and fatal
above $10^{-2}$. Each component is interpolated off its own edge lattice by a separable
Catmull–Rom cubic, gauge-normalised by its boundary-shell mean, and tapered to zero over
`b_taper_root_cells` root cells outside the source box. The target field is the discrete curl,
per MeshBlock, of edge samples of this one global continuous potential (with fine-neighbour
edge averaging where a block abuts a finer one), so **every target block is divergence-free to
machine precision** independent of interpolation error. Excised horizon interiors are not
skipped: the discrete curl identity needs them.

In `keep` mode the field is overwritten only on faces whose curl stencil touches the taper
window; elsewhere the generator's field stands. Two divergence-free fields meeting face to face
are not divergence-free at the seam, so the outermost taper layer can carry
$\nabla\!\cdot\!B \sim |B_{\rm pgen}|/h$; the module measures and reports it (it is not
suppressed by `b_report`). A same-domain remap has no seam.

**Magnetic-energy bookkeeping (Newtonian).** The loader subtracts $\tfrac12|B|^2$ from the source
energy so that the gas engine moves gas energy only, and the field pass adds the new
$\tfrac12|B_{\rm new}|^2$ back, weighted by the same blend weight the gas pass used. In GR the
conserved column is remapped as stored and the $O(h^2)$ energy–field inconsistency is absorbed
by the primitive recovery, first-order flux correction and excision in the first cycles.

### GR runs

The module sorts every source into a *relativity class* from its parameter dump and requires the
target to be of the same class:

| Class | Source signature | Examples |
| --- | --- | --- |
| Newtonian | no relativistic `<coord>` keys | `tde_external`, `remap_test` |
| fixed-metric GR | `<coord> general_rel = true`, no `<adm>` | `gr_torus` |
| dynamical GR, analytic metric | `<adm>` block | |

A source or target with an evolved spacetime (`<z4c>`, `<cce>`) is refused permanently:
resampling a numerically evolved metric with an interpolant that knows nothing of the
Hamiltonian and momentum constraints yields constraint-violating data that looks fine. Special
relativity is refused too. Parameter dumps of source and target are compared and only warn
(fatal for a fixed-metric differing spin `a` or `minkowski`); `<units>` differences
(`bhmass_msun`, `density_cgs`, `mu`) warn in both classes.

On the dynamical-GR path every conserved column carries a factor $\sqrt\gamma$. Interpolating
the stored columns would put the metric's curvature inside the smoothing stencil, worst at the
punctures, so the module divides $\sqrt\gamma$ out at each source cell centre (the analytic
binary metric evaluated at the **source time**, since the punctures have moved since the
generator built its $t=0$ state), interpolates, and multiplies by $\sqrt\gamma$ at the target
centre. A stored (non-analytic) metric cannot be evaluated this way and falls back to
interpolating as stored, with a warning. The fixed-metric class does not densitize, so nothing
is divided out there.

Every GR fluid carries the gas entropy $Ds$ in the column after the passive scalars (the
entropy fallback of the primitive recovery). It is remapped like a scalar, so the first
recovery after the remap can fall back on it where the remapped energy has no root; outside
the source box it blends with the entropy of the target's own state. A GR checkpoint written
before the column existed is still accepted (the loader recognises it by its payload width,
as a restart does): the target then takes the column from the remapped state after its first
recovery. The banner says which (`gas entropy D s remapped` or `set from the remapped state`).

After the state is replaced, the module calls the metric callback at the source time, so the
lapse, shift and excision masks match the remapped state with
no generator code.

### Radiation

- **Angular intensities `i0`**: remapped when both sides carry `<radiation>` with identical
  `nlevel`, `rotate_geo` and `angular_fluxes` (a mismatch is fatal).
- A group present in the source but not in the target is skipped with a warning; the reverse is
  reported in the banner and left to the generator.

## Deck parameters

| Key | Default | Meaning |
| --- | --- | --- |
| `remap/enable` | `true` (when the block exists) | Master switch; can be flipped from the command line. |
| `remap/source` | `""` | Source restart path; empty is fatal. |
| `remap/copy_output_state` | `true` | Carry each `<outputN>`'s `file_number` and `last_time` forward so dumps continue numbering. |
| `remap/transition_band` | `true` | Reconstruct the boundary band; `false` is pure interpolation. |
| `remap/band_mode` | `auto` | `auto` (keep for GR, floor for Newtonian), `floor` or `keep`; case-insensitive. |
| `remap/b_coarsen_ok` | `false` | Accept a multi-level MHD source. $B$ is then restricted to the source root grid and loses sub-root structure; cell-centred fields keep full detail. |
| `remap/b_taper_root_cells` | 4 | Width, in source root cells, of the taper of $A$ and of the keep-mode weight outside the source box. |
| `remap/b_report` | `true` | Print the inverse-curl and boundary-field diagnostics. The sheet-current warning, the residual warning and the keep-mode seam report are always printed. |
| `remap/radiation_i0` | `true` | Remap `i0` when both sides carry `<radiation>`. |
| `remap/settle_steps` | 0 (`tde_external`: 20) | Plain steps after the startup remap before the state is remapped again onto the adapted mesh; must be $\ge0$. |
| `remap/settle_passes` | 1 | Number of settle passes, each preceded by `settle_steps`; must be $\ge0$. |
| `remap/inherit_tree` | `false` | Build the target MeshBlock tree from the source's tree (see [Tree inheritance and automatic expansion](#tree-inheritance-and-automatic-expansion)). |
| `remap/auto_expand` | `false` | Grow the domain when gas the generator needs nears a face; needs a generator hook (`tde_external` has one). |
| `remap/expand_coarsen` | `false` | Every expansion also doubles the root cell. |
| `remap/expand_predict` | `false` | Also grow at once to the box the generator predicts its needed gas reaches; otherwise a face moves only when needed gas comes within its buffer. |
| `remap/expand_rho_min` | $100\,$`dfloor` | Density above which the generator counts gas as needed (code units). |
| `remap/expand_bound_only` | `false` | `true`: only gas bound to the BH whose Kepler orbit reaches its next pericentre before `tlim` is needed; `false`: all gas above `expand_rho_min`, bound or not. |
| `remap/expand_max_x1min` ... `expand_max_x3max` | none | Outer limits of the grown domain; all six are required with `auto_expand` and must contain the mesh. |
| `remap/floor_reset` | `false` | Newtonian hydro only. Every remap, manual or automatic, lowers `hydro/dfloor` to $10^{-10}$ times the largest source density when that is below it, the other density thresholds scaled with it, and turns every source cell below 1.1 times the source's `dfloor` into the new floor at rest (see [Floor reset](#floor-reset)). |

A generator may seed another default by reading a key before the remap runs; it does so only if
the block already exists. Do not add `remap` keys to a deck that is not meant to remap: the mere
presence of the block arms the module.

### Examples

A TDE-style Newtonian remap onto an adapting mesh:

```ini
<remap>
enable        = true
source        = ../04_bh_frame/rst/TDEExternalLTEPrad.00016.rst
settle_steps  = 10
settle_passes = 1
```

## Settle steps

The first remap lays the source state on the target mesh as it was built at the start; the mesh
then adapts to that state, so an adaptive target can end up refined differently from the mesh
the state was laid on. With `settle_steps = N` the run takes $N$ plain steps with AMR checking
every cycle, and then remaps the source onto the adapted mesh again; `settle_passes` repeats this.
While a settle is pending, outputs are suppressed, the configured AMR cadence is restored after
the last pass, and with `time/lat = true` LAT is paused (the steps are plain steps of the global
time step). Each pass calls the generator's hooks like the first remap. Time, step and cycle go
back to the source's at each pass, so `nlim` counts from the source's cycle. Settle steps work
for any generator on an adaptive mesh.

## Tree inheritance and automatic expansion

**Tree inheritance.** With `inherit_tree = true` the target tree is the source's tree placed on
a root grid that contains the source box or crops it (source leaves outside the target box are
dropped, and their gas with them). The target root cell must be $2^k$ source root cells
($k\ge0$, the same on every axis), the MeshBlocks must have the source's size, and the target
origin must sit a whole number of source root MeshBlocks from the source's; anything else is
fatal. A source block of physical level $p$ becomes a target block of level $p+k$ holding the
same cells. A block deeper than the target's `num_levels` allows is merged into its ancestor at
the deepest target level. With AMR, a tree that would hold more than
$0.9\times$ ranks $\times$ `max_nmb_per_rank` MeshBlocks (counting the 2:1 neighbours the tree
adds) loses its deepest levels the same way, one at a time, until it fits; the run prints the
deepest level kept and the MeshBlock count. Every coincident cell is then an exact copy, a
merged cell is sampled like any remapped cell (two trilinear sub-samples per axis: the mean of
its $2^3$ children one level down, of its $4^3$ grandchildren two levels down), and the region
beyond the source box is ambient, so no settle pass is needed. The source's MeshBlock list is read in the
layout the restart writer stores it. A damaged list, or a source leaf outside the source's own root
grid, is a fatal error that names the entry.

**Automatic expansion.** With `auto_expand = true` the driver asks the generator at every
synchronized end of an outer step whether needed gas lies within 4 root cells of a face: the
first call (also after a restart) scans the whole mesh, later calls scan only the blocks at the
faces and rescan the whole mesh when a face is hit. In `tde_external` the needed gas
is gas above `expand_rho_min`; with `expand_bound_only = true` only the part of it that is bound
to the BH and whose Kepler orbit reaches its next pericentre before `tlim`.

- By default each face with needed gas within its buffer moves out by the current length of its
  axis, doubling that side.
- With `expand_predict = true`, in the BH frame the generator also returns the union of the exact
  bounding boxes of the Kepler ellipses of the needed bound gas, and every face moves out to cover
  it plus the buffer (the buffer of the grown root cell). Needed gas on an open orbit has no such
  box: a face it comes near moves out by the length of its axis. In a translating frame no orbit
  is a fixed curve of the grid, so no box is predicted there.

Faces move by whole root MeshBlocks; periodic faces never move. With `expand_coarsen = true` the
new length on each axis is a whole number of two root MeshBlocks and the root cell doubles (the
levels keep their number, so the finest cell doubles too); if that cannot be done within the
limits, the expansion keeps the root cell and says why. Since the doubled root cell doubles the
buffer of the next pass, a doubling expansion looks for needed gas within that doubled buffer, so
it also moves every face the next pass would otherwise find already reached. No face moves past
its `expand_max_*`.
The MeshBlock budget does not stop an expansion, since the next pass coarsens the inherited
tree to fit it (above); only a grown root grid whose root MeshBlocks alone exceed
$0.9\times$ ranks $\times$ `max_nmb_per_rank` is not built. Without `expand_coarsen` the
start-up refuses limits whose root MeshBlocks alone exceed that budget. A face that has to move but cannot is
reported once and is left to its boundary condition.

A planned expansion ends the pass at that step. The final checkpoint (an `<outputN>` with
`file_type = rst`, not `single_file_per_rank`, is required) holds exactly that state. The
stopping pass and the pass the expansion starts write their checkpoints only, no other dumps, so
the other streams keep their numbering and their cadence. The same
process then builds the grown mesh from this input with the grown `<mesh>`, `start_time = 0` and
a `<remap>` block that inherits the tree from that checkpoint with `transition_band = false`,
and `settle_steps = 0`, and carries on from it. One `mpirun` therefore runs through every expansion, under one wall-clock limit.

## Floor reset

As debris spreads its densities fall while `hydro/dfloor` stays put, so the floor gas grows in
relative weight. With `floor_reset = true` a remap lowers `dfloor` to $10^{-10}\rho_{\max}$, the
largest density of the source, when that is below it, and multiplies every threshold tied to it by the same factor:
`hydro/pfloor` (the floor gas keeps its temperature), `problem/rho_floor`,
`problem/bh_grav_rho_min`, `problem/frame_rho_min`, `gravity/rho_grav_min` and
`remap/expand_rho_min`, each only if the input has it. Their ratios to the floor stay those of
the deck. `tde_amr/rho_a`, the ambient cut of the refinement, is set to $10^{-6}\rho_{\max}$ at
every remap when the input gives it, whether or not the floor moves. Source cells below $1.1$ times the source's own `dfloor` (the
floor gas of the old run) become the new floor state at rest before the interpolation.

The physics modules read these values when they are built. A manual remap whose `dfloor` does
not match therefore writes the matching values into the input and builds the run again with
them (the source is read twice); an automatic expansion sets them from the stopped state, so its
next pass matches at once. The values are marked as updated at run time, so a later `-r`
restart keeps them over the deck's. The floor only falls: a target at or above the present
`dfloor` leaves the floors, the thresholds and the floor gas as they are. The reset is refused
for MHD and GR.

## Accuracy and conservation

**The remap is second-order accurate and is not conservative.** Each target cell is the plain mean
of a few point samples of an interpolant, with no cell-volume weights and no donor–acceptor
intersection volumes. Total mass, momentum and energy change by $O(h^2)$ of whatever the source
resolves, and by $O(1)$ wherever the source structure is unresolved on the target grid (any
coarsening loses what it cannot represent). Re-measure the conserved totals after a remap rather
than assuming they carried over. Making the transfer conservative would need an exact
intersection-volume algorithm over two AMR hierarchies ($\sqrt\gamma$-weighted in GR) and a
matching flux-conservative restriction of $B$; the module transfers a *state*, not a budget.

**Total energy is not conserved, by design.** Because the engine carries thermal energy, the
grid-scale kinetic-energy variance a coarser target cannot represent is dropped: on a
$32^3\to16^3$ Orszag–Tang remap $\Delta{\rm KE}/{\rm KE} = -3.8\%$ while $\Delta{\rm IE}/{\rm IE}$ is
$1.8\times10^{-7}$ (it was $+0.96\%$ when $E$ was interpolated, exactly the lost kinetic energy
re-deposited as heat). A state that merely lacks unresolved kinetic energy is still the right
thermal state; a heated one is not. The thermal carry applies to the floor mode only: with an
explicit `band_mode = keep` in a Newtonian run the stored $E$ is interpolated and the heating
returns.

Two things are exact: coincident cells are copied, and $\nabla\!\cdot\!B = 0$ holds on every target
block (up to the keep-mode seam).

## Restrictions

All fail with a clear message at start-up:

- The source must carry exactly one of `<hydro>` or `<mhd>`, and so must the target; the module
  type must match (hydro→hydro, mhd→mhd). A two-fluid target is refused.
- No `<z4c>`/`<cce>`, `<turb_driving>`, `<sink_particles>` or `<shearing_box>` in the source,
  because those restart sections carry no length markers the loader can skip. 
- Source and target must have the same relativity class; special relativity is refused.
- Both ends must be **3D**, whether or not there is a magnetic field.
- The fluid EOS must match: the same number of conserved columns, tabulated versus analytic,
  and for a gamma-law gas the same `gamma` to $10^{-12}$. Floors (`dfloor`, `pfloor`, `tfloor`)
  only warn. An `isothermal` EOS (no energy equation) is refused on either end.
- The number of passive scalars must match, and a dual-energy source needs a dual-energy target.
  A GR checkpoint written with the removed GR dual energy (`dual_energy = true`) is refused.
- A multi-level MHD source is refused unless `b_coarsen_ok = true`.
- `band_mode = floor` with a GR class.
- A remap while LAT windows are running (see below).
- `<problem>/remap*` keys are not read; the remap is configured in `<remap>` only.

## Interactions

- **LAT.** A remap can share a run with LAT but not a LAT window. The startup remap runs before
  the first window and the bins are built from the remapped state at the first time step; the
  settle passes run with LAT paused. A mid-run caller must pause LAT first; any other remap under
  `time/lat = true` is fatal. After the state is replaced the module invalidates the LAT bins and
  rewinds the AMR and rebalance cadence anchors to the source's cycle. The older two-step route
  (remap with LAT off, then `athena -r` with LAT on) still works. See
  [Local Adaptive Time Stepping](Local-Adaptive-Time-Stepping).
- **AMR/SMR targets** are fully supported; the fine-neighbour edge averaging keeps the target's
  fine/coarse faces restriction-consistent.
- **Restart.** The module never runs on `athena -r`.
- **Self-gravity.** The potential is not in the checkpoint; it is recomputed from the remapped
  density. See [Multigrid self-gravity](Multigrid-Self-Gravity).
- **Generator hooks.** A generator can enrol three optional hooks inside its problem function:
  one to mark cells that keep the ambient state (the TDE black-hole excision), one called after
  the source parameter dump is read and before the state is applied (to restore the generator's
  own metadata, such as the TDE black-hole/frame record), and one called after the remap with a
  summary of what was applied (frame switches). A generator that builds a full initial state itself should skip
  that work when remap is enabled, unless its keep-mode blend needs it. `tde_external`,
  the built-in `remap_test` uses them; `gr_torus` needs none.
- **Mid-run use.** The programmatic entry point is available for mid-run re-remaps (the settle
  passes are its own use); callers must re-initialise boundary values and primitives afterwards
  because the module performs no boundary communication, prolongation, physical boundary
  conditions or conserved-to-primitive conversion.

## Outputs and diagnostics

Rank 0 prints a `--- Remap (<remap> block) ---` banner: source path, relativity class, band mode (with
`[auto]` or `[explicit]`), source time and cycle, number of source blocks loaded, the gas group
(hydro or mhd), and for each of the face-field and i0 groups whether
it was applied and, if not, why (disabled by option, source group discarded, source has none).
Check this banner on every new remap. The face-field pass prints the inverse-curl residual, the
boundary-field (sheet-current) warning and, in keep mode, the seam divergence. The built-in
`remap_test` generator prints a `remap_test_errors:` line (divergence, field, density and energy
errors against an analytic reference) that the test battery parses.

## Limitations and known issues

- Not conservative in mass, momentum or energy (above).
- A multi-level MHD source loses sub-root $B$ structure (opt-in, warned).
- A source whose $B$ threads its own boundary gets sheet currents in the taper outside the old
  box (warned above $10^{-2}$ relative); unavoidable for a domain-widening remap.
- The face data of **every** source block is read on **every** rank because the covering-grid
  potential is global; this is a one-time, page-cached I/O cost and makes the result
  independent of the rank count.
- In GR the gas is remapped verbatim and $B$ is rebuilt from the interpolated potential, so a
  strongly magnetised, poorly resolved region pays a one-time floor/FOFC transient in the first
  cycles.
- An exactly divergence-free keep-mode seam would need the generator's own vector potential;
  not implemented.
- The loader identifies a stored `<adm>` payload purely by residual byte count; a future ADM
  backend with a different record shape would trip the "unexpected payload residual" error
  rather than be mis-parsed. The GR gas-entropy column is recognised the same way, by width.
- 3D only.

## Validation

The decks in `inputs/remap/` drive the test battery: a same-grid $B$ round trip to
$10^{-11}$, second-order convergence in $\rho$, $e_{\rm int}$ and $B$, domain widening with
taper, SMR targets, 1-versus-4-rank bitwise identity of the double-precision payload and
multi-level-source rejection and acceptance. The LAT interaction is pinned by
`tst/test_suite/nr/test_nr_remap_lat_cpu.py`.

## See also

- [Local Adaptive Time Stepping](Local-Adaptive-Time-Stepping) – remap with LAT and settle
  steps.
- [Multigrid self-gravity](Multigrid-Self-Gravity), [Dual energy](Dual-Energy) – what is
  recomputed or rebuilt after a remap.
- [Remap usage](Remap-Usage) – how to run a remap, step by step.
- [TDE](TDE) – the main user.

## Where in the code

- `src/remap/remap.cpp`, `remap.hpp` – the `<remap>` block, orchestration, settle steps.
- `src/remap/remap_load.cpp` – restart parsing, the physics whitelist, relativity classes and
  the EOS, width and dimensionality audits.
- `src/remap/remap_cc.cpp` – the cell-centred engine: sampling, thermal-energy carry, band
  modes.
- `src/remap/remap_fc.cpp` – the divergence-free field: inverse curl, Catmull–Rom potential,
  target curl.
- `src/pgen/pgen.cpp` – where the module is invoked after the generator, on fresh starts only.
- `src/pgen/tests/remap_test.cpp` – the analytic test generator.
- `inputs/remap/` – decks for the test battery.
