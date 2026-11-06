# Tidal disruption events (`tde_external`)

The `tde_external` problem generator follows one star from hydrostatic equilibrium through
tidal disruption by a black hole (BH) and on through the return of the bound debris, in a
single Newtonian hydrodynamic calculation. The BH is a softened point mass that is advanced
together with the gas, gas self-gravity is solved with multigrid, and the gas obeys a
tabulated hydrogen and helium equation of state (EOS) with radiation pressure. Use it for
the end-to-end workflow of Jiang et al. (ApJS, 2026): a five-step chain of decks that
enlarges the domain as the debris spreads, converts to the BH rest frame, and finishes with
localized adaptive time stepping. Radiative post-processing of the snapshots is a separate
package, `athenak_rt` (`docs/athenak_rt.md`).

The supported TDE workflow is the one of the paper, built from the modules linked below (tabulated
EOS, multigrid self-gravity, dual energy, LAT, remap). For the rest of AthenaK see the
[upstream documentation](https://ias-astrophysics.github.io/athenak-docs).

## The workflow at a glance

The example decks in `inputs/TDE_examples/` reproduce the fiducial calculation of the paper:
a $1\,M_\odot$, $1\,R_\odot$ star disrupted by a $1000\,M_\odot$ BH on a parabolic orbit
with $\beta = 1$, followed to five fallback times. Each step starts from the last restart file
of the previous step through a `<remap>` block.

| Step | Deck | Time ($t_0$) | Box ($R_\odot$) | AMR levels | Frame | Purpose |
|---|---|---|---|---|---|---|
| 1 | `tde_01_disruption` | 0 to 21 | $64^3$ | 9 | translating | EOS-balanced star, first pericenter passage |
| 2 | `tde_02_remap_box256` | 21 to 24 | $256^3$ | 10 | translating | remap to a larger box |
| 3 | `tde_03_remap_box512` | 24 to 40 | $512\times512\times256$ | 9 | translating | second remap, `<tde_amr>` refinement |
| 4 | `tde_04_bh_frame` | 40 to 70 | $512\times512\times256$ | 8 | BH rest frame | frame conversion |
| 5 | `tde_05_fallback_lat` | 70 to 440 | $1024\times1280\times256$ | 8 | BH rest frame | fallback with LAT |

Step 5 runs with [LAT](Local-Adaptive-Time-Stepping) and the reciprocal BH force on, to
$t = 440.2\,t_0 = 5\,P_{\rm mb}$. The settle steps of its remap run without LAT; LAT starts
after the last remap pass. The finest cell of step 5 is $0.0625\,R_\odot$, i.e. $r_t/\Delta x = 160$.

## Physical setup and code units

The example decks use these code units, set in `<units>`:

| Quantity | Value |
|---|---|
| Mass $M_0$ | $1\,M_\odot = 1.98841586\times10^{33}$ g |
| Length $L_0$ | $2\,R_\odot = 1.3914\times10^{11}$ cm |
| Time $t_0$ | $\sqrt{L_0^3/(4\pi G M_0)} = 1270.915$ s |
| Gravity | $4\pi G = 1$, so $G = 1/(4\pi)$ |
| Velocity, density | $1.0948\times10^8$ cm/s, $0.7382$ g/cm$`^3`$ |

The stellar mass is fixed to 1 in code units. `time_cgs` must be consistent with
`gravity/four_pi_G`, because the BH potential uses $G = \texttt{four\_pi\_G}/4\pi$.

For a star of mass $M_\star$ and radius $R_\star$ and a BH of mass $M_{\rm BH}$,

```math
r_t = R_\star\left(\frac{M_{\rm BH}}{M_\star}\right)^{1/3},\qquad
\beta = \frac{r_t}{r_p},\qquad
P_{\rm mb} = 2\pi G M_{\rm BH}\,(2\,\Delta E_{\rm mb})^{-3/2},\quad
\Delta E_{\rm mb} = \frac{G M_{\rm BH} R_\star}{r_t^{2}} .
```

$P_{\rm mb}$ is the orbital period of the most bound debris and sets the fallback time. With
the defaults of the decks (`mass_ratio = 1000`, `beta = 1`, `star_radius = 0.5`):

| Quantity | Code units | Physical |
|---|---|---|
| $M_{\rm BH}$ | 1000 | $1000\,M_\odot$ |
| $r_t = r_p$ | 5.0 | $10\,R_\odot$ |
| initial separation $10\,r_t$ | 50 | $100\,R_\odot$ |
| $P_{\rm mb}$ | 88.04 | 1.295 d |
| pericenter passage | about 21 | about $0.24\,P_{\rm mb}$ |
| softening `bh_softening` | $5\times10^{-3}$ | $0.01\,R_\odot$ |
| excision radius `bh_excise_radius` | 0.2 | $0.4\,R_\odot$ |

To change the setup:

| To change | Set | Notes |
|---|---|---|
| BH mass | `problem/mass_ratio` | Keep it identical in all five decks. $r_t\propto q^{1/3}$ and $P_{\rm mb}\propto q^{1/2}$, so adapt box sizes, `tlim` and output intervals. |
| Penetration factor | `problem/beta` | $r_p = r_t/\beta$; keep $r_p$ well above `bh_excise_radius` and resolved. |
| Initial separation | `problem/sep_initial` (units of $r_t$) | Parabolic orbits only; sets the time to pericenter, so extend `tlim` of step 1. |
| Eccentricity, inclination | `problem/ecc_bh` ($0<e\le1$), `problem/theta_bh` (degrees) | For $e<1$ the star starts at apoapsis. |
| Star, composition | `<units>`, `problem/star_radius`, another EOS table | Set `mass_cgs` to the stellar mass, choose `length_cgs`, `star_radius` $=R_\star/L_0$, and `time_cgs` from the formula above; the star must lie inside the density and temperature range of the EOS table (another composition: `scripts/generate_lte_table.py`; `athenak_rt` assumes $X=0.7$, $Y=0.3$). |

## How it works

### The initial star

With `stellar_structure_mode = eos_balanced` the star is a spherical hydrostatic profile
integrated with the same EOS table the hydrodynamics uses:

```math
\frac{dm}{dr} = 4\pi r^2\rho,\qquad
\frac{dP}{dr} = -\frac{G\,m\,\rho}{r^2},\qquad
\frac{d\rho}{dr} = \frac{\rho}{\Gamma_1 P}\frac{dP}{dr}.
```

The central density is adjusted until the surface pressure vanishes at `star_radius` with total
mass $M_0$; the internal energy then follows from the table. This avoids the closure mismatch
of a polytrope mapped onto a tabulated EOS. The mode `legacy_polytrope` builds an $n$ =
`poly_n` Lane-Emden star instead (the default, so the example decks set `eos_balanced`
explicitly). The star is placed on a floor atmosphere of density `rho_floor`. A short velocity
damping (`relax_damp`: the momentum inside `relax_radius` is multiplied by
$e^{-\Delta t/\tau}$ each step until `relax_t_end`) removes the residual motion of the
discretized star; it acts only on cells above `bh_grav_rho_min`.

### The black hole and the excision sink

The BH is an external, softened Newtonian point mass,

$$\Phi_{\rm BH}(r) = -\frac{G M_{\rm BH}}{\sqrt{r^2+\epsilon^2}},\qquad \epsilon = \texttt{bh\_softening}.$$

It acts on the gas through a source term whose stencil is the same one used to compute the
force on the BH. The coupling weight $w(\rho)$ is 0 at the density floor, 1 at
`bh_grav_rho_min` and ramps between them as a cubic smoothstep in $\ln\rho$, so floor-density
gas does not feel the BH (a threshold at or below the floor gives a sharp cut). The BH mass is
constant: absorbed gas is not added to it.

Cells within `bh_excise_radius` of the BH form a one-way sink. They are held at rest at
`bh_excise_density` (the density floor by default) and at the thermal floor of that density.
With a floor far below the debris a sink at
the floor is a vacuum, and the gas falling into it sees a rarefaction faster than its own
signal speed; a denser sink avoids that. The only fluxes through the surface of the sphere are those that
carry gas inward. The LAT factor of blocks near the BH (the excision radius plus the ghost
width, widened by the distance the BH moves in a window) and of their neighbors is limited to 1.

### Frames

While the debris is compact, the run is done in a non-rotating frame that translates with the
dense gas (`use_translating_frame = true`). The frame acceleration is the mass-weighted BH
acceleration of the gas with $\rho >$ `frame_rho_min` (excised cells excluded), so floor
atmosphere does not move the frame. In this frame the BH moves through the mesh because of
both its own motion and the frame motion.

Setting `use_translating_frame = false` in a remap (or restart) of a translating-frame source
converts the state: velocities are boosted by the frame velocity (skipped in floor-density
cells), the mesh coordinates are kept, and the frame stops moving. The resulting "BH rest
frame" is the inertial frame in which the BH was initially at rest; the live BH keeps its
small reflex velocity and moves in it. A fresh start needs the translating frame.

The frame correction is applied in the same form as the BH pull: momentum and mass-flux work on
the stage state, with the frame's linear potential. The two cancel to round-off cell by cell. At a
coarse/fine face the fine face cells are also paid the face term of `SourceTerms::Gravity` for the BH
potential, which holds the BH's uniform acceleration along the face; the frame is paid the same term
for its own potential $a_f\cdot x$, so the cancellation holds on those cells too. With `time/lat` the
window-end refund of the frame's share of the refluxed mass-flux work is booked in `W_frame`, so the
energy ledger closes.

### The live black hole

With `bh_live = true` the BH state is advanced once per step (once per synchronization window
with LAT) by a kick-drift-kick leapfrog under the gas pull on it. By default that pull is
sampled from the multigrid self-gravity potential at the BH position, so the BH and the gas
potential evolve from the same density. With `bh_reciprocal_force = true` the acceleration is
instead the equal and opposite of the BH force on the gas, summed over the mesh with the same
stencil, density weight and masks as the gas source term, and the BH velocity is reconciled
with the gas momentum impulse at the end of each LAT window; this needs LAT, self-gravity and
`external_bh_gravity_source`. `bh_pair_force = analytic` uses the analytic derivative of the
softened potential for the reciprocal force and the BH source instead of the finite-difference
stencil.

In step 1 the BH starts outside the 64 $R_\odot$ box, where the gas potential cannot be
sampled, so it is fixed in the inertial frame (`bh_live = false`) and the star follows the
two-body orbit with the translating frame. The live BH starts from rest in the inertial frame
at the first remap and from then on carries its position, velocity and last gas acceleration
through every remap and restart.

### Self-gravity, EOS and dual energy

The gas potential solves $\nabla^2\Phi = 4\pi G\rho$ with geometric multigrid on the AMR
hierarchy ([Multigrid self-gravity](Multigrid-Self-Gravity)). Solving at every
Runge-Kutta stage would dominate the cost, so the potential is refreshed every
`gravity/solve_dt` (the decks use 0.03, about 40 solves per stellar dynamical time of
$1.25\,t_0$), starting each solve from the previous one. With LAT the synchronization window
is limited to `solve_dt`, which must therefore be positive. Gas below `gravity/rho_grav_min` is
excluded from the source with the same smooth ramp as above.

`hydro/eos = lte_table_chabrier2021_t13_helm_union_prad` reads the table named by
`hydro/table`; see [Tabulated EOS](Tabulated-EOS). The cold debris stream is highly
supersonic, so internal energy obtained as total minus kinetic energy becomes inaccurate;
`hydro/dual_energy = true` evolves an auxiliary internal-energy density
([Dual energy](Dual-Energy)). Use $\eta_1 = 10^{-2}$ and $\eta_2 = 10^{-1}$: a small $\eta_2$
lets the truncation noise of $E-E_K$ in the fast cold stream overwrite the auxiliary and shows up
as pressure noise, while shocks are resynchronised by the convergence test regardless of
$\eta_2$.

### Adaptive mesh refinement

Refinement combines generic `<amr_criterionN>` criteria (a density `min_max` on `hydro_w_d` in
steps 1 to 3), optional `<refined_regionN>` boxes (minimum level), and, from step 3 on, the
`<tde_amr>` scheme, switched on by that block together with an `<amr_criterionN> method = user`.
The scheme refines only gas that is bound to the BH,
$\tfrac12|\mathbf v-\mathbf v_{\rm BH}|^2-GM_{\rm BH}/\sqrt{r^2+\epsilon^2}<0$, and denser than
$\rho_a$ (`rho_a`, by default $100\,$`dfloor`); unbound debris is left to the other criteria. Two rules set the target level
of such a cell, the smallest level whose cell width meets them:

* **Rule A, the angular ladder.** $\Delta x_1\le\max(r,r_p)/N_\theta$ for every such cell inside
  $R_c=2a_{\rm mb}$, with $a_{\rm mb}=r_t^2/2R_\star$ the semimajor axis of the most bound debris
  ($R_c=50$ for the decks), and for spine cells (below) beyond it. Inside $R_c$ this covers the loop
  of processed gas, whose apocentres lie inside $2a_{\rm mb}$, and the shocks and hot gas near the
  BH; inside pericentre the scale is $r_p$, not $r$. Beyond $R_c$ only the tail's spine is followed.
* **Rule B, the stream thickness.** Cells are binned by distance to the BH ($\Delta\ln r=0.05$, from
  the excision radius out to the farthest mesh corner) and by azimuth in the orbital plane (128
  bins, the plane through the BH normal to the initial $\mathbf r\times\mathbf v$). A spine cell has
  $\rho\ge f\rho_{\rm bin}$, with $\rho_{\rm bin}$ the densest bound cell of its bin. $T$ is the
  extent of the spine of a bin normal to the plane, between the outermost cell edges, and a spine
  cell at $r_p/2\le r<R_c$ gets $\Delta x_3\le T/N_T$. The vertical resolution thus follows the
  stream's own thickness, through the pericentre nozzle while it is thin and released as it
  thickens. Inside $r_p/2$ no thin stream lives, and bins a few cells wide would read a lone dense
  cell as a one-cell-thick spine and drive the sink inflow to the finest level.

Two caps follow the rules. Inside $r_p/2$, the sink neighbourhood, no cell asks for more than
`max_level - sink_levels` (2 by default), and a block that reaches into the excision sphere is held at
`max_level - sink_levels` at least;
the nozzle and the shocks get the finer levels. With `midplane_finest` a block the orbital plane does
not cross stops at max_level$`-1`$.

A cell that matches neither rule asks for the root level. A block refines if it is coarser than the
finest target of its cells, $T_{\rm up}$, and derefines if it is finer than $T_{\rm down}$, the same
rules with $\rho_a/2$, $\sqrt2R_c$, $\max(r,r_p)/(\sqrt2N_\theta)$ and the spine at $f/2$ with
$T/(\sqrt2N_T)$ (never below $T_{\rm up}$); this hysteresis stops the mesh from flipping. The request
joins the generic criteria (whose flags start at 0 every check): a TDE refinement always wins, a TDE
derefinement stands only where no other criterion flagged the block, and a block at its TDE target is not derefined by
another criterion. All reductions are maxima, so the targets do not depend on the rank count. A
block changes by one level per AMR pass, so the settle steps of a remap must provide at least as
many passes as levels to build.

The keys of the earlier schemes, `problem/bh_max_amr*`, `problem/unbound_amr*` and
`problem/stream_shell_*`, are retired, and so is any `<tde_amr>` key that is neither one of the five
scheme keys nor one of the eleven rule keys in the tables above. In `tde_external` a fresh start that
carries one stops with a message, and a restart, whose file keeps the keys of the run that wrote it,
warns and ignores them. A `method = user`
criterion without `<tde_amr>` stops the run.

**Memory.** With `<tde_amr>` the scheme checks, after every criterion, the block count the pass
would leave, $N+7n_+$ for $n_+$ blocks flagged to refine (flags of blocks in their refinement
cooldown, which the pass clears, are not counted; derefinements are not credited, since only
complete sibling groups merge), against a budget of $0.9$ times the total
`mesh_refinement/max_nmb_total`. The total belongs to the mesh, not to the cards, so the same deck
builds the same mesh on 4, 8 or 10 ranks (and a restart on another rank count continues the same
refinement). The adaptive remap (auto expansion and the merge of an inherited tree) uses the same
total. With `<tde_amr>` (always adaptive) and with the adaptive remap, `<mesh_refinement>/max_nmb_per_rank` and `max_nmb_total` are both
required, and a run without `max_nmb_per_rank` stops at startup saying so. The total is a target: the
estimate leaves 10 % headroom and does not include the 2:1 closure, so when `ranks` times `max_nmb_per_rank` exceeds the total
the mesh can end above it, and the per-rank cap is the hard memory limit (the decks set the total equal to that product, which
keeps the mesh at the budget). The decks write ten times
`max_nmb_per_rank`, the ten-card layout they were sized for (5200 for 520, 4000 for 400), and a run
stops at startup if the total exceeds `ranks` times `max_nmb_per_rank`, naming the smallest rank
count that fits. Over the budget the refinements in the nozzle zone are kept first, then the others
nearest the BH; the refinements farthest from the BH are denied until the estimate fits, and distance
ties are broken in tree (Z) order, so a restart ranks the blocks the same way. Rank 0 prints
`TDE AMR budget: denied ...`.
Denied refinements make the far end give way, so a full mesh does not freeze: the farthest blocks at $r \ge R_c = 2a_{\rm mb}$ are derefined, two sibling groups per denied nozzle-zone refinement and one per other denied refinement (those only beyond the nearest denied request that is not a zone refinement and not among the blocks the zone share took, at most 2% of the budget per pass, so the far debris coarsens gradually). An eviction also sets a
hold radius at the nearest evicted block: requests from outside the zone at or beyond it are denied until they
would fit within 95% of the budget, when the radius moves outward by 20% per pass, so the evicted blocks are not
rebuilt in the next pass. The hold is released once it reaches the farthest mesh corner from the BH, beyond which
it denies nothing. The hold radius is kept in the restart file (`<tde_amr_state>/hold_radius`, 0 for none or
released), so a restart resumes with the hold of the checkpoint.
Behind it `max_nmb_per_rank` stays the per-rank memory cap and a hard cap: if a refinement pass would exceed it, the pass's
refinements are cancelled and its derefinements kept (the whole pass is cancelled if derefinement
alone cannot meet the cap), with a message. With LAT, `ncycle_check` and `refinement_interval` are
512, so AMR checks happen about once per window.

### Remap and frame conversion

A `<remap>` block on a fresh start loads the source restart file and resamples mass, momentum
and thermal energy onto the new mesh; the total energy is rebuilt from them, so kinetic energy
the new grid cannot represent is dropped instead of becoming heat. The remap is therefore not
conservative (see [Remapping](Remapping)). The BH and frame state are restored from metadata
in the restart file. When a remap or a restart switches a translating-frame state into the BH frame,
the dual-energy auxiliary is first set to the internal energy the inversion reads in the old frame,
because the boost raises the total energy and can move that inversion's energy-channel test. The boost
then leaves rest mass, momentum and total energy as before. Each remap first runs `settle_steps` steps with AMR checked every cycle
and outputs suppressed, to build the hierarchy, then projects the source state again onto it.
A remap never runs on a restart (`-r`). Remap and LAT can share a run because LAT is paused
during the settle steps.

The fallback step of the generated chain (`make_tde_chain.py`) changes the frame and grows the box
by more than ten times, so its root cell is far coarser than the debris. Resampled there, the thin
stream would be aliased by the $2^3$ sub-samples of each root cell and most of its mass lost before
the settle steps could refine on it. That step therefore inherits the MeshBlock tree of the step
before it (`inherit_tree = true`, `transition_band = false`, `settle_steps = 0`): its root cell is
$2^k$ times the source's and its corner sits on the source's root-block lattice, so every source
cell lands on a coincident cell (a cell finer than the step's deepest level becomes the mean of its
children): the mass crosses unchanged and the energy changes only by the frame boost.

### Localized adaptive time stepping

With `time/lat = true` each MeshBlock advances with a step that is a power of two times the
smallest, up to $2^{\texttt{lat\_levels}}$ (7 gives 128), so the few blocks near the BH and the
nozzle shock do not set the step of the whole box. Neighbors differ by at most a factor of two
and conservation across mixed-step interfaces is enforced with time-integrated flux corrections
([LAT](Local-Adaptive-Time-Stepping)). In the production benchmark of the paper LAT
reduces the number of MeshBlock updates by 85% and raises the throughput by a factor of 3.1.

## Deck parameters

Only keys a user sets are listed. The remap is configured in the module-level `<remap>` block
(`Remapping.md`); `<problem>/remap*` keys are not read.

### `<problem>`: star, orbit, BH

| Key | Default | Meaning |
|---|---|---|
| `star_radius` | 0.5 | stellar radius (code length), $>0$ |
| `stellar_structure_mode` | `legacy_polytrope` | `eos_balanced` (all example decks) or `legacy_polytrope` |
| `poly_n` | 1.5 | $0<n<5$; defines the legacy star, only a starting guess for `eos_balanced` |
| `rho_floor` | 1e-8 | ambient density added to the star (decks: 1e-10) |
| `amp` | 0 | $m=2$ perturbation of the legacy polytrope; must be 0 with `eos_balanced` |
| `x_center`, `y_center`, `z_center` | -1, 0, 0 | star center; use 0 for all three |
| `vx_star`, `vy_star`, `vz_star` | 0 | gas velocity relative to the frame |
| `relax_damp`, `relax_tau`, `relax_t_end`, `relax_radius` | false, 0, 0, `star_radius` | velocity damping; the three numbers must be $>0$ when on (decks: 0.02, 0.2, 0.5) |
| `mass_ratio` | 1000 | $M_{\rm BH}/M_\star$, also the BH mass in code units |
| `beta` | 1 | penetration factor |
| `ecc_bh` | 1 | eccentricity, $0<e\le1$ |
| `theta_bh` | 0 | inclination (degrees); rotates the orbit about $y$, tilting $\mathbf L$ from $+z$ toward $+x$ |
| `sep_initial` | 10 | initial separation in $r_t$ (parabolic orbit) |
| `provide_params`, `x1`..`z1`, `vx1`..`vz1` | false, 0 | give the star state relative to the BH directly |
| `bh_softening` | 0.01 | softening $\epsilon$ (decks: 0.005) |
| `bh_potential` | `newtonian` | `newtonian` (softened point mass) or `wegg`: the Wegg (2012, ApJ 749, 183) pseudo-Newtonian potential (Eq. 1, Table 1 potential C), $\Phi=-\alpha GM/r_s-(1-\alpha)GM/(r_s-R_x)-GMR_y/r_s^2$ with $\alpha=-\frac43(2+\sqrt6)$, $R_x=(4\sqrt6-9)r_g$, $R_y=-\frac43(2\sqrt6-3)r_g$, $r_g=GM/c^2$; it carries the Schwarzschild apsidal precession of low-energy orbits to within 1% (12.1$`^\circ`$ per passage at $r_p=47\,r_g$). Needs `<units>` and `bh_excise_radius` $\ge2r_g$; held at its value inside $1.6\,r_g$. Every module that applies or measures the BH potential uses it (`pgen/bh_force_pair.hpp`); a parabolic start is parabolic in it, with the pericentre at $r_p$ |
| `bh_grav_rho_min` | 0 | density above which the BH pull is full (decks: 1e-9) |
| `bh_inner_boundary`, `bh_excise_radius` | true, 0.5 | excision sink (decks: radius 0.2) |
| `bh_excise_density` | `dfloor` | density held inside the excision sink (at least the floor) |
| `external_bh_gravity_source` | true | BH pull applied as an external source; required by `bh_live` and `bh_reciprocal_force`. `false` applies the pull in the generator's own kernel and is refused with `time/lat` and an energy equation, which has no window-end refund for it |
| `external_bh_dt_factor` | 0.5 | time-step limit $\Delta t\le f\sqrt{\Delta x/a_{\rm BH}}$; 0 disables |
| `use_translating_frame` | false | `true` for a fresh start; `false` only with a remap or restart |
| `frame_rho_min` | 0 | only gas above this density sets the frame acceleration (decks: 1e-9) |
| `bh_live` | false | the BH responds to the gas |
| `bh_reciprocal_force` | false | reciprocal force; needs `bh_live`, LAT, self-gravity |
| `bh_pair_force` | `finite_difference` | or `analytic` (needs the reciprocal force) |

### `<tde_amr>`

The block, with an `<amr_criterionN> method = user`, switches the scheme on; an empty block takes
the defaults. Any other key in it stops the run.

| Key | Default | Meaning |
|---|---|---|
| `cells_per_radian` | 64 | $N_\theta$ of rule A, $>0$ |
| `cells_per_thickness` | 16 | $N_T$ of rule B, cells across the vertical extent of the spine, $>0$ |
| `spine_frac` | 0.1 | $f$, spine threshold relative to the densest cell of the bin, $0<f<1$ |
| `midplane_finest` | true | the finest level only on blocks the orbital plane through the BH crosses: one finest block holds a stream thin enough to need it across its own height, so blocks off the plane stop one level coarser |
| `bound_only` | true | only gas bound to the BH counts; `false` follows all gas, for the star and the fresh debris around pericentre |
| `rho_a` | 0 | $\rho_a$, the ambient cut in code density: gas below it is never refined; 0 takes $100\,$`dfloor`. Set it when the floor is far below the densities worth resolving, e.g. `dfloor` = 1e-15 |

Optional rules, each off by default (the scheme above):

| Key | Default | Meaning |
|---|---|---|
| `plane_only` | false | only blocks the orbital plane crosses refine, the rest follow the 2:1 rule |
| `dense_frac` | 0 | $f_d$: a spine cell also needs $\rho\ge f_d\rho_{\rm ring}$, the densest bound cell of its $\ln r$ ring |
| `cells_per_radian_envelope`, `cells_per_radian_tail` | $N_\theta$ | rule A for the bound non-spine gas inside $R_c$, and for the spine beyond it |
| `cells_per_thickness_tail` | 0 | rule B on the spine beyond $R_c$ |
| `shock_mach`, `cells_per_radian_shock` | 0, $N_\theta$ | bound gas inside $R_c$ with a compression number $m_c=-\nabla\cdot v\,\Delta x/c_s$ above `shock_mach` takes the shock ladder |
| `nozzle_radius`, `nozzle_mode` | 0, `max` | the spine on plane blocks within $d_n$ of the nozzle point goes to `max_level` (`thickness`: to its uncapped level, rule B counting $\min(N_T, N_{zone})$ cells across, $N_{zone}$ the largest power of two not above $T_{min}/\Delta x_{fin}$ and at least 2, $T_{min}$ the thinnest spine of the zone, so only the stretch within about a factor two of the thinnest reaches `max_level`) |
| `cap_levels`, `sink_levels` | 0, 2 | all but the nozzle stop at `max_level - cap_levels`; the sink cap and floor sit at `max_level - sink_levels` |

The orbit model gives what is Newtonian in these rules. `tde_external`: $c_s^2=\tfrac{10}{9}e/\rho$ and
$\nabla\cdot\mathbf v$; the nozzle point from the mass-weighted eccentricity vector and pericentre distance
$h^2/[GM(1+e)]$ of the dense bound gas falling in between $2r_p$ and $R_c$.

Fixed and printed at startup: $\rho_a$, $R_c=r_t^2/R_\star$, $r_p=r_t/\beta$, 128 azimuth bins with
$\Delta\ln r=0.05$ from the excision radius (the softening without excision, at least one finest
cell), and the block budget $`0.9\times`$`max_nmb_total`.

### Keys of other blocks that the decks rely on

| Key | Deck value | Meaning |
|---|---|---|
| `hydro/eos`, `hydro/table` | `lte_table_chabrier2021_t13_helm_union_prad`, `../chabrier2021_t13_helm_union_prad_640.table` | EOS and table; relative paths resolve from the working directory, then `$ATHENAK_DATA`, then `<source>/data` |
| `hydro/lte_bounds` | `clamp` | out-of-range states clamp to the table edge (default `error`) |
| `hydro/dfloor`, `pfloor`, `tfloor_kelvin`, `cs_ceil`, `vceil` | 1e-10, 1e-12, 1, 15, 15 | floors and ceilings (code units; K) |
| `hydro/dual_energy`, `dual_energy_eta1`, `dual_energy_eta2` | true, 1e-2, 1e-1 | dual energy |
| `hydro/allow_reinterpretive_restart` | true from step 2 | accept a restart or remap source written with another EOS or table |
| `gravity/self_gravity`, `four_pi_G` | true, 1.0 | `four_pi_G` is required |
| `gravity/solve_dt`, `rho_grav_min` | 0.03, 1e-9 | interval between Poisson solves (required $>0$ with LAT); density above which gas sources and feels self-gravity fully |
| `remap/source` | `../0N_.../rst/<file>.rst` | restart file of the previous step; required |
| `remap/settle_steps`, `settle_passes` | 10, 1 | settling (the module default is 0 and 1; `tde_external` seeds 20 steps) |
| `time/lat`, `lat_levels` | step 5: true, 7 | LAT and its largest factor $2^7$; the decks also set `lat_same_level = true`, `lat_same_level_max_ratio = 2`, `lat_neighbor_limiter = all` |
| `time/lat_min_bin_count` | default $4\times$ ranks | smallest bin population; set it for reproducibility across rank counts |
| `mesh_refinement/max_nmb_per_rank` | 520 | cap on blocks per rank, bounds device memory |
| `mesh_refinement/max_nmb_total` | 5200, required | blocks of the whole mesh; the AMR and remap budgets are 0.9 times it, whatever the rank count. A restart file written before the key existed lacks it: `athena -r rst/<file>.rst -i deck.athinput` |
| `gravity/mg_bc`, `threshold` | `multipole`, 1e-10 | isolated-system potential at the box faces; the multigrid iterates to the absolute defect 1e-10 |

## A minimal example

A `<problem>` block for step 1, and the AMR switch of steps 3 to 5:

```
<problem>
pgen_name = tde_external
stellar_structure_mode = eos_balanced
star_radius = 0.5
x_center = 0.0           # y_center, z_center = 0.0 too
mass_ratio = 1000.0
beta = 1.0
use_translating_frame = true
bh_live = false          # true from step 2 on
bh_softening = 5.0e-3
bh_excise_radius = 0.2
bh_grav_rho_min = 1.0e-9
frame_rho_min = 1.0e-9

<amr_criterion0>
method = user

<tde_amr>
cells_per_radian = 48
cells_per_thickness = 16
spine_frac = 0.1
```

## Build and run

The workflow needs only the `tde_external` problem generator:

```bash
cmake -S . -B build_tde -D PROBLEM=tde_external -D Athena_ENABLE_MPI=ON \
      -D Kokkos_ENABLE_CUDA=ON -D Kokkos_ARCH_VOLTA70=ON \
      -D CMAKE_CXX_COMPILER=$PWD/kokkos/bin/nvcc_wrapper
cmake --build build_tde -j 16          # executable: build_tde/src/athena
```

Set `Kokkos_ARCH_*` for your GPU (the paper used V100 cards, one MPI rank per GPU). A CPU build
without the CUDA options is suitable for small tests only.

**EOS table.** The decks read `chabrier2021_t13_helm_union_prad_640.table` (Tomida et al. 2013
at low density, Chabrier et al. dense fluid, a fully ionized HELM-style branch at high
temperature, with radiation pressure; $640\times640$ in $\ln\rho$, $\ln T$; $X=0.7$, $Y=0.3$).
Binary tables are not in git. Build it with `./get_eos_table.sh` in the top-level directory
(needs Python with numpy and scipy and curl or wget; about two minutes), or download it from the
release assets, and place it one directory above the step directories. The chemistry plots of
`plot_slice.py` additionally need `lte_t13_prad_eos.table` (`python3 scripts/generate_lte_table.py
lte_t13_prad`). See [Tabulated EOS](Tabulated-EOS).

**Running the chain.** Run every step from its own directory, with the directories side by side,
because each `<remap>` source is a relative path to the previous step's last restart file:

```bash
for step in 01_disruption 02_remap_box256 03_remap_box512 04_bh_frame 05_fallback_lat; do
  mkdir -p $step && (cd $step && mpirun -np 10 $ATHENA -i $DECKS/tde_$step.athinput)
done
```

The chain used 10 GPUs of 16 GB; step 2 needs all of them, and the memory of step 5 grows as the
debris spreads. A step that stops early continues with `athena -r rst/<last file>.rst` in its
directory; command-line overrides such as `time/tlim=` apply on top of the stored input. Exact
restart file names are in [`inputs/TDE_examples/README.md`](https://github.com/HongxuanJiang/athenak-transients/blob/main/inputs/TDE_examples/README.md).

**Variations of the paper.** The self-gravity cadence test repeats steps 1 to 4 with
`gravity/solve_dt=0.0025` on the command line. The LAT benchmark runs step 5 with
`time/lat=false problem/bh_reciprocal_force=false` to a checkpoint at $t=71$, then restarts it with
and without LAT (the reciprocal force needs LAT) on the same rank count.

## Outputs and diagnostics

Every example deck writes a history file (`<output1> file_type = hst`, `dt = 0.1`), `hydro_w`
binary snapshots in `bin/` and restart files in `rst/`, all with base name `TDEExternalLTEPrad`.
File numbers continue across remaps.

* **History.** `TDEExternalLTEPrad.hydro.hst` holds volume integrals over the whole box (`mass`,
  `1-mom`..`3-mom`, `tot-E`, `1-KE`..`3-KE`), including the floor atmosphere and the excised region.
  With `gravity/lat_time_centered_work = true` (needs LAT, `rho_grav_min` $\le$ `dfloor`, no
  `mask_radius` and no sink particles; off in the example decks) a second file `.user.hst` adds an energy ledger (`K_BH`,
  `U_bhgas`, `U_self`, `W_cent`, `B_mass`, `B_E`, `B_Wself`, `B_Wbh`, `E_remap`, BH position and
  velocity, `E_floor`, `E_asym`, `W_frame`).
* **Snapshots** contain `dens`, `velx`, `vely`, `velz`, `eint` and, with self-gravity, `grav_phi`, in
  code units. The header copies the input including `<units>` and the runtime state, so a snapshot is
  self-describing: `bh_live_x/y/z`, `bh_live_vx/vy/vz` (BH in the simulation frame), `frame_live_*`
  (origin and velocity of the moving frame; zero in the BH rest frame), `bh_live_inertial_*`,
  `bh_live_ax/ay/az` (last gas acceleration) and, with the reciprocal force, `bh_pair_gas_impulse_*`
  and `bh_pair_impulse_residual`. These are written by the code; do not set them by hand.
* **Restarts** hold the full state, the parameter input and the BH and frame metadata that the next
  remap reads.

**Slices.** `scripts/TDE/bin/plot_slice.py <snapshot.bin> <quantity> <out.png> [options]` plots a
stored variable or `derived:<name>` (`T`, `pgas`, `c_s`, `vel_norm`, `mu`, `xion`, `xh2`, `xhe1`,
`xhe2`, `gamma1`, `gamma3m1`, `beta_rad`; `derived:?` lists them), recovering thermodynamics from the
table in the header. Options include `-d/-l` (slice normal and position), `--x1_min`.. (window),
`-c/-n/--vmin/--vmax`, `--orthogonal_triptych`, `--streamlines`, `--bound_unbound_contour`, `--grid`,
`--bh_marker`, `--no_bh_mask`, `--notex`. `plot_density_midplane_parallel.sh` and
`plot_density_slices_parallel.sh` run it over all snapshots in the current directory (jobs and
output directory as arguments; windows and limits as environment variables listed in each script).

**Budgets and distributions.** `python3 scripts/TDE/bin/analyze_tde.py [INPUT_DIR [OUTPUT_DIR
[CONFIG.json]]]` (numpy, numba) reads the `*.bin` snapshots one at a time and writes one
`<snapshot>.analysis.dat` each (default `INPUT_DIR/analysis_dat`, with `SUMMARY`, `HIST` and `HIST2D`
records) plus a `README_ANALYSIS.md`. For gas above `density_threshold_factor` (10) times `dfloor`,
relative to the BH state in the header and with the softened BH potential, it computes the specific
orbital energy, angular momentum $j$, Bernoulli-like energy and eccentricity; their 1D and 2D mass
distributions for the channels `in_domain` (bound/unbound and inward/outward splits), `sink_rate`,
`outer_rate` and fallback shells at 0.5, 1.0, 1.2, 1.5 times $a_{\rm mb}=r_t^2/2R_\star$; and mass
budgets of the domain, the sink and the outer boundary with $\dot M_{\rm sink}$, $\dot M_{\rm outer}$
and shell rates. Keys of `analysis_config.json` that matter: `snapshot_selection`, `sink_radius_code`
(0.2; set it to your `bh_excise_radius`), `r_tide_code` and `stellar_radius_code` (null: derived from
the header), `fallback_radius_multipliers`, the binning, `resume_existing_outputs` (skip snapshots
whose analysis file matches) and `numba_threads`/`batch_blocks` (shipped as 32; lower on a laptop).
`plot_analysis.py [ANALYSIS_DIR [FIGURE_DIR]]` (matplotlib) draws seven figures into
`paper_figures/` (mass budget, rates, circularization, distribution budget, selected snapshots,
in-domain evolution, flux evolution).

Read the caveats before quoting numbers. The dumps carry no face fluxes, so sink and outer losses are
proxies from snapshot-local shells; the in-domain histograms are not the full late-time distribution
(gas also leaves through the sink and the outer boundary, so combine the channels); fallback-shell
inward flux is not the sink accretion rate; the `compressed_proxy` subsets use negative velocity
divergence and are not shock tracers; and the $t^{-5/3}$ columns and deep-tail fits are descriptive. For
luminosities, spectra and images use `athenak_rt` (`docs/athenak_rt.md`).

## Limitations and known issues

* The model is Newtonian hydrodynamics with a softened potential, no magnetic fields and no
  relativistic effects. Only the fiducial configuration (a $1\,M_\odot$, $1\,R_\odot$ star, a
  $1000\,M_\odot$ BH, $\beta=1$) is validated. The generator needs a
  `<hydro>` block and stops if an `<mhd>`, `<radiation>`, turbulence-driving or spacetime block is
  present.
* The excision sphere is a one-way sink that does not grow the BH. The separate sink-particle
  module is in the tree but combining it with `tde_external` is not validated.
* The remap is not conservative (kinetic energy the new grid cannot represent is dropped); the
  paper reports conservation audits along the chain.
* `plot_analysis.py` expects `analysis_global_timeseries.dat` and
  `analysis_cumulative_distributions.dat` in the analysis directory, but the shipped
  `analyze_tde.py` removes these names at start and never writes them (its routine that builds them
  is not called), so the plotting step stops with a "missing" message. The per-snapshot
  `.analysis.dat` files, whose `SUMMARY` blocks carry the budget scalars, are written correctly.
* LAT results depend on the number of ranks unless `time/lat_min_bin_count` is set.

**Troubleshooting.**

| Symptom | Cause and remedy |
|---|---|
| `Failed to read lte_table` | The table path is resolved from the working directory; run from the step directory with the table above it, set `$ATHENAK_DATA`, or pass `hydro/table=/absolute/path`. |
| The EOS-balanced star fails to build | The star lies outside the density-temperature range of the table; use a table that covers it. Clamping is not used here. |
| `use_translating_frame = false is only supported for restart/remap-based TDE conversion` | A fresh start needs the translating frame; use `false` only with `<remap>` or on a restart. |
| `Restart compatibility check failed` | The source used another EOS or table; set `hydro/allow_reinterpretive_restart = true` only if intended. |
| `retired or unknown TDE AMR keys` | Remove the listed keys; the TDE refinement is `<tde_amr>` (`cells_per_radian`, `cells_per_thickness`, `spine_frac`, `rho_a`, ...) with an `<amr_criterionN> method = user`. |
| Out of device memory | Lower `max_nmb_per_rank` and use more ranks, keeping `max_nmb_total` at or below ranks times `max_nmb_per_rank`; the hierarchy grows fastest after pericenter. |

## See also

* `athenak_rt` (`docs/athenak_rt.md`): luminosities, spectra and images from the snapshots
* [Tabulated EOS](Tabulated-EOS), [Dual energy](Dual-Energy), [Multigrid self-gravity](Multigrid-Self-Gravity)
* [Local adaptive time stepping](Local-Adaptive-Time-Stepping), [Remapping](Remapping)
* [Home](Home)

If you use this workflow, cite Jiang et al., *An End-to-End Numerical Framework for Tidal Disruption
Events with AthenaK*, ApJS (accepted, arXiv:2609.37859), and the AthenaK paper (Stone et al. 2026,
ApJS 283, 27). Cite Tomida & Stone (2023) for the multigrid solver, Liska et al. (2022) for LAT,
and Tomida et al. (2013), Chabrier et al. and Timmes & Swesty (2000) for the EOS where relevant.

## Where in the code

* `src/pgen/tde_external.cpp`, `tde_external.hpp`: the problem generator (star, BH, frames, AMR hooks)
* `src/pgen/tde_amr.hpp`: per-cell rules of the `<tde_amr>` scheme
* `src/pgen/bh_dynamics.hpp`, `bh_force_pair.hpp`, `star_bh_orbit.hpp`: live-BH helpers
* `src/srcterms/`, `src/hydro/hydro_fluxes_excision.cpp`: BH source term and excision fluxes
* `src/remap/`, `src/gravity/`, `src/eos/`: remap, multigrid and the tabulated EOS
* `inputs/TDE_examples/`: the five chain decks; `inputs/TDE_q1e6/`: the first three steps of a mass-ratio $10^6$ chain
* `src/pgen/stellar_profile.hpp`: the hydrostatic star
* `scripts/TDE/bin/`: `plot_slice.py`, the batch plot scripts, `analyze_tde.py`, `plot_analysis.py`, `analysis_config.json`
