# Multigrid Self-Gravity

The `<gravity>` block turns on Newtonian self-gravity: a cell-centred geometric multigrid
solver for $\nabla^2\phi = 4\pi G\rho$ on the whole MeshBlock hierarchy (uniform, SMR or AMR),
plus a source-term coupling that adds $-\rho\nabla\phi$ and the matching gravitational work to
the conserved variables of the hydro or MHD fluid. It is used by the TDE, sink-particle and Jeans/collapse test
problems. The coupling described here is Newtonian; the general-relativistic paths drive the
same solver with the relativistic source $E+S$ and a four-force. It works on Cartesian
(possibly anisotropic) cells only. The cost
is dominated by the Poisson solve, not by the coupling. Under
[LAT](Local-Adaptive-Time-Stepping) the potential is frozen over a whole window. In the
default mode it is solved at a window start: on the solve_dt grid, brought forward to the start
of a window that would cross a due time, and again whenever a regrid or a restart invalidates it.
With `gravity/lat_time_centered_work` it is solved at every window end (see Energy ledger).

## How it works

### Equation and discretisation

The finest-level source is the conserved density of whichever fluid is active, weighted by the
coupling weight $w(\rho)$ described below and scaled by $-4\pi G$. The operator is the standard
7-point Laplacian, scaled by $\Delta x^2$, with per-axis weights for non-cubic cells,

```math
 L\phi = \mathrm{diag}\;\phi_{ijk} - (\phi_{i\pm1}) - w_y\,(\phi_{j\pm1}) - w_z\,(\phi_{k\pm1}), \qquad
   w_y = \left(\tfrac{\Delta x}{\Delta y}\right)^2,\ w_z = \left(\tfrac{\Delta x}{\Delta z}\right)^2,\ \mathrm{diag} = 2(1+w_y+w_z),
```

built from the root-grid spacings. MeshBlocks must be logically cubic ($n_x = n_y = n_z$) and a
power of two in size; both are fatal errors otherwise.

### The multigrid cycle

The solver is a Full-Approximation-Storage (FAS) multigrid with red–black Gauss–Seidel
smoothing and over-relaxation weight $\omega$ (`omega`). Restriction is the plain 8-cell
average; prolongation is trilinear or a factorised tricubic stencil with weights
$(5,30,-3)/32$ per axis. One V-cycle goes down through the hierarchy, solves the coarsest
grid, and comes back up; on a *cold* start (no previous potential) a full-multigrid (FMG) ramp
first builds an initial solution from the coarsest level upwards, then V-cycles refine it.

The hierarchy has three kinds of level. MeshBlock levels (device) coarsen each block down to
one cell. Refinement levels between the root grid and the blocks are represented by
**octets** – one $2\times2\times2$ parent cell per refined region – which take part fully in
the V-cycle but live on the host. Root levels coarsen the grid of root blocks while all three
block counts stay even; when the root grid cannot be reduced to one cell, or has more than 100
coarsest cells, a warning is printed and the bottom solve becomes iterative (correct, but
slower). A coarsest grid of a single cell is solved directly.

### Convergence control

`threshold` selects one of three policies:

- **`threshold < 0`:** exactly `niteration` V-cycles per solve.
- **`threshold > 0`:** V-cycles in bursts until the L2 defect drops below `threshold` (an absolute
  value). A burst is never longer than `defect_check_interval`; it is shorter when the measured
  contraction per V-cycle (of this solve, or of the last iterative solve before the first burst)
  predicts that the threshold is reached sooner, so the defect is checked where the solve is
  expected to end and the stopping test itself does not change. The solve is capped at
  `niteration` x `defect_check_interval` V-cycles (`niteration` 160 if not positive; the interval is
  1 with `show_defect >= 2`), so a shorter burst never shrinks the budget. It also stops when three
  checks in a row fail to bring the defect below 0.995 of the best defect of the solve: the
  threshold then lies below the round-off floor of the problem, which grows with the source while
  an absolute threshold does not (the polytropic star with $\rho_c = 10^8$ stalls between
  $2\times10^{-9}$ and $3\times10^{-8}$ against a threshold of $10^{-10}$),
  and further V-cycles would only redraw the same noise up to the cap. A V-cycle that contracts the
  defect faster than 0.995 per check never takes this exit. The stop prints "Stopping at practical
  convergence floor" with the defect, the best defect and the threshold on rank 0, the first time
  in a run and every time with `show_defect`; slow convergence is reported when `show_defect` is on.
- **`threshold = 0`:** the automatic controller. A short preset burst is followed by extra
  cycles until the defect falls below $10^{-6}$ of the source norm (a fixed fraction, floored at
  $10^{-12}$, not a deck key), or `auto_max_extra_cycles` is reached, or two checks stagnate. On a warm solve in the last RK
  stage of an unchanged mesh only `warm_final_niter` cycles run (fewer if the previous defect
  was already small).

One of `threshold` or `niteration` must be set; `niteration = 0` requires `threshold >= 0`.

### Boundary conditions

Periodic mesh faces stay periodic. Every other face is an isolated-system face, `multipole`, unless
`mg_bc` sets another value for all of them or `ix1_bc`…`ox3_bc` set single faces. `mg_bc = none`
takes the mesh boundaries instead (outflow becomes zero-gradient). A mesh with both periodic and
non-periodic faces and no `mg_bc` stops with an error: the multipole expansion of an isolated mass
is not the potential of a system that is periodic in some direction, so the deck must say what it
means. A reflecting mesh face is a symmetry plane, and the multipole expansion of the mass inside
the domain misses the mirror mass there, so without `mg_bc` such a face needs its own `ix1_bc`…`ox3_bc`
key (`zerograd` is the symmetry condition) or the run stops with an error naming the face. This
holds for Newtonian and GR self-gravity alike. A fully periodic mesh needs no key. Three rules exist for a non-periodic face:

- `zerograd` (Neumann, alias `outflow`): the ghost cell equals the interior cell.
- `zerofixed` (Dirichlet $\phi=0$ on the face): the ghost cell is minus the interior cell.
- `multipole` (isolated): $\phi_{\rm ghost} = 2\phi_{\rm mp} - \phi_{\rm interior}$, with
  $\phi_{\rm mp}$ from a real-spherical-harmonic expansion of the source about the centre of
  mass, to quadrupole (`mporder = 2`, 9 coefficients) or hexadecapole (`mporder = 4`, 25).

**The Jeans swindle.** With no `zerofixed` or `multipole` face anywhere, all-Neumann walls
would demand zero enclosed mass. The solver then forces `subtract_average = true` (with a
warning) and solves $\nabla^2\phi = 4\pi G(\rho-\langle\rho\rangle)$, which is equivalent to a
uniform negative background and produces a spurious outward acceleration
$(4\pi G/3)\langle\rho\rangle r$ that grows with radius. In the TDE campaign this
was measured at 0.03% of the true field at the stellar surface and 0.3–0.7% at the black
hole's initial radius, isotropic and identical across similarity-rescaled runs, so it cancels
in comparisons. It appears only when a deck asks for `mg_bc = zerograd` or `none` with outflow
faces; the default `multipole` solves the isolated problem, and the shipped decks state
`mg_bc = multipole` explicitly. With any multipole face `subtract_average` is forced off.

### Coupling into the fluid

The gravity source is applied once per Runge–Kutta stage in *source form*, following Mullen,
Hanawa & Gammie (2020). With $\delta\phi_L = -(\phi_i-\phi_{i-1})$ and
$\delta\phi_R = -(\phi_{i+1}-\phi_i)$ the potential differences across the two faces of a cell,
$F^\rho$ the Godunov mass fluxes, and $w$ the coupling weight below,

```math
 \Delta(\rho v_x) = \tfrac12\,\frac{\Delta t}{\Delta x}\,w\,\rho\,(\delta\phi_L+\delta\phi_R),
\qquad
\Delta E = \tfrac12\,\frac{\Delta t}{\Delta x}\,w\,\bigl(F^{\rho}_{i}\,\delta\phi_L + F^{\rho}_{i+1}\,\delta\phi_R\bigr),
```

and likewise in $y$ and $z$. Tying the work to the mass flux makes the energy change consistent
with the mass actually moved (the work term exists only for an EOS with an energy equation).
Exact conservation is limited by the multigrid residual and by any
time-lagged potential. The source does not update the dual-energy auxiliary (gravity is not a
thermal source); any reliable total-energy change is folded back into it later by the dual-energy
synchronization.

**The coupling weight.** `rho_grav_min` is not a hard cut. Both the Poisson source and the
force are multiplied by a weight that ramps smoothly in $\ln\rho$ from 0 at the hydro density
floor to 1 at $\rho_{\min}$,

$$ w(\rho) = 3t^2 - 2t^3, \qquad t = \frac{\ln(\rho/\rho_{\rm floor})}{\ln(\rho_{\min}/\rho_{\rm floor})}, $$

with $w=1$ above $\rho_{\min}$ and $w=0$ at or below the floor. The same weight enters the
source and the force, so the self-gravity pair still conserves momentum. If
$\rho_{\min}\le\rho_{\rm floor}$ the ramp is empty and $w$ is a step at $\rho_{\min}$ (the
default, $\rho_{\min}=0$, couples every cell). No weight can make the total energy exactly
conserved when $w'(\rho)\ne0$; the ramp trades a small energy source in compressed ramp gas for
a force without discontinuities.

**Black holes and sinks.** An analytic softened black-hole potential
(`problem/external_bh_gravity_source`, on by default for the TDE and star–BH generators with
hydro) is added in the same kernel, separately from $\phi$. Sink particles are deliberately
**not** in the Poisson right-hand side; their potential enters through a separate source term
built from the same softened potential the sink–gas reaction uses, so the pair force is equal
and opposite cell by cell. Two masks exist: a generator-published *black-hole sink mask*
(the source is zeroed inside the sink radius), and the generic `mask_radius`, which zeroes the
source *outside* a radius.

### Cadence: `solve_every` and `solve_dt`

By default the potential is re-solved at every RK stage. `solve_every = N` solves only on
cycles with `ncycle % N == 0`; `solve_dt = T` solves when `time` reaches the next point of the
fixed grid $kT$ or reaches the time of a checkpoint (see below), so the schedule is a function of
time and of the checkpoint times alone and a restarted run solves on the same cycles as the
uninterrupted one (`solve_dt` overrides `solve_every`). Between solves the
potential is simply reused. Under LAT `solve_dt > 0` is mandatory and bounds the window length.

### AMR, SMR and restart

After a regrid the potential is invalidated unless `reuse_phi_after_amr = true`, in which case
the transferred potential is the next initial guess; the first solve after a mesh change uses
exact halo exchanges and a larger cycle budget. The potential travels with the blocks only in that case (and in a pure rebalance, which moves it verbatim); a derefined block carries the restriction of its children. Otherwise it is not migrated at all, since the next solve is cold and never loads it. The potential is **not** written to restart
files, by decision: the first solve after a restart is a cold FMG solve, so a restarted run agrees with the
uninterrupted one only to the multigrid tolerance (velocities to $\sim10^{-11}$ with five FMG
cycles; to $10^{-16}$ with `threshold = 1e-12`), identically for hydro and MHD.
Writing a checkpoint marks a solve due at its time (not with `lat_time_centered_work`, which solves
at every window end anyway), so under LAT the run that wrote it solves at its next window start, as
a restart from the file does at its first; the solve schedule therefore follows the checkpoint times
as well as the $kT$ grid. Without LAT and with `solve_dt`, the restart also solves at the first RK
stage of that cycle, which the uninterrupted run skips while it reuses the potential of its last
solve, so there the stage-1 force differs by the age of that potential and not only by the
multigrid tolerance.

### Initial guess

A cadenced solve (`solve_dt` or `solve_every > 1`) in the last RK stage of an unchanged mesh starts
from the potential of the previous solve. If the potential of the solve before that is also known on
the same mesh, the extrapolation $\phi + r(\phi-\phi_{\rm prev})$ with $r$ the ratio of the time
since the last solve to the interval between the last two (accepted for $0<r\le2$) is a second
candidate. The solver evaluates the defect of both and iterates from the smaller one, so it never
starts from a worse iterate than the plain previous potential; the equation and the stopping test
do not change. The extrapolated candidate needs `threshold > 0` and a solve that does not take the
FMG ramp: the automatic controller (`threshold = 0`) and fixed-cycle solves (`threshold < 0`) start
from the plain potential, as do the solve after a regrid or a remap, a cold solve and a solve at
every RK stage. The history costs one more potential array per rank (0.37 MB per
$32^3$ MeshBlock) and is not written to restart files.

## Deck parameters

There is no separate `<multigrid>` block; every key lives in `<gravity>`.

### Essential keys

| Key | Default | Meaning |
| --- | --- | --- |
| `self_gravity` | `true` when `<gravity>` exists | Master switch for the solver and the source term. |
| `four_pi_G` | none (must be set) | $4\pi G$ in code units; must be positive, the run stops if it is unset or zero. |
| `threshold` | −1 | Target L2 defect: `<0` fixed cycles, `>0` iterate to it, `0` automatic controller. |
| `niteration` | −1 | V-cycles (`threshold<0`), V-cycle cap `niteration` x `defect_check_interval` (`threshold>0`) or preset burst (`threshold=0`). |
| `rho_grav_min` | 0 | Density at which the coupling weight reaches 1 (ramp from the hydro floor). |
| `mg_bc` | `multipole` | Value for all non-periodic faces: `zerofixed`, `zerograd`/`outflow`, `multipole`, `none` (take the mesh boundaries). `periodic` is fatal here. Required when the mesh mixes periodic and non-periodic faces, or has a reflecting face without its own per-face key. |
| `solve_dt` | none | Solve interval in code time; `>0` overrides `solve_every`. Required with LAT. |
| `solve_every` | 1 | Solve on cycles with `ncycle % N == 0`. |
| `reuse_phi_after_amr` | `false` | Keep the AMR-transferred potential as the next initial guess. |

### Solver and convergence

| Key | Default | Meaning |
| --- | --- | --- |
| `full_multigrid` | `true` | Use the FMG ramp on a cold solve. A warm potential always uses plain V-cycles. |
| `fmg_ncycle` | 1 | V-cycles per FMG level. |
| `omega` | 1.15 | Over-relaxation weight of the smoother. |
| `npresmooth`, `npostsmooth` | 1, and 2 on a refined mesh (else 1) | Smoothing sweeps per level. |
| `mg_prolongation` | `tricubic` on a refined mesh, else `trilinear` | Prolongation stencil; any value other than `tricubic` means trilinear. |
| `defect_check_interval` | 2 | Upper bound of the V-cycles between defect evaluations (forced to 1 for `show_defect >= 2`); with `threshold > 0` a burst is shortened to the cycle at which the measured contraction predicts the threshold is reached. |
| `auto_max_extra_cycles` | −1 (computed) | Extra-cycle cap of the automatic controller: $\max(6, 3\,\texttt{fmg\_ncycle})$, at least 12 on a refined mesh and 32 right after a regrid. Read only when `threshold = 0`. |
| `warm_final_niter` | 4 | Cycles on the last stage of a warm solve. Used only when `threshold = 0` and `niteration <= 0`. |
| `coarsest_min_sweeps` | 64 | Minimum relaxations on a multi-cell coarsest grid; inert on a one-cell grid. |
| `subtract_average` | `true` for a strictly periodic mesh | Remove the mean of source and solution (forced on with all-Neumann faces, off with a multipole face). |
| `mg_nghost` | 1 | Ghost cells on multigrid levels; values above 1 are fatal on a refined mesh. |

### Boundary conditions and masks

| Key | Default | Meaning |
| --- | --- | --- |
| `ix1_bc`, `ox1_bc`, `ix2_bc`, `ox2_bc`, `ix3_bc`, `ox3_bc` | mesh-derived | Per-face override with the `mg_bc` vocabulary. |
| `mporder` | 4 | Multipole order, 2 or 4. Read only when a face is `multipole`. |
| `auto_mporigin` | `true` | Use the source centre of mass as the expansion origin at every solve. |
| `nodipole` | `false` | Drop the $\ell=1$ terms; fatal together with `auto_mporigin`. |
| `mporigin_x1`, `mporigin_x2`, `mporigin_x3` | required if `auto_mporigin = false` | Fixed expansion origin. |
| `mask_radius` | −1 (off) | Zero the Poisson source outside this radius of `mask_origin_x1..3` (default 0). |

### Parallel and performance options

The defaults suit most runs; they matter only on many ranks or on deep AMR hierarchies.

| Key | Default | Meaning |
| --- | --- | --- |
| `ca_pack_smoother` | `rbgs` | Smoother variant on the block levels: `rbgs` (red–black Gauss–Seidel), or the communication-avoiding `jacobi` and `chebyshev`/`chebyshev2`. |
| `same_exchange_stride` | 1 | Exchange same-level halos only every N smoother steps (multi-rank, coarse blocks only). |
| `local_sweeps_per_exchange` | 1 | Local red–black sweeps between exchanges. |
| `reduced_exchange_max_edge` | 4 | Largest block edge (cells) on which the reduced exchange is used. |
| `ca_pack_cheb_lambda_min`, `ca_pack_cheb_lambda_max` | 0.7, 1.95 | Chebyshev eigenvalue bounds. |
| `distributed_coarse_solve` | `false` | Solve the root/octet levels on one rank and broadcast the result (only on multi-rank AMR runs). |
| `root_on_host` | `true` if the root grid has at most 4096 blocks | Keep root-grid arrays in host memory. |

The host path of the fine/coarse ghost fill (remote edge and corner neighbours) copies each run of
flagged blocks in one transfer, and a level of at most 8 MiB as a whole, instead of one block at a
time. This changes the cost of a V-cycle, not the values it produces.

### Diagnostics and ledger

| Key | Default | Meaning |
| --- | --- | --- |
| `show_defect` | 0 | `0`/`false` quiet, `1`/`true` final defect each solve, `>=2` per iteration. |
| `show_timing` | `false` | Print a per-solve phase-timing line. |
| `mg_verbose` | 0 | Gates only the warning for a multipole source with cancelling total mass. |
| `lat_time_centered_work` | `false` | LAT, hydro only: freeze $\phi$ over each window, re-solve at its end and add the time-centering work (see below). Needs `rho_grav_min <= hydro/dfloor`, no `mask_radius`, no sink, and `solve_dt > 0` or `solve_every = 1`. |
| `lat_boundary_flux_diagnostics` | `false` | Tally mass, energy and gravitational work leaving through physical faces (needs `lat_time_centered_work`). |
| `lat_ledger_debug` | `false` | Print per-window identity terms on rank 0. |
| `mg_fc_symmetric` | `false` | Make the coarse/fine interface operator exactly symmetric (see below). |
| `reciprocity_test` | `false` | At start-up, solve for two density states and report the reciprocity defect. |

### Keys in other blocks

| Key | Default | Meaning |
| --- | --- | --- |
| `problem/external_bh_gravity_source` | `true` for TDE/star–BH with hydro | Add the analytic softened black-hole potential in the same kernel. |
| `problem/bh_grav_rho_min` | 0 | Density at which the black-hole coupling weight reaches 1. |
| `sink_particles/newton_g` | $\texttt{four\_pi\_G}/4\pi$ | $G$ for sink potentials when `<units>` is absent. |

## A minimal example

The self-gravity block of a TDE deck on a refined mesh under LAT:

```ini
<gravity>
self_gravity = true
four_pi_G    = 1.0
mg_bc        = multipole     # isolated problem
threshold    = 1.0e-8
fmg_ncycle   = 1
rho_grav_min = 1.0e-10
solve_dt     = 0.03          # required with time/lat
reuse_phi_after_amr = true
mg_prolongation = tricubic
coarsest_min_sweeps = 32
```

For a test on a periodic box, the Jeans-wave deck (`inputs/tests/jeans_wave.athinput`) uses a
fixed budget, `threshold = -1` with `niteration = 5`, and the Poisson tests iterate to
`threshold = 1e-8`; both keep the default boundaries.

## Outputs and diagnostics

- **`grav_phi`** is an output variable holding the potential, including the analytic
  black-hole potential when present. It is appended automatically to the standard hydro and MHD
  bundles (`hydro_u`, `hydro_w`, `mhd_u_bcc`, `mhd_w_bcc`, …) whenever a gravity potential
  exists, and is fatal to request without one.
- `show_defect` prints `MGGravityDriver::Solve: Final defect norm = …` for the last RK stage
  of each solve; a warning is printed once per 1000 cycles when the automatic controller ends
  far from its target.
- `show_timing` prints one `MGGravityDriver::Timing:` line per solve (source load, setup,
  multigrid time, V-cycles, per-phase times).
- At start-up the root-grid level count and any coarsest-grid warning are printed.

### Energy ledger (hydro, LAT)

With `lat_time_centered_work = true` the potential is frozen over each LAT window, solved
again at its end, and $-\tfrac12\sum\Delta\rho\,\Delta\phi\,dV$ is added to the gas energy so
that, for a symmetric Poisson operator, $E_{\rm gas} + \tfrac12\sum\rho\phi\,dV$ is exact to
round-off apart from floors, AMR remaps and boundary transport. Those are tallied in double
precision and exposed as user history columns shared by the TDE and star–BH generators:
`K_BH`, `U_bhgas`, `U_self`, `W_cent`, `B_mass`, `B_E`, `B_Wself`, `B_Wbh` (the last four need
`lat_boundary_flux_diagnostics`), `E_remap`, `E_floor`, `E_asym`, the black-hole state and the
age of the potential. The tallies are carried through regrids, rebalances, checkpoints and restarts.
A checkpoint stores the floor energy booked since the last window end as `<gravity>/lat_floor_energy_pending`,
with `<gravity>/lat_floor_tally_on`; both keys are optional, and a file without them restarts as before.
Checkpoint replays and restart recoveries leave the floor ledgers unchanged. These tallies are diagnostic
and never change the evolved state. Closed-box tests close the
energy ledger at second order in the time step (residuals of order $10^{-6}$–$`10^{-5}\,|U_{\rm self}|`$); see
`tst/bh_trajectory/ENERGY_CLOSURE.md`.

`E_floor` is the energy the EOS floors add to a stage state, weighted by that state's share of the
completed step ($\tfrac12$ for the stage-1 state of rk2, 1 for the stage-2 state). Under the LAT union
predictor the stage-1 state is recovered in the corrector's boundary refresh, which carries the stage-1
share too; before this, the share was 1 and a live-BH deck with mixed cadence windows showed a closure
drift of $`1.5\times10^{-4}\,|U_{\rm self}|`$ per unit time (at a window of two fine ticks) that was only
the stage-1 floors counted at twice their weight, while the gas and BH energies themselves were unchanged.

The composite coarse/fine operator on the MeshBlock levels is not exactly symmetric: the
fine-side ghost fill couples a fine cell to the coarse cell's tangential neighbours with no
reciprocal entry. The measured reciprocity defect is $6\times10^{-12}$ on a uniform mesh,
$2.5\times10^{-5}$ on two levels and $6\times10^{-5}$ on three. `mg_fc_symmetric = true` drops
the tangential terms: symmetric to round-off at the same V-cycle count, changing the self-energy
of the test star by 0.14%. The asymmetry is booked in `E_asym` either way.

The gravitational work of the gas telescopes across a coarse/fine face as well. A fine face cell
pays $-\tfrac12 F_f A_f\,dt\,(g_f-\phi_f)$ with its ghost $g_f$ and the coarse cell pays
$\tfrac23 F_f A_f\,dt\,(\bar\phi_f-\phi_C)$ per fine face through the restricted flux, while the mass
$F_f A_f\,dt$ changes $\sum\rho\phi\,dV$ by $F_f A_f\,dt\,(\phi_C-\phi_f)$. The two sides differ by the
covariance of the face flux with the tangential variation of $\phi$, so after the work of
`SourceTerms::Gravity` each fine face cell is paid $\tfrac12 F_f A_f\,dt\,[(\phi_f-\bar\phi)+(g_f-\bar g)]$ with
the means over the fine faces of one coarse face. It is zero for a uniform flux and needs
no data from the coarse side. Without it, a star centred on a refinement boundary lost 3% of
its self-gravity work, independent of the time step; with it the work closes to $10^{-3}$ of that.
The external black hole is paid the same way with its analytic potential: the fine face cell receives
$F_f A_f\,dt\,[\tfrac12(\Phi_f+\Phi_g)-\tfrac12(\Phi_C+\Phi_{Cg})]$, with $\Phi$ evaluated at the fine centre
and ghost and at the coarse centre and ghost, which removes the $\tfrac{3}{32}\Phi''\,\Delta x_c^2$ per unit
mass that the two one-sided stencils leave. The term needs the $2\times2$ fine faces of a coarse face, so it
is applied on three-dimensional meshes with refinement only. For a uniform BH acceleration $a$ the bracket
equals $-a\cdot(x_q-\bar x)$ per unit mass along the face, so the fine cell receives
$-F_f A_f\,dt\,a\cdot(x_q-\bar x)$, and the translating frame of `tde_external` pays for its own potential
with the same grouping.

## Tuning notes

Measured on the TDE production meshes (about 10 GPUs); the defaults were
chosen from them.

- The multigrid took about three quarters of the wall time on the star–BH mesh before tuning.
  Reusing the potential after AMR (`reuse_phi_after_amr`) together with a correct `threshold`
  made the gravity about 1.7 times faster.
- Tricubic prolongation is both faster and lower in residual than trilinear on the refined TDE
  restart.
- On the TDE restart, `auto_max_extra_cycles = 1` cut the solve cost by 40–45% relative to the
  older open-ended loop while recovering most of the residual in a few cycles; use 2 when the
  residual matters more than speed and 0 only for exploration.
- Switch `show_defect` off for throughput runs; the defect norm is an extra global reduction
  and cost a few per cent in the TDE benchmark.
- The root and bottom solve was not the bottleneck: fine-level correction dominated. Setting
  `coarsest_min_sweeps` to 1 (+67%) or 64 (+35%) instead of the 32 used in the star–BH decks,
  `mg_prolongation = trilinear` (+43%) and reduced exchanges (`same_exchange_stride` > 1,
  +23%) were all slower on that mesh.
- A root grid that cannot coarsen to a single cell (for example $2\times2\times1$ blocks)
  makes the bottom solve iterative; pick root block counts that are powers of two in every
  direction if the bottom solve ever matters.
- On the star–BH mesh `threshold` values of $10^{-7}$ and $10^{-6}$ ran slower than $10^{-8}$
  and changed the trajectory; $10^{-8}$ is the validated setting.

### Same-level halo traffic on the finest level

A finest-level round (one same-level face exchange between the ranks) costs a fixed latency
plus its bytes, and on a multi-GPU node it is mostly bytes. Two exact reductions apply to the
manual finest-level paths (no deck key, results bitwise unchanged):

- The first colour sweep of a V-cycle's descent runs without an exchange. The halo is exact on
  entry: the previous V-cycle's tail ended on a full refresh and nothing writes the potential
  in between (the defect reads it only), or the refresh just before has run.
- After a one-colour sweep only the cells of that colour differ from the copy the neighbour
  holds, and the 7-point red–black smoother reads only cells of the other colour. The next
  face-only exchange therefore sends just the swept colour, half the same-level bytes. A
  prolongation rewrites every cell and is followed by a whole-face exchange. The colour is the
  smoothers' own $(i+j+k-n_{gh}) \bmod 2$ and is the same number on both sides of a block face
  when the block edge is even, so a level with an odd edge, a one-dimensional mesh, a
  two-layer halo and the exchanges with diagonals send the whole face. Same-rank neighbours and
  the coarse–fine shell are not reduced.

MPI buffer registration is not a steady-state cost. Every MPI-facing multigrid buffer is
persistent and grow-only, and a probe of seven consecutive solves saw device-IPC mappings and
host registrations only in start-up and the first solves; the second and third solve after a
restart still map the buffers of each peer once, and the first solve after a regrid grows the
pool. Profile the multigrid only from the fourth solve after a restart and away from a regrid,
or the trace measures this warm-up.

## Limitations and known issues

- MeshBlocks must be logically cubic and a power of two in size.
- Self-gravity needs a hydro or MHD fluid; with LAT it needs hydro with exactly one fluid (a two-fluid
  ion-neutral pack has no single source density and is refused).
- The potential is not in the checkpoint: restarts are not bitwise (see above).
- With all-Neumann boundaries the solved equation is the Jeans swindle, not the isolated
  problem; the solver only warns.
- Conservation is only as good as the multigrid residual and the age of the potential.
- Without LAT the potential is solved at every stage; with `solve_every > 1` or `solve_dt > 0`
  it is frozen between solves and the lag shows up as an $O(\texttt{solve\_dt})$ error (about 2%
  in the growth rate of a Jeans wave at the gate's `solve_dt`, 0.14% at a 35 times smaller one).
- The octet (refinement) levels are host-resident by design, so a very deep hierarchy can be
  dominated by them; a device implementation was slower and less converged on the TDE benchmark.
- Dual energy: the gravity source does not update the internal-energy auxiliary directly; the
  dual-energy synchronization absorbs the change afterwards.
- The weighted coupling is not energy conserving where $w'(\rho)\neq0$ (see above); the ledger
  closes only up to that term.
- `mg_verbose` gates a single message and is not a general verbosity switch.

## Tests

`tst/test_suite/multigrid/` (CPU, MPI-CPU and GPU variants selected by file name) covers:
two-sphere binary gravity on uniform and SMR meshes (defect $10^{-9}$ within 10 V-cycles,
average reduction below 0.0625, consistent across rank counts); stable and unstable Jeans
waves with both `fmg` and iterative solves, growth rates within 1–3% of
$\omega^2 = k^2c_s^2(1-n_J^2)$ and convergent in resolution (including a 4-rank adaptive case);
64³ hydro and MHD Poisson problems with defect independent of MeshBlock size; and
`test_mg_lat_gravity_cpu.py` for the frozen-potential LAT gates. The decks are `inputs/tests/{binary_gravity,jeans_wave,selfgravity}.athinput`,
`tst/inputs/selfgravity*.athinput` and `inputs/LAT/`. Decks that switch self-gravity on:
`inputs/TDE_examples/`, `inputs/remap/tde_golden_*.athinput`,
`inputs/sink_particles/creation.athinput` and the collapse decks `inputs/tests/be_collapse*.athinput`,
`polytropic_star.athinput`.

## See also

- [Local Adaptive Time Stepping](Local-Adaptive-Time-Stepping) – window length, the frozen
  potential and the window-end work refund.
- [TDE](TDE) – the main user.
- [Dual energy](Dual-Energy) – the auxiliary variable the gravity source does not update directly.
- [Remapping](Remapping) – remap of a self-gravitating run (the potential is recomputed).
- Mullen, Hanawa & Gammie (2020), *ApJS* – the flux-consistent gravitational-work source form.

## Where in the code

- `src/gravity/` – the module: `Gravity` (potential storage and validity, energy ledger) and
  `MGGravityDriver` (all `<gravity>` keys, the solve, cadence, masks, bottom solve).
- `src/multigrid/` – the generic multigrid: level arrays, restriction/prolongation,
  smoothers, octets, root–block transfer, boundary rules, multipole expansion, V-cycle/FMG.
- `src/srcterms/` – `SourceTerms::Gravity`, the force and work source (self, black hole, sink).
- `src/hydro/hydro_update.cpp` – the LAT window-end gravitational-work
  refund.
- `src/driver/driver.cpp` – where the solve is called and the LAT scheduling guards.
- `src/mesh/mesh_refinement.cpp` – potential handling after a regrid.
