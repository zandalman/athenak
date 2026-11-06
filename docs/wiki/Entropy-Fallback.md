# Gas entropy fallback (general relativity)

Every general-relativistic run (fixed or dynamical metric, with or without a magnetic field)
carries one extra conserved column, the gas entropy, and uses it as the KORAL code (Sadowski et al. 2013) does: where the
energy inversion fails, the primitive variables are recovered from the transported entropy
instead. The column is always on in GR and has no deck key; Newtonian runs do not have it
(their cancellation problem is handled by [Dual energy](Dual-Energy)). It replaces the
two-threshold adiabat channel that earlier versions offered in GR.

## The variable

The column is $D\,s$, with $s$ the entropy per unit rest mass of the gas,

```math
s=\frac{1}{\Gamma-1}\ln\frac{p}{\rho^{\Gamma}},
```

so that $S=\rho s$ is the gas entropy of that method. $\Gamma$ is the index of
the cell: the deck's. For `dyn_eos = lte_table` the transported quantity
is the adiabat label $L=\ln\varepsilon_{\rm top}$, the specific energy reached at the density
of the table's top row along the table's own adiabat $d\varepsilon=-p\,d(1/\rho)$, integrated
once at table load. An ideal-gas table gives $L=(\Gamma-1)s+{\rm const}$. Other dynamical
policies (`compose`, `hybrid`, `piecewise_poly`) provide their tabulated entropy per baryon
or the thermal adiabat; a policy without one is refused at set-up.

In smooth adiabatic flow $u^\mu\partial_\mu s=0$, so $\partial_\mu(\sqrt{-g}\,\rho s\,u^\mu)=0$
holds exactly and $s$ is a passive scalar with no source. The column lives in the auxiliary
slot after the passive scalars and is invisible to problem generators and to the scalar loops.
Outputs name it `s` (primitive) and `dens_s` (conserved).

### The table label

The label is integrated from every node of the table to its top row when the table is loaded
(`lte_eos::BuildAdiabatLabelAtNode` on the device, then `lte_table_utils::FinalizeAdiabatLabel`
on the host). The substeps per density interval are doubled from one until two successive
doublings each change the label by at most $10^{-12}$ relative. One agreeing pair is not
enough: two coarse integrations can take one step across the same feature of the path and
agree on its error. Every step also resolves, with the same number of substeps, the thickness
of the table cell it is in and the length on which the slope under the path changes. Where
the bundle of neighbouring adiabats spreads by a factor $A$ on its way to the top row, a
rounding error is multiplied by $A$, and a node that does not reach $10^{-12}$ is accepted at
$10^{-12}A$. On the production tables these are 140 to 240 nodes at $T\lesssim100$ K whose
adiabats pass through the cold dense corner below ($A$ up to $2\times10^{5}$, last change at
most $2\times10^{-10}$); the load reports them.

Where the specific energy does not depend on $T$ ($c_v\to0$) neither does the entropy, and the
label is flat in $T$. The Chabrier union tables have such runs in their cold dense corner
($\rho\approx10^{-3}$ to $10^{3}$ g/cc, $T\approx10^{2}$ to $10^{5}$ K, 9686 of the 408,960
temperature intervals), where the generator raised $\ln\varepsilon$ to $1+10^{-12}$ times that
of the colder node wherever $\varepsilon$ did not increase. The adiabat through a node of such
a run leaves it at one of its edges within a negligible range of density, so the nodes that
leave through the same edge share that edge's label. Where the table's pressure rises along a
run, the nodes on either side of some point leave through different edges and the label steps
there: by up to 3.7 in $L$ at $\rho\approx0.4$ g/cc, $T\approx130$ K, where $p$ rises by 40%
within one interval of constant $\varepsilon$. That is the table's own thermodynamics, which
the label follows. A label row is therefore non-decreasing in $T$, not strictly increasing. The load raises a node whose label lies below that of a colder
node of its row by no more than the two labels' accuracy (70 to 80 nodes on the production
tables, by at most $3\times10^{-15}$ relative). It stops with a fatal error if a label lies
further below, if a node has no finite label, or if a node meets neither convergence
criterion; none does on the production tables.

The inverse $T(\rho,L)$ is exact wherever the label increases. On a run of equal labels it
returns the hottest temperature of the run. The specific energy recovered from a label agrees
with that of any temperature on the run to the run's span: the round trip
$\varepsilon(\rho,T(\rho,L(\rho,T)))$ against $\varepsilon(\rho,T)$ is within $3\times10^{-13}$
over both production tables. Temperature and pressure are not determined there. The table's
pressure varies along these runs, so the recovered pressure differs from the original by up to
$2\times10^{-3}$ relative in the corner (within $10^{-11}$ elsewhere).

## Transport

The face flux is the face's final rest-mass flux times the donor cell's own $s$, the value of the
cell the mass leaves (the cell-centred $s$ of the stage start, `w0`), not the reconstructed face value
the other passive scalars take. It is formed after every replacement of the mass flux (FOFC, excision,
runtime sinks), so it rides FOFC,
LAT dense output, AMR, load balance and restart like any conserved column. The LAT
delayed reflux is the exception: the column was resynchronised at the end of the coarse
cell's own step (below), so that cell no longer holds the entropy its face flux moved,
and the window-end correction moves the corrected mass at the cell's own $s$,
$\Delta(Ds) = s\,\Delta D$ (`lat_reflux::GasEntropyCorrection`). Refluxing the
transported mismatch instead made $s \sim 10^2$ in a vacuum cell whose correction returned
its density to the floor, and the window-end entropy root a $10^{14}$ pressure.
The column and the GR passive scalars are conserved as $D\,q$, and every prolongation
of them follows one ratio rule (`ProlongCCRatio`): the specific value $q$ is prolongated, and the conserved
value is the child's own density times it. No child or ghost then carries a ratio that no coarse neighbour
holds. The rule is applied in these places:

- Regrid children take the parent's $s$ times their own $D$ (`RefinedFirstOrderWhereInadmissible` in
  `src/mesh/mesh_refinement.cpp`).
- Fixed-metric and special-relativistic ghosts on the default conserved fill keep $D$ from the conserved
  prolongation and take $D_{\rm ghost}$ times the min-mod prolongation of the specific value. With
  `<mesh_refinement>/prolong_primitives` the specific $s$ is prolongated and the fine ghost carries $D s$.
- Dynamical-GR ghosts take $D_{\rm ghost}$ times the prolongated specific value, on the accepted conserved
  ghosts and on the rebuilt ones (`src/dyn_grmhd/dyn_grmhd_prolong_impl.hpp`).

A stencil with a non-positive or non-finite $D$ keeps the conserved prolongation.

**Why the donor value, and the drained cell.** The column obeys $\partial_t(Ds) + \nabla\cdot(Ds\,V) = 0$
with no source and is used only by the entropy root, which maps it exponentially to the pressure. For a
cell $i$ with the outgoing and incoming face weights $w_f = \beta\,\Delta t\,|F_{D,f}|/\Delta x_f$ a stage
gives

$$(Ds)_i^{\rm new} = \gamma D_i s_i - \sum_{\rm out} w_f s_f + \sum_{\rm in} w_f s_f,\qquad
D_i^{\rm new} = \gamma D_i - \sum_{\rm out} w_f + \sum_{\rm in} w_f ,$$

with $\gamma$ the Shu-Osher weight of the state the fluxes act on ($\gamma_0 + \gamma_1$ at the first stage,
where both registers hold it, $\gamma_0$ later). The new $s$ lies between the values the column holds iff
the outgoing faces carry the cell's own $s_i$ and the remainder $R_i = \gamma D_i - \sum_{\rm out} w_f$ is
not negative (consistency with continuity, Larrouturou 1991, J. Comput. Phys. 95, 59). Both failed
for a high-order scheme that does not flatten at an extremum (WENOZ, PPMX): the face $s$ was the
reconstructed value, and a one-cell density hole exports many times its own mass through the high-order
face density of its neighbour side, $R_i < 0$. The update then extrapolates, $s$ reached $10^2$ to $10^3$
against neighbours near $50$, the root turned into an energy of $10^{266}$, and the run blew up in two
steps. The two rules that restore the invariant, neither a cap nor a filter:

- **Donor value.** The entropy flux is $F_D$ times the donor cell's $s$ (above). The column lives one step
  (it is resynchronised at the end of every step, below), so the first-order label error does not
  accumulate; the gas update is unchanged.
- **Drained cell.** With FOFC, every tested cell whose stage update would leave $R_i < 0$, computed from the
  trial's own face mass fluxes, is flagged and takes first-order fluxes on its six faces
  (`StageDrainsCell` in `src/utils/mass_drain.hpp`, called from each `BuildFOFCTrial`). For a Rusanov face
  the outflow of a cell is at most $\sum_d \lambda_d a_d D_i$ ($a \le 1$), so $R_i \ge 0$ for $\mathrm{cfl}
  \le 1/3$ in three dimensions; in smooth flow $R_i/D_i \ge 1 - \sqrt{3}\,\mathrm{cfl}\,|V|$ and the flag
  never fires. It reads $D$ only, so the entropy slot stays out of every admissibility test. It also
  protects every other label the mass flux carries in a drained cell. It is skipped at a stage whose
  weight $\gamma_0$ is not positive (the non-SSP `rk4` tableau has negative weights, for which no convex
  form exists). With `fofc = false` the test does not run, and `wenoz` or `ppmx` is refused.
- **Closure.** The test sees the high-order fluxes, and the replacement of a flagged neighbour's face
  changes the outflow of an unflagged cell afterwards, so the same test is repeated on the replaced
  fluxes until no cell is added (`CloseDrainedCells`). Without it a hole
  beside four flagged cells kept $R=-0.17\,D$ and an entropy $s=242$ against neighbours at 51 to 76,
  the one blow-up in four seeds of the WENOZ-HLLE deck.

The reference method reconstructs the entropy at high order and clamps the ratio $S/\rho$ at a fixed ceiling of 100; the donor value with the drained-cell flag replaces that clamp here, which is a
cap and would hide the defect. This departs from KORAL and changes the face value, not the quantity. The
reconstructed face value stays in the face states (the reconstruction is unchanged), and no other code reads
it.

The slot is never part of an admissibility test (FOFC flags, scalar maximum principle,
refinement criteria, time-step bounds), so the other variables are unaffected by it.

## Inversion sequence

For every cell that is not excised, in the full recovery (never in the FOFC trial):

1. The energy root, as before.
2. A hot verdict, the reference method's hot-failure set evaluated on the cell's own conserved state:
   no root or iteration limit, a non-finite result, an internal energy below zero, or a root
   that does not reproduce the conserved state. A floor that acted on the energy root is
   judged by solving the cell's own state once more with every floor at zero. Beyond KORAL's
   set, a root whose internal energy does not exceed the rounding of the sums that form it
   holds no significant digit of $\epsilon$ and counts as a failure. The test holds for
   the fixed-metric GR hydrodynamics (bound $6\,\epsilon_{\rm mach}(W|q|+zr+z^2/(1+W))$ for
   $\epsilon=Wq-zr+z^2/(1+W)$), for its FOFC candidate test, and in the coarse-buffer and refinement-time recoveries. Where the
   entropy root fails too, a root that only lacks digits stays the cell's.
3. On a hot failure with $D>0$, the entropy root from $s=u_0^{\rm s}/D$, the reference method's entropy inversion:
   the momentum, field and density equations with the adiabat $p=K\rho^\Gamma$,
   $\ln K=(\Gamma-1)s$, in place of the energy equation. The state it returns replaces the
   energy state. On the fixed metric it is solved as the Kastaun master function with the
   enthalpy of the adiabat at $\rho=D/W$ (`RecoverSRStateOnAdiabat` in
   `src/eos/gr_gas_entropy.hpp`): $[0,\mu_+]$ brackets its root for any gas with $h\ge1$, so
   every state with $D>0$ has one, and no seed is needed. KORAL runs a damped Newton iteration
   on $\rho hW^2$ from the previous primitives instead, which on a host scan of states with
   $\sigma\le10^6$, $W\le50$ finds no root for 3 to 6 % of them; the bracketed solve finds all
   of them in 6 to 20 evaluations, with the pressure of the adiabat at the root. The dynamical
   root is the PrimitiveSolver's bracketed root with $T$ taken from $(n,s)$.
4. If both fail, the cell keeps its last primitives (the reference method's double failure).
5. The Lorentz and magnetisation ceilings, the floors and the drift-frame re-injection, as for
   any recovered state.
6. Publication: $s$ is left as transported where nothing acted, and is rewritten from the
   published state wherever the fallback, a floor, a ceiling or a limit acted; the conserved state is rebuilt from the published primitives there.
   In fixed-metric GR hydro every such rebuild of an owned cell (the kept primitives of a
   double failure included, whose $D$ is not the one the fluxes produced) books its change of
   $D$ and $\tau$, weighted with the stage's share of the step, in the floor ledger
   (`Mesh::ecounter.eos_floor_mass`, `eos_floor_energy`), as Newtonian hydro does.

There is no ratio test, shock detector or threshold: only the failure set above triggers the
fallback. An energy root that is accepted is never replaced, whatever $s$ says.

The inversion of a coarse buffer for the primitive ghost fill, and that of a parent for the
primitive refinement of its children, run the same steps with that coarse cell's
$s=u_0^{\rm s}/D$, and the coarse buffer publishes $s$ by rule 6. A coarse cell has no last
primitives. Where both inversions fail it takes the atmosphere, and keeps its own
transported $s$ when $D>0$ and $s/D$ is finite; otherwise it takes the atmosphere's $s$
(`src/bvals/prolong_prims.cpp`).

## Resynchronisation

Shocks heat the gas, which the entropy equation does not know about, so $s$ is rewritten from
the thermal state of every cell, ghosts included, once at the end of every full step, after
the last endpoint pass (the reference method's entropy update): after the final
conserved-to-primitive conversion and the LAT delayed-reflux reconciliation. Intermediate stages are not resynchronised. Remaps into
a GR target from a source without the column, and checkpoints without it (below), are
resynchronised from their state after the first recovery.

A fresh start follows KORAL's order: the
initial data carry the entropy of the problem generator's own primitives, set in the owned
cells before the startup exchange
(`SetGasEntropyFromGenerator` in `driver.cpp`, through the same hook as the resync), so the
startup recovery runs KORAL's sequence like every later one. Where both of its roots fail,
a cell keeps the generator's primitives, which are KORAL's previous primitives at its first
inversion. A generator state that is not a gas state ($\rho$ or $p$ not positive) has no
entropy: its column is NaN, so no entropy root is taken there, and every ghost cell's column
is NaN until the exchange fills it. The ordinary resync follows the startup recovery, so
wherever that recovery took the energy root the column is the entropy of the recovered
state. A start from a `<remap>` source keeps the remap's rule.

Operators that change the thermal state inside a stage keep the column consistent by the exact
fixed-density change of the state they were evaluated on, at the cell's $D$:

- ISM, relativistic and disk cooling add
  $D\ln(1-\delta e/e)/(\Gamma-1)$ for a comoving internal-energy removal $\delta e$. A sink
  that removes all of $e$ or more takes the column to the entropy of the energy-floor state.

## Restart

Layouts are matched by width. A GR checkpoint without the column (every checkpoint of earlier
versions) restarts with the column written from the restored state; a checkpoint with it is
read verbatim and continues bit for bit. A checkpoint whose header says `dual_energy = true`
holds $D\kappa$ in that slot and is refused.

## Removed keys

`<mhd>/dual_energy = true`, `dual_energy_eta1` and `dual_energy_eta2` (and the `<hydro>`
equivalents) are fatal on every GR path; `dual_energy = false` is accepted. Delete them.

## Cost and memory

One more fluid variable: about 2.9 MiB per $32^3$ block at two ghost zones (3.6 at four),
counted for the 32^3 block. A cell that passes the energy inversion costs
one comparison; a cell where a floor acted costs one more inversion bracketed at the floored
root.

## Limitations

- The fallback cell's conserved energy is rebuilt from its entropy state, so energy is not
  conserved there, as in KORAL. The cells are those whose energy equation had already failed.
- For a tabulated gas the label inherits the bilinear interpolation error of the stored field;
  the end-of-step resynchronisation bounds it per step.
- In the cold dense corner of the Chabrier union tables, where the table's specific energy does
  not resolve the temperature, a fallback cell recovers its specific energy but not its
  temperature or pressure (above). The energy inversion has no more information there.
- Special relativity is not covered.

## Where in the code

- `src/eos/gr_gas_entropy.hpp`, `ideal_grmhd_c2p_impl.hpp`, `ideal_grhyd.cpp`: fixed-metric
  inversion; `src/eos/primitive_solver_hyd.hpp`, `src/eos/primitive-solver/`: dynamical.
- `src/eos/lte_table_core.hpp`: the adiabat label.
- `src/driver/driver.cpp` (`SetGasEntropyFromGenerator`): the column of a fresh start.
- `src/srcterms/gas_entropy_debit.hpp`, `srcterms.cpp`: in-stage cooling.
