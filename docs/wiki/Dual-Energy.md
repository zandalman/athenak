# Dual energy

The dual-energy formalism evolves a second thermodynamic variable alongside the total
energy, so that pressure and temperature can still be recovered in cells where the usual
inversion loses them to round-off: $e_{\rm int}=E-E_K$ in highly supersonic gas (and
$E-E_K-B^2/2$ in low-$`\beta`$ MHD). Switch it on for Newtonian hydro or MHD where bulk
kinetic energy exceeds the thermal energy by many orders of magnitude, as in stellar
disruption and collisions; it is the production setting of all tabulated-EOS runs.

It is Newtonian only. General-relativistic runs do not use this formalism: they carry a gas
entropy column and recover from it where the energy inversion fails, always on and without a
key ([Gas entropy fallback](Entropy-Fallback)).

## How it works

One extra conserved column, the auxiliary $e_{\rm aux}$, is added to the fluid. It is an
internal-energy density with the equation $\dot e=-p\,\nabla\!\cdot\!\mathbf v$ and an
operator-split source, transported with its own upwind flux and face-velocity and heat buffers, for any
EOS with an internal energy (ideal, tabulated). General relativity and special relativity are
refused.

*Advection and compression.* $e_{\rm aux}$ is reconstructed like every other variable and
advected with the mass flux, $\mathcal F=\dot m\,(e_{\rm aux}/\rho)$ taken from the upwind
side. After the flux update the auxiliary takes a $p\,dV$ step with two parts.

*Volume change.* The cell follows its reversible adiabat on the face velocities the solver
used: the contact velocity on subsonic HLLC faces, the upwind velocity on supersonic ones,
$\dot m/\rho$ for the other solvers and for FOFC faces, and the live state's velocity at
excised faces.

```math
\nabla\!\cdot\!\mathbf v=\sum_d\frac{v_{d,i+1/2}-v_{d,i-1/2}}{\Delta x_d},\qquad
e_{\rm aux}\leftarrow
\begin{cases}
e_{\rm aux}\,e^{-(\Gamma-1)\nabla\cdot\mathbf v\,\Delta t} & \text{gamma-law (exact)}\\[2pt]
e_{\rm aux}-p(\rho,e_{\rm aux})\,\nabla\!\cdot\!\mathbf v\,\Delta t & \text{tabulated (explicit)}
\end{cases}
```

*Heat.* Each face's Riemann fan adds heat to each of its two sides. At the two-wave contact
$(S^\ast,p^\ast)$ of the face's reconstructed states, the side $s$ takes
$q_s=-(p^\ast-p_s)(S^\ast-u_s)\cdot\mathbf n\ge0$. The expression is written in velocity
differences, so it is Galilean invariant, and it never lowers the entropy. A side whose star
pressure is negative (a cavitating fan) takes $q_s=p_s\,|S^\ast-u_s|$, because its gas expands
into vacuum without work. In MHD the contact is taken on the total pressure $p+B^2/2$ with the
fast speeds, so a tangential discontinuity in total-pressure balance gives no heat. At $B=0$
the result is hydro's, bit for bit. The heat rate of a cell is $\dot q=\sum_f q_f A_f/V$ over
its faces, and the auxiliary takes $e_{\rm aux}\leftarrow e_{\rm aux}+\Delta t\,\dot q$ after
the volume change. Each face stores its velocity and its two heats in the flux-correction
buffers, and they are refluxed like fluxes.

Thermal floors are re-applied before and after, so a tabulated EOS never sees a state
outside its table.

*Which energy is used ($`\eta_1`$).* At every conserved-to-primitive conversion of an owned
cell the internal energy is taken from the total energy, $e_{\rm cons}=E-E_K$ (minus
$B^2/2$ in MHD), if it is positive and not lost in the subtraction,

```math
e_{\rm cons}>\eta_1\,E ,
```

and from the auxiliary otherwise ($\eta_1\le0$ always prefers $e_{\rm cons}$ when positive).
In MHD an additional test applies, because the energy and the constrained-transport update
discretise the magnetic energy differently and the mismatch is first order in the field
change: $e_{\rm cons}$ must also exceed $0.1\,B^2/2$. When the energy channel is rejected
the conserved energy is rewritten as $e_{\rm aux}+E_K\,(+B^2/2)$ (hydro books the residual
in the floor ledger), so total energy is not conserved in such a cell by exactly the
residual that was not thermal.

*Hydro and MHD: the owner decides.* Before a block's state leaves it (restriction, boundary
send, LAT refresh) the owner applies the same test to its cells: an accepted $e_{\rm cons}$
becomes the auxiliary, a rejected one is replaced as above. Every state a block hands on
satisfies $E-E_K\,(-B^2/2)=e_{\rm aux}$, ghost zones and coarse buffers read the auxiliary as
their thermal state, and the inversion publishes its thermal state in the auxiliary and
settles the primitives on the state it writes back. Both fluids follow this rule. There is
no second threshold in either fluid.

*The shock rule.* The auxiliary only compresses adiabatically, so it cannot know about shock
heating. A cell that is being shocked therefore takes $e_{\rm cons}$ whatever $\eta_1$ or the
magnetic test say, provided $e_{\rm cons}$ is resolved above its rounding
error, $e_{\rm cons}>32\,\epsilon\,(E+E_K\,[+B^2/2])$. It is taken as it is, above or below
the auxiliary: keeping only the stages where it exceeds the auxiliary rectifies the
stage-to-stage scatter of the update and over-heats the shocked gas. A cell is shocked when
gas converges into it faster than sound:

- across its own stencil, $\big[\sum_d (v_{d,-1}-v_{d,+1})\big]^2 > c_{s,{\rm aux}}^2$
  (hydro and MHD);
- into the steady compression it sits in: the constant mass flux
  $\rho(v-s)$ through a steady structure gives its speed $s=[\rho v]/[\rho]$ from the cells
  two off along each axis, and the inflow $|v-s|$ at the upstream-most stencil cell, summed
  in quadrature over the axes, must exceed $M_c\,c_s$ there, the Mach number of a shock
  whose velocity jump equals the upstream sound speed,
  $M_c=q+\sqrt{q^2+1}$, $q=(\Gamma_1+1)/4$ (1.87 for $\Gamma_1=5/3$). MHD measures the
  inflow per family (below).

The second test is needed because an upwind scheme spreads a shock over about
$|v_{\rm grid}|/\Delta v$ cells: colliding $\pm10\,c_s$ streams carried at 0, 300 and
1000 $c_s$ gave shocks 2, 6 and 11 cells wide, which the stencil test alone does not see.
A compression wave has $|v-s|=c_s$ and stays on the auxiliary (its damping by the scheme
at a large bulk speed is not physical heat). The same scheme puts part of a shock's
dissipation upwind of the front: where the hot downstream gas is upwind of the grid, into
the shock's post-sonic tail, whose stencil holds only shocked gas flowing in at 1.1-1.7
$c_s$, below $M_c$. So, where `nghost` leaves room, the test also counts a cell that
converges along an axis across both stencil pairs and has a shocked cell up to `nghost`
$-$ 2 cells to either side of it along that axis (the front is visible from the tail only
through those neighbours, whose test reads two cells further). Hydro and MHD apply the
rule to the auxiliary right after the stage's p dV update, on the owner: ghost zones and
coarse buffers receive the owner's auxiliary.

*One decision per step, and no shock that cools.* Hydro and MHD test for shocks once per
step, at the first stage on the step-start primitives, and hold the decision through the
step (`dual_shocked`). A stage's $E-E_K$ is not a thermal energy on its own: the energy
update does the work of the stage-start velocity, so $E-E_K$ misses the kinetic energy
$\tfrac12\rho|\Delta{\bf v}|^2$ of the cell's velocity change over the stage, and an
averaging stage adds the kinetic energy that mixing the time levels dissipates. Over a
whole RK step the two cancel to the order of the scheme; taken at some stages and not at
others they leave a bias of about $\tfrac12\rho|\Delta{\bf v}|^2$ (in a 1D model of the
update, taking $e_{\rm cons}$ only at the averaging stage gives 1.4-2.9 times the
Rankine–Hugoniot entropy at a bulk Mach number of 300-1000). A shocked cell takes
$e_{\rm cons}$ at every stage before the step's last; at the last stage it takes the step's
$e_{\rm cons}$ only if that has not fallen below $e_{\rm aux}$ as the first stage's channel
left it, since a shock never lowers the entropy, and otherwise returns to the adiabat it
would have followed without the earlier stages' channel (`dual_shock_debt` holds how far
they moved it, and the first stage's part). The reference carries no later stage's
channel: that of an averaging stage holds the mixing heat of the time levels, and with it
rk3 lost 11 and 59 % of the post-shock entropy on the tail side of the colliding streams
at 300 and 1000 $c_s$ (`nghost` = 2; 2.3 and 5.6 % without it, the post-sonic tail below).
For a two-stage step the reference is $e_{\rm aux}$ itself. With rk4 the offsets carried by
the second register $u_1$ are tracked as well. With PLM the condition selects
nothing in a real shock at the step's end: the colliding streams at 0-1000 $c_s$ keep every printed digit
with rk2 and rk3 (applied at every stage it would rectify the stages' errors, 1.4-2.9
times the Rankine–Hugoniot entropy; against the bare adiabat it rectifies the scatter of
the smeared front, 1.09-1.16 with rk3). MHD applies the same step-final condition and
`dual_shock_debt` as hydro, with the magnetic channel $E-E_K-|{\bf b}|^2/2$. With PPM4 and rk3
or rk4 at 300 $c_s$ the condition rectifies the front-cell scatter of a smeared shock in both
fluids: K/K_RH 1.52-1.65 in MHD, and 1.56 and 1.04 with rk3, 1.50 and 1.04 with rk4 in hydro,
against 0.977-0.996 for the conservative runs. No dual-energy deck uses that combination; the
defect is open. It matters where a flagged cell's channel drifts below
the adiabat step after step: in the q = $10^4$ TDE restart a vertical sheet one to two
cells thick near the black hole converges supersonically onto itself at a bulk Mach number
of 100-4000, and its $E-E_K$ falls about 0.4 % of the thermal energy per stage below the
adiabat; taken as it was, patches of the stream fell to the temperature floor.

*MHD: the steady test per family.* The jump threshold stays on the gas sound speed $c_s$,
the speed of the energy the auxiliary carries. A shock of family $k$ (fast or slow) with
upstream speed $c_k$ along the normal and adiabatic index $\Gamma_k$ has the velocity jump
$\Delta v=2(u^2-c_k^2)/((\Gamma_k+1)u)$; setting it to $c_s$ gives the inflow threshold

```math
u_k=q_k\,c_s+\sqrt{q_k^2c_s^2+c_k^2},\qquad q_k=\frac{\Gamma_k+1}{4},\qquad
\Gamma_f=\frac{\rho c_f^2}{p+B^2/2},\quad \Gamma_s=\frac{\rho c_s^2}{p}
```

(`DualEnergyShockInflowSq`), with $c_f$ and $c_{\rm slow}=c_s|B_d|/(\sqrt\rho\,c_f)$ along
the axis at the upstream point. The squares $|v-s|^2/u_k^2$ are summed over the axes per
family; the slow family counts only where the inflow is below the Alfvén speed along the
axis, as a slow shock's is. A simple wave of either family flows in at its own speed and
is never flagged; at $B=0$ the test is hydro's. A low-$`\beta`$ fast shock that is weak
against $c_f$ still doubles the gas entropy ($`\beta`$ = 0.06: $\Delta v=0.52\,c_f=2.4\,c_s$),
which a threshold on $c_f$ would miss.

*MHD: the channel of a shocked cell.* A shocked cell reads $e_{\rm cons}=E-E_K-|{\bf b}|^2/2$
with the cell field ${\bf b}$ that E's fluxes carried, not the constrained-transport field:
E's Poynting flux through a face is consistent with that face's EMFs (the Riemann
solver's), CT moves the field with the edge EMFs, and the two cell fields differ to first
order in the field change, which is exactly the magnetic error the $0.1\,B^2/2$ test
exists for. ${\bf b}$ is the CT field after the stage plus the curl of the face EMFs minus
the face-averaged curl of the edge EMFs, with CT's stage weight; in 1D, and for planes
along the grid, the two fields coincide. A cell the inversion then rejects publishes that
channel and resets E from the CT field, so E is not conserved there by the mismatch. With
the first-order error out of the channel the steady test runs in every cell: where it
misfires (in a uniform low-$`\beta`$ atmosphere the steady speed of two counter-propagating
waves takes any value) the cell takes the scheme's damping heat of the waves, second order
in their amplitude. Read against the CT field, the steady test (on the fast speed) heated a
low-$`\beta`$ star atmosphere 2.8-15 times. The correction uses the corner EMFs' edge EMFs only. The
edge EMFs that other modules add after them (resistivity, the 2D shearing box) are kept apart
(`dual_efld_added`, computed only with dual energy) and enter ${\bf b}$ through CT, as in the conservative
scheme. On a coarse-fine edge the added EMF is fine minus coarse.

## Deck parameters

Three keys for MHD, two for hydro; the auxiliary column's index is derived.

| key | default | meaning |
|---|---|---|
| `<hydro>/dual_energy`, `<mhd>/dual_energy` | `false` | enable the formalism |
| `dual_energy_eta1` | `1.0e-3` | energy-channel acceptance ratio $e_{\rm cons}/E$; $\le0$ always prefers the energy channel |

The two fluids use the same keys. A deck that sets `<mhd>/dual_energy_eta2` is fatal, since
neither fluid has a second threshold.

Requirements, all checked at startup: `<time>/evolution = dynamic`; an EOS with an internal
energy and a Riemann solver that supplies a face velocity (hydro `llf`, `hlle`, `hllc`, `roe`;
MHD `llf`, `hlle`, `hlld`); not in special or general relativity (the keys are fatal in
general relativity: [Gas entropy fallback](Entropy-Fallback)).

## Minimal example

Newtonian, tabulated EOS (`inputs/TDE_examples/tde_01_disruption.athinput`):

```
<hydro>
dual_energy       = true
dual_energy_eta1  = 1.0e-2
```

## Outputs and diagnostics

- MHD output bundles (`mhd_u`, `mhd_w` and the `bcc` and radiation variants) include the
  auxiliary under the names `reint_aux` (conserved) and `eint_aux` (primitive). The hydro
  bundles do not write it.
- The AMR criteria accept `hydro_w_eaux` and `mhd_w_eaux` as `rvariable`; using them
  without the formalism aborts at set-up.
- The tabulated-EOS startup line reports `dual_energy=on|off`.

## Interactions

| module | behaviour |
|---|---|
| [Tabulated EOS](Tabulated-EOS) | production use; the Newtonian flavour compresses with the table pressure; AMR prolongs primitives |
| AMR/SMR | prolongation goes through primitives. After a regrid the auxiliary of new cells is re-floored |
| LAT | hydro only (MHD with LAT is refused). Every refresh makes the owner's decision before it sends, whether or not it restricts. Coarse/fine corrections reflux the face velocity and the heats of the auxiliary's $p\,dV$ |
| FOFC | flagged faces take the first-order solve's face velocity and heat for the auxiliary |
| cooling source terms | ISM, relativistic and disk cooling charge the auxiliary additively. Any new internal-energy sink must do the same, or the $\eta_1$ test hands the energy back |
| [Remapping](Remapping) | Newtonian remap floors the auxiliary of the source's outer ghost ring. With `band_mode = keep` the target's own cells are re-seeded from the total energy at start-up; in floor-fade mode the remap writes the auxiliary itself (the source's, or $e_{\rm int}$ of the remapped state) |
| restart | a checkpoint carrying the column is used verbatim. Switching the formalism on against a checkpoint without it seeds the column once; switching it off drops it with a one-line notice |

## Limitations and known issues

- **Cost.** One more column in every array, face-velocity buffers and a flux-correction
  payload with the face heats; in both fluids a flag and a real per owned cell (two with
  three-stage integrators, four with rk4) and, once per step, a shock test, which reads two cells either side along each axis of a
  converging cell, and `nghost` cells when `nghost` $\ge$ 3 (the downwind neighbours'
  tests).
- **Hydro output.** The hydro bundles do not write the auxiliary; use MHD output, an AMR
  `rvariable`, or the restart file to inspect it.
- **Energy in MHD.** Energy is not conserved in cells where the magnetic test rejects the
  energy channel, by the amount that was truncation error.
- **PPM4 at rk2 and a large bulk Mach number.** At 1000 $c_s$ the left slab of two colliding
  streams is about 0.02-0.03 K/K_RH above the conservative scheme, in both fluids. The likely
  cause is the uneven flagging of the ramp of a wide captured shock. It is not yet fixed.
- **MHD shocks at a large bulk Mach number.** The steady family test recovers the
  shocks an upwind scheme smears at a large bulk speed (1D model of the update, rk2:
  colliding $\pm10\,c_s$ streams with a perpendicular field carried at 300 and 1000
  $c_s$ keep $K/K_{\rm RH}$ 0.976-1.010 at $`\beta`$ = 1 and 0.935-1.004 at
  $`\beta`$ = 0.1 with `nghost` = 2, 0.988-1.011 with `nghost` = 4; the stencil test alone
  0.63-0.72 and 0.19-0.59). Open: the low-$`\beta`$ Brio–Wu tube carried at Mach 300,
  whose slow shock, contact and compound wave overlap in one ~85-cell structure that forms
  at $t=0$ and is never steady, keeps 0.87-0.92 of the internal energy of the same tube at
  rest (dx to dx/4; 0.77 with the stencil test alone). An oracle that takes every
  converging cell of that structure gives 1.05; a flag held while the cell keeps converging
  takes the compression heat behind a slow shock (6-11 % above the rest frame). The
  conservative scheme is no reference: at Mach 300 it holds 1.28 times the heat at rest
  (1.10 at dx/4), in cells that do not converge. See `general_c2p_mhd.hpp`.
- **Hydro shocks at a large bulk Mach number.** With `nghost` = 2 (every PLM deck) the
  post-sonic tail upwind of the front keeps the auxiliary: on the side where the hot
  downstream gas is upwind, 0.5 and 3 % of the shock entropy are lost at a bulk Mach
  number of 300 and 1000 ($K/K_{\rm RH}$ 0.995, 0.972; 0.961, 0.932 when the shock test
  was decided per stage). The test reaches `nghost` $-$ 2 cells to both sides along each axis, in hydro and
  MHD. With PPM4 at beta 0.1, 300 $c_s$ and rk2 the K/K_RH is 0.957-0.962 with the downwind reach and 0.973 with
  both sides (conservative 0.9735). Larger `nghost` (1D model: 1.007,
  0.993 with `nghost` = 3; 1.009, 1.006 with 4; 1.009, 1.015 with 6; the other side
  within 0.991-1.011).

## Tests

`tst/test_suite/nr/test_nr_mhd_dual_hydro_parity_cpu.py` (MHD with $B=0$ matches hydro on the
same deck: seven cases of shock tubes and colliding sheets on one and two blocks, tolerance
$10^{-12}$ on every primitive), `tst/test_suite/nr/test_nr_mhd_lowbeta_shock_dual_cpu.py`
(low-$`\beta`$ Brio–Wu shock keeps its heating), `tst/test_suite/nr/test_nr_mhd_boost_dual_cpu.py` (Galilean gates at Mach
300: perpendicular colliding streams within 2 % of the Rankine–Hugoniot entropy with
`nghost` = 4, and the low-$`\beta`$ Brio–Wu structure holds at least 0.84 of the internal
energy of the tube at rest). Colliding-stream tests: streams carried at 0, 300 and 1000 $c_s$ (post-shock
pressure within 2 % of Rankine–Hugoniot, HLLE and HLLC; with `nghost` = 4 also the
entropy of both shocked slabs, along $x_1$ and $x_2$; rk2, and rk3 and rk4 against their
conservative runs) and an entropy wave carried at 1000 $c_s$ (pressure uniform to
$10^{-10}$); in MHD the same with a perpendicular field ($`\beta`$ = 1 and 0.1, against the
MHD Rankine–Hugoniot state), an entropy wave at uniform pressure and field carried at
1000 $c_s$ (pressure uniform to $10^{-10}$, rk1-rk4) and a $`\beta`$ = 0.01 fast wave at rest
and at 1000 $c_s$ (no heating beyond the stencil test's). Where the conservative scheme of the same
reconstruction and integrator misses the Rankine–Hugoniot state at that Mach number, the band of the PPM4 rows
is the conservative value plus or minus 0.01 (PPM4 at 300 $c_s$: 0.9735 and 0.9969 at beta 0.1 with rk2,
0.9765-0.9779 at beta 1). The dual scheme with every cell flagged reproduces it (0.9732).

## See also

[Gas entropy fallback](Entropy-Fallback) (the general-relativistic counterpart),
[Tabulated EOS](Tabulated-EOS), [TDE](TDE),
[Local adaptive time stepping](Local-Adaptive-Time-Stepping), [Remapping](Remapping).
Method reference: Bryan et al. 1995 (the $\eta_1$ switch).

## Where in the code

- `src/eos/dual_energy_shock_rule.hpp`: the shock rule, the auxiliary's $p\,dV$ and the face
  heat, shared by both fluids (`DualEnergyStencilShocked`, `DualEnergyCellShocked`,
  `DualEnergyAuxiliaryCompressed`, `DualEnergyFaceContact`, `DualEnergyMHDFaceContact`,
  `DualEnergyShockedStageEint`, and the MHD family test `DualEnergyAddShockInflow`).
- `src/hydro/hydro_dual_energy.cpp`, `src/mhd/mhd_dual_energy.cpp`: seeding, the owner's
  decision (`ResolveDualEnergyChannel`), the stage update, regrid repair.
- `src/hydro/hydro.cpp`, `src/mhd/mhd.cpp`: key reading, refusals.
- `src/eos/general_c2p_hyd.hpp`, `general_c2p_mhd.hpp`, `ideal_hyd.cpp`, `ideal_mhd.cpp`,
  `saha_table_hyd.cpp`, `saha_table_mhd.cpp`: Newtonian $\eta_1$ switch and the primitive
  recovery.
- `src/hydro/hydro_fluxes_impl.hpp`, `src/mhd/mhd_fluxes_impl.hpp`, `*_fofc.cpp`: face velocity
  and heat payloads.
- `src/srcterms/srcterms.cpp`: source-term coupling.
- `src/mesh/mesh_refinement.cpp`, `src/bvals/prolong_prims.cpp`, `src/remap/`,
  `src/pgen/pgen.cpp`, `src/driver/driver.cpp`: AMR, remap, restart, seeding.
