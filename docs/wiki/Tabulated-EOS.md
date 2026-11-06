# Tabulated EOS (LTE and Saha)

AthenaK can replace the gamma-law closure of Newtonian hydro and MHD with a precomputed
table of an H/He gas in local thermodynamic equilibrium (LTE): pressure, internal energy,
sound speed, adiabatic indices and the ionisation and dissociation state, all as functions
of density and temperature. Use it when a constant $\gamma$ is wrong because the gas
ionises, dissociates H$`_2`$, becomes degenerate or radiation-dominated, as in stellar
envelopes, tidal-disruption debris and star-black-hole collisions. A simpler hydrogen-only
Saha table is also provided. The tables are made offline by a Python generator and read at
startup; at run time every conversion between conserved and primitive variables is one
table lookup. The Newtonian path described first is not available in special or general
relativity; the general-relativistic MHD solver on a dynamical spacetime reads the same
gas-only tables through its own policy, `dyn_eos = lte_table`, described in
[Dynamical-spacetime GR](#dynamical-spacetime-gr-dyn_eos--lte_table).

## How it works

### What a table holds

A table is a 2-D grid uniform in $(\ln\rho,\ \ln T)$ with $\rho$ in g cm$`^{-3}`$ and $T$ in
K. On every node it stores $\ln P$, $\ln\varepsilon$ (specific internal energy),
$\ln c_s^2$, the adiabatic indices $\Gamma_1$ and $\Gamma_3-1$, the species fractions
(H$`_2`$, H$`^+`$, He$`^+`$, He$`^{2+}`$), the mean molecular weight $\mu$ and the radiation
pressure fraction $\beta_{\rm rad}$. The generated H/He tables also carry about twenty
diagnostic columns (heat capacities, entropy, pair-gas pressure, the map of which model
supplied each cell) that the run ignores. The thermodynamic closure is

```math
\Gamma_1=\chi_\rho+\chi_T(\Gamma_3-1),\qquad
\Gamma_3-1=\frac{P\chi_T}{\rho\,T\,c_v},\qquad
c_s^2=\Gamma_1\frac{P}{\rho},
```

with $`\chi_\rho=(\partial\ln P/\partial\ln\rho)_T`$ and $`\chi_T=(\partial\ln P/\partial\ln T)_\rho`$.
In the `_prad` tables a blackbody component $P_{\rm rad}=a_{\rm rad}T^4/3$,
$\varepsilon_{\rm rad}=a_{\rm rad}T^4/\rho$ is added on top of the gas.

### Table families

The generator `scripts/generate_lte_table.py` builds three H/He models and
`scripts/generate_saha_table.py` the hydrogen-only reference. The method write-up with all
derivations is in [Further reading](#further-reading).

| model | what it is | useful range |
|---|---|---|
| `saha_table` | pure hydrogen, H $\rightleftharpoons$ H$`^+`$ + e$`^-`$ Saha equilibrium, $x^2/(1-x)=C\,T^{3/2}e^{-T_{\rm ion}/T}/\rho$; $P=\rho R_{\rm H}T(1+x)$ | $\rho$ $10^{-20}$–$`10^{2}`$, $T$ $10^{-8}$–$`10^{12}`$ K |
| `t13` | Tomida et al. (2013) partition-function equilibrium of H$`_2`$, H, H$`^+`$, He, He$`^+`$, He$`^{2+}`$, e$`^-`$; relativistic e$`^-`$/e$`^+`$ pair gas above $10^7$ K | $\rho$ $10^{-20}$–$`10^{2}`$, $T$ $0.5$–$`10^{9}`$ K |
| `scvh_t13_cp_helm_union` | Saumon–Chabrier–Van Horn (1995) wherever it is valid, a fully ionised Coulomb-liquid corner, a HELM pair-gas corner, and `t13` at low density | $\rho$ $10^{-20}$–$`10^{2}`$, $T$ $1$–$`10^{9}`$ K |
| `chabrier2021_t13_helm_union` | `t13` at low density, blended to the Chabrier et al. (2021) dense H/He fluid between $10^{-3}$ and $10^{-2}$ g cm$`^{-3}`$, HELM above $10^8$ K at high density | $\rho$ $10^{-20}$–$`10^{3}`$ or more, $T$ $1$–$`10^{10}`$ K |

Composition is a per-table property: hydrogen mass fraction $X$ and helium mass fraction
$Y=1-X$. Metals are not in the EOS, so a metal-enriched gas is represented by folding $Z$
into $Y$, which gives the closest mean molecular weight. Where the Chabrier
mixture tables (published only for $Y=0.275$–$`0.297`$) cannot represent the requested $Y$,
the generator builds the dense-fluid block from the pure-H and pure-He tables with an
additive-volume mixing law.

### Run-time lookup

Inverting $\varepsilon(\rho,T)$ for $T(\rho,\varepsilon)$ is done once at startup, on the
host, for every density row, on a grid uniform in $\ln\varepsilon$ with
`lte_cache_eps_factor` $\times$ (number of temperature nodes) points. The result, an inverse
cache of $T$, $P$, $c_s^2$, $\Gamma_1$, $\Gamma_3-1$ and the composition fields, is
uploaded to the device. A conversion then takes $(\rho,e_{\rm int})$ from code units to
cgs and does one bilinear interpolation in $(\ln\rho,\ln\varepsilon)$; there is no root
find in the hot path. Temperature from $(\rho,P)$ binary-searches the monotonic
$\ln P(\ln T)$ row.

### Floors, ceilings and out-of-range queries

Density, pressure, temperature and sound-speed limits are turned into one energy floor
and one optional energy ceiling per density row, computed at startup. The floor is the
specific energy at the temperature

```math
T_{\min}(\rho)=\max\bigl\{T_{\rm table},\ T_{\rm floor},\ T(\rho,P_{\rm floor})\bigr\},
```

the largest of the table's lowest temperature, the deck's temperature floor and the
temperature at which the pressure equals `pfloor`; so `pfloor` acts as a
density-dependent temperature floor. The ceiling is the energy at which $c_s$ first
exceeds `cs_ceil`. A cell with $\rho<$ `dfloor`
is reset to the atmosphere (density `dfloor`, zero velocity, floor energy).

Queries outside the table follow `lte_bounds`. With `error` (default) the run aborts with
a message naming the quantity. With `clamp` density is held at the table edge, energies
below the cold edge are held at it, and above the last temperature row $P$, $\varepsilon$
and $c_s^2$ continue as the power law of the last tabulated temperature interval
(radiation and pairs make them scale as $T^4$ there, so the continuation is accurate);
bounded fields such as $\mu$ and the fractions hold their top value.

### Face states

For Newtonian hydro and MHD with any reconstruction except donor cell, the specific energy
$\varepsilon=e_{\rm int}/\rho$ is reconstructed with the run's own limiter and the face
internal energy is rebuilt as $e_{\rm int}^{\rm face}=\rho^{\rm face}\varepsilon^{\rm face}$.
Limiting $\rho$ and $e_{\rm int}$ independently can give an unbounded $e_{\rm int}/\rho$ at a
steep atmosphere gradient, and with it an artificially hot Riemann state. The same is done
for the dual-energy auxiliary ([Dual energy](Dual-Energy)). The conservative total
energy update is unchanged.

### Temperature unit

The code temperature of a tabulated run is not the one of `<units>/mu`. The tables carry
the mean molecular weight themselves, so the code unit is fixed to
$T_{\rm unit}=v_{\rm unit}^2\,m_{\rm H}/k_{\rm B}$ with $v_{\rm unit}=$ `length_cgs`/`time_cgs`,
independent of `mu`. Every consumer of a temperature scale (radiation constants, opacity
conversion, cooling, problem generators) must read it from the active
EOS. Using the ideal-gas unit instead is wrong by
$\mu m_{\rm u}/m_{\rm H}$ in $T$ and four times that in $a_{\rm R}T^4$.

## Choosing the EOS name and table

The deck names the family with `eos` and the file with `table`. The name must agree with
the `table_type` stored in the file, and a `_prad` name requires a table that includes
radiation (a name without `_prad` requires one that does not); otherwise the run stops at
startup.

| `eos` | table | notes |
|---|---|---|
| `saha_table` | `saha_hydrogen.table` | hydrogen only, $X=1$ |
| `lte_table_t13_prad` | `lte_t13_prad_eos.table` | $T$ from 0.5 K; H$`_2`$ zero-point energy subtracted |
| `lte_table_scvh_t13_cp_helm_union_prad` | `scvh_t13_cp_helm_union_prad_640.table` | SCvH union |
| `lte_table_chabrier2021_t13_helm_union_prad` | `chabrier2021_t13_helm_union_prad_640*.table` | production choice; includes $a_{\rm rad}T^4$ |
| `lte_table_chabrier2021_t13_helm_union` | `chabrier2021_t13_helm_union_640.table` | same gas without radiation |

**Radiation.** Use a `_prad` table when radiation is only a pressure and energy
reservoir; the gas-only table is for runs where an explicit radiation module exchanges
energy with the gas, otherwise the trapped radiation would be counted twice.

The loader also recognises other names (`lte_table_hhe*`, `lte_table_scvh1995_hhe*`,
`lte_table_scvh_t13_union*`, `lte_table_scvh_t13_helm_union*`, `lte_table_scvh_t13_cp_union*`,
`lte_table_hybrid_hhe_t13*`, non-radiation `lte_table_t13`), but the generator cannot make
tables for them; treat them as unavailable.

## Deck parameters

All keys live in `<hydro>` or `<mhd>` and are read once at startup. A `<units>` block is
mandatory.

| key | default | meaning |
|---|---|---|
| `eos` | required | `saha_table` or one of the `lte_table_*` names above |
| `table` | required | table file; a relative path is searched in the working directory, then `$ATHENAK_DATA`, then `<source>/data` |
| `dfloor` | `FLT_MIN` | density floor (code units) |
| `pfloor` | `FLT_MIN` | pressure floor; raises the temperature floor row by row |
| `tfloor` | `FLT_MIN` | temperature floor in code units |
| `tfloor_kelvin` | unset | temperature floor in kelvin; if present it replaces `tfloor` |
| `sfloor` | `FLT_MIN` | **must be set to `0.0`**: any positive value is a fatal error, and the default is positive |
| `cs_ceil` | `0` | sound-speed ceiling in code units (0 = off) |
| `vceil` | `FLT_MAX` | velocity ceiling |
| `lte_bounds` | `saha_bounds`, else `error` | `error` or `clamp`, see above (`saha_bounds` for `saha_table`) |
| `lte_debug_checks` | `saha_debug_checks`, else `false` | extra load-time checks: $\Gamma_1=\rho c_s^2/P$ and an $\varepsilon\to T$ round trip |
| `lte_cache_eps_factor` | `8` | resolution of the inverse cache in $\ln\varepsilon$, in multiples of the table's temperature nodes (`saha_cache_eps_factor` for `saha_table`) |
| `lte_floor_cache_factor` | `16` | density resolution of the precomputed floor curve (`saha_floor_cache_factor` for `saha_table`) |
| `allow_reinterpretive_restart` | `false` | allow a restart with a different EOS or table than the checkpoint recorded |

The composition, radiation, H$`_2`$ and zero-point flags are read from the table file, not
from the deck. `dual_energy`, `dual_energy_eta1` and `dual_energy_eta2` are documented on
[Dual energy](Dual-Energy); the production decks run with `dual_energy = true`.

## Minimal example

From the first step of the TDE chain (`inputs/TDE_examples/tde_01_disruption.athinput`):

```
<hydro>
eos              = lte_table_chabrier2021_t13_helm_union_prad
table            = eos_tables/chabrier2021_t13_helm_union_prad_640.table
reconstruct      = plm
rsolver          = hlle
dfloor           = 1.0e-10
pfloor           = 1.0e-12
tfloor_kelvin    = 1
sfloor           = 0.0
cs_ceil          = 15.0
vceil            = 15.0
dual_energy      = true
lte_bounds       = clamp
```

For the hydrogen-only table use `eos = saha_table`, `table = eos_tables/saha_hydrogen.table`
and `saha_bounds` instead of `lte_bounds`.

## Getting the tables

The authors of the Chabrier et al. dense-fluid tables distribute them without a licence
that permits redistribution (`scripts/chabrier2021_data/README.md`). The script
`get_eos_table.sh` in the repository root downloads and checksums the five files and
builds the production table `chabrier2021_t13_helm_union_prad_640` (`--output-dir DIR`
redirects the output); it reports whether the numbers are identical to the table used for
the TDE paper. Any other table comes from the generator, which needs Python 3 with numpy
and scipy and finds the Chabrier files in `scripts/chabrier2021_data/`:

```
python3 scripts/generate_lte_table.py                         # every job
python3 scripts/generate_lte_table.py chabrier2021_t13_helm_union_prad_640_X745Y255
python3 scripts/generate_lte_table.py <job> --output-dir $ATHENAK_DATA/eos_tables
python3 scripts/generate_saha_table.py                        # saha_hydrogen.table
```

All jobs are 640 $\times$ 640 in $(\ln\rho,\ln T)$ and write to `eos_tables/` unless
`--output-dir` is given; an unknown job name prints the list. The SCvH source tables are in
`scripts/scvh95_data/`.

| job (file stem) | $X$ / $Y$ | $\log_{10}\rho$ | $\log_{10}T$ | in git |
|---|---|---|---|---|
| `lte_t13_prad` (`lte_t13_prad_eos`) | 0.70 / 0.30 | $-20$…2 | $-0.3$…9 | yes |
| `scvh_t13_cp_helm_union_prad_640` | 0.70 / 0.30 | $-20$…2 | 0…9 | yes |
| `chabrier2021_t13_helm_union_prad_640` | 0.70 / 0.30 | $-20$…3 | 0…10 | yes |
| `chabrier2021_t13_helm_union_640` (no radiation) | 0.70 / 0.30 | $-20$…3 | 0…10 | yes |
| `…_prad_640_X745Y255` | 0.745 / 0.255 | $-20$…4 | 0…10 | no |
| `…_prad_640_X620Y380` | 0.62 / 0.38 | $-20$…4 | 0…10 | no |
| `…_prad_640_X550Y450` | 0.55 / 0.45 | $-20$…4 | 0…10 | no |
| `…_prad_640_X200Y800` | 0.20 / 0.80 | $-20$…4 | 0…10 | no |
| `…_prad_640_X038Y062` | 0.38 / 0.62 | $-20$…4 | 0…10 | yes |
| `…_prad_640_X000Y100` | 0 / 1 | $-20$…5 | 0…10 | yes |
| `saha_hydrogen` (256 $\times$ 512) | 1 / 0 | $-20$…2 | $-8$…12 | yes |

The `…_prad_640_X…Y…` rows are the uniform-composition tables of the star-black-hole
composition campaign; the shared prefix is `chabrier2021_t13_helm_union`. Tables are kept
out of git by `.gitignore` (the seven already tracked stay tracked), so share new ones as
release assets with a `SHA256SUMS` file, the generator version and the job names. Because
a relative `table =` path is also searched under `$ATHENAK_DATA`, a deck can say
`eos_tables/<file>` and run from any directory once the table sits in
`$ATHENAK_DATA/eos_tables/`.

## Dynamical-spacetime GR (`dyn_eos = lte_table`)

The same gas-only H/He tables drive the general-relativistic MHD solver on a dynamical
spacetime (a deck with `<adm>` or `<z4c>`). The deck changes in one
value, `<mhd>/dyn_eos = lte_table`. The table is the file of `<mhd>/table`, and the code
units are the GR ones of `<units>`. The policy is `Primitive::EOSLTETable` in
PrimitiveSolver; the Newtonian path above
is untouched. **Status.** Written to the design of 2026-10-22. The tests V1 to V7 below are
written and have not been run; treat every number on this page as a specification until they
have.

### Units

The deck gives the black-hole mass and the density unit, `<units>/bhmass_msun` and
`<units>/density_cgs`. The length unit is $L=GM/c^2$, the time unit $L/c$, the velocity unit
$c$ and the pressure unit $\rho_u c^2$. The temperature unit is the Newtonian rule with
$v_{\rm unit}=c$,

```math
T_u=\frac{c^2 m_{\rm H}}{k_{\rm B}}=1.0894\times10^{13}\ {\rm K},
```

published through `EOS_Data::temp_unit_cgs`, so `MeshBlockPack::TemperatureUnitCGS()` stays
the one place every consumer reads it. `<units>/mu` is not used. The table is shifted into
these units once, when it is read, so the device never converts: $\ln\rho$ by
$\ln\rho_u$, $\ln T$ by $\ln T_u$, $\ln\varepsilon$ and $\ln c_{s,N}^2$ by $\ln c^2$ and
$\ln p$ by $\ln(\rho_uc^2)$. The baryon mass is the code mass unit, so $n=\rho$, the total
energy density is $e=\rho(1+\varepsilon)$, the enthalpy per baryon is
$h=1+\varepsilon+p/\rho$, and the sound speed is $c_s^2=c_{s,N}^2/h$, where the table's
$c_{s,N}^2=\Gamma_1p/\rho$. A run with `<gravity>` on has $G=G_{\rm cgs}\rho_ut_u^2$ in code units, with the code mass unit
$\rho_uL^3$ and the hole's mass $1/G$ in it; $G=1$ for $\rho_u=c^6/(G^3M_{\rm BH}^2)$
($6.18\times10^5$ g cm$`^{-3}`$ for $10^6\,M_\odot$). The GR TDE generator writes the power of ten nearest to the
star's central density over $10^4$, which puts the star's code density near $10^4$.

### Inversion

The conserved-to-primitive root of PrimitiveSolver evaluates $e(n,T(n,e))$ and expects $e$
back. A $(\rho,\varepsilon)$ cache whose temperature is not the inverse of the forward
$(\rho,T)$ interpolant would leave a residual of the size of its interpolation error (up to
$5\times10^{-3}$ between rows in a stellar box). The policy therefore inverts the forward
interpolant exactly, with no cache. The forward law is bilinear in $(\ln\rho,\ln T)$. At
fixed $\rho$ the $\varepsilon$ row blended with the density weights is piecewise linear in
$\ln T$ with the table nodes as breakpoints and strictly increasing (the loader checks it),
so a bisection over the nodes (about ten steps for 640 nodes) and one linear solve invert it
to round-off. $T(\rho,p)$ is inverted the same way. The pressure rows are monotone except in
the Chabrier cold-dense corner (below); there the first bracket the bisection meets is
returned.

### Outside the table

| Query | Behaviour |
|---|---|
| $T$ above the top | $\ln p$, $\ln\varepsilon$, $\ln c_{s,N}^2$ continue linearly in $\ln T$ with the slope of the last interval ($T^4$ for radiation and pairs); $\Gamma_3-1$ is held |
| $T$ below the bottom | $T$ is held at the table floor $T_{\min}$ |
| $\rho$ below the first row | the specific quantities $\varepsilon$, $p/\rho$, $c_{s,N}^2$, $\Gamma_3-1$ are held at the first row: the dilute ideal-gas limit, not the Newtonian density clamp (which keeps $p$ and lets $p/\rho$ blow up) |
| $\rho$ above the last row | the log fields continue as the power law of the last interval, and the startup line reports it |

Pairs and radiation in the hot top rows are the table's physics; the table has no
$\varepsilon<0$ and no $p\le0$, so $h\ge1$ everywhere and the policy's minimum enthalpy is 1.

### Floors, atmosphere and excision

The atmosphere is the dyn-GR `reset_floor` one: density `dfloor`, temperature `tfloor` (code
units, $T_u$) or `tfloor_kelvin` (kelvin, the Newtonian rule). The atmosphere pressure comes
from the table, so the floor temperature is a physical number. The conserved floor, the
Lorentz, magnetisation and $b^2/u$ ceilings and the mass loading go through the policy's
exact $T(n,e)$ and are correct in the ionisation zone, where $T$ is not linear in $u$. The
excision state is `dexcise` at `texcise`, a code temperature. PrimitiveSolver hands the
policy the total $e$ and the policy forms $\varepsilon=e/\rho-1$, so $\varepsilon$ carries a
relative round-off of about $10^{-16}/\varepsilon$ ($10^{-7}$ for 1e4 K gas, $10^{-10}$
inside a star). This is the same as `dyn_eos = ideal` and is not masked.

### Deck keys

One new value and no new key. The table, the units and the floors are read from keys that
already exist.

| Key | Value | Meaning |
|---|---|---|
| `<mhd>/eos` | `ideal` | must stay `ideal` (the placeholder EOS of every dynamical deck); any other value is refused |
| `<mhd>/dyn_eos` | `lte_table` | selects the policy |
| `<mhd>/table` | required | a gas-only table of the families above. A `_prad` table is allowed, its radiation pressure being part of the gas |
| `<units>/bhmass_msun`, `density_cgs` | required | the GR scales; `general_rel = true` is required |
| `<mhd>/dfloor`, `tfloor` or `tfloor_kelvin` | as for `dyn_eos = ideal` | atmosphere; the kelvin form replaces `tfloor` |
| `<mhd>/allow_reinterpretive_restart` | `false` | accept a restart written with another `dyn_eos`, table or composition |

### What is refused

| Configuration | Why |
|---|---|
| `<units>` missing or `<coord>/general_rel` false | no GR scales |
| `<mhd>/eos` other than `ideal` on a dynamical deck | a Newtonian LTE EOS would be built silently |
| `dyn_eos = lte_table` on a deck that is not dynamical | the key would be unread and the run ideal |
| disk cooling, a remap whose source or target closure differs | assume an ideal gas, or not validated |
| a restart whose `dyn_eos`, table, table type, $X$, $Y$ or radiation, $\mathrm{H_2}$ and zero-point flags differ | unless `allow_reinterpretive_restart = true` |

Supported: LAT, AMR, FOFC, `llf`/`hlle`/`hlld`, excision, GR self-gravity and passive scalars.

### Limitations

- **Cold-dense corner.** For $\rho\gtrsim10^{-2}$ g cm$`^{-3}`$ and $T\lesssim10^5$ K,
  Chabrier's printed $\log U$ cannot resolve the thermal energy: $\varepsilon(T)$ has plateaus
  with slope $10^{-14}$ and $p(T)$ is not monotone in 4442 cells of the $X=0.7$ table. $T(\varepsilon)$ stays unique but
  carries little information, $p$ is insensitive to the $T$ error (at most $10^{-2}$), and
  $T(p)$ on faces may pick another branch. A solar-type TDE box ($\rho$ 1e-3 to 1e2,
  $T$ 1e5 to 1e8 K) has $T$ errors of at most $5\times10^{-3}$. The table is not fixed here;
  the repair belongs in the generator.
- **Density above the table top.** A deep encounter compresses the gas by about
  $\beta^3\rho_c$; above $10^3$ g cm$`^{-3}`$ the EOS is the last interval's power law. Use a
  denser table for $\beta\gtrsim2$ with a solar star.
- Pair equilibrium above $10^{10}$ K in very dilute hot gas near the hole is the table's
  assumption.

## Outputs and diagnostics

At startup, rank 0 prints one line, `LTE table EOS startup: block=... table=... X=... Y=...
h2=... prad=... zpe_subtracted=... dual_energy=... cache_neps=... bounds=... restart=...`
(`Saha EOS startup: ...` for the hydrogen table). It also reports a restart that
reinterprets the EOS. Check it once for every new deck.

Nine derived output variables expose the table state of each cell, evaluated from its
density and internal energy: `hydro_temperature`, `hydro_xh2`, `hydro_xion`, `hydro_xhe1`,
`hydro_xhe2`, `hydro_gamma1`, `hydro_gamma3m1`, `hydro_mu`, `hydro_beta_rad`, and the
`mhd_*` equivalents. Temperature is in code units; multiply by $T_{\rm unit}$ above for
kelvin. A variable the table cannot supply stops the run at output setup: `*_xh2` needs a
table with H$`_2`$, `*_xhe1`/`*_xhe2` a table with helium, `*_beta_rad` a radiation table.
Floor, ceiling and failure counts appear in the event log (`file_type = log`).

## Interactions

- **AMR and SMR.** A Newtonian tabulated fluid always prolongs primitives rather than conserved
  energy across coarse/fine boundaries, so the thermal state stays consistent. On a dynamical spacetime
  (`dyn_eos = lte_table`) the fluid chooses its own coarse/fine ghost prolongation.
- **LAT.** The conversions run only over active blocks ([LAT](Local-Adaptive-Time-Stepping)).
- **FOFC.** The first-order flux correction flags a cell whenever a floor or ceiling would
  fire; there is no separate tabulated path.
- **Restart.** Every checkpoint records the EOS name, table path and type, $X$, $Y$ and the
  radiation, H$`_2`$ and zero-point flags. A mismatch at restart is fatal unless
  `allow_reinterpretive_restart = true`, in which case the conserved state is resumed but
  the thermodynamics are not those that produced it.
- **Remap.** Source and target runs must both be ideal or both tabulated
  ([Remapping](Remapping)).
- **Self-gravity and stellar set-ups.** The star and TDE problem generators integrate
  hydrostatic equilibrium through the same EOS ([TDE](TDE)).

## Limitations and known issues

- The Newtonian EOS is refused under special and general relativity at startup. The
  dynamical-spacetime GR MHD path has its own policy, `dyn_eos = lte_table`, section above.
  Fixed-metric GR has no tabulated H/He EOS.
- `sfloor` must be exactly zero; there is no entropy floor.
- No metals: $X+Y=1$ is enforced by loader and generator.
- ISM, relativistic and disk cooling source terms, and thermal conduction, are refused with
  a tabulated EOS because they assume a gamma-law temperature. The Roe solver is refused
  for hydro; use `llf`, `hlle` or `hllc` (MHD: `llf`, `hlle`, `hlld`).
- Memory: the inverse cache dominates. For a 640 $\times$ 640 H/He table at the default
  factor, the raw table is about 34 MiB and the cache about 275 MiB on the device, plus an
  equal host copy, per `<hydro>` or `<mhd>` block and per rank.
- Every rank reads and checks the table file itself at startup.
- The Chabrier et al. dense-fluid data cover $10^2$–$`10^8`$ K; outside that, and below
  $10^{-3}$ g cm$`^{-3}`$, the `t13` or HELM branches supply the state, and the transitions
  are blended, not exact.
- In the union tables the species fractions on dense Chabrier cells are diagnostic proxies
  from a simple atomic Saha solver; they do not feed back on the thermodynamics.
- Coverage by tests is mostly indirect: shock-tube and star-surface decks under
  `tst/inputs/` (`hydro_lte_thermo_tube`, `*_star_surface_eps*`),
  and the table checks run at every load.

## See also

- [Dual energy](Dual-Energy): the auxiliary internal energy that all production
  tabulated runs use.
- [EOS tables](EOS-Tables): building the tables.
- [TDE](TDE): the problem generator that runs on these tables

### Further reading

- Tomida et al. 2013, ApJ 763, 6 (Appendix 1); Saumon, Chabrier & Van Horn 1995, ApJS 99, 713;
  Chabrier, Mazevet & Soubiran 2019, ApJ 872, 51; Chabrier & Debras 2021, ApJ 917, 4;
  Timmes & Swesty 2000, ApJS 126, 501 and Timmes & Arnett 1999, ApJS 125, 277 (HELM
  electron–positron gas).

## Where in the code

- `src/eos/lte_table_utils.hpp`, `src/eos/saha_table_utils.hpp`: table loaders, checks,
  inverse cache, floor curves, restart gate, startup line.
- `src/eos/lte_table_core.hpp`, `src/eos/primitive-solver/eos_lte_table.{hpp,cpp}`: the
  dynamical-GR path (device table view, code units, exact inverses, the PrimitiveSolver policy).
- `src/eos/eos.hpp`, `src/eos/eos.cpp`: `EOS_Data`, interpolation kernels, base-class floors.
- `src/eos/saha_table_hyd.cpp`, `src/eos/saha_table_mhd.cpp`, `src/eos/general_c2p_hyd.hpp`,
  `src/eos/general_c2p_mhd.hpp`: conserved/primitive conversion.
- `src/reconstruct/specific_energy_recon.hpp`: specific-energy face states.
- `src/utils/tr_table.{hpp,cpp}`, `src/utils/data_path.{hpp,cpp}`: table container and file search.
- `scripts/generate_lte_table.py`, `scripts/generate_saha_table.py`,
  `scripts/chabrier2021_eos.py`, `scripts/scvh95_eos.py`, `get_eos_table.sh`, `eos_tables/`.
