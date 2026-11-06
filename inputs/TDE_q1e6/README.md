# TDE: 1 Msun star, M_BH/M_star = 1e+06, beta = 1, wegg BH potential

| Quantity | Code units | |
|---|---|---|
| r_t = R_* q^(1/3) | 50 | 100 Rsun |
| r_p = r_t/beta | 50 |  |
| r_g = G M/c^2 | 1.06125 | r_p/r_g = 47.1 |
| a_mb = r_t^2/(2 R_*) | 2500 |  |
| P_mb | 2784.16 | 40.95 d |
| v_p | 56.419 | 0.206 c |
| pericentre time | 21.27 |  |
| frame conversion | 174.231 | 0.05 P_mb after pericentre |
| end | 8373.76 | 3.00 P_mb after pericentre |
| sink radius | 6.36753 |  |
| softening | 0.05 |  |
| vceil, cs_ceil (BH frame) | 169.257 | 3 v_p |
| pfloor | 3.42288e-15 | T(dfloor, pfloor) = 5772 K, the stellar T_eff, with the EOS table |

| Step | t start | t end | Frame | Radiation | Box |
|---|---|---|---|---|---|
| 01_disruption | 0 | 21.12 | translating | LTE table | [-16, 16] x [-16, 16] x [-16, 16] |
| 02_debris | 21.12 | 60.94 | translating | LTE table | [-64, 64] x [-64, 64] x [-64, 64] |
| 03_debris_wide | 60.94 | 174.2 | translating | LTE table | [-256, 256] x [-256, 256] x [-128, 128] |


Steps 01_disruption, 02_debris, 03_debris_wide run on the LTE table with radiation pressure `chabrier2021_t13_helm_union_prad_640.table`, which sits in the parent directory of the step directories: a_R T^4 stays in the gas, as it does while the photon diffusion and emission times exceed the step.


## Running the chain, one step at a time

```bash
ATHENA=/path/to/build/src/athena        # PROBLEM=tde_external
DECKS=/path/to/athenak-transients/inputs/TDE_q1e6
export ATHENAK_DATA=/path/to/athenak-transients/data
mkdir -p run && cd run
ln -s /path/to/athenak-transients/eos_tables/chabrier2021_t13_helm_union_prad_640.table .

# step 01_disruption
mkdir -p 01_disruption && cd 01_disruption
cp $DECKS/tde_01_disruption.athinput deck.athinput
mpirun -np $NRANK <gpu binding wrapper> $ATHENA -i deck.athinput -d .
# an interrupted step continues with -r rst/<newest>.rst -d . (add -i deck.athinput for a file written before max_nmb_total)
cd ..

# step 02_debris
mkdir -p 02_debris && cd 02_debris
cp $DECKS/tde_02_debris.athinput deck.athinput
# set <remap>/source in deck.athinput to the newest ../01_disruption/rst/TDE_q1e06.NNNNN.rst
mpirun -np $NRANK <gpu binding wrapper> $ATHENA -i deck.athinput -d .
# an interrupted step continues with -r rst/<newest>.rst -d . (add -i deck.athinput for a file written before max_nmb_total)
cd ..

# step 03_debris_wide
mkdir -p 03_debris_wide && cd 03_debris_wide
cp $DECKS/tde_03_debris_wide.athinput deck.athinput
# set <remap>/source in deck.athinput to the newest ../02_debris/rst/TDE_q1e06.NNNNN.rst
mpirun -np $NRANK <gpu binding wrapper> $ATHENA -i deck.athinput -d .
# an interrupted step continues with -r rst/<newest>.rst -d . (add -i deck.athinput for a file written before max_nmb_total)
cd ..

```

A remap never runs on a restart (`-r`), so a step that stops is continued with `-r`; only the first start of a step remaps.  A restart file written before `mesh_refinement/max_nmb_total` existed lacks the key and the command line cannot add it, so continue such a run with the deck too: `-r rst/<newest>.rst -i deck.athinput -d .`.

Every deck states `mesh_refinement/max_nmb_total = 4000`, the number of MeshBlocks of the whole mesh (the refinement and the remap use 0.9 times it, whatever the rank count): 10 x `max_nmb_per_rank` for a ten-card layout, a choice of this release because no rank count was documented.  The run stops at startup if the total exceeds ranks x `max_nmb_per_rank`; with fewer than ten ranks raise `max_nmb_per_rank` to what the cards hold or lower the total.  Every deck also states `gravity/mg_bc = multipole` and `gravity/threshold = 1e-10`.

`<remap>/auto_expand` is off in every step.  Every face of every step is `outflow_only`: gas leaves the box and none enters. 
