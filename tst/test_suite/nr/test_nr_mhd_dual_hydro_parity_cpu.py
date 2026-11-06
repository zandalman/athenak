"""MHD dual energy with B = 0 reproduces hydro dual energy.

The Newtonian dual-energy formalism is one rule for both fluids (eos/general_c2p_hyd.hpp,
eos/dual_energy_shock_rule.hpp): the shock test once per step on the step-start state with
its two-sided reach, the step-final guard of a shocked cell, the owner's eta1 decision
before the exchange, and inversions in which only owned cells may read the energy
channel.  The fluids differ only in the field: E_M in the energy split, the field E's
fluxes carried, the MHD shock families.  With B = 0 those are exactly hydro's, so an MHD
run of a hydro deck must give hydro's answer.  Hydro's rule is the one validated on the
TDE stream head; a change to one fluid's rule that the other does not mirror fails here.

The cases are the regimes where the rule acts.  Colliding +-10 c streams carried at 300 c
(PLM and PPM4 with rk2, PLM with rk3 and rk4 for the guard's multi-stage offsets) and at
1000 c (PLM rk2, nghost = 2), where the eta1 test rejects the energy channel everywhere
and only the shock rule heats.  And a sheet: streams colliding at 7 c along x2, carried at
1000 c along x1 under uniform gravity along x1, where the source term moves the shocked
cells' channel after the rule took it and the step-final guard decides what they keep;
once on one block and once on two that meet at the collision, so that the owner's
decision before the exchange and the ghost zones that read the auxiliary act on the
shocked cells.

The gate is agreement to 1e-12 of every primitive of the final state (density, velocity
relative to the largest speed, internal energy per cell): with B = 0 both fluids run the
same reconstruction, HLLE fluxes, source terms and recoveries, which give the same
floating-point results, and the runs agree to the bit.  No looser gate would do.  Before
the MHD inversion settled its primitives as hydro's does, a last-place difference of the
velocity in a rejected cell grew, through the shock decisions and the guard that read it,
to 1e-9-2e-7 of the internal energy, the size of the difference the guard-free MHD rule
made in the rk2 tubes (2e-7-4e-5; 0.9 in the rk4 tube and on the sheet).
"""

from pathlib import Path
import os
import subprocess
import sys

import numpy as np
import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPOSITORY_ROOT / "vis" / "python"))

import bin_convert  # noqa: E402

TUBE = "inputs/dual_parity_tube.athinput"
SHEET = "inputs/dual_parity_sheet.athinput"
ZERO_FIELD = "".join(f"{key} = 0.0\n"
                     for key in ("bxl", "byl", "bzl", "bxr", "byr", "bzr"))
TOLERANCE = 1.0e-12

# (id, deck, overrides): {fluid} is the deck's fluid block, hydro or mhd
CASES = [("tube-U300-plm-rk2", TUBE, []),
         ("tube-U300-ppm4-rk2", TUBE, ["{fluid}/reconstruct=ppm4"]),
         ("tube-U300-plm-rk3", TUBE, ["time/integrator=rk3"]),
         ("tube-U300-plm-rk4", TUBE, ["time/integrator=rk4"]),
         ("tube-U1000-plm-rk2", TUBE, ["problem/ul=1010.0", "problem/ur=990.0",
                                       "mesh/nghost=2"]),
         ("sheet-gravity-plm-rk2", SHEET, []),
         ("sheet-gravity-plm-rk2-two-blocks", SHEET, ["meshblock/nx2=32"])]


def _mhd_deck(hydro_deck, run_directory):
    """The hydro deck with its fluid blocks and output renamed for MHD, and B = 0."""
    text = Path(hydro_deck).read_text()
    for old, new in (("<hydro>", "<mhd>"), ("<hydro_srcterms>", "<mhd_srcterms>"),
                     ("hydro_w", "mhd_w"), ("<problem>\n", "<problem>\n" + ZERO_FIELD)):
        text = text.replace(old, new)
    deck = run_directory / "mhd.athinput"
    deck.write_text(text)
    return str(deck)


def _run(run_directory, fluid, deck, overrides):
    run_directory.mkdir(parents=True)
    if fluid == "mhd":
        deck = _mhd_deck(deck, run_directory)
    command = ["./athena", "-i", deck,
               *[item.format(fluid=fluid) for item in overrides],
               "-d", str(run_directory)]
    environment = os.environ.copy()
    environment.setdefault("OMPI_MCA_coll_hcoll_enable", "0")
    result = subprocess.run(command, capture_output=True, check=False,
                            env=environment, text=True, timeout=600)
    assert result.returncode == 0, (
        f"{fluid} run of {deck} failed with return code {result.returncode}:\n"
        f"{(result.stdout + result.stderr)[-8000:]}"
    )
    dumps = sorted((run_directory / "bin").glob(f"*.{fluid}_w.*.bin"))
    assert len(dumps) >= 2, f"the {fluid} run wrote no final dump"
    data = bin_convert.read_binary(str(dumps[-1]))["mb_data"]
    return {name: np.asarray(data[name]) for name in
            ("dens", "velx", "vely", "velz", "eint")}


def _differences(hydro, mhd):
    """Largest relative differences of the final states, per primitive."""
    speed = max(np.abs(hydro[name]).max() for name in ("velx", "vely", "velz"))
    result = {"dens": np.abs(mhd["dens"] - hydro["dens"]).max()/hydro["dens"].max(),
              "eint": (np.abs(mhd["eint"] - hydro["eint"])/hydro["eint"]).max()}
    for name in ("velx", "vely", "velz"):
        result[name] = np.abs(mhd[name] - hydro[name]).max()/speed
    return result


@pytest.mark.parametrize("deck,overrides", [case[1:] for case in CASES],
                         ids=[case[0] for case in CASES])
def test_mhd_dual_energy_with_no_field_is_hydro(tmp_path, deck, overrides):
    """The MHD run with B = 0 ends on the hydro run's state to 1e-12."""

    hydro = _run(tmp_path / "hydro", "hydro", deck, overrides)
    mhd = _run(tmp_path / "mhd", "mhd", deck, overrides)
    assert (hydro["eint"] > 0.0).all(), "the hydro run holds a non-positive energy"
    differences = _differences(hydro, mhd)
    worst = max(differences, key=differences.get)
    assert differences[worst] <= TOLERANCE, (
        f"MHD with B = 0 departs from hydro by {differences[worst]:.3e} in {worst} "
        f"({deck} {' '.join(overrides)}; all: "
        f"{', '.join(f'{k} {v:.2e}' for k, v in differences.items())}): the dual-energy "
        f"rule of one fluid changed without the other (eos/dual_energy_shock_rule.hpp, "
        f"Hydro/MHD::ResolveDualEnergyChannel, the dual-energy inversions)"
    )
