"""The LAT gravitational-work refund must not depend on how the blocks sit on the ranks.

The window-end refund pays the work of the mass mismatch a faster neighbour moved through
a coarse/fine or mixed-cadence face, less the part SourceTerms::Gravity recorded as
refused by the coupling weight w(rho) of the face cell.  The record is made in the stage
the mass arrives.  On a rank whose blocks are all slow, the ticks between their steps
have no active block, and a source term that returns before the record leaves that mass
unweighed until a later stage, where the face cell holds other gas: the refund then pays
work the source term never applied.  One rank always has an active block, so the total
gravitational work of the run, the energy gained, then differs between one rank and ten.

The deck keeps the weight below one for the whole run (rho_grav_min above every density),
so every mismatch has a refused part.  15 blocks on 10 ranks leave ranks that hold only slow
blocks, and the two vacuum blocks away from the gas make same-level pairs of two factors
(a block's CFL step includes its ghost cells, so a vacuum block next to the gas is as fast
as the gas).  The work is the same to round-off for any layout once the record follows the
fluxes booked on every rank: 1 and 10 ranks agree to 1e-13 on this deck.  A build that
returns early differed by 3.0e-2 of the work on the previous, 11-block deck (measured then;
not re-measured on this one).
"""

from pathlib import Path
import os
import re
import subprocess

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
INPUT = REPOSITORY_ROOT / "tst" / "inputs" / "hydro_lat_grav_refund_ranks.athinput"
NRANKS = int(os.environ.get("ATHENAK_TEST_RANKS", "10"))
MAX_RELATIVE_DIFFERENCE = 1.0e-8


def _run(nranks, run_directory):
    command = ["mpirun", "--oversubscribe", "-np", str(nranks), "./athena",
               "-i", str(INPUT), "-d", str(run_directory)]
    environment = os.environ.copy()
    environment.setdefault("OMPI_MCA_coll_hcoll_enable", "0")
    result = subprocess.run(command, capture_output=True, check=False,
                            env=environment, text=True, timeout=600)
    assert result.returncode == 0, (
        f"refund run on {nranks} ranks failed with return code "
        f"{result.returncode}:\n{(result.stdout + result.stderr)[-8000:]}")
    return result.stdout


def _work(run_directory):
    """Total energy the run gained, which is the gravitational work it applied."""
    history = np.loadtxt(next(run_directory.glob("*.hst")))
    return history[-1][6] - history[0][6]


def test_lat_gravity_refund_is_independent_of_the_rank_count(tmp_path):
    """One rank and NRANKS ranks apply the same gravitational work."""

    log = _run(1, tmp_path / "one")
    _run(NRANKS, tmp_path / "many")

    mixed = [int(value) for value in
             re.findall(r"AMR-interface status.*?mixed_factor=(\d+)", log)]
    assert mixed and max(mixed) > 0, "the deck has no coarse/fine face between factors"
    same_level = [int(value) for value in
                  re.findall(r"same-level status .*?mixed_factor=(\d+)", log)]
    assert same_level and max(same_level) > 0, "the deck has no mixed same-level pair"

    work_one = _work(tmp_path / "one")
    work_many = _work(tmp_path / "many")
    assert abs(work_one) > 0.0
    difference = abs(work_many - work_one)/abs(work_one)
    assert difference <= MAX_RELATIVE_DIFFERENCE, (
        f"the work is {work_one:.13e} on one rank and {work_many:.13e} on {NRANKS}, "
        f"{difference:.3e} apart: the refund depends on the rank layout"
    )
