"""
Multigrid self-gravity under local adaptive time stepping (LAT), CPU.

The gravity kernels and the multigrid source loader sit behind the LAT scheduling:
one Poisson solve per window at the window start, the window shortened so it never
exceeds ``gravity/solve_dt``.  These tests check the hydro Jeans growth rate under
LAT against the analytic rate, and that the refusal paths still fire.

Decks live in the repository's ``inputs/LAT/`` tree.  The suite runs from
``tst/build/src``, where ``inputs`` is a symlink to ``tst/inputs``, so the LAT
decks are reached through the source tree instead.
"""

import os
import shutil
import tempfile


# mg_utils pulls in testutils, which is what puts vis/python on sys.path.
from test_suite.multigrid.mg_utils import (
    run_athenak, parse_jeans_omega, cleanup,
)

LAT_INPUTS = os.path.join("..", "..", "..", "inputs", "LAT")

JEANS_HYDRO = os.path.join(LAT_INPUTS, "jeans_hydro_LAT.athinput")

# The unstable-Jeans tolerance of test_mg_jeans3d_cpu at its coarsest resolution.
MAX_ANALYTIC_REL_ERROR = 0.03
# The three runs share one mesh, one gravity cadence and one frozen-potential rule,
# so they may differ from each other by far less than they differ from the analytic
# rate.  2% is the contract; the measured spread is ~2e-4.
MAX_SPREAD = 0.02


def _omega(label, inputfile, flags):
    """Run one Jeans deck and return (measured, analytical) omega."""
    results = run_athenak(inputfile, flags)
    assert results[0], f"{label}: run failed"
    data = parse_jeans_omega(results[1])
    assert data is not None, f"{label}: no omega in output"
    return data


def _rundir():
    return tempfile.mkdtemp(prefix="lat_gravity_")


def test_lat_gravity_jeans_growth_rate_cpu():
    """Growth rate must not depend on whether LAT is on."""
    tmp = _rundir()
    try:
        cases = {
            "hydro+LAT": (JEANS_HYDRO, []),
            "hydro no LAT": (JEANS_HYDRO, ["time/lat=false"]),
        }
        measured = {}
        analytic = None
        for label, (deck, flags) in cases.items():
            data = _omega(label, deck, flags + ["-d", os.path.join(tmp, "run")])
            measured[label] = data["measured"]
            analytic = data["analytical"]

        for label, value in measured.items():
            rel = abs(value - analytic)/analytic
            assert rel <= MAX_ANALYTIC_REL_ERROR, (
                f"{label}: omega {value:.6e} is {rel:.3%} from the analytic "
                f"{analytic:.6e} (limit {MAX_ANALYTIC_REL_ERROR:.1%})")

        lo, hi = min(measured.values()), max(measured.values())
        spread = (hi - lo)/lo
        assert spread <= MAX_SPREAD, (
            f"growth rates disagree by {spread:.3%} (limit {MAX_SPREAD:.1%}): "
            + ", ".join(f"{k}={v:.8e}" for k, v in measured.items()))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
        cleanup()


def test_lat_gravity_requires_solve_dt_cpu():
    """The frozen-potential window needs gravity/solve_dt > 0."""
    tmp = _rundir()
    try:
        results = run_athenak(
            JEANS_HYDRO, ["gravity/solve_dt=0", "time/nlim=1",
                        "-d", os.path.join(tmp, "run")])
        assert not results[0], "hydro + LAT gravity ran without gravity/solve_dt"
        assert "requires gravity/solve_dt > 0" in results[1], (
            f"unexpected refusal message:\n{results[1][-2000:]}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
        cleanup()
