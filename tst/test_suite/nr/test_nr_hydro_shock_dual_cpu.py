"""A hydro shock carried at a large bulk speed keeps its entropy under dual energy.

Two +-10 c streams (rho = 1, p = 0.6, gamma = 5/3, so c = 1) collide while carried at
300 c.  There the internal energy is 1e-5 of E and the eta1 test rejects the energy
channel everywhere; only the shock rule of the dual-energy formalism
(eos/general_c2p_hyd.hpp) gives the shocked gas its heat.  The upwind scheme spreads each
shock over several cells and, on the side whose hot downstream gas is upwind of the grid,
puts part of its dissipation into a post-sonic tail whose +-2 stencil holds only shocked
gas; with nghost = 4 the rule reaches the tail through its downwind neighbours.  With
nghost = 2 that side keeps 0.961 of the Rankine-Hugoniot entropy.

The gate: the entropy K = p/rho^gamma of both shocked slabs (0.3 to 0.8 of the shock's
distance from the collision point) and the pressure within 0.4 of it are within 2 % of
Rankine-Hugoniot.
"""

from pathlib import Path
import os
import subprocess

import numpy as np

DECK = "inputs/hydro_dshock_dual.athinput"
GAMMA = 5.0/3.0
RHO1, P1, VREL = 1.0, 0.6, 10.0
BULK, LENGTH, TIME = 300.0, 0.1, 0.002
TOLERANCE = 0.02


def _run(run_directory):
    command = ["./athena", "-i", DECK, "-d", str(run_directory)]
    environment = os.environ.copy()
    environment.setdefault("OMPI_MCA_coll_hcoll_enable", "0")
    result = subprocess.run(command, capture_output=True, check=False,
                            env=environment, text=True, timeout=600)
    assert result.returncode == 0, (
        f"colliding-stream run failed with return code {result.returncode}:\n"
        f"{(result.stdout + result.stderr)[-8000:]}"
    )
    dumps = sorted(Path(run_directory, "tab").glob("*.hydro_w.*.tab"))
    assert len(dumps) >= 2, "the run wrote no final dump"
    data = np.loadtxt(dumps[-1])
    return data[:, 2], data[:, 3], (GAMMA - 1.0)*data[:, 7]


def _rankine_hugoniot():
    """Post-shock pressure and density of a stream stopped by a wall at speed VREL."""

    def velocity_jump(p2):
        return (p2 - P1)*np.sqrt(2.0/RHO1/((GAMMA + 1.0)*p2 + (GAMMA - 1.0)*P1))

    low, high = P1, 1.0e6
    for _ in range(200):
        mid = 0.5*(low + high)
        if velocity_jump(mid) < VREL:
            low = mid
        else:
            high = mid
    p2 = 0.5*(low + high)
    rho2 = RHO1*((GAMMA + 1.0)*p2 + (GAMMA - 1.0)*P1)/((GAMMA - 1.0)*p2 +
                                                        (GAMMA + 1.0)*P1)
    return p2, rho2


def test_hydro_shock_at_bulk_mach_300_keeps_its_entropy(tmp_path):
    """Both shocked slabs carry the Rankine-Hugoniot entropy to 2 %."""

    x, rho, p = _run(tmp_path / "dshock")
    p2, rho2 = _rankine_hugoniot()
    k2 = p2/rho2**GAMMA
    shock_distance = RHO1*VREL/(rho2 - RHO1)*TIME
    centre = (BULK*TIME + LENGTH/2.0) % LENGTH - LENGTH/2.0
    xr = (x - centre + LENGTH/2.0) % LENGTH - LENGTH/2.0

    centre_p = p[np.abs(xr) < 0.4*shock_distance].mean()/p2
    assert abs(centre_p - 1.0) <= TOLERANCE, (
        f"pressure at the collision point {centre_p:.4f} of Rankine-Hugoniot")
    for side, name in ((-1.0, "left"), (1.0, "right")):
        slab = (side*xr > 0.3*shock_distance) & (side*xr < 0.8*shock_distance)
        entropy = np.mean(p[slab]/rho[slab]**GAMMA)/k2
        assert abs(entropy - 1.0) <= TOLERANCE, (
            f"{name} shocked slab: K/K_RH = {entropy:.4f}; the shock rule of the "
            f"dual-energy formalism missed part of the shock's heat")
