"""Shocks carried at a large bulk Mach number keep their heating under MHD dual energy.

An upwind scheme spreads a shock over about |v_grid|/dv cells, because its numerical
viscosity grows with the speed of the gas across the grid.  At a bulk Mach number of a few
hundred no stencil pair of a cell then converges faster than sound, and the dual-energy
shock rule sees the shock only through the steady compression the cell sits in: the
steady test per MHD family with the jump on the gas sound speed (eos/general_c2p_mhd.hpp).
Without it the auxiliary, which only compresses adiabatically, kept 0.63-0.72 of the
post-shock entropy of the streams below at beta = 1 and 0.19-0.59 at beta = 0.1.

Both gates are Galilean.  The conservative scheme is no reference at Mach 300: it carries
a frame-dependent heat, a quarter more internal energy in the Brio-Wu structure below than
the same tube at rest (1D model of the update), which falls only slowly with resolution.
A gate on the peak specific energy against it would reward spurious heat.

- Perpendicular colliding +-10 c streams (rho = 1, p = 0.6, gamma = 5/3, beta = 1 and
  0.1) carried at 300 c: the entropy K = p/rho^gamma of both shocked slabs within 2 % of
  the Rankine-Hugoniot value.  nghost = 4, so the test reaches two cells to both sides of
  a converging cell (DualEnergyShockReach): the post-sonic tail of each front is counted.
- The low-beta Brio-Wu tube (inputs/mhd_bw_lowbeta_dual.athinput states) on a doubled
  periodic domain, at rest and carried at 300 sound speeds of its left state: the internal
  energy of the structure (within 0.6 of the contact) at Mach 300 at least 0.84 of the
  same run at rest.  The slow shock, the contact and the compound wave overlap there in
  one structure of ~85 cells that forms at t = 0 and is never steady; the stencil test
  alone kept 0.78 of the heat, and about a tenth remains lost (an open limitation,
  general_c2p_mhd.hpp).
"""

from pathlib import Path
import os
import subprocess

import numpy as np

STREAMS_DECK = "inputs/mhd_boost_dual_streams.athinput"
BW_DECK = "inputs/mhd_boost_dual_bw.athinput"
GAMMA = 5.0/3.0
STREAM_SPEED = 10.0
STREAM_PRESSURE = 0.6
STREAMS_LENGTH = 0.1
STREAMS_TLIM = 0.002
BOOST = 300.0
# 300 sound speeds of the Brio-Wu left state (rho = 1, p = 0.1, gamma = 2)
BW_BOOST = 134.0
BW_TLIM = 0.1
BW_LENGTH = 2.0
BW_MIN_HEAT_RATIO = 0.84
ENTROPY_TOLERANCE = 0.02


def _run(run_directory, deck, overrides):
    command = ["./athena", "-i", deck, *overrides, "-d", str(run_directory)]
    environment = os.environ.copy()
    environment.setdefault("OMPI_MCA_coll_hcoll_enable", "0")
    result = subprocess.run(command, capture_output=True, check=False,
                            env=environment, text=True, timeout=600)
    assert result.returncode == 0, (
        f"run of {deck} failed with return code {result.returncode}:\n"
        f"{(result.stdout + result.stderr)[-8000:]}"
    )
    dumps = sorted(Path(run_directory, "tab").glob("*.mhd_w.*.tab"))
    assert len(dumps) >= 2, f"the run of {deck} wrote no final dump"
    with open(dumps[-1]) as table:
        table.readline()
        names = table.readline().lstrip("#").split()
    data = np.loadtxt(dumps[-1])
    # the header names the j/x2v/k/x3v columns, which a 1D table does not hold
    names = [n for n in names if n not in ("j", "x2v", "k", "x3v")]
    assert len(names) == data.shape[1], f"unexpected table layout {names}"
    return {name: data[:, column] for column, name in enumerate(names)}


def _relative_position(x, boost, tlim, length):
    """Position relative to the point x = 0 carried by the boost (periodic domain)."""
    centre = ((boost*tlim + 0.5*length) % length) - 0.5*length
    return ((x - centre + 0.5*length) % length) - 0.5*length


def _perpendicular_shock(field):
    """Shock speed (downstream frame) and post-shock density and pressure of a stream of
    the given perpendicular field running into its mirror image at STREAM_SPEED."""
    rho1, p1, dv = 1.0, STREAM_PRESSURE, STREAM_SPEED

    def states(speed):
        w1 = dv + speed
        rho2 = rho1*w1/speed
        b2 = field*w1/speed
        p2 = p1 + 0.5*(field**2 - b2**2) + rho1*w1*dv
        return w1, rho2, b2, p2

    def energy_flux_jump(speed):
        w1, rho2, b2, p2 = states(speed)
        upstream = 0.5*w1**2 + GAMMA/(GAMMA - 1.0)*p1/rho1 + field**2/rho1
        downstream = 0.5*speed**2 + GAMMA/(GAMMA - 1.0)*p2/rho2 + b2**2/rho2
        return upstream - downstream

    low, high = 1.0e-3*dv, 10.0*dv + 100.0
    low_sign = energy_flux_jump(low) > 0.0
    assert low_sign != (energy_flux_jump(high) > 0.0)
    for _ in range(200):
        middle = 0.5*(low + high)
        if (energy_flux_jump(middle) > 0.0) == low_sign:
            low = middle
        else:
            high = middle
    speed = 0.5*(low + high)
    _, rho2, _, p2 = states(speed)
    return speed, rho2, p2


def test_boosted_perpendicular_shocks_keep_their_entropy(tmp_path):
    """K/K_RH of both shocked slabs of colliding streams carried at 300 c."""

    for beta in (1.0, 0.1):
        field = float(np.sqrt(2.0*STREAM_PRESSURE/beta))
        state = _run(tmp_path / f"streams_beta{beta}", STREAMS_DECK,
                     [f"problem/ul={BOOST + STREAM_SPEED!r}",
                      f"problem/ur={BOOST - STREAM_SPEED!r}",
                      f"problem/byl={field!r}", f"problem/byr={field!r}"])
        speed, rho2, p2 = _perpendicular_shock(field)
        entropy_rh = p2/rho2**GAMMA
        position = _relative_position(state["x1v"], BOOST, STREAMS_TLIM,
                                      STREAMS_LENGTH)
        entropy = (GAMMA - 1.0)*state["eint"]/state["dens"]**GAMMA
        width = speed*STREAMS_TLIM
        for side, slab in (("left", (position > -0.75*width) & (position < -0.25*width)),
                           ("right", (position > 0.25*width) & (position < 0.75*width))):
            assert slab.any(), f"no cell inside the {side} shocked slab"
            ratio = float(entropy[slab].mean())/entropy_rh
            assert abs(ratio - 1.0) <= ENTROPY_TOLERANCE, (
                f"colliding streams at beta = {beta} carried at {BOOST:g} c: the {side} "
                f"shocked slab has K/K_RH = {ratio:.4f}, more than "
                f"{ENTROPY_TOLERANCE:.0%} from the Rankine-Hugoniot entropy: the shock "
                f"smeared by the bulk flow is not recognised (or over-heated) by the "
                f"dual-energy shock rule"
            )


def test_boosted_brio_wu_keeps_its_heat(tmp_path):
    """Internal energy of the boosted low-beta Brio-Wu structure against the tube
    at rest."""

    heat = {}
    for boost in (0.0, BW_BOOST):
        state = _run(tmp_path / f"bw_{boost:g}", BW_DECK,
                     [f"problem/ul={boost!r}", f"problem/ur={boost!r}"])
        position = _relative_position(state["x1v"], boost, BW_TLIM, BW_LENGTH)
        cell = BW_LENGTH/state["x1v"].size
        heat[boost] = float((state["eint"]*cell)[np.abs(position) < 0.6].sum())
    ratio = heat[BW_BOOST]/heat[0.0]
    assert ratio >= BW_MIN_HEAT_RATIO, (
        f"low-beta Brio-Wu tube carried at Mach 300: the structure holds an internal "
        f"energy of {heat[BW_BOOST]:.5f}, {ratio:.3f} of the {heat[0.0]:.5f} of the tube "
        f"at rest (gate {BW_MIN_HEAT_RATIO}): the heat of the smeared slow shock is lost "
        f"to the adiabatic auxiliary"
    )
