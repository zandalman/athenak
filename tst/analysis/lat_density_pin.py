#!/usr/bin/env python3
"""Exercise core LAT density pinning with the built-in shock_tube generator.

Run against a CPU MPI build:
  python3 tst/analysis/lat_density_pin.py --binary build/src/athena --outdir pin_out

The density jump lies INSIDE block 8, owned by rank 1 in a two-rank run. Its
factor-4 neighbour (block 7) belongs to rank 0, where no block meets the contrast
criterion. The pre-limiter and final bin counts therefore test the cross-rank
neighbour pin, as well as both density criteria and the hydro/MHD paths.
"""

import argparse
import json
import math
import os
from pathlib import Path
import re
import subprocess


DECK = """<job>
basename = density_pin
<mesh>
nghost = 2
nx1 = 128
x1min = 0.0
x1max = 1.0
ix1_bc = outflow
ox1_bc = outflow
nx2 = 1
x2min = 0.0
x2max = 1.0
ix2_bc = periodic
ox2_bc = periodic
nx3 = 1
x3min = 0.0
x3max = 1.0
ix3_bc = periodic
ox3_bc = periodic
<meshblock>
nx1 = 8
nx2 = 1
nx3 = 1
<time>
evolution = dynamic
integrator = rk2
cfl_number = 0.3
nlim = 4
tlim = 1.0
ndiag = 1
lat = true
lat_levels = 2
lat_diagnostics = true
lat_same_level = true
lat_same_level_max_ratio = 8
hydro_lat_min_bin_count = 0
hydro_lat_gid_reorder = false
hydro_lat_post_amr_rebalance = false
{pin_options}
<{fluid}>
eos = ideal
reconstruct = plm
rsolver = hlle
gamma = 1.4
<problem>
pgen_name = shock_tube
shock_dir = 1
xshock = 0.52
dl = 16.0
pl = 1.0
ul = 0.0
vl = 0.0
wl = 0.0
dr = 1.0
pr = 1.0
ur = 0.0
vr = 0.0
wr = 0.0
bxl = 0.0
byl = 0.0
bzl = 0.0
bxr = 0.0
byr = 0.0
bzr = 0.0
<output1>
file_type = hst
data_format = %.17e
dt = 0.001
"""


def first_bins(log, label):
    match = re.search(r"Mesh: HD LAT " + re.escape(label) + r" bins([^\n]*)", log)
    assert match, f"missing {label} diagnostics"
    return {int(f): int(n) for f, n in re.findall(r"f(\d+)=(\d+)", match[1])}


def check_conservation(directory):
    files = list(directory.rglob("*.hst"))
    assert len(files) == 1, f"expected one history file, found {files}"
    text = files[0].read_text()
    columns = {name: int(i) - 1 for i, name in re.findall(r"\[(\d+)\]=(\S+)", text)}
    rows = [[float(v) for v in line.split()] for line in text.splitlines()
            if line.strip() and not line.startswith("#")]
    assert len(rows) >= 2, "no evolved history samples"
    assert all(math.isfinite(v) for row in rows for v in row), "non-finite history"
    errors = {}
    for name in ("mass", "tot-E"):
        index = columns[name]
        initial = rows[0][index]
        errors[name] = max(abs(row[index] - initial) / abs(initial) for row in rows)
        assert errors[name] < 1.0e-12, f"{name} drift = {errors[name]}"
    return errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--mpirun", default="mpirun")
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    binary = args.binary.resolve()
    environment = os.environ.copy()
    environment.setdefault("OMPI_MCA_coll_hcoll_enable", "0")
    cases = (
        ("default", "", None, {1: 8, 2: 1, 4: 7}),
        ("disabled", "lat_pin_density_contrast = 1\nlat_pin_density = 0",
         None, {1: 8, 2: 1, 4: 7}),
        ("below_threshold", "lat_pin_density_contrast = 32\nlat_pin_density = 32",
         {1: 8, 4: 8}, {1: 8, 2: 1, 4: 7}),
        ("contrast", "lat_pin_density_contrast = 8",
         {1: 9, 4: 7}, {1: 9, 2: 1, 4: 6}),
        ("density", "lat_pin_density = 8", {1: 16}, {1: 16}),
        ("nonfinite", "lat_pin_density_contrast = nan", None, None),
    )
    results = {}
    for fluid in ("hydro", "mhd"):
        for ranks in (1, 2):
            for name, options, pinned, final in cases:
                label = f"{fluid}_{ranks}rank_{name}"
                directory = args.outdir / label
                directory.mkdir(exist_ok=False)
                deck = directory / "input.athinput"
                deck.write_text(DECK.format(fluid=fluid, pin_options=options))
                command = [str(binary), "-i", str(deck.resolve()),
                           "-d", str(directory.resolve())]
                command = [args.mpirun, "-n", str(ranks), "--bind-to", "none"] + command
                run = subprocess.run(command, capture_output=True, text=True,
                                     env=environment, timeout=60)
                log = run.stdout + run.stderr
                (directory / "run.log").write_text(log)
                if final is None:
                    assert run.returncode != 0 and ("must be finite" in log
                                                    or "Invalid real value" in log), log[-4000:]
                    results[label] = {"status": "PASS", "rejected_nonfinite": True}
                else:
                    assert run.returncode == 0, log[-4000:]
                    assert first_bins(log, "raw local-CFL") == {1: 8, 4: 8}, label
                    if pinned is not None:
                        assert first_bins(log, "density pinned") == pinned, label
                    else:
                        assert "density pinned bins" not in log, label
                    assert first_bins(log, "final") == final, label
                    results[label] = {"status": "PASS", "initial_bins": final,
                                      "relative_drift": check_conservation(directory)}
                print(f"PASS {label}", flush=True)
    (args.outdir / "results.json").write_text(json.dumps(results, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
