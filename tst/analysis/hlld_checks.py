#!/usr/bin/env python3
"""Special-relativistic HLLD checks: the Riemann problems and the magnetic loop of
Fields, Wong & Stone (2026, arXiv:2609.06150) Secs. III.A and III.B.

`riemann` runs the three shock tubes of the paper's Table I with both HLLE and HLLD at
several resolutions, and reports (a) the L1 errors of rho and B^y at the paper's 400
cells against a 6400-cell WENOZ HLLD reference, and (b) the self-convergence slope of the
entropy S = int s D dV (Eq. 15) over 100-1600 cells, which for a solution dominated by a
contact discontinuity should be about one half.

`loop` advects the magnetic loop of Eq. 16 twice across a periodic box and reports the
fraction of int b^2/2 that survives; the less diffusive solver keeps more.

See tst/analysis/HLLD_TESTS.md.
"""

import argparse
import os
import shutil
import subprocess
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

RIEMANN = {
    "contact": ("hlld_contact", 4.0 / 3.0),
    "alfven": ("hlld_alfven", 5.0 / 3.0),
    "balsara1": ("hlld_balsara1", 2.0),
    "balsara2": ("hlld_balsara2", 5.0 / 3.0),
}
RESOLUTIONS = (100, 200, 400, 800, 1600)
REFERENCE_CELLS = 6400
LOOP_DECK = "hlld_loop"


def run(binary, deck, workdir, args, mpirun, ranks):
    """One AthenaK run in its own directory; returns that directory."""
    if os.path.isdir(workdir):
        shutil.rmtree(workdir)
    os.makedirs(workdir)
    cmd = []
    if mpirun:
        cmd += [mpirun, "-n", str(ranks), "--bind-to", "none"]
    cmd += [binary, "-i", os.path.join(REPO, "inputs", "srmhd", deck + ".athinput")]
    cmd += args
    with open(os.path.join(workdir, "run.log"), "w") as log:
        rc = subprocess.call(cmd, cwd=workdir, stdout=log, stderr=subprocess.STDOUT)
    if rc != 0:
        sys.exit("run failed (%d): %s\n  see %s/run.log" % (rc, " ".join(cmd), workdir))
    return workdir


def read_tab(path):
    """Columns of an AthenaK .tab dump, keyed by the names on its second header line."""
    with open(path) as f:
        f.readline()
        names = f.readline().lstrip("#").split()
    data = np.loadtxt(path)
    # The first two columns are the meshblock id and the cell index.
    return {n: data[:, k] for k, n in enumerate(names)}


def shocktube(binary, deck, cells, solver, recon, workdir, mpirun):
    d = run(binary, deck, workdir,
            ["job/basename=st", "mesh/nx1=%d" % cells, "meshblock/nx1=%d" % cells,
             "mhd/rsolver=" + solver, "mhd/reconstruct=" + recon],
            mpirun, 1)
    tab = read_tab(os.path.join(d, "tab", "st.mhd_w_bcc.00001.tab"))
    order = np.argsort(tab["x1v"])
    return {k: v[order] for k, v in tab.items()}


def coarsen(values, cells):
    """Block-average a fine solution onto `cells` cells."""
    factor = len(values) // cells
    return values.reshape(cells, factor).mean(axis=1)


def entropy(tab, gamma):
    """S = int s D dV with s = ln(P/rho^gamma) and D = rho W (Eq. 15), on a uniform grid."""
    rho = tab["dens"]
    pgas = (gamma - 1.0) * tab["eint"]
    lorentz = np.sqrt(1.0 + tab["velx"]**2 + tab["vely"]**2 + tab["velz"]**2)
    dx = 1.0 / len(rho)
    return float(np.sum(np.log(pgas / rho**gamma) * rho * lorentz) * dx)


def check_riemann(args):
    print("%-10s %-6s %6s %12s %12s" % ("test", "solver", "cells", "L1(rho)", "L1(B^y)"))
    l1 = {}
    slopes = {}
    for test, (deck, gamma) in RIEMANN.items():
        work = os.path.join(args.workdir, test)
        ref = shocktube(args.binary, deck, REFERENCE_CELLS, "hlld", "wenoz",
                        os.path.join(work, "ref"), args.mpirun)
        s_ref = entropy(ref, gamma)
        for solver in ("hlle", "hlld"):
            errors = {}
            for cells in RESOLUTIONS:
                tab = shocktube(args.binary, deck, cells, solver, "dc",
                                os.path.join(work, "%s_%d" % (solver, cells)),
                                args.mpirun)
                errors[cells] = abs(entropy(tab, gamma) - s_ref)
                if cells == 400:
                    dx = 1.0 / cells
                    l1[(test, solver, "dens")] = float(
                        np.sum(np.abs(tab["dens"] - coarsen(ref["dens"], cells))) * dx)
                    l1[(test, solver, "bcc2")] = float(
                        np.sum(np.abs(tab["bcc2"] - coarsen(ref["bcc2"], cells))) * dx)
                    print("%-10s %-6s %6d %12.5e %12.5e" %
                          (test, solver, cells, l1[(test, solver, "dens")],
                           l1[(test, solver, "bcc2")]))
            n = np.array(RESOLUTIONS, dtype=float)
            e = np.array([errors[c] for c in RESOLUTIONS])
            slopes[(test, solver)] = float(
                np.polyfit(np.log(n), np.log(e), 1)[0])
    print()
    print("%-10s %-6s %12s" % ("test", "solver", "d log S/d log N"))
    for key in sorted(slopes):
        print("%-10s %-6s %12.3f" % (key[0], key[1], slopes[key]))
    print()
    for test in RIEMANN:
        for field in ("dens", "bcc2"):
            better = l1[(test, "hlld", field)] < l1[(test, "hlle", field)]
            print("%-10s %-5s HLLD below HLLE: %s" % (test, field, better))


_bin_convert = None


def bin_reader():
    """vis/python/bin_convert.py, loaded without its optional h5py dependency."""
    global _bin_convert
    if _bin_convert is None:
        path = os.path.join(REPO, "vis", "python", "bin_convert.py")
        src = open(path).read().replace("import h5py", "h5py = None")
        _bin_convert = type(sys)("bin_convert")
        exec(compile(src, path, "exec"), _bin_convert.__dict__)
    return _bin_convert


def read_bin(path):
    """Cell-centred fields of an AthenaK .bin dump, keyed by variable name."""
    dump = bin_reader().read_binary(path)
    return {n: np.asarray(dump["mb_data"][n]).ravel() for n in dump["var_names"]}


def magnetic_energy(path):
    """Sum of b^2/2 over cells, which on this uniform grid is the integral up to dV."""
    f = read_bin(path)
    return float(np.sum(f["bcc1"]**2 + f["bcc2"]**2 + f["bcc3"]**2) * 0.5)


def check_loop(args):
    print("%-6s %14s %14s %10s" % ("solver", "int b^2/2 t=0", "t=24", "retained"))
    for solver in ("hlle", "hlld"):
        d = run(args.binary, LOOP_DECK, os.path.join(args.workdir, "loop_" + solver),
                ["job/basename=loop", "mhd/rsolver=" + solver], args.mpirun, args.ranks)
        initial = magnetic_energy(os.path.join(d, "bin", "loop.mhd_bcc.00000.bin"))
        final = magnetic_energy(os.path.join(d, "bin", "loop.mhd_bcc.00001.bin"))
        print("%-6s %14.6e %14.6e %10.4f" % (solver, initial, final, final / initial))


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--binary", required=True, help="path to the athena executable")
    p.add_argument("--workdir", required=True, help="directory for the runs")
    p.add_argument("--check", default="riemann,loop",
                   help="comma-separated subset of riemann,loop")
    p.add_argument("--mpirun", default="", help="mpirun path ('' to launch directly)")
    p.add_argument("--ranks", type=int, default=1, help="ranks for the loop test")
    args = p.parse_args()
    args.binary = os.path.abspath(args.binary)
    args.workdir = os.path.abspath(args.workdir)
    checks = args.check.split(",")
    if "riemann" in checks:
        check_riemann(args)
    if "loop" in checks:
        check_loop(args)


if __name__ == "__main__":
    main()
