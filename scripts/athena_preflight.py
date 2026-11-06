#!/usr/bin/env python3
"""Job preflight checker for AthenaK (Python 3 standard library only).

A job that waited a day in a Slurm queue must not die within seconds because of a
missing or broken input.  Run this on the login node (or at the top of the sbatch,
before srun); it reads files only, uses no GPU and needs no build.

Usage
-----
    athena_preflight.py --deck DECK [--rst RST | --rst-dir DIR] [--exe ATHENA] [--ranks N]

    --deck     input deck (<block> lines, key = value, # comments)
    --rst      restart file to resume from
    --rst-dir  directory of *.rst; the file is chosen like the production sbatch does:
               newest by name among files whose size is at least 90 % of the largest
    --exe      athena executable; checked for existence, the x bit and (via ldd, when
               available) missing shared libraries
    --ranks    MPI ranks of the job; checks rst blocks <= ranks * max_nmb_per_rank

Exit code 0: all checks passed (warnings allowed).  Nonzero: at least one check failed;
each failure is printed as "[FAIL] what" followed by "fix: one-line hint".

Example sbatch lines
    python3 scripts/athena_preflight.py --deck run.athinput --rst-dir rst \\
        --exe ./athena --ranks $SLURM_NTASKS || exit 1
    srun ./athena -r "$RST" -i run.athinput

Checks
------
 1. executable exists / is executable / has no missing shared libraries (ldd)
 2. deck parses (rules of src/parameter_input.cpp LoadFromStream/ParseLine) and has no
    duplicate key in a block (the code silently keeps the last one)
 3. every absolute-path value (tables, remap source, ...) exists and is readable
 4. keys whose absence is FATAL (RULES below, with file:line of the source rule)
 5. restart file: header parses up to <par_end>, binary layout of
    src/outputs/restart.cpp WriteOutputFile, file size equals the size implied, the
    modules of the deck have their restart sections; time, cycle, block count printed
 6. block budget: rst blocks <= max_nmb_total and <= ranks * max_nmb_per_rank
 7. disk: free space (df) and quota (diskusage_report / quota, if present) against the
    rst size; a warning below 3 rst sizes free, a failure below 1

With a restart the effective parameters are the rst header overridden by the deck, as in
`athena -r RST -i DECK`.

Rst layout (restart.cpp WriteOutputFile, line numbers of the source this was written for)
    text   : ParameterDump ending "<par_end>\\n"           (714; parameter_input.cpp:1019)
    int32  : nmb_total, root_level                         (717-720)
    RegionSize(9 Real), RegionIndcs(19 int) x2 (mesh, mb) (721-726)
    Real time, Real dt, int32 ncycle                       (728-733)
    LogicalLocation[nmb] (4 int32), float cost[nmb]        (739-742)
    [z4c, tracker, turb RNG]  [int nsinks + long n, long created, n*SinkData]
    [int32 ncyc_since_ref[nmb] when restart_ncyc_since_ref_count > 0]   (771-782)
    size_t data_size (bytes per MeshBlock)                 (829)
    nmb * data_size payload
"""

import argparse
import os
import re
import shutil
import struct
import subprocess
import sys

FAILS = []
WARNS = []

# Largest header the code accepts: parameter_input.cpp:429 (<par_end> within 11 reads of 4096)
MAX_HEADER = 4096 * 11


def fail(msg, fix):
    FAILS.append((msg, fix))
    print("[FAIL] %s\n       fix: %s" % (msg, fix))


def warn(msg):
    WARNS.append(msg)
    print("[WARN] %s" % msg)


def ok(msg):
    print("[ OK ] %s" % msg)


# ----------------------------------------------------------------------------------------
# input parsing (parameter_input.cpp:311-388, ParseLine 477-507)

def parse_params(text, label, check_dups=True):
    """Return ({block: {key: value}}, [(error, fix)]).  Stops at <par_end>."""
    blocks = {}
    errs = []
    cur = None
    for lineno, raw in enumerate(text.split("\n"), 1):
        line = raw.replace("\t", "")          # tabs are erased (321-323)
        s = line.lstrip(" ")
        if not s.strip(" "):
            continue
        if s.startswith("#"):
            continue
        if s.startswith("<par_end>"):
            break
        if s.startswith("<"):
            end = s.find(">")
            if end < 0:
                errs.append(("%s:%d block name not properly ended: %r" % (label, lineno, s),
                             "close the block header with '>'"))
                continue
            cur = s[1:end]
            blocks.setdefault(cur, {})        # blocks that reappear are merged (FindOrAddBlock)
            continue
        if cur is None:
            errs.append(("%s:%d parameter before the first <block>" % (label, lineno),
                         "put a <block> line above the first key = value"))
            continue
        eq = s.find("=")
        if eq < 0:
            errs.append(("%s:%d line has no '=': %r" % (label, lineno, s.strip()),
                         "write it as key = value or comment it out with #"))
            continue
        key = s[:eq].rstrip(" ")
        hsh = s.find("#")
        val = s[eq + 1:(hsh if hsh >= 0 else len(s))].strip(" ")
        if hsh >= 0 and hsh < eq:
            errs.append(("%s:%d '#' before '=' in %r" % (label, lineno, s.strip()),
                         "remove the stray '#' (the code would parse garbage)"))
            continue
        if check_dups and key in blocks[cur]:
            errs.append(("%s:%d duplicate key <%s>/%s (the last value silently wins)"
                         % (label, lineno, cur, key), "delete the repeated line"))
        blocks[cur][key] = val
    return blocks, errs


def getb(params, block, key, default=False):
    v = params.get(block, {}).get(key)
    if v is None:
        return default
    return v.strip().lower() in ("true", "1", "yes", "t")


def has(params, block, key):
    return key in params.get(block, {})


# ----------------------------------------------------------------------------------------
# rule table for check 4: (applies(params) -> bool, [(block, key)], source, why)
# Keep small.  Each rule is a key whose absence aborts the run (GetXxx without default,
# or an explicit FATAL).  file:line refer to this source tree.

def _always(p):
    return True


def _nx(p, n):
    try:
        return int(p.get("mesh", {}).get("nx%d" % n, "1")) > 1
    except ValueError:
        return False


def _tde_amr(p):
    return "tde_amr" in p


def _expand(p):
    return getb(p, "remap", "auto_expand")


def _expand_adaptive(p):
    return _expand(p) or _tde_amr(p)


def _adaptive(p):
    return p.get("mesh_refinement", {}).get("refinement") == "adaptive"


def _dynamic(p):
    return p.get("time", {}).get("evolution") != "static"


RULES = [
    (_always, [("mesh", k) for k in ("x1min", "x1max", "x2min", "x2max", "x3min", "x3max",
                                      "nx1", "nx2", "nx3")],
     "src/mesh/mesh.cpp:110-120", "mesh extent and cell counts have no default"),
    (_always, [("mesh", "ix1_bc"), ("mesh", "ox1_bc")],
     "src/mesh/mesh.cpp:205-206", "x1 boundary conditions have no default"),
    (lambda p: _nx(p, 2), [("mesh", "ix2_bc"), ("mesh", "ox2_bc")],
     "src/mesh/mesh.cpp:244-245", "x2 boundary conditions (nx2 > 1)"),
    (lambda p: _nx(p, 3), [("mesh", "ix3_bc"), ("mesh", "ox3_bc")],
     "src/mesh/mesh.cpp:271-272", "x3 boundary conditions (nx3 > 1)"),
    (_always, [("time", "evolution")],
     "src/driver/driver.cpp:358", "time evolution has no default"),
    (_dynamic, [("time", "tlim")],
     "src/driver/driver.cpp:375", "end time of a non-static run"),
    (_always, [("job", "basename")],
     "src/outputs/outputs.cpp:145", "every output block names its files from it"),
    (_adaptive, [("mesh_refinement", "max_nmb_per_rank")],
     "src/mesh/build_tree.cpp:378-392,1056-1070", "adaptive runs need the per-rank block cap"),
    (_tde_amr, [("mesh_refinement", "max_nmb_per_rank"), ("mesh_refinement", "max_nmb_total")],
     "src/pgen/tde_amr.hpp:517-525 -> src/mesh/mesh_refinement.cpp:2373",
     "<tde_amr> reads the per-rank cap and the total block budget"),
    (_expand, [("mesh_refinement", "max_nmb_total")],
     "src/remap/remap_expand.cpp:262 and 525-530 -> src/mesh/mesh_refinement.cpp:2373",
     "remap auto_expand reads the total block budget"),
    (_expand, [("remap", "expand_max_" + f) for f in
               ("x1min", "x1max", "x2min", "x2max", "x3min", "x3max")],
     "src/remap/remap_expand.cpp:205-212", "auto_expand needs all six expansion limits"),
]


def check_rules(p, restart):
    n = 0
    for applies, keys, src, why in RULES:
        if not applies(p):
            continue
        for blk, key in keys:
            if key not in p.get(blk, {}):
                n += 1
                fail("missing <%s>/%s: %s (%s)" % (blk, key, why, src),
                     "add '%s = ...' to <%s> in the deck" % (key, blk))
    # explicit FATAL checks that are not a single missing key
    # tde_amr.hpp... tde_external.cpp:572-580: <tde_amr> needs adaptive refinement + a user criterion
    if _tde_amr(p):
        if not _adaptive(p):
            n += 1
            fail("<tde_amr> without <mesh_refinement>/refinement = adaptive "
                 "(src/pgen/tde_external.cpp:577-579)",
                 "set 'refinement = adaptive' in <mesh_refinement>")
        if not any(b.startswith("amr_criterion") and v.get("method") == "user"
                   for b, v in p.items()):
            n += 1
            fail("<tde_amr> needs an <amr_criterionN> block with method = user "
                 "(src/pgen/tde_external.cpp:580)",
                 "add <amr_criterion0> with 'method = user'")
    # remap_expand.cpp:221-248: auto_expand needs an rst output with a positive dt/dcycle
    # and single_file_per_rank = false
    if _expand(p):
        good = False
        for b, v in p.items():
            if not b.startswith("output") or v.get("file_type") != "rst":
                continue
            cadence = (v["dcycle"] not in ("0", "")) if "dcycle" in v else \
                (_pos(v.get("dt")))
            if cadence and not getb(p, b, "single_file_per_rank"):
                good = True
        if not good:
            n += 1
            fail("remap auto_expand = true needs an <outputN> with file_type = rst, a "
                 "positive dt or dcycle and single_file_per_rank = false "
                 "(src/remap/remap_expand.cpp:221-248)",
                 "add <output9> file_type = rst, dt = <interval>")
    # outputs.cpp:95-98: every <outputN> needs dt or dcycle; 146: file_type; 154: variable
    for b, v in sorted(p.items()):
        if not re.match(r"output\d+$", b):
            continue
        if "dt" not in v and "dcycle" not in v:
            n += 1
            fail("<%s> has neither dt nor dcycle (src/outputs/outputs.cpp:95-98)" % b,
                 "add 'dt = ...' or 'dcycle = ...' to <%s>" % b)
        if "file_type" not in v:
            n += 1
            fail("<%s> has no file_type (src/outputs/outputs.cpp:146)" % b,
                 "add 'file_type = bin|rst|hst|...' to <%s>" % b)
        elif v["file_type"] not in ("hst", "rst", "log", "trk") and "variable" not in v:
            n += 1
            fail("<%s> (file_type = %s) has no variable (src/outputs/outputs.cpp:154)"
                 % (b, v["file_type"]), "add 'variable = ...' to <%s>" % b)
    if n == 0:
        ok("required keys present (%d rules)" % len(RULES))


def _pos(s):
    try:
        return float(s) > 0.0
    except (TypeError, ValueError):
        return False


# ----------------------------------------------------------------------------------------
# check 3: absolute paths

PATH_RE = re.compile(r"^/[^\s,;=]+$")


def check_paths(deck, header, restart):
    seen = set()
    bad = 0
    n = 0
    for source, params in (("deck", deck), ("rst header", header)):
        for b, kv in params.items():
            for k, v in kv.items():
                if not PATH_RE.match(v) or (source, v) in seen:
                    continue
                if source == "rst header" and k in deck.get(b, {}):
                    continue                  # overridden by the deck
                seen.add((source, v))
                n += 1
                # a remap source is not read when resuming from a checkpoint
                soft = source == "rst header" or (restart and b == "remap" and k == "source")
                if os.path.exists(v) and os.access(v, os.R_OK):
                    continue
                bad += 1
                what = "missing" if not os.path.exists(v) else "not readable"
                if soft:
                    warn("<%s>/%s = %s is %s (%s; not needed unless the code re-reads it)"
                         % (b, k, v, what, source))
                else:
                    fail("<%s>/%s = %s is %s" % (b, k, v, what),
                         "restore/copy the file or fix the path in the deck")
    if bad == 0:
        ok("%d absolute path value(s) exist and are readable" % n)


# ----------------------------------------------------------------------------------------
# check 1: executable

def check_exe(exe):
    if not exe:
        print("[SKIP] no --exe given")
        return
    if not os.path.isfile(exe):
        fail("executable %s does not exist" % exe, "build or point --exe at the right binary")
        return
    if not os.access(exe, os.X_OK):
        fail("executable %s is not executable" % exe, "chmod +x %s" % exe)
        return
    ldd = shutil.which("ldd")
    if not ldd:
        ok("executable %s (ldd not available, libraries not checked)" % exe)
        return
    try:
        out = subprocess.run([ldd, exe], capture_output=True, text=True, timeout=60).stdout
    except (subprocess.TimeoutExpired, OSError) as e:
        warn("ldd %s failed: %s" % (exe, e))
        return
    miss = [l.split("=>")[0].strip() for l in out.splitlines() if "not found" in l]
    if miss:
        fail("executable %s misses shared libraries: %s" % (exe, ", ".join(miss)),
             "load the build-time modules (module load ...) before running the check")
    else:
        ok("executable %s, no missing shared libraries" % exe)


# ----------------------------------------------------------------------------------------
# check 5: restart file

def pick_rst(rst_dir):
    try:
        names = sorted(f for f in os.listdir(rst_dir) if f.endswith(".rst"))
    except OSError as e:
        fail("cannot list %s: %s" % (rst_dir, e), "check --rst-dir")
        return None
    files = [(os.path.join(rst_dir, f), os.path.getsize(os.path.join(rst_dir, f)))
             for f in names if os.path.isfile(os.path.join(rst_dir, f))]
    if not files:
        fail("no *.rst in %s" % rst_dir, "point --rst-dir at the checkpoint directory")
        return None
    big = max(s for _, s in files)
    good = [(f, s) for f, s in files if s >= 0.9 * big]
    for f, s in files:
        if f > good[-1][0]:
            warn("%s (%d B) is newer than the chosen file but under 90%% of the largest: "
                 "skipped as incomplete" % (f, s))
    return good[-1][0]


class RstInfo:
    pass


def read_rst(path, deck):
    """Parse header and layout.  Returns RstInfo or None (failures already reported)."""
    size = os.path.getsize(path)
    try:
        with open(path, "rb") as fh:
            head = fh.read(MAX_HEADER)
            loc = head.find(b"<par_end>")
            if loc < 0:
                fail("%s: <par_end> not found in the first %d bytes (code limit, "
                     "parameter_input.cpp:429)" % (path, MAX_HEADER),
                     "the file is not a restart file or is damaged: use an earlier rst")
                return None
            text = head[:loc + 10].decode("ascii", "replace")
            params, _ = parse_params(text, os.path.basename(path), check_dups=False)
            r = RstInfo()
            r.params, r.size, r.path = params, size, path
            pos = loc + 10                    # header = loc + 10 (parameter_input.cpp:425)
            best = None
            for rs in (8, 4):                 # Real = double or float
                res = _layout(fh, pos, rs, params, size)
                if res is not None and (best is None or res["ok"]):
                    best = res
                    if res["ok"]:
                        break
    except OSError as e:
        fail("cannot read %s: %s" % (path, e), "check the file permissions / filesystem")
        return None
    if best is None:
        fail("%s: file ends inside the header (size %d B)" % (path, size),
             "the checkpoint is truncated: use the previous rst")
        return None
    r.__dict__.update(best)
    return r


def _layout(fh, pos, rs, params, size):
    """Walk the binary header for Real size rs.  Returns dict or None."""
    sz_region = 9 * rs
    sz_indcs = 19 * 4
    fixed = 8 + sz_region + 2 * sz_indcs + 2 * rs + 4
    fh.seek(pos)
    buf = fh.read(fixed)
    if len(buf) < fixed:
        return None
    nmb, root_level = struct.unpack_from("<ii", buf, 0)
    o = 8 + sz_region
    ng, nx1, nx2, nx3 = struct.unpack_from("<4i", buf, o + sz_indcs)   # mb_indcs
    fmt = "<d" if rs == 8 else "<f"
    o2 = 8 + sz_region + 2 * sz_indcs
    t = struct.unpack_from(fmt, buf, o2)[0]
    dt = struct.unpack_from(fmt, buf, o2 + rs)[0]
    ncyc = struct.unpack_from("<i", buf, o2 + 2 * rs)[0]
    res = dict(nmb=nmb, root_level=root_level, ng=ng, nx=(nx1, nx2, nx3), time=t, dt=dt,
               cycle=ncyc, rs=rs, ok=False, notes=[], pos_hdr=pos + fixed)
    if not (0 < nmb < 10 ** 8 and 0 <= ng < 16 and nx1 > 0 and nx2 > 0 and nx3 > 0):
        return res
    p = pos + fixed + nmb * (16 + 4)          # lloc + cost
    nsinks_bytes = 0
    if "z4c" in params or "turb_driving" in params:
        res["notes"].append("z4c/turbulence state precedes the payload: size is a lower bound")
        res["lower_bound_only"] = True
        return _finish(res, fh, p, size, params, rs, locate=False)
    if "sink_particles" in params:
        fh.seek(p)
        b = fh.read(4 + 16)
        if len(b) < 20:
            return res
        nsinks = struct.unpack_from("<i", b, 0)[0]
        n = struct.unpack_from("<q", b, 4)[0]
        if nsinks < 0 or n != nsinks:
            return res
        res["nsinks"] = nsinks
        nsinks_bytes = 4 + 16 + nsinks * (8 + 10 * rs)   # int, 2 long, SinkData: long id, Real m, 3 x Real[3]
        p += nsinks_bytes
    ncyc_since = int(params.get("mesh_refinement", {}).get("restart_ncyc_since_ref_count", "0") or 0)
    p += 4 * ncyc_since
    return _finish(res, fh, p, size, params, rs, locate=True)


def _finish(res, fh, p, size, params, rs, locate):
    nout = [res["nx"][0] + 2 * res["ng"],
            res["nx"][1] + 2 * res["ng"] if res["nx"][1] > 1 else 1,
            res["nx"][2] + 2 * res["ng"] if res["nx"][2] > 1 else 1]
    res["ncell"] = nout[0] * nout[1] * nout[2]
    if not locate:
        res["expected"] = None
        return res
    fh.seek(p)
    b = fh.read(8)
    if len(b) < 8:
        res["expected"] = None
        res["ok"] = False
        res["trunc_in_header"] = True
        return res
    ds = struct.unpack("<Q", b)[0]
    res["data_size"] = ds
    if ds == 0 or ds % (res["ncell"] * rs) != 0 or ds > 10 ** 12:
        return res
    res["ok"] = True
    res["nvar_total"] = ds // (res["ncell"] * rs)
    res["expected"] = p + 8 + ds * res["nmb"]
    return res


def check_rst(r, deck, merged):
    if r is None:
        return
    print("       %s" % r.path)
    print("       time=%.17g dt=%.6g cycle=%d blocks=%d size=%d B (Real=%d B)"
          % (r.time, r.dt, r.cycle, r.nmb, r.size, r.rs))
    for n in getattr(r, "notes", []):
        warn(n)
    if getattr(r, "trunc_in_header", False):
        fail("%s ends inside the binary header (size %d B)" % (r.path, r.size),
             "the checkpoint is incomplete (job killed while writing?): use the previous rst")
        return
    exp = getattr(r, "expected", None)
    if exp is None and not getattr(r, "lower_bound_only", False):
        fail("%s: binary layout does not parse (blocks=%d, data_size=%s)"
             % (r.path, r.nmb, getattr(r, "data_size", "?")),
             "the file is damaged or not an AthenaK restart: use the previous rst")
    elif exp is not None:
        if r.size < exp:
            fail("%s is truncated: size %d B, header implies %d B (%.1f%%)"
                 % (r.path, r.size, exp, 100.0 * r.size / exp),
                 "the write was interrupted: use the previous complete rst")
        elif r.size > exp:
            fail("%s is %d B longer than the header implies (%d B)"
                 % (r.path, r.size - exp, exp), "file is damaged: use the previous rst")
        else:
            ok("rst size matches the header (%d B payload)" % exp)
    # restart sections the deck's modules need: the module blocks must be in the header,
    # else the payload layout the new run expects (sum over modules) differs
    for blk, what in (("hydro", "hydro u0"), ("mhd", "mhd u0 + B"),
                      ("radiation", "radiation i0"), ("z4c", "z4c u0"),
                      ("turb_driving", "turbulence forcing + RNG"),
                      ("sink_particles", "sink list")):
        if blk in deck and blk not in r.params:
            fail("deck has <%s> but the rst has no <%s> section (%s)" % (blk, blk, what),
                 "use an rst written by a run with this module, or drop the module")
    # tde_amr state is optional (tde_amr.hpp:471); nothing to check


# ----------------------------------------------------------------------------------------
# check 6, 7

def check_budget(r, merged, ranks):
    if r is None:
        return
    nmb = r.nmb
    tot = merged.get("mesh_refinement", {}).get("max_nmb_total")
    per = merged.get("mesh_refinement", {}).get("max_nmb_per_rank")
    bad = 0
    try:
        if tot is not None and nmb > int(float(tot)):
            bad += 1
            fail("rst has %d blocks, above <mesh_refinement>/max_nmb_total = %s" % (nmb, tot),
                 "raise max_nmb_total (and max_nmb_per_rank or the rank count)")
        elif tot is not None and getb(merged, "remap", "auto_expand") and nmb > 0.9 * float(tot):
            warn("rst has %d blocks, above 0.9 x max_nmb_total = %s (remap_expand.cpp:520 "
                 "merges blocks to fit)" % (nmb, tot))
        if ranks and per is not None and nmb > ranks * int(float(per)):
            bad += 1
            fail("rst has %d blocks, above ranks x max_nmb_per_rank = %d x %s = %d"
                 % (nmb, ranks, per, ranks * int(float(per))),
                 "use at least %d ranks or raise max_nmb_per_rank"
                 % -(-nmb // int(float(per))))
    except ValueError:
        fail("max_nmb_total / max_nmb_per_rank is not a number", "fix the deck value")
        return
    if not bad:
        ok("block budget: %d blocks, max_nmb_total=%s, max_nmb_per_rank=%s, ranks=%s"
           % (nmb, tot, per, ranks))


def run_text(cmd):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=30).stdout
    except (subprocess.TimeoutExpired, OSError):
        return ""


def check_disk(outdir, rst_size):
    d = outdir if os.path.isdir(outdir) else "."
    out = run_text(["df", "-Pk", d]).strip().splitlines()
    free = None
    if len(out) >= 2:
        try:
            free = int(out[-1].split()[3]) * 1024
        except (ValueError, IndexError):
            pass
    if free is None:
        free = shutil.disk_usage(d).free
    msg = "free space in %s: %.1f GB" % (os.path.abspath(d), free / 1e9)
    if rst_size:
        msg += " = %.1f rst sizes" % (free / rst_size)
    if rst_size and free < rst_size:
        fail(msg, "free disk space or write to another filesystem")
    elif rst_size and free < 3 * rst_size:
        warn(msg + " (less than 3)")
    else:
        ok(msg)
    for tool, args in (("diskusage_report", []), ("quota", ["-s"])):
        if shutil.which(tool):
            txt = run_text([tool] + args).strip().splitlines()
            if txt:
                print("       %s:" % tool)
                for l in txt[:6]:
                    print("         " + l)
            break


# ----------------------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="AthenaK job preflight checker")
    ap.add_argument("--deck", required=True)
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--rst")
    g.add_argument("--rst-dir")
    ap.add_argument("--exe")
    ap.add_argument("--ranks", type=int)
    a = ap.parse_args()

    check_exe(a.exe)

    try:
        with open(a.deck, "r", errors="replace") as fh:
            deck, errs = parse_params(fh.read(), os.path.basename(a.deck))
    except OSError as e:
        fail("cannot read deck %s: %s" % (a.deck, e), "check --deck")
        return finish()
    for e, fx in errs:
        fail(e, fx)
    if not errs:
        ok("deck parses (%d blocks, %d keys)" % (len(deck), sum(len(v) for v in deck.values())))

    rst = a.rst
    if a.rst_dir:
        rst = pick_rst(a.rst_dir)
    elif rst and not os.path.isfile(rst):
        fail("restart file %s does not exist" % rst, "check --rst")
        rst = None
    info = read_rst(rst, deck) if rst else None

    header = info.params if info else {}
    merged = {b: dict(v) for b, v in header.items()}
    for b, v in deck.items():
        merged.setdefault(b, {}).update(v)

    check_paths(deck, header, bool(rst))
    check_rules(merged, bool(rst))
    if rst:
        check_rst(info, deck, merged)
        check_budget(info, merged, a.ranks)
        outdir = a.rst_dir or os.path.dirname(os.path.abspath(rst))
        check_disk(outdir, os.path.getsize(rst))
    else:
        check_disk(".", None)
    return finish()


def finish():
    print("---- preflight: %d failure(s), %d warning(s)" % (len(FAILS), len(WARNS)))
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
