"""HLLD uses its star states only where the fan they form is admissible.

The adiabatic HLLD solver (mhd/rsolvers/hlld_mhd.hpp) builds the Miyoshi & Kusano star
states from the jump conditions across the outer waves, with den = rho*(S - S_M)^2 - Bx^2
in the denominator of the transverse ones.  (1) Where den <= 0 a side's fast and Alfven
waves have merged, and its star state is M&K's degenerate one (B_t* = B_t, v_t* = v_t).
(2) Where a star state has a negative density or internal energy, the face takes the
HLL flux of the same outer speeds.  The faces below were met in the low-beta Brio-Wu
tube (inputs/mhd_bw_lowbeta_dual.athinput, recorded with a 1D mirror of the update):

- closing: den = 1.4e-4 p_T*.  The left star state has B_y* = 359 and an internal energy
  of -15 p_T*; its energy flux is -1.86 against +0.20 from HLL.
- reversed: den = -2.7e-3 p_T*.  The jump conditions would reverse B_y (B_y*/B_y = -97).
- ordinary: den = 0.11 p_T* and both star states have a positive internal energy, but the
  star gas pressure p_T* - B*^2/2 is negative (-0.67 p_T*), since p_T* is only an HLL
  estimate.  This fan is admissible and keeps the Miyoshi & Kusano flux.

Each face is the initial discontinuity of a 16-cell tube advanced one forward-Euler step
with donor-cell reconstruction.  The changes of the two cells beside the face give the
face flux, which is compared with the HLL and Miyoshi & Kusano fluxes computed here.
"""

from pathlib import Path
import os
import subprocess

import numpy as np

DECK = "inputs/mhd_bw_lowbeta_dual.athinput"
GAMMA = 2.0
BX = 0.75
NX = 16

# primitive face states (rho, vx, vy, vz, eint, by, bz), left and right
FACES = {
    "closing": (
        (0.7386649233626502, 0.4464566366919992, -0.7503082529990804, 0.0,
         0.09136630990083751, 0.18846533450830844, 0.0),
        (0.9640584771943665, 0.24493888343469283, -0.9278517800586137, 0.0,
         0.23007644945210665, 0.044512839218149705, 0.0)),
    "reversed": (
        (0.8064907270696342, 0.3623405088218731, -0.8465089238207023, 0.0,
         0.132813079525128, 0.11625355943729396, 0.0),
        (1.0969931942473778, 0.17509518063295748, -1.0056454127470187, 0.0,
         0.2968650651728536, -0.0466522962815992, 0.0)),
    "ordinary": (
        (0.8749543324102219, 0.2987661226417741, -0.5747065697381902, 0.0,
         0.04506150719447672, 0.3723022075023131, 0.0),
        (1.1262535831803573, 0.16619181581463915, -0.7626678558677484, 0.0,
         0.1879498641475956, 0.217802826418828, 0.0)),
}


def _conserved_and_flux(w):
    d, vx, vy, vz, e, by, bz = w
    pt = (GAMMA - 1.0)*e + 0.5*(BX*BX + by*by + bz*bz)
    etot = e + 0.5*d*(vx*vx + vy*vy + vz*vz) + 0.5*(BX*BX + by*by + bz*bz)
    u = np.array([d, d*vx, d*vy, d*vz, etot, by, bz])
    f = np.array([d*vx, d*vx*vx + pt - BX*BX, d*vx*vy - BX*by, d*vx*vz - BX*bz,
                  vx*(etot + pt) - BX*(vx*BX + vy*by + vz*bz),
                  by*vx - BX*vy, bz*vx - BX*vz])
    return u, f, pt


def _fast_speed(w):
    d, _, _, _, e, by, bz = w
    asq = GAMMA*(GAMMA - 1.0)*e
    ct2 = by*by + bz*bz
    tmp = BX*BX + ct2 - asq
    return np.sqrt(0.5*(BX*BX + ct2 + asq + np.sqrt(tmp*tmp + 4.0*asq*ct2))/d)


def _fan(wl, wr):
    """Outer speeds, HLL flux and the Miyoshi & Kusano star states of one face."""
    ul, fl, ptl = _conserved_and_flux(wl)
    ur, fr, ptr = _conserved_and_flux(wr)
    sl = min(wl[1] - _fast_speed(wl), wr[1] - _fast_speed(wr))
    sr = max(wl[1] + _fast_speed(wl), wr[1] + _fast_speed(wr))
    hll = (sr*fl - sl*fr + sr*sl*(ur - ul))/(sr - sl)
    sm = ((sr - wr[1])*ur[1] - (sl - wl[1])*ul[1] + ptl - ptr)/(
        (sr - wr[1])*ur[0] - (sl - wl[1])*ul[0])
    ptst = 0.5*(ptl + ul[0]*(sl - wl[1])*(sm - wl[1]) +
                ptr + ur[0]*(sr - wr[1])*(sm - wr[1]))
    stars = []
    for w, u, pt, s in ((wl, ul, ptl, sl), (wr, ur, ptr, sr)):
        d, vx, vy, vz, _, by, bz = w
        ds = d*(s - vx)/(s - sm)
        den = d*(s - vx)*(s - sm) - BX*BX
        star = {}
        for kind in ("jump", "solver"):
            if kind == "solver" and (abs(den) < 1.0e-4*ptst or not den > 0.0):
                vys, vzs, bys, bzs = vy, vz, by, bz      # degenerate: merged waves
            else:
                rat = (d*(s - vx)**2 - BX*BX)/den
                tmp = BX*(sm - vx)/den
                vys, vzs, bys, bzs = vy - by*tmp, vz - bz*tmp, by*rat, bz*rat
            vbs = sm*BX + vys*bys + vzs*bzs
            es = ((s - vx)*u[4] - pt*vx + ptst*sm +
                  BX*(vx*BX + vy*by + vz*bz - vbs))/(s - sm)
            eint = (es - 0.5*ds*(sm*sm + vys*vys + vzs*vzs) -
                    0.5*(BX*BX + bys*bys + bzs*bzs))
            star[kind] = dict(u=np.array([ds, ds*sm, ds*vys, ds*vzs, es, bys, bzs]),
                              eint=eint/ptst, by_ratio=bys/by, vb=vbs,
                              pgas=(ptst - 0.5*(BX*BX + bys*bys + bzs*bzs))/ptst)
        star["den"] = den/ptst
        stars.append(star)
    return dict(sl=sl, sr=sr, sm=sm, ul=ul, ur=ur, fl=fl, fr=fr, hll=hll, stars=stars)


def _mk_flux(fan):
    """Miyoshi & Kusano flux with the double-star states, eqns (59)-(66)."""
    stl, str_ = (star["solver"] for star in fan["stars"])
    usl, usr = stl["u"], str_["u"]
    sl, sr, sm = fan["sl"], fan["sr"], fan["sm"]
    sqdl, sqdr = np.sqrt(usl[0]), np.sqrt(usr[0])
    sgn = 1.0 if BX > 0.0 else -1.0
    vy2 = (sqdl*usl[2]/usl[0] + sqdr*usr[2]/usr[0] + sgn*(usr[5] - usl[5]))/(sqdl + sqdr)
    vz2 = (sqdl*usl[3]/usl[0] + sqdr*usr[3]/usr[0] + sgn*(usr[6] - usl[6]))/(sqdl + sqdr)
    by2 = (sqdl*usr[5] + sqdr*usl[5] +
           sgn*sqdl*sqdr*(usr[2]/usr[0] - usl[2]/usl[0]))/(sqdl + sqdr)
    bz2 = (sqdl*usr[6] + sqdr*usl[6] +
           sgn*sqdl*sqdr*(usr[3]/usr[0] - usl[3]/usl[0]))/(sqdl + sqdr)
    vb2 = sm*BX + vy2*by2 + vz2*bz2
    udl = np.array([usl[0], usl[1], usl[0]*vy2, usl[0]*vz2,
                    usl[4] - sqdl*sgn*(stl["vb"] - vb2), by2, bz2])
    udr = np.array([usr[0], usr[1], usr[0]*vy2, usr[0]*vz2,
                    usr[4] + sqdr*sgn*(str_["vb"] - vb2), by2, bz2])
    sls, srs = sm - abs(BX)/sqdl, sm + abs(BX)/sqdr
    fls = fan["fl"] + sl*(usl - fan["ul"])
    frs = fan["fr"] + sr*(usr - fan["ur"])
    if sl >= 0.0:
        return fan["fl"]
    if sr <= 0.0:
        return fan["fr"]
    if sls >= 0.0:
        return fls
    if sm >= 0.0:
        return fls + sls*(udl - usl)
    if srs > 0.0:
        return frs + srs*(udr - usr)
    return frs


def _face_flux(run_directory, wl, wr):
    """The face flux AthenaK applies, from one forward-Euler step of the tube."""
    keys = ("d", "u", "v", "w", "p", "by", "bz")
    overrides = [f"mesh/nx1={NX}", f"meshblock/nx1={NX}", "time/integrator=rk1",
                 "time/nlim=1", "time/cfl_number=0.4", "mhd/reconstruct=dc",
                 "mhd/dual_energy=false", "output1/variable=mhd_u", "output1/dt=1.0e-12",
                 f"problem/bxl={BX!r}", f"problem/bxr={BX!r}"]
    for side, w in (("l", wl), ("r", wr)):
        values = list(w)
        values[4] = (GAMMA - 1.0)*w[4]
        overrides += [f"problem/{k}{side}={float(v)!r}" for k, v in zip(keys, values)]
    command = ["./athena", "-i", DECK, *overrides, "-d", str(run_directory)]
    environment = os.environ.copy()
    environment.setdefault("OMPI_MCA_coll_hcoll_enable", "0")
    result = subprocess.run(command, capture_output=True, check=False, env=environment,
                            text=True, timeout=120)
    assert result.returncode == 0, (result.stdout + result.stderr)[-4000:]
    dumps = sorted(Path(run_directory, "tab").glob("*.mhd_u.*.tab"))
    assert len(dumps) >= 2, "the run wrote no dump after its step"
    before, after = np.loadtxt(dumps[0])[:, 3:8], np.loadtxt(dumps[-1])[:, 3:8]
    change_l = after[NX//2 - 1] - before[NX//2 - 1]
    change_r = after[NX//2] - before[NX//2]
    # The cells either side of the face change by -dt/dx (F - F_L) and -dt/dx (F_R - F);
    # their sum gives dt/dx without the dump's rounded time.
    _, flux_l, _ = _conserved_and_flux(wl)
    _, flux_r, _ = _conserved_and_flux(wr)
    jump = flux_r[:5] - flux_l[:5]
    dt_dx = -np.dot(change_l + change_r, jump)/np.dot(jump, jump)
    return flux_l[:5] - change_l/dt_dx


def _check(tmp_path, name, expected, what):
    wl, wr = (np.array(w) for w in FACES[name])
    fan = _fan(wl, wr)
    flux = _face_flux(tmp_path / name, wl, wr)
    scale = np.max(np.abs(fan["fl"][:5]) + np.abs(fan["fr"][:5]))
    error = np.max(np.abs(flux - expected(fan)[:5]))/scale
    assert np.all(np.isfinite(flux)) and error < 1.0e-9, (
        f"{name} face: flux {flux} is not {what} {expected(fan)[:5]} (relative "
        f"difference {error:.3e}; HLL {fan['hll'][:5]})")


def test_star_state_of_negative_internal_energy_takes_the_hll_flux(tmp_path):
    left = _fan(*(np.array(w) for w in FACES["closing"]))["stars"][0]
    assert 0.0 < left["den"] < 1.0e-3 and left["solver"]["eint"] < 0.0, left
    _check(tmp_path, "closing", lambda fan: fan["hll"], "the HLL flux")


def test_merged_fast_and_alfven_waves_take_the_degenerate_state(tmp_path):
    left = _fan(*(np.array(w) for w in FACES["reversed"]))["stars"][0]
    assert left["den"] < 0.0 and left["jump"]["by_ratio"] < 0.0, left
    assert left["solver"]["eint"] > 0.0, left
    _check(tmp_path, "reversed", _mk_flux, "the flux of the degenerate star state")


def test_admissible_fan_keeps_its_star_states(tmp_path):
    fan = _fan(*(np.array(w) for w in FACES["ordinary"]))
    # an admissible fan whose star gas pressure p_T* - B*^2/2 is negative
    for star in fan["stars"]:
        assert star["den"] > 1.0e-4 and star["solver"]["eint"] > 0.0, star
    assert fan["stars"][0]["solver"]["pgas"] < 0.0
    _check(tmp_path, "ordinary", _mk_flux, "the Miyoshi & Kusano flux")
