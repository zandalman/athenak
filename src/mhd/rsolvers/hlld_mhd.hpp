#ifndef MHD_RSOLVERS_HLLD_MHD_HPP_
#define MHD_RSOLVERS_HLLD_MHD_HPP_
//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file hlld_mhd.hpp
//! \brief HLLD Riemann solver for MHD.  Per-face implementation, split-kernel path.
//!
//! REFERENCES:
//! - T. Miyoshi & K. Kusano, "A multi-state HLL approximate Riemann solver for ideal
//!   MHD", JCP, 208, 315 (2005)

#include <cmath>
#include <limits>

#include "eos/general_c2p_hyd.hpp"

namespace mhd {

#define HLLD_SMALL_NUMBER 1.0e-4

//----------------------------------------------------------------------------------------
//! \fn HLLDStarAdmissible
//! \brief Condition (2) of the fan, rho* > 0 and 2 e - m^2/rho - B^2 >= 0, for a * state.
//! The internal energy is the difference of terms of at most 2 e + m^2/rho + B^2, which
//! carry the rounding of the star-state algebra: a state whose internal energy is
//! negative by no more than the cancellation scale of the dual-energy channel
//! (eos_general::kDualEnergyChannelRoundingEps) is admissible to round-off, and only
//! a physically inadmissible one collapses the fan.  A NaN fails.
KOKKOS_INLINE_FUNCTION
bool HLLDStarAdmissible(const Real d, const Real kinetic, const Real e2,
                        const Real bsq) {
  const Real eint2 = e2 - kinetic - bsq;
  const Real scale = e2 + kinetic + bsq;
  return (d > 0.0) &&
         (eint2 >= -eos_general::kDualEnergyChannelRoundingEps*
                    std::numeric_limits<Real>::epsilon()*scale);
}

//----------------------------------------------------------------------------------------
//! \fn HLLD<ivx>()
//! \brief HLLD Riemann solver for MHD (energy-evolving and isothermal), single face.
template <int ivx>
KOKKOS_INLINE_FUNCTION
Real HLLD(const EOS_Data &eos,
          const int m, const int mb, const int k, const int j, const int i,
          const int is, const int js, const int ks,
          const DvceArray5D<Real> &wl, const DvceArray5D<Real> &wr,
          const DvceArray5D<Real> &bl, const DvceArray5D<Real> &br,
          const DvceArray4D<Real> &bx,
          const BandView5D<Real> &flx,
          const BandView4D<Real> &ey, const BandView4D<Real> &ez) {
  constexpr int ivy = IVX + ((ivx-IVX)+1)%3;
  constexpr int ivz = IVX + ((ivx-IVX)+2)%3;
  constexpr int iby = ((ivx-IVX) + 1)%3;
  constexpr int ibz = ((ivx-IVX) + 2)%3;
  Real spd[5];         // signal speeds, left to right
  Real interface_velocity = 0.0;


  //------------------------ ADIABATIC HLLD solver ---------------------------------------
  if (eos.use_e) {
    //--- Step 1.  Load L/R states

    Real wl_idn = wl(mb,IDN,k,j,i);
    Real wl_ivx = wl(mb,ivx,k,j,i);
    Real wl_ivy = wl(mb,ivy,k,j,i);
    Real wl_ivz = wl(mb,ivz,k,j,i);
    Real wl_iby = bl(mb,iby,k,j,i);
    Real wl_ibz = bl(mb,ibz,k,j,i);

    Real wr_idn = wr(mb,IDN,k,j,i);
    Real wr_ivx = wr(mb,ivx,k,j,i);
    Real wr_ivy = wr(mb,ivy,k,j,i);
    Real wr_ivz = wr(mb,ivz,k,j,i);
    Real wr_iby = br(mb,iby,k,j,i);
    Real wr_ibz = br(mb,ibz,k,j,i);

    Real wl_ipr, wl_cs2, wr_ipr, wr_cs2;
    eos.EvalPressureCs2FromRhoEint(
        wl_idn, wl(mb,IEN,k,j,i), wl_ipr, wl_cs2);
    eos.EvalPressureCs2FromRhoEint(
        wr_idn, wr(mb,IEN,k,j,i), wr_ipr, wr_cs2);

    Real bxi = bx(m,k,j,i);

    // Compute L/R states for selected conserved variables
    Real bxsq = bxi*bxi;
    // (KGF): group transverse components for floating-point associativity symmetry
    Real pbl = 0.5*(bxsq + (SQR(wl_iby) + SQR(wl_ibz)));  // magnetic pressure (l/r)
    Real pbr = 0.5*(bxsq + (SQR(wr_iby) + SQR(wr_ibz)));
    Real kel = 0.5*wl_idn*(SQR(wl_ivx) + (SQR(wl_ivy) + SQR(wl_ivz)));
    Real ker = 0.5*wr_idn*(SQR(wr_ivx) + (SQR(wr_ivy) + SQR(wr_ivz)));

    MHDCons1D ul,ur;  // L/R states, conserved variables (computed)
    ul.d  = wl_idn;
    ul.mx = wl_ivx*ul.d;
    ul.my = wl_ivy*ul.d;
    ul.mz = wl_ivz*ul.d;
    ul.e  = wl(mb,IEN,k,j,i) + kel + pbl;
    ul.by = wl_iby;
    ul.bz = wl_ibz;

    ur.d  = wr_idn;
    ur.mx = wr_ivx*ur.d;
    ur.my = wr_ivy*ur.d;
    ur.mz = wr_ivz*ur.d;
    ur.e  = wr(mb,IEN,k,j,i) + ker + pbr;
    ur.by = wr_iby;
    ur.bz = wr_ibz;

    //--- Step 2.  Compute L & R wave speeds according to Miyoshi & Kusano, eqn. (67)

    Real cfl = eos.FastMagnetosonicSpeedFromSoundSpeed2(
        wl_idn, wl_cs2, bxi, wl_iby, wl_ibz);
    Real cfr = eos.FastMagnetosonicSpeedFromSoundSpeed2(
        wr_idn, wr_cs2, bxi, wr_iby, wr_ibz);

    spd[0] = fmin( wl_ivx-cfl, wr_ivx-cfr );
    spd[4] = fmax( wl_ivx+cfl, wr_ivx+cfr );

    //--- Step 3.  Compute L/R fluxes

    Real ptl = wl_ipr + pbl; // total pressures L,R
    Real ptr = wr_ipr + pbr;

    MHDCons1D fl,fr,flxi;           // Fluxes for left & right states
    fl.d  = ul.mx;
    fl.mx = ul.mx*wl_ivx + ptl - bxsq;
    fl.my = ul.my*wl_ivx - bxi*ul.by;
    fl.mz = ul.mz*wl_ivx - bxi*ul.bz;
    fl.e  = wl_ivx*(ul.e + ptl - bxsq) - bxi*(wl_ivy*ul.by + wl_ivz*ul.bz);
    fl.by = ul.by*wl_ivx - bxi*wl_ivy;
    fl.bz = ul.bz*wl_ivx - bxi*wl_ivz;

    fr.d  = ur.mx;
    fr.mx = ur.mx*wr_ivx + ptr - bxsq;
    fr.my = ur.my*wr_ivx - bxi*ur.by;
    fr.mz = ur.mz*wr_ivx - bxi*ur.bz;
    fr.e  = wr_ivx*(ur.e + ptr - bxsq) - bxi*(wr_ivy*ur.by + wr_ivz*ur.bz);
    fr.by = ur.by*wr_ivx - bxi*wr_ivy;
    fr.bz = ur.bz*wr_ivx - bxi*wr_ivz;

    //--- Step 4.  Compute middle and Alfven wave speeds

    Real sdl = spd[0] - wl_ivx;  // S_i-u_i (i=L or R)
    Real sdr = spd[4] - wr_ivx;

    // S_M: eqn (38) of Miyoshi & Kusano
    // (KGF): group ptl, ptr terms for floating-point associativity symmetry
    spd[2] = (sdr*ur.mx - sdl*ul.mx + (ptl - ptr))/(sdr*ur.d - sdl*ul.d);

    Real sdml   = spd[0] - spd[2];  // S_i-S_M (i=L or R)
    Real sdmr   = spd[4] - spd[2];
    Real sdml_inv = 1.0/sdml;
    Real sdmr_inv = 1.0/sdmr;

    MHDCons1D ulst,uldst,urdst,urst;   // intermediate states for conserved variables
    // eqn (43) of Miyoshi & Kusano
    ulst.d = ul.d * sdl * sdml_inv;
    urst.d = ur.d * sdr * sdmr_inv;
    Real ulst_d_inv = 1.0/ulst.d;
    Real urst_d_inv = 1.0/urst.d;
    Real sqrtdl = sqrt(ulst.d);
    Real sqrtdr = sqrt(urst.d);

    // eqn (51) of Miyoshi & Kusano
    spd[1] = spd[2] - fabs(bxi)/sqrtdl;
    spd[3] = spd[2] + fabs(bxi)/sqrtdr;

    //--- Step 5.  Compute intermediate states
    // eqn (23) explicitly becomes eq (41) of Miyoshi & Kusano
    // TODO(felker): place an assertion that ptstl==ptstr
    Real ptstl = ptl + ul.d*sdl*(spd[2]-wl_ivx);
    Real ptstr = ptr + ur.d*sdr*(spd[2]-wr_ivx);
    Real ptst = 0.5*(ptstr + ptstl);  // total pressure (star state)

    // ul* - eqn (39) of M&K
    // Admissibility of the fan, from its jump and integral conditions.  Across S_L the
    // transverse momentum and induction jumps, with the mass flux rho (S - u) =
    // rho* (S - S_M), give B_t* den = B_t num (eqns (44)-(47)), where
    //   num = rho (S - u)^2 - Bx^2 >= rho c_f^2 - Bx^2 >= 0 (eqn (67)), and
    //   den = rho* (S - S_M)^2 - Bx^2 = rho* [(S - S_M)^2 - (S* - S_M)^2].
    // (1) A fast wave is at least as fast as the Alfven wave of the state behind it and
    //     does not reverse B_t.  den <= 0 (S* at or beyond S) therefore means that this
    //     side's fast and Alfven waves have merged: the merged wave carries the rotation
    //     of B_t in its Alfven part (the ** states) and no transverse jump in its fast
    //     part.  That is M&K's degenerate state for den = 0, which den <= 0 joins.  (The
    //     jump conditions would otherwise reverse B_t, and as den -> 0- amplify it
    //     without bound.)
    // (2) Every state of the fan has rho > 0 and e - rho v^2/2 - B^2/2 >= 0: the update
    //     is a convex combination of the fan's states (CFL <= 1/2), and only then stays
    //     physical.  As den -> 0+ the * region (width S* - S, proportional to den) closes
    //     while B_t* grows as 1/den, so its magnetic energy (S* - S) B_t*^2/2 grows as
    //     1/den against the fan's fixed total, sum (width e) = (S_R - S_L) E_hll: the
    //     energy jump (eqn (48)) leaves the * state a negative internal energy.  The
    //     rotational jumps (eqns (59)-(63)) change e by exactly the change of kinetic
    //     plus magnetic energy: the ** states have the internal energy of the * states.
    //     A fan failing (2) collapses onto the single HLL state of the same outer speeds,
    //     which meets the integral condition by construction (after ur* below).
    // Neither test reads the star gas pressure p_T* - B*^2/2: p_T* (eqn (41)) is an HLL
    // estimate whose error exceeds the gas pressure at low beta.  (A face of the
    // low-beta Brio-Wu tube with den = 1.4e-4 p_T* gave B_y* = 359, a * state internal
    // energy of -15 p_T*, and an energy flux of -1.86 against +0.20 from HLL.)
    ulst.mx = ulst.d * spd[2];
    const Real denl = ul.d*sdl*sdml - bxsq;
    if (fabs(denl) < (HLLD_SMALL_NUMBER)*ptst || !(denl > 0.0)) {
      // Degenerate case
      ulst.my = ulst.d * wl_ivy;
      ulst.mz = ulst.d * wl_ivz;

      ulst.by = ul.by;
      ulst.bz = ul.bz;
    } else {
      // eqns (44) and (46) of M&K
      Real tmp = bxi*(sdl - sdml)/denl;
      ulst.my = ulst.d * (wl_ivy - ul.by*tmp);
      ulst.mz = ulst.d * (wl_ivz - ul.bz*tmp);

      // eqns (45) and (47) of M&K
      tmp = (ul.d*SQR(sdl) - bxsq)/denl;
      ulst.by = ul.by * tmp;
      ulst.bz = ul.bz * tmp;
    }
    // v_i* dot B_i*
    // (KGF): group transverse momenta terms for floating-point associativity symmetry
    Real vbstl = (ulst.mx*bxi+(ulst.my*ulst.by+ulst.mz*ulst.bz))*ulst_d_inv;
    // eqn (48) of M&K
    // (KGF): group transverse by, bz terms for floating-point associativity symmetry
    ulst.e = (sdl*ul.e - ptl*wl_ivx + ptst*spd[2] +
              bxi*(wl_ivx*bxi + (wl_ivy*ul.by + wl_ivz*ul.bz) - vbstl))*sdml_inv;
    // (2) for ul*, doubled (a NaN fails it)
    bool fan_admissible = HLLDStarAdmissible(
        ulst.d, (SQR(ulst.mx) + (SQR(ulst.my) + SQR(ulst.mz)))*ulst_d_inv,
        2.0*ulst.e, bxsq + (SQR(ulst.by) + SQR(ulst.bz)));

    // ur* - eqn (39) of M&K, admissible as ul*
    urst.mx = urst.d * spd[2];
    const Real denr = ur.d*sdr*sdmr - bxsq;
    if (fabs(denr) < (HLLD_SMALL_NUMBER)*ptst || !(denr > 0.0)) {
      // Degenerate case
      urst.my = urst.d * wr_ivy;
      urst.mz = urst.d * wr_ivz;

      urst.by = ur.by;
      urst.bz = ur.bz;
    } else {
      // eqns (44) and (46) of M&K
      Real tmp = bxi*(sdr - sdmr)/denr;
      urst.my = urst.d * (wr_ivy - ur.by*tmp);
      urst.mz = urst.d * (wr_ivz - ur.bz*tmp);

      // eqns (45) and (47) of M&K
      tmp = (ur.d*SQR(sdr) - bxsq)/denr;
      urst.by = ur.by * tmp;
      urst.bz = ur.bz * tmp;
    }
    // v_i* dot B_i*
    // (KGF): group transverse momenta terms for floating-point associativity symmetry
    Real vbstr = (urst.mx*bxi+(urst.my*urst.by+urst.mz*urst.bz))*urst_d_inv;
    // eqn (48) of M&K
    // (KGF): group transverse by, bz terms for floating-point associativity symmetry
    urst.e = (sdr*ur.e - ptr*wr_ivx + ptst*spd[2] +
              bxi*(wr_ivx*bxi + (wr_ivy*ur.by + wr_ivz*ur.bz) - vbstr))*sdmr_inv;
    // (2) for ur*
    fan_admissible = fan_admissible && HLLDStarAdmissible(
        urst.d, (SQR(urst.mx) + (SQR(urst.my) + SQR(urst.mz)))*urst_d_inv,
        2.0*urst.e, bxsq + (SQR(urst.by) + SQR(urst.bz)));
    // A fan failing (2) collapses onto its HLL state: U_L* = U_R* = U_hll, with the
    // rotational waves on the contact, so that Fl* and Fr* below both equal the HLL flux.
    if (!fan_admissible) {
      Real idspd = 1.0/(spd[4] - spd[0]);
      ulst.d  = (spd[4]*ur.d  - spd[0]*ul.d  - (fr.d  - fl.d ))*idspd;
      ulst.mx = (spd[4]*ur.mx - spd[0]*ul.mx - (fr.mx - fl.mx))*idspd;
      ulst.my = (spd[4]*ur.my - spd[0]*ul.my - (fr.my - fl.my))*idspd;
      ulst.mz = (spd[4]*ur.mz - spd[0]*ul.mz - (fr.mz - fl.mz))*idspd;
      ulst.e  = (spd[4]*ur.e  - spd[0]*ul.e  - (fr.e  - fl.e ))*idspd;
      ulst.by = (spd[4]*ur.by - spd[0]*ul.by - (fr.by - fl.by))*idspd;
      ulst.bz = (spd[4]*ur.bz - spd[0]*ul.bz - (fr.bz - fl.bz))*idspd;
      urst.d  = ulst.d;
      urst.mx = ulst.mx;
      urst.my = ulst.my;
      urst.mz = ulst.mz;
      urst.e  = ulst.e;
      urst.by = ulst.by;
      urst.bz = ulst.bz;
      spd[1] = spd[2];
      spd[3] = spd[2];
    }
    // ul** and ur** - eqns (59)-(63) of M&K.  For any B_x != 0 the rotational waves
    // stand |B_x|/sqrt(rho*) either side of the contact (eqn (51)), and the ** flux
    // differs from the * flux by S*(U** - U*), a term of order B_x, not B_x^2.
    // Collapsing U** onto U* below a finite 0.5 B_x^2 < 1e-4 p_T* (M&K do not have that
    // test) drops the transverse jump across the rotational waves and overshoots v_t and
    // B_t there (Athena++ PR 620, AthenaK PR 771).  Only at B_x = 0 exactly do the
    // rotational waves merge with the contact, so that no ** flux is selected below;
    // there U** = U*.  The same holds for a collapsed fan (!fan_admissible): U* is the HLL
    // state, the rotational waves sit on the contact, and no ** flux is selected.
    if (bxi == 0.0 || !fan_admissible) {
      uldst.d = ulst.d;
      uldst.mx = ulst.mx;
      uldst.my = ulst.my;
      uldst.mz = ulst.mz;
      uldst.e = ulst.e;
      uldst.by = ulst.by;
      uldst.bz = ulst.bz;

      urdst.d = urst.d;
      urdst.mx = urst.mx;
      urdst.my = urst.my;
      urdst.mz = urst.mz;
      urdst.e = urst.e;
      urdst.by = urst.by;
      urdst.bz = urst.bz;
    } else {
      Real invsumd = 1.0/(sqrtdl + sqrtdr);
      Real bxsig = (bxi > 0.0 ? 1.0 : -1.0);

      uldst.d = ulst.d;
      urdst.d = urst.d;

      uldst.mx = ulst.mx;
      urdst.mx = urst.mx;

      // eqn (59) of M&K
      Real tmp = invsumd*(sqrtdl*(ulst.my*ulst_d_inv) + sqrtdr*(urst.my*urst_d_inv) +
                          bxsig*(urst.by - ulst.by));
      uldst.my = uldst.d * tmp;
      urdst.my = urdst.d * tmp;

      // eqn (60) of M&K
      tmp = invsumd*(sqrtdl*(ulst.mz*ulst_d_inv) + sqrtdr*(urst.mz*urst_d_inv) +
                     bxsig*(urst.bz - ulst.bz));
      uldst.mz = uldst.d * tmp;
      urdst.mz = urdst.d * tmp;

      // eqn (61) of M&K
      tmp = invsumd*(sqrtdl*urst.by + sqrtdr*ulst.by +
                     bxsig*sqrtdl*sqrtdr*((urst.my*urst_d_inv) - (ulst.my*ulst_d_inv)));
      uldst.by = urdst.by = tmp;

      // eqn (62) of M&K
      tmp = invsumd*(sqrtdl*urst.bz + sqrtdr*ulst.bz +
                     bxsig*sqrtdl*sqrtdr*((urst.mz*urst_d_inv) - (ulst.mz*ulst_d_inv)));
      uldst.bz = urdst.bz = tmp;

      // eqn (63) of M&K
      tmp = spd[2]*bxi + (uldst.my*uldst.by + uldst.mz*uldst.bz)/uldst.d;
      uldst.e = ulst.e - sqrtdl*bxsig*(vbstl - tmp);
      urdst.e = urst.e + sqrtdr*bxsig*(vbstr - tmp);
    }

    //--- Step 6.  Compute flux
    uldst.d = spd[1] * (uldst.d - ulst.d);
    uldst.mx = spd[1] * (uldst.mx - ulst.mx);
    uldst.my = spd[1] * (uldst.my - ulst.my);
    uldst.mz = spd[1] * (uldst.mz - ulst.mz);
    uldst.e = spd[1] * (uldst.e - ulst.e);
    uldst.by = spd[1] * (uldst.by - ulst.by);
    uldst.bz = spd[1] * (uldst.bz - ulst.bz);

    ulst.d = spd[0] * (ulst.d - ul.d);
    ulst.mx = spd[0] * (ulst.mx - ul.mx);
    ulst.my = spd[0] * (ulst.my - ul.my);
    ulst.mz = spd[0] * (ulst.mz - ul.mz);
    ulst.e = spd[0] * (ulst.e - ul.e);
    ulst.by = spd[0] * (ulst.by - ul.by);
    ulst.bz = spd[0] * (ulst.bz - ul.bz);

    urdst.d = spd[3] * (urdst.d - urst.d);
    urdst.mx = spd[3] * (urdst.mx - urst.mx);
    urdst.my = spd[3] * (urdst.my - urst.my);
    urdst.mz = spd[3] * (urdst.mz - urst.mz);
    urdst.e = spd[3] * (urdst.e - urst.e);
    urdst.by = spd[3] * (urdst.by - urst.by);
    urdst.bz = spd[3] * (urdst.bz - urst.bz);

    urst.d = spd[4] * (urst.d  - ur.d);
    urst.mx = spd[4] * (urst.mx - ur.mx);
    urst.my = spd[4] * (urst.my - ur.my);
    urst.mz = spd[4] * (urst.mz - ur.mz);
    urst.e = spd[4] * (urst.e - ur.e);
    urst.by = spd[4] * (urst.by - ur.by);
    urst.bz = spd[4] * (urst.bz - ur.bz);

    if (spd[0] >= 0.0) {
      // return Fl if flow is supersonic
      interface_velocity = wl_ivx;
      flxi.d = fl.d;
      flxi.mx = fl.mx;
      flxi.my = fl.my;
      flxi.mz = fl.mz;
      flxi.e  = fl.e;
      flxi.by = fl.by;
      flxi.bz = fl.bz;
    } else if (spd[4] <= 0.0) {
      // return Fr if flow is supersonic
      interface_velocity = wr_ivx;
      flxi.d = fr.d;
      flxi.mx = fr.mx;
      flxi.my = fr.my;
      flxi.mz = fr.mz;
      flxi.e  = fr.e;
      flxi.by = fr.by;
      flxi.bz = fr.bz;
    } else if (spd[1] >= 0.0) {
      // return Fl*
      interface_velocity = spd[2];
      flxi.d = fl.d  + ulst.d;
      flxi.mx = fl.mx + ulst.mx;
      flxi.my = fl.my + ulst.my;
      flxi.mz = fl.mz + ulst.mz;
      flxi.e  = fl.e  + ulst.e;
      flxi.by = fl.by + ulst.by;
      flxi.bz = fl.bz + ulst.bz;
    } else if (spd[2] >= 0.0) {
      // return Fl**
      interface_velocity = spd[2];
      flxi.d = fl.d  + ulst.d + uldst.d;
      flxi.mx = fl.mx + ulst.mx + uldst.mx;
      flxi.my = fl.my + ulst.my + uldst.my;
      flxi.mz = fl.mz + ulst.mz + uldst.mz;
      flxi.e  = fl.e  + ulst.e + uldst.e;
      flxi.by = fl.by + ulst.by + uldst.by;
      flxi.bz = fl.bz + ulst.bz + uldst.bz;
    } else if (spd[3] > 0.0) {
      // return Fr**
      interface_velocity = spd[2];
      flxi.d = fr.d + urst.d + urdst.d;
      flxi.mx = fr.mx + urst.mx + urdst.mx;
      flxi.my = fr.my + urst.my + urdst.my;
      flxi.mz = fr.mz + urst.mz + urdst.mz;
      flxi.e  = fr.e + urst.e + urdst.e;
      flxi.by = fr.by + urst.by + urdst.by;
      flxi.bz = fr.bz + urst.bz + urdst.bz;
    } else {
      // return Fr*
      interface_velocity = spd[2];
      flxi.d = fr.d  + urst.d;
      flxi.mx = fr.mx + urst.mx;
      flxi.my = fr.my + urst.my;
      flxi.mz = fr.mz + urst.mz;
      flxi.e  = fr.e  + urst.e;
      flxi.by = fr.by + urst.by;
      flxi.bz = fr.bz + urst.bz;
    }

    flx(m,IDN,k,j,i) = flxi.d;
    flx(m,ivx,k,j,i) = flxi.mx;
    flx(m,ivy,k,j,i) = flxi.my;
    flx(m,ivz,k,j,i) = flxi.mz;
    flx(m,IEN,k,j,i) = flxi.e;
    ey(m,k,j,i) = -flxi.by;
    ez(m,k,j,i) =  flxi.bz;

  //------------------------- ISOTHERMAL HLLD solver -------------------------------------
  } else {
    auto &dfloor_ = eos.dfloor;
    Real iso_cs = eos.iso_cs;
    //--- Step 1.  Load L/R states into local variables

    Real wl_idn = wl(mb,IDN,k,j,i);
    Real wl_ivx = wl(mb,ivx,k,j,i);
    Real wl_ivy = wl(mb,ivy,k,j,i);
    Real wl_ivz = wl(mb,ivz,k,j,i);
    Real wl_iby = bl(mb,iby,k,j,i);
    Real wl_ibz = bl(mb,ibz,k,j,i);

    Real wr_idn = wr(mb,IDN,k,j,i);
    Real wr_ivx = wr(mb,ivx,k,j,i);
    Real wr_ivy = wr(mb,ivy,k,j,i);
    Real wr_ivz = wr(mb,ivz,k,j,i);
    Real wr_iby = br(mb,iby,k,j,i);
    Real wr_ibz = br(mb,ibz,k,j,i);

    Real bxi = bx(m,k,j,i);

    // Compute L/R states for selected conserved variables
    MHDCons1D ul,ur;
    ul.d  = wl_idn;
    ul.mx = wl_ivx*ul.d;
    ul.my = wl_ivy*ul.d;
    ul.mz = wl_ivz*ul.d;
    ul.by = wl_iby;
    ul.bz = wl_ibz;

    ur.d  = wr_idn;
    ur.mx = wr_ivx*ur.d;
    ur.my = wr_ivy*ur.d;
    ur.mz = wr_ivz*ur.d;
    ur.by = wr_iby;
    ur.bz = wr_ibz;

    //--- Step 2.  Compute L & R wave speeds according to Miyoshi & Kusano, eqn. (67)

    Real cfl = eos.IdealMHDFastSpeed(wl_idn, bxi, wl_iby, wl_ibz);
    Real cfr = eos.IdealMHDFastSpeed(wr_idn, bxi, wr_iby, wr_ibz);

    spd[0] = fmin( wl_ivx-cfl, wr_ivx-cfr );
    spd[4] = fmax( wl_ivx+cfl, wr_ivx+cfr );

    //--- Step 3.  Compute L/R fluxes

    // total pressures L,R
    Real bxsq = bxi*bxi;
    Real ptl = SQR(iso_cs)*wl_idn + 0.5*(bxsq + SQR(wl_iby) + SQR(wl_ibz));
    Real ptr = SQR(iso_cs)*wr_idn + 0.5*(bxsq + SQR(wr_iby) + SQR(wr_ibz));

    MHDCons1D fl,fr,flxi;  // Fluxes for left & right states, and interface
    fl.d  = ul.mx;
    fl.mx = ul.mx*wl_ivx + ptl - bxsq;
    fl.my = ul.my*wl_ivx - bxi*ul.by;
    fl.mz = ul.mz*wl_ivx - bxi*ul.bz;
    fl.by = ul.by*wl_ivx - bxi*wl_ivy;
    fl.bz = ul.bz*wl_ivx - bxi*wl_ivz;

    fr.d  = ur.mx;
    fr.mx = ur.mx*wr_ivx + ptr - bxsq;
    fr.my = ur.my*wr_ivx - bxi*ur.by;
    fr.mz = ur.mz*wr_ivx - bxi*ur.bz;
    fr.by = ur.by*wr_ivx - bxi*wr_ivy;
    fr.bz = ur.bz*wr_ivx - bxi*wr_ivz;

    //--- Step 4.  Compute hll averages and Alfven wave speed

    // inverse of difference between right and left signal speeds
    Real idspd = 1.0/(spd[4]-spd[0]);

    // rho component of U^{hll} from Mignone eqn. (15); uses F_L and F_R from eqn. (6)
    Real dhll = (spd[4]*ur.d - spd[0]*ul.d - fr.d + fl.d)*idspd;
    dhll = fmax(dhll, dfloor_);
    Real sqrtdhll = sqrt(dhll);

    // rho and mx components of F^{hll} from Mignone eqn. (17)
    Real fdhll  = (spd[4]*fl.d  - spd[0]*fr.d  + spd[4]*spd[0]*(ur.d -ul.d ))*idspd;
    Real fmxhll = (spd[4]*fl.mx - spd[0]*fr.mx + spd[4]*spd[0]*(ur.mx-ul.mx))*idspd;

    // ustar from paragraph between eqns. (23) and (24)
    Real ustar = fdhll/dhll;

    // mx component of U^{hll} from Mignone eqn. (15); paragraph referenced
    // above states that mxhll should NOT be used to compute ustar
    Real mxhll = (spd[4]*ur.mx - spd[0]*ul.mx - fr.mx + fl.mx)*idspd;

    // S*_L and S*_R from Mignone eqn. (29)
    spd[1] = ustar - fabs(bxi)/sqrtdhll;
    spd[3] = ustar + fabs(bxi)/sqrtdhll;

    //--- Step 5. Compute intermediate states

    MHDCons1D ulst,urst,ucst;  // Conserved variable for all states
    // Ul* - eqn. (20) of Mignone
    ulst.d  = dhll;
    ulst.mx = mxhll; // eqn. (24) of Mignone

    Real tmp = (spd[0]-spd[1])*(spd[0]-spd[3]);
    if (fabs(spd[0]-spd[1]) < (HLLD_SMALL_NUMBER)*iso_cs) {
      // degenerate case described below eqn. (39)
      ulst.my = ul.my;
      ulst.mz = ul.mz;
      ulst.by = ul.by;
      ulst.bz = ul.bz;
    } else {
      Real mfact = bxi*(ustar-wl_ivx)/tmp;
      Real bfact = (ul.d*SQR(spd[0]-wl_ivx) - bxsq)/(dhll*tmp);

      ulst.my = dhll*wl_ivy - ul.by*mfact; // eqn. (30) of Mignone
      ulst.mz = dhll*wl_ivz - ul.bz*mfact; // eqn. (31) of Mignone
      ulst.by = ul.by*bfact; // eqn. (32) of Mignone
      ulst.bz = ul.bz*bfact; // eqn. (33) of Mignone
    }

    // Ur* - eqn. (20) of Mignone */
    urst.d  = dhll;
    urst.mx = mxhll; // eqn. (24) of Mignone

    tmp = (spd[4]-spd[1])*(spd[4]-spd[3]);
    if (fabs(spd[4]-spd[3]) < (HLLD_SMALL_NUMBER)*iso_cs) {
      // degenerate case described below eqn. (39)
      urst.my = ur.my;
      urst.mz = ur.mz;
      urst.by = ur.by;
      urst.bz = ur.bz;
    } else {
      Real mfact = bxi*(ustar-wr_ivx)/tmp;
      Real bfact = (ur.d*SQR(spd[4]-wr_ivx) - bxsq)/(dhll*tmp);

      urst.my = dhll*wr_ivy - ur.by*mfact; // eqn. (30) of Mignone
      urst.mz = dhll*wr_ivz - ur.bz*mfact; // eqn. (31) of Mignone
      urst.by = ur.by*bfact; // eqn. (32) of Mignone
      urst.bz = ur.bz*bfact; // eqn. (33) of Mignone
    }

    // Uc*
    Real x = sqrtdhll*(bxi > 0.0 ? 1.0 : -1.0); // from below eqn. (37) of Mignone
    ucst.d  = dhll;  // eqn. (20) of Mignone
    ucst.mx = mxhll; // eqn. (24) of Mignone
    ucst.my = 0.5*(ulst.my + urst.my + (urst.by-ulst.by)*x); // eqn. (34) of Mignone
    ucst.mz = 0.5*(ulst.mz + urst.mz + (urst.bz-ulst.bz)*x); // eqn. (35) of Mignone
    ucst.by = 0.5*(ulst.by + urst.by + (urst.my-ulst.my)/x); // eqn. (36) of Mignone
    ucst.bz = 0.5*(ulst.bz + urst.bz + (urst.mz-ulst.mz)/x); // eqn. (37) of Mignone

    //--- Step 6.  Compute flux

    if (spd[0] >= 0.0) {
      // return Fl if flow is supersonic, eqn. (38a) of Mignone
      interface_velocity = wl_ivx;
      flxi.d  = fl.d;
      flxi.mx = fl.mx;
      flxi.my = fl.my;
      flxi.mz = fl.mz;
      flxi.by = fl.by;
      flxi.bz = fl.bz;
    } else if (spd[4] <= 0.0) {
      // return Fr if flow is supersonic, eqn. (38e) of Mignone
      interface_velocity = wr_ivx;
      flxi.d  = fr.d;
      flxi.mx = fr.mx;
      flxi.my = fr.my;
      flxi.mz = fr.mz;
      flxi.by = fr.by;
      flxi.bz = fr.bz;
    } else if (spd[1] >= 0.0) {
      // return (Fl+Sl*(Ulst-Ul)), eqn. (38b) of Mignone
      interface_velocity = ustar;
      flxi.d  = fl.d  + spd[0]*(ulst.d  - ul.d);
      flxi.mx = fl.mx + spd[0]*(ulst.mx - ul.mx);
      flxi.my = fl.my + spd[0]*(ulst.my - ul.my);
      flxi.mz = fl.mz + spd[0]*(ulst.mz - ul.mz);
      flxi.by = fl.by + spd[0]*(ulst.by - ul.by);
      flxi.bz = fl.bz + spd[0]*(ulst.bz - ul.bz);
    } else if (spd[3] <= 0.0) {
      // return (Fr+Sr*(Urst-Ur)), eqn. (38d) of Mignone
      interface_velocity = ustar;
      flxi.d  = fr.d  + spd[4]*(urst.d  - ur.d);
      flxi.mx = fr.mx + spd[4]*(urst.mx - ur.mx);
      flxi.my = fr.my + spd[4]*(urst.my - ur.my);
      flxi.mz = fr.mz + spd[4]*(urst.mz - ur.mz);
      flxi.by = fr.by + spd[4]*(urst.by - ur.by);
      flxi.bz = fr.bz + spd[4]*(urst.bz - ur.bz);
    } else {
      // return Fcst, eqn. (38c) of Mignone, using eqn. (24)
      interface_velocity = ustar;
      flxi.d  = dhll*ustar;
      flxi.mx = fmxhll;
      flxi.my = ucst.my*ustar - bxi*ucst.by;
      flxi.mz = ucst.mz*ustar - bxi*ucst.bz;
      flxi.by = ucst.by*ustar - bxi*ucst.my/ucst.d;
      flxi.bz = ucst.bz*ustar - bxi*ucst.mz/ucst.d;
    }

    flx(m,IDN,k,j,i) = flxi.d;
    flx(m,ivx,k,j,i) = flxi.mx;
    flx(m,ivy,k,j,i) = flxi.my;
    flx(m,ivz,k,j,i) = flxi.mz;
    ey(m,k,j,i) = -flxi.by;
    ez(m,k,j,i) =  flxi.bz;
  } // end ideal gas/isothermal solvers
  return interface_velocity;
}
} // namespace mhd
#endif // MHD_RSOLVERS_HLLD_MHD_HPP_
