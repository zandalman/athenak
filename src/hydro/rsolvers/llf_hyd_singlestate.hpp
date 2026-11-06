#ifndef HYDRO_RSOLVERS_LLF_HYD_SINGLESTATE_HPP_
#define HYDRO_RSOLVERS_LLF_HYD_SINGLESTATE_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file llf_hyd_singlestate.hpp
//! \brief various Local Lax Friedrichs (LLF) Riemann solvers, also known as Rusanov's
//! method, for NR/SR/GR hydrodynamics.  This flux is very diffusive, even more diffusive
//! than HLLE, and so it is not recommended for use in applications.  However, it is
//! useful for testing, or for problems where other Riemann solvers fail.
//!
//! Each solver in this file works on a single L/R state
//!
//! REFERENCES:
//! - E.F. Toro, "Riemann Solvers and numerical methods for fluid dynamics", 2nd ed.,
//!   Springer-Verlag, Berlin, (1999) chpt. 10.

#include "coordinates/cartesian_ks.hpp"

namespace hydro {
//----------------------------------------------------------------------------------------
//! \fn void SingleStateLLF_HYD
//  \brief The LLF Riemann solver for hydrodynamics for a single L/R state

KOKKOS_INLINE_FUNCTION
void SingleStateLLF_Hyd(const HydPrim1D &wl, const HydPrim1D &wr, const EOS_Data &eos,
                        HydCons1D &flux) {
  Real qa = wl.d*wl.vx;
  Real qb = wr.d*wr.vx;

  // Compute sum of L/R fluxes
  HydCons1D fsum;
  fsum.d  = qa        + qb;
  fsum.mx = qa*wl.vx + qb*wr.vx;
  fsum.my = qa*wl.vy + qb*wr.vy;
  fsum.mz = qa*wl.vz + qb*wr.vz;

  Real el,er,pl,pr;
  EOS_Data::ThermoState wl_thermo, wr_thermo;
  if (eos.use_e) {
    wl_thermo = eos.EvalThermoStateFromRhoEint(wl.d, wl.e);
    wr_thermo = eos.EvalThermoStateFromRhoEint(wr.d, wr.e);
    pl = wl_thermo.pressure;
    pr = wr_thermo.pressure;
    el = wl.e + 0.5*wl.d*(SQR(wl.vx) + SQR(wl.vy) + SQR(wl.vz));
    er = wr.e + 0.5*wr.d*(SQR(wr.vx) + SQR(wr.vy) + SQR(wr.vz));
    fsum.mx += (pl + pr);
    fsum.e  = (el + pl)*wl.vx + (er + pr)*wr.vx;
  } else {
    fsum.mx += SQR(eos.iso_cs)*(wl.d + wr.d);
  }

  // Compute max wave speed in L,R states (see Toro eq. 10.43)
  if (eos.use_e) {
    qa = sqrt(fmax(wl_thermo.cs2, 0.0));
    qb = sqrt(fmax(wr_thermo.cs2, 0.0));
  } else {
    qa = eos.iso_cs;
    qb = eos.iso_cs;
  }
  Real a = fmax( (fabs(wl.vx) + qa), (fabs(wr.vx) + qb) );

  // Compute difference in L/R states dU, multiplied by max wave speed
  HydCons1D du;
  du.d  = a*(wr.d       - wl.d);
  du.mx = a*(wr.d*wr.vx - wl.d*wl.vx);
  du.my = a*(wr.d*wr.vy - wl.d*wl.vy);
  du.mz = a*(wr.d*wr.vz - wl.d*wl.vz);
  if (eos.use_e) du.e = a*(er - el);

  // Compute the LLF flux at interface (see Toro eq. 10.42).
  flux.d  = 0.5*(fsum.d  - du.d );
  flux.mx = 0.5*(fsum.mx - du.mx);
  flux.my = 0.5*(fsum.my - du.my);
  flux.mz = 0.5*(fsum.mz - du.mz);
  if (eos.use_e) {flux.e = 0.5*(fsum.e - du.e);}

  return;
}

//----------------------------------------------------------------------------------------
//! \fn void SingleStateLLF_SRHyd
//  \brief The LLF Riemann solver for SR hydrodynamics for a single L/R state

KOKKOS_INLINE_FUNCTION
void SingleStateLLF_SRHyd(const HydPrim1D &wl, const HydPrim1D &wr, const EOS_Data &eos,
                        HydCons1D &flux) {
  // Recall in SR the primitive variables are (\rho, u^i, P_g), where
  //  \rho is the mass density in the comoving/fluid frame,
  //  u^i = \gamma v^i are the spatial components of the 4-velocity (v^i is the 3-vel),
  //  P_g is the pressure.

  Real u0l  = sqrt(1.0 + SQR(wl.vz) + SQR(wl.vy) + SQR(wl.vx)); // Lorentz fact in L
  Real u0r  = sqrt(1.0 + SQR(wr.vz) + SQR(wr.vy) + SQR(wr.vx)); // Lorentz fact in R
  // FIXME ERM: Ideal fluid for now
  Real wgas_l = wl.d + eos.gamma * wl.e;  // total enthalpy in L-state
  Real wgas_r = wr.d + eos.gamma * wr.e;  // total enthalpy in R-state

  // Compute wave speeds in L,R states (see Toro eq. 10.43)
  Real pl = eos.IdealGasPressure(wl.e);
  Real lp_l, lm_l;
  eos.IdealSRHydroSoundSpeeds(wl.d, pl, wl.vx, u0l, lp_l, lm_l);

  Real pr = eos.IdealGasPressure(wr.e);
  Real lp_r, lm_r;
  eos.IdealSRHydroSoundSpeeds(wr.d, pr, wr.vx, u0r, lp_r, lm_r);

  Real qa = fmax(-fmin(lm_l,lm_r), 0.0);
  Real a = fmax(fmax(lp_l,lp_r), qa);

  // Compute sum of L/R fluxes
  qa = wgas_l * wl.vx;
  Real qb = wgas_r * wr.vx;

  HydCons1D fsum;
  fsum.d  = wl.d * wl.vx + wr.d * wr.vx;
  fsum.mx = qa*wl.vx + qb*wr.vx + (pl + pr);
  fsum.my = qa*wl.vy + qb*wr.vy;
  fsum.mz = qa*wl.vz + qb*wr.vz;
  fsum.e  = qa*u0l + qb*u0r;

  // Compute difference dU = U_R - U_L multiplied by max wave speed
  HydCons1D du;
  qa = wgas_r*u0r;
  qb = wgas_l*u0l;
  Real er = qa*u0r - pr;
  Real el = qb*u0l - pl;
  du.d  = a*(u0r*wr.d  - u0l*wl.d);
  du.mx = a*( qa*wr.vx -  qb*wl.vx);
  du.my = a*( qa*wr.vy -  qb*wl.vy);
  du.mz = a*( qa*wr.vz -  qb*wl.vz);
  du.e  = a*(er - el);

  // Compute the LLF flux at the interface
  flux.d  = 0.5*(fsum.d  - du.d );
  flux.mx = 0.5*(fsum.mx - du.mx);
  flux.my = 0.5*(fsum.my - du.my);
  flux.mz = 0.5*(fsum.mz - du.mz);
  flux.e  = 0.5*(fsum.e  - du.e );

  // We evolve tau = E - D
  flux.e -= flux.d;

  return;
}

//----------------------------------------------------------------------------------------
//! \fn void SingleStateLLF_GRHyd
//! \brief The LLF Riemann solver for GR hydrodynamics for a single L/R state

KOKKOS_INLINE_FUNCTION
void SingleStateLLF_GRHyd(const HydPrim1D wl, const HydPrim1D wr,
                       const Real x1v, const Real x2v, const Real x3v, const int ivx,
                       const CoordData &coord, const EOS_Data &eos, HydCons1D &flux) {
  // Cyclic permutation of array indices
  int ivy = IVX + ((ivx-IVX)+1)%3;
  int ivz = IVX + ((ivx-IVX)+2)%3;
  const Real gamma_prime = eos.gamma/(eos.gamma - 1.0);

  // References to left primitives
  const Real &wl_idn=wl.d;
  const Real &wl_ivx=wl.vx;
  const Real &wl_ivy=wl.vy;
  const Real &wl_ivz=wl.vz;

  // References to right primitives
  const Real &wr_idn=wr.d;
  const Real &wr_ivx=wr.vx;
  const Real &wr_ivy=wr.vy;
  const Real &wr_ivz=wr.vz;

  Real wl_ipr, wr_ipr;
  wl_ipr = eos.IdealGasPressure(wl.e);
  wr_ipr = eos.IdealGasPressure(wr.e);

  // The Cartesian Kerr-Schild metric is g = eta + f l l, so its null form (f, l_1, l_2,
  // l_3) in place of glower[4][4]/gupper[4][4] keeps 4 live doubles in place of 32; every
  // component used below is recovered exactly by the accessors in cartesian_ks.hpp (the
  // same form as hlle_grmhd.hpp).
  KSNullForm nf;
  ComputeKSNullForm(x1v, x2v, x3v, coord.Hole(), nf);

  // Calculate 4-velocity in left state (contravariant compt)
  // q = gamma_ij v^i v^j = |v|^2 + f (l_i v^i)^2
  Real q = KSSpatialNormSq(nf, ivx, ivy, ivz, wl_ivx, wl_ivy, wl_ivz);

  // alpha = sqrt(-1/g^{00}) with g^{00} = -(1+f)
  Real alpha = KSAlpha(nf);
  Real gamma = sqrt(1.0 + q);
  Real uul[4];
  uul[0] = gamma / alpha;
  Real ag = alpha * gamma;
  uul[ivx] = wl_ivx - ag * KSGupper0i(nf, ivx);
  uul[ivy] = wl_ivy - ag * KSGupper0i(nf, ivy);
  uul[ivz] = wl_ivz - ag * KSGupper0i(nf, ivz);

  // lower vector indices (covariant compt)
  Real ull[4];
  KSLowerVec(nf, ivx, ivy, ivz, uul[0], uul[ivx], uul[ivy], uul[ivz],
             ull[0], ull[ivx], ull[ivy], ull[ivz]);

  // Calculate 4-velocity in right state (contravariant compt)
  q = KSSpatialNormSq(nf, ivx, ivy, ivz, wr_ivx, wr_ivy, wr_ivz);

  gamma = sqrt(1.0 + q);
  Real uur[4];
  uur[0] = gamma / alpha;
  ag = alpha * gamma;
  uur[ivx] = wr_ivx - ag * KSGupper0i(nf, ivx);
  uur[ivy] = wr_ivy - ag * KSGupper0i(nf, ivy);
  uur[ivz] = wr_ivz - ag * KSGupper0i(nf, ivz);

  // lower vector indices (covariant compt)
  Real ulr[4];
  KSLowerVec(nf, ivx, ivy, ivz, uur[0], uur[ivx], uur[ivy], uur[ivz],
             ulr[0], ulr[ivx], ulr[ivy], ulr[ivz]);

  // Contravariant metric components entering the wavespeed solve
  const Real gu00 = KSGupper00(nf);
  const Real gu0x = KSGupper0i(nf, ivx);
  const Real guxx = KSGupperij(nf, ivx, ivx);

  // Calculate wavespeeds in left state
  Real lp_l, lm_l;
  eos.IdealGRHydroSoundSpeeds(wl_idn, wl_ipr, uul[0], uul[ivx], gu00, gu0x, guxx,
                              lp_l, lm_l);

  // Calculate wavespeeds in right state
  Real lp_r, lm_r;
  eos.IdealGRHydroSoundSpeeds(wr_idn, wr_ipr, uur[0], uur[ivx], gu00, gu0x, guxx,
                              lp_r, lm_r);

  // Calculate extremal wavespeeds
  Real lambda_l = fmin(lm_l, lm_r);
  Real lambda_r = fmax(lp_l, lp_r);
  Real lambda = fmax(lambda_r, -lambda_l);

  // Calculate difference du =  U_R - U_l in conserved quantities (rho u^0 and T^0_\mu)
  HydCons1D du;
  Real wgas_l = wl_idn + gamma_prime * wl_ipr;
  Real wgas_r = wr_idn + gamma_prime * wr_ipr;
  Real qa = wgas_r * uur[0];
  Real qb = wgas_l * uul[0];
  du.d  = wr_idn * uur[0] - wl_idn * uul[0];
  du.mx = qa * ulr[ivx] - qb * ull[ivx];
  du.my = qa * ulr[ivy] - qb * ull[ivy];
  du.mz = qa * ulr[ivz] - qb * ull[ivz];
  du.e  = qa * ulr[0] - qb * ull[0] + wr_ipr - wl_ipr;

  // Calculate fluxes in L region (rho u^i and T^i_\mu, where i = ivx)
  HydCons1D fl;
  qa = wgas_l * uul[ivx];
  fl.d  = wl_idn * uul[ivx];
  fl.mx = qa * ull[ivx] + wl_ipr;
  fl.my = qa * ull[ivy];
  fl.mz = qa * ull[ivz];
  fl.e  = qa * ull[0];

  // Calculate fluxes in R region (rho u^i and T^i_\mu, where i = ivx)
  HydCons1D fr;
  qa = wgas_r * uur[ivx];
  fr.d  = wr_idn * uur[ivx];
  fr.mx = qa * ulr[ivx] + wr_ipr;
  fr.my = qa * ulr[ivy];
  fr.mz = qa * ulr[ivz];
  fr.e  = qa * ulr[0];

  // Compute the LLF flux at the interface
  flux.d  = 0.5 * (fl.d  + fr.d  - lambda * du.d);
  flux.mx = 0.5 * (fl.mx + fr.mx - lambda * du.mx);
  flux.my = 0.5 * (fl.my + fr.my - lambda * du.my);
  flux.mz = 0.5 * (fl.mz + fr.mz - lambda * du.mz);
  flux.e  = 0.5 * (fl.e  + fr.e  - lambda * du.e);

  // We evolve tau = T^t_t + D
  flux.e  += flux.d;
  return;
}

} // namespace hydro
#endif // HYDRO_RSOLVERS_LLF_HYD_SINGLESTATE_HPP_
