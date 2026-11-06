#ifndef HYDRO_RSOLVERS_HLLE_GRHYD_HPP_
#define HYDRO_RSOLVERS_HLLE_GRHYD_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file hlle_grhyd.hpp
//! \brief HLLE Riemann solver for general relativistic hydrodynamics.  Per-face
//! implementation for the split-kernel path.
//!
//! Notes:
//!  - cf. HLLE solver in hlle_rel_no_transform.cpp in Athena++

#include <algorithm>  // max(), min()
#include <cmath>      // sqrt()

#include "coordinates/cartesian_ks.hpp"
#include "coordinates/cell_locations.hpp"

namespace hydro {
//----------------------------------------------------------------------------------------
//! \fn HLLE_GR<ivx>()
//! \brief HLLE flux at face (m,k,j,i) for GR hydrodynamics
template <int ivx>
KOKKOS_INLINE_FUNCTION
void HLLE_GR(const EOS_Data &eos, const RegionIndcs &indcs,
             const DualArray1D<RegionSize> &size, const CoordData &coord,
             const int m, const int mb, const int k, const int j, const int i,
             const int is, const int js, const int ks,
             const DvceArray5D<Real> &wl,
             const DvceArray5D<Real> &wr,
             const BandView5D<Real> &flx) {
  constexpr int ivy = IVX + ((ivx-IVX)+1)%3;
  constexpr int ivz = IVX + ((ivx-IVX)+2)%3;
  const Real gamma_prime = eos.gamma/(eos.gamma - 1.0);
  const KerrHole hole = coord.Hole();


  // Left/right primitives
  Real wl_idn = wl(mb,IDN,k,j,i);
  Real wl_ivx = wl(mb,ivx,k,j,i);
  Real wl_ivy = wl(mb,ivy,k,j,i);
  Real wl_ivz = wl(mb,ivz,k,j,i);

  Real wr_idn = wr(mb,IDN,k,j,i);
  Real wr_ivx = wr(mb,ivx,k,j,i);
  Real wr_ivy = wr(mb,ivy,k,j,i);
  Real wr_ivz = wr(mb,ivz,k,j,i);

  Real wl_ipr = eos.IdealGasPressure(wl(mb,IEN,k,j,i));
  Real wr_ipr = eos.IdealGasPressure(wr(mb,IEN,k,j,i));

  // Extract position of interface
  Real &x1min = size.d_view(m).x1min;
  Real &x1max = size.d_view(m).x1max;
  Real &x2min = size.d_view(m).x2min;
  Real &x2max = size.d_view(m).x2max;
  Real &x3min = size.d_view(m).x3min;
  Real &x3max = size.d_view(m).x3max;
  Real x1v,x2v,x3v;
  if (ivx == IVX) {
    x1v = LeftEdgeX  (i-is, indcs.nx1, x1min, x1max);
    x2v = CellCenterX(j-js, indcs.nx2, x2min, x2max);
    x3v = CellCenterX(k-ks, indcs.nx3, x3min, x3max);
  } else if (ivx == IVY) {
    x1v = CellCenterX(i-is, indcs.nx1, x1min, x1max);
    x2v = LeftEdgeX  (j-js, indcs.nx2, x2min, x2max);
    x3v = CellCenterX(k-ks, indcs.nx3, x3min, x3max);
  } else {
    x1v = CellCenterX(i-is, indcs.nx1, x1min, x1max);
    x2v = CellCenterX(j-js, indcs.nx2, x2min, x2max);
    x3v = LeftEdgeX  (k-ks, indcs.nx3, x3min, x3max);
  }
  // The Cartesian Kerr-Schild metric is g = eta + f l l, so its null form (f, l_1, l_2,
  // l_3) in place of glower[4][4]/gupper[4][4] keeps 4 live doubles in place of 32; every
  // component used below is recovered exactly by the accessors in cartesian_ks.hpp (the
  // same form as hlle_grmhd.hpp).
  KSNullForm nf;
  ComputeKSNullForm(x1v, x2v, x3v, hole, nf);

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

  // Calculate fluxes in HLL region
  HydCons1D flux_hll;
  qa = lambda_r*lambda_l;
  qb = 1.0/(lambda_r - lambda_l);
  flux_hll.d  = (lambda_r*fl.d  - lambda_l*fr.d  + qa*du.d ) * qb;
  flux_hll.mx = (lambda_r*fl.mx - lambda_l*fr.mx + qa*du.mx) * qb;
  flux_hll.my = (lambda_r*fl.my - lambda_l*fr.my + qa*du.my) * qb;
  flux_hll.mz = (lambda_r*fl.mz - lambda_l*fr.mz + qa*du.mz) * qb;
  flux_hll.e  = (lambda_r*fl.e  - lambda_l*fr.e  + qa*du.e ) * qb;

  // Determine region of wavefan
  HydCons1D *flux_interface;
  if (lambda_l >= 0.0) {  // L region
    flux_interface = &fl;
  } else if (lambda_r <= 0.0) { // R region
    flux_interface = &fr;
  } else {  // HLL region
    flux_interface = &flux_hll;
  }

  // Set fluxes
  flx(m,IDN,k,j,i) = flux_interface->d;
  flx(m,ivx,k,j,i) = flux_interface->mx;
  flx(m,ivy,k,j,i) = flux_interface->my;
  flx(m,ivz,k,j,i) = flux_interface->mz;
  flx(m,IEN,k,j,i) = flux_interface->e;

  // We evolve tau = T^t_t + D
  flx(m,IEN,k,j,i) += flx(m,IDN,k,j,i);
}
} // namespace hydro
#endif // HYDRO_RSOLVERS_HLLE_GRHYD_HPP_
