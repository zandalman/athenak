#ifndef EOS_DUAL_ENERGY_SHOCK_RULE_HPP_
#define EOS_DUAL_ENERGY_SHOCK_RULE_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file dual_energy_shock_rule.hpp
//! \brief The dual-energy shock rule of the Newtonian fluids (eos/general_c2p_hyd.hpp),
//! one implementation for Hydro::ApplyDualEnergyFormalism and MHD::
//! ApplyDualEnergyFormalism: the shock test on the step-start state and its reach, the
//! auxiliary's compression, and a shocked cell's stage with the step-final guard.  The
//! fluids differ only in the field: MHD's steady test measures the inflow per MHD family
//! (DualEnergyCellField), hydro's has no field (DualEnergyNoField), and the caller forms
//! the energy channel E - E_K (- E_M) and its magnitude.  The auxiliary's p dV with the
//! entropy of each face's Riemann fan (the face payload, DualEnergyFaceContact,
//! DualEnergyCellPdV, DualEnergyAuxiliaryPdV) is shared as well.  With B = 0 every
//! function here returns for MHD what it returns for hydro, bit for bit
//! (tst/test_suite/nr/test_nr_mhd_dual_hydro_parity_cpu.py).

#include <cmath>

#include "athena.hpp"
#include "eos/eos.hpp"
#include "eos/general_c2p_hyd.hpp"

namespace eos_general {

KOKKOS_INLINE_FUNCTION
Real DualEnergyStencilValue(const DvceArray5D<Real> &a, const int m, const int n,
                            const int dir, const int offset, const int k, const int j,
                            const int i) {
  return a(m, n, (dir == 2) ? (k + offset) : k, (dir == 1) ? (j + offset) : j,
           (dir == 0) ? (i + offset) : i);
}

// The field the shock test of a fluid reads: none for hydro, the stage-start cell-centred
// field bcc0 for MHD.
struct DualEnergyNoField {};
struct DualEnergyCellField {
  DvceArray5D<Real> bcc;
};

// The steady test's inflow u2 = |v - s|^2 at the upstream point (offset `off` along axis
// dir) against the shock whose velocity jump equals the upstream gas sound speed, summed
// over the axes as planar oblique compression adds them (each axis sees M_n |n_d|).
// Hydro: the gas shock (DualEnergyShockMachSq).
KOKKOS_INLINE_FUNCTION
void DualEnergyAddShockInflow(const DualEnergyNoField &, const int, const int, const int,
                              const int, const int, const int, const Real u2,
                              const Real rho_up, const Real p_up, const Real c2_up,
                              Real &fast_ratio_sq, Real &) {
  fast_ratio_sq += u2/(c2_up*DualEnergyShockMachSq(rho_up*c2_up/p_up));
}

// MHD: the fast and the slow shock (DualEnergyShockFamilyMachSq, with the family's speed
// along the axis and Gamma_f = rho c_f^2/P_T, Gamma_s = rho c_s^2/p), the slow family
// only where the inflow is slower than the Alfven speed along the axis, as a slow
// shock's is.
// A simple wave of either family flows in at its own speed and is never flagged.  The
// slow speed takes the product form of the roots (c_f^2 c_sl^2 = a^2 b_d^2), free of
// cancellation; c_f^2 >= a^2 > 0.  With B = 0 the fast speed is a^2 to the bit
// (sqrt(a2*a2) == a2), the fast term is hydro's and the slow family is empty.
KOKKOS_INLINE_FUNCTION
void DualEnergyAddShockInflow(const DualEnergyCellField &field, const int m,
                              const int dir, const int off, const int k, const int j,
                              const int i, const Real u2, const Real rho_up,
                              const Real p_up, const Real a2, Real &fast_ratio_sq,
                              Real &slow_ratio_sq) {
  const Real bx = DualEnergyStencilValue(field.bcc, m, IBX, dir, off, k, j, i);
  const Real by = DualEnergyStencilValue(field.bcc, m, IBY, dir, off, k, j, i);
  const Real bz = DualEnergyStencilValue(field.bcc, m, IBZ, dir, off, k, j, i);
  const Real bd = (dir == 0) ? bx : ((dir == 1) ? by : bz);
  const Real bsq = SQR(bx) + SQR(by) + SQR(bz);
  const Real b2 = bsq/rho_up;
  const Real bd2 = SQR(bd)/rho_up;
  const Real disc = sqrt(fmax(SQR(a2 + b2) - 4.0*a2*bd2, 0.0));
  const Real cf2 = 0.5*(a2 + b2 + disc);
  fast_ratio_sq += u2/(a2*DualEnergyShockFamilyMachSq(rho_up*cf2/(p_up + 0.5*bsq),
                                                      cf2/a2));
  if (u2 < bd2) {
    slow_ratio_sq += u2/(a2*DualEnergyShockFamilyMachSq(rho_up*a2/p_up, bd2/cf2));
  }
}

//----------------------------------------------------------------------------------------
//! \brief Whether gas converges into cell (m,k,j,i) faster than sound: the shock test of
//! the shock rule, on the stage-start primitives w0 (and field) that the stage's fluxes
//! read, owners and ghosts alike.  w0(IEN) is the published thermal state, which is the
//! auxiliary of every recovered cell; (dens, eint_aux) is the cell's thermal state the
//! test measures its own sound speed with.  The threshold is the GAS sound speed in MHD
//! too: a low-beta fast shock weak against c_f still heats the gas, and the auxiliary
//! carries the gas's energy.  Reads w0 (and the field) two cells off along each axis.

template <class Field>
KOKKOS_INLINE_FUNCTION
bool DualEnergyStencilShocked(const EOS_Data &eos, const DvceArray5D<Real> &w0,
                              const Field &field, const int m, const int k, const int j,
                              const int i, const int ndim, const Real dens,
                              const Real eint_aux) {
  Real converging_dv = 0.0;
  bool steady_candidate = false;
  for (int dir = 0; dir < ndim; ++dir) {
    const Real dv_inner = DualEnergyStencilValue(w0, m, IVX + dir, dir, -1, k, j, i) -
                          DualEnergyStencilValue(w0, m, IVX + dir, dir, 1, k, j, i);
    const Real dv_outer = DualEnergyStencilValue(w0, m, IVX + dir, dir, -2, k, j, i) -
                          DualEnergyStencilValue(w0, m, IVX + dir, dir, 2, k, j, i);
    converging_dv += dv_inner;
    steady_candidate = steady_candidate || ((dv_inner > 0.0) && (dv_outer > 0.0));
  }
  // no EOS call where the gas does not converge
  if (!(converging_dv > 0.0) && !steady_candidate) return false;
  Real pressure, cs2;
  eos.EvalPressureCs2FromRhoEint(dens, eint_aux, pressure, cs2);
  if (DualEnergyStencilConvergesSupersonically(converging_dv, cs2)) return true;
  const Real c_self = sqrt(fmax(cs2, 0.0));
  Real fast_ratio_sq = 0.0, slow_ratio_sq = 0.0;
  for (int dir = 0; dir < ndim; ++dir) {
    Real rho[5], vel[5];
    for (int o = 0; o < 5; ++o) {
      rho[o] = DualEnergyStencilValue(w0, m, IDN, dir, o - 2, k, j, i);
      vel[o] = DualEnergyStencilValue(w0, m, IVX + dir, dir, o - 2, k, j, i);
    }
    Real inflow, rho_up;
    int up;
    if (DualEnergySteadyInflowSpeed(rho, vel, c_self, inflow, up, rho_up)) {
      Real p_up, c2_up;
      const Real eint_up = DualEnergyStencilValue(w0, m, IEN, dir, up - 2, k, j, i);
      eos.EvalPressureCs2FromRhoEint(rho_up, eint_up, p_up, c2_up);
      if ((p_up > 0.0) && (c2_up > 0.0)) {
        DualEnergyAddShockInflow(field, m, dir, up - 2, k, j, i, SQR(inflow), rho_up,
                                 p_up, c2_up, fast_ratio_sq, slow_ratio_sq);
      }
    }
  }
  return (fast_ratio_sq > 1.0) || (slow_ratio_sq > 1.0);
}

//----------------------------------------------------------------------------------------
//! \brief The shock test of cell (m,k,j,i), extended by `reach` cells
//! (DualEnergyShockReach) to both sides along each axis: a cell that converges along an
//! axis across both stencil pairs is shocked also when a cell up to `reach` cells away
//! from it along that axis passes DualEnergyStencilShocked on the step-start state.  A
//! captured shock dissipates in converging cells on both sides of the cells whose stencil
//! sees the supersonic inflow: its post-sonic tail lies upwind of the front on the grid
//! when the shock overtakes its gas and downwind when it moves slower than its gas, and a
//! shock structure has no preferred side in the frame of the grid.  Reads w0 (and the
//! field) reach + 2 cells off.  One call site of the stencil test inside the loop, so the
//! kernel carries one copy of it.

template <class Field>
KOKKOS_INLINE_FUNCTION
bool DualEnergyCellShocked(const EOS_Data &eos, const DvceArray5D<Real> &w0,
                           const Field &field, const int m, const int k, const int j,
                           const int i, const int ndim, const int reach, const Real dens,
                           const Real eint_aux) {
  const int nside = ndim*reach;
  for (int c = 0; c <= 2*nside; ++c) {
    int kk = k, jj = j, ii = i;
    Real dens_c = dens, eint_c = eint_aux;
    if (c > 0) {
      const int dir = ((c - 1) % nside)/reach;
      const int dist = (c - 1) % reach + 1;
      if (!(DualEnergyStencilValue(w0, m, IVX + dir, dir, -1, k, j, i) >
            DualEnergyStencilValue(w0, m, IVX + dir, dir, 1, k, j, i)) ||
          !(DualEnergyStencilValue(w0, m, IVX + dir, dir, -2, k, j, i) >
            DualEnergyStencilValue(w0, m, IVX + dir, dir, 2, k, j, i))) {
        continue;
      }
      const int offset = (c <= nside) ? dist : -dist;
      kk += (dir == 2) ? offset : 0;
      jj += (dir == 1) ? offset : 0;
      ii += (dir == 0) ? offset : 0;
      dens_c = w0(m, IDN, kk, jj, ii);
      eint_c = w0(m, IEN, kk, jj, ii);
    }
    if (DualEnergyStencilShocked(eos, w0, field, m, kk, jj, ii, ndim, dens_c, eint_c)) {
      return true;
    }
  }
  return false;
}

//----------------------------------------------------------------------------------------
//! \brief The auxiliary's operator-split compression of one stage, between the thermal
//! floors: the adiabat exp(-(gamma - 1) divv dt) of a gamma-law gas, p divv dt otherwise.

KOKKOS_INLINE_FUNCTION
Real DualEnergyAuxiliaryCompressed(const EOS_Data &eos, const Real dens, Real eint,
                                   const Real divv_dt, const Real eint_floor,
                                   const Real eint_ceil, const Real log_rho_cgs) {
  // The advected auxiliary can undershoot before this update: reapply the thermal floor
  // before any EOS query so the tabulated Saha inversion never sees an out-of-bounds
  // state.
  bool efloor_used = false, tfloor_used = false;
  eint = ApplyHydroThermalFloorsWithBounds(eos, dens, eint, eint_floor, eint_ceil,
                                           efloor_used, tfloor_used);
  if (eos.is_gamma_law) {
    eint *= exp(-(eos.gamma - 1.0)*divv_dt);
  } else {
    const Real pressure = eos.UsesTabulatedLTE() ?
        eos.PressureFromLogRhoEintTabulated(dens, eint, log_rho_cgs) :
        eos.PressureFromRhoEint(dens, eint);
    eint -= pressure*divv_dt;
  }
  efloor_used = false;
  tfloor_used = false;
  return ApplyHydroThermalFloorsWithBounds(eos, dens, eint, eint_floor, eint_ceil,
                                           efloor_used, tfloor_used);
}

//----------------------------------------------------------------------------------------
//! The auxiliary's p dV: the cell's own adiabat on the volume change at the solver's
//! face velocities, plus the entropy each face's Riemann fan produces on its side.  With
//! the face states (p_s, u_s) of a side and the contact (S*, p*) of the two-wave fan
//! between them, the work the contact does on that side's gas beyond its own pressure
//! is -(p* - p_s)(S* - u_s).n, which the fan's jump condition across the
//! outer wave, p* - p_s = d_s sigma_s (S* - u_s), makes d_s |sigma_s| (S* - u_s)^2: never
//! negative, so the auxiliary's entropy never falls below its adiabat, and Galilean
//! invariant (velocity differences only).  It is the heat an unresolved expansion or
//! compression leaves in the cell's average: a cell expanding into gas at a lower
//! pressure does work only against the lower face pressure.  With the cell's own pressure
//! alone a sheet two cells thick expanding into gas at 1/100 of its pressure, carried at
//! 1000 c, ends 15-20 % colder than the conservative scheme and its converged solution;
//! with the fans' heat 1-4 % colder.
//! Face payload of hydro's and MHD's dual_vf, one value per component and face: the face
//! velocity, the heating rate of the cell on the left of the face and that of the cell
//! on its right (per unit area).  Each component is a flux, so a coarse/fine or
//! LAT reflux replaces or corrects all three as it does the conservative fluxes.

enum DualFaceComponent : int {kDualFaceVelocity = 0, kDualFaceHeatLeft = 1,
                               kDualFaceHeatRight = 2, kDualFacePayload = 3};

//----------------------------------------------------------------------------------------
//! \brief The heat the two-wave (HLLC) model of a face's Riemann fan leaves on each side,
//! from the face states (density d, normal velocity u, pressure p, signal speed c), in
//! velocity differences only (Toro, Riemann Solvers, sect. 10.4-10.5, outer waves of
//! Davis): sigma_l = S_L - u_l, sigma_r = S_R - u_r and the contact S* - u_l = (p_r - p_l
//! - d_r sigma_r (u_r - u_l)) / (d_l sigma_l - d_r sigma_r).  Where a side's star
//! pressure p_s + d_s sigma_s (S* - u_s) is negative the fan is a strong two-sided
//! rarefaction close to vacuum, whose face pressure is zero: that side's gas expands into
//! it without work, and its heat is p_s |S* - u_s|.  Two states at one velocity without
//! signal speed have their common velocity as contact and no heat.

KOKKOS_INLINE_FUNCTION
void DualEnergyFaceContact(const Real dl, const Real ul, const Real pl, const Real cl,
                           const Real dr, const Real ur, const Real pr, const Real cr,
                           Real &q_left, Real &q_right) {
  const Real du = ur - ul;
  const Real sigma_l = fmin(-cl, du - cr);
  const Real sigma_r = fmax(cr, cl - du);
  const Real denom = dl*sigma_l - dr*sigma_r;
  const Real w_l = (denom < 0.0) ? (pr - pl - dr*sigma_r*du)/denom : 0.0;
  const Real w_r = w_l - du;
  // sigma_l <= 0 <= sigma_r: each heat is a product of non-negative factors
  q_left = (pl + dl*sigma_l*w_l >= 0.0) ? dl*(-sigma_l)*SQR(w_l) : pl*w_l;
  q_right = (pr + dr*sigma_r*w_r >= 0.0) ? dr*sigma_r*SQR(w_r) : pr*(-w_r);
}

//----------------------------------------------------------------------------------------
//! \brief The heat of an MHD face: the same two-wave model on the total pressure
//! p + B^2/2 with the fast speeds (the outer waves of HLLD and of HLLC for MHD, Li 2005,
//! J. Comput. Phys. 203, 344).  A tangential discontinuity in total-pressure balance has
//! no contact motion and no heat.  With B = 0 the call is DualEnergyFaceContact on the
//! gas states, bit for bit (the fast speed of a state without field is its sound speed).

KOKKOS_INLINE_FUNCTION
Real DualEnergyFastSpeed(const Real dens, const Real cs2, const Real bn, const Real bt1,
                         const Real bt2) {
  const Real va2 = (SQR(bn) + SQR(bt1) + SQR(bt2))/dens;
  const Real qsq = cs2 + va2;
  const Real cf2 = 0.5*(qsq + sqrt(fmax(SQR(qsq) - 4.0*cs2*SQR(bn)/dens, 0.0)));
  return sqrt(fmax(cf2, 0.0));
}

KOKKOS_INLINE_FUNCTION
void DualEnergyMHDFaceContact(const Real dl, const Real ul, const Real pl,
                              const Real cs2l, const Real bl1, const Real bl2,
                              const Real dr, const Real ur, const Real pr,
                              const Real cs2r, const Real br1, const Real br2,
                              const Real bn, Real &q_left, Real &q_right) {
  DualEnergyFaceContact(dl, ul, pl + 0.5*(SQR(bn) + SQR(bl1) + SQR(bl2)),
                        DualEnergyFastSpeed(dl, cs2l, bn, bl1, bl2),
                        dr, ur, pr + 0.5*(SQR(bn) + SQR(br1) + SQR(br2)),
                        DualEnergyFastSpeed(dr, cs2r, bn, br1, br2), q_left, q_right);
}

//----------------------------------------------------------------------------------------
//! \brief Write the payload of one face.

KOKKOS_INLINE_FUNCTION
void SetDualFacePayload(const BandView5D<Real> &vf, const int m, const int k,
                        const int j, const int i, const Real v_face,
                        const Real q_left, const Real q_right) {
  vf(m, kDualFaceVelocity, k, j, i) = v_face;
  vf(m, kDualFaceHeatLeft, k, j, i) = q_left;
  vf(m, kDualFaceHeatRight, k, j, i) = q_right;
}

//----------------------------------------------------------------------------------------
//! \brief A cell's volume change rate divv = sum_f v_f.n_f A_f / V and heating rate
//! heat = sum_f q_f A_f / V from the payload of its faces: the cell is the left side of
//! its upper face and the right side of its lower one along each axis.

KOKKOS_INLINE_FUNCTION
void DualEnergyCellPdV(const BandView5D<Real> &vf1, const BandView5D<Real> &vf2,
                       const BandView5D<Real> &vf3, const int m, const int k,
                       const int j, const int i, const Real dx1, const Real dx2,
                       const Real dx3, const bool multi_d, const bool three_d,
                       Real &divv, Real &heat) {
  constexpr int fv = kDualFaceVelocity, fl = kDualFaceHeatLeft, fr = kDualFaceHeatRight;
  divv = (vf1(m,fv,k,j,i+1) - vf1(m,fv,k,j,i))/dx1;
  heat = (vf1(m,fl,k,j,i+1) + vf1(m,fr,k,j,i))/dx1;
  if (multi_d) {
    divv += (vf2(m,fv,k,j+1,i) - vf2(m,fv,k,j,i))/dx2;
    heat += (vf2(m,fl,k,j+1,i) + vf2(m,fr,k,j,i))/dx2;
  }
  if (three_d) {
    divv += (vf3(m,fv,k+1,j,i) - vf3(m,fv,k,j,i))/dx3;
    heat += (vf3(m,fl,k+1,j,i) + vf3(m,fr,k,j,i))/dx3;
  }
}

//----------------------------------------------------------------------------------------
//! \brief The auxiliary's p dV of one stage: the adiabat on divv_dt
//! (DualEnergyAuxiliaryCompressed) and the faces' heat heat_dt, between the floors.

KOKKOS_INLINE_FUNCTION
Real DualEnergyAuxiliaryPdV(const EOS_Data &eos, const Real dens, const Real eint,
                            const Real divv_dt, const Real heat_dt,
                            const Real eint_floor, const Real eint_ceil,
                            const Real log_rho_cgs) {
  Real e = DualEnergyAuxiliaryCompressed(eos, dens, eint, divv_dt, eint_floor, eint_ceil,
                                         log_rho_cgs) + heat_dt;
  bool efloor_used = false, tfloor_used = false;
  return ApplyHydroThermalFloorsWithBounds(eos, dens, e, eint_floor, eint_ceil,
                                           efloor_used, tfloor_used);
}

//----------------------------------------------------------------------------------------
//! \brief A LAT reflux of the auxiliary's p dV at one face of a cell, on an auxiliary
//! eint the caller has floored (and floors again).  chi_v = theta sgn dv / dx is the
//! volume change and chi_q = -theta dq / dx the heat the stage updates took in excess,
//! with theta the admissible fraction of the pending reflux and dv, dq the
//! time-integrated mismatch (own minus fine) over the window of the face velocity and
//! of the heat of the cell's side of the face; they are given back as
//! DualEnergyAuxiliaryPdV takes them.

KOKKOS_INLINE_FUNCTION
Real DualEnergyRefluxedPdV(const EOS_Data &eos, const Real dens, const Real eint,
                           const Real chi_v, const Real chi_q) {
  if (eos.is_gamma_law) return eint*exp((eos.gamma - 1.0)*chi_v) + chi_q;
  return eint + eos.PressureFromRhoEint(dens, eint)*chi_v + chi_q;
}

//----------------------------------------------------------------------------------------
//! \brief The thermal state a shocked cell leaves this stage with (the shock rule,
//! eos/general_c2p_hyd.hpp).  eint is the auxiliary this stage compressed, eint_advected
//! the advected one it compressed, eint_cons the stage's energy channel E - E_K (- E_M)
//! and magnitude_sum E + E_K (+ E_M), its rounding scale.  A shocked cell takes the
//! channel at every stage before the step's last; the owner's decision and the inversion
//! then read the shock heat from the auxiliary, whatever the eta1 (and magnetic) test
//! says.  At the last stage it takes the step's channel only if that has not fallen
//! below its auxiliary as the first stage's channel left it: a shock never lowers the
//! entropy, and over the whole step the stages' channel errors cancel to the order of
//! the scheme.  Otherwise the step leaves it on the adiabat it would have followed
//! without the channels of the earlier stages.
//!
//! debt(m, ., kb, jb, ib) holds how far the channel taken at the step's earlier stages
//! moved the auxiliary off its adiabat: the whole offset, the part the first stage's
//! channel left (integrators of three or more stages), and for the two-register rk4 the
//! same two parts in the register u1.  The state of the previous stage enters this
//! stage's update with weight gam0, and with rk4 the second register u1, which
//! accumulates delta times each stage's state, with weight gam1.  The reference of the
//! step-final condition keeps the first stage's channel: an averaging stage's E - E_K
//! carries the kinetic energy that mixing the time levels dissipates, which is not heat
//! (rk3: with it the reference exceeded the step's channel in real shocks); a two-stage
//! step has no other.

KOKKOS_INLINE_FUNCTION
Real DualEnergyShockedStageEint(const EOS_Data &eos, const DvceArray5D<Real> &debt,
                                const int m, const int kb, const int jb, const int ib,
                                const int stage, const bool step_final,
                                const bool two_register, const Real gam0,
                                const Real gam1, const Real delta, const Real dens,
                                Real eint, const Real eint_advected, const Real divv_dt,
                                const Real eint_floor, const Real eint_ceil,
                                const Real log_rho_cgs, const Real eint_cons,
                                const Real magnitude_sum) {
  const bool first_stage = (stage == 1);
  Real eint_adiabat = eint, eint_reference = eint;
  if (!first_stage) {
    const Real debt_prev = debt(m, 0, kb, jb, ib);
    const Real first_prev = (stage == 2) ? debt_prev : debt(m, 1, kb, jb, ib);
    Real carried = gam0*debt_prev;
    Real carried_later = gam0*(debt_prev - first_prev);  // zero at the second stage
    if (two_register) {
      const Real reg = ((stage == 2) ? 0.0 : debt(m, 2, kb, jb, ib)) + delta*debt_prev;
      const Real reg_first = ((stage == 2) ? 0.0 : debt(m, 3, kb, jb, ib)) +
                             delta*first_prev;
      if (!step_final) {
        debt(m, 2, kb, jb, ib) = reg;
        debt(m, 3, kb, jb, ib) = reg_first;
      }
      carried += gam1*reg;
      carried_later += gam1*(reg - reg_first);
    }
    eint_adiabat = DualEnergyAuxiliaryCompressed(eos, dens, eint_advected - carried,
                                                 divv_dt, eint_floor, eint_ceil,
                                                 log_rho_cgs);
    eint_reference = eint - carried_later;
  }
  if (DualEnergyChannelResolved(eint_cons, magnitude_sum) &&
      (!step_final || (eint_cons >= eint_reference))) {
    bool efloor_used = false, tfloor_used = false;
    eint = ApplyHydroThermalFloorsWithBounds(eos, dens, eint_cons, eint_floor, eint_ceil,
                                             efloor_used, tfloor_used);
  } else if (step_final) {
    eint = eint_adiabat;
  }
  if (!step_final) {
    debt(m, 0, kb, jb, ib) = eint - eint_adiabat;
    if (!first_stage) debt(m, 1, kb, jb, ib) = eint_reference - eint_adiabat;
  }
  return eint;
}

}  // namespace eos_general

#endif  // EOS_DUAL_ENERGY_SHOCK_RULE_HPP_
