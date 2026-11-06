#ifndef EOS_LTE_TABLE_CORE_HPP_
#define EOS_LTE_TABLE_CORE_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file lte_table_core.hpp
//! \brief The H+He LTE table as a device-usable view, with its code units and the exact
//! inverses of its interpolant.  Used by the dynamical-GR tabulated EOS policy, by the
//! GR M1 exchange and by host-side initial-data builders.
//!
//! The table stores ln p, ln eps, ln cs2N, Gamma3 - 1 and the adiabat label L on a
//! uniform (ln rho, ln T) grid.  Every axis and field is shifted into CODE units once, when the table is
//! packed, so that no evaluation converts units.  The forward interpolant is bilinear in
//! (ln rho, ln T).  T(rho, eps) and T(rho, p) are its EXACT inverses along fixed rho:
//! the row blended with the rho weights is piecewise linear in ln T with the T nodes as
//! breakpoints, so a bisection over the nodes and one linear solve in the bracketing
//! interval invert it to round-off.  E(n, T(n, e)) = e holds without iteration, which
//! the conserved-to-primitive root requires.
//!
//! Outside the table:
//!  - ln T above the top: ln p, ln eps and ln cs2N continue linearly in ln T with the
//!    slope kTopSlope = 4 from the last node.  At the top of the table (1e10 K) the gas
//!    is pairs and photons, for which e = 3p and every energy density scales as T^4 at
//!    fixed rho.  The slopes of the last interval itself (about 4.14 for p, 4.06 for eps)
//!    differ from 4 only because the pairs are still freezing out; carrying them upward
//!    changes p/(rho eps) steadily and gives cs^2 > 1 beyond 3e11 K, which no
//!    continuation of this plasma may do.  One common exponent keeps p/(rho eps),
//!    Gamma1 and so cs^2 constant above the top.
//!  - ln T below the bottom: the fields are held at the first node (T is held at T_min).
//!  - rho below the first row: the specific quantities eps, p/rho, cs2N, Gamma3 are
//!    held at the first row, i.e. the dilute ideal-gas limit (ln p continues with unit
//!    slope in ln rho).
//!  - rho above the last row: the log fields continue linearly in ln rho with the slope
//!    of the last interval (a power law).
//!  - Gamma3 - 1, a bounded field, is never extrapolated.
//!
//! The adiabat label (kAdLabel, the GR entropy fallback's transported entropy on this
//! EOS): L(rho, T) = ln eps at the last row rho_top on the adiabat of the interpolant
//! through (rho, eps(rho, T)), i.e. the solution of d ln eps/d ln rho = p/(rho eps)
//! (the first law at zero heat, d eps = -p d(1/rho)).  It is integrated once, at table
//! load, from every node straight to rho_top (BuildAdiabatLabelAtNode), so no
//! row-to-row interpolation error accumulates, and it is interpolated like the log
//! fields: bilinear, slope kTopSlope above the last T node (above the top every log
//! field moves by kTopSlope ln(T/T_top) at fixed rho, and p/(rho eps) and d ln eps/d ln
//! rho do not depend on T there, so two adiabats keep their ln eps offset all the way to
//! rho_top), held below the first T node (the EOS lower bound: an entropy root that
//! lands there fails), and linear in ln rho above the last row (where the fallback has
//! no label and reports failure: LnTFromAdiabatLabel returns NaN).  Below the first row
//! the table holds eps(T) and p/rho(T), so adiabats obey C(T) - ln rho = const with
//! C(T) = integral of d eps/(p/rho) along the first row, exact for the interpolant and
//! stored as nt node values after the fields (TableView::CBelow); there
//! L(rho, T) = L(rho_0, C^-1(C(T) + ln(rho_0/rho))).
//!
//! Where eps does not depend on T (c_v -> 0) neither does the entropy, and the label is
//! flat in T.  In the cold dense corner of the Chabrier union tables (rho 1e-3 to 1e3
//! g/cc, T 1e2 to 1e5 K) the generator raised ln eps (cgs) to 1 + 1e-12 times that of the
//! colder node wherever eps did not increase, so Delta ln eps is ~3e-11 per T interval
//! there (1e-8 and more elsewhere).  The adiabat through a node of such a run leaves the
//! run at one of its edges within a negligible range of rho (at the colder edge where
//! the pdV work raises eps more slowly than the run's own eps rises with rho, else at the
//! hotter one), so the nodes that leave through one edge share its label to the accuracy
//! of the build.  The label rows are therefore non-decreasing in T, not strictly
//! increasing: the load (lte_table_utils, FinalizeAdiabatLabel) raises a node whose label
//! lies below that of a colder node of its row by no more than their combined accuracy
//! to that label, and refuses the table if one lies further below (the label would then
//! not be monotone in the entropy).  On a run of equal labels the inverse returns the
//! hottest T of the run.

#include <cmath>
#include <limits>
#include <string>
#include <vector>

#include <Kokkos_Core.hpp>

#include "athena.hpp"

namespace units {
class Units;
}

namespace lte_eos {

// Constants of the LTE table generator (the same ones lte_table_utils uses).
constexpr Real kMHydrogenCGS = 1.6735575e-24;   // g
constexpr Real kBoltzmannCGS = 1.380649e-16;    // erg/K
//! d ln(p, eps, cs2N) / d ln T above the top of the table: the T^4 law of an
//! ultra-relativistic pair and photon plasma (e = 3p).
constexpr Real kTopSlope = 4.0;

enum Field : int { kLnP = 0, kLnEps = 1, kLnCs2N = 2, kG3m1 = 3, kAdLabel = 4,
                   kNumFields = 5 };

//! Reals in a packed table: the fields, then the nt node values of C along the first
//! row (see the file comment).
KOKKOS_INLINE_FUNCTION size_t TableSize(const int nr, const int nt) {
  return static_cast<size_t>(kNumFields)*nr*nt + static_cast<size_t>(nt);
}

//! Table view.  All axes and fields are in code units.
struct TableView {
  const Real *f = nullptr;    // f[(iv*nr + ir)*nt + it], device or host memory
  int nr = 0, nt = 0;
  Real lr0 = 0.0, idlr = 0.0; // ln rho_code of node 0, 1/d ln rho
  Real lt0 = 0.0, idlt = 0.0; // ln T_code of node 0, 1/d ln T
  Real lt_min = 0.0;          // = lt0 (T floor of the table)

  KOKKOS_INLINE_FUNCTION Real At(const int iv, const int ir, const int it) const {
    return f[(static_cast<size_t>(iv)*nr + ir)*nt + it];
  }
  //! C at T node it of the first row (C = 0 at the first node).
  KOKKOS_INLINE_FUNCTION Real CBelow(const int it) const {
    return f[static_cast<size_t>(kNumFields)*nr*nt + it];
  }
  KOKKOS_INLINE_FUNCTION size_t Size() const {
    return TableSize(nr, nt);
  }
};

//! Quantities at a point, code units.
struct State {
  Real p, eps, cs2N, g3m1;
};

//! Exact derivatives of the interpolant in the cell the query lands in.  They are
//! piecewise constant in (ln rho, ln T), which is what a Newton Jacobian needs.
struct Partials {
  Real dlneps_dlnT, dlneps_dlnrho, dlnp_dlnT, dlnp_dlnrho;
};

//! The cell a query lands in, with the weights of the bilinear interpolant.  The weight
//! wr1 exceeds 1 above the last rho row, which is the linear continuation of the log
//! fields in ln rho; wr1c is clamped to [0, 1] for the bounded field.  Above the last T
//! node wt1 = 1 and lnt_over is the excess of ln T over the node.
struct Cell {
  int ir, it;
  Real wr0, wr1, wt0, wt1, wr1c;
  Real lnp_below;       // ln rho - lr0 when rho is below the first row, else 0
  Real lnt_over;        // ln T - ln T_top when T is above the last node, else 0
  bool rho_below, t_below, t_above;
};

//! Locate a point.  A NaN coordinate falls in the lower branches (never an
//! out-of-range index), and the result then carries the NaN.
KOKKOS_INLINE_FUNCTION Cell Locate(const TableView &v, const Real ln_rho,
                                   const Real ln_T) {
  Cell c;
  const Real x = (ln_rho - v.lr0)*v.idlr;
  if (!(x >= 0.0)) {
    c.rho_below = true;
    c.ir = 0;
    c.wr1 = 0.0;
    c.lnp_below = ln_rho - v.lr0;
  } else {
    c.rho_below = false;
    c.lnp_below = 0.0;
    if (x >= static_cast<Real>(v.nr - 1)) {
      c.ir = v.nr - 2;
      c.wr1 = x - static_cast<Real>(c.ir);
    } else {
      c.ir = static_cast<int>(x);
      c.wr1 = x - static_cast<Real>(c.ir);
    }
  }
  c.wr0 = 1.0 - c.wr1;
  c.wr1c = (c.wr1 > 1.0) ? 1.0 : c.wr1;

  const Real y = (ln_T - v.lt0)*v.idlt;
  c.t_below = false;
  c.t_above = false;
  c.lnt_over = 0.0;
  if (!(y >= 0.0)) {
    c.t_below = true;
    c.it = 0;
    c.wt1 = 0.0;
  } else if (y >= static_cast<Real>(v.nt - 1)) {
    c.it = v.nt - 2;
    c.wt1 = 1.0;
    if (y > static_cast<Real>(v.nt - 1)) {
      c.t_above = true;
      c.lnt_over = (y - static_cast<Real>(v.nt - 1))/v.idlt;
    }
  } else {
    c.it = static_cast<int>(y);
    c.wt1 = y - static_cast<Real>(c.it);
  }
  c.wt0 = 1.0 - c.wt1;
  return c;
}

//! A log field at a located point (kLnP, kLnEps, kLnCs2N or kAdLabel).
KOKKOS_INLINE_FUNCTION Real LnAt(const TableView &v, const Cell &c, const int iv) {
  const Real f00 = v.At(iv, c.ir, c.it);
  const Real f01 = v.At(iv, c.ir, c.it + 1);
  const Real f10 = v.At(iv, c.ir + 1, c.it);
  const Real f11 = v.At(iv, c.ir + 1, c.it + 1);
  Real r = c.wr0*(c.wt0*f00 + c.wt1*f01) + c.wr1*(c.wt0*f10 + c.wt1*f11);
  if (iv == kLnP) {
    r += c.lnp_below;
  }
  return r + kTopSlope*c.lnt_over;
}

//! Gamma3 - 1 at a located point.
KOKKOS_INLINE_FUNCTION Real G3m1At(const TableView &v, const Cell &c) {
  const Real f00 = v.At(kG3m1, c.ir, c.it);
  const Real f01 = v.At(kG3m1, c.ir, c.it + 1);
  const Real f10 = v.At(kG3m1, c.ir + 1, c.it);
  const Real f11 = v.At(kG3m1, c.ir + 1, c.it + 1);
  return (1.0 - c.wr1c)*(c.wt0*f00 + c.wt1*f01) + c.wr1c*(c.wt0*f10 + c.wt1*f11);
}

//! Slopes of a log field at a located point.
KOKKOS_INLINE_FUNCTION Real DLnDLnT(const TableView &v, const Cell &c, const int iv) {
  if (c.t_below) {
    return 0.0;
  }
  if (c.t_above) {
    return kTopSlope;
  }
  const Real d0 = v.At(iv, c.ir, c.it + 1) - v.At(iv, c.ir, c.it);
  const Real d1 = v.At(iv, c.ir + 1, c.it + 1) - v.At(iv, c.ir + 1, c.it);
  return v.idlt*(c.wr0*d0 + c.wr1*d1);
}

KOKKOS_INLINE_FUNCTION Real DLnDLnRho(const TableView &v, const Cell &c, const int iv) {
  if (c.rho_below) {
    return (iv == kLnP) ? 1.0 : 0.0;
  }
  const Real d0 = v.At(iv, c.ir + 1, c.it) - v.At(iv, c.ir, c.it);
  const Real d1 = v.At(iv, c.ir + 1, c.it + 1) - v.At(iv, c.ir, c.it + 1);
  return v.idlr*(c.wt0*d0 + c.wt1*d1);
}

//! Evaluate the table at (ln rho, ln T), code units.
KOKKOS_INLINE_FUNCTION State Eval(const TableView &v, const Real ln_rho, const Real ln_T,
                                  Partials *part = nullptr) {
  const Cell c = Locate(v, ln_rho, ln_T);
  State s;
  s.p = Kokkos::exp(LnAt(v, c, kLnP));
  s.eps = Kokkos::exp(LnAt(v, c, kLnEps));
  s.cs2N = Kokkos::exp(LnAt(v, c, kLnCs2N));
  s.g3m1 = G3m1At(v, c);
  if (part != nullptr) {
    part->dlneps_dlnT = DLnDLnT(v, c, kLnEps);
    part->dlneps_dlnrho = DLnDLnRho(v, c, kLnEps);
    part->dlnp_dlnT = DLnDLnT(v, c, kLnP);
    part->dlnp_dlnrho = DLnDLnRho(v, c, kLnP);
  }
  return s;
}

namespace detail {

//! ln T at which the rho-blended row of log field iv equals target.  The rho weights are
//! fixed once; the bisection runs over the T nodes and one linear solve follows.  Below
//! the first node the result is lt_min; above the last it is the inverse of the linear
//! continuation.  If the row is not monotone (the p rows in the Chabrier cold-dense
//! corner) the first bracket the bisection meets is returned.
KOKKOS_INLINE_FUNCTION Real LnTFromLnField(const TableView &v, const int iv,
                                           const Real ln_rho, Real target,
                                           Real *dlnT_dlnf) {
  const Cell c = Locate(v, ln_rho, v.lt0);
  if (iv == kLnP) {
    target -= c.lnp_below;
  }
  const int ir = c.ir;
  const Real wr0 = c.wr0, wr1 = c.wr1;
  auto g = [&](const int it) {
    return wr0*v.At(iv, ir, it) + wr1*v.At(iv, ir + 1, it);
  };
  const Real dlt = 1.0/v.idlt;
  const Real glo = g(0);
  if (!(target > glo)) {
    if (dlnT_dlnf != nullptr) {
      *dlnT_dlnf = 0.0;
    }
    return v.lt_min;
  }
  const int itop = v.nt - 1;
  const Real ghi = g(itop);
  if (target >= ghi) {
    if (dlnT_dlnf != nullptr) {
      *dlnT_dlnf = 1.0/kTopSlope;
    }
    return v.lt0 + static_cast<Real>(itop)*dlt + (target - ghi)/kTopSlope;
  }
  int ilo = 0, ihi = itop;
  while (ihi - ilo > 1) {
    const int imid = ilo + (ihi - ilo)/2;
    if (g(imid) <= target) {
      ilo = imid;
    } else {
      ihi = imid;
    }
  }
  const Real a = g(ilo);
  const Real b = g(ihi);
  if (dlnT_dlnf != nullptr) {
    *dlnT_dlnf = dlt/(b - a);
  }
  return v.lt0 + (static_cast<Real>(ilo) + (target - a)/(b - a))*dlt;
}

}  // namespace detail

//! ln T at which eps(rho, T) = exp(ln_eps).  Exact inverse of Eval.  Returns lt_min
//! when ln_eps is at or below the first node of the row.
KOKKOS_INLINE_FUNCTION Real LnTFromLnEps(const TableView &v, const Real ln_rho,
                                         const Real ln_eps, Real *dlnT_dlneps = nullptr) {
  return detail::LnTFromLnField(v, kLnEps, ln_rho, ln_eps, dlnT_dlneps);
}

//! ln T at which p(rho, T) = exp(ln_p).  Exact inverse of Eval where the row is monotone.
KOKKOS_INLINE_FUNCTION Real LnTFromLnP(const TableView &v, const Real ln_rho,
                                       const Real ln_p) {
  return detail::LnTFromLnField(v, kLnP, ln_rho, ln_p, nullptr);
}

namespace detail {

//! The slope of C along the first row in T interval j (j = nt - 1: above the last node),
//! dC/d ln T = k exp(z t) with t = ln T - ln T_j: ln eps = a + alpha t and
//! ln(p/rho) = b + beta t there, so d eps/(p/rho) = alpha exp(a - b + (alpha - beta) t).
//! Above the top both slopes are kTopSlope, so z = 0.
struct CSegment {
  Real k, z;
};

KOKKOS_INLINE_FUNCTION CSegment CBelowSegment(const TableView &v, const int j) {
  CSegment g;
  const Real a = v.At(kLnEps, 0, j);
  const Real b = v.At(kLnP, 0, j) - v.lr0;
  if (j >= v.nt - 1) {
    g.k = kTopSlope*Kokkos::exp(a - b);
    g.z = 0.0;
  } else {
    const Real alpha = (v.At(kLnEps, 0, j + 1) - a)*v.idlt;
    const Real beta = (v.At(kLnP, 0, j + 1) - v.At(kLnP, 0, j))*v.idlt;
    g.k = alpha*Kokkos::exp(a - b);
    g.z = alpha - beta;
  }
  return g;
}

//! integral_0^t k exp(z t') dt' = k t expm1(z t)/(z t).
KOKKOS_INLINE_FUNCTION Real CSegmentIntegral(const CSegment &g, const Real t) {
  const Real zt = g.z*t;
  return (zt == 0.0) ? g.k*t : g.k*t*(Kokkos::expm1(zt)/zt);
}

//! C(ln T) along the first row: 0 at and below the first node (the fields are held
//! there, so d eps = 0), exact within each interval, linear above the last node.
KOKKOS_INLINE_FUNCTION Real CBelowAt(const TableView &v, const Real ln_T) {
  const Real y = (ln_T - v.lt0)*v.idlt;
  if (!(y > 0.0)) {
    return 0.0;
  }
  const int itop = v.nt - 1;
  const int j = (y >= static_cast<Real>(itop)) ? itop : static_cast<int>(y);
  const Real t = ln_T - (v.lt0 + static_cast<Real>(j)/v.idlt);
  return v.CBelow(j) + CSegmentIntegral(CBelowSegment(v, j), t);
}

//! ln T at which C(ln T) = c on the first row; lt_min where c <= 0.  Exact inverse of
//! CBelowAt (C increases strictly with T, which the build checks).
KOKKOS_INLINE_FUNCTION Real LnTFromCBelow(const TableView &v, const Real c) {
  if (!(c > 0.0)) {
    return v.lt_min;
  }
  const int itop = v.nt - 1;
  int j;
  if (c >= v.CBelow(itop)) {
    j = itop;
  } else {
    int ilo = 0, ihi = itop;
    while (ihi - ilo > 1) {
      const int imid = ilo + (ihi - ilo)/2;
      if (v.CBelow(imid) <= c) {
        ilo = imid;
      } else {
        ihi = imid;
      }
    }
    j = ilo;
  }
  const CSegment g = CBelowSegment(v, j);
  const Real dc = c - v.CBelow(j);
  const Real x = g.z*dc/g.k;
  const Real t = (x == 0.0) ? dc/g.k : Kokkos::log1p(x)/g.z;
  return v.lt0 + static_cast<Real>(j)/v.idlt + t;
}

}  // namespace detail

//! The adiabat label L(rho, T) (see the file comment), code units.
KOKKOS_INLINE_FUNCTION Real AdiabatLabel(const TableView &v, const Real ln_rho,
                                         const Real ln_T) {
  const Cell c = Locate(v, ln_rho, ln_T);
  if (!c.rho_below) {
    return LnAt(v, c, kAdLabel);
  }
  // Below the first row: carry (rho, T) along its adiabat to the first row, where
  // C(T*) = C(T) + ln(rho_0/rho), and read the label there.
  const Real ln_T_row0 =
      detail::LnTFromCBelow(v, detail::CBelowAt(v, ln_T) + (v.lr0 - ln_rho));
  return LnAt(v, Locate(v, v.lr0, ln_T_row0), kAdLabel);
}

//! ln T at which AdiabatLabel(rho, T) = L at fixed rho: the exact inverse of the
//! interpolant (the label rows are piecewise linear in ln T and non-decreasing).  Where
//! the blended row is flat at L (a run of equal labels, where eps does not resolve T) it
//! returns the hottest T of the run: the bisection keeps the last node whose label is at
//! or below L.  eps there equals eps anywhere on the run to the run's span of eps; p is
//! the table's p at that T.  Returns lt_min where L lies at or below the label of the
//! first T node (the EOS lower bound) and NaN above the last density row, where there is
//! no label.
KOKKOS_INLINE_FUNCTION Real LnTFromAdiabatLabel(const TableView &v, const Real ln_rho,
                                                const Real L) {
  const Real x = (ln_rho - v.lr0)*v.idlr;
  if (!(x <= static_cast<Real>(v.nr - 1))) {
    return std::numeric_limits<Real>::quiet_NaN();
  }
  if (x >= 0.0) {
    return detail::LnTFromLnField(v, kAdLabel, ln_rho, L, nullptr);
  }
  const Real ln_T_row0 = detail::LnTFromLnField(v, kAdLabel, v.lr0, L, nullptr);
  if (!(ln_T_row0 > v.lt_min)) {
    return v.lt_min;
  }
  return detail::LnTFromCBelow(v, detail::CBelowAt(v, ln_T_row0) - (v.lr0 - ln_rho));
}

namespace detail {

//! One smooth piece of the interpolant along an adiabat: rho between rows ir and ir+1
//! and T between nodes jt and jt+1, or above the last T node (jt = nt - 1), where ln eps
//! and ln p continue with slope kTopSlope in ln T.  The adiabat is integrated for
//! u = ln eps as a function of x = ln rho, du/dx = p/(rho eps): bounded (Gamma - 1 of
//! the gas) and so not stiff, also where eps hardly depends on T (degenerate matter)
//! and d ln T/d ln rho along an adiabat is huge.  Within a piece ln eps = c0(x) + c1(x) w
//! and ln p = d0(x) + d1(x) w with w the T weight, all linear in x, so p at (x, u) is
//! exact and smooth; the kinks of the interpolant are the node lines w = 0 and w = 1,
//! u = c0(x) and u = c0(x) + c1(x).  The polynomial is evaluated as it stands slightly
//! outside the piece too, which the Runge-Kutta stages need.
struct AdiabatPiece {
  int jt;
  bool top;
  Real x0, idx;
  Real e00, e01, e10, e11, p00, p01, p10, p11;

  KOKKOS_INLINE_FUNCTION void Set(const TableView &v, const int ir, const int j) {
    jt = j;
    top = (j >= v.nt - 1);
    x0 = v.lr0 + static_cast<Real>(ir)/v.idlr;
    idx = v.idlr;
    const int jl = top ? v.nt - 1 : j;
    e00 = v.At(kLnEps, ir, jl);
    e10 = v.At(kLnEps, ir + 1, jl);
    p00 = v.At(kLnP, ir, jl);
    p10 = v.At(kLnP, ir + 1, jl);
    if (top) {
      e01 = e11 = p01 = p11 = 0.0;
    } else {
      e01 = v.At(kLnEps, ir, j + 1);
      e11 = v.At(kLnEps, ir + 1, j + 1);
      p01 = v.At(kLnP, ir, j + 1);
      p11 = v.At(kLnP, ir + 1, j + 1);
    }
  }

  //! ln eps on the lower node line of the piece, and its slope in x.
  KOKKOS_INLINE_FUNCTION Real Lo(const Real x) const {
    const Real a = (x - x0)*idx;
    return (1.0 - a)*e00 + a*e10;
  }
  KOKKOS_INLINE_FUNCTION Real LoSlope() const { return (e10 - e00)*idx; }
  //! ln eps on the upper node line (none above the last T node).
  KOKKOS_INLINE_FUNCTION Real Hi(const Real x) const {
    const Real a = (x - x0)*idx;
    return top ? std::numeric_limits<Real>::max() : (1.0 - a)*e01 + a*e11;
  }
  KOKKOS_INLINE_FUNCTION Real HiSlope() const { return (e11 - e01)*idx; }

  //! du/dx = p/(rho eps) at (x, u), and its derivative in u (J).  Above the last T node
  //! ln p - ln eps does not depend on T, so J = 0 there.  A piece in which eps does not
  //! increase with T returns NaN.
  KOKKOS_INLINE_FUNCTION Real Slope(const Real x, const Real u, Real *jac = nullptr) const {
    const Real a = (x - x0)*idx;
    const Real c0 = (1.0 - a)*e00 + a*e10;
    const Real d0 = (1.0 - a)*p00 + a*p10;
    if (top) {
      if (jac != nullptr) {
        *jac = 0.0;
      }
      return Kokkos::exp(d0 - c0 - x);
    }
    const Real c1 = (1.0 - a)*(e01 - e00) + a*(e11 - e10);
    if (!(c1 > 0.0)) {
      return std::numeric_limits<Real>::quiet_NaN();
    }
    const Real d1 = (1.0 - a)*(p01 - p00) + a*(p11 - p10);
    const Real w = (u - c0)/c1;
    const Real f = Kokkos::exp(d0 + d1*w - x - u);
    if (jac != nullptr) {
      *jac = f*(d1/c1 - 1.0);
    }
    return f;
  }

  //! |d ln f/dx| along the path through (x, u) whose slope there is f: the rate at which
  //! the polynomial changes under the path.  Within a piece ln f = d0 + d1 w - x - u with
  //! w = (u - c0)/c1, so d ln f/dx = d0' + d1' w + d1 w' - 1 - f and
  //! w' = (f - c0' - c1' w)/c1, the rate at which the path crosses the piece in T.  Where
  //! the path crosses a piece fast and p changes across it (d1 w' large), f changes on a
  //! length much shorter than the row.
  KOKKOS_INLINE_FUNCTION Real SlopeLogRate(const Real x, const Real u, const Real f) const {
    const Real dd0 = (p10 - p00)*idx, dc0 = (e10 - e00)*idx;
    if (top) {
      return Kokkos::fabs(dd0 - dc0 - 1.0);
    }
    const Real a = (x - x0)*idx;
    const Real c0 = (1.0 - a)*e00 + a*e10;
    const Real c1 = (1.0 - a)*(e01 - e00) + a*(e11 - e10);
    const Real d1 = (1.0 - a)*(p01 - p00) + a*(p11 - p10);
    const Real dc1 = ((e11 - e10) - (e01 - e00))*idx;
    const Real dd1 = ((p11 - p10) - (p01 - p00))*idx;
    const Real w = (u - c0)/c1;
    const Real dw = (f - dc0 - dc1*w)/c1;
    return Kokkos::fabs(dd0 + dd1*w + d1*dw - 1.0 - f);
  }

  //! The increment of u over one step of the three-stage Radau IIA method (order 5,
  //! L-stable; E. Hairer and G. Wanner, Solving ODEs II, Sect. IV.5, Table 5.6) from
  //! (x, u) over h, NaN when its Newton iteration does not converge.  Implicit because
  //! the equation is stiff where eps hardly depends on T: there J = f (dlnp/dlneps|_rho
  //! - 1) is huge and neighbouring adiabats collapse onto one within a tiny range of
  //! rho, which an explicit method can follow only with steps of that size.  int_jac
  //! receives the quadrature of J over the step with the method's weights and its stage
  //! Jacobians: the logarithm of the factor by which the step stretches the distance
  //! between neighbouring adiabats.
  KOKKOS_INLINE_FUNCTION Real Step(const Real x, const Real u, const Real h,
                                   Real *int_jac = nullptr) const {
    const Real s6 = Kokkos::sqrt(6.0);
    const Real c[3] = {(4.0 - s6)/10.0, (4.0 + s6)/10.0, 1.0};
    const Real A[3][3] = {
        {(88.0 - 7.0*s6)/360.0, (296.0 - 169.0*s6)/1800.0, (-2.0 + 3.0*s6)/225.0},
        {(296.0 + 169.0*s6)/1800.0, (88.0 + 7.0*s6)/360.0, (-2.0 - 3.0*s6)/225.0},
        {(16.0 - s6)/36.0, (16.0 + s6)/36.0, 1.0/9.0}};
    const Real f0 = Slope(x, u);
    Real K[3] = {f0, f0, f0};
    Real last_change = std::numeric_limits<Real>::max();
    for (int it = 0; it < 30; ++it) {
      Real R[3], J[3];
      for (int i = 0; i < 3; ++i) {
        const Real ui = u + h*(A[i][0]*K[0] + A[i][1]*K[1] + A[i][2]*K[2]);
        R[i] = K[i] - Slope(x + c[i]*h, ui, &J[i]);
      }
      // (I - h diag(J) A) dK = -R, by Gaussian elimination with partial pivoting.
      Real M[3][4];
      for (int i = 0; i < 3; ++i) {
        for (int k = 0; k < 3; ++k) {
          M[i][k] = ((i == k) ? 1.0 : 0.0) - h*J[i]*A[i][k];
        }
        M[i][3] = -R[i];
      }
      for (int col = 0; col < 3; ++col) {
        int piv = col;
        for (int i = col + 1; i < 3; ++i) {
          if (Kokkos::fabs(M[i][col]) > Kokkos::fabs(M[piv][col])) piv = i;
        }
        if (piv != col) {
          for (int k = 0; k < 4; ++k) {
            const Real t = M[col][k];
            M[col][k] = M[piv][k];
            M[piv][k] = t;
          }
        }
        for (int i = col + 1; i < 3; ++i) {
          const Real fct = M[i][col]/M[col][col];
          for (int k = col; k < 4; ++k) M[i][k] -= fct*M[col][k];
        }
      }
      Real dK[3];
      for (int i = 2; i >= 0; --i) {
        Real sum = M[i][3];
        for (int k = i + 1; k < 3; ++k) sum -= M[i][k]*dK[k];
        dK[i] = sum/M[i][i];
      }
      // Converged when the correction is at round-off, or when it no longer contracts:
      // where J is huge, f carries the rounding of u amplified by J, and that noise is
      // the floor of the iteration (the L-stable step damps it).
      bool done = true;
      Real change = 0.0;
      for (int i = 0; i < 3; ++i) {
        K[i] += dK[i];
        const Real rel = Kokkos::fabs(dK[i])/Kokkos::fabs(K[i]);
        change = Kokkos::fmax(change, rel);
        if (!(rel <= 4.0*std::numeric_limits<Real>::epsilon())) {
          done = false;
        }
      }
      if (it >= 2 && change >= 0.5*last_change && change < 1.0e-6) {
        done = true;
      }
      last_change = change;
      if (done) {
        if (int_jac != nullptr) {
          *int_jac = h*(A[2][0]*J[0] + A[2][1]*J[1] + A[2][2]*J[2]);
        }
        return h*(A[2][0]*K[0] + A[2][1]*K[1] + A[2][2]*K[2]);
      }
      if (!(K[0] == K[0] && K[1] == K[1] && K[2] == K[2])) {
        break;
      }
    }
    return std::numeric_limits<Real>::quiet_NaN();
  }
};

//! Status of BuildAdiabatLabelAtNode.
enum AdiabatStatus : int {
  kAdiabatOK = 0,              // converged to kAdiabatTolerance
  kAdiabatLeavesBottom = 1,    // the adiabat left the table below its first T node
  kAdiabatNotMonotone = 2,     // eps does not increase with T somewhere on the path
  kAdiabatStuck = 3,           // switches pieces or halves its step without end
  kAdiabatNotConverged = 4,    // neither criterion met by kAdiabatMaxSubsteps
  kAdiabatConditioned = 5      // converged to the tolerance times its amplification
};
//! The label is accepted when two successive doublings of the substeps each change it by
//! at most this much, relative to max(1, |L|).  It sits a few times above the rounding
//! floor of an integration over the whole table (~1e-13 in ln eps).
constexpr Real kAdiabatTolerance = 1.0e-12;
constexpr int kAdiabatMaxSubsteps = 1024;

//! L at node (r, j) with n substeps per row interval.  The steps end exactly on the
//! rows; a step that leaves its T interval is cut where it reaches the node line
//! (safeguarded Newton on the step length, to two ulps of ln eps), so the method keeps
//! its order across the kinks of the interpolant.  ln eps is accumulated with Neumaier's
//! compensated sum: a path takes up to ~nr n steps, and the plain sum's rounding alone
//! would approach the build tolerance.
//!
//! Every step resolves with n substeps each length on which the polynomial under the
//! path changes: the row (the steps end on its n-th parts), the thickness of the piece
//! (c1/|c1'|, which goes to zero where the piece tapers towards a row on which eps does
//! not resolve T), and the slope along the path (1/|d ln f/dx|, short where the path
//! crosses a thin piece fast while p changes across it).  A feature shorter than the
//! row would otherwise be crossed in one step at every n up to some size, and the
//! doubling test would converge to the error of that step.  No step is shorter than the
//! resolution of x.
//!
//! f is continuous across a node line, so the two pieces of a line disagree about the
//! side a path on it moves to only by the rounding of f: a path that each of them sends
//! into the other is tangent to the line to rounding, and it rides the line, u = line(x),
//! for one step.
//!
//! ln_amp receives ln A, A = max over x of exp(integral of J from x to the last row):
//! the factor by which the neighbouring adiabats spread from the narrowest point of the
//! bundle to the last row, i.e. the amplification of an error made anywhere on the path.
KOKKOS_INLINE_FUNCTION int IntegrateAdiabatToTop(const TableView &v, const int r,
                                                 const int j, const int n, Real &label,
                                                 Real *ln_amp = nullptr) {
  const int itop = v.nt - 1;
  Real x = v.lr0 + static_cast<Real>(r)/v.idlr;
  Real u = v.At(kLnEps, r, j);
  Real uc = 0.0;    // compensation: ln eps = u + uc
  Real ln_s = 0.0, ln_s_min = 0.0;    // ln of the spread since the start, its minimum
  if (ln_amp != nullptr) {
    *ln_amp = 0.0;
  }
  if (r >= v.nr - 1) {
    label = u;
    return kAdiabatOK;
  }
  AdiabatPiece pc;
  // On a node line the path enters the piece above when it rises faster than the line.
  auto enter = [&](const int ir, const int j_line) {
    pc.Set(v, ir, j_line);
    const Real f = pc.Slope(x, u);
    if (f == f && f < pc.LoSlope()) {
      if (j_line == 0) {
        return kAdiabatLeavesBottom;
      }
      pc.Set(v, ir, j_line - 1);
    }
    return (f == f) ? kAdiabatOK : kAdiabatNotMonotone;
  };
  {
    const int st = enter(r, j);
    if (st != kAdiabatOK) {
      return st;
    }
  }
  auto add = [&](const Real du) {
    const Real t = u + du;
    uc += (Kokkos::fabs(u) >= Kokkos::fabs(du)) ? (u - t) + du : (du - t) + u;
    u = t;
  };
  auto spread = [&](const Real ln_factor) {
    ln_s += ln_factor;
    ln_s_min = Kokkos::fmin(ln_s_min, ln_s);
  };
  const Real tol_u = 2.0*std::numeric_limits<Real>::epsilon();
  const Real nsub = static_cast<Real>(n);
  int crossings = 0;
  const int max_crossings = 8*(v.nr + v.nt) + 64;
  Real h_cap = std::numeric_limits<Real>::max();
  // The last switch of pieces on a node line: where, and the piece it left.
  Real x_switch = std::numeric_limits<Real>::quiet_NaN();
  int jt_switch_from = -1;
  for (int ir = r; ir < v.nr - 1; ++ir) {
    if (ir != r) {
      pc.Set(v, ir, pc.top ? itop : pc.jt);
    }
    for (int s = 1; s <= n; ++s) {
      const Real x_end = (s == n) ? v.lr0 + static_cast<Real>(ir + 1)/v.idlr :
          v.lr0 + (static_cast<Real>(ir) + static_cast<Real>(s)/n)/v.idlr;
      while (x < x_end) {
        // A step may reach past its piece by at most about the piece's own width in
        // ln eps: beyond, the extrapolated bilinear is meaningless, and where eps hardly
        // depends on T (the plateaus of the cold dense corner, Delta ln eps ~ 1e-11 per
        // T interval) it overflows.  So the step is cut to twice the predicted distance
        // to the node line the path heads for, and a path on a node line that it leaves
        // switches pieces without a step.
        Real jac0 = 0.0;
        const Real f0 = pc.Slope(x, u, &jac0);
        if (!(f0 == f0)) {
          return kAdiabatNotMonotone;
        }
        const Real g_hi = (u - pc.Hi(x)) + uc;
        const Real g_lo = (u - pc.Lo(x)) + uc;
        const Real rate_hi = f0 - pc.HiSlope();
        const Real rate_lo = pc.LoSlope() - f0;
        bool leave_up = !pc.top && rate_hi > 0.0 && !(g_hi < 0.0);
        bool leave_down = rate_lo > 0.0 && !(g_lo > 0.0);
        // The lengths on which the polynomial under the path changes, n substeps each.
        Real h_res;
        {
          Real scale = std::numeric_limits<Real>::max();
          if (!pc.top) {
            const Real dc1 = Kokkos::fabs(pc.HiSlope() - pc.LoSlope());
            if (dc1 > 0.0) {
              scale = (pc.Hi(x) - pc.Lo(x))/dc1;
            }
          }
          const Real rate = pc.SlopeLogRate(x, u + uc, f0);
          if (rate > 0.0) {
            scale = Kokkos::fmin(scale, 1.0/rate);
          }
          h_res = Kokkos::fmax(scale/nsub, 4.0*std::numeric_limits<Real>::epsilon()*
                                           Kokkos::fmax(1.0, Kokkos::fabs(x)));
        }
        Real h = Kokkos::fmin(Kokkos::fmin(x_end - x, h_cap), h_res);
        if (!leave_up && !leave_down) {
          if (!pc.top && rate_hi > 0.0) {
            h = Kokkos::fmin(h, -2.0*g_hi/rate_hi);
          }
          if (rate_lo > 0.0) {
            h = Kokkos::fmin(h, 2.0*g_lo/rate_lo);
          }
          if (!(x + h > x)) {
            leave_up = !pc.top && rate_hi > 0.0;
            leave_down = !leave_up;
          }
        }
        if (leave_up || leave_down) {
          if (leave_down && pc.jt == 0) {
            return kAdiabatLeavesBottom;
          }
          const int jt_new = leave_up ? pc.jt + 1 : pc.jt - 1;
          if (x == x_switch && jt_new == jt_switch_from) {
            // Sent back to the piece it has just left: the path rides the line.
            const Real x_new = (h_res < x_end - x) ? x + h_res : x_end;
            spread(jac0*(x_new - x));
            u = leave_up ? pc.Hi(x_new) : pc.Lo(x_new);
            uc = 0.0;
            x = x_new;
            x_switch = std::numeric_limits<Real>::quiet_NaN();
            h_cap = std::numeric_limits<Real>::max();
            continue;
          }
          // A path that runs along a node line would switch pieces forever.
          if (++crossings > max_crossings) {
            return kAdiabatStuck;
          }
          x_switch = x;
          jt_switch_from = pc.jt;
          pc.Set(v, ir, jt_new);
          h_cap = std::numeric_limits<Real>::max();
          continue;
        }
        const Real x_try = (h < x_end - x) ? x + h : x_end;
        h = x_try - x;
        Real int_jac = 0.0;
        const Real du = pc.Step(x, u, h, &int_jac);
        if (!(du == du)) {
          // The implicit stages did not converge: a shorter step.
          if (!(0.5*h > 0.0) || ++crossings > max_crossings) {
            return kAdiabatStuck;
          }
          h_cap = 0.5*h;
          continue;
        }
        // Where the step ends relative to the piece, from the compensated ln eps.
        const Real over_hi = (u - pc.Hi(x_try)) + (du + uc);
        const Real under_lo = (u - pc.Lo(x_try)) + (du + uc);
        if (under_lo >= 0.0 && over_hi <= 0.0) {
          add(du);
          spread(int_jac);
          x = x_try;
          h_cap = std::numeric_limits<Real>::max();
          continue;
        }
        const bool up = (over_hi > 0.0);
        const Real g0 = up ? g_hi : g_lo;
        if (up ? !(g0 < 0.0) : !(g0 > 0.0)) {
          // The path set off inward from this very node line and came back to it within
          // the step: halve the step until it ends inside, then cut it where it returns.
          if (!(0.5*h > 0.0) || ++crossings > max_crossings) {
            return kAdiabatStuck;
          }
          h_cap = 0.5*h;
          continue;
        }
        if (!up && pc.jt == 0) {
          return kAdiabatLeavesBottom;
        }
        if (++crossings > max_crossings) {
          return kAdiabatStuck;
        }
        const Real line_slope = up ? pc.HiSlope() : pc.LoSlope();
        // Step length to the node line: g(s) = U(s) - line(x + s) changes sign on (0, h].
        const Real g1 = g0 + du - line_slope*h;
        Real sl = 0.0, sh = h;
        Real sc = h*g0/(g0 - g1);
        Real line = 0.0, int_jac_sc = int_jac;
        for (int it = 0; it < 100; ++it) {
          line = up ? pc.Hi(x + sc) : pc.Lo(x + sc);
          const Real ds = pc.Step(x, u, sc, &int_jac_sc);
          if (!(ds == ds)) {
            sh = sc;
            sc = 0.5*(sl + sh);
            continue;
          }
          const Real g = (u - line) + (ds + uc);
          if (Kokkos::fabs(g) <= tol_u*Kokkos::fmax(1.0, Kokkos::fabs(line))) {
            break;
          }
          if ((g > 0.0) == up) {
            sh = sc;
          } else {
            sl = sc;
          }
          const Real fs = pc.Slope(x + sc, u + ds) - line_slope;
          Real sn = sc - g/fs;
          if (!(sn > sl && sn < sh)) {
            sn = 0.5*(sl + sh);
          }
          if (sn == sc) {
            break;
          }
          sc = sn;
        }
        spread((sc < h) ? int_jac_sc : int_jac);
        if (sc < h) {
          x += sc;
        } else {
          x = x_try;
          line = up ? pc.Hi(x) : pc.Lo(x);
        }
        u = line;
        uc = 0.0;
        pc.Set(v, ir, up ? pc.jt + 1 : pc.jt - 1);
        h_cap = std::numeric_limits<Real>::max();
      }
    }
  }
  label = u + uc;
  if (ln_amp != nullptr) {
    *ln_amp = ln_s - ln_s_min;
  }
  return (label == label) ? kAdiabatOK : kAdiabatNotMonotone;
}

}  // namespace detail

//! The label of node (r, j): BuildAdiabatLabelAtNode doubles the substeps per row from
//! one, and accepts the label when two successive doublings each change it by at most
//! kAdiabatTolerance relative (kAdiabatOK).  One agreeing pair is not enough: two coarse
//! integrations can share a step that is longer than a feature of the path and agree on
//! its error (on the production tables up to 3e-10 relative with the length caps of
//! IntegrateAdiabatToTop, 7e-6 without them).  An integration that fails at one n (a path
//! that the pieces of a node line cannot settle at that step length) does not end the
//! node: the next n is tried, and agreement is counted between successive successful
//! integrations.  If the tolerance is not met by kAdiabatMaxSubsteps, the finest label is
//! accepted when its last change is within the tolerance times the amplification A of
//! its adiabat bundle (kAdiabatConditioned): an error made where the bundle is narrowest
//! is multiplied by A on the way to the last row, so such a label is the exact label of
//! a path perturbed by the tolerance.  Otherwise the status is kAdiabatNotConverged, or
//! the last failure if fewer than two integrations succeeded.  `substeps` is the n of
//! the label, `change` the relative change of its last doubling, ln_amp receives ln A.
KOKKOS_INLINE_FUNCTION int BuildAdiabatLabelAtNode(const TableView &v, const int r,
                                                   const int j, Real &label,
                                                   int &substeps, Real &change,
                                                   Real *ln_amp = nullptr) {
  Real prev = 0.0, ln_a = 0.0;
  int n_ok = 0, agree = 0, last_fail = detail::kAdiabatNotConverged;
  substeps = 0;
  change = 0.0;
  if (ln_amp != nullptr) {
    *ln_amp = 0.0;
  }
  for (int n = 1; n <= detail::kAdiabatMaxSubsteps; n *= 2) {
    Real cur = 0.0, la = 0.0;
    const int status = detail::IntegrateAdiabatToTop(v, r, j, n, cur, &la);
    if (status != detail::kAdiabatOK) {
      last_fail = status;
      continue;
    }
    label = cur;
    substeps = n;
    ln_a = la;
    if (ln_amp != nullptr) {
      *ln_amp = la;
    }
    if (r >= v.nr - 1) {
      return detail::kAdiabatOK;
    }
    if (n_ok > 0) {
      change = Kokkos::fabs(cur - prev)/Kokkos::fmax(1.0, Kokkos::fabs(cur));
      agree = (change <= detail::kAdiabatTolerance) ? agree + 1 : 0;
      if (agree >= 2) {
        return detail::kAdiabatOK;
      }
    }
    prev = cur;
    ++n_ok;
  }
  if (n_ok < 2) {
    return last_fail;
  }
  if (change <= detail::kAdiabatTolerance*Kokkos::exp(ln_a)) {
    return detail::kAdiabatConditioned;
  }
  return detail::kAdiabatNotConverged;
}

//! Scales that turn the cgs table into code units: the GR scales L = G M/c^2, time L/c
//! and density rho_u of <units>, and the temperature unit T_u = c^2 m_H/k_B (the rule of
//! the Newtonian LTE EOS with the velocity unit c).  Specific energies carry c^2 and
//! pressure rho_u c^2; with the baryon mass equal to the code mass unit, n = rho.
struct LTECodeUnits {
  Real rho_u;                                // g cm^-3
  Real c;                                    // cm s^-1
  Real T_u;                                  // K
  Real ln_rho_u, ln_c2, ln_p_u, ln_T_u;

  static LTECodeUnits Make(const Real density_cgs, const Real velocity_cgs) {
    LTECodeUnits u;
    u.rho_u = density_cgs;
    u.c = velocity_cgs;
    u.T_u = velocity_cgs*velocity_cgs*kMHydrogenCGS/kBoltzmannCGS;
    u.ln_rho_u = std::log(density_cgs);
    u.ln_c2 = 2.0*std::log(velocity_cgs);
    u.ln_p_u = u.ln_rho_u + u.ln_c2;
    u.ln_T_u = std::log(u.T_u);
    return u;
  }
  //! From the deck's <units>.  Fatal unless the code velocity unit is c (a GR run).
  static LTECodeUnits FromUnits(const units::Units &punit);
};

//! A packed table in host memory.  The view is rebuilt on request, so the object may be
//! moved freely.
struct HostTable {
  std::vector<Real> data;
  TableView geom;            // everything but the data pointer
  TableView View() const {
    TableView v = geom;
    v.f = data.data();
    return v;
  }
};

//! What the table says about itself, for the startup line and the restart check.
struct LTETableMeta {
  std::string table_type;
  Real x_h = 0.0, y_he = 0.0;
  bool has_radiation = false, has_h2 = false, zpe_subtracted = false;
  int nrho = 0, ntemp = 0;
  Real rho_min_cgs = 0.0, rho_max_cgs = 0.0;   // table range, g cm^-3
  Real t_min_k = 0.0, t_max_k = 0.0;           // table range, K
};

}  // namespace lte_eos

#endif  // EOS_LTE_TABLE_CORE_HPP_
