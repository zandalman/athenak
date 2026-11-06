//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file excision.cpp
//! \brief sets boolean masks for horizon excision

#include <float.h>

#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <string>

#include "athena.hpp"
#include "pgen/problem_name.hpp"
#include "globals.hpp"
#include "coordinates/cartesian_ks.hpp"
#include "mesh/mesh.hpp"
#include "coordinates.hpp"
#include "cell_locations.hpp"
#include "coordinates/adm.hpp"
#include "z4c/z4c.hpp"
#include "z4c/fastflow.hpp"

// inlined spherical Kerr-Schild r evaluated at CKS x1, x2, x3
KOKKOS_INLINE_FUNCTION
Real KSRX(const Real x1, const Real x2, const Real x3, const KerrHole &hole) {
  return CartesianKerrSchildRadius(x1, x2, x3, hole);
}

//----------------------------------------------------------------------------------------
//! \fn void Coordinates::SetExcisionMasks()
//  \brief Sets boolean masks for the excision in CKS
//
//  Assigns every cell of every live MeshBlock, ghosts included, so a second call with
//  smaller radii also clears the cells that left the excised region.

void Coordinates::SetExcisionMasks(DvceArray4D<bool> &excision_floor,
                                   DvceArray4D<bool> &excision_flux) {
  // capture variables for kernel
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int is = indcs.is; int js = indcs.js; int ks = indcs.ks;
  int &ng = indcs.ng;
  int n1 = indcs.nx1 + 2*ng;
  int n2 = (indcs.nx2 > 1)? (indcs.nx2 + 2*ng) : 1;
  int n3 = (indcs.nx3 > 1)? (indcs.nx3 + 2*ng) : 1;
  int nmb1 = pmy_pack->nmb_thispack - 1;
  auto &size = pmy_pack->pmb->mb_size;
  const KerrHole hole = coord_data.Hole();
  auto &excision_radius = coord_data.rexcise;

  auto &flux_excise_r = coord_data.flux_excise_r;

  // NOTE(@pdmullen):
  // excision_floor: - if r_ks evaluated at this CC is <= excision_radius, mask the cell.
  // excision_flux:  - if r_ks evaluated at any portion of the two cells connecting
  //                   each face of this cell is <= excision_radius, mask the cell.
  par_for("set_excision", DevExeSpace(), 0, nmb1, 0, (n3-1), 0, (n2-1), 0, (n1-1),
  KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
    // NOTE(@pdmullen): In some instances, calls to x? will access coordinate information
    // for which there is *no corresponding logical counterpart*, however, the
    // LeftEdgeX/CellCenterX functions can handle "out-of-range" queries.
    Real &x1min = size.d_view(m).x1min;
    Real &x1max = size.d_view(m).x1max;
    Real &x2min = size.d_view(m).x2min;
    Real &x2max = size.d_view(m).x2max;
    Real &x3min = size.d_view(m).x3min;
    Real &x3max = size.d_view(m).x3max;

    Real x1v   = CellCenterX(i  -is, indcs.nx1, x1min, x1max);
    Real x1vm1 = CellCenterX(i-1-is, indcs.nx1, x1min, x1max);
    Real x1vp1 = CellCenterX(i+1-is, indcs.nx1, x1min, x1max);
    Real x1f   = LeftEdgeX  (i  -is, indcs.nx1, x1min, x1max);
    Real x1fm1 = LeftEdgeX  (i-1-is, indcs.nx1, x1min, x1max);
    Real x1fp1 = LeftEdgeX  (i+1-is, indcs.nx1, x1min, x1max);
    Real x1fp2 = LeftEdgeX  (i+2-is, indcs.nx1, x1min, x1max);

    Real x2v   = CellCenterX(j  -js, indcs.nx2, x2min, x2max);
    Real x2vm1 = CellCenterX(j-1-js, indcs.nx2, x2min, x2max);
    Real x2vp1 = CellCenterX(j+1-js, indcs.nx2, x2min, x2max);
    Real x2f   = LeftEdgeX  (j  -js, indcs.nx2, x2min, x2max);
    Real x2fm1 = LeftEdgeX  (j-1-js, indcs.nx2, x2min, x2max);
    Real x2fp1 = LeftEdgeX  (j+1-js, indcs.nx2, x2min, x2max);
    Real x2fp2 = LeftEdgeX  (j+2-js, indcs.nx2, x2min, x2max);

    Real x3v   = CellCenterX(k  -ks, indcs.nx3, x3min, x3max);
    Real x3vm1 = CellCenterX(k-1-ks, indcs.nx3, x3min, x3max);
    Real x3vp1 = CellCenterX(k+1-ks, indcs.nx3, x3min, x3max);
    Real x3f   = LeftEdgeX  (k  -ks, indcs.nx3, x3min, x3max);
    Real x3fm1 = LeftEdgeX  (k-1-ks, indcs.nx3, x3min, x3max);
    Real x3fp1 = LeftEdgeX  (k+1-ks, indcs.nx3, x3min, x3max);
    Real x3fp2 = LeftEdgeX  (k+2-ks, indcs.nx3, x3min, x3max);

    // Set excision floor mask
    excision_floor(m,k,j,i) = (KSRX(x1v, x2v, x3v, hole) <= excision_radius);

    // Set excision flux mask
    bool flux = false;
    Real x1, x2, x3;

    // Check face at i
    x1 = x1v;
    x1 = (fabs(x1) < fabs(x1vm1)) ? x1 : x1vm1;
    x1 = (fabs(x1) < fabs(x1f))   ? x1 : x1f;
    x1 = (fabs(x1) < fabs(x1fm1)) ? x1 : x1fm1;
    x1 = (fabs(x1) < fabs(x1fp1)) ? x1 : x1fp1;
    x2 = x2v;
    x2 = (fabs(x2) < fabs(x2f))   ? x2 : x2f;
    x2 = (fabs(x2) < fabs(x2fp1)) ? x2 : x2fp1;
    x3 = x3v;
    x3 = (fabs(x3) < fabs(x3f))   ? x3 : x3f;
    x3 = (fabs(x3) < fabs(x3fp1)) ? x3 : x3fp1;
    flux = flux || (KSRX(x1, x2, x3, hole) <= flux_excise_r);

    // check face at i+1
    x1 = x1vp1;
    x1 = (fabs(x1) < fabs(x1v))   ? x1 : x1v;
    x1 = (fabs(x1) < fabs(x1fp1)) ? x1 : x1fp1;
    x1 = (fabs(x1) < fabs(x1f))   ? x1 : x1f;
    x1 = (fabs(x1) < fabs(x1fp2)) ? x1 : x1fp2;
    flux = flux || (KSRX(x1, x2, x3, hole) <= flux_excise_r);

    // Check face at j
    x1 = x1v;
    x1 = (fabs(x1) < fabs(x1f))   ? x1 : x1f;
    x1 = (fabs(x1) < fabs(x1fp1)) ? x1 : x1fp1;
    x2 = x2v;
    x2 = (fabs(x2) < fabs(x2vm1)) ? x2 : x2vm1;
    x2 = (fabs(x2) < fabs(x2f))   ? x2 : x2f;
    x2 = (fabs(x2) < fabs(x2fm1)) ? x2 : x2fm1;
    x2 = (fabs(x2) < fabs(x2fp1)) ? x2 : x2fp1;
    x3 = x3v;
    x3 = (fabs(x3) < fabs(x3f))   ? x3 : x3f;
    x3 = (fabs(x3) < fabs(x3fp1)) ? x3 : x3fp1;
    flux = flux || (KSRX(x1, x2, x3, hole) <= flux_excise_r);

    // Check face at j+1
    x2 = x2vp1;
    x2 = (fabs(x2) < fabs(x2v))   ? x2 : x2v;
    x2 = (fabs(x2) < fabs(x2fp1)) ? x2 : x2fp1;
    x2 = (fabs(x2) < fabs(x2f))   ? x2 : x2f;
    x2 = (fabs(x2) < fabs(x2fp2)) ? x2 : x2fp2;
    flux = flux || (KSRX(x1, x2, x3, hole) <= flux_excise_r);

    // Check face at k
    x1 = x1v;
    x1 = (fabs(x1) < fabs(x1f))   ? x1 : x1f;
    x1 = (fabs(x1) < fabs(x1fp1)) ? x1 : x1fp1;
    x2 = x2v;
    x2 = (fabs(x2) < fabs(x2f))   ? x2 : x2f;
    x2 = (fabs(x2) < fabs(x2fp1)) ? x2 : x2fp1;
    x3 = x3v;
    x3 = (fabs(x3) < fabs(x3vm1)) ? x3 : x3vm1;
    x3 = (fabs(x3) < fabs(x3f))   ? x3 : x3f;
    x3 = (fabs(x3) < fabs(x3fm1)) ? x3 : x3fm1;
    x3 = (fabs(x3) < fabs(x3fp1)) ? x3 : x3fp1;
    flux = flux || (KSRX(x1, x2, x3, hole) <= flux_excise_r);

    // Check face at k+1
    x3 = x3vp1;
    x3 = (fabs(x3) < fabs(x3v))   ? x3 : x3v;
    x3 = (fabs(x3) < fabs(x3fp1)) ? x3 : x3fp1;
    x3 = (fabs(x3) < fabs(x3f))   ? x3 : x3f;
    x3 = (fabs(x3) < fabs(x3fp2)) ? x3 : x3fp2;
    flux = flux || (KSRX(x1, x2, x3, hole) <= flux_excise_r);
    excision_flux(m,k,j,i) = flux;
  });

  return;
}

//----------------------------------------------------------------------------------------
//! \fn void Coordinates::CheckExcisionResolved()
//! \brief FATAL when the excision sphere reaches into the mesh but no cell of any rank is
//! excised.
//!
//! A fixed-metric run evolves whatever the mask leaves unexcised as fluid.  If the root
//! grid is too coarse for the sphere (or a regrid coarsened it away) no cell centre lies
//! inside r_KS <= rexcise, the mask is empty, and the interior of the hole is advanced as
//! gas.  Collective: the count and the finest cell width are reduced over all ranks, so
//! it is called from the constructor, which runs on every rank at startup and after
//! every regrid, and never from the per-move call of the evolving hole.
//!
//! The sphere r_KS <= R is the oblate spheroid (x^2+y^2)/(R^2+a^2) + z^2/R^2 <= 1, so it
//! meets the mesh box exactly when the point of the box nearest the hole centre, taken
//! per axis, lies inside it.

void Coordinates::CheckExcisionResolved() {
  // The analytic binary writes its own masks from the two horizons' trajectories; the
  // mask built here is only the placeholder around the origin, where that problem has no
  // hole.
  if (std::string(ProblemGeneratorName()) == "BBH") return;
  auto *pm = pmy_pack->pmesh;
  const Real R = coord_data.rexcise;
  const Real a = coord_data.Hole().a;
  auto nearest = [](Real lo, Real hi) {
    return std::min(std::max(static_cast<Real>(0.0), lo), hi);
  };
  const Real cx = nearest(pm->mesh_size.x1min, pm->mesh_size.x1max);
  const Real cy = nearest(pm->mesh_size.x2min, pm->mesh_size.x2max);
  const Real cz = nearest(pm->mesh_size.x3min, pm->mesh_size.x3max);
  const bool reaches_mesh =
      (SQR(cx) + SQR(cy))/(SQR(R) + SQR(a)) + SQR(cz)/SQR(R) <= 1.0;
  if (!reaches_mesh) return;

  auto &indcs = pm->mb_indcs;
  const int is = indcs.is, js = indcs.js, ks = indcs.ks;
  const int nx1 = indcs.nx1, nx2 = indcs.nx2, nx3 = indcs.nx3;
  const int nmb = pmy_pack->nmb_thispack;
  const long long ncell = static_cast<long long>(nmb)*nx3*nx2*nx1;
  auto mask = excision_floor;
  long long nexcised = 0;
  Kokkos::parallel_reduce("excision_count",
                          Kokkos::RangePolicy<>(DevExeSpace(), 0, ncell),
  KOKKOS_LAMBDA(const long long idx, long long &count) {
    const int i = static_cast<int>(idx % nx1) + is;
    const int j = static_cast<int>((idx / nx1) % nx2) + js;
    const int k = static_cast<int>((idx / (static_cast<long long>(nx1)*nx2)) % nx3) + ks;
    const int m = static_cast<int>(idx / (static_cast<long long>(nx1)*nx2*nx3));
    if (mask(m,k,j,i)) ++count;
  }, nexcised);

  Real dx_finest = std::numeric_limits<Real>::max();
  auto &size = pmy_pack->pmb->mb_size;
  for (int m = 0; m < nmb; ++m) {
    dx_finest = std::min(dx_finest, size.h_view(m).dx1);
    if (nx2 > 1) dx_finest = std::min(dx_finest, size.h_view(m).dx2);
    if (nx3 > 1) dx_finest = std::min(dx_finest, size.h_view(m).dx3);
  }
#if MPI_PARALLEL_ENABLED
  MPI_Allreduce(MPI_IN_PLACE, &nexcised, 1, MPI_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &dx_finest, 1, MPI_ATHENA_REAL, MPI_MIN, MPI_COMM_WORLD);
#endif
  if (nexcised == 0) {
    if (global_variable::my_rank == 0) {
      std::cout << "### FATAL ERROR in Coordinates::CheckExcisionResolved" << std::endl
                << "The excision sphere (r_KS <= rexcise = " << R << ") reaches into the "
                << "mesh but no cell centre of any MeshBlock lies inside it, so the "
                << "interior of the hole would be evolved as fluid.  The finest cell "
                << "width at the hole is dx = " << dx_finest << ".  Refine the mesh at "
                << "the hole (a finer root grid or a static refined region) so that "
                << "at least one cell centre lies within rexcise." << std::endl;
    }
    std::exit(EXIT_FAILURE);
  }
}

//----------------------------------------------------------------------------------------
//! \fn void Coordinates::UpdateExcisionMasks()
//! \brief Re-derive the excision masks from the current geometry.
//!
//! Deliberately a NO-OP for ExcisionScheme::fixed, which is what a prescribed-metric BBH
//! run uses: there the pgen callback writes excision_floor/excision_flux in the same
//! kernel that writes the metric (and on the same LAT active mask), so the masks are
//! already current when this returns and a second full-grid pass would only duplicate it.
//! The call sites that pair SetADMVariablesAtTime(t) with this function are therefore
//! correct but inert under `fixed`; do not read them as the thing that moves the masks.
//! The other schemes run on a dynamical metric only.
void Coordinates::UpdateExcisionMasks() {
  // The horizon scheme reads the Z4c horizon finders.  Under LAT pz4c is guaranteed null
  // (Driver refuses evolved Z4c with LAT), so this used to be a null dereference inside a
  // host loop rather than a diagnosed refusal.  Only `fixed` and `lapse` are reachable
  // without Z4c.
  if (coord_data.excision_scheme == ExcisionScheme::horizon &&
      pmy_pack->pz4c == nullptr) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "coord/excision_scheme = horizon needs the Z4c horizon finders, but Z4c "
              << "is not evolved in this configuration (e.g. a prescribed metric, or "
              << "local adaptive timestepping, which refuses evolved Z4c)." << std::endl;
    std::exit(EXIT_FAILURE);
  }

  if (coord_data.excision_scheme == ExcisionScheme::lapse) {
    // capture variables for kernel
    auto &indcs = pmy_pack->pmesh->mb_indcs;
    int &ng = indcs.ng;
    int n1 = indcs.nx1 + 2*ng;
    int n2 = (indcs.nx2 > 1)? (indcs.nx2 + 2*ng) : 1;
    int n3 = (indcs.nx3 > 1)? (indcs.nx3 + 2*ng) : 1;
    const int nwork1 = pmy_pack->nmb_thispack - 1;
    if (nwork1 < 0) return;
    const auto metric = pmy_pack->padm->GetMetricView();
    auto &floor = excision_floor;
    auto &flux = excision_flux;

    Real &excise_lapse = coord_data.excise_lapse;

    par_for("set_excision", DevExeSpace(), 0, nwork1, 0, (n3-1), 0, (n2-1), 0, (n1-1),
    KOKKOS_LAMBDA(const int a, const int k, const int j, const int i) {
      const int m = a;
      adm::ADMMetricPoint metric_point{};
      metric.CellMetric(m, k, j, i, metric_point);
      bool excise = (metric_point.alpha < excise_lapse);
      floor(m,k,j,i) = excise;
      flux(m,k,j,i) = excise;
    });
  }

  if (coord_data.excision_scheme == ExcisionScheme::horizon) {
    // capture variables for kernel
    auto &indcs = pmy_pack->pmesh->mb_indcs;
    auto &size = pmy_pack->pmb->mb_size;
    int &ng = indcs.ng;
    int is = indcs.is; int js = indcs.js; int ks = indcs.ks;
    int n1 = indcs.nx1 + 2 * ng;
    int n2 = (indcs.nx2 > 1) ? (indcs.nx2 + 2 * ng) : 1;
    int n3 = (indcs.nx3 > 1) ? (indcs.nx3 + 2 * ng) : 1;
    int nmb1 = pmy_pack->nmb_thispack - 1;
    auto &floor = excision_floor;
    auto &flux = excision_flux;

    // set up arrays to hold horizon information
    Real &horizon_factor = coord_data.horizon_factor;
    int hsize = pmy_pack->pz4c->pfastflow.size();
    DualArray2D<Real> hcenter("hcenter", hsize, 3);
    DualArray2D<Real> hradius("hradius", hsize, 1);
    DualArray2D<bool> hfound("hfound", hsize, 1);

    // fill horizon arrays on host
    for (int h = 0; h < hsize; ++h) {
      hcenter.h_view(h,0) = pmy_pack->pz4c->pfastflow[h]->center[0]; // center x-coord.
      hcenter.h_view(h,1) = pmy_pack->pz4c->pfastflow[h]->center[1]; // center y-coord.
      hcenter.h_view(h,2) = pmy_pack->pz4c->pfastflow[h]->center[2]; // center z-coord.
      hradius.h_view(h,0) = pmy_pack->pz4c->pfastflow[h]->rr_min; // minimum radius
      hfound.h_view(h,0) = pmy_pack->pz4c->pfastflow[h]->ah_found; // found/not found
    }

    // sync to device
    hcenter.template modify<HostMemSpace>();
    hcenter.template sync<DevExeSpace>();
    hradius.template modify<HostMemSpace>();
    hradius.template sync<DevExeSpace>();
    hfound.template modify<HostMemSpace>();
    hfound.template sync<DevExeSpace>();

    par_for("set_excision_horizon", DevExeSpace(),0,nmb1,0,(n3-1),0,(n2-1),0,(n1-1),
    KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
      // Set MB specifics
      Real &x1min = size.d_view(m).x1min;
      Real &x1max = size.d_view(m).x1max;
      Real &x2min = size.d_view(m).x2min;
      Real &x2max = size.d_view(m).x2max;
      Real &x3min = size.d_view(m).x3min;
      Real &x3max = size.d_view(m).x3max;

      Real x1 = CellCenterX(i - is, indcs.nx1, x1min, x1max);
      Real x2 = CellCenterX(j - js, indcs.nx2, x2min, x2max);
      Real x3 = CellCenterX(k - ks, indcs.nx3, x3min, x3max);

      bool excise = false;
      for (int h = 0; h < hsize; ++h) {
        if (!hfound.d_view(h,0)) continue;

        const Real r2 = SQR(x1 - hcenter.d_view(h,0)) +
                        SQR(x2 - hcenter.d_view(h,1)) +
                        SQR(x3 - hcenter.d_view(h,2));

        if (r2 < SQR(hradius.d_view(h,0) * horizon_factor)) {
          excise = true;
          break;
        }
      }

      floor(m,k,j,i) = excise;
      flux(m,k,j,i) = excise;
    });
  }
}
