//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file history.cpp
//  \brief writes history output data, volume-averaged quantities that are output
//         frequently in time to trace their evolution.

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

#include "athena.hpp"
#include "coordinates/cartesian_ks.hpp"
#include "coordinates/cell_locations.hpp"
#include "coordinates/kerr_hole.hpp"
#include "globals.hpp"
#include "mesh/mesh.hpp"
#include "eos/eos.hpp"
#include "hydro/hydro.hpp"
#include "mhd/mhd.hpp"
#include "z4c/z4c.hpp"
#include "coordinates/adm.hpp"
#include "dyn_grmhd/dyn_grmhd.hpp"
#include "outputs.hpp"

namespace {
//----------------------------------------------------------------------------------------
//! \fn void RankZeroMeshTotals()
//! \brief The mesh totals of per-MeshBlock history sums, the same for every distribution
//! of the blocks over the ranks.
//!
//! Floating-point addition is not associative.  A rank-local reduction followed by the
//! rank sum of WriteOutputFile rounds differently for every partition of the same mesh,
//! so a LAT repartition, a load-balance choice or a different cost model (the deferred
//! bin-final exchange changes the exchange count the LAT partition prices) moved the last
//! bits of the history columns of a bitwise identical state (T1(a): mass 5.6e-14,
//! tot-E 1.6e-14).  The block sums (each block's cells in one fixed order, block_sums)
//! are therefore added on rank 0 in the depth-first order of the MeshBlockTree
//! (ZOrderPrecedes), which neither the partition nor a gid renumbering changes.  The
//! table that carries them has one owner per entry, so its rank sum is exact.  Rank 0
//! returns the totals and every other rank zeros, which the rank sum of WriteOutputFile
//! reproduces exactly.

void RankZeroMeshTotals(Mesh *pm, const DvceArray2D<Real> &block_sums, const int nhist,
                        Real *totals) {
  for (int n = 0; n < nhist; ++n) totals[n] = 0.0;
  const int nmb = pm->pmb_pack->nmb_thispack;
  const int nmb_total = pm->nmb_total;
  auto host_sums = Kokkos::create_mirror_view_and_copy(HostMemSpace(), block_sums);
  const LogicalLocation *lloc = pm->lloc_eachmb;
  std::vector<int> zorder(nmb_total);
  std::iota(zorder.begin(), zorder.end(), 0);
  std::sort(zorder.begin(), zorder.end(), [lloc](const int a, const int b) {
    return ZOrderPrecedes(lloc[a], lloc[b]);
  });
  std::vector<int> zpos(nmb_total);
  for (int p = 0; p < nmb_total; ++p) zpos[zorder[p]] = p;
  std::vector<Real> table(static_cast<std::size_t>(nmb_total)*nhist, 0.0);
  auto &mb_gid = pm->pmb_pack->pmb->mb_gid;
  for (int m = 0; m < nmb; ++m) {
    const std::size_t row = static_cast<std::size_t>(zpos[mb_gid.h_view(m)])*nhist;
    for (int n = 0; n < nhist; ++n) table[row + n] = host_sums(m, n);
  }
#if MPI_PARALLEL_ENABLED
  const int count = nmb_total*nhist;
  if (global_variable::my_rank == 0) {
    MPI_Reduce(MPI_IN_PLACE, table.data(), count, MPI_ATHENA_REAL, MPI_SUM, 0,
               MPI_COMM_WORLD);
  } else {
    MPI_Reduce(table.data(), nullptr, count, MPI_ATHENA_REAL, MPI_SUM, 0,
               MPI_COMM_WORLD);
  }
#endif
  if (global_variable::my_rank != 0) return;
  for (int p = 0; p < nmb_total; ++p) {
    for (int n = 0; n < nhist; ++n) {
      totals[n] += table[static_cast<std::size_t>(p)*nhist + n];
    }
  }
}
}  // namespace

//----------------------------------------------------------------------------------------
// Constructor: also calls BaseTypeOutput base class constructor

HistoryOutput::HistoryOutput(ParameterInput *pin, Mesh *pm, OutputParameters op) :
  BaseTypeOutput(pin, pm, op) {
  // cycle through physics modules and add HistoryData struct for each
  hist_data.clear();

  if (pm->pgen->user_hist && op.user_hist_only) {
    hist_data.emplace_back(PhysicsModule::UserDefined);
  } else {
    if (pm->pmb_pack->phydro != nullptr) {
      hist_data.emplace_back(PhysicsModule::HydroDynamics);
    }
    if (pm->pmb_pack->pmhd != nullptr) {
      hist_data.emplace_back(PhysicsModule::MagnetoHydroDynamics);
    }
    if (pm->pgen->user_hist) {
      hist_data.emplace_back(PhysicsModule::UserDefined);
    }
  }

  if (pm->pmb_pack->pz4c != nullptr) {
    hist_data.emplace_back(PhysicsModule::SpaceTimeDynamics);
  }
  if (kerr_hole::Evolves()) {
    hist_data.emplace_back(PhysicsModule::KerrBlackHole);
  }
}

//----------------------------------------------------------------------------------------
//! \fn void HistoryOutput::LoadOutputData()
//  \brief Wrapper function that cycles through hist_data vector and calls
//  appropriate LoadXXXData() function for that physics

void HistoryOutput::LoadOutputData(Mesh *pm) {
  for (auto &data : hist_data) {
    if (data.physics == PhysicsModule::HydroDynamics) {
      LoadHydroHistoryData(&data, pm);
    } else if (data.physics == PhysicsModule::MagnetoHydroDynamics) {
      LoadMHDHistoryData(&data, pm);
    } else if (data.physics == PhysicsModule::SpaceTimeDynamics) {
      LoadZ4cHistoryData(&data, pm);
    } else if (data.physics == PhysicsModule::KerrBlackHole) {
      // The bookkept hole (the metric lags it by at most 1e-5 M): mass, and the
      // dimensionless spin J/M^2.  Global values, so only rank 0 contributes to the
      // rank sum of WriteOutputFile.
      data.nhist = 2;
      data.label[0] = "bh_M";
      data.label[1] = "bh_a";
      Real bh_m = 0.0, bh_a = 0.0;
      kerr_hole::HistoryValues(bh_m, bh_a);
      const bool root = (global_variable::my_rank == 0);
      data.hdata[0] = root ? bh_m : 0.0;
      data.hdata[1] = root ? bh_a : 0.0;
    } else if (data.physics == PhysicsModule::UserDefined) {
      (pm->pgen->user_hist_func)(&data, pm);
    }
  }
}

//----------------------------------------------------------------------------------------
//! \fn void HistoryOutput::LoadHydroHistoryData()
//  \brief Compute and store history data over all MeshBlocks on this rank
//  Data is stored in a Real array defined in derived class.

void HistoryOutput::LoadHydroHistoryData(HistoryData *pdata, Mesh *pm) {
  auto &eos_data = pm->pmb_pack->phydro->peos->eos_data;
  int &nhydro_ = pm->pmb_pack->phydro->nhydro;
  int &nscalars_ = pm->pmb_pack->phydro->nscalars;

  // set number of and names of history variables for hydro
  if (eos_data.is_ideal) {
    pdata->nhist = 8;
  } else {
    pdata->nhist = 7;
  }
  if (nscalars_>0) {
    pdata->nhist += nscalars_;
  }
  pdata->label[IDN] = "mass";
  pdata->label[IM1] = "1-mom";
  pdata->label[IM2] = "2-mom";
  pdata->label[IM3] = "3-mom";
  if (eos_data.is_ideal) {
    pdata->label[IEN] = "tot-E";
  }
  pdata->label[nhydro_  ] = "1-KE";
  pdata->label[nhydro_+1] = "2-KE";
  pdata->label[nhydro_+2] = "3-KE";
  for (int s=0; s<nscalars_; ++s) {
    std::ostringstream labelSS;
    labelSS << "scal-" << s;
    pdata->label[nhydro_+3+s] = labelSS.str();
  }

  // capture class variables for kernel
  auto &u0_ = pm->pmb_pack->phydro->u0;
  auto &size = pm->pmb_pack->pmb->mb_size;
  int &nhist_ = pdata->nhist;

  // loop over all MeshBlocks in this pack
  auto &indcs = pm->pmb_pack->pmesh->mb_indcs;
  int is = indcs.is; int nx1 = indcs.nx1;
  int js = indcs.js; int nx2 = indcs.nx2;
  int ks = indcs.ks; int nx3 = indcs.nx3;
  const int nmb = pm->pmb_pack->nmb_thispack;
  const int nkji = nx3*nx2*nx1;
  const int nji  = nx2*nx1;
  // per-block sums, added in a partition-independent order (RankZeroMeshTotals)
  DvceArray2D<Real> block_sums("hist_block_sums", nmb, NHISTORY_VARIABLES);
  Kokkos::parallel_for("HistSums", Kokkos::TeamPolicy<>(DevExeSpace(), nmb, Kokkos::AUTO),
  KOKKOS_LAMBDA(TeamMember_t tmember) {
    const int m = tmember.league_rank();
    const Real vol = size.d_view(m).dx1*size.d_view(m).dx2*size.d_view(m).dx3;
    array_sum::GlobalSum block_sum;
    Kokkos::parallel_reduce(Kokkos::TeamThreadRange(tmember, nkji),
        [&](const int idx, array_sum::GlobalSum &mb_sum) {
      // compute k,j,i indices of thread
      int k = idx/nji;
      int j = (idx - k*nji)/nx1;
      int i = (idx - k*nji - j*nx1) + is;
      k += ks;
      j += js;

      // Hydro conserved variables:
      array_sum::GlobalSum hvars;
      hvars.the_array[IDN] = vol*u0_(m,IDN,k,j,i);
      hvars.the_array[IM1] = vol*u0_(m,IM1,k,j,i);
      hvars.the_array[IM2] = vol*u0_(m,IM2,k,j,i);
      hvars.the_array[IM3] = vol*u0_(m,IM3,k,j,i);
      if (eos_data.is_ideal) {
        hvars.the_array[IEN] = vol*u0_(m,IEN,k,j,i);
      }

      // Hydro KE
      hvars.the_array[nhydro_  ] = vol*0.5*SQR(u0_(m,IM1,k,j,i))/u0_(m,IDN,k,j,i);
      hvars.the_array[nhydro_+1] = vol*0.5*SQR(u0_(m,IM2,k,j,i))/u0_(m,IDN,k,j,i);
      hvars.the_array[nhydro_+2] = vol*0.5*SQR(u0_(m,IM3,k,j,i))/u0_(m,IDN,k,j,i);

      // Scalar masses
      for (int s=0; s<nscalars_; ++s) {
        hvars.the_array[nhydro_+3+s] = vol*u0_(m,nhydro_+s,k,j,i);
      }

      // fill rest of the_array with zeros, if nhist < NHISTORY_VARIABLES
      for (int n=nhist_; n<NHISTORY_VARIABLES; ++n) {
        hvars.the_array[n] = 0.0;
      }

      // sum into parallel reduce
      mb_sum += hvars;
    }, Kokkos::Sum<array_sum::GlobalSum>(block_sum));
    Kokkos::single(Kokkos::PerTeam(tmember), [&]() {
      for (int n=0; n<NHISTORY_VARIABLES; ++n) block_sums(m,n) = block_sum.the_array[n];
    });
  });

  // store data into hdata array
  RankZeroMeshTotals(pm, block_sums, pdata->nhist, pdata->hdata);

  // What the floors, ceilings and failures of the fixed-metric GR hydro recovery wrote
  // into U since this output's previous row (IdealGRHydro::ConsToPrim books it in
  // Mesh::ecounter at the weight each rewrite reaches the end of its step with): floor-E
  // in the measure of tot-E, floor-M in that of mass, as for the dynamical metric.
  if (pm->pmb_pack->pcoord->is_general_relativistic && pm->pmb_pack->pdyngr == nullptr) {
    const int n = pdata->nhist;
    if (n + 2 > NHISTORY_VARIABLES) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "the hydro history columns and floor-E/floor-M exceed "
                << "NHISTORY_VARIABLES = " << NHISTORY_VARIABLES << std::endl;
      std::exit(EXIT_FAILURE);
    }
    const Real energy = pm->ecounter.eos_floor_energy;
    const Real mass = pm->ecounter.eos_floor_mass;
    pdata->label[n] = "floor-E";
    pdata->label[n+1] = "floor-M";
    pdata->hdata[n] = energy - floor_written_[0];
    pdata->hdata[n+1] = mass - floor_written_[1];
    floor_written_[0] = energy;
    floor_written_[1] = mass;
    pdata->nhist += 2;
  }

  return;
}

//----------------------------------------------------------------------------------------
//! \fn void HistoryOutput::LoadZ4cHistoryData()
//  \brief Compute and store history data over all MeshBlocks on this rank
//  Data is stored in a Real array defined in derived class.

void HistoryOutput::LoadZ4cHistoryData(HistoryData *pdata, Mesh *pm) {
  // set number of and names of history variables for z4c
  pdata->nhist = 9;
  pdata->label[0] = "C-norm2";
  pdata->label[1] = "H-norm2";
  pdata->label[2] = "M-norm2";
  pdata->label[3] = "Z-norm2";
  pdata->label[4] = "Mx-norm2";
  pdata->label[5] = "My-norm2";
  pdata->label[6] = "Mz-norm2";
  pdata->label[7] = "Theta-norm2";
  pdata->label[8] = "Volume";

  // capture class variabels for kernel
  auto &u0_ = pm->pmb_pack->pz4c->u0;
  auto &u_con_ = pm->pmb_pack->pz4c->u_con;
  const int &I_Z4c_Theta_ =  pm->pmb_pack->pz4c->I_Z4C_THETA;
  auto &z4c = pm->pmb_pack->pz4c->z4c;
  auto &adm = pm->pmb_pack->padm->adm;

  auto &size = pm->pmb_pack->pmb->mb_size;
  int &nhist_ = pdata->nhist;
  auto &opt = pm->pmb_pack->pz4c->opt;

  // loop over all MeshBlocks in this pack
  auto &indcs = pm->pmb_pack->pmesh->mb_indcs;
  int is = indcs.is; int nx1 = indcs.nx1;
  int js = indcs.js; int nx2 = indcs.nx2;
  int ks = indcs.ks; int nx3 = indcs.nx3;
  const int nmkji = (pm->pmb_pack->nmb_thispack)*nx3*nx2*nx1;
  const int nkji = nx3*nx2*nx1;
  const int nji  = nx2*nx1;
  array_sum::GlobalSum sum_this_mb;
  Kokkos::parallel_reduce("HistSums",Kokkos::RangePolicy<>(DevExeSpace(), 0, nmkji),
  KOKKOS_LAMBDA(const int &idx, array_sum::GlobalSum &mb_sum) {
    // compute n,k,j,i indices of thread
    int m = (idx)/nkji;
    int k = (idx - m*nkji)/nji;
    int j = (idx - m*nkji - k*nji)/nx1;
    int i = (idx - m*nkji - k*nji - j*nx1) + is;
    k += ks;
    j += js;

    Real detg = adm::SpatialDet(adm.g_dd(m,0,0,k,j,i), adm.g_dd(m,0,1,k,j,i),
                                adm.g_dd(m,0,2,k,j,i), adm.g_dd(m,1,1,k,j,i),
                                adm.g_dd(m,1,2,k,j,i), adm.g_dd(m,2,2,k,j,i));

    Real vol = size.d_view(m).dx1*size.d_view(m).dx2*size.d_view(m).dx3
               * std::sqrt(std::abs(detg));

    // Excise the punctures based on chi
    array_sum::GlobalSum hvars;
    if (z4c.chi(m,k,j,i)>=opt.excise_chi) {
      hvars.the_array[0] = vol*u_con_(m,0,k,j,i); // ||C||^2 (comes already squared)
      hvars.the_array[1] = vol*SQR(u_con_(m,1,k,j,i)); //||H||^2
      hvars.the_array[2] = vol*u_con_(m,2,k,j,i); // ||M||^2 (comes already squared)
      hvars.the_array[3] = vol*u_con_(m,3,k,j,i); // ||Z||^2 (comes already squared)
      hvars.the_array[4] = vol*SQR(u_con_(m,4,k,j,i));      // ||Mx||^2
      hvars.the_array[5] = vol*SQR(u_con_(m,5,k,j,i));      // ||My||^2
      hvars.the_array[6] = vol*SQR(u_con_(m,6,k,j,i));      // ||Mz||^2
      hvars.the_array[7] = vol*SQR(u0_(m,I_Z4c_Theta_,k,j,i)); // ||Theta||^2
      hvars.the_array[8] = vol;
    } else {
      hvars.the_array[0] = 0;
      hvars.the_array[1] = 0;
      hvars.the_array[2] = 0;
      hvars.the_array[3] = 0;
      hvars.the_array[4] = 0;
      hvars.the_array[5] = 0;
      hvars.the_array[6] = 0;
      hvars.the_array[7] = 0;
      hvars.the_array[8] = 0;
    }

    // fill rest of the_array with zeros, if nhist < NHISTORY_VARIABLES
    for (int n=nhist_; n<NHISTORY_VARIABLES; ++n) {
      hvars.the_array[n] = 0.0;
    }

    // sum into parallel reduce
    mb_sum += hvars;
  }, Kokkos::Sum<array_sum::GlobalSum>(sum_this_mb));

  // store data into hdata array
  for (int n=0; n<pdata->nhist; ++n) {
    pdata->hdata[n] = sum_this_mb.the_array[n];
  }

  return;
}

//----------------------------------------------------------------------------------------
//! \fn void HistoryOutput::LoadMHDHistoryData()
//  \brief Compute and store history data over all MeshBlocks on this rank
//  Data is stored in a Real array defined in derived class.

void HistoryOutput::LoadMHDHistoryData(HistoryData *pdata, Mesh *pm) {
  auto &eos_data = pm->pmb_pack->pmhd->peos->eos_data;
  int &nmhd_ = pm->pmb_pack->pmhd->nmhd;
  int &nscalars_ = pm->pmb_pack->pmhd->nscalars;

  // A field-free fluid (GRHD) has no magnetic energy to sum, and its column group is
  // absent rather than zero.
  const bool has_field = pm->pmb_pack->pmhd->HasField();
  const int nme = has_field ? 3 : 0;

  // set number of and names of history variables for mhd
  if (eos_data.is_ideal) {
    pdata->nhist = 8 + nme;
  } else {
    pdata->nhist = 7 + nme;
  }
  if (nscalars_>0) {
    pdata->nhist += nscalars_;
  }
  pdata->label[IDN] = "mass";
  pdata->label[IM1] = "1-mom";
  pdata->label[IM2] = "2-mom";
  pdata->label[IM3] = "3-mom";
  if (eos_data.is_ideal) {
    pdata->label[IEN] = "tot-E";
  }
  pdata->label[nmhd_  ] = "1-KE";
  pdata->label[nmhd_+1] = "2-KE";
  pdata->label[nmhd_+2] = "3-KE";
  if (has_field) {
    pdata->label[nmhd_+3] = "1-ME";
    pdata->label[nmhd_+4] = "2-ME";
    pdata->label[nmhd_+5] = "3-ME";
  }

  for (int s=0; s<nscalars_; ++s) {
    std::ostringstream labelSS;
    labelSS << "scal-" << s;
    pdata->label[nmhd_+3+nme+s] = labelSS.str();
  }

  // capture class variabels for kernel
  auto &u0_ = pm->pmb_pack->pmhd->u0;
  auto &bx1f = pm->pmb_pack->pmhd->b0.x1f;
  auto &bx2f = pm->pmb_pack->pmhd->b0.x2f;
  auto &bx3f = pm->pmb_pack->pmhd->b0.x3f;
  auto &size = pm->pmb_pack->pmb->mb_size;
  int &nhist_ = pdata->nhist;

  // loop over all MeshBlocks in this pack
  auto &indcs = pm->pmb_pack->pmesh->mb_indcs;
  int is = indcs.is; int nx1 = indcs.nx1;
  int js = indcs.js; int nx2 = indcs.nx2;
  int ks = indcs.ks; int nx3 = indcs.nx3;
  const int nmb = pm->pmb_pack->nmb_thispack;
  const int nkji = nx3*nx2*nx1;
  const int nji  = nx2*nx1;
  // per-block sums, added in a partition-independent order (RankZeroMeshTotals)
  DvceArray2D<Real> block_sums("hist_block_sums", nmb, NHISTORY_VARIABLES);
  Kokkos::parallel_for("HistSums", Kokkos::TeamPolicy<>(DevExeSpace(), nmb, Kokkos::AUTO),
  KOKKOS_LAMBDA(TeamMember_t tmember) {
    const int m = tmember.league_rank();
    const Real vol = size.d_view(m).dx1*size.d_view(m).dx2*size.d_view(m).dx3;
    array_sum::GlobalSum block_sum;
    Kokkos::parallel_reduce(Kokkos::TeamThreadRange(tmember, nkji),
        [&](const int idx, array_sum::GlobalSum &mb_sum) {
      // compute k,j,i indices of thread
      int k = idx/nji;
      int j = (idx - k*nji)/nx1;
      int i = (idx - k*nji - j*nx1) + is;
      k += ks;
      j += js;

      // MHD conserved variables:
      array_sum::GlobalSum hvars;
      hvars.the_array[IDN] = vol*u0_(m,IDN,k,j,i);
      hvars.the_array[IM1] = vol*u0_(m,IM1,k,j,i);
      hvars.the_array[IM2] = vol*u0_(m,IM2,k,j,i);
      hvars.the_array[IM3] = vol*u0_(m,IM3,k,j,i);
      if (eos_data.is_ideal) {
        hvars.the_array[IEN] = vol*u0_(m,IEN,k,j,i);
      }

      // MHD KE
      hvars.the_array[nmhd_  ] = vol*0.5*SQR(u0_(m,IM1,k,j,i))/u0_(m,IDN,k,j,i);
      hvars.the_array[nmhd_+1] = vol*0.5*SQR(u0_(m,IM2,k,j,i))/u0_(m,IDN,k,j,i);
      hvars.the_array[nmhd_+2] = vol*0.5*SQR(u0_(m,IM3,k,j,i))/u0_(m,IDN,k,j,i);

      // MHD ME
      if (has_field) {
        hvars.the_array[nmhd_+3] = vol*0.25*(SQR(bx1f(m,k,j,i+1)) + SQR(bx1f(m,k,j,i)));
        hvars.the_array[nmhd_+4] = vol*0.25*(SQR(bx2f(m,k,j+1,i)) + SQR(bx2f(m,k,j,i)));
        hvars.the_array[nmhd_+5] = vol*0.25*(SQR(bx3f(m,k+1,j,i)) + SQR(bx3f(m,k,j,i)));
      }

      // Scalar masses
      for (int s=0; s<nscalars_; ++s) {
        hvars.the_array[nmhd_+3+nme+s] = vol*u0_(m,nmhd_+s,k,j,i);
      }

      // fill rest of the_array with zeros, if nhist < NHISTORY_VARIABLES
      for (int n=nhist_; n<NHISTORY_VARIABLES; ++n) {
        hvars.the_array[n] = 0.0;
      }

      // sum into parallel reduce
      mb_sum += hvars;
    }, Kokkos::Sum<array_sum::GlobalSum>(block_sum));
    Kokkos::single(Kokkos::PerTeam(tmember), [&]() {
      for (int n=0; n<NHISTORY_VARIABLES; ++n) block_sums(m,n) = block_sum.the_array[n];
    });
  });

  // store data into hdata array
  RankZeroMeshTotals(pm, block_sums, pdata->nhist, pdata->hdata);

  // What the floors and ceilings of the dynamical-GR recovery injected into the evolved
  // state since this output's previous row (since the start or restart for its first),
  // at the weight each repair reaches the end of its step with: floor-E in the measure of
  // tot-E (sqrt(gamma) tau dV), floor-M in that of mass (sqrt(gamma) D dV).  Subtracting
  // their sums since a reference row from tot-E and mass removes the floors from the
  // budget.
  if (pm->pmb_pack->pdyngr != nullptr) {
    const int n = pdata->nhist;
    if (n + 2 > NHISTORY_VARIABLES) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "the MHD history columns and floor-E/floor-M exceed "
                << "NHISTORY_VARIABLES = " << NHISTORY_VARIABLES << std::endl;
      std::exit(EXIT_FAILURE);
    }
    Real energy = 0.0, mass = 0.0;
    pdata->label[n] = "floor-E";
    pdata->label[n+1] = "floor-M";
    pdata->hdata[n] = energy - floor_written_[0];
    pdata->hdata[n+1] = mass - floor_written_[1];
    floor_written_[0] = energy;
    floor_written_[1] = mass;
    pdata->nhist += 2;
  }
  // The fixed-metric twin, from Mesh::ecounter as for GR hydro: what an AMR regrid moved
  // when it rebuilt the energy of refined cells from their parent's entropy
  // (MeshRefinement, RefinedThermalStateFromParentEntropy), in the measure of tot-E and
  // of mass, since this output's previous row.
  else if (pm->pmb_pack->pcoord->is_general_relativistic) {
    const int n = pdata->nhist;
    if (n + 2 > NHISTORY_VARIABLES) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "the MHD history columns and floor-E/floor-M exceed "
                << "NHISTORY_VARIABLES = " << NHISTORY_VARIABLES << std::endl;
      std::exit(EXIT_FAILURE);
    }
    // The fixed-metric recovery books on the device (IdealGRMHD floor ledger): bring it
    // into the ecounter before it is read.
    if (pm->pmb_pack->pmhd != nullptr) pm->pmb_pack->pmhd->peos->FlushFloorLedger();
    const Real energy = pm->ecounter.eos_floor_energy;
    const Real mass = pm->ecounter.eos_floor_mass;
    pdata->label[n] = "floor-E";
    pdata->label[n+1] = "floor-M";
    pdata->hdata[n] = energy - floor_written_[0];
    pdata->hdata[n+1] = mass - floor_written_[1];
    floor_written_[0] = energy;
    floor_written_[1] = mass;
    pdata->nhist += 2;
  }

  return;
}

//----------------------------------------------------------------------------------------
//! \fn void HistoryOutput::WriteOutputFile()
//  \brief Cycles through hist_data vector and writes history file for each component

void HistoryOutput::WriteOutputFile(Mesh *pm, ParameterInput *pin) {
  for (auto &data : hist_data) {
    // first, perform in-place sum over all MPI ranks
#if MPI_PARALLEL_ENABLED
    if (global_variable::my_rank == 0) {
      MPI_Reduce(MPI_IN_PLACE, &(data.hdata[0]), data.nhist, MPI_ATHENA_REAL,
         MPI_SUM, 0, MPI_COMM_WORLD);
    } else {
      MPI_Reduce(&(data.hdata[0]), &(data.hdata[0]), data.nhist,
         MPI_ATHENA_REAL, MPI_SUM, 0, MPI_COMM_WORLD);
    }
#endif

    // only the master rank writes the file
    if (global_variable::my_rank == 0) {
      // create filename: "file_basename" + ".physics" + ".hst"
      // There is no file number or id in history output filenames.
      std::string fname;
      fname.assign(out_params.file_basename);
      switch (data.physics) {
        case PhysicsModule::HydroDynamics:
          fname.append(".hydro");
          break;
        case PhysicsModule::MagnetoHydroDynamics:
          fname.append(".mhd");
          break;
        case PhysicsModule::SpaceTimeDynamics:
          fname.append(".z4c");
          break;
        case PhysicsModule::KerrBlackHole:
          fname.append(".bh");
          break;
        case PhysicsModule::UserDefined:
          fname.append(".user");
          break;
        default:
          break;
      }
      fname.append(".hst");

      // open file for output
      FILE *pfile;
      if ((pfile = std::fopen(fname.c_str(),"a")) == nullptr) {
        std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
          << std::endl << "Output file '" << fname << "' could not be opened" <<std::endl;
        exit(EXIT_FAILURE);
      }

      // Write header, if it has not been written already
      if (!(data.header_written)) {
        int iout = 1;
        std::fprintf(pfile,"# Athena++ history data\n");
        std::fprintf(pfile,"#  [%d]=time      ", iout++);
        std::fprintf(pfile,"[%d]=dt       ", iout++);
        for (int n=0; n<data.nhist; ++n) {
          std::fprintf(pfile,"[%d]=%.10s    ", iout++, data.label[n].c_str());
        }
        std::fprintf(pfile,"\n");                              // terminate line
        data.header_written = true;
      }

      // write history variables
      std::fprintf(pfile, out_params.data_format.c_str(), pm->time);
      std::fprintf(pfile, out_params.data_format.c_str(), pm->dt);
      for (int n=0; n<data.nhist; ++n)
        std::fprintf(pfile, out_params.data_format.c_str(), data.hdata[n]);
      std::fprintf(pfile,"\n"); // terminate line
      std::fclose(pfile);
    }
  } // End loop over hist_data vector

  // increment counters, clean up
  AdvanceOutputTime(pm, pin);
  return;
}
