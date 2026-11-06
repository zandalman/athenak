//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file load_balance.cpp
//! \brief Contains various Mesh and MeshRefinement functions associated with
//! load balancing when MPI is used, both for uniform grids and with SMR/AMR.

#include <iostream>
#include <limits> // numeric_limits<>
#include <map>
#include <algorithm> // max
#include <cmath>
#include <cstdint>
#include <unordered_map>
#include <utility> // make_pair
#include <vector>

#include "athena.hpp"
#include "globals.hpp"
#include "mesh.hpp"
#include "eos/eos.hpp"
#include "hydro/hydro.hpp"
#include "mhd/mhd.hpp"
#include "radiation/radiation.hpp"
#include "gravity/gravity.hpp"
#include "z4c/z4c.hpp"
#include "utils/mpi_buffer_pool.hpp"

#if MPI_PARALLEL_ENABLED
#include <mpi.h>
#endif

namespace {
#if MPI_PARALLEL_ENABLED
// A message is split when a peer's payload exceeds what one MPI count can carry.
constexpr std::size_t kAMRMaxMessageElements = std::size_t{1} << 30;

int AMRMessageCount(int ncc, int nfc, int cntcc, int cntfc) {
  const std::int64_t count = static_cast<std::int64_t>(ncc)*cntcc +
                             static_cast<std::int64_t>(nfc)*cntfc;
  if (count < 0 || count > std::numeric_limits<int>::max()) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl
              << "AMR load-balance message exceeds the MPI integer count limit."
              << std::endl;
    std::exit(EXIT_FAILURE);
  }
  return static_cast<int>(count);
}

void ValidateAMRDescriptorCount(const char *direction, int actual, int expected) {
  if (actual == expected) return;
  std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
            << std::endl << "AMR load-balance " << direction
            << " descriptor count mismatch: built=" << actual
            << ", expected=" << expected << std::endl;
  std::exit(EXIT_FAILURE);
}

// The descriptors of one direction are put in (peer rank, ordering key) order and given
// consecutive offsets, so the payload of each peer is one contiguous range of Reals.  The
// sender and the receiver of a block build the same key (CreateAMR_MPI_Tag of the
// destination), hence both sides order a peer's blocks identically and a message lands in
// the receiver's descriptors without any datatype.  Returns the total payload.
std::size_t OrderAMRDescriptors(DualArray1D<AMRBuffer> &buffers, int nbuffer,
                                const std::vector<int> &peer_ranks,
                                const std::vector<int> &keys,
                                std::vector<AMRRankMessage> &messages,
                                const char *direction) {
  messages.clear();
  if (static_cast<int>(peer_ranks.size()) != nbuffer ||
      static_cast<int>(keys.size()) != nbuffer) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << "AMR load-balance " << direction
              << " peer metadata count mismatch" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  std::vector<int> order(nbuffer);
  for (int n=0; n<nbuffer; ++n) order[n] = n;
  std::sort(order.begin(), order.end(), [&](int a, int b) {
    return (peer_ranks[a] != peer_ranks[b]) ? (peer_ranks[a] < peer_ranks[b])
                                            : (keys[a] < keys[b]);
  });
  std::vector<AMRBuffer> sorted(nbuffer);
  std::size_t total = 0;
  for (int n=0; n<nbuffer; ++n) {
    const int index = order[n];
    if (n > 0 && peer_ranks[order[n - 1]] == peer_ranks[index] &&
        keys[order[n - 1]] == keys[index]) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "AMR rank-packed " << direction
                << " has duplicate peer ordering key " << keys[index]
                << " for rank " << peer_ranks[index] << std::endl;
      std::exit(EXIT_FAILURE);
    }
    sorted[n] = buffers.h_view(index);
    sorted[n].offset = total;
    total += sorted[n].cnt;
    if (messages.empty() || messages.back().rank != peer_ranks[index]) {
      AMRRankMessage msg;
      msg.rank = peer_ranks[index];
      msg.begin = n;
      msg.end = n;
      msg.offset = sorted[n].offset;
      msg.count = 0;
      msg.nrequest = 0;
      msg.pending = 0;
      messages.push_back(msg);
    }
    messages.back().end = n + 1;
    messages.back().count += sorted[n].cnt;
  }
  for (auto &msg : messages) {
    msg.nrequest = static_cast<int>((msg.count + kAMRMaxMessageElements - 1)/
                                    kAMRMaxMessageElements);
  }
  for (int n=0; n<nbuffer; ++n) buffers.h_view(n) = sorted[n];
  return total;
}

// Payload buffer of one transfer event, handed to MPI.
// A payload within the device budget comes from the process-wide pool and returns to it
// at the end of the event.  It is requested with the pool's geometric growth target,
// capped at the budget, so a mesh whose payloads keep setting new maxima adds buffers of
// at least 1.5x the previous size until one of budget size serves every later event: the
// buffers one role leaves in the pool sum to at most 3x the budget, not one buffer per
// new maximum.
// A larger payload lives in pinned host memory, which the pack and unpack kernels
// address directly, so no device copy of it is made.  That buffer is kept from event to
// event (MPI registers it once) and regrown by the same 1.5x target.  It is freed once
// a rank that transfers anything needs less than a quarter of it, so it follows the
// payload within a factor-6 band, and a one-off large transfer does not keep its memory
// pinned for the rest of the run.
Real *AcquireAMRBuffer(std::size_t n, std::size_t device_budget,
                       DvceArray1D<Real> &pooled, AMRPinnedArray1D &pinned) {
  const std::size_t grown = mpi_buffer_pool::GrowTarget(n, 0, 0);
  if (n > 0 && n < pinned.extent(0)/4) pinned = AMRPinnedArray1D();
  if (n <= device_budget) {
    pooled = mpi_buffer_pool::Acquire1D<Real>(n, false, std::min(grown, device_budget));
    return pooled.data();
  }
  if (pinned.extent(0) < n) {
    pinned = AMRPinnedArray1D();
    pinned = AMRPinnedArray1D(
        Kokkos::view_alloc(Kokkos::WithoutInitializing, "lb amr pinned buffer"), grown);
  }
  return pinned.data();
}

// End of the event: the pooled buffer returns to the pool; the pinned one is kept.
void ReleaseAMRBuffer(DvceArray1D<Real> &pooled) {
  mpi_buffer_pool::Release(pooled);
}

#endif
} // namespace

//----------------------------------------------------------------------------------------
int Mesh::ApplyHydroLATLoadBalanceCosts(ParameterInput *pin, LogicalLocation *lloc_list,
                                        float *cost_list, int nb, int *lat_factor_list,
                                        int *lat_sync_factor, bool costs_measured) {
  if (lat_sync_factor != nullptr) *lat_sync_factor = 1;
  if (pin == nullptr || cost_list == nullptr || nb <= 0) return 0;
  const bool hydro_lat = pin->IsLATEnabled();
  auto configured_amr_block_cap = [&]() {
    if (pin->DoesParameterExist("mesh_refinement", "max_nmb_per_rank")) {
      const int cap = pin->GetInteger("mesh_refinement", "max_nmb_per_rank");
      if (cap > 0) return cap;
    }
    return 0;
  };
  if (!hydro_lat) {
    // Restart files may carry stale cost data from a previous LAT partition.
    // With LAT disabled, recover the original block-count balance and leave
    // LoadBalance on the standard Z-order scalar path used by the non-LAT code.
    for (int i=0; i<nb; ++i) {
      cost_list[i] = 1.0;
      if (lat_factor_list != nullptr) lat_factor_list[i] = 1;
    }
    return 0;
  }

  int configured_levels = 1;
  if (pin->DoesParameterExist("time", "lat_levels")) {
    configured_levels = pin->GetInteger("time", "lat_levels");
  }
  configured_levels = std::max(1, std::min(configured_levels, 20));
  int configured_max_factor = 1;
  for (int n=0; n<configured_levels; ++n) configured_max_factor *= 2;

  int max_present_factor = 1;
  const bool have_exact_lat_metadata =
      hydro_lat_metadata_valid && hydro_lat_factor_eachmb != nullptr &&
      hydro_lat_metadata_nmb == nb &&
      (lloc_list == nullptr || lloc_list == lloc_eachmb);
  const bool have_amr_estimated_metadata =
      (!have_exact_lat_metadata) && hydro_lat_metadata_valid &&
      hydro_lat_factor_eachmb != nullptr && hydro_lat_metadata_nmb == nmb_total &&
      lloc_eachmb != nullptr && lloc_list != nullptr && nmb_total > 0;
  std::unordered_map<LogicalLocationKey, int, LogicalLocationKeyHash> factor_by_location;
  std::unordered_map<LogicalLocationKey, int, LogicalLocationKeyHash> min_descendant_factor;
  // The measured per-step work travels with the factor: by gid when the metadata is
  // exact, by logical location for a block that survives unchanged (a pure rebalance is
  // all such blocks), and 1 for a block that is new or re-levelled, since nothing has
  // been measured on it yet.
  std::unordered_map<LogicalLocationKey, Real, LogicalLocationKeyHash> work_by_location;
  std::vector<Real> work_list(nb, 1.0);
  if (have_amr_estimated_metadata) {
    factor_by_location.reserve(static_cast<std::size_t>(hydro_lat_metadata_nmb)*2);
    min_descendant_factor.reserve(static_cast<std::size_t>(hydro_lat_metadata_nmb)*2);
    if (hydro_lat_work_eachmb != nullptr) {
      work_by_location.reserve(static_cast<std::size_t>(hydro_lat_metadata_nmb)*2);
    }
    for (int old_gid=0; old_gid<hydro_lat_metadata_nmb; ++old_gid) {
      const LogicalLocation &old_lloc = lloc_eachmb[old_gid];
      const int old_factor = std::max(1, hydro_lat_factor_eachmb[old_gid]);
      factor_by_location.emplace(LogicalLocationKey(old_lloc), old_factor);
      if (hydro_lat_work_eachmb != nullptr) {
        work_by_location.emplace(LogicalLocationKey(old_lloc),
                                 hydro_lat_work_eachmb[old_gid]);
      }
      LogicalLocationKey ancestor(old_lloc);
      int depth_to_ancestor = 0;
      while (ancestor.level > 0) {
        ancestor = ParentKey(ancestor);
        ++depth_to_ancestor;
        int ancestor_factor = old_factor;
        for (int n=0; n<depth_to_ancestor && ancestor_factor<configured_max_factor; ++n) {
          ancestor_factor = std::min(configured_max_factor, 2*ancestor_factor);
        }
        auto it = min_descendant_factor.find(ancestor);
        if (it == min_descendant_factor.end()) {
          min_descendant_factor.emplace(ancestor, ancestor_factor);
        } else {
          it->second = std::min(it->second, ancestor_factor);
        }
      }
    }
  }
  auto estimate_factor = [&](const LogicalLocation &new_lloc) {
    int factor_estimate = 0;
    auto clamp_factor = [&](int factor) {
      return std::max(1, std::min(configured_max_factor, factor));
    };

    // Direct carry-over for unchanged MeshBlocks.
    const auto direct = factor_by_location.find(LogicalLocationKey(new_lloc));
    if (direct != factor_by_location.end()) {
      factor_estimate = direct->second;
    }

    // New refined children inherit a conservative half-step factor from the nearest
    // old ancestor, matching the AMR 2:1 timestep expectation without scanning all MBs.
    LogicalLocationKey ancestor(new_lloc);
    int depth_from_ancestor = 0;
    while (factor_estimate == 0 && ancestor.level > 0) {
      ancestor = ParentKey(ancestor);
      ++depth_from_ancestor;
      const auto it = factor_by_location.find(ancestor);
      if (it == factor_by_location.end()) continue;
      int factor = it->second;
      for (int n=0; n<depth_from_ancestor && factor>1; ++n) factor /= 2;
      factor_estimate = factor;
    }

    // Newly derefined parents inherit the minimum descendant factor, adjusted by
    // one factor increase per coarsening level.
    if (factor_estimate == 0) {
      const auto descendant = min_descendant_factor.find(LogicalLocationKey(new_lloc));
      if (descendant != min_descendant_factor.end()) {
        factor_estimate = descendant->second;
      }
    }

    return clamp_factor((factor_estimate > 0) ? factor_estimate : 1);
  };
  // Precompute max AMR level for level-based factor estimation fallback
  int max_level_present = root_level;
  if (!have_exact_lat_metadata && !have_amr_estimated_metadata && lloc_list != nullptr) {
    for (int i=0; i<nb; ++i) {
      max_level_present = std::max(max_level_present,
                                   static_cast<int>(lloc_list[i].level));
    }
  }
  for (int i=0; i<nb; ++i) {
    int factor = 1;
    if (have_exact_lat_metadata) {
      factor = std::max(1, hydro_lat_factor_eachmb[i]);
      if (hydro_lat_work_eachmb != nullptr) work_list[i] = hydro_lat_work_eachmb[i];
    } else if (have_amr_estimated_metadata) {
      factor = estimate_factor(lloc_list[i]);
      const auto work = work_by_location.find(LogicalLocationKey(lloc_list[i]));
      if (work != work_by_location.end()) work_list[i] = work->second;
    } else if (lloc_list != nullptr && !costs_measured) {
      // Level-based estimation fallback for a cold start:
      // No CFL metadata available yet. Estimate factors from AMR level alone.
      // Finest level gets factor=1, each coarser level doubles the factor.
      const int level_diff = max_level_present - lloc_list[i].level;
      factor = 1;
      for (int n=0; n<level_diff && factor<configured_max_factor; ++n) {
        factor = std::min(configured_max_factor, factor*2);
      }
    }
    if (lat_factor_list != nullptr) lat_factor_list[i] = factor;
    max_present_factor = std::max(max_present_factor, factor);
  }
  const bool have_level_estimated_metadata =
      (!have_exact_lat_metadata) && (!have_amr_estimated_metadata) &&
      lloc_list != nullptr && max_present_factor > 1;
  const bool have_lat_metadata = have_exact_lat_metadata ||
      have_amr_estimated_metadata || have_level_estimated_metadata;
  if (have_lat_metadata && lat_factor_list != nullptr) {
    // Same rank-dependent default as Mesh::UpdateHydroLATMetadata; see the comment there
    // for why no rank-independent constant can replace it.
    int min_bin_count = 4*global_variable::nranks;
    if (pin->DoesParameterExist("time", "lat_min_bin_count")) {
      min_bin_count = pin->GetInteger("time", "lat_min_bin_count");
      if (min_bin_count < -1) {
        std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                  << std::endl
                  << "time/hydro_lat_min_bin_count must be >= 0, or -1 for the default."
                  << std::endl;
        std::exit(EXIT_FAILURE);
      }
      if (min_bin_count == -1) min_bin_count = 4*global_variable::nranks;
    }
    min_bin_count = std::max(0, min_bin_count);
    if (min_bin_count > 0) {
      bool changed = true;
      while (changed) {
        changed = false;
        for (int factor=max_present_factor; factor>1; factor/=2) {
          int count = 0;
          for (int i=0; i<nb; ++i) {
            if (lat_factor_list[i] == factor) ++count;
          }
          if (count > 0 && count < min_bin_count) {
            const int clamped_factor = std::max(1, factor/2);
            for (int i=0; i<nb; ++i) {
              if (lat_factor_list[i] == factor) lat_factor_list[i] = clamped_factor;
            }
            changed = true;
          }
        }

        max_present_factor = 1;
        for (int i=0; i<nb; ++i) {
          max_present_factor = std::max(max_present_factor, lat_factor_list[i]);
        }
      }
    }
  }
  // A checkpoint carries the writing run's own per-block costs, and on a restart they are
  // the only true cost that exists: this run has measured nothing yet.  Keep them, and
  // do not put the level-based estimate in their place -- that estimate is pure geometry,
  // it reads 2^(max_level - level) off the tree and knows nothing about what actually
  // sets the local timestep, so on a deck whose dt does not scale with dx (the star-BH
  // deck: the softened BH CFL sets one dt for every block, and the live factors are 1
  // everywhere) it invents a ladder up to 2^time/lat_levels and a sync factor to match.
  // The LAT partition (BuildLATPartition) then balances the classes of a tick schedule
  // that does not exist, which is worse than no bin information at all.  The per-block
  // loop above already leaves the ladder flat on this path; the first
  // Mesh::UpdateHydroLATMetadata of the restarted run installs the real one, which is how
  // the writing run reached the partition the checkpoint carries.
  const bool keep_restart_costs =
      costs_measured && !have_exact_lat_metadata && !have_amr_estimated_metadata;
  if (keep_restart_costs) {
    for (int i=0; i<nb; ++i) {
      if (!std::isfinite(cost_list[i]) || !(cost_list[i] > 0.0f)) cost_list[i] = 1.0;
    }
  } else if (have_lat_metadata) {
    // Window cost of a block = steps per window x measured cost of one of its steps
    // (Mesh::hydro_lat_work_eachmb, 1 where nothing was measured).  The LAT partition
    // recovers the per-step part as cost*factor/sync.
    for (int i=0; i<nb; ++i) {
      const int factor = (lat_factor_list != nullptr) ? std::max(1,
          lat_factor_list[i]) : 1;
      cost_list[i] = static_cast<float>(
          static_cast<Real>(max_present_factor/factor)*
          std::max(work_list[i], static_cast<Real>(1.0)));
    }
  } else {
    // Restart files can carry stale per-block costs from a non-LAT or older LAT
    // partition.  Before local-CFL metadata exists, balance by block count so the
    // memory cap is an actual per-rank allocation bound instead of packing many
    // cheap coarse blocks onto one GPU.
    for (int i=0; i<nb; ++i) {
      cost_list[i] = 1.0;
    }
  }
  if (lat_sync_factor != nullptr) *lat_sync_factor = max_present_factor;

  const int nranks = std::max(1, global_variable::nranks);
  const int min_cap = (nb + nranks - 1)/nranks;
  // One cap only: the allocation bound mesh_refinement/max_nmb_per_rank.  A separate,
  // lower LAT partition cap (the former time/hydro_lat_max_nmb_per_rank) could not be
  // honoured once AMR filled the blocks between the two caps and aborted the run.
  int cap = adaptive ? configured_amr_block_cap() : 0;
  if (cap <= 0) cap = min_cap;
  if (cap < min_cap || static_cast<long long>(cap)*static_cast<long long>(nranks) < nb) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl
              << "HD LAT load-balance MeshBlocks/rank cap cannot hold current mesh: cap="
              << cap << ", minimum=" << min_cap
              << ", nranks=" << nranks << ", MeshBlocks=" << nb << std::endl;
    std::exit(EXIT_FAILURE);
  }
  if (cap > (1 << NUM_BITS_LID)) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl
              << "HD LAT load-balance memory cap exceeds MPI tag local-id capacity: cap="
              << cap << ", limit=" << (1 << NUM_BITS_LID) << std::endl;
    std::exit(EXIT_FAILURE);
  }

  return cap;
}

//----------------------------------------------------------------------------------------
void Mesh::ApplyHydroLATGIDMapToArrays(LogicalLocation *lloc_list, float *cost_list,
                                       int *lat_factor_list, int **restart_gid_map,
                                       int nb, const int *newtoold, bool reordered) {
  if (lloc_list == nullptr || cost_list == nullptr || newtoold == nullptr || nb <= 0) {
    return;
  }
  std::vector<LogicalLocation> old_lloc(lloc_list, lloc_list + nb);
  std::vector<float> old_cost(cost_list, cost_list + nb);
  std::vector<int> old_factor;
  if (lat_factor_list != nullptr) {
    old_factor.assign(lat_factor_list, lat_factor_list + nb);
  }
  std::vector<int> old_restart_map;
  if (restart_gid_map != nullptr && *restart_gid_map != nullptr) {
    old_restart_map.assign(*restart_gid_map, *restart_gid_map + nb);
  }

  for (int gid=0; gid<nb; ++gid) {
    const int old_gid = newtoold[gid];
    if (old_gid < 0 || old_gid >= nb) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl
                << "LAT GID reorder map references invalid old gid=" << old_gid
                << " at new gid=" << gid << std::endl;
      std::exit(EXIT_FAILURE);
    }
    lloc_list[gid] = old_lloc[old_gid];
    cost_list[gid] = old_cost[old_gid];
    if (lat_factor_list != nullptr) lat_factor_list[gid] = old_factor[old_gid];
  }

  if (restart_gid_map != nullptr) {
    std::vector<int> composed(nb, 0);
    for (int gid=0; gid<nb; ++gid) {
      const int old_gid = newtoold[gid];
      composed[gid] = old_restart_map.empty() ? old_gid : old_restart_map[old_gid];
    }
    delete [] *restart_gid_map;
    *restart_gid_map = new int[nb];
    for (int gid=0; gid<nb; ++gid) (*restart_gid_map)[gid] = composed[gid];
  }

  for (int gid=0; gid<nb; ++gid) {
    ptree->SetLeafGID(lloc_list[gid], gid);
  }
  hydro_lat_gid_reordered = reordered;
  hydro_lat_lb_topology_version = kInvalidLATVersion;
  hydro_lat_lb_metadata_version = kInvalidLATVersion;
  hydro_lat_lb_work_version = kInvalidLATVersion;
}

//----------------------------------------------------------------------------------------
//! \fn void Mesh::LoadBalance(double *clist, int *rlist, int *slist, int *nlist, int nb)
//! \brief Calculate distribution of MeshBlocks across ranks based on input cost list
//! input: clist = cost of each MB (array of length nmbtotal)
//!        nb = number of MeshBlocks
//! output: rlist = rank to which each MB is assigned (array of length nmbtotal)
//!         slist = starting grid ID (gid) for MB on each rank (array of length nrank)
//!         nlist = number of MBs on each rank (array of length nrank)
//! With multiple ranks in MPI, this function is needed even on a uniform mesh and not
//! just for SMR/AMR, which is why it is part of the Mesh and not MeshRefinement class.

void Mesh::LoadBalance(float *clist, int *rlist, int *slist, int *nlist, int nb,
                       int max_blocks_per_rank) {
  float min_cost = std::numeric_limits<float>::max();
  float max_cost = 0.0, totalcost = 0.0;
  // find min/max and total cost in clist
  for (int i=0; i<nb; i++) {
    totalcost += clist[i];
    min_cost = std::min(min_cost,clist[i]);
    max_cost = std::max(max_cost,clist[i]);
  }

  if (max_blocks_per_rank > 0 && global_variable::nranks > 1) {
    const int nranks = global_variable::nranks;
    const int min_cap = (nb + nranks - 1)/nranks;
    const int block_cap = max_blocks_per_rank;
    if (block_cap < min_cap ||
        static_cast<long long>(block_cap)*static_cast<long long>(nranks) < nb) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl
                << "Load-balance cap is infeasible: cap=" << block_cap
                << ", nranks=" << nranks << ", MeshBlocks=" << nb << std::endl;
      std::exit(EXIT_FAILURE);
    }
    std::vector<double> prefix(nb + 1, 0.0);
    for (int i=0; i<nb; ++i) prefix[i + 1] = prefix[i] + clist[i];

    auto feasible = [&](double limit) {
      int used = 1;
      int nblocks = 0;
      double cost = 0.0;
      for (int i=0; i<nb; ++i) {
        if (clist[i] > limit) return false;
        if (nblocks > 0 && (nblocks + 1 > block_cap || cost + clist[i] > limit)) {
          used++;
          nblocks = 0;
          cost = 0.0;
          if (used > nranks) return false;
        }
        nblocks++;
        cost += clist[i];
      }
      return used <= nranks;
    };

    double lo = std::max(static_cast<double>(max_cost),
                         static_cast<double>(totalcost)/static_cast<double>(nranks));
    double hi = static_cast<double>(totalcost);
    for (int iter=0; iter<64; ++iter) {
      const double mid = 0.5*(lo + hi);
      if (feasible(mid)) {
        hi = mid;
      } else {
        lo = mid;
      }
    }
    const double limit = hi;

    struct Partition {int start, end; double cost;};
    std::vector<Partition> parts;
    int start = 0;
    int nblocks = 0;
    double cost = 0.0;
    for (int i=0; i<nb; ++i) {
      if (nblocks > 0 && (nblocks + 1 > block_cap || cost + clist[i] > limit)) {
        parts.push_back({start, i, cost});
        start = i;
        nblocks = 0;
        cost = 0.0;
      }
      nblocks++;
      cost += clist[i];
    }
    parts.push_back({start, nb, cost});

    while (static_cast<int>(parts.size()) < nranks) {
      int split_idx = -1;
      int split_count = 0;
      for (int p=0; p<static_cast<int>(parts.size()); ++p) {
        const int count = parts[p].end - parts[p].start;
        if (count > split_count) {
          split_count = count;
          split_idx = p;
        }
      }
      if (split_idx < 0 || split_count <= 1) break;
      Partition part = parts[split_idx];
      const int mid = part.start + split_count/2;
      const double left_cost = prefix[mid] - prefix[part.start];
      const double right_cost = prefix[part.end] - prefix[mid];
      parts[split_idx] = {part.start, mid, left_cost};
      parts.insert(parts.begin() + split_idx + 1, {mid, part.end, right_cost});
    }

    if (static_cast<int>(parts.size()) == nranks) {
      for (int rank=0; rank<nranks; ++rank) {
        slist[rank] = parts[rank].start;
        nlist[rank] = parts[rank].end - parts[rank].start;
        for (int i=parts[rank].start; i<parts[rank].end; ++i) rlist[i] = rank;
      }
      return;
    }
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl
              << "Load balancer could not find a partition under hard cap "
              << block_cap << std::endl;
    std::exit(EXIT_FAILURE);
  }

  int j = (global_variable::nranks) - 1;
  float targetcost = totalcost/global_variable::nranks;
  float mycost = 0.0;
  // create rank list from the end: the master MPI rank should have less load
  for (int i=nb-1; i>=0; i--) {
    if (targetcost == 0.0) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "There is at least one process which has no MeshBlock"
                << std::endl << "Decrease the number of processes or use smaller "
                << "MeshBlocks." << std::endl;
      std::exit(EXIT_FAILURE);
    }
    mycost += clist[i];
    rlist[i] = j;
    if (mycost >= targetcost && j>0) {
      j--;
      totalcost -= mycost;
      mycost = 0.0;
      targetcost = totalcost/(j+1);
    }
  }
  slist[0] = 0;
  j = 0;
  for (int i=1; i<nb; i++) { // make the list of nbstart and nblocks
    if (rlist[i] != rlist[i-1]) {
      nlist[j] = i-slist[j];
      slist[++j] = i;
    }
  }
  nlist[j] = nb-slist[j];

#if MPI_PARALLEL_ENABLED
  if (nb % global_variable::nranks != 0
     && !adaptive && max_cost == min_cost && global_variable::my_rank == 0) {
    std::cout << "### WARNING in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "Number of MeshBlocks cannot be divided evenly by number of MPI ranks. "
              << "This will result in poor load balancing." << std::endl;
  }
#endif
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void MeshRefinement::InitRecvAMR()
//! \brief Allocates and initializes receive buffers, and posts non-blocking receives,
//! for communicating MeshBlocks during load balancing. Equivalent to some of the work
//! done inside MPI_PARALLEL block in the Mesh::RedistributeAndRefineMeshBlocks()
//! function in amr_loadbalance.cpp in Athena++

//----------------------------------------------------------------------------------------
//! \fn int MeshRefinement::FullBlockTransferPrimitiveCount()
//! \brief Number of extra cell-centered variables a pure rebalance appends to every
//! load-balance message: the fluid primitives (and the cell-centered field), which
//! travel with the conserved state instead of being re-derived after the move, the MHD
//! inversion's warm-start history (EquationOfState::CarriedC2PHistory), which are read
//! before the next step rewrites them.

int MeshRefinement::FullBlockTransferPrimitiveCount() const {
  if (!lb_full_block_transfer) return 0;
  int n = 0;
  if (pmy_mesh->pmb_pack->phydro != nullptr) {
    n += pmy_mesh->pmb_pack->phydro->w0.extent_int(1);
  }
  if (pmy_mesh->pmb_pack->pmhd != nullptr) {
    n += pmy_mesh->pmb_pack->pmhd->w0.extent_int(1);
    // A field-free fluid (GRHD) has no cell-centered field to move.
    if (pmy_mesh->pmb_pack->pmhd->HasField()) {
      n += pmy_mesh->pmb_pack->pmhd->bcc0.extent_int(1);
    }
    DvceArray5D<Real> *hist = pmy_mesh->pmb_pack->pmhd->peos->CarriedC2PHistory();
    if (hist != nullptr) n += hist->extent_int(1);
  }
  return n;
}

//----------------------------------------------------------------------------------------
//! \fn void MeshRefinement::SetSameLevelTransferExtent(AMRBuffer &buf)
//! \brief Cell extent of a same-level load-balance transfer: the active cells, or the
//! whole MeshBlock (ghost zones included) for a pure rebalance, whose target must hold
//! the moved block verbatim.  Face-centered counts follow PackAMRBuffersFC (one extra
//! face per direction).

#if MPI_PARALLEL_ENABLED
void MeshRefinement::SetSameLevelTransferExtent(AMRBuffer &buf) const {
  auto &indcs = pmy_mesh->mb_indcs;
  const int ng = indcs.ng;
  int n1 = indcs.nx1, n2 = indcs.nx2, n3 = indcs.nx3;
  if (lb_full_block_transfer) {
    n1 += 2*ng;
    if (pmy_mesh->multi_d) n2 += 2*ng;
    if (pmy_mesh->three_d) n3 += 2*ng;
    buf.bis = 0; buf.bjs = 0; buf.bks = 0;
  } else {
    buf.bis = indcs.is; buf.bjs = indcs.js; buf.bks = indcs.ks;
  }
  buf.bie = buf.bis + n1 - 1;
  buf.bje = buf.bjs + n2 - 1;
  buf.bke = buf.bks + n3 - 1;
  buf.cntcc = n1*n2*n3;
  buf.cntfc = 3*n1*n2*n3 + n2*n3 + n1*n3 + n1*n2;
}
#endif

void MeshRefinement::InitRecvAMR(int nleaf) {
#if MPI_PARALLEL_ENABLED
  // Step 1. (InitRecvAMR)
  // loop over new MBs on this rank, count number of MeshBlocks received by this rank
  nmb_recv = 0;
  int nmbs = new_gids_eachrank[global_variable::my_rank];
  int nmbe = nmbs + new_nmb_eachrank[global_variable::my_rank] - 1;
  for (int newm=nmbs; newm<=nmbe; newm++) {
    int oldm = newtoold[newm];
    LogicalLocation &old_lloc = pmy_mesh->lloc_eachmb[oldm];
    LogicalLocation &new_lloc = new_lloc_eachmb[newm];
    if (old_lloc.level > new_lloc.level) {          // old MB was de-refined
      const int root_rank = pmy_mesh->rank_eachmb[oldm];
      for (int l=0; l<nleaf; l++) {
        const int old_child = DerefOldChildGID(newm, l, nleaf);
        if (old_child < 0) continue;
        // recv whenever root MB changes rank, or if any leaf on different rank than root
        if ((root_rank != global_variable::my_rank) ||
            (pmy_mesh->rank_eachmb[old_child] != global_variable::my_rank)) {
          nmb_recv++;
        }
      }
    } else if (old_lloc.level == new_lloc.level) {  // old MB at same level
      if (pmy_mesh->rank_eachmb[oldm] != global_variable::my_rank) {
        nmb_recv++;
      }
    } else {                                        // old MB was refined
      // recv whenever refined MB changes rank, or if any leaf on different rank than root
      if ((new_rank_eachmb[oldtonew[oldm]] != global_variable::my_rank) ||
          (pmy_mesh->rank_eachmb[oldm] != global_variable::my_rank)) {
        nmb_recv++;
      }
    }
  }
  if (nmb_recv == 0) return;  // nothing to do

  // allocate array of recv buffers
  Kokkos::realloc(recvbuf, nmb_recv);

  // count number of cell- and face-centered variables communicated depending on physics
  int ncc_tosend=0, nfc_tosend=0;
  if (pmy_mesh->pmb_pack->phydro != nullptr) {
    ncc_tosend += pmy_mesh->pmb_pack->phydro->nvars;
  }
  if (pmy_mesh->pmb_pack->pmhd != nullptr) {
    ncc_tosend += pmy_mesh->pmb_pack->pmhd->nvars;
    nfc_tosend += 1;
  }
  if (pmy_mesh->pmb_pack->prad != nullptr) {
    ncc_tosend += (pmy_mesh->pmb_pack->prad->prgeo->nangles);
  }
  if (pmy_mesh->pmb_pack->pgrav != nullptr && lb_transfer_gravity) {
    ncc_tosend += 1;
  }
  if (pmy_mesh->pmb_pack->pz4c != nullptr) {
    ncc_tosend += (pmy_mesh->pmb_pack->pz4c->nz4c);
  }
  ncc_tosend += FullBlockTransferPrimitiveCount();

  // Step 2. (InitRecvAMR)
  // loop over new MBs on this rank, initialize recv buffers
  auto &indcs = pmy_mesh->mb_indcs;
  auto &is = indcs.is, &ie = indcs.ie;
  auto &js = indcs.js, &je = indcs.je;
  auto &ks = indcs.ks, &ke = indcs.ke;
  auto &cis = indcs.cis, &cie = indcs.cie;
  auto &cjs = indcs.cjs, &cje = indcs.cje;
  auto &cks = indcs.cks, &cke = indcs.cke;
  auto &cnx1 = indcs.cnx1, &cnx2 = indcs.cnx2, &cnx3 = indcs.cnx3;
  auto &nx1 = indcs.nx1, &nx2 = indcs.nx2, &nx3 = indcs.nx3;
  auto &ng = indcs.ng;
  int il = cis - ng, iu = cie + ng;
  int jl = cjs,      ju = cje;
  int kl = cks,      ku = cke;
  if (pmy_mesh->multi_d) {
    jl -= ng; ju += ng;
  }
  if (pmy_mesh->three_d) {
    kl -= ng; ku += ng;
  }

  int rb_idx = 0;   // recv buffer index
  std::vector<int> recv_peer_ranks, recv_keys;
  recv_peer_ranks.reserve(nmb_recv);
  recv_keys.reserve(nmb_recv);
  for (int newm=nmbs; newm<=nmbe; newm++) {
    int oldm = newtoold[newm];
    LogicalLocation &old_lloc = pmy_mesh->lloc_eachmb[oldm];
    LogicalLocation &new_lloc = new_lloc_eachmb[newm];
    if (old_lloc.level > new_lloc.level) {        // old MB was de-refined
      const int root_rank = pmy_mesh->rank_eachmb[oldm];
      for (int l=0; l<nleaf; l++) {
        const int old_child = DerefOldChildGID(newm, l, nleaf);
        if (old_child < 0) continue;
        // recv whenever root MB changes rank, or if any leaf on different rank than root
        if ((root_rank != global_variable::my_rank) ||
            (pmy_mesh->rank_eachmb[old_child] != global_variable::my_rank)) {
          LogicalLocation &lloc = pmy_mesh->lloc_eachmb[old_child];
          int ox1 = ((lloc.lx1 & 1) == 1);
          int ox2 = ((lloc.lx2 & 1) == 1);
          int ox3 = ((lloc.lx3 & 1) == 1);
          recvbuf.h_view(rb_idx).bis = cis + ox1*cnx1;
          recvbuf.h_view(rb_idx).bie = cie + ox1*cnx1;
          recvbuf.h_view(rb_idx).bjs = cjs + ox2*cnx2;
          recvbuf.h_view(rb_idx).bje = cje + ox2*cnx2;
          recvbuf.h_view(rb_idx).bks = cks + ox3*cnx3;
          recvbuf.h_view(rb_idx).bke = cke + ox3*cnx3;
          recvbuf.h_view(rb_idx).cntcc = cnx1*cnx2*cnx3;
          recvbuf.h_view(rb_idx).cntfc = 3*cnx1*cnx2*cnx3 + cnx2*cnx3 +
                                          cnx1*cnx3 + cnx1*cnx2;
          recvbuf.h_view(rb_idx).cnt = AMRMessageCount(
              ncc_tosend, nfc_tosend, recvbuf.h_view(rb_idx).cntcc,
              recvbuf.h_view(rb_idx).cntfc);
          recvbuf.h_view(rb_idx).lid   = newm - nmbs;
          recvbuf.h_view(rb_idx).use_coarse = false;
          recv_peer_ranks.push_back(pmy_mesh->rank_eachmb[old_child]);
          recv_keys.push_back(CreateAMR_MPI_Tag(newm-nmbs, ox1, ox2, ox3));
          rb_idx++;
        }
      }
    } else if (old_lloc.level == new_lloc.level) {   // old MB at same level
      if (pmy_mesh->rank_eachmb[oldm] != global_variable::my_rank) {
        SetSameLevelTransferExtent(recvbuf.h_view(rb_idx));
        recvbuf.h_view(rb_idx).cnt = AMRMessageCount(
            ncc_tosend, nfc_tosend, recvbuf.h_view(rb_idx).cntcc,
            recvbuf.h_view(rb_idx).cntfc);
        recvbuf.h_view(rb_idx).lid = newm - nmbs;
        recvbuf.h_view(rb_idx).use_coarse = false;
        recv_peer_ranks.push_back(pmy_mesh->rank_eachmb[oldm]);
        recv_keys.push_back(CreateAMR_MPI_Tag(newm-nmbs, 0, 0, 0));
        rb_idx++;
      }
    } else {                                        // old MB was refined
      // recv whenever refined MB changes rank, or if any leaf on different rank than root
      if ((new_rank_eachmb[oldtonew[oldm]] != global_variable::my_rank) ||
          (pmy_mesh->rank_eachmb[oldm] != global_variable::my_rank)) {
        recvbuf.h_view(rb_idx).bis = il; // note il:iu etc. includes ghost zones
        recvbuf.h_view(rb_idx).bie = iu;
        recvbuf.h_view(rb_idx).bjs = jl;
        recvbuf.h_view(rb_idx).bje = ju;
        recvbuf.h_view(rb_idx).bks = kl;
        recvbuf.h_view(rb_idx).bke = ku;
        recvbuf.h_view(rb_idx).cntcc = (iu-il+1)*(ju-jl+1)*(ku-kl+1);
        recvbuf.h_view(rb_idx).cntfc = (iu-il+2)*(ju-jl+1)*(ku-kl+1) +
             (iu-il+1)*(ju-jl+2)*(ku-kl+1) + (iu-il+1)*(ju-jl+1)*(ku-kl+2);
        recvbuf.h_view(rb_idx).cnt = AMRMessageCount(
            ncc_tosend, nfc_tosend, recvbuf.h_view(rb_idx).cntcc,
            recvbuf.h_view(rb_idx).cntfc);
        recvbuf.h_view(rb_idx).lid = newm - nmbs;
        recvbuf.h_view(rb_idx).use_coarse = true;
        recv_peer_ranks.push_back(pmy_mesh->rank_eachmb[oldm]);
        recv_keys.push_back(CreateAMR_MPI_Tag(newm-nmbs, 0, 0, 0));
        rb_idx++;
      }
    }
  }
  ValidateAMRDescriptorCount("receive", rb_idx, nmb_recv);
  // Order the descriptors by (source rank, key), sync them, take the receive buffer and
  // post one receive per source rank.
  const std::size_t ndata = OrderAMRDescriptors(recvbuf, nmb_recv, recv_peer_ranks,
                                                recv_keys, rank_recv_messages, "receive");
  recvbuf.template modify<HostMemSpace>();
  recvbuf.template sync<DevExeSpace>();
  recv_ptr = AcquireAMRBuffer(ndata, lb_chunk_elements, recv_pool, recv_pinned);

  // Step 3. (InitRecvAMR)
  // Receive requests will only be accessed on host, so no need to sync after this step.
  bool no_errors=true;
  recv_requests.clear();
  recv_request_message.clear();
  for (int im=0; im<static_cast<int>(rank_recv_messages.size()); ++im) {
    AMRRankMessage &msg = rank_recv_messages[im];
    msg.pending = msg.nrequest;
    for (int j=0; j<msg.nrequest; ++j) {
      const std::size_t first = static_cast<std::size_t>(j)*kAMRMaxMessageElements;
      const int count = static_cast<int>(
          std::min(kAMRMaxMessageElements, msg.count - first));
      MPI_Request request;
      int ierr = MPI_Irecv(recv_ptr + msg.offset + first, count, MPI_ATHENA_REAL,
                           msg.rank, j, amr_comm, &request);
      if (ierr != MPI_SUCCESS) no_errors=false;
      recv_requests.push_back(request);
      recv_request_message.push_back(im);
    }
  }
  // Quit if MPI error detected
  if (!(no_errors)) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
       << std::endl << "MPI error in posting non-blocking receives with AMR" << std::endl;
    std::exit(EXIT_FAILURE);
  }
#endif
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void MeshRefinement::PackAndSendAMR()
//! \brief Allocates and initializes send buffers for communicating MeshBlocks during
//! load balancing, calls function to pack data into buffers, and posts non-blocking sends
//! Equivalent to some of the work done inside MPI_PARALLEL block in the
//! Mesh::RedistributeAndRefineMeshBlocks() function in amr_loadbalance.cpp in Athena++

void MeshRefinement::PackAndSendAMR(int nleaf) {
#if MPI_PARALLEL_ENABLED
  // Step 1. (PackAndSendAMR)
  // loop over old MBs on this rank, count number of MeshBlocks to send on this rank
  nmb_send = 0;
  int ombs = pmy_mesh->gids_eachrank[global_variable::my_rank];
  int ombe = ombs + pmy_mesh->nmb_eachrank[global_variable::my_rank] - 1;
  for (int oldm=ombs; oldm<=ombe; oldm++) {
    int newm = oldtonew[oldm];
    LogicalLocation &old_lloc = pmy_mesh->lloc_eachmb[oldm];
    LogicalLocation &new_lloc = new_lloc_eachmb[newm];
    if (old_lloc.level < new_lloc.level) {          // old MB was refined
      for (int l=0; l<nleaf; l++) {
        const int new_child = RefineNewChildGID(oldm, l, nleaf);
        if (new_child < 0) continue;
        // send if refined MB changes rank, or if any leaf on different rank than root
        if ((new_rank_eachmb[newm] != global_variable::my_rank) ||
            (new_rank_eachmb[new_child] != global_variable::my_rank)) {
          nmb_send++;
        }
      }
    } else if (old_lloc.level == new_lloc.level) {  // old MB on same level
      if (new_rank_eachmb[newm] != global_variable::my_rank) {
        nmb_send++;
      }
    } else {                                        // old MB was de-refined
      // send if root MB changes rank, or if any leaf on different rank than root
      if ((pmy_mesh->rank_eachmb[newtoold[newm]] != global_variable::my_rank) ||
          (new_rank_eachmb[newm] != global_variable::my_rank)) {
        nmb_send++;
      }
    }
  }

  if (nmb_send == 0) return;  // nothing to do

  // allocate array of send buffers
  Kokkos::realloc(sendbuf, nmb_send);

  // count number of cell- and face-centered variables communicated depending on physics
  int ncc_tosend=0, nfc_tosend=0;
  if (pmy_mesh->pmb_pack->phydro != nullptr) {
    // Hydro AMR/load-balance buffers must include auxiliary carried fields such as
    // dual-energy internal energy, not just the standard hydro+scalar state.
    ncc_tosend += pmy_mesh->pmb_pack->phydro->nvars;
  }
  if (pmy_mesh->pmb_pack->pmhd != nullptr) {
    ncc_tosend += pmy_mesh->pmb_pack->pmhd->nvars;
    nfc_tosend += 1;
  }
  if (pmy_mesh->pmb_pack->prad != nullptr) {
    ncc_tosend += (pmy_mesh->pmb_pack->prad->prgeo->nangles);
  }
  if (pmy_mesh->pmb_pack->pgrav != nullptr && lb_transfer_gravity) {
    ncc_tosend += 1;
  }
  if (pmy_mesh->pmb_pack->pz4c != nullptr) {
    ncc_tosend += (pmy_mesh->pmb_pack->pz4c->nz4c);
  }
  ncc_tosend += FullBlockTransferPrimitiveCount();

  // Step 2. (PackAndSendAMR)
  // loop over old MBs on this rank, initialize send buffers
  auto &indcs = pmy_mesh->mb_indcs;
  auto &is = indcs.is, &ie = indcs.ie;
  auto &js = indcs.js, &je = indcs.je;
  auto &ks = indcs.ks, &ke = indcs.ke;
  auto &cis = indcs.cis, &cie = indcs.cie;
  auto &cjs = indcs.cjs, &cje = indcs.cje;
  auto &cks = indcs.cks, &cke = indcs.cke;
  auto &cnx1 = indcs.cnx1, &cnx2 = indcs.cnx2, &cnx3 = indcs.cnx3;
  auto &nx1 = indcs.nx1, &nx2 = indcs.nx2, &nx3 = indcs.nx3;
  auto &ng = indcs.ng;
  int il = cis - ng, iu = cie + ng;
  int jl = cjs,      ju = cje;
  int kl = cks,      ku = cke;
  if (pmy_mesh->multi_d) {
    jl -= ng; ju += ng;
  }
  if (pmy_mesh->three_d) {
    kl -= ng; ku += ng;
  }

  int sb_idx = 0;   // send buffer index
  std::vector<int> send_peer_ranks, send_keys;
  send_peer_ranks.reserve(nmb_send);
  send_keys.reserve(nmb_send);
  for (int oldm=ombs; oldm<=ombe; oldm++) {
    int newm = oldtonew[oldm];
    LogicalLocation &old_lloc = pmy_mesh->lloc_eachmb[oldm];
    LogicalLocation &new_lloc = new_lloc_eachmb[newm];
    if (old_lloc.level < new_lloc.level) {  // old MB was refined
      for (int l=0; l<nleaf; l++) {
        const int new_child = RefineNewChildGID(oldm, l, nleaf);
        if (new_child < 0) continue;
        // send if refined MB changes rank, or if any leaf on different rank than root
        if ((new_rank_eachmb[newm] != global_variable::my_rank) ||
            (new_rank_eachmb[new_child] != global_variable::my_rank)) {
          LogicalLocation &lloc = new_lloc_eachmb[new_child];
          int ox1 = ((lloc.lx1 & 1) == 1);
          int ox2 = ((lloc.lx2 & 1) == 1);
          int ox3 = ((lloc.lx3 & 1) == 1);
          sendbuf.h_view(sb_idx).bis = il + ox1*cnx1;  // il:iu etc. includes ghost zones
          sendbuf.h_view(sb_idx).bie = iu + ox1*cnx1;
          sendbuf.h_view(sb_idx).bjs = jl + ox2*cnx2;
          sendbuf.h_view(sb_idx).bje = ju + ox2*cnx2;
          sendbuf.h_view(sb_idx).bks = kl + ox3*cnx3;
          sendbuf.h_view(sb_idx).bke = ku + ox3*cnx3;
          sendbuf.h_view(sb_idx).cntcc = (iu-il+1)*(ju-jl+1)*(ku-kl+1);
          sendbuf.h_view(sb_idx).cntfc = (iu-il+2)*(ju-jl+1)*(ku-kl+1) +
               (iu-il+1)*(ju-jl+2)*(ku-kl+1) + (iu-il+1)*(ju-jl+1)*(ku-kl+2);
          sendbuf.h_view(sb_idx).cnt = AMRMessageCount(
              ncc_tosend, nfc_tosend, sendbuf.h_view(sb_idx).cntcc,
              sendbuf.h_view(sb_idx).cntfc);
          sendbuf.h_view(sb_idx).lid   = oldm - ombs;
          sendbuf.h_view(sb_idx).use_coarse = false;
          send_peer_ranks.push_back(new_rank_eachmb[new_child]);
          send_keys.push_back(CreateAMR_MPI_Tag(
              new_child - new_gids_eachrank[new_rank_eachmb[new_child]], 0, 0, 0));
          sb_idx++;
        }
      }
    } else {   // same level or de-refinement
      if (old_lloc.level == new_lloc.level) { // old MB on same level
        if (new_rank_eachmb[newm] != global_variable::my_rank) {
          SetSameLevelTransferExtent(sendbuf.h_view(sb_idx));
          sendbuf.h_view(sb_idx).cnt = AMRMessageCount(
              ncc_tosend, nfc_tosend, sendbuf.h_view(sb_idx).cntcc,
              sendbuf.h_view(sb_idx).cntfc);
          sendbuf.h_view(sb_idx).lid = oldm - ombs;
          sendbuf.h_view(sb_idx).use_coarse = false;
          send_peer_ranks.push_back(new_rank_eachmb[newm]);
          send_keys.push_back(CreateAMR_MPI_Tag(
              newm - new_gids_eachrank[new_rank_eachmb[newm]], 0, 0, 0));
          sb_idx++;
        }
      } else {                                // old MB was de-refined
        // send whenever root MB changes rank, or if any leaf on different rank than root
        if ((pmy_mesh->rank_eachmb[newtoold[newm]] != global_variable::my_rank) ||
            (new_rank_eachmb[newm] != global_variable::my_rank)) {
          sendbuf.h_view(sb_idx).bis = cis;
          sendbuf.h_view(sb_idx).bie = cie;
          sendbuf.h_view(sb_idx).bjs = cjs;
          sendbuf.h_view(sb_idx).bje = cje;
          sendbuf.h_view(sb_idx).bks = cks;
          sendbuf.h_view(sb_idx).bke = cke;
          sendbuf.h_view(sb_idx).cntcc = cnx1*cnx2*cnx3;
          sendbuf.h_view(sb_idx).cntfc = 3*cnx1*cnx2*cnx3 + cnx2*cnx3 + cnx1*cnx3
                                          + cnx1*cnx2;
          sendbuf.h_view(sb_idx).use_coarse = true;
          sendbuf.h_view(sb_idx).cnt = AMRMessageCount(
              ncc_tosend, nfc_tosend, sendbuf.h_view(sb_idx).cntcc,
              sendbuf.h_view(sb_idx).cntfc);
          sendbuf.h_view(sb_idx).lid = oldm - ombs;
          send_peer_ranks.push_back(new_rank_eachmb[newm]);
          send_keys.push_back(CreateAMR_MPI_Tag(
              newm - new_gids_eachrank[new_rank_eachmb[newm]],
              ((old_lloc.lx1 & 1) == 1), ((old_lloc.lx2 & 1) == 1),
              ((old_lloc.lx3 & 1) == 1)));
          sb_idx++;
        }
      }
    }
  }
  ValidateAMRDescriptorCount("send", sb_idx, nmb_send);
  // Order the descriptors by (destination rank, key), sync them and take the send buffer.
  const std::size_t ndata = OrderAMRDescriptors(sendbuf, nmb_send, send_peer_ranks,
                                                send_keys, rank_send_messages, "send");
  sendbuf.template modify<HostMemSpace>();
  sendbuf.template sync<DevExeSpace>();
  send_ptr = AcquireAMRBuffer(ndata, lb_chunk_elements, send_pool, send_pinned);

  // Step 3. (PackAndSendAMR)
  // Pack every outgoing value before any local or received destination slot can overwrite
  // its source.
  hydro::Hydro* phydro = pmy_mesh->pmb_pack->phydro;
  mhd::MHD* pmhd = pmy_mesh->pmb_pack->pmhd;
  radiation::Radiation* prad = pmy_mesh->pmb_pack->prad;
  gravity::Gravity* pgrav = pmy_mesh->pmb_pack->pgrav;
  z4c::Z4c* pz4c = pmy_mesh->pmb_pack->pz4c;

  {
    const int mb_begin = 0, mb_end = nmb_send;

    int ncc_sent = 0, nfc_sent = 0;
    if (phydro != nullptr) {
      PackAMRBuffersCC(phydro->u0, phydro->coarse_u0, ncc_sent, nfc_sent,
                       mb_begin, mb_end);
      ncc_sent += phydro->nvars;
    }
    if (pmhd != nullptr) {
      PackAMRBuffersCC(pmhd->u0, pmhd->coarse_u0, ncc_sent, nfc_sent,
                       mb_begin, mb_end);
      ncc_sent += pmhd->nvars;
      if (pmhd->HasField()) {
        PackAMRBuffersFC(pmhd->b0, pmhd->coarse_b0, ncc_sent, nfc_sent,
                         mb_begin, mb_end);
        nfc_sent += 1;
      }
    }
    if (prad != nullptr) {
      PackAMRBuffersCC(prad->i0, prad->coarse_i0, ncc_sent, nfc_sent,
                       mb_begin, mb_end);
      ncc_sent += prad->prgeo->nangles;
    }
    if (pgrav != nullptr && lb_transfer_gravity) {
      PackAMRBuffersCC(pgrav->phi, pgrav->coarse_phi, ncc_sent, nfc_sent,
                       mb_begin, mb_end);
      ncc_sent += 1;
    }
    if (pz4c != nullptr) {
      PackAMRBuffersCC(pz4c->u0, pz4c->coarse_u0, ncc_sent, nfc_sent,
                       mb_begin, mb_end);
      ncc_sent += pz4c->nz4c;
    }
    // Pure rebalance: the fluid primitives travel with the conserved state (see
    // lb_full_block_transfer).  Same-level only, so the coarse argument is never read.
    if (lb_full_block_transfer) {
      if (phydro != nullptr) {
        PackAMRBuffersCC(phydro->w0, phydro->w0, ncc_sent, nfc_sent,
                         mb_begin, mb_end);
        ncc_sent += phydro->w0.extent_int(1);
      }
      if (pmhd != nullptr) {
        PackAMRBuffersCC(pmhd->w0, pmhd->w0, ncc_sent, nfc_sent,
                         mb_begin, mb_end);
        ncc_sent += pmhd->w0.extent_int(1);
        if (pmhd->HasField()) {
          PackAMRBuffersCC(pmhd->bcc0, pmhd->bcc0, ncc_sent, nfc_sent,
                           mb_begin, mb_end);
          ncc_sent += pmhd->bcc0.extent_int(1);
        }
        DvceArray5D<Real> *hist = pmhd->peos->CarriedC2PHistory();
        if (hist != nullptr) {
          PackAMRBuffersCC(*hist, *hist, ncc_sent, nfc_sent, mb_begin, mb_end);
          ncc_sent += hist->extent_int(1);
        }
      }
    }

  }
  Kokkos::fence("AMR load-balance send pack");

  // Step 4. (PackAndSendAMR)
  // One non-blocking send per destination rank, after a single fence on the pack.
  // Send requests will only be accessed on host, so no need to sync after this step.
  bool no_errors=true;
  send_requests.clear();
  for (auto &msg : rank_send_messages) {
    for (int j=0; j<msg.nrequest; ++j) {
      const std::size_t first = static_cast<std::size_t>(j)*kAMRMaxMessageElements;
      const int count = static_cast<int>(
          std::min(kAMRMaxMessageElements, msg.count - first));
      MPI_Request request;
      int ierr = MPI_Isend(send_ptr + msg.offset + first, count, MPI_ATHENA_REAL,
                           msg.rank, j, amr_comm, &request);
      if (ierr != MPI_SUCCESS) no_errors=false;
      send_requests.push_back(request);
    }
  }
  // Quit if MPI error detected
  if (!(no_errors)) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << "MPI error in posting non-blocking sends with AMR"
              << std::endl;
    std::exit(EXIT_FAILURE);
  }
#endif
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void MeshRefinement::PackAMRBuffersCC()
//! \brief Packs cell-centered data into AMR communication buffers for all MBs being sent
//! Equivalent to PrepareSendSameLevel(), PrepareSendCoarseToFineAMR(), and
//! PrepareSendFineToCoarseAMR() functions in amr_loadbalance.cpp

void MeshRefinement::PackAMRBuffersCC(DvceArray5D<Real> &a, DvceArray5D<Real> &ca,
                                      int ncc, int nfc, int mb_begin, int mb_end,
                                      int ko, int jo, int io) {
#if MPI_PARALLEL_ENABLED
  auto &sbuf = sendbuf;
  Real *sdata = send_ptr;
  // Outer loop over (# of MeshBlocks sent)*(# of variables)
  int nvar = a.extent_int(1);  // TODO(@user): 2nd index from L of in array must be NVAR
  int nnv = (mb_end - mb_begin)*nvar;
  Kokkos::TeamPolicy<> policy(DevExeSpace(), nnv, Kokkos::AUTO);
  Kokkos::parallel_for("SendBuff", policy, KOKKOS_LAMBDA(TeamMember_t tmember) {
    const int local_n = (tmember.league_rank())/nvar;
    const int n = mb_begin + local_n;
    const int v = (tmember.league_rank() - local_n*nvar);

    const int il = sbuf.d_view(n).bis;
    const int jl = sbuf.d_view(n).bjs;
    const int kl = sbuf.d_view(n).bks;
    const int ni = sbuf.d_view(n).bie - il + 1;
    const int nj = sbuf.d_view(n).bje - jl + 1;
    const int nk = sbuf.d_view(n).bke - kl + 1;
    const int nkji = nk*nj*ni;
    const int nji  = nj*ni;
    const int m  = sbuf.d_view(n).lid;
    const std::size_t offset = sbuf.d_view(n).offset +
        static_cast<std::size_t>(ncc)*sbuf.d_view(n).cntcc +
        static_cast<std::size_t>(nfc)*sbuf.d_view(n).cntfc;

    // Middle loop over k,j,i
    Kokkos::parallel_for(Kokkos::TeamThreadRange<>(tmember, nkji), [&](const int idx) {
      int k = (idx)/nji;
      int j = (idx - k*nji)/ni;
      int i = (idx - k*nji - j*ni) + il;
      k += kl;
      j += jl;
      if (sbuf.d_view(n).use_coarse) {
        // if de-refinement, load data from coarse_a.  Banded arrays (nonzero origin)
        // must never take the coarse branch: ca is indexed without the band origin.
        if (ko != 0 || jo != 0 || io != 0) {
          Kokkos::abort("AMR pack: a banded array took the coarse branch");
        }
        sdata[offset + (i-il + ni*(j-jl + nj*(k-kl + nk*v)))] = ca(m,v,k,j,i);
      } else {
        // if refinement or same level, load data from a.  An array stored on a band
        // (origin ko,jo,io) holds only its band: the rest of the block is sent as 0.
        const int kb = k - ko, jb = j - jo, ib = i - io;
        const bool in_band = (kb >= 0 && kb < static_cast<int>(a.extent(2)) &&
                              jb >= 0 && jb < static_cast<int>(a.extent(3)) &&
                              ib >= 0 && ib < static_cast<int>(a.extent(4)));
        sdata[offset + (i-il + ni*(j-jl + nj*(k-kl + nk*v)))] =
            in_band ? a(m,v,kb,jb,ib) : 0.0;
      }
    });
  }); // end par_for_outer
#endif
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void MeshRefinement::PackAMRBuffersFC()
//! \brief Packs face-centered data into AMR communication buffers for all MBs being sent

void MeshRefinement::PackAMRBuffersFC(DvceFaceFld4D<Real> &b, DvceFaceFld4D<Real> &cb,
                                      int ncc, int nfc, int mb_begin, int mb_end) {
#if MPI_PARALLEL_ENABLED
  auto &sbuf = sendbuf;
  Real *sdata = send_ptr;
  // Outer loop over (# of MeshBlocks sent)*(3 compnts of field)
  int nn = 3*(mb_end - mb_begin);
  Kokkos::TeamPolicy<> policy(DevExeSpace(), nn, Kokkos::AUTO);
  Kokkos::parallel_for("SendBuff", policy, KOKKOS_LAMBDA(TeamMember_t tmember) {
    const int local_n = (tmember.league_rank())/3;
    const int n = mb_begin + local_n;
    const int v = (tmember.league_rank() - 3*local_n);

    const int il = sbuf.d_view(n).bis;
    const int jl = sbuf.d_view(n).bjs;
    const int kl = sbuf.d_view(n).bks;
    const int m  = sbuf.d_view(n).lid;
    const int nicc = sbuf.d_view(n).bie - il + 1;
    const int njcc = sbuf.d_view(n).bje - jl + 1;
    const int nkcc = sbuf.d_view(n).bke - kl + 1;

    // pack x1 component
    if (v==0) {
      const std::size_t offset = sbuf.d_view(n).offset +
          static_cast<std::size_t>(ncc)*sbuf.d_view(n).cntcc +
          static_cast<std::size_t>(nfc)*sbuf.d_view(n).cntfc;
      const int ni = nicc + 1;  // add b.x1f at (ie+1)
      const int nj = njcc;
      const int nk = nkcc;
      const int nkji = nk*nj*ni;
      const int nji  = nj*ni;

      // Middle loop over k,j,i
      Kokkos::parallel_for(Kokkos::TeamThreadRange<>(tmember, nkji), [&](const int idx) {
        int k = (idx)/nji;
        int j = (idx - k*nji)/ni;
        int i = (idx - k*nji - j*ni) + il;
        k += kl;
        j += jl;
        if (sbuf.d_view(n).use_coarse) {
          // if de-refinement, load data from coarse_a
          sdata[offset + (i-il + ni*(j-jl + nj*(k-kl)))] = cb.x1f(m,k,j,i);
        } else {
          // if refinement or same level, load data from a
          sdata[offset + (i-il + ni*(j-jl + nj*(k-kl)))] = b.x1f(m,k,j,i);
        }
      });

    // pack x2 component
    } else if (v==1) {
      const std::size_t offset = sbuf.d_view(n).offset +
          static_cast<std::size_t>(ncc)*sbuf.d_view(n).cntcc +
          static_cast<std::size_t>(nfc)*sbuf.d_view(n).cntfc +
          static_cast<std::size_t>(nicc + 1)*njcc*nkcc;
      const int ni = nicc;
      const int nj = njcc + 1;  // add b.x2f at (je+1)
      const int nk = nkcc;
      const int nkji = nk*nj*ni;
      const int nji  = nj*ni;

      // Middle loop over k,j,i
      Kokkos::parallel_for(Kokkos::TeamThreadRange<>(tmember, nkji), [&](const int idx) {
        int k = (idx)/nji;
        int j = (idx - k*nji)/ni;
        int i = (idx - k*nji - j*ni) + il;
        k += kl;
        j += jl;
        if (sbuf.d_view(n).use_coarse) {
          // if de-refinement, load data from coarse_a
          sdata[offset + (i-il + ni*(j-jl + nj*(k-kl)))] = cb.x2f(m,k,j,i);
        } else {
          // if refinement or same level, load data from a
          sdata[offset + (i-il + ni*(j-jl + nj*(k-kl)))] = b.x2f(m,k,j,i);
        }
      });

    // pack x3 component
    } else {
      const std::size_t offset = sbuf.d_view(n).offset +
          static_cast<std::size_t>(ncc)*sbuf.d_view(n).cntcc +
          static_cast<std::size_t>(nfc)*sbuf.d_view(n).cntfc +
          static_cast<std::size_t>(nicc + 1)*njcc*nkcc +
          static_cast<std::size_t>(nicc)*(njcc + 1)*nkcc;
      const int ni = nicc;
      const int nj = njcc;
      const int nk = nkcc + 1;  // add b.x3f at (ke+1)
      const int nkji = nk*nj*ni;
      const int nji  = nj*ni;

      // Middle loop over k,j,i
      Kokkos::parallel_for(Kokkos::TeamThreadRange<>(tmember, nkji), [&](const int idx) {
        int k = (idx)/nji;
        int j = (idx - k*nji)/ni;
        int i = (idx - k*nji - j*ni) + il;
        k += kl;
        j += jl;
        if (sbuf.d_view(n).use_coarse) {
          // if de-refinement, load data from coarse_a
          sdata[offset + (i-il + ni*(j-jl + nj*(k-kl)))] = cb.x3f(m,k,j,i);
        } else {
          // if refinement or same level, load data from a
          sdata[offset + (i-il + ni*(j-jl + nj*(k-kl)))] = b.x3f(m,k,j,i);
        }
      });
    }
  }); // end par_for_outer
#endif
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void MeshRefinement::ClearRecvAndUnpackAMR()
//! \brief Checks non-blocking receives have finished, calls function to unpack buffers,
//! deletes receive buffers. Equivalent to some of the work done inside MPI_PARALLEL block
//! in the Mesh::RedistributeAndRefineMeshBlocks() function in amr_loadbalance.cpp

void MeshRefinement::ClearRecvAndUnpackAMR() {
#if MPI_PARALLEL_ENABLED
  // Each source rank's payload is one contiguous range of the receive buffer, so the
  // blocks of a rank are unpacked as soon as its last message has arrived.
  hydro::Hydro* phydro = pmy_mesh->pmb_pack->phydro;
  mhd::MHD* pmhd = pmy_mesh->pmb_pack->pmhd;
  radiation::Radiation* prad = pmy_mesh->pmb_pack->prad;
  gravity::Gravity* pgrav = pmy_mesh->pmb_pack->pgrav;
  z4c::Z4c* pz4c = pmy_mesh->pmb_pack->pz4c;

  auto unpack_range = [&](int mb_begin, int mb_end) {
    int ncc_recv = 0, nfc_recv = 0;
    if (phydro != nullptr) {
      UnpackAMRBuffersCC(phydro->u0, phydro->coarse_u0, ncc_recv, nfc_recv,
                         mb_begin, mb_end);
      ncc_recv += phydro->nvars;
    }
    if (pmhd != nullptr) {
      UnpackAMRBuffersCC(pmhd->u0, pmhd->coarse_u0, ncc_recv, nfc_recv,
                         mb_begin, mb_end);
      ncc_recv += pmhd->nvars;
      if (pmhd->HasField()) {
        UnpackAMRBuffersFC(pmhd->b0, pmhd->coarse_b0, ncc_recv, nfc_recv,
                           mb_begin, mb_end);
        nfc_recv += 1;
      }
    }
    if (prad != nullptr) {
      UnpackAMRBuffersCC(prad->i0, prad->coarse_i0, ncc_recv, nfc_recv,
                         mb_begin, mb_end);
      ncc_recv += prad->prgeo->nangles;
    }
    if (pgrav != nullptr && lb_transfer_gravity) {
      UnpackAMRBuffersCC(pgrav->phi, pgrav->coarse_phi, ncc_recv, nfc_recv,
                         mb_begin, mb_end);
      ncc_recv += 1;
    }
    if (pz4c != nullptr) {
      UnpackAMRBuffersCC(pz4c->u0, pz4c->coarse_u0, ncc_recv, nfc_recv,
                         mb_begin, mb_end);
      ncc_recv += pz4c->nz4c;
    }
    if (lb_full_block_transfer) {
      if (phydro != nullptr) {
        UnpackAMRBuffersCC(phydro->w0, phydro->w0, ncc_recv, nfc_recv,
                           mb_begin, mb_end);
        ncc_recv += phydro->w0.extent_int(1);
      }
      if (pmhd != nullptr) {
        UnpackAMRBuffersCC(pmhd->w0, pmhd->w0, ncc_recv, nfc_recv,
                           mb_begin, mb_end);
        ncc_recv += pmhd->w0.extent_int(1);
        if (pmhd->HasField()) {
          UnpackAMRBuffersCC(pmhd->bcc0, pmhd->bcc0, ncc_recv, nfc_recv,
                             mb_begin, mb_end);
          ncc_recv += pmhd->bcc0.extent_int(1);
        }
        DvceArray5D<Real> *hist = pmhd->peos->CarriedC2PHistory();
        if (hist != nullptr) {
          UnpackAMRBuffersCC(*hist, *hist, ncc_recv, nfc_recv, mb_begin, mb_end);
          ncc_recv += hist->extent_int(1);
        }
      }
    }
  };

  const int nrequest = static_cast<int>(recv_requests.size());
  bool no_errors=true;
  for (int nleft=nrequest; nleft>0; --nleft) {
    int index = MPI_UNDEFINED;
    MPI_Status status;
    int ierr = MPI_Waitany(nrequest, recv_requests.data(), &index, &status);
    if (ierr != MPI_SUCCESS || index == MPI_UNDEFINED) {
      no_errors = false;
      break;
    }
    AMRRankMessage &msg = rank_recv_messages[recv_request_message[index]];
    int slot = 0;
    for (int k=0; k<index; ++k) {
      if (recv_request_message[k] == recv_request_message[index]) ++slot;
    }
    const std::size_t first = static_cast<std::size_t>(slot)*kAMRMaxMessageElements;
    const int expected = static_cast<int>(
        std::min(kAMRMaxMessageElements, msg.count - first));
    int count = MPI_UNDEFINED;
    if (MPI_Get_count(&status, MPI_ATHENA_REAL, &count) != MPI_SUCCESS ||
        count != expected) {
      no_errors = false;
      break;
    }
    if (--msg.pending == 0) unpack_range(msg.begin, msg.end);
  }
  // Quit if MPI error detected
  if (!(no_errors)) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << "MPI error while waiting for AMR receives"
              << std::endl;
    std::exit(EXIT_FAILURE);
  }
  Kokkos::fence("AMR load-balance receive unpack");
  ReleaseAMRBuffer(recv_pool);
  recv_ptr = nullptr;
  rank_recv_messages.clear();
  recv_requests.clear();
  recv_request_message.clear();
  nmb_recv = 0;
#endif
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void MeshRefinement::UnpackAMRBuffersCC()
//! \brief Unpacks face-centered data from AMR communication buffers into appropriate
//! coarse or fine arrays for all MBs received during load balancing.
//! Equivalent to FinishRecvSameLevel(), FinishRecvCoarseToFineAMR(), and
//! FinishRecvFineToCoarseAMR() functions in amr_loadbalance.cpp

void MeshRefinement::UnpackAMRBuffersCC(DvceArray5D<Real> &a, DvceArray5D<Real> &ca,
                                        int ncc, int nfc, int mb_begin, int mb_end,
                                        int ko, int jo, int io) {
#if MPI_PARALLEL_ENABLED
  auto &rbuf = recvbuf;
  Real *rdata = recv_ptr;
  // Outer loop over (# of MeshBlocks recv)*(# of variables)
  int nvar = a.extent_int(1);  // TODO(@user): 2nd index from L of in array must be NVAR
  int nnv = (mb_end - mb_begin)*nvar;
  Kokkos::TeamPolicy<> policy(DevExeSpace(), nnv, Kokkos::AUTO);
  Kokkos::parallel_for("SendBuff", policy, KOKKOS_LAMBDA(TeamMember_t tmember) {
    const int local_n = (tmember.league_rank())/nvar;
    const int n = mb_begin + local_n;
    const int v = (tmember.league_rank() - local_n*nvar);

    const int il = rbuf.d_view(n).bis;
    const int jl = rbuf.d_view(n).bjs;
    const int kl = rbuf.d_view(n).bks;
    const int ni = rbuf.d_view(n).bie - il + 1;
    const int nj = rbuf.d_view(n).bje - jl + 1;
    const int nk = rbuf.d_view(n).bke - kl + 1;
    const int nkji = nk*nj*ni;
    const int nji  = nj*ni;
    const int m  = rbuf.d_view(n).lid;
    const std::size_t offset = rbuf.d_view(n).offset +
        static_cast<std::size_t>(ncc)*rbuf.d_view(n).cntcc +
        static_cast<std::size_t>(nfc)*rbuf.d_view(n).cntfc;

    // Middle loop over k,j,i
    Kokkos::parallel_for(Kokkos::TeamThreadRange<>(tmember, nkji), [&](const int idx) {
      int k = (idx)/nji;
      int j = (idx - k*nji)/ni;
      int i = (idx - k*nji - j*ni) + il;
      k += kl;
      j += jl;
      if (rbuf.d_view(n).use_coarse) {
        // if refinement, load data into coarse_a.  Banded arrays (nonzero origin) must
        // never take the coarse branch: ca is indexed without the band origin.
        if (ko != 0 || jo != 0 || io != 0) {
          Kokkos::abort("AMR unpack: a banded array took the coarse branch");
        }
        ca(m,v,k,j,i) = rdata[offset + (i-il + ni*(j-jl + nj*(k-kl + nk*v)))];
      } else {
        // if de-refinement or same level, load data into a (only the part of the block
        // inside a's band when a is stored on one, see PackAMRBuffersCC)
        const int kb = k - ko, jb = j - jo, ib = i - io;
        if (kb >= 0 && kb < static_cast<int>(a.extent(2)) &&
            jb >= 0 && jb < static_cast<int>(a.extent(3)) &&
            ib >= 0 && ib < static_cast<int>(a.extent(4))) {
          a(m,v,kb,jb,ib) = rdata[offset + (i-il + ni*(j-jl + nj*(k-kl + nk*v)))];
        }
      }
    });
  }); // end par_for_outer
#endif
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void MeshRefinement::UnpackAMRBuffersFC()
//! \brief Unpacks face-centered data from AMR communication buffers into appropriate
//! coarse or fine arrays for all MBs received during load balancing.

void MeshRefinement::UnpackAMRBuffersFC(DvceFaceFld4D<Real> &b, DvceFaceFld4D<Real> &cb,
                                        int ncc, int nfc, int mb_begin, int mb_end) {
#if MPI_PARALLEL_ENABLED
  auto &rbuf = recvbuf;
  Real *rdata = recv_ptr;
  // Outer loop over (# of MeshBlocks recv)*(3 compnts of field)
  int nnv = 3*(mb_end - mb_begin);
  Kokkos::TeamPolicy<> policy(DevExeSpace(), nnv, Kokkos::AUTO);
  Kokkos::parallel_for("SendBuff", policy, KOKKOS_LAMBDA(TeamMember_t tmember) {
    const int local_n = (tmember.league_rank())/3;
    const int n = mb_begin + local_n;
    const int v = (tmember.league_rank() - local_n*3);

    const int il = rbuf.d_view(n).bis;
    const int jl = rbuf.d_view(n).bjs;
    const int kl = rbuf.d_view(n).bks;
    const int m  = rbuf.d_view(n).lid;
    const int nicc = rbuf.d_view(n).bie - il + 1;
    const int njcc = rbuf.d_view(n).bje - jl + 1;
    const int nkcc = rbuf.d_view(n).bke - kl + 1;

    // unpack x1 component
    if (v==0) {
      const std::size_t offset = rbuf.d_view(n).offset +
          static_cast<std::size_t>(ncc)*rbuf.d_view(n).cntcc +
          static_cast<std::size_t>(nfc)*rbuf.d_view(n).cntfc;
      const int ni = nicc + 1;  // add b.x1f at (ie+1)
      const int nj = njcc;
      const int nk = nkcc;
      const int nkji = nk*nj*ni;
      const int nji  = nj*ni;

      // Middle loop over k,j,i
      Kokkos::parallel_for(Kokkos::TeamThreadRange<>(tmember, nkji), [&](const int idx) {
        int k = (idx)/nji;
        int j = (idx - k*nji)/ni;
        int i = (idx - k*nji - j*ni) + il;
        k += kl;
        j += jl;
        if (rbuf.d_view(n).use_coarse) {
          // if refinement, load data into coarse_a
          cb.x1f(m,k,j,i) = rdata[offset + (i-il + ni*(j-jl + nj*(k-kl)))];
        } else {
          // if de-refinement or same level, load data into a
          b.x1f(m,k,j,i) = rdata[offset + (i-il + ni*(j-jl + nj*(k-kl)))];
        }
      });

    // unpack x2 component
    } else if (v==1) {
      const std::size_t offset = rbuf.d_view(n).offset +
          static_cast<std::size_t>(ncc)*rbuf.d_view(n).cntcc +
          static_cast<std::size_t>(nfc)*rbuf.d_view(n).cntfc +
          static_cast<std::size_t>(nicc + 1)*njcc*nkcc;
      const int ni = nicc;
      const int nj = njcc + 1;  // add b.x2f at (je+1)
      const int nk = nkcc;
      const int nkji = nk*nj*ni;
      const int nji  = nj*ni;

      // Middle loop over k,j,i
      Kokkos::parallel_for(Kokkos::TeamThreadRange<>(tmember, nkji), [&](const int idx) {
        int k = (idx)/nji;
        int j = (idx - k*nji)/ni;
        int i = (idx - k*nji - j*ni) + il;
        k += kl;
        j += jl;
        if (rbuf.d_view(n).use_coarse) {
          // if refinement, load data into coarse_a
          cb.x2f(m,k,j,i) = rdata[offset + (i-il + ni*(j-jl + nj*(k-kl)))];
        } else {
          // if de-refinement or same level, load data into a
          b.x2f(m,k,j,i) = rdata[offset + (i-il + ni*(j-jl + nj*(k-kl)))];
        }
      });

    // unpack x3 component
    } else {
      const std::size_t offset = rbuf.d_view(n).offset +
          static_cast<std::size_t>(ncc)*rbuf.d_view(n).cntcc +
          static_cast<std::size_t>(nfc)*rbuf.d_view(n).cntfc +
          static_cast<std::size_t>(nicc + 1)*njcc*nkcc +
          static_cast<std::size_t>(nicc)*(njcc + 1)*nkcc;
      const int ni = nicc;
      const int nj = njcc;
      const int nk = nkcc + 1;  // add b.x3f at (ke+1)
      const int nkji = nk*nj*ni;
      const int nji  = nj*ni;

      // Middle loop over k,j,i
      Kokkos::parallel_for(Kokkos::TeamThreadRange<>(tmember, nkji), [&](const int idx) {
        int k = (idx)/nji;
        int j = (idx - k*nji)/ni;
        int i = (idx - k*nji - j*ni) + il;
        k += kl;
        j += jl;
        if (rbuf.d_view(n).use_coarse) {
          // if refinement, load data into coarse_a
          cb.x3f(m,k,j,i) = rdata[offset + (i-il + ni*(j-jl + nj*(k-kl)))];
        } else {
          // if de-refinement or same level, load data into a
          b.x3f(m,k,j,i) = rdata[offset + (i-il + ni*(j-jl + nj*(k-kl)))];
        }
      });
    }
  }); // end par_for_outer
#endif
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void MeshRefinement::ClearSendAMR()
//! \brief Checks all non-blocking sends completed, deletes send buffers.

void MeshRefinement::ClearSendAMR() {
#if MPI_PARALLEL_ENABLED
  int ierr = MPI_Waitall(static_cast<int>(send_requests.size()), send_requests.data(),
                         MPI_STATUSES_IGNORE);
  if (ierr != MPI_SUCCESS) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << "MPI error in clearing non-blocking sends with AMR"
              << std::endl;
    std::exit(EXIT_FAILURE);
  }
  ReleaseAMRBuffer(send_pool);
  send_ptr = nullptr;
  rank_send_messages.clear();
  send_requests.clear();
  nmb_send = 0;
#endif
  return;
}
