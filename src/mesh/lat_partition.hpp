#ifndef MESH_LAT_PARTITION_HPP_
#define MESH_LAT_PARTITION_HPP_
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file lat_partition.hpp
//! \brief The LAT block->rank partition, shared by every LAT module.  A rank must hold an
//! even share of every timestep class, since the classes step on different ticks; within
//! that constraint the partition keeps the ghost exchange on the rank and on the fast
//! links of the machine.  Included by mesh.hpp after LogicalLocation.

// Measured per-step work above a plain block (1) by more than this marks a block as
// heavy.  Below it the placement is inside the +-1 block the partition quotas round by.
constexpr double kLATHeavyExcess = 0.05;

struct LogicalLocationKey {
  std::int32_t lx1, lx2, lx3, level;

  explicit LogicalLocationKey(const LogicalLocation &loc)
      : lx1(loc.lx1), lx2(loc.lx2), lx3(loc.lx3), level(loc.level) {}

  LogicalLocationKey(std::int32_t x1, std::int32_t x2, std::int32_t x3,
                     std::int32_t lev)
      : lx1(x1), lx2(x2), lx3(x3), level(lev) {}

  bool operator==(const LogicalLocationKey &other) const {
    return lx1 == other.lx1 && lx2 == other.lx2 && lx3 == other.lx3 &&
           level == other.level;
  }
};

struct LogicalLocationKeyHash {
  std::size_t operator()(const LogicalLocationKey &key) const {
    std::uint64_t h = 1469598103934665603ull;
    auto mix = [&](std::uint32_t v) {
      h ^= static_cast<std::uint64_t>(v);
      h *= 1099511628211ull;
    };
    mix(static_cast<std::uint32_t>(key.lx1));
    mix(static_cast<std::uint32_t>(key.lx2));
    mix(static_cast<std::uint32_t>(key.lx3));
    mix(static_cast<std::uint32_t>(key.level));
    return static_cast<std::size_t>(h);
  }
};

inline LogicalLocationKey ParentKey(LogicalLocationKey key) {
  if (key.level > 0) {
    key.lx1 >>= 1;
    key.lx2 >>= 1;
    key.lx3 >>= 1;
    --key.level;
  }
  return key;
}

// Z (Morton) order of two MeshBlock locations of any levels.
bool LATZLess(const LogicalLocation &la, const LogicalLocation &lb, bool multi_d,
              bool three_d);

// The block count a LAT partition may give a rank: within a slack of the mean, and never
// beyond the allocation bound.  Device memory grows with a rank's block count.
struct LATCountBounds {
  int balance_min, balance_cap;
};
LATCountBounds ComputeLATCountBounds(int nb, int nranks, int block_cap);

//----------------------------------------------------------------------------------------
//! \struct LATRankTree
//! \brief The machine as the partition sees it: the node of every rank and, inside a
//! node, the clique of ranks joined by links much faster than the node's typical one
//! (NVLink against PCIe).  Both are named by their lowest rank.  Measured once per launch
//! (ProbeLATRankTree), never stored: it is machine data, not run data.

struct LATRankTree {
  std::vector<int> node;    // node of every rank (its lowest rank)
  std::vector<int> clique;  // fast clique of every rank (its lowest rank)
  bool probed = false;      // the cliques were measured (else every rank is its own)
  // Split an ascending rank group in two along the tree: whole nodes while the group
  // spans several, else whole cliques, else halves; each part as near half the ranks as
  // the units allow.  An empty tree is one node of single ranks.
  void Split(const std::vector<int> &group, std::vector<int> &first,
             std::vector<int> &second) const;
  std::string Describe(int nranks) const;
};

// Collective over all ranks.  Pairs of ranks on a node of >= 3 ranks exchange a device
// buffer in turn; a link at least 4x the node's median bandwidth is fast, and the fast
// cliques are the connected components of fast links.  Every rank gets the same tree.
LATRankTree ProbeLATRankTree();

//----------------------------------------------------------------------------------------
//! \struct LATPartitionRequest
//! \brief Inputs of BuildLATPartition, identical on every rank.

struct LATPartitionRequest {
  const LogicalLocation *lloc = nullptr;
  int nb = 0;
  const int *factor = nullptr;   // LAT factor of every block
  const Real *work = nullptr;    // measured per-step work, nullptr = every block 1
  const int *owner = nullptr;    // current rank of every block, nullptr = none (a seed)
  // false: a regrid before any LAT metadata (the owed pass of a restart); owners are
  // kept and only ranks outside the count bounds give blocks away
  bool balance_classes = true;
  int block_cap = 0;             // hard per-rank block cap (allocation bound)
  int window = 0;                // realised window length W in ticks, <= 1 if unknown
  double horizon = 0.0;          // ticks until the next partition event, H
  int exchanges = 3;             // state exchanges per block step, E
  double halo_bytes = 8.0;       // bytes per ghost cell, b_h
  double move_bytes = 0.0;       // bytes per moved block, b_m
  bool multi_d = false, three_d = false;
  int nx1 = 1, nx2 = 1, nx3 = 1, ng = 1;
  const LATRankTree *tree = nullptr;
};

struct LATPartitionResult {
  std::vector<int> new_to_input;  // input index placed at each new gid
  std::vector<int> rank_of_new;   // rank of each new gid (ascending: one range per rank)
  std::int64_t cut_old = -1;      // face-graph cut of the owners (-1 without owners)
  std::int64_t cut_new = 0;       // face-graph cut of the result
  std::int64_t cut_total = 0;     // total face-graph weight
  int moved = 0;                  // blocks not on their owner
  double cost_old = 0.0, cost_new = 0.0;  // bytes over the horizon (2.5 of the design)
};

// The partition of nb blocks over the ranks: a rank holds floor or ceil of every class
// min(f, W) (with heavy work: the class work within one block of the mean, and counts
// within ComputeLATCountBounds, then the sum over the window's tick phases of the busiest
// rank lowered by same-class swaps and single moves, TickBalance); among such partitions the cheapest in halo plus
// migration bytes of an owner-keeping repair and a machine-aligned recursive bisection.
// GIDs are numbered by (rank, Morton).
void BuildLATPartition(const LATPartitionRequest &req, LATPartitionResult &res);

// The largest per-rank block count of a LAT partition of these factors without owners:
// the capacity a static mesh reserves at construction.
int LATProjectedMaxBlocks(const int *factor, int nb, int nranks, int block_cap);

// Halo bytes per ghost cell and bytes per moved block of the modules of this mesh; a
// full-block transfer (a pure rebalance) carries ghost zones and the primitives too.
void LATTransferPrices(const Mesh *pm, bool full_block, double &halo_bytes,
                       double &move_bytes);

// Fill a LATPartitionRequest's mesh, window and price fields from the mesh.
void FillLATPartitionRequest(const Mesh *pm, bool full_block, LATPartitionRequest &req);

// Rank 0 prints how a LAT partition (rank r holds the blocks [cuts[r], cuts[r + 1]) of
// the GID list) spreads each class min(f, window), the busy fraction of each rank over
// the window under the measured work and, with stats, the cut and migration figures.
void PrintLATPartition(const float *clist, const int *lat_factor_list,
                       int lat_sync_factor, int window, int nb,
                       const std::vector<int> &cuts,
                       const LATPartitionResult *stats = nullptr);

#endif  // MESH_LAT_PARTITION_HPP_
