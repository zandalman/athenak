//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file lat_partition.cpp
//! \brief The LAT block->rank partition (BuildLATPartition) and the machine tree it is
//! aligned with (ProbeLATRankTree).
//!
//! The classes c = min(f, W) of a LAT ladder step on different ticks of a window of W
//! ticks, so every rank holds floor or ceil of every class (the quotas of
//! LATQuotasFor); that fixes how many blocks of each class a rank holds and leaves free
//! WHICH ones.  That freedom is spent on the halo: the block graph weighs every face by
//! the ghost cells it carries per window, and the partition is the cheaper, in halo bytes
//! until the next partition event plus migrated bytes, of
//!  - an owner-keeping repair: every block stays on its owner except the fewest that
//!    restore the quotas, chosen by their face weight to the receiver, and
//!  - a seed: a recursive bisection down the machine tree in which every split halves
//!    every class at once along one common coordinate, so the classes, which nest around
//!    the same centre, stay together on a rank,
//! each refined by same-class swaps between ranks that share faces.  Everything is a
//! deterministic serial function of data that is identical on every rank.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <queue>
#include <set>
#include <sstream>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

#include "athena.hpp"
#include "globals.hpp"
#include "mesh.hpp"
#include "hydro/hydro.hpp"
#include "mhd/mhd.hpp"
#include "radiation/radiation.hpp"
#include "z4c/z4c.hpp"

#if MPI_PARALLEL_ENABLED
#include <mpi.h>
#endif

//----------------------------------------------------------------------------------------
bool LATZLess(const LogicalLocation &la, const LogicalLocation &lb, bool multi_d,
              bool three_d) {
  auto child_index = [&](const LogicalLocation &loc, int depth) {
    const int shift = loc.level - depth - 1;
    const int ox = (shift >= 0) ? ((loc.lx1 >> shift) & 1) : 0;
    const int oy = (multi_d && shift >= 0) ? ((loc.lx2 >> shift) & 1) : 0;
    const int oz = (three_d && shift >= 0) ? ((loc.lx3 >> shift) & 1) : 0;
    return ox + (oy << 1) + (oz << 2);
  };
  const int min_level = std::min(la.level, lb.level);
  for (int depth=0; depth<min_level; ++depth) {
    const int ca = child_index(la, depth);
    const int cb = child_index(lb, depth);
    if (ca != cb) return ca < cb;
  }
  if (la.level != lb.level) return la.level < lb.level;
  if (la.lx1 != lb.lx1) return la.lx1 < lb.lx1;
  if (la.lx2 != lb.lx2) return la.lx2 < lb.lx2;
  return la.lx3 < lb.lx3;
}

//----------------------------------------------------------------------------------------
LATCountBounds ComputeLATCountBounds(int nb, int nranks, int block_cap) {
  const int avg_ceil = (nb + nranks - 1)/nranks;
  const int balance_slack = std::max(8, (avg_ceil + 19)/20);
  LATCountBounds bounds;
  bounds.balance_min = std::max(std::max(1, nb - block_cap*(nranks - 1)),
                                std::max(1, nb/nranks - balance_slack));
  bounds.balance_cap = std::min(block_cap, avg_ceil + balance_slack);
  return bounds;
}

//----------------------------------------------------------------------------------------
void LATRankTree::Split(const std::vector<int> &group, std::vector<int> &first,
                        std::vector<int> &second) const {
  first.clear();
  second.clear();
  const int n = static_cast<int>(group.size());
  if (n <= 1) {
    first = group;
    return;
  }
  auto node_of = [&](int r) {
    return (r < static_cast<int>(node.size())) ? node[r] : 0;
  };
  auto clique_of = [&](int r) {
    return (r < static_cast<int>(clique.size())) ? clique[r] : r;
  };
  bool many_nodes = false, many_cliques = false;
  for (int r : group) {
    many_nodes = many_nodes || (node_of(r) != node_of(group[0]));
    many_cliques = many_cliques || (clique_of(r) != clique_of(group[0]));
  }
  // units of the level at which the group first divides, in order of their lowest rank
  std::map<int, int> unit_index;
  std::vector<std::vector<int>> units;
  for (int r : group) {
    const int u = many_nodes ? node_of(r) : (many_cliques ? clique_of(r) : r);
    auto it = unit_index.find(u);
    if (it == unit_index.end()) {
      unit_index.emplace(u, static_cast<int>(units.size()));
      units.push_back({r});
    } else {
      units[it->second].push_back(r);
    }
  }
  std::vector<int> order(units.size());
  for (std::size_t k=0; k<units.size(); ++k) order[k] = static_cast<int>(k);
  std::sort(order.begin(), order.end(), [&](int a, int b) {
    return units[a].front() < units[b].front();
  });
  // the prefix of whole units nearest half of the ranks (ties: the shorter prefix)
  int best = 1, best_diff = std::numeric_limits<int>::max(), acc = 0;
  for (std::size_t k=1; k<order.size(); ++k) {
    acc += static_cast<int>(units[order[k - 1]].size());
    const int diff = std::abs(2*acc - n);
    if (diff < best_diff) {
      best_diff = diff;
      best = static_cast<int>(k);
    }
  }
  for (std::size_t k=0; k<order.size(); ++k) {
    std::vector<int> &part = (static_cast<int>(k) < best) ? first : second;
    part.insert(part.end(), units[order[k]].begin(), units[order[k]].end());
  }
  std::sort(first.begin(), first.end());
  std::sort(second.begin(), second.end());
}

std::string LATRankTree::Describe(int nranks) const {
  std::ostringstream os;
  std::map<int, std::map<int, std::vector<int>>> nodes;
  for (int r=0; r<nranks; ++r) {
    const int nd = (r < static_cast<int>(node.size())) ? node[r] : 0;
    const int cq = (r < static_cast<int>(clique.size())) ? clique[r] : r;
    nodes[nd][cq].push_back(r);
  }
  for (auto &nd : nodes) {
    if (nodes.size() > 1) os << "[";
    for (auto &cq : nd.second) {
      os << "{";
      for (std::size_t k=0; k<cq.second.size(); ++k) {
        if (k > 0) os << ",";
        os << cq.second[k];
      }
      os << "}";
    }
    if (nodes.size() > 1) os << "]";
  }
  os << (probed ? " (probed)" : " (flat)");
  return os.str();
}

//----------------------------------------------------------------------------------------
LATRankTree ProbeLATRankTree() {
  LATRankTree tree;
  const int nranks = global_variable::nranks;
  tree.node.assign(nranks, 0);
  tree.clique.resize(nranks);
  for (int r=0; r<nranks; ++r) tree.clique[r] = r;
#if MPI_PARALLEL_ENABLED
  if (nranks <= 1) return tree;
  const int me = global_variable::my_rank;
  MPI_Comm node_comm;
  MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, me, MPI_INFO_NULL,
                      &node_comm);
  int nn = 1, nme = 0;
  MPI_Comm_size(node_comm, &nn);
  MPI_Comm_rank(node_comm, &nme);
  std::vector<int> world_of(nn, 0);
  MPI_Allgather(&me, 1, MPI_INT, world_of.data(), 1, MPI_INT, node_comm);
  const int my_node = *std::min_element(world_of.begin(), world_of.end());
  int my_clique = me;
  int probed = 0;
  // The halo path this classifies runs between devices.  A host build exchanges
  // through shared memory, whose links differ by far less than the 4x below, so its
  // nodes are flat; timing them would only read the load of the machine.
  constexpr bool device_halos = !std::is_same<DevMemSpace, HostMemSpace>::value;
  if (device_halos && nn >= 3) {
    const int nbytes = 2 << 20;
    Kokkos::View<char*, DevMemSpace> buffer("lat_probe", nbytes);
    Kokkos::fence();
    std::vector<double> bandwidth(static_cast<std::size_t>(nn)*nn, 0.0);
    for (int i=0; i<nn; ++i) {
      for (int j=i+1; j<nn; ++j) {
        MPI_Barrier(node_comm);
        if (nme != i && nme != j) continue;
        const int peer = (nme == i) ? j : i;
        double best = std::numeric_limits<double>::max();
        for (int trip=0; trip<2; ++trip) {
          const double t0 = MPI_Wtime();
          if (nme == i) {
            MPI_Send(buffer.data(), nbytes, MPI_BYTE, peer, 0, node_comm);
            MPI_Recv(buffer.data(), nbytes, MPI_BYTE, peer, 0, node_comm,
                     MPI_STATUS_IGNORE);
          } else {
            MPI_Recv(buffer.data(), nbytes, MPI_BYTE, peer, 0, node_comm,
                     MPI_STATUS_IGNORE);
            MPI_Send(buffer.data(), nbytes, MPI_BYTE, peer, 0, node_comm);
          }
          best = std::min(best, MPI_Wtime() - t0);
        }
        if (nme == i) {
          bandwidth[static_cast<std::size_t>(i)*nn + j] =
              2.0*static_cast<double>(nbytes)/std::max(best, 1.0e-12);
        }
      }
    }
    // one contributor per entry: the sum is exact and the same on every rank of the node
    MPI_Allreduce(MPI_IN_PLACE, bandwidth.data(), nn*nn, MPI_DOUBLE, MPI_SUM, node_comm);
    std::vector<double> sorted;
    for (int i=0; i<nn; ++i) {
      for (int j=i+1; j<nn; ++j) {
        sorted.push_back(bandwidth[static_cast<std::size_t>(i)*nn + j]);
      }
    }
    std::sort(sorted.begin(), sorted.end());
    const double median = sorted[sorted.size()/2];
    // fast cliques: connected components of the links at least 4x the median
    std::vector<int> parent(nn);
    for (int i=0; i<nn; ++i) parent[i] = i;
    auto find = [&](int x) {
      while (parent[x] != x) x = parent[x] = parent[parent[x]];
      return x;
    };
    for (int i=0; i<nn; ++i) {
      for (int j=i+1; j<nn; ++j) {
        if (bandwidth[static_cast<std::size_t>(i)*nn + j] >= 4.0*median) {
          const int a = find(i), b = find(j);
          if (a != b) parent[std::max(a, b)] = std::min(a, b);
        }
      }
    }
    my_clique = me;
    for (int i=0; i<nn; ++i) {
      if (find(i) == find(nme)) my_clique = std::min(my_clique, world_of[i]);
    }
    probed = 1;
  }
  int mine[2] = {my_node, my_clique};
  std::vector<int> all(2*static_cast<std::size_t>(nranks), 0);
  MPI_Allgather(mine, 2, MPI_INT, all.data(), 2, MPI_INT, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &probed, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
  MPI_Comm_free(&node_comm);
  for (int r=0; r<nranks; ++r) {
    tree.node[r] = all[2*r];
    tree.clique[r] = all[2*r + 1];
  }
  tree.probed = (probed != 0);
#endif
  return tree;
}

//----------------------------------------------------------------------------------------
namespace {
using Weight = std::int64_t;
constexpr int kMaxSwapPasses = 8;
constexpr int kSwapCandidates = 8;  // top candidates of each side compared per swap

// The quotas q[c][r] of every class over the ranks: floor(n_c/R) each, and the n_c mod R
// extra blocks to the ranks already holding most of the class (held, by the owners; all
// zero without owners), ties to the rank with the fewest extras so far, then the lower
// rank.  A rank whose total would pass cap takes no extra; if a class then finds too few
// ranks, the extras of all classes rotate over the ranks, which keeps every total within
// one block of the others.
std::vector<std::vector<int>> LATQuotasFor(const std::vector<int> &n_c,
                                           const std::vector<std::vector<int>> &held,
                                           int nranks, int cap) {
  const int ncls = static_cast<int>(n_c.size());
  std::vector<std::vector<int>> quotas(ncls, std::vector<int>(nranks, 0));
  long long base_total = 0;
  for (int c=0; c<ncls; ++c) base_total += n_c[c]/nranks;
  std::vector<int> plus(nranks, 0), order(nranks);
  bool ok = true;
  for (int c=0; c<ncls && ok; ++c) {
    const int base = n_c[c]/nranks, extra = n_c[c] % nranks;
    for (int r=0; r<nranks; ++r) {
      quotas[c][r] = base;
      order[r] = r;
    }
    std::stable_sort(order.begin(), order.end(), [&](int a, int b) {
      const int ha = held.empty() ? 0 : held[c][a];
      const int hb = held.empty() ? 0 : held[c][b];
      if (ha != hb) return ha > hb;
      return plus[a] < plus[b];
    });
    int given = 0;
    for (int k=0; k<nranks && given<extra; ++k) {
      const int r = order[k];
      if (base_total + plus[r] + 1 > cap) continue;
      ++quotas[c][r];
      ++plus[r];
      ++given;
    }
    ok = (given == extra);
  }
  // a rank left without blocks (classes smaller than the rank count whose extras all go
  // to the same holders) takes the rotation too
  for (int r=0; r<nranks && ok; ++r) {
    int total = 0;
    for (int c=0; c<ncls; ++c) total += quotas[c][r];
    ok = (total > 0);
  }
  if (!ok) {
    int offset = 0;
    for (int c=0; c<ncls; ++c) {
      const int base = n_c[c]/nranks, extra = n_c[c] % nranks;
      for (int r=0; r<nranks; ++r) {
        const int slot = ((r - offset) % nranks + nranks) % nranks;
        quotas[c][r] = base + ((slot < extra) ? 1 : 0);
      }
      offset = (offset + extra) % nranks;
    }
  }
  return quotas;
}

// Heavy work: the class work of every label must stay within tol[c] of its target.  A
// swap may not take a label out of its band, nor further out of it.
struct LATBand {
  bool active = false;
  int nlabels = 0;
  std::vector<double> work, target, tol;
  double Excess(int c, int l) const {
    return work[static_cast<std::size_t>(c)*nlabels + l] -
           target[static_cast<std::size_t>(c)*nlabels + l];
  }
  // a (class c) goes p -> q and b (class c) q -> p, dw = work(a) - work(b)
  bool Allows(int c, int p, int q, double dw) const {
    if (!active) return true;
    auto ok = [&](int l, double delta) {
      const double before = Excess(c, l);
      return std::abs(before + delta) <= std::max(tol[c], std::abs(before));
    };
    return ok(p, -dw) && ok(q, dw);
  }
  void Apply(int c, int p, int q, double dw) {
    if (!active) return;
    work[static_cast<std::size_t>(c)*nlabels + p] -= dw;
    work[static_cast<std::size_t>(c)*nlabels + q] += dw;
  }
  // At the rank level single blocks may also move, as long as every rank's block count
  // stays within [count_min, count_max] (ComputeLATCountBounds); none when count is
  // empty.
  std::vector<int> count;
  int count_min = 0, count_max = 0;
};

class LATPartitioner {
 public:
  explicit LATPartitioner(const LATPartitionRequest &req);
  void Run(LATPartitionResult &res);

 private:
  const LATPartitionRequest &rq_;
  int nb_, nranks_, ncls_, ndim_, cap_;
  bool heavy_;
  double price_cut_, price_move_;
  std::vector<int> zpos_, by_z_, cls_, class_value_;
  std::vector<double> wk_, class_work_, class_wmax_;
  std::vector<int> begin_, nbr_;
  std::vector<Weight> w_;
  std::vector<std::vector<int>> quota_;
  std::vector<std::vector<std::int64_t>> keys_;
  std::vector<int> side_;

  Weight Between(int a, int b) const {
    for (int k=begin_[a]; k<begin_[a + 1]; ++k) {
      if (nbr_[k] == b) return w_[k];
    }
    return 0;
  }
  Weight Cut(const std::vector<int> &label) const;
  int Swap(std::vector<int> &label, int nlabels, const std::vector<int> &part, bool kway,
           double scale, double threshold, LATBand *band);
  void RepairBand(std::vector<int> &label, const std::vector<int> &part, int c,
                  LATBand &band);
  LATBand RankBand(const std::vector<int> &rank) const;
  void RepairClass(int c, std::vector<int> &rank);
  void RepairCounts(std::vector<int> &rank);
  void Bisect(const std::vector<int> &set, const std::vector<int> &group,
              std::vector<int> &rank);
  void TickBalance(std::vector<int> &rank) const;
  std::vector<int> Seed();
  std::vector<int> Incremental();
};

LATPartitioner::LATPartitioner(const LATPartitionRequest &req)
    : rq_(req), nb_(req.nb), nranks_(std::max(1, global_variable::nranks)), ncls_(1),
      ndim_(req.three_d ? 3 : (req.multi_d ? 2 : 1)), cap_(0), heavy_(false),
      price_cut_(0.0), price_move_(0.0) {
  const int nb = nb_;
  const int min_cap = (nb + nranks_ - 1)/nranks_;
  cap_ = (req.block_cap > 0) ? req.block_cap : min_cap;
  if (cap_ < min_cap || static_cast<long long>(cap_)*nranks_ < nb) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << "HD LAT load-balance cap is infeasible: cap=" << cap_
              << ", nranks=" << nranks_ << ", MeshBlocks=" << nb << std::endl;
    std::exit(EXIT_FAILURE);
  }
  // Morton order
  by_z_.resize(nb);
  for (int i=0; i<nb; ++i) by_z_[i] = i;
  std::stable_sort(by_z_.begin(), by_z_.end(), [&](int a, int b) {
    return LATZLess(req.lloc[a], req.lloc[b], req.multi_d, req.three_d);
  });
  zpos_.resize(nb);
  for (int k=0; k<nb; ++k) zpos_[by_z_[k]] = k;

  // classes min(f, W); one class for a regrid before any metadata
  const int window = (req.window > 1) ? req.window : std::numeric_limits<int>::max();
  std::vector<int> cval(nb, 1);
  for (int i=0; i<nb; ++i) {
    const int f = (req.factor != nullptr) ? std::max(1, req.factor[i]) : 1;
    cval[i] = std::min(f, window);
  }
  int wc = 1;
  for (int i=0; i<nb; ++i) wc = std::max(wc, cval[i]);
  cls_.assign(nb, 0);
  if (req.balance_classes) {
    std::vector<int> values(cval);
    std::sort(values.begin(), values.end());
    values.erase(std::unique(values.begin(), values.end()), values.end());
    ncls_ = static_cast<int>(values.size());
    class_value_ = values;
    for (int i=0; i<nb; ++i) {
      cls_[i] = static_cast<int>(std::lower_bound(values.begin(), values.end(), cval[i]) -
                                 values.begin());
    }
  }

  // work: 1 per block unless some block is heavy.  Before any metadata (a flat ladder)
  // the costs are the checkpoint's window costs, not per-step work, and are not read.
  wk_.assign(nb, 1.0);
  if (req.work != nullptr && req.balance_classes) {
    for (int i=0; i<nb; ++i) {
      heavy_ = heavy_ || (static_cast<double>(req.work[i]) - 1.0 > kLATHeavyExcess);
    }
    if (heavy_) {
      for (int i=0; i<nb; ++i) wk_[i] = std::max(0.0, static_cast<double>(req.work[i]));
    }
  }
  class_work_.assign(ncls_, 0.0);
  class_wmax_.assign(ncls_, 0.0);
  for (int k=0; k<nb; ++k) {
    const int i = by_z_[k];
    class_work_[cls_[i]] += wk_[i];
    class_wmax_[cls_[i]] = std::max(class_wmax_[cls_[i]], wk_[i]);
  }

  // Face graph of the blocks: the same-level block, else the coarser one, else the finer
  // ones touching the face.  A face carries per window h(a<-b) W/c_a + h(b<-a) W/c_b
  // ghost cells, h the cells of one ghost buffer (bvals/buffs_cc.cpp): n_t1 n_t2 ng on
  // one level, (n_t1/2)(n_t2/2) ng from a finer block, (n_t1/2 + ng)(n_t2/2 + ng) ng from
  // a coarser one.  Edges and corners are left out.
  std::unordered_map<LogicalLocationKey, int, LogicalLocationKeyHash> index_of;
  index_of.reserve(static_cast<std::size_t>(nb)*2);
  for (int i=0; i<nb; ++i) index_of.emplace(LogicalLocationKey(req.lloc[i]), i);
  const int nx[3] = {req.nx1, req.nx2, req.nx3};
  auto ghost_cells = [&](int dim, int level_to, int level_from) {
    Weight h = req.ng;
    for (int t=0; t<3; ++t) {
      if (t == dim) continue;
      const Weight n = nx[t];
      const Weight half = (n > 1) ? n/2 : 1;
      const Weight pad = (n > 1) ? req.ng : 0;
      if (level_to == level_from) {
        h *= n;
      } else if (level_to < level_from) {
        h *= half;
      } else {
        h *= half + pad;
      }
    }
    return h;
  };
  begin_.assign(nb + 1, 0);
  auto visit = [&](const int i, auto &&emit) {
    const LogicalLocation &loc = req.lloc[i];
    for (int dim=0; dim<ndim_; ++dim) {
      for (int sign=-1; sign<=1; sign+=2) {
        std::int32_t l[3] = {loc.lx1, loc.lx2, loc.lx3};
        l[dim] += sign;
        auto it = index_of.find(LogicalLocationKey(l[0], l[1], l[2], loc.level));
        if (it != index_of.end()) {
          emit(it->second, dim);
          continue;
        }
        if (loc.level > 0) {
          it = index_of.find(LogicalLocationKey(l[0] >> 1, l[1] >> 1, l[2] >> 1,
                                                loc.level - 1));
          if (it != index_of.end()) {
            emit(it->second, dim);
            continue;
          }
        }
        const int near = (sign > 0) ? 0 : 1;
        for (int c=0; c<(1 << ndim_); ++c) {
          if (((c >> dim) & 1) != near) continue;
          std::int32_t f[3] = {2*l[0], 2*l[1], 2*l[2]};
          for (int d=0; d<ndim_; ++d) f[d] += (c >> d) & 1;
          it = index_of.find(LogicalLocationKey(f[0], f[1], f[2], loc.level + 1));
          if (it != index_of.end()) emit(it->second, dim);
        }
      }
    }
  };
  for (int i=0; i<nb; ++i) {
    int n = 0;
    visit(i, [&n](const int, const int) { ++n; });
    begin_[i + 1] = begin_[i] + n;
  }
  nbr_.resize(begin_[nb]);
  w_.resize(begin_[nb]);
  for (int i=0; i<nb; ++i) {
    int n = begin_[i];
    visit(i, [&](const int j, const int dim) {
      const int li = req.lloc[i].level, lj = req.lloc[j].level;
      nbr_[n] = j;
      w_[n] = ghost_cells(dim, li, lj)*static_cast<Weight>(wc/cval[i]) +
              ghost_cells(dim, lj, li)*static_cast<Weight>(wc/cval[j]);
      ++n;
    });
  }

  // bisection keys: block centre on the finest lattice along each axis, then Morton
  int max_level = 0;
  for (int i=0; i<nb; ++i) max_level = std::max(max_level, req.lloc[i].level);
  keys_.assign(ndim_ + 1, std::vector<std::int64_t>(nb, 0));
  for (int i=0; i<nb; ++i) {
    const LogicalLocation &loc = req.lloc[i];
    const std::int64_t l[3] = {loc.lx1, loc.lx2, loc.lx3};
    for (int d=0; d<ndim_; ++d) {
      keys_[d][i] = (2*l[d] + 1) << (max_level - loc.level);
    }
    keys_[ndim_][i] = zpos_[i];
  }

  // quotas, the extras to the ranks already holding most of a class
  std::vector<int> n_c(ncls_, 0);
  for (int i=0; i<nb; ++i) ++n_c[cls_[i]];
  std::vector<std::vector<int>> held;
  if (req.owner != nullptr) {
    held.assign(ncls_, std::vector<int>(nranks_, 0));
    for (int i=0; i<nb; ++i) {
      if (req.owner[i] < 0 || req.owner[i] >= nranks_) {
        std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                  << std::endl << "LAT partition: block " << i
                  << " has invalid owner rank " << req.owner[i] << std::endl;
        std::exit(EXIT_FAILURE);
      }
      ++held[cls_[i]][req.owner[i]];
    }
  }
  quota_ = LATQuotasFor(n_c, held, nranks_, cap_);

  // prices: bytes over the horizon per unit of graph weight, and per moved block
  const double windows = (req.horizon > 0.0) ? req.horizon/static_cast<double>(wc) : 0.0;
  price_cut_ = static_cast<double>(std::max(1, req.exchanges))*windows*req.halo_bytes;
  price_move_ = req.move_bytes;
  side_.assign(nb, -1);
}

Weight LATPartitioner::Cut(const std::vector<int> &label) const {
  Weight cut = 0;
  for (int i=0; i<nb_; ++i) {
    for (int k=begin_[i]; k<begin_[i + 1]; ++k) {
      const int j = nbr_[k];
      if (i < j && label[i] >= 0 && label[j] >= 0 && label[i] != label[j]) cut += w_[k];
    }
  }
  return cut;
}

//----------------------------------------------------------------------------------------
// Same-class swaps between labels (ranks, or the two sides of a bisection); counts per
// class and label are kept.  Swapping a (on p) with b (on q) gains
//   g = w_q(a) - w_p(a) + w_p(b) - w_q(b) - 2 w(a,b),
// accepted when g > 0 and g*scale >= threshold.  Classes fine to coarse, label pairs that
// share a face in ascending order, candidates by gain (ties by Morton), each block moved
// at most once a pass; at most kMaxSwapPasses passes, or until a pass accepts nothing.
int LATPartitioner::Swap(std::vector<int> &label, int nlabels,
                         const std::vector<int> &part, bool kway, double scale,
                         double threshold, LATBand *band) {
  int total = 0;
  std::vector<char> locked(nb_, 0);
  std::vector<Weight> gain(nb_, 0);
  for (int pass=0; pass<kMaxSwapPasses; ++pass) {
    for (int x : part) locked[x] = 0;
    std::vector<std::vector<int>> members(static_cast<std::size_t>(ncls_)*nlabels);
    for (int x : part) {
      if (label[x] >= 0) {
        members[static_cast<std::size_t>(cls_[x])*nlabels + label[x]].push_back(x);
      }
    }
    std::vector<std::int64_t> pairs;
    if (kway) {
      for (int x : part) {
        for (int k=begin_[x]; k<begin_[x + 1]; ++k) {
          const int y = nbr_[k];
          if (label[x] >= 0 && label[y] >= 0 && label[x] < label[y]) {
            pairs.push_back(static_cast<std::int64_t>(label[x])*nlabels + label[y]);
          }
        }
      }
      std::sort(pairs.begin(), pairs.end());
      pairs.erase(std::unique(pairs.begin(), pairs.end()), pairs.end());
    } else {
      pairs.push_back(1);
    }
    int accepted = 0;
    for (int c=0; c<ncls_; ++c) {
      for (const std::int64_t pq : pairs) {
        const int p = static_cast<int>(pq/nlabels), q = static_cast<int>(pq % nlabels);
        // the sets hold (-gain, Morton) of the unlocked candidates of each side
        std::set<std::pair<Weight, int>> side_p, side_q;
        auto collect = [&](int from, int to, std::set<std::pair<Weight, int>> &set) {
          for (int x : members[static_cast<std::size_t>(c)*nlabels + from]) {
            if (label[x] != from || locked[x]) continue;
            Weight w_from = 0, w_to = 0;
            for (int k=begin_[x]; k<begin_[x + 1]; ++k) {
              const int l = label[nbr_[k]];
              if (l == from) w_from += w_[k];
              if (l == to) w_to += w_[k];
            }
            gain[x] = w_to - w_from;
            set.emplace(-gain[x], zpos_[x]);
          }
        };
        collect(p, q, side_p);
        collect(q, p, side_q);
        auto update = [&](int y, Weight delta) {
          auto &set = (label[y] == p) ? side_p : side_q;
          set.erase(std::make_pair(-gain[y], zpos_[y]));
          gain[y] += delta;
          set.emplace(-gain[y], zpos_[y]);
        };
        while (!side_p.empty() && !side_q.empty()) {
          int best_a = -1, best_b = -1;
          Weight best = 0;
          int na = 0;
          for (auto ia = side_p.begin(); ia != side_p.end() && na < kSwapCandidates;
               ++ia, ++na) {
            const int a = by_z_[ia->second];
            int nbq = 0;
            for (auto ib = side_q.begin(); ib != side_q.end() && nbq < kSwapCandidates;
                 ++ib, ++nbq) {
              const int b = by_z_[ib->second];
              if (band != nullptr && !band->Allows(c, p, q, wk_[a] - wk_[b])) continue;
              const Weight g = gain[a] + gain[b] - 2*Between(a, b);
              if (best_a < 0 || g > best) {
                best = g;
                best_a = a;
                best_b = b;
              }
            }
          }
          if (best_a < 0 || best <= 0 ||
              static_cast<double>(best)*scale < threshold) {
            break;
          }
          side_p.erase(std::make_pair(-gain[best_a], zpos_[best_a]));
          side_q.erase(std::make_pair(-gain[best_b], zpos_[best_b]));
          label[best_a] = q;
          label[best_b] = p;
          locked[best_a] = 1;
          locked[best_b] = 1;
          if (band != nullptr) band->Apply(c, p, q, wk_[best_a] - wk_[best_b]);
          // a left p for q, b left q for p: the gains of the candidates next to them
          for (int k=begin_[best_a]; k<begin_[best_a + 1]; ++k) {
            const int y = nbr_[k];
            if (locked[y] || cls_[y] != c || (label[y] != p && label[y] != q)) continue;
            update(y, (label[y] == p) ? 2*w_[k] : -2*w_[k]);
          }
          for (int k=begin_[best_b]; k<begin_[best_b + 1]; ++k) {
            const int y = nbr_[k];
            if (locked[y] || cls_[y] != c || (label[y] != p && label[y] != q)) continue;
            update(y, (label[y] == p) ? -2*w_[k] : 2*w_[k]);
          }
          ++accepted;
        }
      }
    }
    total += accepted;
    if (accepted == 0) break;
  }
  return total;
}

//----------------------------------------------------------------------------------------
// Heavy work: balance the class-c work of the labels between the label furthest above
// and the one furthest below their targets, by the same-class swap (or, at the rank
// level, the single move within the count bounds) that moves as near half their gap as
// the blocks allow, while one narrows the gap; at equal distance a swap goes first.  The
// band of the class then shrinks to the excess reached, never beyond one block (the
// largest block work of the class), so the swap refinement after it trades no balance
// for locality (the design's rule against a locality slack).  If nothing brings a label
// within one block the counts win.
void LATPartitioner::RepairBand(std::vector<int> &label, const std::vector<int> &part,
                                int c, LATBand &band) {
  const int nl = band.nlabels;
  std::vector<std::vector<int>> lists(nl);
  int n_c = 0;
  for (int x : part) {
    if (cls_[x] == c && label[x] >= 0) {
      lists[label[x]].push_back(x);
      ++n_c;
    }
  }
  for (int iter=0; iter<4*n_c + 16; ++iter) {
    int lmax = 0, lmin = 0;
    for (int l=1; l<nl; ++l) {
      if (band.Excess(c, l) > band.Excess(c, lmax)) lmax = l;
      if (band.Excess(c, l) < band.Excess(c, lmin)) lmin = l;
    }
    const double emax = band.Excess(c, lmax), emin = band.Excess(c, lmin);
    const double gap = emax - emin;
    if (!(gap > 0.0)) break;
    std::vector<int> &from = lists[lmax];
    std::vector<int> &to = lists[lmin];
    std::sort(to.begin(), to.end(), [&](int a, int b) {
      return (wk_[a] != wk_[b]) ? (wk_[a] < wk_[b]) : (zpos_[a] < zpos_[b]);
    });
    auto first_at_or_above = [&](double v) {
      return static_cast<int>(std::lower_bound(to.begin(), to.end(), v,
          [&](int x, double value) { return wk_[x] < value; }) - to.begin());
    };
    const bool can_move = !band.count.empty() &&
        band.count[lmax] - 1 >= band.count_min && band.count[lmin] + 1 <= band.count_max;
    // best (a, b): b = -1 is the move of a; order by miss, swap before move, Morton
    int best_a = -1, best_b = -1;
    double best_miss = 0.0;
    auto consider = [&](int a, int b, double dw) {
      if (!(dw > 0.0) || !(dw < gap)) return;
      const double miss = std::abs(dw - 0.5*gap);
      bool better = (best_a < 0 || miss < best_miss);
      if (!better && miss == best_miss) {
        const bool move = (b < 0), best_move = (best_b < 0);
        if (move != best_move) {
          better = !move;
        } else if (zpos_[a] != zpos_[best_a]) {
          better = zpos_[a] < zpos_[best_a];
        } else {
          better = !move && zpos_[b] < zpos_[best_b];
        }
      }
      if (better) {
        best_a = a;
        best_b = b;
        best_miss = miss;
      }
    };
    for (int a : from) {
      const double want = wk_[a] - 0.5*gap;
      const int above = first_at_or_above(want);
      if (above < static_cast<int>(to.size())) {
        consider(a, to[above], wk_[a] - wk_[to[above]]);
      }
      if (above > 0) {
        const int b = to[first_at_or_above(wk_[to[above - 1]])];
        consider(a, b, wk_[a] - wk_[b]);
      }
      if (can_move) consider(a, -1, wk_[a]);
    }
    if (best_a < 0) break;
    label[best_a] = lmin;
    from.erase(std::find(from.begin(), from.end(), best_a));
    to.push_back(best_a);
    if (best_b >= 0) {
      label[best_b] = lmax;
      band.Apply(c, lmax, lmin, wk_[best_a] - wk_[best_b]);
      to.erase(std::find(to.begin(), to.end(), best_b));
      from.push_back(best_b);
    } else {
      band.Apply(c, lmax, lmin, wk_[best_a]);
      --band.count[lmax];
      ++band.count[lmin];
    }
  }
  double reached = 0.0;
  for (int l=0; l<nl; ++l) reached = std::max(reached, std::abs(band.Excess(c, l)));
  band.tol[c] = std::min(band.tol[c], reached);
}

LATBand LATPartitioner::RankBand(const std::vector<int> &rank) const {
  LATBand band;
  band.active = heavy_;
  if (!heavy_) return band;
  band.nlabels = nranks_;
  band.work.assign(static_cast<std::size_t>(ncls_)*nranks_, 0.0);
  band.target.assign(static_cast<std::size_t>(ncls_)*nranks_, 0.0);
  band.tol = class_wmax_;
  for (int k=0; k<nb_; ++k) {
    const int i = by_z_[k];
    band.work[static_cast<std::size_t>(cls_[i])*nranks_ + rank[i]] += wk_[i];
  }
  const LATCountBounds bounds = ComputeLATCountBounds(nb_, nranks_, cap_);
  band.count.assign(nranks_, 0);
  for (int i=0; i<nb_; ++i) ++band.count[rank[i]];
  band.count_min = bounds.balance_min;
  band.count_max = bounds.balance_cap;
  for (int c=0; c<ncls_; ++c) {
    for (int r=0; r<nranks_; ++r) {
      band.target[static_cast<std::size_t>(c)*nranks_ + r] =
          class_work_[c]/static_cast<double>(nranks_);
    }
  }
  return band;
}

//----------------------------------------------------------------------------------------
// Owner-keeping repair of class c: while a rank holds more of the class than its quota,
// the class-c block on an over-quota rank with the largest face weight gained by moving
// to an under-quota rank (weight to the receiver minus weight to the giver; ties to the
// lower Morton key, then the lower receiver) moves.
void LATPartitioner::RepairClass(int c, std::vector<int> &rank) {
  std::vector<int> count(nranks_, 0);
  for (int i=0; i<nb_; ++i) {
    if (cls_[i] == c) ++count[rank[i]];
  }
  auto over = [&](int r) { return count[r] > quota_[c][r]; };
  auto under = [&](int r) { return count[r] < quota_[c][r]; };
  bool any_over = false;
  for (int r=0; r<nranks_; ++r) any_over = any_over || over(r);
  if (!any_over) return;
  struct Entry {
    Weight gain;
    int z, x, dest, version;
  };
  auto worse = [](const Entry &a, const Entry &b) {
    return (a.gain != b.gain) ? (a.gain < b.gain) : (a.z > b.z);
  };
  std::priority_queue<Entry, std::vector<Entry>, decltype(worse)> heap(worse);
  std::vector<int> version(nb_, 0);
  std::vector<std::pair<int, Weight>> to_rank;
  auto evaluate = [&](int x, Weight &best_gain, int &best_dest) {
    const int src = rank[x];
    to_rank.clear();
    Weight w_src = 0;
    for (int k=begin_[x]; k<begin_[x + 1]; ++k) {
      const int r = rank[nbr_[k]];
      if (r == src) {
        w_src += w_[k];
      } else {
        to_rank.emplace_back(r, w_[k]);
      }
    }
    best_dest = -1;
    for (int r=0; r<nranks_ && best_dest<0; ++r) {
      if (under(r)) best_dest = r;
    }
    best_gain = -w_src;
    std::sort(to_rank.begin(), to_rank.end());
    for (std::size_t k=0; k<to_rank.size();) {
      const int r = to_rank[k].first;
      Weight w = 0;
      for (; k<to_rank.size() && to_rank[k].first == r; ++k) w += to_rank[k].second;
      if (!under(r)) continue;
      if (w - w_src > best_gain || (w - w_src == best_gain && r < best_dest)) {
        best_gain = w - w_src;
        best_dest = r;
      }
    }
  };
  auto push = [&](int x) {
    Weight g;
    int d;
    evaluate(x, g, d);
    if (d >= 0) heap.push(Entry{g, zpos_[x], x, d, ++version[x]});
  };
  for (int k=0; k<nb_; ++k) {
    const int x = by_z_[k];
    if (cls_[x] == c && over(rank[x])) push(x);
  }
  while (!heap.empty()) {
    const Entry e = heap.top();
    heap.pop();
    if (e.version != version[e.x] || !over(rank[e.x])) continue;
    Weight g;
    int d;
    evaluate(e.x, g, d);
    if (d < 0) break;
    if (g != e.gain || d != e.dest) {
      heap.push(Entry{g, e.z, e.x, d, ++version[e.x]});
      continue;
    }
    --count[rank[e.x]];
    ++count[d];
    rank[e.x] = d;
    ++version[e.x];
    for (int k=begin_[e.x]; k<begin_[e.x + 1]; ++k) {
      const int y = nbr_[k];
      if (cls_[y] == c && over(rank[y])) push(y);
    }
  }
}

// A regrid before any LAT metadata: owners are kept and only a rank outside the count
// bounds gives blocks away, the one with the largest face-weight gain to the receiver.
void LATPartitioner::RepairCounts(std::vector<int> &rank) {
  const LATCountBounds bounds = ComputeLATCountBounds(nb_, nranks_, cap_);
  std::vector<int> count(nranks_, 0);
  for (int i=0; i<nb_; ++i) ++count[rank[i]];
  for (;;) {
    int src = 0, dst = 0;
    for (int r=1; r<nranks_; ++r) {
      if (count[r] > count[src]) src = r;
      if (count[r] < count[dst]) dst = r;
    }
    if (count[src] <= bounds.balance_cap && count[dst] >= bounds.balance_min) break;
    if (count[src] - count[dst] <= 1) break;
    int best = -1;
    Weight best_gain = 0;
    for (int k=0; k<nb_; ++k) {
      const int x = by_z_[k];
      if (rank[x] != src) continue;
      Weight g = 0;
      for (int j=begin_[x]; j<begin_[x + 1]; ++j) {
        const int r = rank[nbr_[j]];
        if (r == dst) g += w_[j];
        if (r == src) g -= w_[j];
      }
      if (best < 0 || g > best_gain) {
        best = x;
        best_gain = g;
      }
    }
    rank[best] = dst;
    --count[src];
    ++count[dst];
  }
}

//----------------------------------------------------------------------------------------
// Heavy work: balance the critical path over the window's tick phases across classes.
// The per-class repair equalises every class on its own, but the classes step on
// different ticks, and a tick pays the most loaded rank of the classes due on it.  With
// the work W[c][r] of class c on rank r, the window costs (exactly ScoreLATPartition's
// model, sync = the largest class)
//   J = sum over terms t of m_t max_r S_t[r],   S_t[r] = sum over classes with
//   level(c) <= t of W[c][r],
// the terms t = 0 .. T-2 being the m_t = sync/(2 2^t) ticks that advance exactly the
// classes up to level t and t = T-1 the one tick that advances all.  A same-class swap
// of a (on p) and b (on q), or a single move of a, shifts dw = w_a - w_b (w_a) of class
// work from p to q, which changes only S_t[p] by -dw and S_t[q] by +dw for the terms
// that include c; J(dw) is convex and piecewise linear in dw.  Until none is left, the
// move with the lowest J goes, among swaps and (within ComputeLATCountBounds) single
// moves from a rank attaining a term maximum (no other rank can lower one) that keep
// every class within the band LATBand::Allows (one block of the target); it must lower
// J by more than 64 eps (relative).  For each (p, q, c) the dw minimising J on the band
// interval is found by ternary search, and each block a of p takes the blocks of q whose
// work brackets w_a - dw (convexity: the best discrete partner is one of the two).
// Ties in J go to the larger face-weight gain, then to the lower Morton positions.  At
// most 4 nb moves; every rank holds the same inputs, no communication.
void LATPartitioner::TickBalance(std::vector<int> &rank) const {
  if (!heavy_ || nranks_ < 2) return;
  const int nr = nranks_, nc = ncls_;
  const int sync = class_value_.back();
  int nterm = 1;
  for (long long f=1; f<sync; f*=2) ++nterm;
  std::vector<int> level(nc, 0);
  for (int c=0; c<nc; ++c) {
    int f = class_value_[c], l = 0;
    while (f > 1 && l < nterm - 1) {
      f >>= 1;
      ++l;
    }
    level[c] = l;
  }
  std::vector<double> mult(nterm, 1.0);
  for (int t=0, f=1; t<nterm - 1; ++t, f*=2) mult[t] = static_cast<double>(sync/(2*f));
  const double tol_j = 64.0*std::numeric_limits<double>::epsilon();

  std::vector<double> cum(static_cast<std::size_t>(nterm)*nr), tmax(nterm), m_out(nterm);
  std::vector<int> order(static_cast<std::size_t>(nterm)*nr);
  std::vector<std::vector<int>> lists(static_cast<std::size_t>(nc)*nr);
  auto face_gain = [&](int x, int from, int to) {
    Weight g = 0;
    for (int k=begin_[x]; k<begin_[x + 1]; ++k) {
      const int l = rank[nbr_[k]];
      if (l == to) g += w_[k];
      if (l == from) g -= w_[k];
    }
    return g;
  };
  // the blocks of every (class, rank) by (work, Morton), kept sorted across the moves
  auto by_work = [&](int a, int b) {
    return (wk_[a] != wk_[b]) ? (wk_[a] < wk_[b]) : (zpos_[a] < zpos_[b]);
  };
  for (int k=0; k<nb_; ++k) {
    const int x = by_z_[k];
    lists[static_cast<std::size_t>(cls_[x])*nr + rank[x]].push_back(x);
  }
  for (auto &l : lists) std::sort(l.begin(), l.end(), by_work);
  auto relocate = [&](int x, int from, int to) {
    std::vector<int> &l = lists[static_cast<std::size_t>(cls_[x])*nr + from];
    l.erase(std::lower_bound(l.begin(), l.end(), x, by_work));
    std::vector<int> &m = lists[static_cast<std::size_t>(cls_[x])*nr + to];
    m.insert(std::lower_bound(m.begin(), m.end(), x, by_work), x);
    rank[x] = to;
  };
  for (int iter=0; iter<4*nb_; ++iter) {
    LATBand band = RankBand(rank);
    std::fill(cum.begin(), cum.end(), 0.0);
    for (int t=0; t<nterm; ++t) {
      for (int c=0; c<nc; ++c) {
        if (level[c] > t) continue;
        for (int r=0; r<nr; ++r) {
          cum[static_cast<std::size_t>(t)*nr + r] += band.work[static_cast<std::size_t>(c)*nr + r];
        }
      }
    }
    double j0 = 0.0;
    for (int t=0; t<nterm; ++t) {
      int *o = &order[static_cast<std::size_t>(t)*nr];
      for (int r=0; r<nr; ++r) o[r] = r;
      const double *s = &cum[static_cast<std::size_t>(t)*nr];
      std::stable_sort(o, o + nr, [&](int a, int b) { return s[a] > s[b]; });
      tmax[t] = s[o[0]];
      j0 += mult[t]*tmax[t];
    }
    const double j_floor = j0 - tol_j*j0;
    int best_a = -1, best_b = -1, best_q = -1;
    double best_j = 0.0;
    Weight best_gain = 0;
    auto j_with = [&](int c, int p, int q, double dw) {
      double j = 0.0;
      for (int t=0; t<nterm; ++t) {
        double v = m_out[t];
        const double sp = cum[static_cast<std::size_t>(t)*nr + p];
        const double sq = cum[static_cast<std::size_t>(t)*nr + q];
        if (level[c] <= t) {
          v = std::max(v, std::max(sp - dw, sq + dw));
        } else {
          v = std::max(v, std::max(sp, sq));
        }
        j += mult[t]*v;
      }
      return j;
    };
    auto consider = [&](int a, int b, int p, int q, double jn) {
      if (!(jn < j_floor)) return;
      if (best_a >= 0 && jn > best_j + tol_j*best_j) return;
      const Weight g = face_gain(a, p, q) +
                       ((b >= 0) ? face_gain(b, q, p) - 2*Between(a, b) : 0);
      bool better = (best_a < 0) || (jn < best_j - tol_j*best_j);
      if (!better) {
        if (g != best_gain) {
          better = g > best_gain;
        } else if (zpos_[a] != zpos_[best_a]) {
          better = zpos_[a] < zpos_[best_a];
        } else {
          better = (b >= 0 && best_b >= 0) ? zpos_[b] < zpos_[best_b] : (b >= 0);
        }
      }
      if (better) {
        best_a = a;
        best_b = b;
        best_q = q;
        best_j = jn;
        best_gain = g;
      }
    };
    for (int p=0; p<nr; ++p) {
      bool critical = false;
      for (int t=0; t<nterm; ++t) {
        critical = critical || (cum[static_cast<std::size_t>(t)*nr + p] >= tmax[t]);
      }
      if (!critical) continue;
      for (int q=0; q<nr; ++q) {
        if (q == p) continue;
        for (int t=0; t<nterm; ++t) {
          const int *o = &order[static_cast<std::size_t>(t)*nr];
          int k = 0;
          while (k < nr && (o[k] == p || o[k] == q)) ++k;
          m_out[t] = (k < nr) ? cum[static_cast<std::size_t>(t)*nr + o[k]] : 0.0;
        }
        for (int c=0; c<nc; ++c) {
          const std::vector<int> &from = lists[static_cast<std::size_t>(c)*nr + p];
          const std::vector<int> &to = lists[static_cast<std::size_t>(c)*nr + q];
          if (from.empty()) continue;
          // dw allowed by the band of both ranks
          const double ep = band.Excess(c, p), eq = band.Excess(c, q);
          const double mp = std::max(band.tol[c], std::abs(ep));
          const double mq = std::max(band.tol[c], std::abs(eq));
          const double lo = std::max(ep - mp, -eq - mq), hi = std::min(ep + mp, -eq + mq);
          if (!(lo <= hi)) continue;
          const double wmax = class_wmax_[c];
          double a_lo = std::max(lo, -wmax), a_hi = std::min(hi, wmax);
          for (int it=0; it<80 && a_lo < a_hi; ++it) {
            const double m1 = a_lo + (a_hi - a_lo)/3.0, m2 = a_hi - (a_hi - a_lo)/3.0;
            if (j_with(c, p, q, m1) <= j_with(c, p, q, m2)) {
              a_hi = m2;
            } else {
              a_lo = m1;
            }
          }
          const double dstar = 0.5*(a_lo + a_hi);
          const bool can_move = !band.count.empty() &&
              band.count[p] - 1 >= band.count_min && band.count[q] + 1 <= band.count_max;
          for (int a : from) {
            const double wa = wk_[a];
            if (can_move && wa >= lo && wa <= hi && band.Allows(c, p, q, wa)) {
              consider(a, -1, p, q, j_with(c, p, q, wa));
            }
            const int i = static_cast<int>(std::lower_bound(to.begin(), to.end(), wa - dstar,
                [&](int x, double v) { return wk_[x] < v; }) - to.begin());
            for (int ib=i - 1; ib<=i; ++ib) {
              if (ib < 0 || ib >= static_cast<int>(to.size())) continue;
              const int b = to[ib];
              const double dw = wa - wk_[b];
              if (dw < lo || dw > hi || !band.Allows(c, p, q, dw)) continue;
              consider(a, b, p, q, j_with(c, p, q, dw));
            }
          }
        }
      }
    }
    if (best_a < 0) break;
    const int p = rank[best_a];
    relocate(best_a, p, best_q);
    if (best_b >= 0) relocate(best_b, best_q, p);
  }
}

std::vector<int> LATPartitioner::Incremental() {
  std::vector<int> rank(rq_.owner, rq_.owner + nb_);
  if (rq_.balance_classes) {
    for (int c=0; c<ncls_; ++c) RepairClass(c, rank);
  } else {
    RepairCounts(rank);
  }
  LATBand band = RankBand(rank);
  if (heavy_) {
    for (int c=0; c<ncls_; ++c) RepairBand(rank, by_z_, c, band);
  }
  Swap(rank, nranks_, by_z_, true, price_cut_, 2.0*price_move_,
       heavy_ ? &band : nullptr);
  TickBalance(rank);
  return rank;
}

//----------------------------------------------------------------------------------------
// Recursive bisection of a block set over a rank group, down the machine tree.  Each
// split gives the first part exactly the quotas of its ranks in every class, taking the
// blocks of each class lowest (or highest) along one key: the block centre along x1, x2
// or x3, or the Morton rank.  Of these candidates the least cost wins (cut, and with
// owners also the moved blocks), ties to the lower candidate; one 2-way swap refinement
// follows.
void LATPartitioner::Bisect(const std::vector<int> &set, const std::vector<int> &group,
                            std::vector<int> &rank) {
  if (group.size() <= 1) {
    for (int x : set) rank[x] = group.empty() ? 0 : group[0];
    return;
  }
  std::vector<int> first, second;
  const LATRankTree flat;
  (rq_.tree != nullptr ? *rq_.tree : flat).Split(group, first, second);
  std::vector<int> target(ncls_, 0);
  std::vector<char> in_first(nranks_, 0), in_group(nranks_, 0);
  for (int r : first) {
    in_first[r] = 1;
    for (int c=0; c<ncls_; ++c) target[c] += quota_[c][r];
  }
  for (int r : group) in_group[r] = 1;
  std::vector<std::vector<int>> members(ncls_);
  for (int x : set) members[cls_[x]].push_back(x);

  const int ncand = 2*(ndim_ + 1);
  int best = -1;
  double best_cost = 0.0;
  std::vector<int> best_side(set.size(), 1);
  for (int cand=0; cand<ncand; ++cand) {
    const std::vector<std::int64_t> &key = keys_[cand/2];
    const bool high = (cand % 2) == 1;
    auto less = [&](int a, int b) {
      return (key[a] != key[b]) ? (key[a] < key[b]) : (zpos_[a] < zpos_[b]);
    };
    for (int c=0; c<ncls_; ++c) {
      std::vector<int> v(members[c]);
      const int n = static_cast<int>(v.size());
      const int t = std::min(target[c], n);
      const int at = high ? (n - t) : t;
      if (at > 0 && at < n) std::nth_element(v.begin(), v.begin() + at, v.end(), less);
      for (int k=0; k<n; ++k) {
        side_[v[k]] = high ? ((k >= at) ? 0 : 1) : ((k < at) ? 0 : 1);
      }
    }
    Weight cut = 0;
    int moved = 0;
    for (int x : set) {
      for (int k=begin_[x]; k<begin_[x + 1]; ++k) {
        const int y = nbr_[k];
        if (x < y && side_[y] >= 0 && side_[y] != side_[x]) cut += w_[k];
      }
      if (rq_.owner != nullptr && in_group[rq_.owner[x]]) {
        moved += (side_[x] != (in_first[rq_.owner[x]] ? 0 : 1)) ? 1 : 0;
      }
    }
    const double cost = (rq_.owner != nullptr) ?
        static_cast<double>(cut)*price_cut_ + static_cast<double>(moved)*price_move_ :
        static_cast<double>(cut);
    if (best < 0 || cost < best_cost) {
      best = cand;
      best_cost = cost;
      for (std::size_t k=0; k<set.size(); ++k) best_side[k] = side_[set[k]];
    }
  }
  for (std::size_t k=0; k<set.size(); ++k) side_[set[k]] = best_side[k];

  LATBand band;
  if (heavy_) {
    band.active = true;
    band.nlabels = 2;
    band.work.assign(2*static_cast<std::size_t>(ncls_), 0.0);
    band.target.assign(2*static_cast<std::size_t>(ncls_), 0.0);
    band.tol = class_wmax_;
    std::vector<double> total(ncls_, 0.0);
    for (int x : set) {
      band.work[2*static_cast<std::size_t>(cls_[x]) + side_[x]] += wk_[x];
      total[cls_[x]] += wk_[x];
    }
    const double share = static_cast<double>(first.size())/
                         static_cast<double>(group.size());
    for (int c=0; c<ncls_; ++c) {
      band.target[2*c] = total[c]*share;
      band.target[2*c + 1] = total[c] - band.target[2*c];
      RepairBand(side_, set, c, band);
    }
  }
  Swap(side_, 2, set, false, 1.0, 0.0, heavy_ ? &band : nullptr);

  std::vector<int> set_first, set_second;
  for (int x : set) (side_[x] == 0 ? set_first : set_second).push_back(x);
  for (int x : set) side_[x] = -1;
  Bisect(set_first, first, rank);
  Bisect(set_second, second, rank);
}

std::vector<int> LATPartitioner::Seed() {
  std::vector<int> rank(nb_, 0), group(nranks_);
  for (int r=0; r<nranks_; ++r) group[r] = r;
  Bisect(by_z_, group, rank);
  LATBand band = RankBand(rank);
  if (heavy_) {
    for (int c=0; c<ncls_; ++c) RepairBand(rank, by_z_, c, band);
  }
  Swap(rank, nranks_, by_z_, true, 1.0, 0.0, heavy_ ? &band : nullptr);
  TickBalance(rank);
  return rank;
}

//----------------------------------------------------------------------------------------
void LATPartitioner::Run(LATPartitionResult &res) {
  res = LATPartitionResult();
  for (int i=0; i<nb_; ++i) {
    for (int k=begin_[i]; k<begin_[i + 1]; ++k) {
      if (i < nbr_[k]) res.cut_total += w_[k];
    }
  }
  std::vector<int> rank = Seed();
  const Weight cut_seed = Cut(rank);
  res.cut_new = cut_seed;
  if (rq_.owner != nullptr) {
    const std::vector<int> owner(rq_.owner, rq_.owner + nb_);
    auto moved = [&](const std::vector<int> &r) {
      int n = 0;
      for (int i=0; i<nb_; ++i) n += (r[i] != owner[i]) ? 1 : 0;
      return n;
    };
    std::vector<int> incr = Incremental();
    const Weight cut_incr = Cut(incr);
    const int moved_seed = moved(rank), moved_incr = moved(incr);
    const double cost_seed = static_cast<double>(cut_seed)*price_cut_ +
                             static_cast<double>(moved_seed)*price_move_;
    const double cost_incr = static_cast<double>(cut_incr)*price_cut_ +
                             static_cast<double>(moved_incr)*price_move_;
    // the seed migrates more: take it only when it is cheaper over the horizon AND cuts
    // the halo by the same relative margin as any other LAT migration decision
    const bool take_seed = cost_seed < cost_incr &&
        static_cast<double>(cut_seed) <=
        (1.0 - Mesh::kHydroLATRebalanceMinRelativeGain)*static_cast<double>(cut_incr);
    if (!take_seed) {
      rank.swap(incr);
      res.cut_new = cut_incr;
    }
    res.moved = take_seed ? moved_seed : moved_incr;
    res.cut_old = Cut(owner);
    res.cost_old = static_cast<double>(res.cut_old)*price_cut_;
    res.cost_new = take_seed ? cost_seed : cost_incr;
  } else {
    res.cost_new = static_cast<double>(cut_seed)*price_cut_;
  }

  // GIDs by (rank, Morton): every rank holds one contiguous range
  res.new_to_input.resize(nb_);
  for (int i=0; i<nb_; ++i) res.new_to_input[i] = i;
  std::sort(res.new_to_input.begin(), res.new_to_input.end(), [&](int a, int b) {
    return (rank[a] != rank[b]) ? (rank[a] < rank[b]) : (zpos_[a] < zpos_[b]);
  });
  res.rank_of_new.resize(nb_);
  for (int g=0; g<nb_; ++g) res.rank_of_new[g] = rank[res.new_to_input[g]];
  std::vector<int> count(nranks_, 0);
  for (int g=0; g<nb_; ++g) ++count[res.rank_of_new[g]];
  for (int r=0; r<nranks_; ++r) {
    if (count[r] <= 0 || count[r] > cap_) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "HD LAT partition gives rank " << r << " " << count[r]
                << " MeshBlocks; cap=" << cap_ << std::endl;
      std::exit(EXIT_FAILURE);
    }
  }
}
} // namespace

//----------------------------------------------------------------------------------------
void BuildLATPartition(const LATPartitionRequest &req, LATPartitionResult &res) {
  if (req.lloc == nullptr || req.nb <= 0) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << "BuildLATPartition called without blocks" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  LATPartitioner partitioner(req);
  partitioner.Run(res);
}

int LATProjectedMaxBlocks(const int *factor, int nb, int nranks, int block_cap) {
  if (nb <= 0 || nranks <= 0) return 0;
  std::map<int, int> classes;
  for (int i=0; i<nb; ++i) {
    ++classes[(factor != nullptr) ? std::max(1, factor[i]) : 1];
  }
  std::vector<int> n_c;
  for (auto &entry : classes) n_c.push_back(entry.second);
  const int cap = (block_cap > 0) ? block_cap : (nb + nranks - 1)/nranks;
  const std::vector<std::vector<int>> quotas = LATQuotasFor(n_c, {}, nranks, cap);
  int most = 0;
  for (int r=0; r<nranks; ++r) {
    int total = 0;
    for (const auto &q : quotas) total += q[r];
    most = std::max(most, total);
  }
  return most;
}

//----------------------------------------------------------------------------------------
void LATTransferPrices(const Mesh *pm, bool full_block, double &halo_bytes,
                       double &move_bytes) {
  double nhalo = 0.0;
  double nmove = 0.0;
  const MeshBlockPack *pack = pm->pmb_pack;
  if (pack != nullptr) {
    if (pack->phydro != nullptr) nhalo += pack->phydro->nvars;
    if (pack->pmhd != nullptr) {
      nhalo += pack->pmhd->nvars;
      if (pack->pmhd->HasField()) nhalo += 3.0;
    }
    if (pack->prad != nullptr) nhalo += pack->prad->prgeo->nangles;
    if (pack->pz4c != nullptr) nhalo += pack->pz4c->nz4c;
  }
  nhalo = std::max(1.0, nhalo);
  nmove = nhalo;
  if (pm->pmr != nullptr) {
    if (pack != nullptr && pack->pgrav != nullptr && pm->pmr->lb_transfer_gravity) {
      nmove += 1.0;
    }
    if (full_block) nmove += pm->pmr->FullBlockTransferPrimitiveCount();
  }
  const RegionIndcs &indcs = pm->mb_indcs;
  double cells = static_cast<double>(indcs.nx1)*indcs.nx2*indcs.nx3;
  if (full_block) {
    cells = static_cast<double>(indcs.nx1 + 2*indcs.ng)*
            (pm->multi_d ? indcs.nx2 + 2*indcs.ng : 1)*
            (pm->three_d ? indcs.nx3 + 2*indcs.ng : 1);
  }
  halo_bytes = nhalo*sizeof(Real);
  move_bytes = cells*nmove*sizeof(Real);
}

void FillLATPartitionRequest(const Mesh *pm, bool full_block, LATPartitionRequest &req) {
  req.multi_d = pm->multi_d;
  req.three_d = pm->three_d;
  req.nx1 = pm->mb_indcs.nx1;
  req.nx2 = pm->mb_indcs.nx2;
  req.nx3 = pm->mb_indcs.nx3;
  req.ng = pm->mb_indcs.ng;
  req.window = pm->hydro_lat_window_ticks;
  req.horizon = pm->hydro_lat_partition_horizon;
  req.exchanges = pm->hydro_lat_exchanges_per_step;
  req.tree = &pm->lat_rank_tree;
  LATTransferPrices(pm, full_block, req.halo_bytes, req.move_bytes);
}

//----------------------------------------------------------------------------------------
void PrintLATPartition(const float *clist, const int *lat_factor_list,
                       int lat_sync_factor, int window, int nb,
                       const std::vector<int> &cuts, const LATPartitionResult *stats) {
  if (global_variable::my_rank != 0) return;
  const int nranks = global_variable::nranks;
  const int sync = std::max(1, lat_sync_factor);
  // the classes step in a window of wticks ticks: min(f, W)
  const int wticks = (window > 1) ? std::min(window, sync) : sync;
  std::vector<int> factor_values;
  for (int factor=1; factor<=wticks; factor*=2) {
    factor_values.push_back(factor);
    if (factor > (std::numeric_limits<int>::max()/2)) break;
  }
  const int nfactors = static_cast<int>(factor_values.size());
  auto class_of = [&](int i) {
    return std::min(std::max(1, lat_factor_list[i]), wticks);
  };
  std::vector<int> prefix(static_cast<std::size_t>(nfactors)*(nb + 1), 0);
  for (int i=0; i<nb; ++i) {
    for (int n=0; n<nfactors; ++n) {
      prefix[n*(nb + 1) + i + 1] = prefix[n*(nb + 1) + i] +
                                   ((class_of(i) == factor_values[n]) ? 1 : 0);
    }
  }
  std::vector<double> due_prefix(static_cast<std::size_t>(nfactors)*(nb + 1), 0.0);
  std::vector<int> due_multiplicity(nfactors, 1);
  for (int phase=0; phase<nfactors - 1; ++phase) {
    due_multiplicity[phase] = wticks/(2*factor_values[phase]);
  }
  for (int phase=0; phase<nfactors; ++phase) {
    for (int i=0; i<nb; ++i) {
      // per-step work: the window cost over the steps per window of the raw ladder
      const int factor = std::max(1, lat_factor_list[i]);
      const double active_cost = static_cast<double>(clist[i])*
          static_cast<double>(factor)/static_cast<double>(sync);
      due_prefix[phase*(nb + 1) + i + 1] = due_prefix[phase*(nb + 1) + i] +
          ((class_of(i) <= factor_values[phase]) ? active_cost : 0.0);
    }
  }
  std::ostringstream os;
  os << "Mesh: HD LAT per-bin rank distribution";
  for (int n=0; n<nfactors; ++n) {
    const int total_in_bin = prefix[n*(nb + 1) + nb];
    if (total_in_bin <= 0) continue;
    os << " f" << factor_values[n] << "=[";
    for (int rank=0; rank<nranks; ++rank) {
      if (rank > 0) os << ",";
      os << (prefix[n*(nb + 1) + cuts[rank + 1]] - prefix[n*(nb + 1) + cuts[rank]]);
    }
    os << "]";
  }
  double sum_tick_max_work = 0.0;
  for (int phase=0; phase<nfactors; ++phase) {
    double tick_max = 0.0;
    for (int r=0; r<nranks; ++r) {
      tick_max = std::max(tick_max, due_prefix[phase*(nb + 1) + cuts[r + 1]] -
                                    due_prefix[phase*(nb + 1) + cuts[r]]);
    }
    sum_tick_max_work += due_multiplicity[phase]*tick_max;
  }
  if (sum_tick_max_work > 0.0) {
    os << " util%=[";
    for (int rank=0; rank<nranks; ++rank) {
      double rank_total = 0.0;
      for (int phase=0; phase<nfactors; ++phase) {
        rank_total += due_multiplicity[phase]*
            (due_prefix[phase*(nb + 1) + cuts[rank + 1]] -
             due_prefix[phase*(nb + 1) + cuts[rank]]);
      }
      if (rank > 0) os << ",";
      os << static_cast<int>(100.0*rank_total/sum_tick_max_work + 0.5);
    }
    os << "]";
  }
  if (stats != nullptr && stats->cut_total > 0) {
    const double total = static_cast<double>(stats->cut_total);
    os << std::fixed << std::setprecision(1) << " cut%=";
    if (stats->cut_old >= 0) {
      os << 100.0*static_cast<double>(stats->cut_old)/total;
    } else {
      os << "-";
    }
    os << "->" << 100.0*static_cast<double>(stats->cut_new)/total
       << " moved=" << stats->moved;
  }
  std::cout << os.str() << std::endl;
}
