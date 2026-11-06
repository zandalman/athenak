//========================================================================================
// AthenaK astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file remap_expand.cpp
//! \brief MeshBlock-tree inheritance from a <remap> source and automatic domain
//! expansion.  The semantics are specified in remap.hpp.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <set>
#include <sstream>
#include <string>
#include <vector>

#include "athena.hpp"
#include "globals.hpp"
#include "parameter_input.hpp"
#include "bvals/bvals.hpp"
#include "mesh/mesh.hpp"
#include "mesh/mesh_refinement.hpp"
#include "mesh/meshblock_pack.hpp"
#include "eos/eos.hpp"
#include "hydro/hydro.hpp"
#include "outputs/io_wrapper.hpp"
#include "outputs/outputs.hpp"
#include "pgen/pgen.hpp"
#include "remap/remap.hpp"
#include "remap/remap_impl.hpp"

#if MPI_PARALLEL_ENABLED
#include <mpi.h>
#endif

namespace remap {

namespace {

// The largest density of the active cells of the mesh, over all ranks (hydro only).
Real MeshDensityMax(Mesh *pm) {
  auto &u0 = pm->pmb_pack->phydro->u0;
  auto &indcs = pm->mb_indcs;
  const int nmb = pm->pmb_pack->nmb_thispack;
  Real rho_max = 0.0;
  Kokkos::parallel_reduce("remap_density_max",
      Kokkos::MDRangePolicy<DevExeSpace, Kokkos::Rank<4>>(
          {0, indcs.ks, indcs.js, indcs.is},
          {nmb, indcs.ke + 1, indcs.je + 1, indcs.ie + 1}),
      KOKKOS_LAMBDA(const int m, const int k, const int j, const int i, Real &mx) {
        mx = (u0(m, IDN, k, j, i) > mx) ? u0(m, IDN, k, j, i) : mx;
      }, Kokkos::Max<Real>(rho_max));
#if MPI_PARALLEL_ENABLED
  MPI_Allreduce(MPI_IN_PLACE, &rho_max, 1, MPI_ATHENA_REAL, MPI_MAX, MPI_COMM_WORLD);
#endif
  return rho_max;
}

[[noreturn]] void FatalExpand(const std::string &msg) {
  std::cout << "### FATAL ERROR in remap module (tree inheritance / auto-expansion)"
            << std::endl << msg << std::endl;
  std::exit(EXIT_FAILURE);
}

const char *const kFaceName[6] = {"x1min", "x1max", "x2min", "x2max", "x3min", "x3max"};
const char *const kAxisName[3] = {"x1", "x2", "x3"};

//----------------------------------------------------------------------------------------
// The mesh header and the MeshBlock list of a restart file, read exactly as
// LoadRemapSourceData does (remap_load.cpp): every rank opens the file and loads the
// parameter dump, rank 0 reads the binary sections and broadcasts them.

struct RestartTree {
  int nmb_total = 0;
  int root_level = 0;
  RegionSize mesh_size{};
  RegionIndcs mesh_indcs{};
  RegionIndcs mb_indcs{};
  Real time = 0.0;
  int ncycle = 0;
  std::vector<LogicalLocation> lloc;
};

void ReadRestartTree(const std::string &path, const bool read_list, RestartTree &rt) {
  IOWrapper input;
  input.Open(path.c_str(), IOWrapper::FileMode::read, false);
  ParameterInput src_pin;
  src_pin.LoadFromFile(input, false);

  const IOWrapperSizeT headersize = 3*sizeof(int) + 2*sizeof(Real)
                                  + sizeof(RegionSize) + 2*sizeof(RegionIndcs);
  std::vector<char> header(headersize);
  if (global_variable::my_rank == 0) {
    if (input.Read_bytes(header.data(), 1, headersize, false) != headersize) {
      FatalExpand("Failed to read the mesh header of restart file " + path + ".");
    }
  }
#if MPI_PARALLEL_ENABLED
  MPI_Bcast(header.data(), headersize, MPI_CHAR, 0, MPI_COMM_WORLD);
#endif
  IOWrapperSizeT os = 0;
  std::memcpy(&rt.nmb_total, &header[os], sizeof(int));
  os += sizeof(int);
  std::memcpy(&rt.root_level, &header[os], sizeof(int));
  os += sizeof(int);
  std::memcpy(&rt.mesh_size, &header[os], sizeof(RegionSize));
  os += sizeof(RegionSize);
  std::memcpy(&rt.mesh_indcs, &header[os], sizeof(RegionIndcs));
  os += sizeof(RegionIndcs);
  std::memcpy(&rt.mb_indcs, &header[os], sizeof(RegionIndcs));
  os += sizeof(RegionIndcs);
  std::memcpy(&rt.time, &header[os], sizeof(Real));
  os += sizeof(Real);
  os += sizeof(Real);  // dt
  std::memcpy(&rt.ncycle, &header[os], sizeof(int));

  if (read_list) {
    if (rt.nmb_total < 1) {
      FatalExpand("Restart file " + path + " holds " + std::to_string(rt.nmb_total) +
                  " MeshBlocks.");
    }
    const std::size_t listsize = sizeof(LogicalLocation) + sizeof(float);
    std::vector<char> idlist(listsize * rt.nmb_total);
    if (global_variable::my_rank == 0) {
      if (input.Read_bytes(idlist.data(), listsize, rt.nmb_total, false)
          != static_cast<std::size_t>(rt.nmb_total)) {
        FatalExpand("Failed to read the MeshBlock list of restart file " + path + ".");
      }
    }
#if MPI_PARALLEL_ENABLED
    MPI_Bcast(idlist.data(), idlist.size(), MPI_CHAR, 0, MPI_COMM_WORLD);
#endif
    // the writer (src/outputs/restart.cpp, step 2) stores all nmb_total locations and
    // then all nmb_total costs, not (location, cost) pairs
    rt.lloc.resize(rt.nmb_total);
    std::memcpy(rt.lloc.data(), idlist.data(), rt.nmb_total*sizeof(LogicalLocation));
  }
  input.Close(false);
}

bool AxisActive(const Mesh *pm, const int d) {
  return (d == 0) || (d == 1 && pm->multi_d) || (d == 2 && pm->three_d);
}

void MeshBox(const RegionSize &s, Real lo[3], Real hi[3]) {
  lo[0] = s.x1min; lo[1] = s.x2min; lo[2] = s.x3min;
  hi[0] = s.x1max; hi[1] = s.x2max; hi[2] = s.x3max;
}

std::string BoxString(const Real lo[3], const Real hi[3], const bool act[3]) {
  std::ostringstream os;
  os.precision(10);
  bool first = true;
  for (int d = 0; d < 3; ++d) {
    if (!act[d]) continue;
    os << (first ? "" : " x ") << "[" << lo[d] << ", " << hi[d] << "]";
    first = false;
  }
  return os.str();
}

//----------------------------------------------------------------------------------------
// Auto-expansion: see remap.hpp.  A pass plans at most one expansion, so the state is
// file-scope, like the settle state in remap.cpp; ApplyExpansion resets it for the next
// pass.

struct ExpandState {
  bool configured = false;
  bool on = false;
  bool coarsen = false;
  bool predict = false;       // <remap>/expand_predict: grow to the hook's predicted box
  bool scanned = false;       // the first (full) scan has run
  bool requested = false;
  long long budget = 0;       // <mesh_refinement>/max_nmb_total
  bool closed[6] = {false, false, false, false, false, false};  // never move again
  Real cap_min[3] = {0.0, 0.0, 0.0};
  Real cap_max[3] = {0.0, 0.0, 0.0};
  // the planned expansion
  int k = 0;
  Real new_min[3] = {0.0, 0.0, 0.0};
  Real new_max[3] = {0.0, 0.0, 0.0};
  int new_nx[3] = {1, 1, 1};
  Real center[3] = {0.0, 0.0, 0.0};
};
ExpandState expand;
bool expansion_pass_start = false;  // set by ApplyExpansion for the next pass

void ConfigureExpand(ParameterInput *pin, Mesh *pm) {
  expand.configured = true;
  expand.on = pin->DoesBlockExist("remap") &&
              pin->GetOrAddBoolean("remap", "auto_expand", false);
  if (!expand.on) return;
  if (pm->pgen == nullptr || !pm->pgen->user_expand_need_func) {
    FatalExpand("<remap>/auto_expand = true needs a problem generator that enrolls "
                "ProblemGenerator::user_expand_need_func.");
  }
  bool act[3];
  Real lo[3], hi[3];
  MeshBox(pm->mesh_size, lo, hi);
  const int nmb_root[3] = {pm->nmb_rootx1, pm->nmb_rootx2, pm->nmb_rootx3};
  for (int d = 0; d < 3; ++d) {
    act[d] = AxisActive(pm, d);
    for (int side = 0; side < 2; ++side) {
      const std::string key = std::string("expand_max_") + kFaceName[2*d + side];
      if (!pin->DoesParameterExist("remap", key)) {
        FatalExpand("<remap>/auto_expand = true needs all six limits "
                    "<remap>/expand_max_x1min ... expand_max_x3max; " + key +
                    " is missing.");
      }
      (side == 0 ? expand.cap_min[d] : expand.cap_max[d]) = pin->GetReal("remap", key);
    }
    const Real tol = 1.0e-8*(hi[d] - lo[d])/static_cast<Real>(nmb_root[d]);
    if (expand.cap_min[d] > lo[d] + tol || expand.cap_max[d] < hi[d] - tol) {
      FatalExpand("<remap>/expand_max_" + std::string(kFaceName[2*d]) + " .. " +
                  kFaceName[2*d + 1] + " = [" + std::to_string(expand.cap_min[d]) + ", " +
                  std::to_string(expand.cap_max[d]) + "] does not contain the mesh [" +
                  std::to_string(lo[d]) + ", " + std::to_string(hi[d]) + "].");
    }
  }
  expand.coarsen = pin->GetOrAddBoolean("remap", "expand_coarsen", false);
  // A face moves when needed gas (the pgen's hook; <remap>/expand_rho_min is its density
  // threshold) comes within its buffer.  expand_predict = true also grows the mesh at once
  // to the box the hook predicts that gas reaches.
  expand.predict = pin->GetOrAddBoolean("remap", "expand_predict", false);

  // The next pass starts from the checkpoint Driver::Finalize writes, so the run must
  // carry a restart output that writes one global file.
  bool rst_ok = false;
  for (const auto &blk : pin->block) {
    const std::string &name = blk.block_name;
    if (name.compare(0, 6, "output") != 0) continue;
    if (!pin->DoesParameterExist(name, "file_type") ||
        pin->GetString(name, "file_type") != "rst") continue;
    const bool cadence = pin->DoesParameterExist(name, "dcycle") ?
        (pin->GetInteger(name, "dcycle") != 0) :
        (pin->DoesParameterExist(name, "dt") && pin->GetReal(name, "dt") > 0.0);
    const bool per_rank = pin->DoesParameterExist(name, "single_file_per_rank") &&
                          pin->GetBoolean(name, "single_file_per_rank");
    if (cadence && !per_rank) rst_ok = true;
  }
  if (!rst_ok) {
    FatalExpand("<remap>/auto_expand = true needs an <outputN> block with file_type = "
                "rst (a positive dt or dcycle, single_file_per_rank = false): the "
                "next pass starts from the checkpoint written when the run stops.");
  }

  std::string periodic;
  for (int f = 0; f < 6; ++f) {
    const bool per = pm->mesh_bcs[f] == BoundaryFlag::periodic ||
                     pm->mesh_bcs[f] == BoundaryFlag::shear_periodic;
    expand.closed[f] = per || !act[f/2];
    if (per && act[f/2]) periodic += std::string(periodic.empty() ? "" : " ") +
                                     kFaceName[f];
  }

  {
    const std::string bad = ReadBlockBudget(pin, pm->nmb_maxperrank, expand.budget);
    if (!bad.empty()) FatalExpand(bad);
  }
  const double budget = static_cast<double>(expand.budget);
  // root MeshBlocks of the current size that fit between the limits, on the lattice
  // anchored at the current faces (periodic faces fixed)
  double nroot_cap = 1.0;
  for (int d = 0; d < 3; ++d) {
    if (!act[d]) continue;
    const Real bw = (hi[d] - lo[d])/static_cast<Real>(nmb_root[d]);
    auto room = [bw](const Real dist) {
      return std::max(0.0, std::floor(static_cast<double>(dist/bw) + 1.0e-8));
    };
    const double n0 = expand.closed[2*d] ? 0.0 : room(lo[d] - expand.cap_min[d]);
    const double n1 = expand.closed[2*d + 1] ? 0.0 : room(expand.cap_max[d] - hi[d]);
    nroot_cap *= static_cast<double>(nmb_root[d]) + n0 + n1;
  }
  if (global_variable::my_rank == 0) {
    std::cout << std::endl << "--- Remap auto-expansion ---" << std::endl
              << "expansion limits       = "
              << BoxString(expand.cap_min, expand.cap_max, act) << std::endl
              << "expand_coarsen         = " << (expand.coarsen ? "true" : "false")
              << std::endl
              << "expand_predict         = " << (expand.predict ? "true" : "false")
              << std::endl
              << "MeshBlock budget       = " << expand.budget
              << " (<mesh_refinement>/max_nmb_total; an inherited tree beyond 0.9 of it "
              << "loses its deepest levels)"
              << std::endl;
    if (!expand.coarsen) {
      std::cout << "root MeshBlocks at the limits = "
                << static_cast<long long>(nroot_cap) << std::endl;
    }
    std::cout << "periodic faces (fixed) = " << (periodic.empty() ? "none" : periodic)
              << std::endl << std::endl;
  }
  if (!expand.coarsen && nroot_cap > budget) {
    std::ostringstream msg;
    msg << "The expansion limits hold " << static_cast<long long>(nroot_cap) << " root "
        << "MeshBlocks of the current size, more than the MeshBlock budget "
        << "<mesh_refinement>/max_nmb_total = " << expand.budget << ".  Shrink the "
        << "limits, raise max_nmb_total (and max_nmb_per_rank or the rank count with "
        << "it), or set <remap>/expand_coarsen = true.";
    FatalExpand(msg.str());
  }
}

}  // namespace

//----------------------------------------------------------------------------------------
//! \fn bool InheritTreeRequested(ParameterInput *pin)

bool InheritTreeRequested(ParameterInput *pin) {
  if (pin == nullptr || !pin->DoesBlockExist("remap")) return false;
  if (!pin->GetOrAddBoolean("remap", "enable", true)) return false;
  return pin->GetOrAddBoolean("remap", "inherit_tree", false);
}

//----------------------------------------------------------------------------------------
//! \fn void AddSourceTreeLeaves(Mesh *pm, ParameterInput *pin)
//! \brief adds every leaf of the <remap>/source tree to the target tree (remap.hpp).

void AddSourceTreeLeaves(Mesh *pm, ParameterInput *pin) {
  const std::string path = pin->GetOrAddString("remap", "source", "");
  if (path.empty()) {
    FatalExpand("<remap>/inherit_tree = true needs <remap>/source.");
  }
  RestartTree src;
  ReadRestartTree(path, true, src);

  const int src_mb[3] = {src.mb_indcs.nx1, src.mb_indcs.nx2, src.mb_indcs.nx3};
  const int tgt_mb[3] = {pm->mb_indcs.nx1, pm->mb_indcs.nx2, pm->mb_indcs.nx3};
  if (src_mb[0] != tgt_mb[0] || src_mb[1] != tgt_mb[1] || src_mb[2] != tgt_mb[2]) {
    FatalExpand("<remap>/inherit_tree needs MeshBlocks of the source's size: source " +
                std::to_string(src_mb[0]) + " x " + std::to_string(src_mb[1]) + " x " +
                std::to_string(src_mb[2]) + ", target " + std::to_string(tgt_mb[0]) +
                " x " + std::to_string(tgt_mb[1]) + " x " + std::to_string(tgt_mb[2]) +
                ".");
  }
  Real slo[3], shi[3], tlo[3], thi[3];
  MeshBox(src.mesh_size, slo, shi);
  MeshBox(pm->mesh_size, tlo, thi);
  const int snx[3] = {src.mesh_indcs.nx1, src.mesh_indcs.nx2, src.mesh_indcs.nx3};
  const int tnx[3] = {pm->mesh_indcs.nx1, pm->mesh_indcs.nx2, pm->mesh_indcs.nx3};
  bool act[3];
  int k = -1;
  std::int64_t offset[3] = {0, 0, 0};
  for (int d = 0; d < 3; ++d) {
    act[d] = AxisActive(pm, d);
    if (!act[d]) continue;
    const Real sdx = (shi[d] - slo[d])/static_cast<Real>(snx[d]);
    const Real tdx = (thi[d] - tlo[d])/static_cast<Real>(tnx[d]);
    const Real ratio = tdx/sdx;
    const int kd = static_cast<int>(std::lround(std::log2(ratio)));
    const Real pw = std::ldexp(static_cast<Real>(1.0), kd);
    if (kd < 0 || std::abs(ratio - pw) > 1.0e-10*pw) {
      FatalExpand("<remap>/inherit_tree: the target root cell is " +
                  std::to_string(ratio) + " source root cells along " +
                  kAxisName[d] + "; it must be 2^k with k >= 0.");
    }
    if (k >= 0 && kd != k) {
      FatalExpand("<remap>/inherit_tree: the target/source root cell ratio must be the "
                  "same power of two on every axis.");
    }
    k = kd;
    const Real bw = sdx*static_cast<Real>(src_mb[d]);
    const Real o = (slo[d] - tlo[d])/bw;
    const std::int64_t ol = std::llround(o);
    if (std::abs(o - static_cast<Real>(ol)) > 1.0e-8) {
      FatalExpand("<remap>/inherit_tree: the source box must be offset from the target "
                  "box by whole source root MeshBlocks; along " +
                  std::string(kAxisName[d]) + " the source is [" +
                  std::to_string(slo[d]) + ", " + std::to_string(shi[d]) +
                  "], the target [" + std::to_string(tlo[d]) + ", " +
                  std::to_string(thi[d]) + "], offset " + std::to_string(o) +
                  " source root MeshBlocks.");
    }
    offset[d] = ol;
  }
  k = std::max(k, 0);

  // every leaf must be a block of the source's own root grid; placing anything else
  // would walk the target tree outside its root grid
  int src_pmax = 0;
  for (int n = 0; n < src.nmb_total; ++n) {
    const LogicalLocation &ll = src.lloc[n];
    const std::int64_t p = static_cast<std::int64_t>(ll.level) - src.root_level;
    const std::int64_t l[3] = {ll.lx1, ll.lx2, ll.lx3};
    bool inside = (p >= 0 && p < 32);
    for (int d = 0; d < 3 && inside; ++d) {
      const std::int64_t nl = act[d] ? (static_cast<std::int64_t>(snx[d]/src_mb[d]) << p)
                                     : 1;
      inside = (l[d] >= 0 && l[d] < nl);
    }
    if (!inside) {
      FatalExpand("<remap>/inherit_tree: MeshBlock " + std::to_string(n) + " of " +
                  path + " has the logical location (" + std::to_string(ll.lx1) + ", " +
                  std::to_string(ll.lx2) + ", " + std::to_string(ll.lx3) + ") at level " +
                  std::to_string(ll.level) + ", outside the source's root grid of " +
                  std::to_string(snx[0]/src_mb[0]) + " x " +
                  std::to_string(snx[1]/src_mb[1]) + " x " +
                  std::to_string(snx[2]/src_mb[2]) + " MeshBlocks at level " +
                  std::to_string(src.root_level) + "; the file is damaged.");
    }
    src_pmax = std::max(src_pmax, static_cast<int>(p));
  }
  // deepest target physical level; a uniform target holds only its root grid
  const int qmax = pm->multilevel ? (pm->max_level - pm->root_level) : 0;
  // every source leaf on the target tree: depth q = p + k below the target root and its
  // logical location at that depth
  struct TreeLeaf {
    int q;
    std::int64_t l[3];
  };
  // a source leaf outside the target box (a target that crops the source) is dropped:
  // its gas is lost, the target there is ambient
  std::vector<TreeLeaf> mapped;
  mapped.reserve(src.lloc.size());
  int ndropped = 0;
  for (const auto &ll : src.lloc) {
    const int p = ll.level - src.root_level;
    const std::int64_t l[3] = {ll.lx1, ll.lx2, ll.lx3};
    TreeLeaf lf;
    lf.q = p + k;
    bool inside = true;
    for (int d = 0; d < 3; ++d) {
      lf.l[d] = act[d] ? (l[d] + offset[d]*(static_cast<std::int64_t>(1) << p)) : 0;
      const std::int64_t nt = act[d] ? (static_cast<std::int64_t>(tnx[d]/tgt_mb[d]) << lf.q)
                                     : 1;
      if (lf.l[d] < 0 || lf.l[d] >= nt) inside = false;
    }
    if (inside) {
      mapped.push_back(lf);
    } else {
      ++ndropped;
    }
  }
  // the leaves the tree holds already (root grid, refined regions)
  std::vector<TreeLeaf> present;
  {
    int n0 = 0;
    pm->ptree->CountMeshBlocks(n0);
    std::vector<LogicalLocation> lv(n0);
    pm->ptree->CreateZOrderedLLList(lv.data(), nullptr, n0);
    for (const auto &ll : lv) {
      present.push_back({ll.level - pm->root_level, {ll.lx1, ll.lx2, ll.lx3}});
    }
  }

  // MeshBlocks of the tree when every source leaf deeper than qcap is merged into its
  // ancestor at qcap.  The refined nodes are the ancestors of all leaves, closed under the
  // rule MeshBlockTree::Refine keeps (a refined node's neighbours on its own level exist,
  // so their parents are refined); each refined node turns one leaf into 2^dim.
  const std::int64_t nroot[3] = {pm->nmb_rootx1, pm->nmb_rootx2, pm->nmb_rootx3};
  bool per[3];
  per[0] = pm->mesh_bcs[BoundaryFace::inner_x1] == BoundaryFlag::periodic ||
           pm->mesh_bcs[BoundaryFace::inner_x1] == BoundaryFlag::shear_periodic;
  per[1] = pm->mesh_bcs[BoundaryFace::inner_x2] == BoundaryFlag::periodic;
  per[2] = pm->mesh_bcs[BoundaryFace::inner_x3] == BoundaryFlag::periodic;
  std::int64_t nroot_all = 1, nchild = 1;
  for (int d = 0; d < 3; ++d) {
    if (!act[d]) continue;
    nroot_all *= nroot[d];
    nchild *= 2;
  }
  auto count_blocks = [&](const int qcap) -> std::int64_t {
    using Node = std::array<std::int64_t, 4>;  // depth, lx1, lx2, lx3
    std::set<Node> refined;
    std::vector<Node> work;
    auto refine_ancestors = [&](const int q, const std::int64_t *l) {
      for (int a = q - 1; a >= 0; --a) {
        const int s = q - a;
        const Node n = {a, l[0] >> s, l[1] >> s, l[2] >> s};
        if (!refined.insert(n).second) break;  // so are all its ancestors
        work.push_back(n);
      }
    };
    for (const auto &lf : present) refine_ancestors(lf.q, lf.l);
    for (const auto &lf : mapped) {
      const int q = std::min(lf.q, qcap);
      const std::int64_t l[3] = {lf.l[0] >> (lf.q - q), lf.l[1] >> (lf.q - q),
                                 lf.l[2] >> (lf.q - q)};
      refine_ancestors(q, l);
    }
    while (!work.empty()) {
      const Node n = work.back();
      work.pop_back();
      const int d = static_cast<int>(n[0]);
      for (int o3 = -1; o3 <= 1; ++o3) {
        for (int o2 = -1; o2 <= 1; ++o2) {
          for (int o1 = -1; o1 <= 1; ++o1) {
            const int o[3] = {o1, o2, o3};
            bool skip = (o1 == 0 && o2 == 0 && o3 == 0);
            std::int64_t m[3];
            for (int e = 0; e < 3 && !skip; ++e) {
              if (!act[e]) {
                skip = (o[e] != 0);
                m[e] = 0;
                continue;
              }
              const std::int64_t nmax = nroot[e] << d;
              m[e] = n[1 + e] + o[e];
              if (m[e] < 0 || m[e] >= nmax) {
                if (per[e]) {
                  m[e] = (m[e] + nmax) % nmax;
                } else {
                  skip = true;
                }
              }
            }
            if (!skip) refine_ancestors(d, m);
          }
        }
      }
    }
    return nroot_all + (nchild - 1)*static_cast<std::int64_t>(refined.size());
  };
  // An inherited tree the MeshBlock budget cannot hold is coarsened: the deepest levels
  // are merged until it fits 0.9 of <mesh_refinement>/max_nmb_total, the share AMR
  // refinement also stops at.  The merged cells are remapped from the source
  // like every other cell (remap_cc.cpp).
  int qcap = qmax;
  double limit = -1.0;
  if (pm->adaptive && pin->DoesParameterExist("mesh_refinement", "max_nmb_per_rank")) {
    long long total = 0;
    const std::string bad = ReadBlockBudget(
        pin, static_cast<long long>(pin->GetReal("mesh_refinement", "max_nmb_per_rank")),
        total);
    if (!bad.empty()) FatalExpand(bad);
    limit = 0.9*static_cast<double>(total);
    while (qcap > 0 && static_cast<double>(count_blocks(qcap)) > limit) --qcap;
  }
  int nmerged = 0, merge_depth = 0;
  for (const auto &lf : mapped) {
    const int q = std::min(lf.q, qcap);
    const int s = lf.q - q;
    if (s > 0) {  // merge into the ancestor at depth q
      merge_depth = std::max(merge_depth, s);
      ++nmerged;
    }
    if (q == 0) continue;  // a root block: already in the tree
    LogicalLocation loc;
    loc.lx1 = static_cast<std::int32_t>(lf.l[0] >> s);
    loc.lx2 = static_cast<std::int32_t>(lf.l[1] >> s);
    loc.lx3 = static_cast<std::int32_t>(lf.l[2] >> s);
    loc.level = pm->root_level + q;
    int nnew = 0;
    pm->ptree->AddNode(loc, nnew);
  }
  int nfinal = 0;
  pm->ptree->CountMeshBlocks(nfinal);
  if (global_variable::my_rank == 0) {
    std::cout << std::endl << "--- Remap tree inheritance ---" << std::endl
              << "source restart         = " << path << std::endl
              << "source leaves          = " << src.nmb_total << " (" << (src_pmax + 1)
              << " levels)" << std::endl
              << "root cell ratio        = 2^" << k << std::endl
              << "source offset          = (" << offset[0] << ", " << offset[1] << ", "
              << offset[2] << ") source root MeshBlocks" << std::endl
              << "dropped (outside)      = " << ndropped << " leaves" << std::endl
              << "merged into ancestors  = " << nmerged << " leaves (up to "
              << merge_depth << " levels)" << std::endl
              << "deepest level          = " << qcap << " of " << qmax << " below the root"
              << std::endl
              << "MeshBlocks             = " << nfinal;
    if (limit > 0.0) {
      std::cout << " (0.9 of the budget = " << static_cast<long long>(limit) << ")";
    }
    std::cout << std::endl << std::endl;
  }
}

//----------------------------------------------------------------------------------------
//! \fn bool AfterCycleExpand(Driver *driver, ParameterInput *pin, Mesh *pm)
//! \brief plans a domain expansion at a synchronized end of an outer step (remap.hpp).
//! Returns true when one is planned; the Driver then ends the run at this point.

bool AfterCycleExpand(Driver *driver, ParameterInput *pin, Mesh *pm) {
  (void) driver;
  if (pin == nullptr || pm == nullptr) return false;
  if (!expand.configured) ConfigureExpand(pin, pm);
  if (!expand.on || SettleActive() || expand.requested) return false;

  bool act[3];
  Real lo[3], hi[3], dx[3], bw[3], buf[3];
  MeshBox(pm->mesh_size, lo, hi);
  const int nb[3] = {pm->nmb_rootx1, pm->nmb_rootx2, pm->nmb_rootx3};
  const int nx[3] = {pm->mesh_indcs.nx1, pm->mesh_indcs.nx2, pm->mesh_indcs.nx3};
  const int mbnx[3] = {pm->mb_indcs.nx1, pm->mb_indcs.nx2, pm->mb_indcs.nx3};
  for (int d = 0; d < 3; ++d) {
    act[d] = AxisActive(pm, d);
    dx[d] = (hi[d] - lo[d])/static_cast<Real>(nx[d]);
    bw[d] = (hi[d] - lo[d])/static_cast<Real>(nb[d]);
    buf[d] = act[d] ? static_cast<Real>(kExpandBufferRootCells)*dx[d] : 0.0;
  }

  // nothing can move any more: no scan (closed[] is the same on every rank)
  bool all_closed = true;
  for (int f = 0; f < 6; ++f) all_closed = all_closed && expand.closed[f];
  if (all_closed) return false;

  // cheap near-face scan, then the full scan on the first call or when it finds gas
  ExpandNeed need;
  const bool first = !expand.scanned;
  expand.scanned = true;
  if (!first) {
    pm->pgen->user_expand_need_func(pm, buf, false, need);
    bool trigger = false;
    for (int f = 0; f < 6; ++f) trigger = trigger || (!expand.closed[f] &&
                                                      need.near_face[f]);
    if (!trigger) return false;
    need = ExpandNeed();
  }
  pm->pgen->user_expand_need_func(pm, buf, true, need);
  if (!expand.predict) need.have_box = false;
  bool near[6];
  for (int f = 0; f < 6; ++f) near[f] = !expand.closed[f] && need.near_face[f];
  // the same scan with the buffer of the doubled root cell (filled for expand_coarsen)
  ExpandNeed need2;
  bool near2[6] = {false, false, false, false, false, false};

  // Root MeshBlocks each side grows by, on the lattice of the current root blocks, for a
  // root cell ratio 2^kk.  The required box is the predicted reach plus the buffer of the
  // grown mesh's root cell (so the next pass does not find its own buffer entered), else
  // one axis length per face with needed gas within that buffer.  With kk = 1 every axis
  // must hold an even number of root blocks.  want[f]: face f has to move.  Returns the
  // axis that cannot be made even, or -1.
  auto plan = [&](const int kk, std::int64_t g0[3], std::int64_t g1[3],
                  bool want[6]) -> int {
    const ExpandNeed &nd = (kk == 1) ? need2 : need;
    const bool *nr = (kk == 1) ? near2 : near;
    constexpr std::int64_t kHuge = std::int64_t(1) << 40;
    for (int d = 0; d < 3; ++d) {
      g0[d] = 0;
      g1[d] = 0;
      want[2*d] = false;
      want[2*d + 1] = false;
      if (!act[d]) continue;
      auto blocks = [&](const Real dist) -> std::int64_t {
        if (!(dist > 0.0)) return 0;
        const Real n = std::min(std::ceil(dist/bw[d] - 1.0e-8), static_cast<Real>(kHuge));
        return std::max<std::int64_t>(1, static_cast<std::int64_t>(n));
      };
      auto room = [&](const Real dist) -> std::int64_t {
        return std::max<std::int64_t>(0, static_cast<std::int64_t>(
            std::floor(dist/bw[d] + 1.0e-8)));
      };
      Real r0 = lo[d], r1 = hi[d];
      if (nd.have_box) {
        const Real b = std::ldexp(buf[d], kk);
        if (!expand.closed[2*d]) r0 = std::min(lo[d], nd.box_min[d] - b);
        if (!expand.closed[2*d + 1]) r1 = std::max(hi[d], nd.box_max[d] + b);
      } else {
        const Real len = hi[d] - lo[d];
        if (nr[2*d]) r0 = lo[d] - len;
        if (nr[2*d + 1]) r1 = hi[d] + len;
      }
      std::int64_t a0 = blocks(lo[d] - r0);
      std::int64_t a1 = blocks(r1 - hi[d]);
      want[2*d] = a0 > 0;
      want[2*d + 1] = a1 > 0;
      const std::int64_t m0 = expand.closed[2*d] ? 0 : room(lo[d] - expand.cap_min[d]);
      const std::int64_t m1 = expand.closed[2*d + 1] ? 0 :
                              room(expand.cap_max[d] - hi[d]);
      if (kk == 1 && ((nb[d] + a0 + a1) & 1) != 0) {
        if (a1 > 0 || a0 == 0) ++a1; else ++a0;
      }
      a0 = std::min(a0, m0);
      a1 = std::min(a1, m1);
      if (kk == 1 && ((nb[d] + a0 + a1) & 1) != 0) {
        if (a1 > 0) {
          --a1;
        } else if (a0 > 0) {
          --a0;
        } else {
          return d;
        }
      }
      g0[d] = a0;
      g1[d] = a1;
    }
    return -1;
  };
  int k = 0;
  std::int64_t g0[3], g1[3];
  bool want[6];
  (void) plan(0, g0, g1, want);
  bool moves0 = false;  // an expansion is due only if the root-cell-keeping plan moves
  for (int d = 0; d < 3; ++d) moves0 = moves0 || g0[d] > 0 || g1[d] > 0;
  if (expand.coarsen && moves0) {
    // A doubled root cell doubles the next pass's buffer: needed gas within it at a face
    // this plan left in place would start the next expansion at that pass's first scan.
    Real buf2[3];
    for (int d = 0; d < 3; ++d) buf2[d] = 2.0*buf[d];
    pm->pgen->user_expand_need_func(pm, buf2, true, need2);
    if (!expand.predict) need2.have_box = false;
    for (int f = 0; f < 6; ++f) near2[f] = !expand.closed[f] && need2.near_face[f];
    std::int64_t c0[3], c1[3];
    bool cwant[6];
    const int bad = plan(1, c0, c1, cwant);
    int held = -1;  // a face the k = 0 plan moves but the even plan does not
    for (int d = 0; d < 3 && bad < 0 && held < 0; ++d) {
      if (g0[d] > 0 && c0[d] == 0) held = 2*d;
      if (g1[d] > 0 && c1[d] == 0) held = 2*d + 1;
    }
    if (bad < 0 && held < 0) {
      k = 1;
      for (int d = 0; d < 3; ++d) {
        g0[d] = c0[d];
        g1[d] = c1[d];
      }
      for (int f = 0; f < 6; ++f) {
        want[f] = cwant[f];
        near[f] = near2[f];
      }
      need = need2;
    } else if (global_variable::my_rank == 0) {
      std::cout << "remap auto-expansion: this expansion keeps the root cell ("
                << ((bad >= 0) ? (std::string("no even number of root MeshBlocks fits "
                                              "within the limits along ") +
                                  kAxisName[bad])
                               : (std::string("an even number of root MeshBlocks "
                                              "would hold face ") + kFaceName[held]))
                << ")" << std::endl;
    }
  }

  // faces that have to move but cannot are at their limits
  bool moves = false;
  for (int d = 0; d < 3; ++d) {
    moves = moves || g0[d] > 0 || g1[d] > 0;
    for (int side = 0; side < 2; ++side) {
      const int f = 2*d + side;
      if (expand.closed[f] || !(near[f] || want[f]) ||
          (side == 0 ? g0[d] : g1[d]) > 0) continue;
      // a face the doubling plan holds back only to keep its axis even has room left
      const Real room_left = (side == 0) ? lo[d] - expand.cap_min[d] :
                                           expand.cap_max[d] - hi[d];
      if (std::floor(room_left/bw[d] + 1.0e-8) >= 1.0) continue;
      expand.closed[f] = true;
      if (global_variable::my_rank == 0) {
        std::cout << "remap auto-expansion: face " << kFaceName[f] << " at its limit; "
                  << "needed gas reaching it will leave through the boundary"
                  << std::endl;
      }
    }
  }
  if (!moves) return false;

  // MeshBlock budget: the next pass merges the inherited tree's deepest levels until it
  // fits (AddSourceTreeLeaves), so only a root grid that alone exceeds the budget cannot
  // be built
  double nroot_new = 1.0, nroot_old = 1.0;
  for (int d = 0; d < 3; ++d) {
    if (!act[d]) continue;
    nroot_old *= static_cast<double>(nb[d]);
    nroot_new *= static_cast<double>((nb[d] + g0[d] + g1[d]) >> k);
  }
  const double budget = static_cast<double>(expand.budget);
  if (nroot_new > 0.9*budget) {
    std::string faces;
    for (int d = 0; d < 3; ++d) {
      for (int side = 0; side < 2; ++side) {
        const int f = 2*d + side;
        if ((side == 0 ? g0[d] : g1[d]) == 0 && !near[f] && !want[f]) continue;
        if (!expand.closed[f]) faces += std::string(faces.empty() ? "" : " ") +
                                        kFaceName[f];
        expand.closed[f] = true;
      }
    }
    if (global_variable::my_rank == 0) {
      std::cout << "remap auto-expansion: not expanding: the "
                << static_cast<long long>(nroot_new) << " root MeshBlocks of the grown "
                << "mesh alone exceed 0.9 of the budget "
                << static_cast<long long>(budget) << "; faces " << faces
                << " are left to their boundary conditions" << std::endl;
    }
    return false;
  }

  // store the plan; the run ends here (Driver::Execute)
  expand.k = k;
  for (int d = 0; d < 3; ++d) {
    expand.new_min[d] = lo[d] - static_cast<Real>(g0[d])*bw[d];
    expand.new_max[d] = hi[d] + static_cast<Real>(g1[d])*bw[d];
    expand.new_nx[d] = act[d] ? static_cast<int>(((nb[d] + g0[d] + g1[d])*mbnx[d]) >> k)
                              : nx[d];
    expand.center[d] = need.center[d];
  }
  expand.requested = true;
  if (global_variable::my_rank == 0) {
    std::string trigger = need.have_box ? "predicted reach of the needed gas" : "";
    for (int f = 0; f < 6 && !need.have_box; ++f) {
      if (near[f]) trigger += std::string(trigger.empty() ? "needed gas at " : " ") +
                              kFaceName[f];
    }
    std::cout << std::endl << "--- Remap auto-expansion planned ---" << std::endl
              << "cycle / time           = " << pm->ncycle << " / " << pm->time
              << std::endl
              << "trigger                = " << trigger << std::endl
              << "old box                = " << BoxString(lo, hi, act) << std::endl
              << "new box                = "
              << BoxString(expand.new_min, expand.new_max, act) << std::endl
              << "root cell              = "
              << (k == 1 ? "doubled (k = 1)" : "kept (k = 0)") << std::endl
              << "root MeshBlocks        = " << static_cast<long long>(nroot_old)
              << " -> " << static_cast<long long>(nroot_new)
              << std::endl
              << "the pass stops here; the next one starts from the checkpoint"
              << std::endl << std::endl;
  }
  return true;
}

bool ExpandRequested() { return expand.requested; }

//----------------------------------------------------------------------------------------
//! \fn bool ApplyExpansion(Mesh *pm, ParameterInput *pin, Outputs *pout)
//! \brief after Driver::Finalize: puts a planned expansion into the input and returns
//! true, or returns false when none was planned.  The input then describes the next pass,
//! a fresh start on the grown mesh that remaps the checkpoint Finalize has just written.

bool ApplyExpansion(Mesh *pm, ParameterInput *pin, Outputs *pout) {
  if (!expand.requested) return false;
  // the checkpoint Driver::Finalize has just written
  BaseTypeOutput *rst = nullptr;
  if (pout != nullptr) {
    for (auto *out : pout->pout_list) {
      if (rst == nullptr && out->out_params.file_type == "rst" &&
          !out->out_params.single_file_per_rank) {
        rst = out;
      }
    }
  }
  std::string path;
  int ok = (rst != nullptr) ? 1 : 0;
  if (ok != 0) {
    const int number = rst->out_params.file_number - 1;
    char *full = realpath(rst->FileNameForNumber(number).c_str(), nullptr);
    if (full == nullptr) {
      ok = 0;
    } else {
      path = full;
      std::free(full);
    }
  }
#if MPI_PARALLEL_ENABLED
  // also the point at which every rank has finished writing its part of the checkpoint
  MPI_Allreduce(MPI_IN_PLACE, &ok, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
#endif
  if (ok != 0) {  // it must hold exactly this state
    RestartTree chk;
    ReadRestartTree(path, false, chk);
    if (chk.time != pm->time || chk.ncycle != pm->ncycle) ok = 0;
  }
  if (ok == 0) {
    FatalExpand("No global checkpoint of the stopped state was found (" +
                (path.empty() ? std::string("no restart file") : path) +
                "); the expansion cannot start from it.");
  }

  const char *lo_keys[3] = {"x1min", "x2min", "x3min"};
  const char *hi_keys[3] = {"x1max", "x2max", "x3max"};
  const char *nx_keys[3] = {"nx1", "nx2", "nx3"};
  for (int d = 0; d < 3; ++d) {
    if (!AxisActive(pm, d)) continue;
    pin->SetReal("mesh", lo_keys[d], expand.new_min[d]);
    pin->SetReal("mesh", hi_keys[d], expand.new_max[d]);
    pin->SetInteger("mesh", nx_keys[d], expand.new_nx[d]);
  }
  // the next pass's remap then finds its floors matching (<remap>/floor_reset)
  if (pin->GetOrAddBoolean("remap", "floor_reset", false) &&
      pm->pmb_pack->phydro != nullptr && pm->pmb_pack->pmhd == nullptr) {
    MatchFloorsToDensity(pin, MeshDensityMax(pm),
                         pm->pmb_pack->phydro->peos->eos_data.dfloor);
  }
  // the remap installs the source time; a nonzero start time reads as a restart
  pin->SetReal("time", "start_time", 0.0);
  pin->SetBoolean("remap", "enable", true);
  pin->SetString("remap", "source", path);
  pin->SetBoolean("remap", "inherit_tree", true);
  pin->SetBoolean("remap", "transition_band", false);
  pin->SetInteger("remap", "settle_steps", 0);
  pin->SetBoolean("remap", "auto_expand", true);

  // The next pass reads this input as it would read a deck dumped from it, every line of
  // which counts as read from a restart header (ParameterInput::IsFromRestartHeader)
  for (auto &block : pin->block) {
    for (auto &line : block.line) line.from_restart_header = true;
  }
  if (global_variable::my_rank == 0) {
    std::cout << "remap auto-expansion: next pass on the grown mesh, source " << path
              << std::endl;
  }
  expand = ExpandState();  // the next pass plans its own expansions
  expansion_pass_start = true;
  return true;
}

bool ExpansionPlanned() { return expand.requested; }

bool TakeExpansionPassStart() {
  const bool start = expansion_pass_start;
  expansion_pass_start = false;
  return start;
}

}  // namespace remap
