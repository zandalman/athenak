#ifndef UTILS_PHASE_COMMUNICATOR_HPP_
#define UTILS_PHASE_COMMUNICATOR_HPP_
//========================================================================================
// AthenaK astrophysical fluid dynamics & numerical relativity code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the AthenaK collaboration
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file phase_communicator.hpp
//! \brief The communicator of the ranks that execute one stable LAT phase.
//!
//! Under LAT the ranks run different bins and stages at a time, so a collective inside a
//! stage task (the drained-cell closure's round count) is taken over exactly the ranks
//! that execute that bin and stage.  One communicator is split per phase key and cached;
//! the cache is dropped when the topology or the LAT metadata changes.  Hydro (the
//! drained-cell closure) and MHD (canonical FOFC, hybrid HDFIXUP, drained-cell closure)
//! both hold one.

#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <map>

#include "athena.hpp"
#include "globals.hpp"

class PhaseCommunicator {
 public:
  PhaseCommunicator() = default;
  PhaseCommunicator(const PhaseCommunicator &) = delete;
  PhaseCommunicator &operator=(const PhaseCommunicator &) = delete;
  ~PhaseCommunicator() { Free(); }

  //! Selects the communicator of the phase `key`, a collective call on every rank at the
  //! first use of the key; `participates` says whether this rank executes the phase.
  void Configure(const std::uint64_t key, const bool participates,
                 const std::uint64_t topology_version,
                 const std::uint64_t metadata_version) {
#if MPI_PARALLEL_ENABLED
    if (topology_version_ != topology_version || metadata_version_ != metadata_version) {
      Free();
      topology_version_ = topology_version;
      metadata_version_ = metadata_version;
    }
    auto iter = comms_.find(key);
    if (iter == comms_.end()) {
      MPI_Comm phase_comm = MPI_COMM_NULL;
      const int color = participates ? 1 : MPI_UNDEFINED;
      if (MPI_Comm_split(MPI_COMM_WORLD, color, global_variable::my_rank,
                         &phase_comm) != MPI_SUCCESS) {
        Abort("Unable to construct the LAT phase communicator.");
      }
      iter = comms_.emplace(key, phase_comm).first;
    }
    if ((iter->second != MPI_COMM_NULL) != participates) {
      Abort("LAT phase membership changed without a topology or LAT metadata change.");
    }
    current_ = iter->second;
#else
    (void) key;
    (void) participates;
    (void) topology_version;
    (void) metadata_version;
#endif
  }

#if MPI_PARALLEL_ENABLED
  //! The communicator of the configured phase; MPI_COMM_NULL on a rank outside it or
  //! before the first Configure.
  MPI_Comm Comm() const { return current_; }
#endif

  //! The maximum of `local` over the ranks of the configured phase.
  int ReduceMax(const int local) const {
#if MPI_PARALLEL_ENABLED
    if (current_ == MPI_COMM_NULL) {
      Abort("a participating rank has no LAT phase communicator.");
    }
    int global = 0;
    if (MPI_Allreduce(&local, &global, 1, MPI_INT, MPI_MAX, current_) != MPI_SUCCESS) {
      Abort("the LAT phase reduction failed.");
    }
    return global;
#else
    return local;
#endif
  }

 private:
  void Free() {
#if MPI_PARALLEL_ENABLED
    for (auto &entry : comms_) {
      if (entry.second != MPI_COMM_NULL) MPI_Comm_free(&entry.second);
    }
    comms_.clear();
    current_ = MPI_COMM_NULL;
#endif
  }

  [[noreturn]] static void Abort(const char *what) {
    std::cerr << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << what << std::endl;
#if MPI_PARALLEL_ENABLED
    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
#endif
    std::exit(EXIT_FAILURE);
  }

#if MPI_PARALLEL_ENABLED
  std::map<std::uint64_t, MPI_Comm> comms_;
  MPI_Comm current_ = MPI_COMM_NULL;
  std::uint64_t topology_version_ = ~std::uint64_t{0};
  std::uint64_t metadata_version_ = ~std::uint64_t{0};
#endif
};

#endif  // UTILS_PHASE_COMMUNICATOR_HPP_
