#ifndef UTILS_MPI_BUFFER_POOL_HPP_
#define UTILS_MPI_BUFFER_POOL_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file mpi_buffer_pool.hpp
//! \brief Process-wide pool of device buffers that are handed to MPI.
//!
//! With GPU-aware MPI over UCX cuda_ipc (UCX_CUDA_IPC_CACHE=y, the default) every peer
//! keeps a mapping of each device buffer it has exchanged with.  When the owner frees
//! such a buffer the allocation stays alive, owned by no process, for as long as any peer
//! holds the mapping.  Every reallocation of an MPI-facing buffer therefore leaks the old
//! one, and an AMR run that grows its buffers through dozens of sizes leaks them all.
//! (Turning the cache off stops the leak but makes large rendezvous messages ~5x slower.)
//!
//! This pool makes the set of distinct MPI-facing allocations of a process small:
//!   * a released buffer returns to the pool, never to the allocator;
//!   * a request reuses the smallest free buffer that holds it and is at most twice its
//!     size, else allocates a new one rounded up to a whole MiB (powers of two from 4 KiB below 1 MiB);
//!   * callers grow a buffer through GrowTarget(), which sizes the new buffer for the
//!     block budget with a geometric floor: the growth factor is in [1.5, 2], so a ramp
//!     from a few blocks to the budget takes log_1.5(ramp) generations, and a run that
//!     sits at the budget and creeps through new payload maxima still grows by 1.5x per
//!     regrow rather than by a sliver.  Buffers outgrown at one size class are reused by
//!     later roles of that class, and the generations of one role form a geometric
//!     series, so the idle retained total is <= ~2x the final live size per role
//!     (sum of 1.5^-k = 3 in the worst case of never reusing, ~2x with the 2x reuse
//!     window), instead of one retained buffer per new maximum.
//!   * when DevMemSpace is HostSpace there is no CUDA IPC mapping to protect: the pool is
//!     a pass-through (Acquire allocates exactly the request, Release frees).
//! Only buffers that reach MPI belong here: a buffer MPI never sees is never mapped by a
//! peer and is freed normally.
//! The views handed out are unmanaged: they do not own the memory, so release a buffer
//! only after every MPI request and kernel using it has completed.  Release() takes any
//! view by its data pointer, resets it, and ignores views not obtained from the pool.
//! Call Finalize() before Kokkos::finalize().

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <map>
#include <type_traits>
#include <unordered_map>
#include "athena.hpp"

namespace mpi_buffer_pool {

using Storage = Kokkos::View<char *, LayoutWrapper, DevMemSpace>;

struct Entry {
  Storage storage;
  std::size_t bytes;
  bool in_use;
};

struct State {
  std::unordered_map<const void *, Entry> all;
  std::multimap<std::size_t, const void *> free_list;   // capacity in bytes -> pointer
};

inline State &Pool() {
  static State *state = new State();   // intentionally never deleted; see Finalize()
  return *state;
}

//! capacity in bytes allocated for a request of n bytes
inline std::size_t RoundBytes(std::size_t n) {
  constexpr std::size_t kMiB = 1024*1024;
  if (n >= kMiB) return ((n + kMiB - 1)/kMiB)*kMiB;
  std::size_t p = 4096;
  while (p < n) p <<= 1;
  return p;
}

//! Number of elements to allocate when a buffer sized for `need` elements must grow.
//! The need was computed from the blocks this rank holds now (nmb_this); the mesh may
//! still grow to nmb_max, so reserve for that: f = 1.1*nmb_max/nmb_this, limited to
//! [1.5, 2].  The cap of 2 matters because the off-rank payload per block falls as a rank
//! fills up (surface to volume), so need scaled by nmb_max/nmb_this overshoots early in a
//! ramp.  The floor of 1.5 matters at or near the budget (or with no budget), where the
//! ratio collapses to ~1: every new payload maximum would otherwise allocate a fresh
//! buffer and leave the outgrown one in the pool.  Only the first allocation of a buffer
//! on a static mesh (`first`, nmb_max <= nmb_this or unknown) stays exactly sized, since
//! it will never regrow.
inline std::size_t GrowTarget(std::size_t need, int nmb_this, int nmb_max,
                              bool first = false) {
  const bool no_headroom = (nmb_this <= 0 || nmb_max <= nmb_this);
  if (first && no_headroom) return need;
  double f = 1.5;
  if (!no_headroom) {
    f = std::max(1.5, std::min(1.1*static_cast<double>(nmb_max)/nmb_this, 2.0));
  }
  return static_cast<std::size_t>(static_cast<double>(need)*f) + 1;
}

//! raw acquisition of at least `bytes`; returns the pointer and the capacity in bytes
inline void *AcquireBytes(std::size_t bytes, bool zero, std::size_t &capacity,
                          std::size_t grow_bytes = 0) {
  State &s = Pool();
  if (std::is_same<DevMemSpace, Kokkos::HostSpace>::value) {
    // pass-through: no CUDA IPC mapping to protect
    Storage st = zero ? Storage("mpi_buffer_pool", bytes)
                      : Storage(Kokkos::view_alloc(Kokkos::WithoutInitializing,
                                                   "mpi_buffer_pool"), bytes);
    const void *ptr = st.data();
    s.all.emplace(ptr, Entry{st, bytes, true});
    capacity = bytes;
    return const_cast<void *>(ptr);
  }
  const std::size_t cls = RoundBytes(bytes);
  const std::size_t alloc = RoundBytes(std::max(bytes, grow_bytes));
  auto it = s.free_list.lower_bound(bytes);
  if (it != s.free_list.end() && it->first <= 2*cls) {
    const void *ptr = it->second;
    s.free_list.erase(it);
    Entry &e = s.all.at(ptr);
    e.in_use = true;
    capacity = e.bytes;
    if (zero) Kokkos::deep_copy(e.storage, static_cast<char>(0));
    return const_cast<void *>(ptr);
  }
  Storage st = zero ? Storage("mpi_buffer_pool", alloc)
                    : Storage(Kokkos::view_alloc(Kokkos::WithoutInitializing,
                                                 "mpi_buffer_pool"), alloc);
  const void *ptr = st.data();
  s.all.emplace(ptr, Entry{st, alloc, true});
  capacity = alloc;
  return const_cast<void *>(ptr);
}

//! 1D view with extent >= n (extent = the rounded capacity).  A new allocation is made
//! for max(n, grow) elements; a free buffer that holds n is reused as it is.
template <typename T>
DvceArray1D<T> Acquire1D(std::size_t n, bool zero = false, std::size_t grow = 0) {
  std::size_t cap = 0;
  void *p = AcquireBytes((n > 0 ? n : 1)*sizeof(T), zero, cap, grow*sizeof(T));
  return DvceArray1D<T>(static_cast<T *>(p), cap/sizeof(T));
}

//! return the buffer behind `view` to the pool and reset the view; a view that does not
//! come from the pool is just reset
template <typename V>
void Release(V &view) {
  State &s = Pool();
  const void *ptr = static_cast<const void *>(view.data());
  auto it = s.all.find(ptr);
  if (ptr != nullptr && it != s.all.end() && it->second.in_use) {
    Kokkos::fence("mpi_buffer_pool::Release");
    if (std::is_same<DevMemSpace, Kokkos::HostSpace>::value) {
      s.all.erase(it);   // pass-through: free
    } else {
      it->second.in_use = false;
      s.free_list.emplace(it->second.bytes, ptr);
    }
  }
  view = V();
}

//! drop every pooled allocation (end of run, before Kokkos::finalize)
inline void Finalize() {
  State &s = Pool();
  s.free_list.clear();
  s.all.clear();
}

}  // namespace mpi_buffer_pool

#endif  // UTILS_MPI_BUFFER_POOL_HPP_
