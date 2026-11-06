#ifndef UTILS_DEVICE_ARGS_HPP_
#define UTILS_DEVICE_ARGS_HPP_
//========================================================================================
// AthenaK astrophysical fluid dynamics & numerical relativity code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the AthenaK collaboration
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file device_args.hpp
//! \brief One device-memory copy of host objects a kernel reads, for kernels whose
//! captures would otherwise reach a callee the device compiler keeps out of line.
//!
//! A light-weight launch passes the functor as the kernel parameter, and a kernel
//! parameter has no address a callee can take.  Once the address of any captured member
//! reaches a call that stays out of line -- a __noinline__ helper, a solver the compiler
//! declines to inline, or the lambda's own operator() when its body is too large to
//! inline -- the device compiler copies the WHOLE functor into each thread's local memory
//! at kernel entry, ahead of any early return, and every callee then reads the captures
//! from that per-thread copy: kilobytes of stores per thread, and frames that crowd the
//! L1.  Read from one copy in device memory instead, the captures keep a global address
//! every thread shares, and nothing is copied.
//!
//! DeviceArgs::Upload(x) writes the bytes of x into the device copy and returns its
//! address; Upload(x, y, ...) writes several objects into one copy.  The write is
//! enqueued on DevExeSpace(), the instance every kernel that reads the copy runs on: it
//! lands after the launches that read the previous contents and before the ones that
//! follow, and the host does not wait.  The source is pageable host memory, which the
//! runtime stages before the copy call returns.  A call whose bytes equal the ones last
//! written enqueues nothing.  The View handles in a copy are bitwise snapshots, as those
//! of a by-value capture are: the arrays belong to their modules and outlive the
//! kernels, and the host side keeps bytes only, no View, so it holds no reference that
//! would keep an array alive across a regrid.
//!
//! DeviceResident(args, f) does this for a whole kernel functor: the launch carries only
//! the address of the device copy of f, and f runs from there, so a body that hands its
//! captures by reference to an out-of-line callee, or whose operator() is itself out of
//! line, reads them in place.  The body and its arithmetic are unchanged.

#include <cstddef>
#include <cstring>
#include <tuple>
#include <utility>
#include <vector>

#include "athena.hpp"

//----------------------------------------------------------------------------------------
//! \class DeviceArgs
//! \brief The device copy behind Upload and DeviceResident.  One per launch site whose
//! bytes stay the same from call to call, so that the steady state enqueues nothing.

class DeviceArgs {
 public:
  //! The device copy of one object.
  template <typename T>
  const T *Upload(const T &host) {
    return std::get<0>(UploadParts(host));
  }
  //! The device copies of several objects, written as one copy with each object on a
  //! 64-byte boundary.  Pass the objects themselves, not temporaries built from them:
  //! their own bytes, padding included, then repeat from call to call.
  template <typename T0, typename T1, typename... Ts>
  std::tuple<const T0 *, const T1 *, const Ts *...> Upload(const T0 &a, const T1 &b,
                                                           const Ts &...rest) {
    return UploadParts(a, b, rest...);
  }

 private:
  static constexpr std::size_t kPart = 64;

  template <typename... Ts, std::size_t... I>
  static std::tuple<const Ts *...> PartAddresses(const char *base,
                                                 const std::size_t *offset,
                                                 std::index_sequence<I...>) {
    return std::tuple<const Ts *...>(reinterpret_cast<const Ts *>(base + offset[I])...);
  }

  template <typename... Ts>
  std::tuple<const Ts *...> UploadParts(const Ts &...parts) {
    static_assert(((alignof(Ts) <= kPart) && ...),
                  "DeviceArgs: the copy is only 64-byte aligned");
    constexpr std::size_t nparts = sizeof...(Ts);
    const std::size_t size[nparts] = {sizeof(Ts)...};
    const void *source[nparts] = {static_cast<const void *>(&parts)...};
    std::size_t offset[nparts];
    std::size_t nbytes = 0;
    for (std::size_t q = 0; q < nparts; ++q) {
      offset[q] = (nbytes + kPart - 1)/kPart*kPart;
      nbytes = offset[q] + size[q];
    }
    const std::size_t nwords = (nbytes + sizeof(Real) - 1)/sizeof(Real);
    if (buf_.extent(0) < nwords) {
      buf_ = DvceArray1D<Real>(Kokkos::view_alloc(Kokkos::WithoutInitializing,
                                                  "device_args"), nwords);
      last_.clear();
    }
    image_.assign(nwords, 0.0);
    char *bytes = reinterpret_cast<char *>(image_.data());
    for (std::size_t q = 0; q < nparts; ++q) {
      std::memcpy(bytes + offset[q], source[q], size[q]);
    }
    if (last_.size() != nwords ||
        std::memcmp(last_.data(), image_.data(), nwords*sizeof(Real)) != 0) {
      Kokkos::View<const Real *, Kokkos::HostSpace,
                   Kokkos::MemoryTraits<Kokkos::Unmanaged>> src(image_.data(), nwords);
      Kokkos::deep_copy(DevExeSpace(),
                        Kokkos::subview(buf_, std::make_pair(std::size_t{0}, nwords)),
                        src);
      last_.swap(image_);
    }
    return PartAddresses<Ts...>(reinterpret_cast<const char *>(buf_.data()), offset,
                                std::index_sequence_for<Ts...>{});
  }

  DvceArray1D<Real> buf_;
  std::vector<Real> image_;  // the bytes of this call
  std::vector<Real> last_;   // the bytes last written to buf_
};

//----------------------------------------------------------------------------------------
//! \struct DeviceResidentFunctor
//! \brief A kernel functor that runs the device copy of F (see DeviceResident).

template <class F>
struct DeviceResidentFunctor {
  const F *fn;
  template <class... Args>
  KOKKOS_FORCEINLINE_FUNCTION void operator()(const Args &...args) const {
    (*fn)(args...);
  }
};

//! The functor f as a launch that carries only the address of its device copy.  For
//! parallel_for bodies (the forwarding operator() hides the signature a reduction needs).
//! A closure is built anew by every call, so its padding bytes may differ from the last
//! call's even when its captures do not; that costs one more enqueued copy, nothing else.
template <class F>
DeviceResidentFunctor<F> DeviceResident(DeviceArgs &args, const F &f) {
  return DeviceResidentFunctor<F>{args.Upload(f)};
}

//! DeviceResident for the instantiations that need it, f itself for the others.
template <bool kResident, class F>
auto DeviceResidentIf(DeviceArgs &args, const F &f) {
  if constexpr (kResident) {
    return DeviceResident(args, f);
  } else {
    (void)args;
    return f;
  }
}

#endif  // UTILS_DEVICE_ARGS_HPP_
