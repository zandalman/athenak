#ifndef OUTPUTS_RESTART_PRIMITIVES_HPP_
#define OUTPUTS_RESTART_PRIMITIVES_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file restart_primitives.hpp
//! \brief The primitives an ordinary GRMHD checkpoint carries for the inversion's
//! kept-previous branch.
//!
//! A failed GRMHD inversion keeps the primitives the cell last published (KORAL's
//! no-fixup branch; ideal_grmhd_c2p_impl.hpp, primitive_solver_hyd.hpp), which a
//! checkpoint of conserved variables does not hold.  The writer repeats its recovery cold
//! before writing (RestartOutput::LoadOutputData) so that the restart, which recovers
//! cold, continues from the same primitives.  That recovery keeps the primitives it finds
//! in w0, as every recovery of the run does, and the checkpoint carries those primitives
//! (w0 as the writer's recovery found it, ghost zones included) and whether the recovery
//! could keep them (MHD::w0_topology_version current).  The restart puts them back before
//! its own recovery (Driver::InitBoundaryValuesAndPrimitives), so both recoveries start
//! from the same u0, w0, stamp and cold cache and give the same result, and a checkpoint
//! no longer resets the cells that fail at it to the atmosphere.  KORAL's restart files
//! hold the primitives for the same reason.  A fixed-metric GR Hydro fluid carries its w0
//! the same way in its own section (<hydro_restart>); its recovery repeats nothing before
//! the write (it keeps no warm-start cache), so w0 is carried as the run holds it.
//!
//! The payload lives here between the reader (ProblemGenerator, which runs before the
//! driver exists) and the driver's recovery, and between the writer's load and its
//! background write.  Both views are released at Kokkos::finalize.

#include "athena.hpp"

namespace restart_primitives {

struct Payload {
  HostArray5D<Real> written;     // w0 before the writer's recovery; empty: none written
  bool written_kept = false;     // the writer's recovery could keep previous primitives
  DvceArray5D<Real> read;        // w0 of the checkpoint being restarted
  bool read_pending = false;
  bool read_kept = false;
  // The same for a fixed-metric GR Hydro fluid (<hydro_restart> w0_count, w0_kept).
  HostArray5D<Real> hydro_written;
  bool hydro_written_kept = false;
  DvceArray5D<Real> hydro_read;
  bool hydro_read_pending = false;
  bool hydro_read_kept = false;
  bool hook_registered = false;
};

inline Payload &State() {
  static Payload p;
  if (!p.hook_registered) {
    p.hook_registered = true;
    Kokkos::push_finalize_hook([]() {
      Payload &q = State();
      q.written = HostArray5D<Real>();
      q.read = DvceArray5D<Real>();
      q.read_pending = false;
      q.hydro_written = HostArray5D<Real>();
      q.hydro_read = DvceArray5D<Real>();
      q.hydro_read_pending = false;
    });
  }
  return p;
}

}  // namespace restart_primitives

#endif  // OUTPUTS_RESTART_PRIMITIVES_HPP_
