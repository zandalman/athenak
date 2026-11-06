#ifndef MHD_MHD_LAT_EDGES_HPP_
#define MHD_MHD_LAT_EDGES_HPP_
//========================================================================================
// AthenaK astrophysical fluid dynamics & numerical relativity code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the AthenaK collaboration
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file mhd_lat_edges.hpp
//! \brief The block-surface edges the LAT EMF history and the force-free EMF moment store.
//!
//! The history and the moment are read by the endpoint exchanges of the EMF
//! (PackAndSendFluxFC / RecvAndUnpackFluxFC of efld), which touch the edges on the surface
//! of a block only, and by the mismatch kernels, whose result on every other edge is the
//! history minus itself, exactly zero.  They therefore hold the owned edges of a block
//! that lie on its surface: an edge along direction d lies on it iff one of its two
//! transverse node indices is at its minimum or maximum.  An edge is slot s of the
//! enumeration of CellOutsideBox, so the storage is (nmb, 1, 1, count).

#include <string>

#include "athena.hpp"
#include "mesh/mesh.hpp"
#include "utils/cell_box.hpp"

namespace mhd {

struct LATEdgeSurface {
  int K = 0, J = 0, I = 0;  // owned edges of the direction, origin (ks,js,is)
  CellBox interior;         // the edges off the surface, relative to the origin
  int count = 0;            // the edges on the surface

  // Edge of slot s, relative to the origin.
  KOKKOS_INLINE_FUNCTION
  void Decode(const int s, int &k, int &j, int &i) const {
    CellOutsideBox(s, I, J*I, interior.k0, interior.nk, interior.j0, interior.nj,
                   interior.i0, interior.ni, k, j, i);
  }
};

// Direction dir = 0, 1, 2 is the history's x1e, x2e, x3e.
inline LATEdgeSurface LATEdgeSurfaceOf(const RegionIndcs &indcs, const int dir) {
  LATEdgeSurface e;
  e.K = (dir == 2) ? indcs.nx3 : indcs.nx3 + 1;
  e.J = (dir == 1) ? indcs.nx2 : indcs.nx2 + 1;
  e.I = (dir == 0) ? indcs.nx1 : indcs.nx1 + 1;
  // The transverse nodes strictly inside the block; the edge's own direction is whole.
  e.interior.k0 = (dir == 2) ? 0 : 1;
  e.interior.nk = (dir == 2) ? e.K : e.K - 2;
  e.interior.j0 = (dir == 1) ? 0 : 1;
  e.interior.nj = (dir == 1) ? e.J : e.J - 2;
  e.interior.i0 = (dir == 0) ? 0 : 1;
  e.interior.ni = (dir == 0) ? e.I : e.I - 2;
  if (CellBoxEmpty(e.interior)) e.interior = CellBox{};
  e.count = e.K*e.J*e.I - e.interior.nk*e.interior.nj*e.interior.ni;
  return e;
}

// Zero the owned edges of a block off its surface in the blocks recv_indices(0..nrecv-1):
// the mismatch of an edge the exchanges never touched, which was the history minus itself
// when the history held every owned edge.
template <typename IndexView>
void ZeroInteriorLATEdges(const DvceArray4D<Real> &edges, const LATEdgeSurface &surf,
                          const IndexView &recv_indices, const int nrecv, const int ks,
                          const int js, const int is, const std::string &name) {
  const CellBox box = surf.interior;
  if (CellBoxEmpty(box) || nrecv <= 0) return;
  par_for(name, DevExeSpace(), 0, nrecv-1,
          ks + box.k0, ks + box.k0 + box.nk - 1, js + box.j0, js + box.j0 + box.nj - 1,
          is + box.i0, is + box.i0 + box.ni - 1,
  KOKKOS_LAMBDA(int a, int k, int j, int i) {
    edges(recv_indices(a),k,j,i) = 0.0;
  });
}

}  // namespace mhd

#endif // MHD_MHD_LAT_EDGES_HPP_
