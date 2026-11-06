#ifndef UTILS_CELL_BOX_HPP_
#define UTILS_CELL_BOX_HPP_
//========================================================================================
// AthenaK astrophysical fluid dynamics & numerical relativity code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the AthenaK collaboration
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file cell_box.hpp
//! \brief An index box and the enumeration of the cells of an outer box outside it.

#include "athena.hpp"

// A box of cells [k0,k0+nk) x [j0,j0+nj) x [i0,i0+ni); nk == 0 is no box.
struct CellBox {
  int k0 = 0, nk = 0, j0 = 0, nj = 0, i0 = 0, ni = 0;
};

KOKKOS_INLINE_FUNCTION
bool CellBoxEmpty(const CellBox &b) {
  return b.nk <= 0 || b.nj <= 0 || b.ni <= 0;
}

// Cell b, in k-j-i order, of the cells of an outer box (nji per plane, ncells1 per row)
// outside the inner box [k0,k0+nk) x [j0,j0+nj) x [i0,i0+ni), whose coordinates are
// relative to the outer box and which must lie inside it.  The inner box is empty for
// nk = 0.
KOKKOS_INLINE_FUNCTION
void CellOutsideBox(int b, const int ncells1, const int nji,
                    const int k0, const int nk, const int j0, const int nj,
                    const int i0, const int ni, int &k, int &j, int &i) {
  const int row = ncells1 - ni;   // cells of a row through the box
  const int plane = nji - nj*ni;  // cells of a plane through the box
  if (b < k0*nji) {
    k = b/nji;
    b -= k*nji;
  } else if (b - k0*nji < nk*plane) {
    b -= k0*nji;
    k = k0 + b/plane;
    b -= (k - k0)*plane;
    if (b >= j0*ncells1 && b - j0*ncells1 < nj*row) {
      b -= j0*ncells1;
      j = j0 + b/row;
      b -= (j - j0)*row;
      i = (b < i0) ? b : b + ni;
      return;
    }
    if (b >= j0*ncells1) b += nj*ni;
  } else {
    b -= k0*nji + nk*plane;
    k = k0 + nk + b/nji;
    b -= (k - k0 - nk)*nji;
  }
  j = b/ncells1;
  i = b - j*ncells1;
}

#endif // UTILS_CELL_BOX_HPP_
