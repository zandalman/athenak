#ifndef EOS_PRIMITIVE_SOLVER_PS_TYPES_HPP_
#define EOS_PRIMITIVE_SOLVER_PS_TYPES_HPP_
//========================================================================================
// PrimitiveSolver equation-of-state framework
// Copyright(C) 2023 Jacob M. Fields <jmf6719@psu.edu>
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file ps_types.hpp
//  \brief contains some basic type definitions consistent with Athena++.
//
//  Ideally this file shouldn't be required when the code is dropped into Athena.
//  Therefore, all type definitions should be consistent with Athena.
//

#include "athena.hpp"

#define MAX_SPECIES 8
#define NHYDRO ((5) + (MAX_SPECIES))

enum ConsIndex {CDN=0, CSX=1, CSY=2, CSZ=3, CTA=4, CYD=5, NCONS=(NHYDRO)};
// FIXME: Make sure that the position of IYF makes sense.
// It should be okay, since if we're not using any species,
// IBY gets aliased to 6, and IYF should never get called.
// Note that NPRIM does not include IBY and IBZ because NHYDRO doesn't.
enum PrimIndex {PRH=0,PVX=1, PVY=2, PVZ=3, PPR=4, PTM=5, PYF=6, NPRIM=((NHYDRO)+1)};
enum SpatialMetricIndex{S11=0, S12=1, S13=2, S22=3, S23=4, S33=5, NSPMETRIC=6};

class SupportsEntropy{};
//! Tag for EOS policies that label their adiabats: a gas entropy s(n, T), constant along
//! every solution of the first law at zero heat (d eps = -p d(1/rho)), with an inverse
//! T(n, s) at fixed n.  It is the quantity the GR entropy fallback transports (D s rides
//! the mass flux) and inverts when KORAL's energy inversion fails (koral_lite u2p.c:
//! 229-263, U2P_HOT then U2P_ENTROPY).  Every GR policy must carry it.
class SupportsGasEntropy{};
class SupportsChemicalPotentials{};

#endif  // EOS_PRIMITIVE_SOLVER_PS_TYPES_HPP_
