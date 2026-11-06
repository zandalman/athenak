#ifndef MHD_MHD_STEP_WEIGHT_HPP_
#define MHD_MHD_STEP_WEIGHT_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file mhd_step_weight.hpp
//! \brief The weight with which a change of the register u0 reaches u^{n+1}.

#include "athena.hpp"
#include "driver/driver.hpp"
#include "mhd/mhd.hpp"

//----------------------------------------------------------------------------------------
//! \fn Real RKU0StepWeight
//! \brief The weight with which a change of u0 reaches u^{n+1}, made after the register
//! copy (updated = false) or after the update (updated = true) of `stage`.  The change
//! enters as (a0, a1) = (1, 0) in (u0, u1), and the rest of the step combines the
//! registers as MHD::CopyCons (u1 += delta u0 at the later stages of RK4()4[2S]) and
//! MHD::RKUpdate (u0 = gam0 u0 + gam1 u1 + dt L, MHD::TransportStageWeights) do.  Before
//! the stage-1 copy it is 1, the consistency of the scheme, and it is 1 after the last
//! update.  The fluxes are conservative, so the sum over the cells of u^{n+1} is the
//! sum of u^n, the boundary flux, and the sum of these weights times the rewrites.
//! DynGRMHD::U0StepWeight calls this.
inline Real RKU0StepWeight(const Driver *pdrive, const mhd::MHD *pmhd, int stage,
                           bool updated) {
  if (pdrive == nullptr || stage < 1) return 1.0;
  const bool low_storage = (pdrive->integrator == "rk4");
  Real a0 = 1.0, a1 = 0.0;
  for (int s = stage; s <= pdrive->nexp_stages; ++s) {
    if (s > stage && low_storage) a1 += pdrive->delta[s-1]*a0;
    if (s > stage || !updated) {
      Real gam0, gam1;
      pmhd->TransportStageWeights(pdrive, s, gam0, gam1);
      a0 = gam0*a0 + gam1*a1;
    }
  }
  return a0;
}

#endif  // MHD_MHD_STEP_WEIGHT_HPP_
