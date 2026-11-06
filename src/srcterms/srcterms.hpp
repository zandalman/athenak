#ifndef SRCTERMS_SRCTERMS_HPP_
#define SRCTERMS_SRCTERMS_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file srcterms.hpp
//! \brief Data, functions, and classes to implement various source terms in the hydro
//! and/or MHD equations of motion.  Currently implemented:
//!  (1) constant (gravitational) acceleration - for RTI
//!  (2) shearing box in 2D (x-z), for both hydro and MHD
//!  (3) random forcing to drive turbulence - implemented in TurbulenceDriver class

#include <cstdint>
#include <map>
#include <string>

#include "athena.hpp"
#include "mesh/mesh.hpp"
#include "parameter_input.hpp"

//----------------------------------------------------------------------------------------
//! \class SourceTerms
//! \brief data and functions for physical source terms

class SourceTerms {
 public:
  SourceTerms(std::string block, MeshBlockPack *pp, ParameterInput *pin);
  // Which fluid's dual-energy auxiliary a given conserved array belongs to, if any.
  // The cooling terms are handed a bare u0 and do not otherwise know whether they are
  // charging hydro or MHD, so the owner is resolved by identity rather than by a stored
  // block name.  Returns idx = -1 when the formalism is off for that fluid.
  void DualEnergyTarget(const DvceArray5D<Real> &u0, int &idx, bool &pdv) const;
  ~SourceTerms();

  // data
  // flags for various source terms
  bool const_accel;
  bool ism_cooling;
  bool rel_cooling;
  bool disk_cooling;
  bool rad_beam;
  bool self_gravity;
  bool external_bh_gravity;
  // Sink-particle gravity acting on the gas.  Sinks never enter the Poisson RHS, so this
  // is a separate additive term rather than part of the multigrid potential.
  bool sink_gravity;

  // new timestep
  Real dtnew;
  DualArray1D<Real> dtnew_eachmb;

  // data for constant accel
  Real const_accel_val;   // magnitude of accn
  int const_accel_dir;    // direction of accn

  // data for gravitational source terms
  Real rho_grav_min;
  Real rho_external_bh_min;
  Real external_bh_dt_factor;

  // data for ISM cooling
  Real hrate;

  // data for relativistic cooling
  Real crate_rel;
  Real cpower_rel;

  // data for Noble-style disk cooling
  Real disk_cooling_h_over_r;
  Real disk_cooling_noble_s;
  Real disk_cooling_noble_q;
  // where it acts (all off by default: every cell above the density gate is cooled)
  Real disk_cooling_sigma_max = 0.0;   // no cooling where b^2/rho >= this (> 0 enables)
  Real disk_cooling_sin_power = 0.0;   // cooling time / sin^n(theta) about the disk axis
  bool disk_cooling_bound_only = false;  // cool only bound gas, -h u_t < 1

  // data for radiation beam source
  Real dii_dt;            // injection rate
  Real pos1, pos2, pos3;  // position of source
  Real dir1, dir2, dir3;  // direction of source
  Real width, spread;     // spatial width of source region, spread in angles

  // functions
  void ApplySrcTerms(const DvceArray5D<Real> &w0, const EOS_Data &eos,
                     const Real bdt, DvceArray5D<Real> &u0);
  void ApplySrcTerms(DvceArray5D<Real> &i0, const Real bdt);
  void ConstantAccel(const DvceArray5D<Real> &w0, const EOS_Data &eos,
                     const Real bdt, DvceArray5D<Real> &u0);
  void ISMCooling(const DvceArray5D<Real> &w0, const EOS_Data &eos,
                  const Real bdt, DvceArray5D<Real> &u0);
  void RelCooling(const DvceArray5D<Real> &w0, const EOS_Data &eos,
                  const Real bdt, DvceArray5D<Real> &u0);
  void DiskCooling(const DvceArray5D<Real> &w0, const EOS_Data &eos,
                   const Real bdt, DvceArray5D<Real> &u0);
  void Gravity(const DvceArray5D<Real> &w0, const EOS_Data &eos,
               const Real bdt, DvceArray5D<Real> &u0);
  void SinkGravity(const DvceArray5D<Real> &w0, const EOS_Data &eos,
                   const Real bdt, DvceArray5D<Real> &u0);
  // coarse/fine gravitational work, added (sign +1) to the energy flux a fine block
  // restricts to its coarser neighbours, whose own values are restored exactly by
  // sign -1 (Hydro/MHD::SendFlux)
  void CoarseFineGravityWorkFlux(const DvceArray5D<Real> &w0, const EOS_Data &eos,
                                 const Real sign);
  void BeamSource(DvceArray5D<Real> &i0, const Real bdt);
  void NewTimeStep(const DvceArray5D<Real> &w0, const EOS_Data &eos);
  bool ResizeMeshBlockStorage(int nmb, bool exact = false);

 private:
  // The fine block's own coarse/fine energy face fluxes, saved by
  // CoarseFineGravityWorkFlux(+1) and put back by (-1): (a + d) - d is a only to 1 ulp.
  DvceArray3D<Real> cf_work_saved_flux_;
  MeshBlockPack *pmy_pack;
  // The LAT gravity gate (Gravity) visits only the faces its stage wrote, and one full
  // sweep re-establishes acc == cseen on the first call and after a topology change.
  bool lat_gate_swept_ = false;
  std::uint64_t lat_gate_topology_ = 0;
  int lat_gate_nmb_ = 0;
};

#endif  // SRCTERMS_SRCTERMS_HPP_
