#ifndef MHD_MHD_HPP_
#define MHD_MHD_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file mhd.hpp
//  \brief definitions for MHD class

#include <cstdint>
#include <map>
#include <memory>
#include <string>

#include "athena.hpp"
#include "parameter_input.hpp"
#include "tasklist/task_list.hpp"
#include "bvals/bvals.hpp"

// forward declarations
class EquationOfState;
class Coordinates;
class Viscosity;
class Resistivity;
class Conduction;
class SourceTerms;
class OrbitalAdvectionCC;
class OrbitalAdvectionFC;
class ShearingBoxCC;
class ShearingBoxFC;
class Driver;

// function ptr for user-defined MHD boundary functions enrolled in problem generator
namespace mhd {
using MHDBoundaryFnPtr = void (*)(int m, Mesh* pm, MHD* pmhd, DvceArray5D<Real> &u);
}

// constants that enumerate MHD Riemann Solver options
enum class MHD_RSolver {advect, llf, hlle, hlld, roe,   // non-relativistic
                        llf_sr, hlle_sr, hlld_sr,       // SR
                        llf_gr, hlle_gr, hlld_gr};       // GR

//----------------------------------------------------------------------------------------
//! \struct MHDTaskIDs
//  \brief container to hold TaskIDs of all mhd tasks

struct MHDTaskIDs {
  TaskID savest;
  TaskID irecv;
  TaskID copyu;
  TaskID flux;
  TaskID prep_fofc_flags;
  TaskID send_fofc_flags;
  TaskID recv_fofc_flags;
  TaskID replace_fofc;
  TaskID sendf;
  TaskID recvf;
  TaskID rkupdt;
  TaskID duale;
  TaskID gas_entropy;
  TaskID srctrms;
  TaskID sendu_oa;
  TaskID recvu_oa;
  TaskID restu;
  TaskID sendu;
  TaskID recvu;
  TaskID sendu_shr;
  TaskID recvu_shr;
  TaskID efld;
  TaskID sende;
  TaskID recve;
  TaskID ct;
  TaskID sendb_oa;
  TaskID recvb_oa;
  TaskID restb;
  TaskID sendb;
  TaskID recvb;
  TaskID sendb_shr;
  TaskID recvb_shr;
  TaskID bcs;
  TaskID prol;
  TaskID c2p;
  TaskID newdt;
  TaskID csend;
  TaskID crecv;
};

namespace mhd {

//----------------------------------------------------------------------------------------
//! \class MHD

class MHD {
 public:
  MHD(MeshBlockPack *ppack, ParameterInput *pin);
  ~MHD();

  // data
  ReconstructionMethod recon_method;
  MHD_RSolver rsolver_method;
  EquationOfState *peos;   // chosen EOS

  int nmhd;                // number of mhd variables (5/4 for ideal/isothermal EOS)
  int nscalars;            // number of passive scalars
  int naux = 0;            // number of hidden auxiliary MHD variables
  int nvars = 0;           // total MHD-carried variables = nmhd+nscalars+naux
  int dual_energy_idx = -1;
  bool use_dual_energy = false;
  // The dual-energy formalism is non-relativistic only: an internal-energy DENSITY with
  // an operator-split p dV update and the interface velocities the Riemann solver
  // produced.  use_dual_energy and dual_energy_pdv are therefore the same switch.
  bool dual_energy_pdv = false;
  bool dual_energy_needs_init = false;
  // The one threshold of the Newtonian formalism, as hydro's (Hydro::dual_energy_eta1):
  // the Bryan et al. eta1 test, with the magnetic test of the energy split beside it
  // (eos_general::MHDEnergyChannelResolvesEint).  The auxiliary always holds the thermal
  // state the inversion published, so there is no separate eta2.
  Real dual_energy_eta1 = 1.0e-3;
  // General relativity (fixed or dynamical metric) carries the gas entropy instead, in
  // the same hidden column (index nmhd + nscalars): u0 = D s, w0 = s, with s per unit rest
  // mass.  It is always on in GR, never in Newtonian runs, and has no deck key.  It is a
  // passive scalar riding the final mass flux; the inversions use it only where the hot
  // inversion fails, and it is resynchronized from the state at the end of every full
  // step.
  int gas_entropy_idx = -1;
  // The column does not yet hold the entropy of the state: a fresh start (the problem
  // generator's P2C writes D*0), a GR checkpoint written without the column, or a remap
  // from one.  Driver::Initialize resyncs it after the first primitive recovery.
  bool gas_entropy_init_pending = false;
  bool EntropyInitPending() const { return gas_entropy_init_pending; }
  // s = s(published primitives) and u0 = D s on every cell of the pack (ghosts included
  // when `ghosts`), through the metric's own hook: EquationOfState::ResyncGasEntropy on a
  // fixed metric, dyngr::DynGRMHD::ResyncGasEntropy on a dynamical one.  Clears
  // gas_entropy_init_pending.  No-op outside GR.
  void ResyncGasEntropy(bool ghosts);
  bool time_evolving = false;
  DvceArray5D<Real> u0;    // conserved variables
  DvceArray5D<Real> w0;    // primitive variables
  // The key block this fluid reads its parameters from, and whether it carries a magnetic
  // field.  A fluid with a field is the only kind this release builds.
  const std::string &KeyBlock() const { return key_block_; }
  bool HasField() const { return has_field_; }
  // Mesh::topology_version at the last recovery that wrote every cell of w0, or ~0 before
  // the first: a failed GRMHD inversion keeps the primitives w0 holds only while this is
  // the current topology version.
  std::uint64_t w0_topology_version = ~std::uint64_t{0};
  DvceFaceFld4D<Real> b0;  // face-centered magnetic fields
  DvceArray5D<Real> bcc0;  // cell-centered magnetic fields

  DvceArray5D<Real> coarse_u0;    // conserved variables on 2x coarser grid (for SMR/AMR)
  DvceArray5D<Real> coarse_w0;    // primitive variables on 2x coarser grid (for SMR/AMR)
  DvceFaceFld4D<Real> coarse_b0;  // face-centered B-field on 2x coarser grid

  // Objects containing boundary communication buffers and routines for u and b
  MeshBoundaryValuesCC *pbval_u;
  MeshBoundaryValuesFC *pbval_b;
  MeshBoundaryValuesCC *pbval_fofc = nullptr;
  MHDBoundaryFnPtr MHDBoundaryFunc[6];

  // Orbital advection and shearing box BCs
  OrbitalAdvectionCC *porb_u = nullptr;
  OrbitalAdvectionFC *porb_b = nullptr;
  ShearingBoxCC *psbox_u = nullptr;
  ShearingBoxFC *psbox_b = nullptr;

  // Object(s) for extra physics (viscosity, resistivity, thermal conduction, srcterms)
  Viscosity *pvisc = nullptr;
  Resistivity *presist = nullptr;
  Conduction *pcond = nullptr;
  SourceTerms *psrc = nullptr;

  // following only used for time-evolving flow
  DvceArray5D<Real> u1;       // conserved variables, second register
  DvceFaceFld4D<Real> b1;     // face-centered magnetic fields, second register
  // Face fluxes of the conserved variables, stored on the band the flux kernels fill:
  // one face beyond the active domain on each side along the face normal (FOFC replaces
  // the flux at faces is-1 and ie+2) and one cell beyond it transversely (CT needs the
  // area-averaged EMFs at every transverse cell edge), so x1f is
  // (nmb, nvars, nx3+2, nx2+2, nx1+3) with origin (ks-1, js-1, is-1), and likewise
  // x2f/x3f.  A degenerate dimension keeps its single cell at origin 0.  No writer or
  // reader -- the Riemann and FOFC kernels, the divergence, CornerE, the LAT ledger, the
  // diffusion and gravity terms, the sink excision and the flux-correction exchange --
  // goes further, so the ghost-extended layout held 35% dead storage.  Access through
  // FluxBand(); the exchange takes FluxOrigin().  dual_vf and the CornerE scratch
  // arrays below share the same band and origin.
  DvceFaceFld5D<Real> uflx;   // fluxes of conserved quantities on cell faces
  int flux_ko = 0;
  int flux_jo = 0;
  int flux_io = 0;
  BandView5D<Real> FluxBand(const DvceArray5D<Real> &a) const {
    return BandView5D<Real>{a, flux_ko, flux_jo, flux_io};
  }
  BandFaceFld5D<Real> FluxBand(const DvceFaceFld5D<Real> &f) const {
    return BandFaceFld5D<Real>{FluxBand(f.x1f), FluxBand(f.x2f), FluxBand(f.x3f)};
  }
  BandView4D<Real> EmfBand(const DvceArray4D<Real> &a) const {
    return BandView4D<Real>{a, flux_ko, flux_jo, flux_io};
  }
  FaceFldOrigin FluxOrigin() const { return FaceFldOrigin{flux_ko, flux_jo, flux_io}; }
  // Receiver-local union of cells changed by delayed LAT CC reflux or by the adjacent
  // CT face correction.  Delayed C2P/source reconciliation must use this cell mask rather
  // than reprocessing every cell in a receiver MeshBlock.
  // Shared one-cell mask used by delayed LAT correction and by the conservative
  // coarse/fine FOFC order exchange.  The latter is allocated even when LAT is disabled.
  DvceArray5D<Real> lat_correction_mask;
  DvceArray5D<Real> coarse_fofc_mask;
  // dual-energy face payload: the contact velocity of each face and the heat of each side
  // (eos_general::DualFaceComponent, eos/dual_energy_shock_rule.hpp)
  DvceFaceFld5D<Real> dual_vf;
  // The dual-energy shock rule's flag of each owned cell, decided on the step-start state
  // at the first stage and held through the step's later stages, and the offsets of its
  // step-final guard (ApplyDualEnergyFormalism; the components as Hydro::dual_shock_debt,
  // eos_general::DualEnergyShockedStageEint).
  DvceArray4D<bool> dual_shocked;
  DvceArray5D<Real> dual_shock_debt;
  // With dual energy, the edge EMFs other modules add to efld after CornerE (resistivity,
  // the 2D shearing box), kept apart for the field E carried (MHDFieldCarriedByE): it
  // corrects only the mismatch between the Riemann solver's face EMFs and CornerE's.
  DvceEdgeFld4D<Real> dual_efld_added;
  DvceEdgeFld4D<Real> efld;   // edge-centered electric fields (fluxes of B)
  // temporary variables used to store face-centered electric fields returned by RS;
  // each lives on the band of the face flux it accompanies (see uflx), EmfBand() access
  DvceArray4D<Real> e3x1, e2x1;
  DvceArray4D<Real> e1x2, e3x2;
  DvceArray4D<Real> e2x3, e1x3;
  // global per-face L/R buffers for the split-kernel flux path: primitive variables
  // carried by MHD and reconstructed cell-centered B-field (3 components)
  DvceArray5D<Real> wl3d, wr3d;
  DvceArray5D<Real> bl3d, br3d;
  int split_recon_chunk_nmb = 32;
  DualArray1D<Real> dtnew_eachmb;  // local MHD CFL timestep for each MeshBlock
  DualArray1D<int> sink_block_indices;  // reusable active blocks intersecting the sink
  Real dtnew;
  // Optional fixed-spacetime GRMHD fast-magnetosonic CFL estimate.  The default
  // remains false so the legacy unit-coordinate-speed GR timestep is untouched.
  bool gr_dt = false;

  // following used for time derivatives in computation of jcon
  bool wbcc_saved = false;
  DvceArray5D<Real> wsaved;
  DvceArray5D<Real> bccsaved;
  DualArray1D<Real> wbcc_saved_dt;  // elapsed time represented by each saved state

  // following used for FOFC algorithm
  DvceArray4D<bool> fofc;  // flag for each cell to indicate if FOFC is needed
  DvceArray5D<bool> fofc_scal;  // flag to indicate if FOFC for scalar is needed
  bool use_fofc = false;   // flag to enable FOFC

  // container to hold names of TaskIDs
  MHDTaskIDs id;

  // functions...
  void SetSaveWBcc();
  void AssembleMHDTasks(std::map<std::string, std::shared_ptr<TaskList>> tl);
  // Two-phase form used when a radiation module hangs its graph between the halves.  Part A
  // ends at id.srctrms -- the point where the RK update of u0, the corner-E exchange, the
  // CT update of b0, the dual-energy step and the source terms have all run, so u0(IEN)
  // and b0 are stage-consistent and the couple's emag subtraction (B2) is well defined.
  // Part B resumes at the state exchange with BOTH SendU_OA and SendB_OA gated on the
  // radiation couple that has just rewritten the gas u0.  Unlike hydro, AssembleMHDTasks
  // is the COMPOSITION of the two halves, so the radiation-free graph is identical by
  // construction (TaskIDs come from AddTask call order).
  void AssembleMHDTasksPartA(std::map<std::string, std::shared_ptr<TaskList>> tl);
  void AssembleMHDTasksPartB(std::map<std::string, std::shared_ptr<TaskList>> tl,
                             TaskID rad_gate);
  // ...in "before_timeintegrator" task list
  TaskStatus SaveMHDState(Driver *d, int stage);
  // ...in "before_stagen_tl" task list
  TaskStatus InitRecv(Driver *d, int stage);
  // ...in "stagen_tl" task list
  TaskStatus CopyCons(Driver *d, int stage);
  TaskStatus Fluxes(Driver *d, int stage);
  TaskStatus PrepareFOFCFlags(Driver *d, int stage);
  TaskStatus SendFOFCFlags(Driver *d, int stage);
  TaskStatus RecvFOFCFlags(Driver *d, int stage);
  TaskStatus ReplaceCanonicalFOFC(Driver *d, int stage);
  TaskStatus SendFlux(Driver *d, int stage);
  TaskStatus RecvFlux(Driver *d, int stage);
  TaskStatus RKUpdate(Driver *d, int stage);
  // RK base weights of the stage for u0 (RKUpdate), the face fields (CT) and every
  // FOFC candidate.  The driver's (gam0, gam1).
  void TransportStageWeights(const Driver *d, int stage, Real &gam0, Real &gam1) const;
  TaskStatus DualEnergyStep(Driver *d, int stage);
  // End-of-step resync of the GR gas entropy column (ResyncGasEntropy).
  TaskStatus GasEntropyEndpoint(Driver *d, int stage);
  TaskStatus MHDSrcTerms(Driver *d, int stage);
  TaskStatus SendU_OA(Driver *d, int stage);
  TaskStatus RecvU_OA(Driver *d, int stage);
  TaskStatus RestrictU(Driver *d, int stage);
  TaskStatus SendU(Driver *d, int stage);
  TaskStatus RecvU(Driver *d, int stage);
  TaskStatus SendU_Shr(Driver *d, int stage);
  TaskStatus RecvU_Shr(Driver *d, int stage);
  TaskStatus CornerE(Driver *d, int stage);
  TaskStatus EField(Driver *d, int stage);
  TaskStatus SendE(Driver *d, int stage);
  TaskStatus RecvE(Driver *d, int stage);
  TaskStatus CT(Driver *d, int stage);
  TaskStatus SendB_OA(Driver *d, int stage);
  TaskStatus RecvB_OA(Driver *d, int stage);
  TaskStatus RestrictB(Driver *d, int stage);
  TaskStatus SendB(Driver *d, int stage);
  TaskStatus RecvB(Driver *d, int stage);
  TaskStatus SendB_Shr(Driver *d, int stage);
  TaskStatus RecvB_Shr(Driver *d, int stage);
  TaskStatus ApplyPhysicalBCs(Driver* pdrive, int stage);
  TaskStatus Prolongate(Driver* pdrive, int stage);
  TaskStatus ConToPrim(Driver *d, int stage);
  TaskStatus NewTimeStep(Driver *d, int stage);
  // ...in "after_stagen_tl" task list
  TaskStatus ClearSend(Driver *d, int stage);
  TaskStatus ClearRecv(Driver *d, int stage);  // also in Driver::Initialize

  // CalculateFluxes function templated over Riemann Solvers
  template <MHD_RSolver T>
  void CalculateFluxes(Driver *d, int stage);
  void ApplyExcisionSinkBoundary(const Real t);

  // first-order flux correction
  void FOFC(Driver *d, int stage);
  void BuildFOFCTrial(Driver *d, int stage, int il, int iu, int jl, int ju, int kl,
                      int ku);

  // dual-energy support
  void InitializeDualEnergyFieldFromTotal();
  void ApplyDualEnergyFormalism(const Real dt, const int stage, const int nstages,
                                const Real gam0, const Real gam1, const Real delta);
  void ResolveDualEnergyChannel();
  void SynchronizeRestrictedDualEnergyField(bool lat_active_only = false);
  void SynchronizeRestrictedDualEnergyField(DvceArray5D<Real> &coarse_cons,
                                            DvceArray5D<Real> &coarse_prim,
                                            bool lat_active_only = false);
  void RepairRefinedDualEnergyState(DualArray1D<int> &n2o, DualArray1D<int> &rflag,
                                    const int new_gids, const int new_nmb_local);
  bool ResizeMeshBlockStorage(int nmb, bool exact = false,
                              bool allow_shrink = false);

  // FOFC trial state.  Under dynamical GR its one user, DynGRMHDPS::FOFC, touches only
  // the cell band [ks-1,ke+1]x[js-1,je+1]x[is-1,ie+1] of the face fluxes, so there it is
  // stored on that band with the flux origin; the other backends keep the ghost-extended
  // layout at origin 0.  Access through FofcTrialBand().
  DvceArray5D<Real> utest, bcctest;
  int fofc_trial_ko = 0;
  int fofc_trial_jo = 0;
  int fofc_trial_io = 0;
  BandView5D<Real> FofcTrialBand(const DvceArray5D<Real> &a) const {
    return BandView5D<Real>{a, fofc_trial_ko, fofc_trial_jo, fofc_trial_io};
  }

 private:
  std::string key_block_ = "mhd";
  bool has_field_ = true;
  bool FOFCMaskExchangeEnabled() const;
  void ProlongateFaceFieldsFC_();
  bool fofc_replacement_only_ = false;
  MeshBlockPack* pmy_pack;   // ptr to MeshBlockPack containing this MHD
  // cell-centered electric fields for CornerE, on the cells [ks-1,ke+1]x[js-1,je+1]x
  // [is-1,ie+1] that the SG07 corner average reads (the flux band origin, EmfBand())
  DvceArray4D<Real> e1_cc, e2_cc, e3_cc;
};

} // namespace mhd
#endif // MHD_MHD_HPP_
