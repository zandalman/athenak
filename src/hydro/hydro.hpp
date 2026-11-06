#ifndef HYDRO_HYDRO_HPP_
#define HYDRO_HYDRO_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file hydro.hpp
//  \brief definitions for Hydro class

#include <cstdint>
#include <map>
#include <memory>
#include <string>

#include "athena.hpp"
#include "parameter_input.hpp"
#include "tasklist/task_list.hpp"
#include "bvals/bvals.hpp"
#include "utils/phase_communicator.hpp"

// forward declarations
class EquationOfState;
class Coordinates;
class Viscosity;
class Conduction;
class SourceTerms;
class OrbitalAdvectionCC;
class ShearingBoxCC;
class Driver;

// constants that enumerate Hydro Riemann Solver options
enum class Hydro_RSolver {advect, llf, hlle, hllc, roe,    // non-relativistic
                          llf_sr, hlle_sr, hllc_sr,        // SR
                          llf_gr, hlle_gr};                // GR

//----------------------------------------------------------------------------------------
//! \struct HydroTaskIDs
//  \brief container to hold TaskIDs of all hydro tasks

struct HydroTaskIDs {
  TaskID irecv;
  TaskID statefix_pre;
  TaskID statefix;
  TaskID statefix_post;
  TaskID copyu;
  TaskID flux;
  TaskID prep_fofc_flags;
  TaskID send_fofc_flags;
  TaskID recv_fofc_flags;
  TaskID replace_fofc;
  TaskID close_drain;
  TaskID sendf;
  TaskID recvf;
  TaskID rkupdt;
  TaskID duale;
  TaskID srctrms;
  TaskID sendu_oa;
  TaskID recvu_oa;
  TaskID restu;
  TaskID sendu;
  TaskID recvu;
  TaskID sendu_shr;
  TaskID recvu_shr;
  TaskID bcs;
  TaskID prol;
  TaskID c2p;
  TaskID gas_entropy;
  TaskID newdt;
  TaskID csend;
  TaskID crecv;
  TaskID latdense;
};

namespace hydro {

//----------------------------------------------------------------------------------------
//! \class Hydro

class Hydro {
 public:
  Hydro(MeshBlockPack *ppack, ParameterInput *pin);
  ~Hydro();

  // data
  ReconstructionMethod recon_method;
  Hydro_RSolver rsolver_method;
  EquationOfState *peos;  // chosen EOS

  int nhydro;             // number of hydro variables (5/4 for ideal/isothermal EOS)
  int nscalars;           // number of user passive scalars
  int naux = 0;           // number of hidden auxiliary hydro variables
  int nvars = 0;          // total hydro-carried variables = nhydro+nscalars+naux
  // The dual-energy formalism is non-relativistic only (the internal-energy auxiliary
  // with its p dV step): use_dual_energy and dual_energy_pdv are the same switch.
  int dual_energy_idx = -1;
  bool use_dual_energy = false;
  bool dual_energy_pdv = false;
  // GR carries the gas entropy per unit rest mass in the same hidden column instead
  // (u0 = D s, w0 = s; see MHD::gas_entropy_idx).  Always on in GR, no deck key.
  int gas_entropy_idx = -1;
  bool gas_entropy_init_pending = false;
  bool EntropyInitPending() const { return gas_entropy_init_pending; }
  // s from the published primitives and u0 = D s (EquationOfState::ResyncGasEntropy),
  // ghosts included when `ghosts`.  Clears gas_entropy_init_pending.  No-op outside GR.
  void ResyncGasEntropy(bool ghosts);
  bool dual_energy_needs_init = false;
  // Mesh topology on which the last recovery of every cell of every block (ghosts
  // included) wrote w0, or ~0 before the first (as MHD::w0_topology_version).  A failed
  // fixed-metric GR inversion keeps the primitives w0 holds only while this is the
  // current Mesh::topology_version, i.e. while they are that cell's own.  A pure
  // rebalance carries w0 and re-stamps it.
  std::uint64_t w0_topology_version = ~std::uint64_t{0};
  Real dual_energy_eta1 = 1.0e-3;
  // Weight applied to floor-induced energy changes recorded by ConsToPrim: the
  // contribution of the current RK stage's state to the completed step (1 outside
  // the RK stages).
  Real floor_energy_weight = 1.0;
  // Device-side per-block floor energy tally for EOS paths whose C2P is a plain
  // parallel_for (tabulated EOS) and for the Newtonian radiation couple: filled by
  // atomic adds from the first gravity energy window on (floor_energy_ledger), reduced
  // over its whole extent and zeroed at each window end.  A per-rank sum: the slot is
  // the booking block's index at the time, kept across regrids and rebalances
  // (Hydro::ResizeMeshBlockStorage grows it and never shrinks it).
  bool floor_energy_ledger = false;
  DvceArray1D<Real> floor_energy_block;
  // The fixed-metric GR hydro recovery's share of Mesh::ecounter.eos_floor_energy/mass:
  // atomic adds into spread slots (IdealGRHydro::ConsToPrim), read back once per launch.
  DvceArray1D<Real> gr_floor_ledger;
  bool time_evolving = false;
  DvceArray5D<Real> u0;   // conserved variables
  DvceArray5D<Real> w0;   // primitive variables

  DvceArray5D<Real> coarse_u0;  // conserved variables on 2x coarser grid (for SMR/AMR)
  DvceArray5D<Real> coarse_w0;  // primitive variables on 2x coarser grid (for SMR/AMR)

  // Boundary communication buffers and functions for u
  MeshBoundaryValuesCC *pbval_u;

  // Orbital advection and shearing box BCs
  OrbitalAdvectionCC *porb_u = nullptr;
  ShearingBoxCC *psbox_u = nullptr;

  // Object(s) for extra physics (viscosity, thermal conduction, srcterms)
  Viscosity *pvisc = nullptr;
  Conduction *pcond = nullptr;
  SourceTerms *psrc = nullptr;

  // following only used for time-evolving flow
  // Conserved variables at the intermediate step: the RK start state and the LAT
  // interpolation source.  Every reader takes owned cells (RKUpdate, the FOFC trial,
  // RestrictCC with band_source, the LAT packer on its send ranges, the radiation
  // pre-stage gas), so u1 holds the owned cells alone wherever
  // MeshBoundaryValuesCC::LATStage1OwnedOnly allows it and the ghost-extended block
  // otherwise (StartRegisterGhosts); it is read and written through the band view in
  // global cell numbers, and the exchange and RestrictCC take `u1.data`.
  BandView5D<Real> u1;
  DvceArray5D<Real> coarse_u1;  // LAT start-state restriction for fine-to-coarse bvals
  DvceArray5D<Real> lat_u_stage1;  // RK2 stage-1 endpoint for LAT dense bvals
  bool lat_dense_output_enabled = false;
  // The face fluxes and dual_vf live on a band of the block, not on the ghost-extended
  // layout: without FOFC the active cells plus the ie+1 face along each normal, with
  // FOFC one further cell on each side (as MHD, mhd.hpp), so x1f is
  // (nmb, nvars, nx3[+2], nx2[+2], nx1[+2]+1) with origin (ks[-1], js[-1], is[-1]).  A
  // degenerate dimension keeps its single cell at origin 0.  Kernels index the band with
  // the global (m,n,k,j,i) through FluxBand(); the exchange takes FluxOrigin().  Faces
  // outside the band (ghost faces written by the excision flux replacement without
  // FOFC) are skipped.
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
  FaceFldOrigin FluxOrigin() const { return FaceFldOrigin{flux_ko, flux_jo, flux_io}; }
  DvceFaceFld5D<Real> lat_reflux;  // delayed LAT fine/coarse flux correction
  DvceFaceFld5D<Real> lat_dual_vf_reflux;  // delayed LAT reflux of the dual_vf payload
  DvceFaceFld5D<Real> lat_reflux_theta;  // admissible fraction of the pending reflux
  // Gate record of the gravitational work carried by the pending reflux, laid out by
  // lat_reflux::GravWorkComponent.  Allocated only with a gravity source term (self,
  // external BH or sink particles).
  DvceFaceFld5D<Real> lat_grav_reflux;
  bool lat_grav_reflux_allocated = false;
  // dual-energy face payload: the contact velocity of each face and the heat of each side
  // (eos_general::DualFaceComponent, eos/dual_energy_shock_rule.hpp)
  DvceFaceFld5D<Real> dual_vf;
  // The dual-energy shock rule's flag of each owned cell, decided on the step-start state
  // at the first stage and held through the step's later stages, and how far the channel
  // taken at the step's earlier stages moved the cell's auxiliary off its adiabat
  // (ApplyDualEnergyFormalism).  Components of dual_shock_debt: the whole offset, the
  // part the first stage's channel left (integrators of three or more stages), and for
  // the two-register rk4 the same two parts in the register u1.
  DvceArray4D<bool> dual_shocked;
  DvceArray5D<Real> dual_shock_debt;
  DualArray1D<int> sink_block_indices;  // reusable active blocks intersecting the sink
  DualArray1D<Real> dtnew_eachmb;  // local hydro CFL timestep for each MeshBlock
  Real dtnew_hydro_cfl;
  Real dtnew;

  // Per-chunk L/R primitive buffers used by the split-kernel flux path.
  // Shaped (split_recon_chunk_nmb, nvars, ncells3, ncells2, ncells1).
  // The buffers cover the full cell range including ghost zones and are reused
  // sequentially across chunks and coordinate directions.
  DvceArray5D<Real> wl3d;
  DvceArray5D<Real> wr3d;
  int split_recon_chunk_nmb = 32;

  // following used for FOFC
  DvceArray4D<bool> fofc;  // flag for each cell to indicate if FOFC is needed
  bool use_fofc = false;   // flag to enable FOFC
  DvceArray5D<Real> utest;  // scratch array for FOFC
  // FOFC flag exchange (MHD's contract, mhd.hpp): each block tests its owned cells, and
  // the flags of the first ghost layer come from their owners.  fofc_mask is
  // (nmb, 1, n3, n2, n1), > 0 means failed; coarse_fofc_mask is its restriction
  // (multilevel only); pbval_fofc exchanges it with exchange_depth = 1, completed at
  // coarse/fine by injection.  Allocated on every mesh whenever FOFC is on.
  DvceArray5D<Real> fofc_mask;
  DvceArray5D<Real> coarse_fofc_mask;
  MeshBoundaryValuesCC *pbval_fofc = nullptr;
  bool FOFCMaskExchangeEnabled() const { return use_fofc && pbval_fofc != nullptr; }

  // container to hold names of TaskIDs
  HydroTaskIDs id;

  // functions...
  void AssembleHydroTasks(std::map<std::string, std::shared_ptr<TaskList>> tl);
  // Two-phase form used when a radiation module hangs its graph between the halves.  Part A ends at
  // id.statefix (the gate the radiation graph waits on); part B resumes at the state
  // exchange, gated on the radiation couple that has just rewritten the gas u0.
  void AssembleHydroTasksPartA(std::map<std::string, std::shared_ptr<TaskList>> tl);
  void AssembleHydroTasksPartB(std::map<std::string, std::shared_ptr<TaskList>> tl,
                               TaskID rad_gate);
  // ...in "before_stagen_tl" list
  TaskStatus InitRecv(Driver *d, int stage);
  // ...in "stagen_tl" list
  TaskStatus HydroStateFixup(Driver *d, int stage);
  TaskStatus CopyCons(Driver *d, int stage);
  TaskStatus Fluxes(Driver *d, int stage);
  // FOFC flag quartet, between Fluxes and SendFlux: write the owned flags into the
  // mask (and restrict), exchange it, then import the first ghost layer and replace
  // every owned-cell face with a flagged cell on either side.
  TaskStatus PrepareFOFCFlags(Driver *d, int stage);
  TaskStatus SendFOFCFlags(Driver *d, int stage);
  TaskStatus RecvFOFCFlags(Driver *d, int stage);
  TaskStatus ReplaceFOFC(Driver *d, int stage);
  // Grow the first-order set until no unflagged cell is drained by the stage on the
  // fluxes the replacements left (utils/mass_drain.hpp, FlagDrainedCells); GR with the
  // gas-entropy column.  See MHD::CloseDrainedCells.
  TaskStatus CloseDrainedCells(Driver *d, int stage);
  // The drained-cell closure's consumer of the LAT phase communicator: whether it runs,
  // and the selection of the ranks that execute this bin and stage (a collective call).
  bool DrainedCellClosureEnabled() const;
  void ConfigureDrainCommunicator(std::uint64_t phase_key, bool participates);
  TaskStatus SendFlux(Driver *d, int stage);
  TaskStatus RecvFlux(Driver *d, int stage);
  TaskStatus RKUpdate(Driver *d, int stage);
  TaskStatus DualEnergyStep(Driver *d, int stage);
  // End-of-step gas entropy resync (GR, stage == nexp_stages only).
  TaskStatus GasEntropyEndpoint(Driver *d, int stage);
  TaskStatus HydroSrcTerms(Driver *d, int stage);
  TaskStatus SendU_OA(Driver *d, int stage);
  TaskStatus RecvU_OA(Driver *d, int stage);
  TaskStatus RestrictU(Driver *d, int stage);
  TaskStatus SendU(Driver *d, int stage);
  TaskStatus RecvU(Driver *d, int stage);
  TaskStatus SendU_Shr(Driver *d, int stage);
  TaskStatus RecvU_Shr(Driver *d, int stage);
  TaskStatus ApplyPhysicalBCs(Driver* pdrive, int stage);
  TaskStatus Prolongate(Driver* pdrive, int stage);
  TaskStatus ConToPrim(Driver *d, int stage);
  // The stage task: ConToPrim, which a plain fixed-metric GRHD run restricts to the cells
  // its fluxes read (EquationOfState::SetC2PReadSet).
  TaskStatus ConToPrimStage(Driver *d, int stage);
  TaskStatus ConToPrimGhostBands(Driver *d, int stage);
  TaskStatus NewTimeStep(Driver *d, int stage);
  // ...in "after_stagen_tl" list
  TaskStatus ClearSend(Driver *d, int stage);
  TaskStatus ClearRecv(Driver *d, int stage);  // also in Driver::Initialize
  TaskStatus SaveLATDenseOutput(Driver *d, int stage);

  void ConfigureLATDenseOutput(bool enabled);
  void ResetLATFluxCorrection();
  void AccumulateLATCoarseFluxes(Driver *d, int stage);
  bool ApplyLATFluxCorrection(Real sync_time, int sync_factor, int completed_tick);

  // CalculateFluxes function templated over Riemann Solvers
  template <Hydro_RSolver T>
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
  void ResolveDualEnergyChannel(const Real tally_weight);
  void SynchronizeRestrictedDualEnergyField(bool lat_active_only = false);
  void RepairRefinedDualEnergyState(DualArray1D<int> &n2o, DualArray1D<int> &rflag,
                                    const int new_gids, const int new_nmb_local);
  bool ResizeMeshBlockStorage(int nmb, bool exact = false,
                              bool allow_shrink = false);
  // Ghost layers u1 keeps about the owned cells (see u1).
  int StartRegisterGhosts() const;
  // Give u1 the ghost-extended layout.  For the analytic-reference error functions of the
  // test problem generators, which write their reference solution into u1 over the whole
  // block after the last step.  Its contents are not kept.
  void ExpandStartRegister();

 private:
  bool lat_requested_ = false;
  bool lat_correction_storage_ = false;  // LAT delayed-reflux registers are allocated
  PhaseCommunicator drain_phase_comm_;   // the ranks of the current LAT bin and stage
  bool flux_recv_complete_ = false;
  MeshBlockPack* pmy_pack;  // ptr to MeshBlockPack containing this Hydro
};

} // namespace hydro
#endif // HYDRO_HYDRO_HPP_
