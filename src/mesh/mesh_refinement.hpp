#ifndef MESH_MESH_REFINEMENT_HPP_
#define MESH_MESH_REFINEMENT_HPP_
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file mesh_refinement.hpp
//! \brief defines MeshRefinement class containing data and functions controlling SMR/AMR

//----------------------------------------------------------------------------------------
//! \fn int CreateAMR_MPI_Tag(int lid, int ox1, int ox2, int ox3)
//! \brief calculate an MPI tag for AMR communications.  Note maximum size of
//! lid that can be encoded is set by (NUM_BITS_LID) macro.
//! The convention in Athena++ is lid is for the *receiving* process.
//! The MPI standard requires signed int tag, with MPI_TAG_UB>=2^15-1 = 32,767 (inclusive)
inline int CreateAMR_MPI_Tag(int lid, int ox1, int ox2, int ox3) {
  return (ox1<<(NUM_BITS_LID+2)) | (ox2<<(NUM_BITS_LID+1))| (ox3<<(NUM_BITS_LID)) | lid;
}

//----------------------------------------------------------------------------------------
//! \struct AMRBuffer
//! \brief container for index ranges, storage, and flags for AMR buffers used with load
//! balancing.

#if MPI_PARALLEL_ENABLED
struct AMRBuffer {
  int bis, bie, bjs, bje, bks, bke;  // start/end indices of data to be packed/unpacked
  int cntcc, cntfc;          // number of CC and FC array elements to be sent/recv per var
  int cnt;                   // total number of elements stored in buffer incl all vars
  std::size_t offset=0;      // starting index of data for this buffer
  int lid;                   // local ID (gid - gids) of MeshBlock on this rank
  bool use_coarse=false;     // pack/unpack from coarse array when true
};

// The payload exchanged with one peer rank: the descriptors [begin, end) of the send or
// receive table, which are consecutive, occupy the Reals [offset, offset + count) of the
// buffer.  A payload beyond one MPI count travels as nrequest consecutive messages.
struct AMRRankMessage {
  int rank;
  int begin, end;
  std::size_t offset, count;
  int nrequest;
  int pending;   // requests of a receive still outstanding
};

using AMRPinnedArray1D = Kokkos::View<Real *, Kokkos::SharedHostPinnedSpace>;
#endif

// Forward declaration
class RefinementCriteria;

//----------------------------------------------------------------------------------------
//! \class MeshRefinement
//! \brief data/functions associated with SMR/AMR

class MeshRefinement {
 public:
  MeshRefinement(Mesh *pm, ParameterInput *pin);
  ~MeshRefinement();

  // data
  int nmb_created;           // # of MeshBlocks created via AMR across all ranks
  int nmb_deleted;           // # of MeshBlocks deleted via AMR across all ranks
  int nmb_sent_thisrank;     // # of MeshBlocks sent during load balancing on this rank
  int ncyc_check_amr;        // # of cycles between checking mesh for ref/derefinement
  int refinement_interval;   // # of cycles between allowing successive ref/derefinement
  int last_amr_call_cycle;    // last cycle on which AMR bookkeeping was called
  int ncyc_since_amr_check;   // elapsed cycles since the last AMR criterion check
  bool prolong_prims;        // flag to enable prolongation of primitive vars
  bool sticky_load_balance;   // preserve old rank ownership across AMR when possible
  bool lb_transfer_gravity;   // migrate phi with the blocks (set per transaction: a pure
                              // rebalance or a valid phi kept by reuse_phi_after_amr)
  // A pure rebalance (no refinement, only ownership/GID changes) moves every cell of
  // the evolved arrays, ghost zones included, and the fluid primitives with them, so the
  // repartitioned pack holds the synchronized state verbatim and the Driver re-derives
  // nothing from it afterwards (Driver::RebalanceHydroLATMesh).  Frozen together with
  // lb_transfer_gravity (both set once per transaction); every buffer descriptor, pack/unpack offset and same-rank copy
  // of the transaction uses this one layout.
  bool lb_full_block_transfer = false;
  RefinementCriteria* pmrc=nullptr;   // object to control various refinement criteria

  // following View is dimensioned [nmb_total]
  DualArray1D<int> refine_flag;    // refinement flag for each MeshBlock
  DualArray1D<int> refine_hold;    // 1: a criterion wants the block kept at its level
  DualArray1D<int> fc_amr_repair;  // +1 refined, -1 derefined by the last AMR pass
  HostArray1D<int> ncyc_since_ref; // # of cycles since MB last refined/derefined

  // following 4x arrays allocated with length [nranks] only with AMR
  int *nref_eachrank;     // number of MBs refined per rank
  int *nderef_eachrank;   // number of MBs de-refined per rank
  int *nref_rsum;         // running sum of number of MBs refined per rank
  int *nderef_rsum;       // running sum of number of MBs de-refined per rank
  // following 2x arrays allocated with length [nmb_new] and [nmb_old]] only with AMR
  int *newtoold;          // mapping of new gid (index n) to old gid
  int *oldtonew;          // mapping of old gid (index n) to new gid
  std::vector<int> deref_old_gid_each_new;   // [new gid*nleaf + child] -> old fine gid
  std::vector<int> refine_new_gid_each_old;  // [old gid*nleaf + child] -> new fine gid

  // arrays in Mesh class created for new MB heirarchy with AMR
  // following 3x arrays allocated with length [new_nmb_total]
  float *new_cost_eachmb;            // cost of each MeshBlock
  int *new_rank_eachmb;              // rank of each MeshBlock
  LogicalLocation *new_lloc_eachmb;  // LogicalLocations for each MeshBlock
  // following 2x arrays allocated with length [nranks]
  int *new_gids_eachrank;      // starting global ID of MeshBlocks in each rank
  int *new_nmb_eachrank;       // number of MeshBlocks on each rank

  // Lagrange Interpolation weights for prolongation and restriction operators
  // naming convention: {prolong/restrict}_{order of interpolation}_{optional index}
  struct InterpWeight {
    DualArray3D<Real> prolong_2nd;
    DualArray1D<Real> restrict_2nd;
    DualArray3D<Real> prolong_4th;
    DualArray1D<Real> restrict_4th_edge;
    DualArray1D<Real> restrict_4th;
  };
  InterpWeight weights;

#if MPI_PARALLEL_ENABLED
  int nmb_send, nmb_recv;
  std::size_t lb_chunk_elements;      // largest payload packed in pooled device memory
  MPI_Comm amr_comm;                         // unique communicator for AMR
  DualArray1D<AMRBuffer> sendbuf, recvbuf; // send/recv buffers
  std::vector<AMRRankMessage> rank_send_messages, rank_recv_messages;
  std::vector<MPI_Request> send_requests, recv_requests;
  std::vector<int> recv_request_message;     // rank_recv_messages index of each request
  // Payload buffers: pooled device memory for the event, or, above the device budget,
  // pinned host memory kept between events (AcquireAMRBuffer in load_balance.cpp).
  // send_ptr/recv_ptr are what the pack/unpack kernels address.
  DvceArray1D<Real> send_pool, recv_pool;
  AMRPinnedArray1D send_pinned, recv_pinned;
  Real *send_ptr = nullptr, *recv_ptr = nullptr;
#endif

  // functions
  void CheckForRefinement(MeshBlockPack* pmbp);
  void LimitRefinementToMemoryCap();
  int PredictMeshBlockCountAfterAMR();
  void AdaptiveMeshRefinement(Driver *pdrive, ParameterInput *pin);
  void UpdateMeshBlockTree(int &nnew, int &ndel);
  void RedistAndRefineMeshBlocks(ParameterInput *pin, int nnew, int ndel);
  int DerefOldChildGID(int newm, int child, int nleaf) const {
    const int idx = newm*nleaf + child;
    return (idx >= 0 && idx < static_cast<int>(deref_old_gid_each_new.size())) ?
        deref_old_gid_each_new[idx] : -1;
  }
  int RefineNewChildGID(int oldm, int child, int nleaf) const {
    const int idx = oldm*nleaf + child;
    return (idx >= 0 && idx < static_cast<int>(refine_new_gid_each_old.size())) ?
        refine_new_gid_each_old[idx] : -1;
  }

  void DerefineCCSameRank(DvceArray5D<Real> &a, DvceArray5D<Real> &ca);
  void DerefineFCSameRank(DvceFaceFld4D<Real> &b, DvceFaceFld4D<Real> &cb);

  void CopyCC(DvceArray5D<Real> &a);
  void CopyFC(DvceFaceFld4D<Real> &b);

  void CopyForRefinementCC(DvceArray5D<Real> &a, DvceArray5D<Real> &ca);
  void CopyForRefinementFC(DvceFaceFld4D<Real> &b, DvceFaceFld4D<Real> &cb);

  void RefineCC(DualArray1D<int> &n2o, DvceArray5D<Real> &a, DvceArray5D<Real> &ca,
                bool is_z4c=false);
  void RefineFC(DualArray1D<int> &n2o, DvceFaceFld4D<Real> &b, DvceFaceFld4D<Real> &cb);
  void RepairAMRFC(DvceFaceFld4D<Real> &b);

  // band_source: `a` holds a centred band of the block, the owned cells plus g ghost
  // layers with origin ng - g (CentredBand), as a LAT start register or stage-1 snapshot
  // may (MeshBoundaryValuesCC::LATStage1OwnedOnly).  Not with is_z4c.
  void RestrictCC(DvceArray5D<Real> &a, DvceArray5D<Real> &ca, bool is_z4c=false,
                  bool lat_active_only=false, bool band_source=false);
  void RestrictFC(DvceFaceFld4D<Real> &b, DvceFaceFld4D<Real> &cb);
  void HighOrderRestrictCC(DvceArray5D<Real> &a, DvceArray5D<Real> &ca);

  // functions for load balancing (in file load_balance.cpp)
  void InitRecvAMR(int nleaf);
  int FullBlockTransferPrimitiveCount() const;
#if MPI_PARALLEL_ENABLED
  void SetSameLevelTransferExtent(AMRBuffer &buf) const;
#endif
  void PackAndSendAMR(int nleaf);
  // (ko, jo, io) is the global index of a's first cell when a is stored on a sub-block
  // band.  A pure rebalance sends the whole ghost-extended
  // block, so cells outside the band are sent as 0 and skipped on receipt; the default
  // origin is the ordinary full-extent array.
  void PackAMRBuffersCC(DvceArray5D<Real> &a, DvceArray5D<Real> &ca, int ncc, int nfc,
                        int mb_begin, int mb_end, int ko = 0, int jo = 0, int io = 0);
  void PackAMRBuffersFC(DvceFaceFld4D<Real> &b, DvceFaceFld4D<Real> &cb, int ncc, int nfc,
                        int mb_begin, int mb_end);
  void ClearRecvAndUnpackAMR();
  void UnpackAMRBuffersCC(DvceArray5D<Real> &a, DvceArray5D<Real> &ca, int ncc, int nfc,
                          int mb_begin, int mb_end, int ko = 0, int jo = 0, int io = 0);
  void UnpackAMRBuffersFC(DvceFaceFld4D<Real> &b, DvceFaceFld4D<Real> &cb, int ncc,
                          int nfc, int mb_begin, int mb_end);
  void ClearSendAMR();

  // initialize interpolation weights
  void InitInterpWghts();

 private:
  // data
  Mesh *pmy_mesh;

  // The band RestrictCC restricts a conserved register to under the LAT mask
  // (LATBandRestrictable): per local MeshBlock, bit (ox1+1) + 3*(ox2+1) + 9*(ox3+1) is set
  // for every face, edge and corner offset with a coarser neighbour.  Rebuilt from the
  // neighbour table whenever the topology, the local block count or the table changes.
  DualArray1D<int> lat_coarser_offsets_;
  std::uint64_t lat_coarser_offsets_version_ = ~std::uint64_t{0};
  int lat_coarser_offsets_nmb_ = -1;
  const void *lat_coarser_offsets_nghbr_ = nullptr;
  bool LATBandRestrictable(const DvceArray5D<Real> &ca) const;
  DvceArray1D<int> LATCoarserOffsets();
};

// Every ghost cell of the first nmb MeshBlocks of a cell-centred array takes the value of
// the nearest active cell of its own MeshBlock.
void SeedGhostsFromInteriorCC(DvceArray5D<Real> &a, const RegionIndcs &indcs,
                              const int nmb, const bool multi_d, const bool three_d);

// The MeshBlock budget of the mesh, <mesh_refinement>/max_nmb_total, into budget.  It
// replaces the product nranks x max_nmb_per_rank in the AMR block limit of <tde_amr> and
// in the block budget of the remap (the mesh then does not depend on the rank count),
// and is required wherever one of those formulas is used.  max_nmb_per_rank (per_rank)
// stays the per-rank memory cap.  Returns an empty string, or the message of the first
// problem: the key is absent, not positive, or above nranks x per_rank (the message
// names the smallest rank count that holds it).
std::string ReadBlockBudget(ParameterInput *pin, const long long per_rank,
                            long long &budget);
#endif // MESH_MESH_REFINEMENT_HPP_
