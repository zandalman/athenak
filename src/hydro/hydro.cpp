//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file hydro.cpp
//! \brief implementation of Hydro class constructor and assorted other functions

#include <iostream>
#include <string>
#include <algorithm>

#include "athena.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "eos/eos.hpp"
#include "eos/dual_energy_shock_rule.hpp"
#include "eos/lte_table_utils.hpp"
#include "eos/saha_table_utils.hpp"
#include "diffusion/viscosity.hpp"
#include "diffusion/conduction.hpp"
#include "srcterms/srcterms.hpp"
#include "shearing_box/shearing_box.hpp"
#include "shearing_box/orbital_advection.hpp"
#include "bvals/bvals.hpp"
#include "hydro/hydro.hpp"
#include "hydro/removed_gr_dual_energy.hpp"
#include "utils/lat_reflux_limiter.hpp"
#include "../mesh/mb_storage.hpp"

namespace hydro {
using eos_general::kDualFacePayload;

//----------------------------------------------------------------------------------------
// constructor, initializes data structures and parameters

Hydro::Hydro(MeshBlockPack *ppack, ParameterInput *pin) :
    pmy_pack(ppack),
    u0("cons",1,1,1,1,1),
    w0("prim",1,1,1,1,1),
    coarse_u0("ccons",1,1,1,1,1),
    coarse_w0("cprim",1,1,1,1,1),
    u1{DvceArray5D<Real>("cons1",1,1,1,1,1)},
    coarse_u1("ccons1",1,1,1,1,1),
    lat_u_stage1("lat_u_stage1",1,1,1,1,1),
    uflx("uflx",1,1,1,1,1),
    wl3d("wl3d",1,1,1,1,1),
    wr3d("wr3d",1,1,1,1,1),
    lat_reflux("lat_reflux",1,1,1,1,1),
    lat_dual_vf_reflux("lat_dual_vf_reflux",1,1,1,1,1),
    lat_reflux_theta("lat_reflux_theta",1,1,1,1,1),
    lat_grav_reflux("lat_grav_reflux",1,1,1,1,1),
    dual_vf("dual_vf",1,1,1,1,1),
    dual_shocked("dual_shocked",1,1,1,1),
    dual_shock_debt("dual_shock_debt",1,1,1,1,1),
    sink_block_indices("hydro_sink_blocks",1),
    utest("utest",1,1,1,1,1),
    fofc("fofc",1,1,1,1),
    floor_energy_block("floor_energy_block",1) {
  lat_requested_ = pin->IsLATEnabled();
  // The delayed-reflux registers are read only inside LAT windows, and only on a mesh
  // with coarse/fine faces or with same-level LAT (ApplyLATFluxCorrection,
  // ResetLATFluxCorrection, AccumulateLATCoarseFluxes and the masked RecvFlux all
  // return otherwise), which is the condition MHD allocates its twins under.
  lat_correction_storage_ = lat_requested_ &&
      (ppack->pmesh->multilevel || ppack->pmesh->hydro_lat_same_level);

  // Allocate for the current partition. AMR grows these views through
  // ResizeMeshBlockStorage(), so reserving the hard cap here only inflates the
  // persistent CUDA pool and can make an otherwise valid restart run out of memory.
  int nmb = ppack->nmb_thispack;
  // Block extent of the field arrays: exact, or the fixed reserve under preallocation.
  const int nmb_a =
      (MeshBlockStorageReserve() > 0) ? MeshBlockStorageCapacity(nmb) : nmb;
  {
    const auto &bindcs = ppack->pmesh->mb_indcs;
    split_recon_chunk_nmb = pin->GetOrAddInteger(
        "hydro", "split_recon_chunk_nmb",
        DefaultReconChunkNmb(bindcs.nx1*bindcs.nx2*bindcs.nx3));
  }
  if (split_recon_chunk_nmb <= 0) split_recon_chunk_nmb = std::max(1, nmb);

  // (1) construct EOS object (no default)
  std::string eqn_of_state = pin->GetString("hydro","eos");
  // ideal gas EOS
  if (eqn_of_state.compare("ideal") == 0) {
    if (pmy_pack->pcoord->is_special_relativistic) {
      peos = new IdealSRHydro(ppack, pin);
    } else if (pmy_pack->pcoord->is_general_relativistic) {
      peos = new IdealGRHydro(ppack, pin);
    } else {
      peos = new IdealHydro(ppack, pin);
    }
    nhydro = 5;
  // isothermal EOS
  } else if (eqn_of_state.compare("isothermal") == 0) {
    if (pmy_pack->pcoord->is_special_relativistic ||
        pmy_pack->pcoord->is_general_relativistic) {
      std::cout << "### FATAL ERROR in "<< __FILE__ <<" at line " << __LINE__ << std::endl
                << "<hydro>/eos = isothermal cannot be used with SR/GR" << std::endl;
      std::exit(EXIT_FAILURE);
    } else {
      peos = new IsothermalHydro(ppack, pin);
      nhydro = 4;
    }
  // tabulated H-only LTE/Saha EOS
  } else if (eqn_of_state.compare("saha_table") == 0) {
    if (pmy_pack->pcoord->is_special_relativistic ||
        pmy_pack->pcoord->is_general_relativistic) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "<hydro>/eos = saha_table cannot be used with SR/GR"
                << std::endl;
      std::exit(EXIT_FAILURE);
    }
    peos = new SahaTableHydro(ppack, pin);
    nhydro = 5;
  // tabulated H+He LTE EOS
  } else if (!lte_table_utils::ExpectedTableTypeForEosName(eqn_of_state).empty()) {
    if (pmy_pack->pcoord->is_special_relativistic ||
        pmy_pack->pcoord->is_general_relativistic) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "<hydro>/eos = " << eqn_of_state
                << " cannot be used with SR/GR" << std::endl;
      std::exit(EXIT_FAILURE);
    }
    peos = new LTETableHydro(ppack, pin);
    nhydro = 5;
  // EOS string not recognized
  } else {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "<hydro>/eos = '" << eqn_of_state << "' not implemented" << std::endl;
    std::exit(EXIT_FAILURE);
  }

  // (2) Initialize scalars, diffusion, source terms
  nscalars = pin->GetOrAddInteger("hydro","nscalars",0);
  if (pmy_pack->pcoord->is_general_relativistic) {
    // The GR dual energy (adiabat kappa, eta1/eta2) is removed; GR hydro carries the gas
    // entropy per unit rest mass in the hidden column instead, always, with no key.  It
    // crosses coarse/fine boundaries as D*s or, under prolong_primitives, as s.
    RejectRemovedGRDualEnergy(pin, "hydro");
    naux = 1;
    gas_entropy_idx = nhydro + nscalars;
    gas_entropy_init_pending = true;
  } else {
    use_dual_energy = pin->GetOrAddBoolean("hydro", "dual_energy", false);
  }
  if (use_dual_energy) {
    // The non-relativistic formalism: an internal-energy density updated by an
    // operator-split p dV step.  Special relativity has no implementation.
    if (pmy_pack->pcoord->is_special_relativistic) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "<hydro>/dual_energy is implemented for non-relativistic "
                << "hydro only, not for special relativity" << std::endl;
      std::exit(EXIT_FAILURE);
    }
    if (!peos->eos_data.use_e) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "<hydro>/dual_energy requires an EOS with internal energy"
                << std::endl;
      std::exit(EXIT_FAILURE);
    }
    dual_energy_pdv = true;
    naux = 1;
    dual_energy_idx = nhydro + nscalars;
    dual_energy_needs_init = true;
    // The one threshold of the Newtonian hydro formalism (the Bryan et al. eta1 test,
    // eos_general::HydroEnergyChannelResolvesEint): it bounds the truncation error of
    // E - E_K in smooth flow, which no rounding bound replaces (a sound wave carried at
    // 1000 c gives all its kinetic energy to E - E_K as the scheme damps it); shocked
    // cells bypass it (the shock rule, eos/general_c2p_hyd.hpp).  The auxiliary always
    // holds the thermal state the inversion published, so there is no separate eta2.
    dual_energy_eta1 = pin->GetOrAddReal("hydro", "dual_energy_eta1", 1.0e-3);
  }
  nvars = nhydro + nscalars + naux;
  const bool tabulated_eos = peos->eos_data.UsesTabulatedLTE();
  if (pmy_pack->pmesh->pmr != nullptr && (dual_energy_pdv || tabulated_eos)) {
    // Dual-energy hydro and the tabulated Saha EOS should not prolongate total energy in
    // conserved form across AMR boundaries. Rebuild fine-grid states from primitives so
    // the thermal closure stays consistent after prolongation and regrids.
    pmy_pack->pmesh->pmr->prolong_prims = true;
  }

  // Viscosity (if requested in input file)
  if (pin->DoesParameterExist("hydro","viscosity")) {
    pvisc = new Viscosity("hydro", ppack, pin);
  } else {
    pvisc = nullptr;
  }

  // Thermal conduction (if requested in input file)
  if (pin->DoesParameterExist("hydro","conductivity") ||
      pin->DoesParameterExist("hydro","tdep_conductivity")) {
    if (tabulated_eos) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "<hydro>/eos = " << eqn_of_state
                << " does not support thermal "
                << "conduction because the conduction module still assumes gamma-law "
                << "temperature closure." << std::endl;
      std::exit(EXIT_FAILURE);
    }
    pcond = new Conduction("hydro", ppack, pin);
  } else {
    pcond = nullptr;
  }

  // Source terms are needed either for an explicit source-term block or when
  // gravity/self_gravity is enabled and should couple into hydro automatically.
  bool gravity_self_gravity =
      (pin->DoesBlockExist("gravity") &&
       pin->GetOrAddBoolean("gravity", "self_gravity", true));
  const std::string pgen_name =
      pin->DoesBlockExist("problem") ?
      pin->GetOrAddString("problem", "pgen_name", "none") : "none";
  const bool analytic_bh_source_problem =
      (pgen_name == "tde_external");
  const bool external_bh_gravity_source =
      pin->GetOrAddBoolean("problem", "external_bh_gravity_source",
                           analytic_bh_source_problem);
  if (pin->DoesBlockExist("hydro_srcterms") && tabulated_eos) {
    const bool ism_cooling = pin->GetOrAddBoolean("hydro_srcterms", "ism_cooling", false);
    const bool rel_cooling = pin->GetOrAddBoolean("hydro_srcterms", "rel_cooling", false);
    if (ism_cooling || rel_cooling) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "<hydro>/eos = " << eqn_of_state << " does not support "
                << "hydro_srcterms cooling because those source terms still assume "
                << "gamma-law temperature closure." << std::endl;
      std::exit(EXIT_FAILURE);
    }
  }
  if (pin->DoesBlockExist("hydro_srcterms") || gravity_self_gravity ||
      external_bh_gravity_source || pin->DoesBlockExist("sink_particles")) {
    psrc = new SourceTerms("hydro_srcterms", ppack, pin);
  }

  // (3) read time-evolution option [already error checked in driver constructor]
  // Then initialize memory and algorithms for reconstruction and Riemann solvers
  std::string evolution_t = pin->GetString("time","evolution");
  time_evolving = (evolution_t.compare("stationary") != 0);

  // allocate memory for conserved and primitive variables
  // With AMR, maximum size of Views are limited by total device memory through an input
  // parameter, which in turn limits max number of MBs that can be created.
  {
    auto &indcs = pmy_pack->pmesh->mb_indcs;
    int ncells1 = indcs.nx1 + 2*(indcs.ng);
    int ncells2 = (indcs.nx2 > 1)? (indcs.nx2 + 2*(indcs.ng)) : 1;
    int ncells3 = (indcs.nx3 > 1)? (indcs.nx3 + 2*(indcs.ng)) : 1;
    Kokkos::realloc(u0, nmb_a, nvars, ncells3, ncells2, ncells1);
    Kokkos::realloc(w0, nmb_a, nvars, ncells3, ncells2, ncells1);
  }

  // allocate memory for conserved variables on coarse mesh
  if (ppack->pmesh->multilevel) {
    auto &indcs = pmy_pack->pmesh->mb_indcs;
    int n_ccells1 = indcs.cnx1 + 2*(indcs.ng);
    int n_ccells2 = (indcs.cnx2 > 1)? (indcs.cnx2 + 2*(indcs.ng)) : 1;
    int n_ccells3 = (indcs.cnx3 > 1)? (indcs.cnx3 + 2*(indcs.ng)) : 1;
    Kokkos::realloc(coarse_u0, nmb_a, nvars, n_ccells3, n_ccells2, n_ccells1);
    Kokkos::realloc(coarse_w0, nmb_a, nvars, n_ccells3, n_ccells2, n_ccells1);
    if (time_evolving && lat_requested_) {
      Kokkos::realloc(coarse_u1, nmb_a, nvars, n_ccells3, n_ccells2, n_ccells1);
    }
  }

  // allocate boundary buffers for conserved (cell-centered) variables
  pbval_u = new MeshBoundaryValuesCC(ppack, pin, false);
  // Hydro exchanges use only the standard PackAndSendCC/RecvAndUnpackCC machinery,
  // so send-side per-neighbor vars payloads are stubbed on multi-rank runs.
  pbval_u->stub_send_vars = true;
  pbval_u->stub_recv_same_vars = true;
  pbval_u->InitializeBuffers(nvars, nvars + (dual_energy_pdv ? kDualFacePayload : 0));

  // Orbital advection and shearing box BCs (if requested in input file)
  if (pin->DoesBlockExist("shearing_box")) {
    porb_u = new OrbitalAdvectionCC(ppack, pin, nvars);
    psbox_u = new ShearingBoxCC(ppack, pin, nvars);
  } else {
    porb_u = nullptr;
    psbox_u = nullptr;
  }

  // for time-evolving problems, continue to construct methods, allocate arrays
  if (time_evolving) {
    const bool source_coupled_gravity =
        psrc != nullptr && (psrc->self_gravity || psrc->external_bh_gravity);
    ConfigureLATDenseOutput(pmy_pack->pmesh->hydro_lat_same_level &&
                            pmy_pack->pmesh->hydro_lat_same_level_max_ratio > 1 &&
                            !source_coupled_gravity);

    // determine if FOFC is enabled
    use_fofc = pin->GetOrAddBoolean("hydro","fofc",false);

    // select reconstruction method (default PLM)
    std::string xorder = pin->GetOrAddString("hydro","reconstruct","plm");
    if (xorder.compare("dc") == 0) {
      recon_method = ReconstructionMethod::dc;
    } else if (xorder.compare("plm") == 0) {
      recon_method = ReconstructionMethod::plm;
    } else if (xorder.compare("ppm4") == 0 ||
               xorder.compare("ppmx") == 0 ||
               xorder.compare("wenoz") == 0) {
      // check that nghost > 2
      auto &indcs = pmy_pack->pmesh->mb_indcs;
      if (indcs.ng < 3) {
        std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
          << std::endl << xorder << " reconstruction requires at least 3 ghost zones, "
          << "but <mesh>/nghost=" << indcs.ng << std::endl;
        std::exit(EXIT_FAILURE);
      }
      // FOFC needs no ghost layer beyond the reconstruction's: each block tests its own
      // cells and imports the flags of the first ghost layer from their owners.
      if (xorder.compare("ppm4") == 0) {
        recon_method = ReconstructionMethod::ppm4;
      } else if (xorder.compare("ppmx") == 0) {
        recon_method = ReconstructionMethod::ppmx;
      } else if (xorder.compare("wenoz") == 0) {
        recon_method = ReconstructionMethod::wenoz;
      }
    } else {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "<hydro> reconstruct = '" << xorder << "' not implemented"
                << std::endl;
      std::exit(EXIT_FAILURE);
    }
    // A relativistic fluid with the gas-entropy column needs the drained-cell closure of
    // fofc = true (CloseDrainedCells): the column is a convex combination of the cells
    // the mass leaves and enters only where first-order faces bound the drain by the
    // cell's own content, and a wenoz or ppmx face flux can drain a cell below it, after
    // which the entropy root of a failed inversion reads a label no neighbour held.
    if (gas_entropy_idx >= 0 && !use_fofc &&
        (recon_method == ReconstructionMethod::wenoz ||
         recon_method == ReconstructionMethod::ppmx)) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "<hydro>/reconstruct = '" << xorder
                << "' on a relativistic fluid requires <hydro>/fofc = true: "
                << "its closure of drained cells keeps the gas-entropy label inside the "
                << "hull of its neighbours" << std::endl;
      std::exit(EXIT_FAILURE);
    }

    // select Riemann solver (no default).  Test for compatibility of options
    std::string rsolver = pin->GetString("hydro","rsolver");
    // Special relativistic dynamic solvers
    if (pmy_pack->pcoord->is_special_relativistic) {
      if (evolution_t.compare("dynamic") == 0) {
        if (rsolver.compare("llf") == 0) {
          rsolver_method = Hydro_RSolver::llf_sr;
        } else if (rsolver.compare("hlle") == 0) {
          rsolver_method = Hydro_RSolver::hlle_sr;
        } else if (rsolver.compare("hllc") == 0) {
          rsolver_method = Hydro_RSolver::hllc_sr;
        // Error for anything else
        } else {
          std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                    << std::endl << "<hydro> rsolver = '" << rsolver
                    << "' not implemented for SR dynamics" << std::endl;
          std::exit(EXIT_FAILURE);
        }
      } else {
        std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                  << std::endl << "kinematic dynamics not implemented for SR" <<std::endl;
        std::exit(EXIT_FAILURE);
      }

    // General relativistic dynamic solvers
    } else if (pmy_pack->pcoord->is_general_relativistic) {
      if (evolution_t.compare("dynamic") == 0) {
        if (rsolver.compare("llf") == 0) {
          rsolver_method = Hydro_RSolver::llf_gr;
        } else if (rsolver.compare("hlle") == 0) {
          rsolver_method = Hydro_RSolver::hlle_gr;
        // Error for anything else
        } else {
          std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                    << std::endl << "<hydro> rsolver = '" << rsolver
                    << "' not implemented for GR dynamics" << std::endl;
          std::exit(EXIT_FAILURE);
        }
      } else {
        std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                  << std::endl << "kinematic dynamics not implemented for GR" <<std::endl;
        std::exit(EXIT_FAILURE);
      }

    // Non-relativistic dynamic solvers
    } else if (evolution_t.compare("dynamic") == 0) {
      // LLF solver
      if (rsolver.compare("llf") == 0) {
        rsolver_method = Hydro_RSolver::llf;
      // HLLE solver
      } else if (rsolver.compare("hlle") == 0) {
        rsolver_method = Hydro_RSolver::hlle;
      // HLLC solver
      } else if (rsolver.compare("hllc") == 0) {
        if (peos->eos_data.use_e) {
          rsolver_method = Hydro_RSolver::hllc;
        } else {
          std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                    << std::endl << "<hydro>/rsolver = hllc cannot be used with "
                    << "the selected EOS" << std::endl;
          std::exit(EXIT_FAILURE);
        }
      // Roe solver
      } else if (rsolver.compare("roe") == 0) {
        if (peos->eos_data.UsesTabulatedLTE()) {
          std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                    << std::endl
                    << "<hydro>/eos = " << eqn_of_state
                    << " does not support <hydro>/rsolver = roe. "
                    << "Use llf, hlle, or hllc." << std::endl;
          std::exit(EXIT_FAILURE);
        }
        if (!(peos->eos_data.is_gamma_law) &&
            peos->eos_data.hydro_eos != HydroEOSModel::isothermal) {
          std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                    << std::endl << "<hydro>/rsolver = roe requires gamma-law or "
                    << "isothermal EOS" << std::endl;
          std::exit(EXIT_FAILURE);
        }
        rsolver_method = Hydro_RSolver::roe;
      // Error for anything else
      } else {
        std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                  << std::endl << "<hydro> rsolver = '" << rsolver << "' not implemented"
                  << " for dynamic problems" << std::endl;
        std::exit(EXIT_FAILURE);
      }
      if (dual_energy_pdv &&
          !(rsolver_method == Hydro_RSolver::llf ||
            rsolver_method == Hydro_RSolver::hlle ||
            rsolver_method == Hydro_RSolver::hllc ||
            rsolver_method == Hydro_RSolver::roe)) {
        std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                  << std::endl << "<hydro>/dual_energy only supports non-relativistic "
                  << "llf/hlle/hllc/roe solvers"
                  << std::endl;
        std::exit(EXIT_FAILURE);
      }

    // Non-relativistic kinematic solvers
    } else {
      // Advect solver
      if (rsolver.compare("advect") == 0) {
        rsolver_method = Hydro_RSolver::advect;
      } else {
        std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                  << std::endl << "<hydro> rsolver = '" << rsolver << "' not implemented"
                  << " for kinematic problems" << std::endl;
        std::exit(EXIT_FAILURE);
      }
    }
    if (use_dual_energy && evolution_t.compare("dynamic") != 0) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "<hydro>/dual_energy only supports dynamic "
                << "non-relativistic hydro"
                << std::endl;
      std::exit(EXIT_FAILURE);
    }

    // Final memory allocations
    {
      // allocate second registers, fluxes
      auto &indcs = pmy_pack->pmesh->mb_indcs;
      int ncells1 = indcs.nx1 + 2*(indcs.ng);
      int ncells2 = (indcs.nx2 > 1)? (indcs.nx2 + 2*(indcs.ng)) : 1;
      int ncells3 = (indcs.nx3 > 1)? (indcs.nx3 + 2*(indcs.ng)) : 1;
      dtnew_eachmb = DualArray1D<Real>("hydro_dtnew_eachmb", std::max(1, nmb));
      sink_block_indices = DualArray1D<int>("hydro_sink_blocks", std::max(1, nmb_a));
      {
        const int g = StartRegisterGhosts();
        Kokkos::realloc(u1.data, nmb_a, nvars,
                        (indcs.nx3 > 1) ? indcs.nx3 + 2*g : 1,
                        (indcs.nx2 > 1) ? indcs.nx2 + 2*g : 1, indcs.nx1 + 2*g);
        u1 = CentredBand(u1.data, ncells3, ncells2, ncells1);
      }
      if (lat_dense_output_enabled) {
        // Senders read the snapshot on owned send strips only, so it is stored without
        // ghosts wherever the mesh allows it (MeshBoundaryValuesCC::LATStage1OwnedOnly).
        const bool s1_owned = MeshBoundaryValuesCC::LATStage1OwnedOnly(
            indcs, pmy_pack->pmesh->multilevel);
        Kokkos::realloc(lat_u_stage1, nmb_a, nvars, s1_owned ? indcs.nx3 : ncells3,
                        s1_owned ? indcs.nx2 : ncells2, s1_owned ? indcs.nx1 : ncells1);
      }
      // flux band (see hydro.hpp): the faces of the owned cells
      const bool multi_d = pmy_pack->pmesh->multi_d;
      const bool three_d = pmy_pack->pmesh->three_d;
      flux_ko = three_d ? indcs.ks : 0;
      flux_jo = multi_d ? indcs.js : 0;
      flux_io = indcs.is;
      const int nband1 = indcs.nx1;
      const int nband2 = multi_d ? indcs.nx2 : 1;
      const int nband3 = three_d ? indcs.nx3 : 1;
      Kokkos::realloc(uflx.x1f, nmb_a, nvars, nband3, nband2, nband1+1);
      Kokkos::realloc(uflx.x2f, nmb_a, nvars, nband3, nband2+1, nband1);
      Kokkos::realloc(uflx.x3f, nmb_a, nvars, nband3+1, nband2, nband1);
      const int nrecon_nmb = std::max(1, std::min(nmb_a, split_recon_chunk_nmb));
      Kokkos::realloc(wl3d, nrecon_nmb, nvars, ncells3, ncells2, ncells1);
      Kokkos::realloc(wr3d, nrecon_nmb, nvars, ncells3, ncells2, ncells1);
      if (lat_correction_storage_) {
        Kokkos::realloc(lat_reflux.x1f, nmb_a, nvars, ncells3, ncells2, 2);
        Kokkos::realloc(lat_reflux.x2f, nmb_a, nvars, ncells3, 2, ncells1);
        Kokkos::realloc(lat_reflux.x3f, nmb_a, nvars, 2, ncells2, ncells1);
        Kokkos::realloc(lat_reflux_theta.x1f, nmb_a, 1, ncells3, ncells2, 2);
        Kokkos::realloc(lat_reflux_theta.x2f, nmb_a, 1, ncells3, 2, ncells1);
        Kokkos::realloc(lat_reflux_theta.x3f, nmb_a, 1, 2, ncells2, ncells1);
        Kokkos::deep_copy(lat_reflux_theta.x1f, 1.0);
        Kokkos::deep_copy(lat_reflux_theta.x2f, 1.0);
        Kokkos::deep_copy(lat_reflux_theta.x3f, 1.0);
      }
      // Cleared, not merely left unallocated, without LAT storage: SourceTerms::Gravity
      // keys its gate kernels on this flag, and the 1x1x1x1x1 placeholder would pass its
      // extent test on a one-block pack.
      lat_grav_reflux_allocated = lat_correction_storage_ && (psrc != nullptr) &&
          (psrc->self_gravity || psrc->external_bh_gravity || psrc->sink_gravity);
      if (lat_grav_reflux_allocated) {
        const int ngw = lat_reflux::kNGravWork;
        Kokkos::realloc(lat_grav_reflux.x1f, nmb_a, ngw, ncells3, ncells2, 2);
        Kokkos::realloc(lat_grav_reflux.x2f, nmb_a, ngw, ncells3, 2, ncells1);
        Kokkos::realloc(lat_grav_reflux.x3f, nmb_a, ngw, 2, ncells2, ncells1);
        Kokkos::deep_copy(lat_grav_reflux.x1f, 0.0);
        Kokkos::deep_copy(lat_grav_reflux.x2f, 0.0);
        Kokkos::deep_copy(lat_grav_reflux.x3f, 0.0);
      }
      if (dual_energy_pdv) {
        const int ndf = kDualFacePayload;
        if (lat_correction_storage_) {
          Kokkos::realloc(lat_dual_vf_reflux.x1f, nmb_a, ndf, ncells3, ncells2, 2);
          Kokkos::realloc(lat_dual_vf_reflux.x2f, nmb_a, ndf, ncells3, 2, ncells1);
          Kokkos::realloc(lat_dual_vf_reflux.x3f, nmb_a, ndf, 2, ncells2, ncells1);
        }
        Kokkos::realloc(dual_vf.x1f, nmb_a, ndf, nband3, nband2, nband1+1);
        Kokkos::realloc(dual_vf.x2f, nmb_a, ndf, nband3, nband2+1, nband1);
        Kokkos::realloc(dual_vf.x3f, nmb_a, ndf, nband3+1, nband2, nband1);
        Kokkos::realloc(dual_shocked, nmb_a, nband3, nband2, nband1);
        Kokkos::realloc(dual_shock_debt, nmb_a, 1, nband3, nband2, nband1);
      }

      // allocate array of flags used with FOFC
      if (use_fofc) {
        Kokkos::realloc(fofc,  nmb_a, ncells3, ncells2, ncells1);
        Kokkos::realloc(utest, nmb_a, nvars, ncells3, ncells2, ncells1);
        Kokkos::deep_copy(fofc, false);
        // The order-mask exchange on every mesh: the flags of the first ghost layer come
        // from their owners (Hydro::PrepareFOFCFlags .. ReplaceFOFC).  The mask exchange
        // uses the standard CC machinery only: the receive slots drop the same-level
        // footprint the rank-packed path never uses, no flux routine is ever called on
        // it, the import reads the first ghost layer alone and completes its coarse/fine
        // part by injection.
        pbval_fofc = new MeshBoundaryValuesCC(ppack, pin, false);
        pbval_fofc->stub_send_vars = true;
        pbval_fofc->stub_recv_same_vars = true;
        pbval_fofc->exchange_depth = 1;
        pbval_fofc->exchange_injected = true;
        pbval_fofc->InitializeBuffers(1, 0);
        pbval_fofc->PrepareRankPackedVarMetadata(1);
        fofc_mask = DvceArray5D<Real>("hydro_fofc_mask", nmb_a, 1, ncells3, ncells2,
                                      ncells1);
        if (ppack->pmesh->multilevel) {
          const int n_ccells1 = indcs.cnx1 + 2*indcs.ng;
          const int n_ccells2 = (indcs.cnx2 > 1) ? indcs.cnx2 + 2*indcs.ng : 1;
          const int n_ccells3 = (indcs.cnx3 > 1) ? indcs.cnx3 + 2*indcs.ng : 1;
          coarse_fofc_mask = DvceArray5D<Real>("hydro_coarse_fofc_mask", nmb_a, 1,
                                               n_ccells3, n_ccells2, n_ccells1);
        }
      }
    }
  }

  if (peos->eos_data.hydro_eos == HydroEOSModel::saha_table) {
    saha_table_utils::PrintStartupSummary("hydro", pin, peos->eos_data, use_dual_energy);
  } else if (peos->eos_data.hydro_eos == HydroEOSModel::lte_table) {
    lte_table_utils::PrintStartupSummary("hydro", pin, peos->eos_data, use_dual_energy);
  }
}

//----------------------------------------------------------------------------------------
// destructor

Hydro::~Hydro() {
  if (psbox_u != nullptr) {delete psbox_u;}
  if (porb_u != nullptr) {delete porb_u;}
  delete pbval_u;
  if (pbval_fofc != nullptr) {delete pbval_fofc;}
  if (psrc != nullptr) {delete psrc;}
  if (pcond != nullptr) {delete pcond;}
  if (pvisc != nullptr) {delete pvisc;}
  delete peos;
}

//----------------------------------------------------------------------------------------
//! \fn int Hydro::StartRegisterGhosts()
//! \brief The ghost layers u1 keeps about the owned cells: none where the LAT packer may
//! read the owned cells alone (MeshBoundaryValuesCC::LATStage1OwnedOnly), the whole ghost
//! extent otherwise.

int Hydro::StartRegisterGhosts() const {
  const auto &indcs = pmy_pack->pmesh->mb_indcs;
  return MeshBoundaryValuesCC::LATStage1OwnedOnly(indcs, pmy_pack->pmesh->multilevel) ?
         0 : indcs.ng;
}

//----------------------------------------------------------------------------------------
//! \fn void Hydro::ExpandStartRegister()
//! \brief u1 on the ghost-extended block, origin 0.  Its contents are not kept.

void Hydro::ExpandStartRegister() {
  const auto &indcs = pmy_pack->pmesh->mb_indcs;
  const int ncells1 = indcs.nx1 + 2*indcs.ng;
  const int ncells2 = (indcs.nx2 > 1) ? indcs.nx2 + 2*indcs.ng : 1;
  const int ncells3 = (indcs.nx3 > 1) ? indcs.nx3 + 2*indcs.ng : 1;
  const int nmb = u1.data.extent_int(0);
  u1.data = DvceArray5D<Real>();
  u1 = BandView5D<Real>{
      DvceArray5D<Real>("cons1", nmb, nvars, ncells3, ncells2, ncells1), 0, 0, 0};
}

bool Hydro::ResizeMeshBlockStorage(int nmb, bool exact, bool allow_shrink) {
  // Boundary buffers follow the same grow-only high-water discipline as the field
  // arrays below.  Growing them here keeps every topology-changing path covered without
  // the callers in mesh_refinement.cpp having to know about them.  The return value is
  // deliberately not folded into `resized`: that flag means the FIELD storage moved.
  if (pbval_u != nullptr) {
    pbval_u->ResizeBuffers(nmb);
  }
  if (pbval_fofc != nullptr) {
    pbval_fofc->ResizeBuffers(nmb);
  }
  // The floor tally is a per-rank sum whose slots only spread the atomics: a block a
  // regrid or a rebalance removed or renumbered still owes what it booked, and a new
  // block must find a slot for its first recovery.  So it grows with the pack before
  // that recovery, keeps every entry and never shrinks; the gravity energy window sums
  // its whole extent.  Not part of `resized` either.
  if (floor_energy_block.extent_int(0) < nmb) {
    Kokkos::resize(floor_energy_block, nmb);
  }
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int ncells1 = indcs.nx1 + 2*(indcs.ng);
  int ncells2 = (indcs.nx2 > 1)? (indcs.nx2 + 2*(indcs.ng)) : 1;
  int ncells3 = (indcs.nx3 > 1)? (indcs.nx3 + 2*(indcs.ng)) : 1;
  int n_ccells1 = indcs.cnx1 + 2*(indcs.ng);
  int n_ccells2 = (indcs.cnx2 > 1)? (indcs.cnx2 + 2*(indcs.ng)) : 1;
  int n_ccells3 = (indcs.cnx3 > 1)? (indcs.cnx3 + 2*(indcs.ng)) : 1;

  auto resize5 = [exact, allow_shrink](auto &view, int n0, int n1, int n2, int n3,
      int n4) {
    const bool need = ((exact ? (view.extent_int(0) != n0)
                          : MeshBlockStorageShouldResize(view.extent_int(0), n0,
                                                         allow_shrink)) ||
                       view.extent_int(1) != n1 || view.extent_int(2) != n2 ||
                       view.extent_int(3) != n3 || view.extent_int(4) != n4);
    if (need) {
      Kokkos::resize(view, exact ? n0 : MeshBlockStorageCapacity(n0), n1, n2, n3, n4);
    }
    return need;
  };
  auto realloc5 = [](auto &view, int n0, int n1, int n2, int n3, int n4) {
    const bool need = (view.extent_int(0) != n0 || view.extent_int(1) != n1 ||
                       view.extent_int(2) != n2 || view.extent_int(3) != n3 ||
                       view.extent_int(4) != n4);
    if (need) {
      Kokkos::realloc(view, n0, n1, n2, n3, n4);
    }
    return need;
  };
  auto realloc4 = [](auto &view, int n0, int n1, int n2, int n3) {
    const bool need = (view.extent_int(0) != n0 || view.extent_int(1) != n1 ||
                       view.extent_int(2) != n2 || view.extent_int(3) != n3);
    if (need) {
      Kokkos::realloc(view, n0, n1, n2, n3);
    }
    return need;
  };

  bool resized = false;
  if (time_evolving &&
      static_cast<int>(dtnew_eachmb.extent(0)) != std::max(1, nmb)) {
    dtnew_eachmb = DualArray1D<Real>("hydro_dtnew_eachmb", std::max(1, nmb));
    resized = true;
  }
  if (psrc != nullptr) {
    resized = psrc->ResizeMeshBlockStorage(nmb, exact) || resized;
  }
  resized = resize5(u0, nmb, nvars, ncells3, ncells2, ncells1) || resized;
  resized = resize5(w0, nmb, nvars, ncells3, ncells2, ncells1) || resized;
  const int nmb_storage = u0.extent_int(0);
  const int sink_capacity = (MeshBlockStorageReserve() > 0) ?
      MeshBlockStorageCapacity(nmb) : std::max(1, nmb);
  const int old_sink_capacity = static_cast<int>(sink_block_indices.extent(0));
  if (old_sink_capacity < sink_capacity || (exact &&
      old_sink_capacity != sink_capacity)) {
    // AMR may replace this allocation after asynchronous sink kernels have used it.
    DevExeSpace().fence();
    sink_block_indices = DualArray1D<int>("hydro_sink_blocks", sink_capacity);
    resized = true;
  }
  if (pmy_pack->pmesh->multilevel) {
    resized = resize5(coarse_u0, nmb, nvars, n_ccells3, n_ccells2, n_ccells1) || resized;
    resized = resize5(coarse_w0, nmb, nvars, n_ccells3, n_ccells2, n_ccells1) || resized;
    const int coarse_nmb_storage = coarse_u0.extent_int(0);
    if (time_evolving && lat_requested_) {
      resized = realloc5(coarse_u1, coarse_nmb_storage, nvars, n_ccells3, n_ccells2,
                         n_ccells1) || resized;
    }
  }

  bool resized_fofc = false;
  if (time_evolving) {
    {
      const int g = StartRegisterGhosts();
      resized = realloc5(u1.data, nmb_storage, nvars,
                         (indcs.nx3 > 1) ? indcs.nx3 + 2*g : 1,
                         (indcs.nx2 > 1) ? indcs.nx2 + 2*g : 1, indcs.nx1 + 2*g) ||
                resized;
      u1 = CentredBand(u1.data, ncells3, ncells2, ncells1);
    }
    if (lat_dense_output_enabled) {
      // Owned cells only wherever the mesh allows it, as in the constructor.
      const bool s1_owned = MeshBoundaryValuesCC::LATStage1OwnedOnly(
          indcs, pmy_pack->pmesh->multilevel);
      resized = realloc5(lat_u_stage1, nmb_storage, nvars,
                         s1_owned ? indcs.nx3 : ncells3, s1_owned ? indcs.nx2 : ncells2,
                         s1_owned ? indcs.nx1 : ncells1) || resized;
    }
    // flux band extents: the faces of the owned cells (origin fixed in the constructor)
    const int nband1 = indcs.nx1;
    const int nband2 = pmy_pack->pmesh->multi_d ? indcs.nx2 : 1;
    const int nband3 = pmy_pack->pmesh->three_d ? indcs.nx3 : 1;
    resized = realloc5(uflx.x1f, nmb_storage, nvars, nband3, nband2, nband1 + 1) ||
              resized;
    resized = realloc5(uflx.x2f, nmb_storage, nvars, nband3, nband2 + 1, nband1) ||
              resized;
    resized = realloc5(uflx.x3f, nmb_storage, nvars, nband3 + 1, nband2, nband1) ||
              resized;
    const int nrecon_nmb =
        std::max(1, std::min(nmb_storage, split_recon_chunk_nmb));
    resized = realloc5(wl3d, nrecon_nmb, nvars, ncells3, ncells2, ncells1) || resized;
    resized = realloc5(wr3d, nrecon_nmb, nvars, ncells3, ncells2, ncells1) || resized;
    if (lat_correction_storage_) {
      resized = realloc5(lat_reflux.x1f, nmb_storage, nvars, ncells3, ncells2,
          2) || resized;
      resized = realloc5(lat_reflux.x2f, nmb_storage, nvars, ncells3, 2,
          ncells1) || resized;
      resized = realloc5(lat_reflux.x3f, nmb_storage, nvars, 2, ncells2,
          ncells1) || resized;
      resized = realloc5(lat_reflux_theta.x1f, nmb_storage, 1, ncells3, ncells2, 2) ||
          resized;
      resized = realloc5(lat_reflux_theta.x2f, nmb_storage, 1, ncells3, 2, ncells1) ||
          resized;
      resized = realloc5(lat_reflux_theta.x3f, nmb_storage, 1, 2, ncells2, ncells1) ||
          resized;
    }
    if (lat_grav_reflux_allocated) {
      const int ngw = lat_reflux::kNGravWork;
      resized = realloc5(lat_grav_reflux.x1f, nmb_storage, ngw, ncells3, ncells2, 2) ||
          resized;
      resized = realloc5(lat_grav_reflux.x2f, nmb_storage, ngw, ncells3, 2, ncells1) ||
          resized;
      resized = realloc5(lat_grav_reflux.x3f, nmb_storage, ngw, 2, ncells2, ncells1) ||
          resized;
    }
    if (dual_energy_pdv) {
      if (lat_correction_storage_) {
        resized = realloc5(lat_dual_vf_reflux.x1f, nmb_storage, kDualFacePayload,
            ncells3, ncells2, 2) || resized;
        resized = realloc5(lat_dual_vf_reflux.x2f, nmb_storage, kDualFacePayload,
            ncells3, 2, ncells1) || resized;
        resized = realloc5(lat_dual_vf_reflux.x3f, nmb_storage, kDualFacePayload, 2,
            ncells2, ncells1) || resized;
      }
      resized = realloc5(dual_vf.x1f, nmb_storage, kDualFacePayload, nband3, nband2,
                         nband1 + 1) || resized;
      resized = realloc5(dual_vf.x2f, nmb_storage, kDualFacePayload, nband3, nband2 + 1,
                         nband1) || resized;
      resized = realloc5(dual_vf.x3f, nmb_storage, kDualFacePayload, nband3 + 1, nband2,
                         nband1) || resized;
      resized = realloc4(dual_shocked, nmb_storage, nband3, nband2, nband1) || resized;
      resized = realloc5(dual_shock_debt, nmb_storage, dual_shock_debt.extent_int(1),
                         nband3, nband2, nband1) || resized;
    }
    if (use_fofc) {
      resized_fofc = realloc4(fofc, nmb_storage, ncells3, ncells2, ncells1) ||
                     resized_fofc;
      resized_fofc = realloc5(utest, nmb_storage, nvars, ncells3, ncells2, ncells1) ||
                     resized_fofc;
      resized = resized || resized_fofc;
    }
    if (pbval_fofc != nullptr) {
      resized = realloc5(fofc_mask, nmb_storage, 1, ncells3, ncells2, ncells1) ||
                resized;
      if (pmy_pack->pmesh->multilevel) {
        resized = realloc5(coarse_fofc_mask, coarse_u0.extent_int(0), 1, n_ccells3,
                           n_ccells2, n_ccells1) || resized;
      }
    }
  }

  // With preallocation nothing is ever reallocated, so the flags are cleared on every
  // topology change, exactly as a reallocation does in the default mode.
  if (resized_fofc || (use_fofc && time_evolving && MeshBlockStorageReserve() > 0)) {
    Kokkos::deep_copy(fofc, false);
  }
  return resized;
}

void Hydro::ConfigureLATDenseOutput(bool enabled) {
  lat_dense_output_enabled = enabled;
}

} // namespace hydro
