//========================================================================================
// Athena astrophysical MHD code (Kokkos version)
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//////////////////////////////////// Athena Main Program /////////////////////////////////
//! \file main.cpp
//! \brief Athena main program
//!
//! Based on the Athena (Cambridge version) and Athena++ MHD codes. Athena originally was
//! written in 2002-2005 by Jim Stone, Tom Gardiner, and Peter Teuben, with many important
//! contributions by many other developers after that, i.e. 2005-2014.
//!
//! Athena++ was started in Jan 2014, with the core design developed 4-7/2014 during an
//! extended visit to the KITP at UCSB by J. Stone. GR was implemented by Chris White and
//! AMR by Kengo Tomida 2014-2016, with contributions from many others (esp. K. Felker)
//! continuing after that.
//!
//! Athena (Kokkos version) is an outgrowth of the Athena-Parthenon collaboration, and is
//! a completely new implementation based on the Kokkos performance-portability library
//! (which is an external dependency required for this version). It was started 6/2020
//! during the pandemic. As part of the keep-it-simple design, only a fraction of the
//! features of the C++ version are implemented.
//========================================================================================

// C/C++ headers
#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <list>
#include <string>
#include <memory>
#include <vector>
#include <cstdio> // sscanf
#include <fstream>  // Include this for std::ifstream

// Athena headers
#include "athena.hpp"
#include "globals.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "mesh/mesh_refinement.hpp"
#include "outputs/outputs.hpp"
#include "driver/driver.hpp"
#include "remap/remap.hpp"
#include "remap/remap_impl.hpp"
#include "utils/utils.hpp"
#include "utils/env_switch.hpp"
#include "utils/mpi_buffer_pool.hpp"

// MPI/OpenMP headers
#if MPI_PARALLEL_ENABLED
#include <mpi.h>
#endif

#if OPENMP_PARALLEL_ENABLED
#include <omp.h>
#endif

#if defined(KOKKOS_ENABLE_HIP)
#include <hip/hip_runtime.h>
#endif

namespace {

//----------------------------------------------------------------------------------------
//! \struct RestartBlockSnapshot
//! \brief one block of the parameters a restart file's header carried, as loaded

struct RestartBlockSnapshot {
  std::string name;
  std::list<InputLine> line;
  std::size_t max_len_parname = 0;
  std::size_t max_len_parvalue = 0;
};

//----------------------------------------------------------------------------------------
//! \fn std::vector<RestartBlockSnapshot> SnapshotParameters(ParameterInput *pin)
//! \brief copy every block, key, value, comment and provenance flag

std::vector<RestartBlockSnapshot> SnapshotParameters(ParameterInput *pin) {
  std::vector<RestartBlockSnapshot> snap;
  for (const InputBlock &b : pin->block) {
    RestartBlockSnapshot s;
    s.name = b.block_name;
    s.line = b.line;
    if (!b.line.empty()) {  // the lengths of a block without lines are never set
      s.max_len_parname = b.max_len_parname;
      s.max_len_parvalue = b.max_len_parvalue;
    }
    snap.push_back(std::move(s));
  }
  return snap;
}

//----------------------------------------------------------------------------------------
//! \fn void ReconcileDeckWithRestart(ParameterInput *pin,
//!                                   const std::vector<RestartBlockSnapshot> &snap)
//! \brief settle a deck loaded over a restart header (-r and -i together)
//!
//! The deck is typically the run's original deck, edited, or an older full dump, which
//! holds the state the run itself maintains next to the settings a person chose.  Only
//! the settings are the deck's to change.
//!  - <saha_runtime>, <mhd_restart>, <hydro_restart> and <tde_amr_state> describe the checkpoint's own layout
//!    and live state.  They keep the header's content whole: lines the deck added are
//!    removed, and a block only the deck has is removed.
//!  - any other key the header holds is restored line for line (comment and
//!    from_restart_header included) when the deck gives the same value.  A deck that
//!    repeats the header therefore leaves the parameters identical to those of -r alone.
//!  - a key whose header comment says the code updated it at run time keeps the header
//!    value when the deck differs, and rank 0 says so.
//!  - every other value is the deck's.  A key the header lacks is the deck's as well, so
//!    a deck can still add a block or a key, and none of the deck's lines counts as read
//!    from the header (LoadFromFile marks all of them so, because a dump ends in
//!    <par_end>).
//! Command-line block/key=value overrides are applied afterwards and win over all of it.

void ReconcileDeckWithRestart(ParameterInput *pin,
                              const std::vector<RestartBlockSnapshot> &snap) {
  const char *const kInternal[] = {"saha_runtime", "mhd_restart", "hydro_restart",
                                   "tde_amr_state"};
  const std::string kRuntimeComment = "# Updated during run time";
  auto find_snap = [&snap](const std::string &name) -> const RestartBlockSnapshot* {
    for (const auto &s : snap) {
      if (s.name == name) return &s;
    }
    return nullptr;
  };

  for (auto itb = pin->block.begin(); itb != pin->block.end();) {
    const RestartBlockSnapshot *sb = find_snap(itb->block_name);
    const bool internal = std::find(std::begin(kInternal), std::end(kInternal),
                                    itb->block_name) != std::end(kInternal);
    if (internal) {
      if (sb == nullptr) {
        itb = pin->block.erase(itb);
        continue;
      }
      itb->line = sb->line;
      itb->max_len_parname = sb->max_len_parname;
      itb->max_len_parvalue = sb->max_len_parvalue;
      ++itb;
      continue;
    }
    for (InputLine &l : itb->line) {
      const InputLine *sl = nullptr;
      if (sb != nullptr) {
        for (const InputLine &c : sb->line) {
          if (c.param_name == l.param_name) {
            sl = &c;
            break;
          }
        }
      }
      if (sl == nullptr) {
        l.from_restart_header = false;
      } else if (sl->param_value == l.param_value) {
        l = *sl;
      } else if (sl->param_comment.find(kRuntimeComment) != std::string::npos) {
        if (global_variable::my_rank == 0) {
          std::cout << "kept from restart: " << itb->block_name << "/" << l.param_name
                    << " = " << sl->param_value << " (deck had " << l.param_value << ")"
                    << std::endl;
        }
        l = *sl;
      } else {
        l.from_restart_header = false;
      }
    }
    // the column widths of ParameterDump follow the lines that stay
    itb->max_len_parname = (sb != nullptr) ? sb->max_len_parname : 0;
    itb->max_len_parvalue = (sb != nullptr) ? sb->max_len_parvalue : 0;
    for (const InputLine &l : itb->line) {
      itb->max_len_parname = std::max(itb->max_len_parname, l.param_name.length());
      itb->max_len_parvalue = std::max(itb->max_len_parvalue, l.param_value.length());
    }
    ++itb;
  }
}

//----------------------------------------------------------------------------------------
//! \struct FromRestart
//! \brief what the restart file being read says about the equations of state, which the
//! <saha_runtime> keys ending in _from_restart hand to them.  The defaults are those of a
//! fresh start.

struct FromRestart {
  std::string hydro_eos = "none";
  std::string mhd_eos = "none";
  std::string hydro_table = "unknown";
  std::string mhd_table = "unknown";
  std::string hydro_table_type = "unknown";
  std::string mhd_table_type = "unknown";
  Real hydro_lte_x = -1.0;
  Real mhd_lte_x = -1.0;
  Real hydro_lte_y = -1.0;
  Real mhd_lte_y = -1.0;
  Real hydro_lte_prad = -1.0;
  Real mhd_lte_prad = -1.0;
  Real hydro_lte_h2 = -1.0;
  Real mhd_lte_h2 = -1.0;
  Real hydro_lte_zpe = -1.0;
  Real mhd_lte_zpe = -1.0;
  // The checkpoint header's own <mhd|hydro>/dual_energy, before any deck override: a GR
  // file written with the removed GR dual energy holds D*kappa in the auxiliary column
  // (hydro/removed_gr_dual_energy.hpp).
  bool hydro_dual_energy = false;
  bool mhd_dual_energy = false;
};

//----------------------------------------------------------------------------------------
//! \fn void SetFromRestart(ParameterInput *pin, bool res_flag, const FromRestart &r)
//! \brief put the restart-derived values where the equations of state read them

void SetFromRestart(ParameterInput *pin, const bool res_flag, const FromRestart &r) {
  pin->SetBoolean("saha_runtime", "restart_active", res_flag);
  pin->SetString("saha_runtime", "hydro_eos_from_restart", r.hydro_eos);
  pin->SetString("saha_runtime", "mhd_eos_from_restart", r.mhd_eos);
  pin->SetString("saha_runtime", "hydro_table_from_restart", r.hydro_table);
  pin->SetString("saha_runtime", "mhd_table_from_restart", r.mhd_table);
  pin->SetString("saha_runtime", "hydro_table_type_from_restart", r.hydro_table_type);
  pin->SetString("saha_runtime", "mhd_table_type_from_restart", r.mhd_table_type);
  pin->SetReal("saha_runtime", "hydro_lte_x_from_restart", r.hydro_lte_x);
  pin->SetReal("saha_runtime", "mhd_lte_x_from_restart", r.mhd_lte_x);
  pin->SetReal("saha_runtime", "hydro_lte_y_from_restart", r.hydro_lte_y);
  pin->SetReal("saha_runtime", "mhd_lte_y_from_restart", r.mhd_lte_y);
  pin->SetReal("saha_runtime", "hydro_lte_prad_from_restart", r.hydro_lte_prad);
  pin->SetReal("saha_runtime", "mhd_lte_prad_from_restart", r.mhd_lte_prad);
  pin->SetReal("saha_runtime", "hydro_lte_h2_from_restart", r.hydro_lte_h2);
  pin->SetReal("saha_runtime", "mhd_lte_h2_from_restart", r.mhd_lte_h2);
  pin->SetReal("saha_runtime", "hydro_lte_zpe_from_restart", r.hydro_lte_zpe);
  pin->SetReal("saha_runtime", "mhd_lte_zpe_from_restart", r.mhd_lte_zpe);
  pin->SetBoolean("saha_runtime", "hydro_dual_energy_from_restart", r.hydro_dual_energy);
  pin->SetBoolean("saha_runtime", "mhd_dual_energy_from_restart", r.mhd_dual_energy);
}

}  // namespace

//----------------------------------------------------------------------------------------
//! \fn int main(int argc, char *argv[])
//! \brief Athena main program

int main(int argc, char *argv[]) {
  std::string input_file, restart_file, run_dir;
  bool iarg_flag = false;  // set to true if -i <file> argument is on cmdline
  bool marg_flag = false;  // set to true if -m        argument is on cmdline
  bool narg_flag = false;  // set to true if -n        argument is on cmdline
  bool  res_flag = false;  // set to true if -r <file> argument is on cmdline
  Real wtlim = 0;

  //--- Step 1. --------------------------------------------------------------------------
  // Initialize environment (must initialize MPI first, then Kokkos)

#if MPI_PARALLEL_ENABLED
#if defined(KOKKOS_ENABLE_HIP)
  // JMF: This is a bizarre workaround to avoid segmentation faults on Frontier.
  // See OLCFDEV-1655: Occasional seg-fault during MPI_Init inside the Frontier
  // documentation.
  (void) hipInit(0);
#endif
#if OPENMP_PARALLEL_ENABLED
  int mpiprv;
  if (MPI_SUCCESS != MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &mpiprv)) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "MPI Initialization failed." << std::endl;
    return(0);
  }
  if (mpiprv != MPI_THREAD_MULTIPLE) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "MPI_THREAD_MULTIPLE must be supported for hybrid parallelization. "
              << MPI_THREAD_MULTIPLE << " : " << mpiprv
              << std::endl;
    MPI_Finalize();
    return(0);
  }
#else  // no OpenMP
  if (MPI_SUCCESS != MPI_Init(&argc, &argv)) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "MPI Initialization failed." << std::endl;
    return(0);
  }
#endif  // OPENMP_PARALLEL_ENABLED
  // Get process id (rank) in MPI_COMM_WORLD
  if (MPI_SUCCESS != MPI_Comm_rank(MPI_COMM_WORLD, &(global_variable::my_rank))) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "MPI_Comm_rank failed." << std::endl;
    MPI_Finalize();
    return(0);
  }

  // Get total number of MPI processes (ranks)
  if (MPI_SUCCESS != MPI_Comm_size(MPI_COMM_WORLD, &global_variable::nranks)) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "MPI_Comm_size failed." << std::endl;
    MPI_Finalize();
    return(0);
  }
#else  // no MPI
  global_variable::my_rank = 0;
  global_variable::nranks  = 1;
#endif  // MPI_PARALLEL_ENABLED

  // Broadcast rank 0's ATHENAK_* environment switches before anything can read one, so a
  // per-rank difference in the launch environment cannot desynchronise the ranks.
  env_switch::Sync();

  Kokkos::initialize(argc, argv);

  //--- Step 2. --------------------------------------------------------------------------
  // Check for command line options and respond.

  for (int i=1; i<argc; i++) {
    // If argv[i] is a 2 character string of the form "-?" then:
    if (*argv[i] == '-'  && *(argv[i]+1) != '\0' && *(argv[i]+2) == '\0') {
      // check that command line options that require arguments actually have them:
      char opt_letter = *(argv[i]+1);
      switch(opt_letter) {
        case 'c':
        case 'h':
        case 'm':
        case 'n':
          break;
        default:
          if ((i+1 >= argc) // no argument after option
              || (*argv[i+1] == '-') ) { // option is followed by another option
            // every rank parses the same command line, so every rank leaves here; a
            // rank-0-only exit left the other ranks waiting in their first collective
            if (global_variable::my_rank == 0) {
              std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                        << std::endl << "-" << opt_letter
                        << " must be followed by a valid argument" << std::endl;
            }
            Kokkos::finalize();
#if MPI_PARALLEL_ENABLED
            MPI_Finalize();
#endif
            return(1);
          }
      }

      // set arguments, flags, or execute tasks specified by options
      switch(*(argv[i]+1)) {
        case 'i':                      // -i <input_file>
          input_file.assign(argv[++i]);
          iarg_flag = true;
          break;
        case 'r':                      // -r <restart_file>
          restart_file.assign(argv[++i]);
          res_flag = true;
          break;
        case 'd':                      // -d <run_directory>
          run_dir.assign(argv[++i]);
          break;
        case 'n':
          narg_flag = true;
          break;
        case 'm':
          marg_flag = true;
          break;
        case 't':                      // -t <hh:mm:ss>
          int wth, wtm, wts;
          std::sscanf(argv[++i], "%d:%d:%d", &wth, &wtm, &wts);
          wtlim = static_cast<Real>(wth*3600 + wtm*60 + wts);
          break;
        case 'c':
          if (global_variable::my_rank == 0) ShowConfig();
          Kokkos::finalize();
#if MPI_PARALLEL_ENABLED
          MPI_Finalize();
#endif
          return(0);
          break;
        case 'h':
        default:
          if (global_variable::my_rank == 0) {
            std::cout << "Athena v" << ATHENA_VERSION_MAJOR << "."
                                    << ATHENA_VERSION_MINOR << std::endl;
            std::cout << "Usage: " << argv[0] << " [options] [block/par=value ...]\n";
            std::cout << "Options:" << std::endl;
            std::cout << "  -i <file>       specify input file [athinput]\n";
            std::cout << "  -r <file>       restart with this file\n";
            std::cout << "  -d <directory>  specify run dir [current dir]\n";
            std::cout << "  -n              parse input file and quit\n";
            std::cout << "  -c              show configuration and quit\n";
            std::cout << "  -m              output mesh structure and quit\n";
            std::cout << "  -t hh:mm:ss     wall time limit for final output\n";
            std::cout << "  -h              this help\n";
            ShowConfig();
          }
          Kokkos::finalize();
#if MPI_PARALLEL_ENABLED
          MPI_Finalize();
#endif
          return(0);
          break;
      }
    } // else if argv[i] not of form "-?" ignore it here (tested in ModifyFromCmdline)
  }

  // print error if input or restart file not given
  if (restart_file.empty() && input_file.empty()) {
    // no input file is given
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "Either an input or restart file must be specified." << std::endl
              << "See " << argv[0] << " -h for options and usage." << std::endl;
    Kokkos::finalize();
#if MPI_PARALLEL_ENABLED
    MPI_Finalize();
#endif
    return(0);
  }

  // Start the wall clock timer. This is done here rather than in the Driver to ensure
  // that the time taken in ProblemGenerator is also captured.
  Kokkos::Timer timer;

  //--- Step 3. --------------------------------------------------------------------------
  // Construct ParameterInput object and load data either from restart or input file.
  // With MPI, the input is read by every rank in parallel using MPI-IO.

  ParameterInput* pinput = new ParameterInput;
  IOWrapper infile, restartfile;
  FromRestart from_restart;
  // read parameters from restart file
  bool single_file_per_rank = false; // DBF: flag for single_file_per_rank for rst files
  if (res_flag) {
    // Check if the path contains "rank_" directory
    size_t rank_pos = restart_file.find("/rank_");
    single_file_per_rank = (rank_pos != std::string::npos);

    // If single_file_per_rank is true, modify the path for the current rank
    if (single_file_per_rank) {
        // Extract the base directory and file name
        size_t last_slash = restart_file.rfind('/');
        std::string base_dir = restart_file.substr(0, rank_pos);
        std::string file_name = restart_file.substr(last_slash + 1);

        // Construct the path for the current rank
        char rank_dir[20];
        std::snprintf(rank_dir, sizeof(rank_dir), "rank_%08d", global_variable::my_rank);
        restart_file = base_dir + "/" + rank_dir + "/" + file_name;
    }

    // Now use restart_file for opening the file
    std::ifstream file_check(restart_file);
    if (!file_check.good()) {
        std::cerr << "Error: Unable to open restart file: " << restart_file << std::endl;
        // Handle the error (e.g., exit the program or use a default configuration)
    }

    // read parameters from restart file
    restartfile.Open(restart_file.c_str(),IOWrapper::FileMode::read,single_file_per_rank);
    pinput->LoadFromFile(restartfile, single_file_per_rank);
    if (pinput->DoesParameterExist("hydro", "eos")) {
      from_restart.hydro_eos = pinput->GetString("hydro", "eos");
    }
    from_restart.hydro_dual_energy =
        pinput->DoesParameterExist("hydro", "dual_energy") &&
        pinput->GetBoolean("hydro", "dual_energy");
    from_restart.mhd_dual_energy =
        pinput->DoesParameterExist("mhd", "dual_energy") &&
        pinput->GetBoolean("mhd", "dual_energy");
    if (pinput->DoesParameterExist("hydro", "table")) {
      from_restart.hydro_table = pinput->GetString("hydro", "table");
    }
    if (pinput->DoesParameterExist("mhd", "eos")) {
      from_restart.mhd_eos = pinput->GetString("mhd", "eos");
    }
    if (pinput->DoesParameterExist("mhd", "table")) {
      from_restart.mhd_table = pinput->GetString("mhd", "table");
    }
    if (pinput->DoesParameterExist("saha_runtime", "hydro_table_type_runtime")) {
      from_restart.hydro_table_type =
          pinput->GetString("saha_runtime", "hydro_table_type_runtime");
    }
    if (pinput->DoesParameterExist("saha_runtime", "mhd_table_type_runtime")) {
      from_restart.mhd_table_type =
          pinput->GetString("saha_runtime", "mhd_table_type_runtime");
    }
    if (pinput->DoesParameterExist("saha_runtime", "hydro_lte_x_runtime")) {
      from_restart.hydro_lte_x = pinput->GetReal("saha_runtime", "hydro_lte_x_runtime");
    }
    if (pinput->DoesParameterExist("saha_runtime", "mhd_lte_x_runtime")) {
      from_restart.mhd_lte_x = pinput->GetReal("saha_runtime", "mhd_lte_x_runtime");
    }
    if (pinput->DoesParameterExist("saha_runtime", "hydro_lte_y_runtime")) {
      from_restart.hydro_lte_y = pinput->GetReal("saha_runtime", "hydro_lte_y_runtime");
    }
    if (pinput->DoesParameterExist("saha_runtime", "mhd_lte_y_runtime")) {
      from_restart.mhd_lte_y = pinput->GetReal("saha_runtime", "mhd_lte_y_runtime");
    }
    if (pinput->DoesParameterExist("saha_runtime", "hydro_lte_prad_runtime")) {
      from_restart.hydro_lte_prad =
          pinput->GetReal("saha_runtime", "hydro_lte_prad_runtime");
    }
    if (pinput->DoesParameterExist("saha_runtime", "mhd_lte_prad_runtime")) {
      from_restart.mhd_lte_prad =
          pinput->GetReal("saha_runtime", "mhd_lte_prad_runtime");
    }
    if (pinput->DoesParameterExist("saha_runtime", "hydro_lte_h2_runtime")) {
      from_restart.hydro_lte_h2 =
          pinput->GetReal("saha_runtime", "hydro_lte_h2_runtime");
    }
    if (pinput->DoesParameterExist("saha_runtime", "mhd_lte_h2_runtime")) {
      from_restart.mhd_lte_h2 =
          pinput->GetReal("saha_runtime", "mhd_lte_h2_runtime");
    }
    if (pinput->DoesParameterExist("saha_runtime", "hydro_lte_zpe_runtime")) {
      from_restart.hydro_lte_zpe =
          pinput->GetReal("saha_runtime", "hydro_lte_zpe_runtime");
    }
    if (pinput->DoesParameterExist("saha_runtime", "mhd_lte_zpe_runtime")) {
      from_restart.mhd_lte_zpe =
          pinput->GetReal("saha_runtime", "mhd_lte_zpe_runtime");
    }
  }

  // read parameters from input file.  If both -r and -i are specified, this overrides
  // the restart file's parameters except the state the run itself maintains
  // (ReconcileDeckWithRestart)
  if (iarg_flag) {
    std::vector<RestartBlockSnapshot> restart_parameters;
    if (res_flag) restart_parameters = SnapshotParameters(pinput);
    infile.Open(input_file.c_str(), IOWrapper::FileMode::read);
    pinput->LoadFromFile(infile);
    infile.Close();
    if (res_flag) ReconcileDeckWithRestart(pinput, restart_parameters);
    pinput->CheckBlockNames();
  }
  pinput->ModifyFromCmdline(argc, argv);
  SetFromRestart(pinput, res_flag, from_restart);

  // Refuse inputs that enable a module or a combination that this public release omits.
  {
    struct RemovedModule {const char *module; const char *block; const char *key;
                          bool key_is_flag;};
    // an empty key means that the block itself enables the module; a key that is not a
    // flag enables it whenever it is present
    const RemovedModule removed[] = {
      {"GR M1 radiation", "radiation_m1", "", true},
      {"GR M1 radiation", "bns_nurates", "", true},
      {"GR M1 radiation", "photons", "", true},
      {"Newtonian M1 radiation", "radiation_newt", "", true},
      {"Newtonian M1 radiation", "problem", "rad_lte_seed", true},
      {"Newtonian M1 radiation", "problem", "rad_pgen_opacity", true},
      {"Newtonian M1 radiation", "mesh_refinement", "drad_max", false},
      {"hybrid force-free", "forcefree", "", true},
      {"hybrid force-free", "forcefree_restart", "", true},
      {"hybrid force-free", "mhd", "hybrid_forcefree", true},
      {"two-temperature thermodynamics", "two_temperature", "", true},
      {"nonthermal electrons", "two_temperature", "nonthermal_enable", true},
    };
    for (const auto &r : removed) {
      const bool has_key = (r.key[0] != '\0');
      if (!pinput->DoesBlockExist(r.block)) continue;
      if (has_key && !pinput->DoesParameterExist(r.block, r.key)) continue;
      if (has_key && r.key_is_flag && !pinput->GetBoolean(r.block, r.key)) continue;
      if (global_variable::my_rank == 0) {
        std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                  << std::endl << "The " << r.module << " module is not included in "
                  << "this public release of the code (input "
                  << (has_key ? "key " : "block ") << r.block
                  << (has_key ? "/" : "") << r.key << ")." << std::endl;
      }
      std::exit(EXIT_FAILURE);
    }
    if (pinput->IsLATEnabled() && pinput->DoesBlockExist("mhd")) {
      if (global_variable::my_rank == 0) {
        std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                  << std::endl
                  << "LAT for MHD and GRMHD is not included in this public release. "
                  << "It is available on request from Hong-Xuan Jiang "
                  << "(masterjoe2000@outlook.com)." << std::endl;
      }
      std::exit(EXIT_FAILURE);
    }
  }

  const bool hydro_lat_requested = pinput->IsLATEnabled();
  const bool self_gravity_requested =
      pinput->DoesBlockExist("gravity") &&
      pinput->GetOrAddBoolean("gravity", "self_gravity", true);
  const std::string pgen_name =
      pinput->DoesBlockExist("problem") ?
      pinput->GetOrAddString("problem", "pgen_name", "none") : "none";
  const bool analytic_bh_source_problem =
      (pgen_name == "tde_external");
  const bool external_bh_gravity_requested =
      pinput->DoesBlockExist("problem") &&
      pinput->GetOrAddBoolean("problem", "external_bh_gravity_source",
                              analytic_bh_source_problem);
  const bool self_gravity_lat_has_solve_dt =
      pinput->DoesParameterExist("gravity", "solve_dt") &&
      pinput->GetReal("gravity", "solve_dt") > 0.0;
  if (hydro_lat_requested && self_gravity_requested &&
      !self_gravity_lat_has_solve_dt) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl
              << "time/lat with gravity/self_gravity=true requires "
              << "gravity/solve_dt > 0 so the frozen-potential LAT window is bounded."
              << std::endl;
    std::exit(EXIT_FAILURE);
  }
  // The lat_neighbor_limiter refusal for same-level LAT with source-coupled fluids is
  // made by the Driver constructor, from the source terms actually constructed.

  // Dump input parameters and quit if code was run with -n option.
  if (narg_flag) {
    if (global_variable::my_rank == 0) pinput->ParameterDump(std::cout);
    if (res_flag) restartfile.Close(single_file_per_rank);
    delete pinput;
    mpi_buffer_pool::Finalize();
    Kokkos::finalize();
#if MPI_PARALLEL_ENABLED
    MPI_Finalize();
#endif
    return(0);
  }

  //--- Steps 4-8. ----------------------------------------------------------------------
  // One pass builds the Mesh, runs the Driver and cleans up.  A planned
  // <remap>/auto_expand expansion ends a pass at a checkpoint and puts the grown mesh and
  // a <remap> start from that checkpoint into the input (remap::ApplyExpansion), so the
  // next pass is a fresh start on the grown mesh, in this process, with the same timer
  // and run directory.
  bool run_dir_set = false;
  for (;;) {
    //--- Step 4. ------------------------------------------------------------------------
    // Construct Mesh.  Then build MeshBlockTree and add MeshBlockPack containing
    // MeshBlocks on this rank.  Latter cannot be performed in Mesh constructor since it
    // requires pointer to Mesh.

    Mesh* pmesh = new Mesh(pinput);
    if (!res_flag) {
      pmesh->BuildTreeFromScratch(pinput);
    } else {
      pmesh->BuildTreeFromRestart(pinput, restartfile, single_file_per_rank);
    }

    //  If code was run with -m option, write mesh structure to file and quit.
    if (marg_flag) {
      if (global_variable::my_rank == 0) {pmesh->WriteMeshStructure();}
      if (res_flag) {restartfile.Close(single_file_per_rank);}
      delete pmesh;
      delete pinput;
      mpi_buffer_pool::Finalize();
      Kokkos::finalize();
#if MPI_PARALLEL_ENABLED
      MPI_Finalize();
#endif
      return(0);
    }

    //--- Step 5. ------------------------------------------------------------------------
    // Add coordinates and physics modules to MeshBlockPack, and set initial conditions.
    // Note these steps must occur after Mesh (including MeshBlocks and MeshBlockPack)
    // is fully constructed.

    pmesh->AddCoordinatesAndPhysics(pinput);

    // MeshRefinement depends on the enrolled physics modules for refinement data and
    // load-balance metadata.  Construct it before the problem generator, since custom
    // problems in this fork configure AMR cadence and initial refinement through pmr.
    if (pmesh->multilevel) {
      pmesh->pmr = new MeshRefinement(pmesh, pinput);
    }

    if (!res_flag) {
      // set ICs using ProblemGenerator constructor for new runs
      pmesh->pgen = std::make_unique<ProblemGenerator>(pinput, pmesh);
    } else {
      // read ICs from restart file using ProblemGenerator constructor for restarts
      pmesh->pgen = std::make_unique<ProblemGenerator>(pinput,
                                                       pmesh,
                                                       restartfile,
                                                       single_file_per_rank);
      restartfile.Close(single_file_per_rank);
    }
    // A <remap>/floor_reset start whose floors did not match its source has put the
    // matching floors into the input; the physics is built again with them.
    if (remap::TakeFloorResetPass()) {
      delete pmesh;
      continue;
    }

    //--- Step 6. ------------------------------------------------------------------------
    // Construct Driver and Outputs. Actual outputs (including initial conditions) are
    // made in Driver.Initialize(). Add wall clock timer to Driver if necessary.

    if (!run_dir_set) ChangeRunDir(run_dir);
    run_dir_set = true;
    Driver* pdriver = new Driver(pinput, pmesh, wtlim, &timer);
    Outputs* pout = new Outputs(pinput, pmesh);

    //--- Step 7. ------------------------------------------------------------------------
    // Execute Driver.
    //    1. Initial conditions set in Driver::Initialize()
    //    2. TaskList(s) executed in Driver::Execute()
    //    3. Any final analysis or diagnostics run in Driver::Finalize()

    pdriver->Initialize(pmesh, pinput, pout, res_flag);
    pdriver->Execute(pmesh, pinput, pout);
    pdriver->Finalize(pmesh, pinput, pout);
    const bool expand = remap::ApplyExpansion(pmesh, pinput, pout);

    //--- Step 8. ------------------------------------------------------------------------
    // clean up, and start the next pass or terminate
    // Note anything containing a Kokkos::view must be deleted before Kokkos::finalize()

    delete pout;
    delete pdriver;
    delete pmesh;
    if (!expand) break;
    res_flag = false;
    from_restart = FromRestart();
    SetFromRestart(pinput, res_flag, from_restart);
  }
  delete pinput;
  mpi_buffer_pool::Finalize();
  Kokkos::finalize();
#if MPI_PARALLEL_ENABLED
  MPI_Finalize();
#endif
  return(0);
}
