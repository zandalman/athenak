# Master CMakeLists.txt file for Athena++ (Kokkos version)
#
#------ start with setting various compiler, environment, and build defaults -------------

cmake_minimum_required(VERSION 3.10)
project(athena VERSION 0.1 LANGUAGES C CXX)

# Disable in-source builds to prevent source tree corruption.
if( "${CMAKE_SOURCE_DIR}" STREQUAL "${CMAKE_BINARY_DIR}" )
  message(FATAL_ERROR "FATAL: In-source builds disabled. Please create a separate build directory in which to run cmake, e.g. mkdir build; cd build; cmake ../" )
endif()

# Athena++ (and Kokkos) require C++17 standard as minimum
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Force flag to denote C++ standard, as deprecated c++1z is not supported by nvcc.
set(CMAKE_CXX17_STANDARD_COMPILE_OPTION "-std=c++17")

# Kokkos recommends NOT using default compiler extensions
set(CMAKE_CXX_EXTENSIONS OFF)

# following policy required to enable target_link_libraries() below 
cmake_policy(SET CMP0079 NEW)

# Use 'Release' as default build type. Use 'Debug' for testing.
if(NOT CMAKE_BUILD_TYPE)
  message(STATUS "Setting build type to 'Release' as none was specified.")
  set(CMAKE_BUILD_TYPE Release)
endif()

#------ default values for compile time options  -----------------------------------------

option(Athena_SINGLE_PRECISION "Compile for single precision" OFF)
option(Athena_ENABLE_MPI "Compile with MPI parallelism enabled" OFF)
option(Athena_ENABLE_OPENMP "Compile with OpenMP parallelism enabled" OFF)
option(Athena_ENABLE_RNS "Compile with RNS support enabled" OFF)
set(PROBLEM built_in_pgens CACHE STRING "Name of problem generator function")

#------ set macros exported to config.hpp ------------------------------------------------

# set single precision macro (true/false)
if (Athena_SINGLE_PRECISION)
  set(SINGLE_PRECISION_ENABLED 1)
else()
  set(SINGLE_PRECISION_ENABLED 0)
endif()

# set MPI macro (true/false)
set(ENABLE_MPI OFF)
if (Athena_ENABLE_MPI)
  find_package(MPI COMPONENTS CXX)
  if (NOT MPI_FOUND)
    message(FATAL_ERROR "MPI package required but could not be found.")
  endif()
  set(ENABLE_MPI ON)
  set(Kokkos_ENABLE_MPI ON CACHE BOOL "Compile Kokkos with MPI")
endif()
if (ENABLE_MPI)
  set(MPI_PARALLEL_ENABLED 1)
else ()
  set(MPI_PARALLEL_ENABLED 0)
endif()

# set OpenMP macro (true/false)
set(ENABLE_OPENMP OFF)
if (Athena_ENABLE_OPENMP)
  find_package(OpenMP COMPONENTS CXX)
  if (NOT OpenMP_FOUND)
    message(FATAL_ERROR "OpenMP package is required but could not be found.")
  endif()
  set(ENABLE_OPENMP ON)
  set(Kokkos_ENABLE_OPENMP ON CACHE BOOL "Compile Kokkos with OpenMP")
endif()
if (ENABLE_OPENMP)
  set(OPENMP_PARALLEL_ENABLED 1)
else()
  set(OPENMP_PARALLEL_ENABLED 0)
endif()

# Resolve legacy problem-generator basenames after the IAS source-tree reorganization.
# Direct files in src/pgen retain priority; otherwise require one unique recursive match.
function(athena_resolve_problem name outvar)
  set(resolved "${name}")
  if (NOT EXISTS "${CMAKE_SOURCE_DIR}/src/pgen/${name}.cpp")
    file(GLOB_RECURSE matches
         RELATIVE "${CMAKE_SOURCE_DIR}/src/pgen"
         "${CMAKE_SOURCE_DIR}/src/pgen/*/${name}.cpp")
    list(LENGTH matches match_count)
    if (match_count EQUAL 0)
      message(FATAL_ERROR
              "Unable to locate problem generator '${name}' under src/pgen")
    elseif (match_count GREATER 1)
      message(FATAL_ERROR
              "Problem generator '${name}' is ambiguous: ${matches}")
    endif()
    list(GET matches 0 source)
    string(REGEX REPLACE "\\.cpp$" "" resolved "${source}")
  endif()
  set(${outvar} "${resolved}" PARENT_SCOPE)
endfunction()

set(ATHENA_PROBLEM_REQUESTED "${PROBLEM}")
if (NOT "${PROBLEM}" STREQUAL "built_in_pgens")
  athena_resolve_problem("${PROBLEM}" ATHENA_PROBLEM_RESOLVED)
  # src/CMakeLists.txt consumes PROBLEM as a path relative to src/pgen.
  set(PROBLEM "${ATHENA_PROBLEM_RESOLVED}")
endif()

# Further problem generators linked into extra executables athena_<name> that share this
# build's object files (src/CMakeLists.txt).  Each is resolved like PROBLEM.
set(ATHENA_EXTRA_PROBLEMS "" CACHE STRING
    "Semicolon list of extra problem generators; builds athena_<name> next to athena")
set(ATHENA_EXTRA_RESOLVED)
foreach(extra IN LISTS ATHENA_EXTRA_PROBLEMS)
  if ("${PROBLEM}" STREQUAL "built_in_pgens")
    message(FATAL_ERROR "ATHENA_EXTRA_PROBLEMS requires PROBLEM to name a problem generator")
  endif()
  athena_resolve_problem("${extra}" extra_resolved)
  # The external initial-data libraries below are selected from PROBLEM alone and linked
  # into every executable of the build, so an extra problem cannot bring its own.
  if ("${extra_resolved}" MATCHES
      "^(z4c/(two_punctures|spectre)|dyn_grmhd/(elliptica|kadath|lorene|sgrid|rns))/")
    message(FATAL_ERROR
            "ATHENA_EXTRA_PROBLEMS entry '${extra}' (${extra_resolved}) needs an external "
            "library, which is configured from PROBLEM only: build it as PROBLEM in a "
            "build directory of its own")
  endif()
  list(APPEND ATHENA_EXTRA_RESOLVED "${extra}=${extra_resolved}")
endforeach()

# set user problem generator flag
if (NOT ${PROBLEM} STREQUAL "built_in_pgens")
  message(STATUS "Including user-specified problem generator file: ${PROBLEM}")
  set(USER_PROBLEM_ENABLED 1)
else()
  set(USER_PROBLEM_ENABLED 0)
endif()

#------ set various Kokkos option --------------------------------------------------------

# Tell Kokkos to vectorize aggressively
# Kokkos prefers this capitalization for debugging reasons
SET (Kokkos_ENABLE_AGGRESSIVE_VECTORIZATION ON CACHE BOOL
     "Kokkos aggressive vectorization")

# Tell Kokkos we need lambdas in Cuda.
if (Kokkos_ENABLE_CUDA)
  SET (Kokkos_ENABLE_CUDA_LAMBDA ON CACHE BOOL "Enable lambda expressions in CUDA")
endif()

# If this is a debug build, set kokkos debug on
if (${CMAKE_BUILD_TYPE} STREQUAL "Debug")
  message(STATUS "Enabling Kokkos debug mode")
  set(Kokkos_ENABLE_DEBUG ON CACHE BOOL "Most general debug settings")
  set(Kokkos_ENABLE_DEBUG_BOUNDS_CHECK ON CACHE BOOL
     "Bounds checking on Kokkos views")
  set(Kokkos_ENABLE_DEBUG_DUALVIEW_MODIFY_CHECK ON CACHE BOOL
      "Sanity checks on Kokkos DualView")
endif()

#------ create configuration header file from template in /src directory -----------------

# add subdirectories for source code and include files
add_subdirectory(src)
if (EXISTS ${CMAKE_SOURCE_DIR}/kokkos/CMakeLists.txt)
  add_subdirectory(${CMAKE_SOURCE_DIR}/kokkos ${CMAKE_BINARY_DIR}/kokkos-build)
elseif (EXISTS ${CMAKE_SOURCE_DIR}/../athenak/kokkos/CMakeLists.txt)
  add_subdirectory(${CMAKE_SOURCE_DIR}/../athenak/kokkos ${CMAKE_BINARY_DIR}/kokkos-build)
else()
  message(FATAL_ERROR "Unable to locate Kokkos source directory.")
endif()
include_directories(${Kokkos_INCLUDE_DIRS_RET})

# Optional per-translation-unit compile times (tools/compile_timer.sh).  Kokkos refuses a
# CMAKE_CXX_COMPILER_LAUNCHER with a compiler that goes through its own launcher, so the
# timer is prefixed to whatever launch rule is in force.
option(ATHENA_COMPILE_TIMER "Append 'seconds file' per compiled unit to compile_times.log" OFF)
if (ATHENA_COMPILE_TIMER)
  get_property(athena_launch GLOBAL PROPERTY RULE_LAUNCH_COMPILE)
  set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE
      "${CMAKE_COMMAND} -E env ATHENA_COMPILE_TIMER_LOG=${CMAKE_BINARY_DIR}/compile_times.log ${CMAKE_SOURCE_DIR}/tools/compile_timer.sh ${athena_launch}")
endif()

target_link_libraries(athena_usage INTERFACE Kokkos::kokkos)
if (ENABLE_MPI)
  target_link_libraries(athena_usage INTERFACE MPI::MPI_CXX)
endif()
if (ENABLE_OPENMP)
  target_link_libraries(athena_usage INTERFACE OpenMP::OpenMP_CXX)
endif()
if (${PROBLEM} MATCHES "z4c/two_punctures/*")
	target_include_directories(athena_usage INTERFACE ${CMAKE_SOURCE_DIR}/twopuncturesc/include)
	target_link_libraries(athena_usage INTERFACE ${CMAKE_SOURCE_DIR}/twopuncturesc/lib/libTwoPunctures.a)
	#target_include_directories(athena_usage INTERFACE ${CMAKE_SOURCE_DIR}/gsl/include)
	#target_link_libraries(athena_usage INTERFACE ${CMAKE_SOURCE_DIR}/gsl/lib/libgsl.a)
        #target_link_libraries(athena_usage INTERFACE ${CMAKE_SOURCE_DIR}/gsl/lib/libgslcblas.a)
	find_package(GSL REQUIRED)
	target_link_libraries(athena_usage INTERFACE GSL::gsl GSL::gslcblas)
	set(TWO_PUNCTURES 1)
else()
	set(TWO_PUNCTURES 0)
endif()

if (${PROBLEM} MATCHES "z4c/spectre/*")
  # Load binary black hole initial data from the SpECTRE code:
  # https://spectre-code.org
  #
  # Uses the `BundledExporter` library from SpECTRE to read the initial data.
  # You can help CMake find the library by setting
  # `-D SPECTRE_ROOT=path/to/spectre/build` to the path where SpECTRE is
  # installed.
  find_library(
    SPECTRE_EXPORTER_LIB
    NAMES BundledExporter
    PATH_SUFFIXES lib
    PATHS ${SPECTRE_ROOT})
  find_path(
    SPECTRE_EXPORTER_INCLUDE_DIR
    NAMES spectre/Exporter.hpp
    PATH_SUFFIXES include
    PATHS ${SPECTRE_ROOT})
  add_library(spectre::Exporter UNKNOWN IMPORTED)
  set_target_properties(
    spectre::Exporter
    PROPERTIES IMPORTED_LOCATION ${SPECTRE_EXPORTER_LIB}
               INTERFACE_INCLUDE_DIRECTORIES ${SPECTRE_EXPORTER_INCLUDE_DIR})
  target_link_libraries(athena_usage INTERFACE spectre::Exporter)
endif()

if (${PROBLEM} MATCHES "dyn_grmhd/elliptica/*")
  target_include_directories(athena_usage INTERFACE
      ${CMAKE_SOURCE_DIR}/Elliptica_ID_Reader/include)
  target_link_libraries(athena_usage INTERFACE
      ${CMAKE_SOURCE_DIR}/Elliptica_ID_Reader/lib/libelliptica_id_reader.a)
  if (Athena_ENABLE_OPENMP)
  else ()
    find_package(OpenMP COMPONENTS CXX)
    target_link_libraries(athena_usage INTERFACE OpenMP::OpenMP_CXX)
  endif()
endif()

if (Athena_ENABLE_RNS OR "${PROBLEM}" MATCHES "^dyn_grmhd/rns/")
  # Support for RNS library for rotating neutron stars. See
  # https://github.com/computationalrelativity/RNSC
  target_include_directories(athena_usage INTERFACE
      ${CMAKE_SOURCE_DIR}/rnsc/include)
  target_link_libraries(athena_usage INTERFACE
      ${CMAKE_SOURCE_DIR}/rnsc/lib/libRNS.a)
endif()

if (${PROBLEM} MATCHES "dyn_grmhd/sgrid/*")
  target_link_libraries(athena_usage INTERFACE
      ${CMAKE_SOURCE_DIR}/sgrid/lib/libsgrid.a)
endif()

if (${PROBLEM} MATCHES "dyn_grmhd/kadath/*")
  # Kadath (FUKa) BNS initial data reader.
  # Expects Kadath to be linked at ${CMAKE_SOURCE_DIR}/kadath
  # with libkadath.a in its lib/ directory.
  find_package(Boost REQUIRED)
  target_include_directories(athena_usage INTERFACE
      ${CMAKE_SOURCE_DIR}/kadath/include
      ${CMAKE_SOURCE_DIR}/kadath/include/Kadath_point_h
      #${CMAKE_SOURCE_DIR}/kadath/include/Solvers
      ${Boost_INCLUDE_DIRS})
  target_link_libraries(athena_usage INTERFACE
      ${CMAKE_SOURCE_DIR}/kadath/lib/libkadath.a)
  # Kadath depends on Boost, FFTW3 (spectral transforms) and LAPACK (linear algebra)
  find_package(PkgConfig QUIET)
  if (PKG_CONFIG_FOUND)
    pkg_check_modules(FFTW3 fftw3)
  endif()
  if (FFTW3_FOUND)
    target_link_libraries(athena_usage INTERFACE ${FFTW3_LIBRARIES})
  else()
    find_library(FFTW3_LIB NAMES fftw3)
    target_link_libraries(athena_usage INTERFACE ${FFTW3_LIB})
  endif()
  find_package(LAPACK REQUIRED)
  target_link_libraries(athena_usage INTERFACE ${LAPACK_LIBRARIES})
  find_package(GSL REQUIRED)
  target_link_libraries(athena_usage INTERFACE GSL::gsl GSL::gslcblas)
  if (Athena_ENABLE_OPENMP)
  else ()
    find_package(OpenMP COMPONENTS CXX)
    target_link_libraries(athena_usage INTERFACE OpenMP::OpenMP_CXX)
  endif()
endif()

if (${PROBLEM} MATCHES "dyn_grmhd/lorene/*")
  # FIXME(user): The way the Fortran libraries here are linked is not great. Because these
  # are implicit libraries, we shouldn't have to add them manually, but at least on some
  # machines this seems to be necessary. We should find out if there's a better way to
  # do this.
  enable_language(Fortran)
  include(FortranCInterface)
  FortranCInterface_VERIFY(CXX)
  find_package(BLAS REQUIRED)
  find_package(LAPACK REQUIRED)
  find_package(GSL REQUIRED)
  target_include_directories(athena_usage INTERFACE
      ${CMAKE_SOURCE_DIR}/Lorene/Export/C++/Include
      ${CMAKE_SOURCE_DIR}/Lorene/C++/Include)
  target_link_libraries(athena_usage INTERFACE
      ${CMAKE_SOURCE_DIR}/Lorene/Lib/liblorene_export.a
      ${CMAKE_SOURCE_DIR}/Lorene/Lib/liblorene.a
      ${CMAKE_SOURCE_DIR}/Lorene/Lib/liblorenef77.a
      ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES}
      GSL::gsl ${CMAKE_Fortran_IMPLICIT_LINK_LIBRARIES})
endif()

# enable include of header files created in build directories
target_include_directories(athena_usage INTERFACE
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_CURRENT_BINARY_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}/src/pgen
)

configure_file(config.hpp.in config.hpp)

install( TARGETS athena ${ATHENA_EXTRA_TARGETS} DESTINATION bin )
