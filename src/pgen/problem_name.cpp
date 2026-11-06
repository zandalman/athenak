//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file problem_name.cpp
//! \brief Name of the problem generator linked into this executable.  The only
//! per-executable unit besides the problem generator itself: it is compiled once per
//! executable with ATHENA_PROBLEM_NAME defined, so the shared core objects do not depend on
//! which problem generator an executable carries.

#include "pgen/problem_name.hpp"

#ifndef ATHENA_PROBLEM_NAME
#error "ATHENA_PROBLEM_NAME must be defined when compiling problem_name.cpp"
#endif

const char *ProblemGeneratorName() { return ATHENA_PROBLEM_NAME; }
