#!/bin/bash

# Set problem
PROBLEM="injection"

# Default behavior
GPU=false

# Parse arguments
while getopts ":g" opt; do
  case $opt in
    g)
      GPU=true
      ;;
    \?)
      echo "Invalid option: -$OPTARG"
      exit 1
      ;;
  esac
done

#rm -r $ATHENA_DIR/athenak/build/*
if $GPU; then
  cmake -D PROBLEM=$PROBLEM -D Athena_ENABLE_MPI=ON -D Kokkos_ENABLE_CUDA=On -D Kokkos_ARCH_AMPERE80=On -D CMAKE_CXX_COMPILER=$ATHENA_DIR/athenak/kokkos/bin/nvcc_wrapper -S $ATHENA_DIR/athenak -B $ATHENA_DIR/athenak/build
else
  cmake -D PROBLEM=$PROBLEM -D Athena_ENABLE_MPI=ON -S $ATHENA_DIR/athenak -B $ATHENA_DIR/athenak/build
fi
cmake --build $ATHENA_DIR/athenak/build --parallel 8
