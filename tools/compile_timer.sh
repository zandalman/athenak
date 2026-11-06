#!/bin/bash
# Optional per-translation-unit compile timer, used as a CMake compiler launcher:
#   cmake ... -DATHENA_COMPILE_TIMER=ON   (or -DCMAKE_CXX_COMPILER_LAUNCHER=<this script>
#   where Kokkos allows a launcher; Kokkos+nvc++ does not)
# Runs the compile command unchanged and appends "<seconds> <source file>" to
# $ATHENA_COMPILE_TIMER_LOG (ATHENA_COMPILE_TIMER=ON sets it to <build>/compile_times.log;
# otherwise compile_times.log in the current directory, where make
# runs the compiler).  Sort the slowest first with:  sort -rn compile_times.log | head
log="${ATHENA_COMPILE_TIMER_LOG:-compile_times.log}"
src=""
prev=""
for a in "$@"; do
  if [ "$prev" = "-c" ]; then src="$a"; fi
  prev="$a"
done
t0=$(date +%s.%N)
"$@"
rc=$?
t1=$(date +%s.%N)
if [ $rc -eq 0 ]; then
  awk -v a="$t0" -v b="$t1" -v f="$src" 'BEGIN{printf "%.1f %s\n", b-a, f}' >> "$log"
fi
exit $rc
