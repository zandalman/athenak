"""Variable-length part of an AthenaK restart file's binary header.

After the ASCII parameter dump (terminated by <par_end>) a checkpoint holds the mesh
header, each MeshBlock's LogicalLocation (4 x i4) and load-balance cost (f4), and then,
when an adaptive mesh wrote it, each MeshBlock's cycles since its last refinement (i4,
src/outputs/restart.cpp).  The writer records how many in the parameter dump as
<mesh_refinement>/restart_ncyc_since_ref_count, which is how a reader knows the list is
there.  The restart readers of tst/analysis skip it through this function.
"""

import re

_AGE_COUNT = re.compile(rb"^\s*restart_ncyc_since_ref_count\s*=\s*(\d+)", re.M)


def refinement_age_bytes(parameter_dump: bytes) -> int:
    """Bytes of the refinement-age list that follows the cost list (0 if none)."""
    match = _AGE_COUNT.search(parameter_dump)
    return 4 * int(match.group(1)) if match else 0
