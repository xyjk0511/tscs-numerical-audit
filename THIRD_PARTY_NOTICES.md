# Source attribution and file-level rights

This release is multi-licensed. The manifest assigns rights per file. Software
is publicly viewable, but no open-source software license is granted.

## Alvar anatomy and mixed derivatives

Alvar anatomical whole-body model with anisotropic skeletal muscles, version
17.0.0, was created by Otto Kangasmaa (2025),
https://doi.org/10.5281/zenodo.14901199, under CC BY-SA 4.0.
The source README credits Alvar copyright 2018 Ilkka Laakso and Tuukka Lehtinen
under the same license. Project changes include anatomy cropping, meshing,
electrode configuration, field solutions, registration of root trajectories,
and numerical post-processing. See each manifest entry for its transformation.
Alvar-derived and mixed data are offered under CC BY-SA 4.0 to the extent
copyright or database rights apply. This is a conservative treatment of
anatomy-dependent outputs, not a claim that every numerical fact is protected.
https://creativecommons.org/licenses/by-sa/4.0/

## LSSM markers

Markers, QC results and generated models, version 1, were created by Jionghui
Liu, Wenqi Zhang, Yuxing Zhou, Linhao Xu, Yinghua Chu and Fumin Jia (2024),
https://doi.org/10.6084/m9.figshare.26403442.v1, under CC BY 4.0.
This dataset was obtained from the figshare database.
The marker asset retains the original neutral source geometry codes and native
LPS millimetre coordinates. Registered trajectories and mixed numerical outputs
retain this attribution and also carry the Alvar notice above.
https://creativecommons.org/licenses/by/4.0/

## Exclusions and processing record

No original MRI archive, original Alvar anatomical archive, participant identity
key, private Git history, or copied/adapted article figure is distributed.
The source article https://doi.org/10.1038/s41597-024-03919-4 has a separate
CC BY-NC-ND 4.0 license; the dataset license does not replace that article license.
No endorsement by the source creators is implied.

Frozen originals remain unchanged. Public copies replace absolute local-path
strings with `<local-path-redacted>`. Changed field names and original/public
hashes appear in `manifests/sanitization_record.json`. Numerical arrays and
non-path scientific values are preserved. HDF5 dataset names, shapes, dtypes
and content hashes are compared before and after metadata edits. Public source
pseudonyms distinguish geometries; they are not names or an identity key.
The scan records observed content; it is not an institutional ethics decision.

## Project software

Copyright 2026 Buchan Liang. All rights reserved. Source code and tests are
made publicly viewable for inspection. The author has not granted an
open-source license or general permission to modify or redistribute software.
Data licenses do not apply to the software. GitHub's platform terms continue
to apply to platform viewing and other platform operations.
