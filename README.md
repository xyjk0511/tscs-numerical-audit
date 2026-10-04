# tSCS numerical audit: archived data and inspectable analysis

Companion materials for *Spatial scale and normalization alter segmental
comparisons in a transcutaneous spinal cord stimulation model*.

Author: Buchan Liang, University of California, Davis.
This is a preparation-stage numerical study, not an accepted publication.

The release contains the exact archived mesh, four FEM fields, registered root
ensembles and electrode configurations, source-native LSSM markers, and 5,880
saved scalar-potential profiles with 392 path-descriptor records. Portable
analysis tables and complete AF vectors are in the repository. Every scientific
payload is preserved; local-path metadata edits are recorded in the manifest.

## Downloads

Use the assets at the tagged `v1.0.0` GitHub release. Compare their SHA-256 values
with `manifests/SHA256SUMS`. Each archive also contains its own manifest, source
notices, checksums and applicable data license.

- `mesh-fields-specs-v1.0.0.zip`: archived mesh, four fields and electrode specs.
- `registered-ensembles-v1.0.0.zip`: registered trajectories and existing outputs.
- `lssm-native-markers-v1.0.0.zip`: source-native markers, CC BY 4.0.
- `potential-profiles-v1.0.0.zip`: saved potentials and independent-replay inputs.

The original sources are Alvar v17.0.0
(https://doi.org/10.5281/zenodo.14901199) and LSSM v1
(https://doi.org/10.6084/m9.figshare.26403442.v1).
Raw MRI is available from its original repository
(https://doi.org/10.6084/m9.figshare.26403595.v1) and is not mirrored here.

## Audit commands and scope

The commands below document the author's verification workflow. Public viewing
does not grant an open-source software license; see `LICENSE` before software reuse.
The numerical environment used NumPy 2.5.2, SciPy 1.18.0, h5py 3.16.0,
pytest 9.1.1 and Python 3.13.5. Package installation is not automated.

```text
python paper/numerical_audit/matched_analysis.py
python paper/numerical_audit/replay_common_support_tables.py --output replay
python -m pytest tests/test_common_support.py tests/test_paper_matched_analysis.py -q -W error
```

After extracting the profile asset into the repository root:

```text
python paper/numerical_audit/verify_common_support_profiles.py profile_replay --report profile-check.json
```

This independent verifier integrates the sampled potentials and checks the
reported descriptors and geometry-weighted comparisons. It does not solve FEM.
The portable replay regenerates comparison tables from supplied AF vectors.
The archived fields and trajectories are available for inspection and further
analysis; this release does not claim a self-contained original FEM rebuild or
a complete locator-cache environment for the existing field-resampling runner.

The study concerns one shared volume conductor and retrospective numerical
sensitivity. Sampling stability does not establish FEM mesh convergence,
physiological recruitment, patient-specific efficacy or clinical safety.

## Rights and ethics

Code and tests are publicly viewable, with all rights reserved by Buchan Liang;
no open-source software license is granted. Data have separate file-level
Creative Commons licenses. Preserve source attribution and modification notices.
See `THIRD_PARTY_NOTICES.md` and `manifests/release_manifest.json`.

The original LSSM acquisition reports Fudan University approval FE23166I and
consent to anonymized sharing. This secondary computational study recruited no
participants and delivered no intervention. The original approval is not an
institutional approval, exemption or formal determination for this analysis.

## Integrity records

`manifests/release_manifest.json` records every scientific/code release member,
source lineage, license and original/public hash. `sanitization_record.json`
records metadata-only changes. `assets.json` records archive sizes and hashes.
`metadata_identity_validation.json` supplies paired original/public dataset
digests and confirms that each changed value was a complete filesystem locator;
the four changed code literals are `Path` defaults, with all other AST nodes
unchanged. Explicit external-input paths are needed for those ancillary defaults.
No private repository history or frozen original is changed by this release.
