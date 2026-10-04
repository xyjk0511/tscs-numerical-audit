"""Place LSSM root complexes with the level-anchored map and sample the FEM field.

## What this does

`align_roots_by_level.py` replaced the conus-distance anchor with a longitudinal
map fitted on root exit heights against Alvar's named discs, holding the
opposite side out. This executable applies that map to every available
root-complex line of every subject, carries each line into Alvar with the
transverse/angular construction from `place_lumbosacral_roots.py`, and samples
the existing direct-FEM lead-field artifact along the placed lines.

No new FEM solve runs here. The field is the on-disk
`alvar-direct-fem-l9-63e0cadc-field.h5` artifact, which is bound to the exact
mesh the roots are placed on; that binding is re-verified by hash before any
sample is taken.

## Claims and non-claims

The longitudinal map is the only part that is verified (on the held-out side).
The transverse and angular placement is the same construction as before and is
recorded as descriptive geometry screens, not as gates. LSSM does not separate
dorsal from ventral roots, so every line is a root complex and the
dorsal/ventral identity is left unresolved. S1 and S2 exit through fused
sacral foramina, below the disc-anchored fit range, so their longitudinal
placement is an extrapolation and is tagged as such. The field is a provisional
V/A transfer field, so activating functions are relative transfer metrics, not
motor recruitment thresholds.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import sys
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import numpy.typing as npt
import scipy
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).parent))

from align_roots_by_level import DISC_NAMES, fit_map, level_pairs  # noqa: E402
from place_lumbosacral_roots import (  # noqa: E402
    BONE_LABELS,
    CSF_LABEL,
    DEFAULT_MARKERS,
    DEFAULT_MESH,
    POPULATION_ANGLE_DEG,
    AlvarCanal,
    LssmSubject,
    digest_file,
    load_alvar,
    load_subject,
    place,
)
from sweep_root_roll import (  # noqa: E402
    BIN_WIDTHS_M,
    activating_function,
    bin_potential,
    resample_by_arclength,
)

DEFAULT_FIELD = Path(
    '<local-path-redacted>'
)

# The montage family and bin widths are reused from the roll sweep: four
# coefficient directions over the two lead-field bases, and 2/5/10 mm bins.
MONTAGE_ANGLES_DEG = (0.0, 45.0, 90.0, 135.0)
SAMPLE_SPACING_M = 0.0005
MIN_RETAINED_POINTS = 5
DEGENERATE_RADIAL_DISTANCE_M = 1e-9
CENTROID_CHUNK = 500_000
# S1 and S2 exit through fused sacral foramina, which carry no disc, so their
# heights are an extrapolation of the L1-L5-anchored regression line.
EXTRAPOLATED_SEGMENTS = ("S1", "S2")


def is_extrapolated(segment: str) -> bool:
    """Whether one segment sits below the disc-anchored fit range."""
    return segment in EXTRAPOLATED_SEGMENTS


def montage_coefficients(angle_deg: float) -> npt.NDArray[np.float64]:
    """Two-basis montage coefficients at one angle, as in the roll sweep."""
    angle = np.radians(angle_deg)
    return np.array([np.cos(angle), np.sin(angle)], dtype=np.float64)


def electrode_currents(
    basis_matrix: npt.NDArray[np.float64], angle_deg: float
) -> npt.NDArray[np.float64]:
    """Electrode currents implied by one montage coefficient direction."""
    return basis_matrix @ montage_coefficients(angle_deg)


def validate_field_mesh_binding(field: Path, mesh: Path) -> dict[str, Any]:
    """Refuse a field with the wrong mesh binding or incompatible array layout."""
    with h5py.File(mesh, "r") as handle:
        node_count = int(handle["mesh/points_xyz_m"].shape[0])
        tetrahedra_shape = tuple(handle["mesh/tetrahedra"].shape)
    with h5py.File(field, "r") as handle:
        declared = str(handle.attrs.get("mesh_artifact_sha256", ""))
        required = ("L_phi", "basis_matrix", "node_coordinates", "mesh/tetrahedra")
        missing = [name for name in required if name not in handle]
        if missing:
            raise ValueError(f"field artifact missing required datasets: {missing}")
        lead_shape = tuple(handle["L_phi"].shape)
        basis_shape = tuple(handle["basis_matrix"].shape)
        coordinate_shape = tuple(handle["node_coordinates"].shape)
        field_tetrahedra_shape = tuple(handle["mesh/tetrahedra"].shape)
        status = str(handle.attrs.get("status", ""))
    if not declared:
        raise ValueError("field artifact declares no mesh_artifact_sha256")
    actual = digest_file(mesh)
    if declared != actual:
        raise ValueError(f"field declares mesh {declared}, on disk it is {actual}")
    if len(lead_shape) != 2 or lead_shape[1] != node_count:
        raise ValueError(f"L_phi must have shape (N_basis, {node_count}), got {lead_shape}")
    if len(basis_shape) != 2 or basis_shape[1] != lead_shape[0]:
        raise ValueError(
            f"basis_matrix second dimension must equal {lead_shape[0]}, got {basis_shape}"
        )
    if coordinate_shape != (node_count, 3):
        raise ValueError(
            f"node_coordinates must have shape ({node_count}, 3), got {coordinate_shape}"
        )
    if field_tetrahedra_shape != tetrahedra_shape:
        raise ValueError(
            "field tetrahedra shape does not match the bound mesh: "
            f"{field_tetrahedra_shape} != {tetrahedra_shape}"
        )
    return {
        "mesh_sha256": actual,
        "declared_mesh_sha256": declared,
        "status": status,
        "lead_field_shape": list(lead_shape),
        "basis_matrix_shape": list(basis_shape),
    }


def local_cell_context(
    points: npt.NDArray[np.float64],
    tetrahedra: npt.NDArray[np.int64],
    source_label: npt.NDArray[np.int64],
    placed_points: npt.NDArray[np.float64],
    margin_m: float = 0.02,
) -> tuple[cKDTree, npt.NDArray[np.int64], npt.NDArray[np.int64]]:
    """A nearest-cell index over the neighbourhood the placed roots occupy.

    Centroids are computed and windowed in bounded chunks so the peak allocation
    stays low even though the mesh holds 8.1 million tetrahedra.
    """
    origin = placed_points.mean(axis=0)
    z_low, z_high = float(placed_points[:, 2].min()), float(placed_points[:, 2].max())
    reach = (
        float(np.sqrt(((placed_points[:, :2] - origin[:2]) ** 2).sum(axis=1)).max())
        + margin_m
    )

    indices: list[npt.NDArray[np.int64]] = []
    centroids: list[npt.NDArray[np.float64]] = []
    count = tetrahedra.shape[0]
    for start in range(0, count, CENTROID_CHUNK):
        stop = min(start + CENTROID_CHUNK, count)
        nodes = tetrahedra[start:stop]
        block = (
            points[nodes[:, 0]]
            + points[nodes[:, 1]]
            + points[nodes[:, 2]]
            + points[nodes[:, 3]]
        ) * 0.25
        in_z = (block[:, 2] >= z_low - margin_m) & (block[:, 2] <= z_high + margin_m)
        in_radial = np.sqrt(((block[:, :2] - origin[:2]) ** 2).sum(axis=1)) <= reach
        local = np.flatnonzero(in_z & in_radial)
        if local.size:
            indices.append(local + start)
            centroids.append(block[local])
    local_indices = np.concatenate(indices) if indices else np.empty(0, dtype=np.int64)
    if local_indices.size < 1000:
        raise ValueError("neighbourhood around the placed roots holds too few cells")
    local_centroids = np.concatenate(centroids)
    tree = cKDTree(local_centroids)
    return tree, source_label[local_indices], tetrahedra[local_indices]


def apply_local_direction_displacement(
    samples: npt.NDArray[np.float64],
    alvar: AlvarCanal,
    displacement_mm: float,
    angle_deg: float,
) -> tuple[npt.NDArray[np.float64], int]:
    """Displace each point in its own radial-tangential canal frame."""
    moved = np.array(samples, dtype=np.float64, copy=True)
    angle = math.radians(angle_deg)
    cosine, sine = math.cos(angle), math.sin(angle)
    degenerate = 0
    for index, point in enumerate(samples):
        sac_index = int(np.argmin(np.abs(alvar.heights - point[2])))
        radial = point[:2] - alvar.centres[sac_index]
        norm = float(np.linalg.norm(radial))
        if norm < DEGENERATE_RADIAL_DISTANCE_M:
            degenerate += 1
            continue
        radial /= norm
        tangent = np.array([-radial[1], radial[0]], dtype=np.float64)
        moved[index, :2] += displacement_mm * 1e-3 * (
            cosine * radial + sine * tangent
        )
    return moved, degenerate


def geometry_screen(
    key: tuple[str, str],
    samples: npt.NDArray[np.float64],
    labels: npt.NDArray[np.int64],
    alvar: AlvarCanal,
) -> dict[str, Any]:
    """Descriptive self-consistency checks in the population-angle convention."""
    segment, side = key
    offsets = []
    for point in samples:
        index = int(np.argmin(np.abs(alvar.heights - point[2])))
        offsets.append(point[:2] - alvar.centres[index])
    offset = np.median(np.asarray(offsets), axis=0)
    # Alvar dorsal is -y and its +x is to the subject's right, so a left-side
    # root has negative x; recover the population convention of zero at the
    # dorsal midline with positive toward the left.
    angle = float(np.degrees(np.arctan2(-offset[0], -offset[1])))
    expected, spread = POPULATION_ANGLE_DEG[(segment, side)]
    return {
        "in_csf_fraction": float((labels == CSF_LABEL).mean()),
        "in_bone_fraction": float(np.isin(labels, BONE_LABELS).mean()),
        "dorsal_offset_mm": float(-offset[1] * 1000.0),
        "is_dorsal": bool(-offset[1] > 0.0),
        "angle_deg": angle,
        "population_angle_deg": expected,
        "population_sd_deg": spread,
        "angle_residual_deg": float(angle - expected),
        "within_population": bool(abs(angle - expected) <= 2.0 * spread),
    }


def _magnitude_stats(values: npt.NDArray[np.float64], unit: str) -> dict[str, float]:
    """Percentiles of the absolute values, tagged with explicit transfer units."""
    magnitude = np.abs(values)
    return {
        f"p05_{unit}": float(np.percentile(magnitude, 5)),
        f"p25_{unit}": float(np.percentile(magnitude, 25)),
        f"median_{unit}": float(np.median(magnitude)),
        f"p75_{unit}": float(np.percentile(magnitude, 75)),
        f"p95_{unit}": float(np.percentile(magnitude, 95)),
        f"max_{unit}": float(magnitude.max()),
    }


def field_metrics_for_root(
    samples: npt.NDArray[np.float64],
    nearest: npt.NDArray[np.int64],
    montage_potentials: Sequence[tuple[float, npt.NDArray[np.float64]]],
    bin_widths: Sequence[float],
) -> list[dict[str, Any]]:
    """Cell-node-mean potential sampled along one root, binned per width."""
    montages = []
    for angle, cell_potential in montage_potentials:
        sampled = cell_potential[nearest]
        by_width: dict[str, Any] = {}
        for width in bin_widths:
            binned, _ = bin_potential(samples, sampled, width)
            if binned.size < 5:
                by_width[f"{width:g}"] = None
                continue
            af = activating_function(binned, width)
            gradient = np.diff(binned) / width
            by_width[f"{width:g}"] = {
                "bins": int(binned.size),
                "peak_abs_af_v_per_a_m2": _magnitude_stats(af, "v_per_a_m2"),
                "peak_abs_longitudinal_e_v_per_a_m": _magnitude_stats(
                    gradient, "v_per_a_m"
                ),
            }
        montages.append({"coefficient_angle_deg": angle, "by_bin_width_m": by_width})
    return montages


def _field_metrics_finite(per_subject: Sequence[dict[str, Any]]) -> bool:
    """Every recorded field statistic is finite, and at least one was recorded."""
    recorded = 0
    for entry in per_subject:
        for root in entry.get("placed_roots", []):
            for montage in root.get("field", []):
                for block in montage["by_bin_width_m"].values():
                    if block is None:
                        continue
                    for stats in (
                        block["peak_abs_af_v_per_a_m2"],
                        block["peak_abs_longitudinal_e_v_per_a_m"],
                    ):
                        recorded += 1
                        if not all(np.isfinite(value) for value in stats.values()):
                            return False
    return recorded > 0


def _subject_has_sampled_field(entry: dict[str, Any]) -> bool:
    """Whether one subject has at least one real, non-empty field metric block."""
    for root in entry.get("placed_roots", []):
        for montage in root.get("field", []):
            if any(block is not None for block in montage["by_bin_width_m"].values()):
                return True
    return False


def _vertebra_name(height_m: float, alvar: AlvarCanal) -> str:
    """Name the vertebra a height falls in, between two named discs."""
    ordered = sorted(alvar.disc_heights, reverse=True)
    above = [name for name, z in zip(DISC_NAMES, ordered, strict=True) if z > height_m]
    below = [name for name, z in zip(DISC_NAMES, ordered, strict=True) if z <= height_m]
    if above and below:
        return above[-1].split("-")[1]
    return "above" if below else "below"
def place_root(
    subject: LssmSubject,
    alvar: AlvarCanal,
    line_mm: npt.NDArray[np.float64],
    height_map: Callable[[float], float],
) -> tuple[npt.NDArray[np.float64] | None, int]:
    """Place one root, counting points that fall outside Alvar's sac samples.

    S1 and S2 routinely descend below the lowest thecal sac slice, where
    Alvar's sac label stops and `sac_at` returns None; those points are counted
    as dropped rather than silently vanishing.
    """
    dropped = 0
    for point in line_mm:
        if alvar.sac_at(height_map(float(point[2]))) is None:
            dropped += 1
    placed = place(subject, alvar, line_mm, height_map=height_map)
    return placed, dropped


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--markers", type=Path, default=DEFAULT_MARKERS)
    parser.add_argument("--mesh", type=Path, default=DEFAULT_MESH)
    parser.add_argument("--field", type=Path, default=DEFAULT_FIELD)
    parser.add_argument("--fit-side", choices=("L", "R"), default="L")
    parser.add_argument("--sample-spacing-m", type=float, default=SAMPLE_SPACING_M)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args(argv)

    for path in (arguments.mesh, arguments.field):
        if not path.is_file():
            raise FileNotFoundError(f"required input missing: {path}")
    if not arguments.markers.is_dir():
        raise FileNotFoundError(f"markers directory missing: {arguments.markers}")

    field_binding = validate_field_mesh_binding(arguments.field, arguments.mesh)
    mesh_sha = field_binding["mesh_sha256"]

    alvar = load_alvar(arguments.mesh)
    if alvar.disc_heights.size != len(DISC_NAMES):
        raise ValueError(
            f"expected {len(DISC_NAMES)} discs, detected {alvar.disc_heights.size}; "
            "the level naming would be wrong"
        )
    test_side = "R" if arguments.fit_side == "L" else "L"

    with h5py.File(arguments.mesh, "r") as handle:
        points = np.asarray(handle["mesh/points_xyz_m"][...], dtype=np.float64)
        tetrahedra = np.asarray(handle["mesh/tetrahedra"][...], dtype=np.int64)
        source_label = np.asarray(handle["mesh/source_label"][...], dtype=np.int64)
    with h5py.File(arguments.field, "r") as handle:
        lead_field = np.asarray(handle["L_phi"][...], dtype=np.float64)
        basis_matrix = np.asarray(handle["basis_matrix"][...], dtype=np.float64)

    subjects = sorted(p for p in arguments.markers.iterdir() if p.is_dir())
    spacing = float(np.median(np.diff(sorted(alvar.disc_heights))) * 1000.0)
    chance = spacing / 4.0

    per_subject: list[dict[str, Any]] = []
    placed_lines: list[tuple[str, tuple[str, str], npt.NDArray[np.float64]]] = []
    for subject_path in subjects:
        subject = load_subject(subject_path)
        fit_rows = level_pairs(subject, alvar, arguments.fit_side)
        test_rows = level_pairs(subject, alvar, test_side)
        entry: dict[str, Any] = {
            "subject": subject_path.name,
            "fit_side": arguments.fit_side,
            "test_side": test_side,
        }
        if len(fit_rows) < 3 or len(test_rows) < 3:
            entry["processed"] = False
            entry["skip_reason"] = "too few levels to fit a longitudinal map"
            per_subject.append(entry)
            continue

        mapping = fit_map(
            [(row[1], row[2]) for row in fit_rows], tuple(row[0] for row in fit_rows)
        )
        fit_residual = [
            abs(mapping.apply(row[1]) - row[2]) * 1000.0 for row in fit_rows
        ]
        test_residual = [
            abs(mapping.apply(row[1]) - row[2]) * 1000.0 for row in test_rows
        ]
        conus = mapping.apply(subject.conus_z)
        entry.update(
            {
                "processed": True,
                "fit_levels": [row[0] for row in fit_rows],
                "test_levels": [row[0] for row in test_rows],
                "slope": mapping.slope,
                "intercept_m": mapping.intercept,
                "fit_rms_mm": float(np.sqrt(np.mean(np.square(fit_residual)))),
                "test_median_mm": float(np.median(test_residual)),
                "test_max_mm": float(np.max(test_residual)),
                "conus_height_m": conus,
                "conus_vertebra": _vertebra_name(conus, alvar),
            }
        )

        placed_roots: list[dict[str, Any]] = []
        dropped_roots: list[dict[str, Any]] = []
        for key, line in subject.roots.items():
            placed, dropped = place_root(subject, alvar, line, mapping.apply)
            if placed is None or placed.shape[0] < MIN_RETAINED_POINTS:
                dropped_roots.append(
                    {
                        "root": f"{key[0]}_{key[1]}",
                        "root_class": "root_complex",
                        "retained_samples": (
                            0 if placed is None else int(placed.shape[0])
                        ),
                        "reason": (
                            "fewer than five points inside Alvar's thecal sac "
                            "after longitudinal mapping"
                        ),
                    }
                )
                continue
            placed_roots.append(
                {
                    "root": f"{key[0]}_{key[1]}",
                    "root_class": "root_complex",
                    "segment": key[0],
                    "side": key[1],
                    "source_points": int(line.shape[0]),
                    "retained_samples": int(placed.shape[0]),
                    "dropped_points": dropped,
                    "longitudinal_extrapolation": is_extrapolated(key[0]),
                }
            )
            placed_lines.append((subject_path.name, key, placed))
        entry["placed_roots"] = placed_roots
        entry["dropped_roots"] = dropped_roots
        per_subject.append(entry)

    if not placed_lines:
        raise ValueError("no subject produced a placed root to sample")

    stacked = np.vstack([line for _, _, line in placed_lines])
    tree, local_label, cell_nodes = local_cell_context(
        points, tetrahedra, source_label, stacked
    )

    montage_potentials = []
    for angle in MONTAGE_ANGLES_DEG:
        nodal = montage_coefficients(angle) @ lead_field
        montage_potentials.append((angle, nodal[cell_nodes].mean(axis=1)))

    by_name = {entry["subject"]: entry for entry in per_subject}
    for subject_name, key, line in placed_lines:
        entry = by_name[subject_name]
        record = next(
            r for r in entry["placed_roots"] if r["root"] == f"{key[0]}_{key[1]}"
        )
        samples = resample_by_arclength(line, arguments.sample_spacing_m)
        _, nearest = tree.query(samples, k=1)
        record["geometry"] = geometry_screen(key, samples, local_label[nearest], alvar)
        record["field"] = field_metrics_for_root(
            samples, nearest, montage_potentials, BIN_WIDTHS_M
        )

    processed = [entry for entry in per_subject if entry.get("processed")]
    test_medians = np.asarray([entry["test_median_mm"] for entry in processed])
    test_maxes = np.asarray([entry["test_max_mm"] for entry in processed])
    slopes = [entry["slope"] for entry in processed]

    verdict = {
        "all_subjects_processed": bool(
            len(processed) == len(subjects) and len(subjects) > 0
        ),
        "held_out_side_beats_chance": bool(
            np.median(test_medians) < chance if test_medians.size else False
        ),
        "no_subject_off_by_half_a_level": bool(
            test_maxes.max() < spacing / 2.0 if test_maxes.size else False
        ),
        "every_subject_beats_chance": bool(
            test_medians.max() < chance if test_medians.size else False
        ),
        "slopes_near_unity": bool(all(0.85 <= slope <= 1.15 for slope in slopes)),
        "mesh_field_binding_valid": True,
        "every_subject_has_sampled_fields": bool(
            processed and all(_subject_has_sampled_field(entry) for entry in processed)
        ),
        "field_metrics_finite": _field_metrics_finite(per_subject),
    }

    geometry_roots = []
    for entry in per_subject:
        for root in entry.get("placed_roots", []):
            geometry = root.get("geometry")
            if geometry is not None:
                geometry_roots.append(geometry)
    if geometry_roots:
        geometry_screens = {
            "descriptive_only": True,
            "note": (
                "self-consistent construction screens (CSF/bone fractions, "
                "dorsal sign, population angles), recorded for transparency "
                "and not part of `passed`"
            ),
            "summary": {
                "every_root_mostly_in_csf": bool(
                    min(row["in_csf_fraction"] for row in geometry_roots) >= 0.5
                ),
                "every_root_avoids_bone": bool(
                    max(row["in_bone_fraction"] for row in geometry_roots) <= 0.1
                ),
                "all_roots_dorsal": bool(
                    all(row["is_dorsal"] for row in geometry_roots)
                ),
                "angles_within_population": bool(
                    sum(row["within_population"] for row in geometry_roots)
                    >= 0.8 * len(geometry_roots)
                ),
            },
        }
    else:
        geometry_screens = {
            "descriptive_only": True,
            "note": "no placed roots survived to screen",
            "summary": None,
        }

    receipt = {
        "schema_version": 1,
        "generated_at": datetime.now(UTC).isoformat(),
        "status": "LEVEL_ANCHORED_LUMBOSACRAL_PLACEMENT_WITH_FIELDS",
        "provenance": {
            "markers": str(arguments.markers),
            "mesh": str(arguments.mesh.resolve()),
            "mesh_sha256": mesh_sha,
            "field": str(arguments.field.resolve()),
            "field_binding": field_binding,
        },
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
        },
        "units": {
            "coordinates": "m",
            "field_transfer_potential": "V/A",
            "longitudinal_e_field": "V/(A*m)",
            "activating_function": "V/(A*m^2)",
        },
        "method": {
            "longitudinal": (
                "root exit heights onto their own foramen levels, fitted on one "
                "side and tested on the held-out side"
            ),
            "transverse_angular": (
                "existing construction from place_lumbosacral_roots, recorded as "
                "descriptive geometry screens only"
            ),
            "sample_spacing_m": arguments.sample_spacing_m,
            "bin_widths_m": list(BIN_WIDTHS_M),
            "montage_angles_deg": list(MONTAGE_ANGLES_DEG),
            "montage_electrode_currents_a": [
                electrode_currents(basis_matrix, angle).tolist()
                for angle in MONTAGE_ANGLES_DEG
            ],
            "min_retained_points": MIN_RETAINED_POINTS,
            "sampling": (
                "nearest cell centroid within a local neighbourhood; cell "
                "potential is the mean of its four nodes; the local index is "
                "built in bounded chunks"
            ),
        },
        "alignment": {
            "disc_spacing_mm": spacing,
            "chance_mm": chance,
            "per_subject": [
                {
                    "subject": entry["subject"],
                    "processed": entry.get("processed"),
                    "slope": entry.get("slope"),
                    "fit_rms_mm": entry.get("fit_rms_mm"),
                    "test_median_mm": entry.get("test_median_mm"),
                    "test_max_mm": entry.get("test_max_mm"),
                    "conus_vertebra": entry.get("conus_vertebra"),
                }
                for entry in per_subject
            ],
        },
        "geometry_screens": geometry_screens,
        "per_subject": per_subject,
        "checks": verdict,
        "passed": all(verdict.values()),
        "sets_no_gates": True,
        "dorsal_ventral_identity": "unresolved",
        "claim_scope": (
            "Geometric placement and direct FEM lead-field sampling along LSSM "
            "root complexes. The longitudinal map is level-anchored and "
            "validated on held-out roots; the transverse/angular placement is "
            "descriptive; dorsal/ventral identity is unresolved (root_complex); "
            "S1/S2 are extrapolated below the disc-anchored fit; the field is a "
            "provisional V/A transfer field, so no motor recruitment claims are "
            "made."
        ),
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"\nwrote {arguments.output}")
    return 0 if receipt["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
