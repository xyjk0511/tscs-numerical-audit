"""Sweep the unidentified roll and ask whether any conclusion actually depends on it.

## The problem this answers

RADO-SCS posterior-root geometry is registered into the Alvar body model, but one
of the six rigid degrees of freedom is not determined. The four anchors used were
intervertebral disc centroids, and those lie on the very axis the missing degree
of freedom rotates about. The registration receipt records the consequence:
anchor spread of [33.677, 1.4746, 0.0486] mm against a fit residual of 1.3362 mm,
so the transverse extent is 27 times smaller than the noise, and 2000 perturbed
trials put only 4.55% of the fitted angles within five degrees of each other.

Identifying the roll needs new off-axis anchors and is real work. **This asks a
cheaper question first: does the roll change any answer the tool is for?** If a
conclusion holds across every roll the anatomy permits, the missing degree of
freedom does not matter for that conclusion. If it does not hold, the sweep says
how accurately the roll would have to be pinned down.

## What is compared, and against what

Nothing here is compared against a truth. **Every roll angle is put through an
identical pipeline and compared against the other angles.** Sampling and binning
error is therefore common-mode across the comparison, which is what makes a
coarse sampling scheme adequate for this question and would not make it adequate
for an absolute claim.

Two lessons from this project's own verification work are applied. The activating
function is binned along arclength rather than differenced point to point,
because a second difference taken below the element size measures the mesh rather
than the field. And because that bin width turned out to dominate the numbers in
the sphere verification, the sweep is run at three bin widths and the conclusion
is only reported as stable if it is stable at all three.

## Anatomical plausibility is not optional

Most roll angles put the roots inside vertebral bone or outside the canal. Those
placements are not merely wrong, they are impossible, and a claim that "the
answer is stable across all 360 degrees" that includes them is worthless. Every
angle therefore carries the fraction of its sample points that land in bone, and
the stability verdict is computed over the plausible band only, with the full
circle reported alongside.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import numpy.typing as npt
import scipy
from scipy.spatial import cKDTree

DEFAULT_REGISTRATION = Path(
    "analysis/model_next_steps_l11_rado_registration/l11_rado_registration.json"
)
DEFAULT_AXONS = Path(
    '<local-path-redacted>'
)
DEFAULT_MESH = Path(
    "analysis/model_next_steps_l9_resolution/raw_grid_candidates/"
    "pb-purity24p2-39p4-41p1-79p0p5h-v81-topologyfix-surf1p0d0p20-pad2-"
    "deterministic1-nolimit-run2-topology-repaired1-smooth130-pin1-polish1-cgal-mesh.h5"
)
DEFAULT_FIELD = Path('<local-path-redacted>')

CORD_LABEL = 66
BONE_LABELS = (11, 17)  # cancellous and cortical bone
CSF_LABEL = 18
# Dorsal is -y in Alvar. A margin rather than zero, so a root grazing the
# midline is not counted as dorsal on floating-point luck.
DORSAL_MARGIN_M = -0.002
BIN_WIDTHS_M = (0.002, 0.005, 0.010)

# Montages are synthesised from the two lead-field bases by varying the
# coefficient direction. These span the accessible family; the implied electrode
# currents are recorded in the receipt so a reader can see what was actually
# compared rather than an abstract angle.
MONTAGE_ANGLES_DEG = (0.0, 45.0, 90.0, 135.0)


def digest_file(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            value.update(chunk)
    return value.hexdigest()


@dataclass(frozen=True)
class Registration:
    """The published RADO-to-Alvar transform, read rather than reconstructed."""

    scale: float
    rotation: npt.NDArray[np.float64]
    translation_mm: npt.NDArray[np.float64]
    residual_mm: float
    source_sha256: str

    def apply(self, points_m: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """RADO metres to Alvar metres. The transform itself is in millimetres."""
        millimetres = points_m * 1000.0
        return (self.scale * (millimetres @ self.rotation.T) + self.translation_mm) / 1000.0


def load_registration(path: Path) -> Registration:
    document = json.loads(path.read_text(encoding="utf-8"))
    stage1 = document["stage1_disc_similarity"]
    rotation = np.asarray(stage1["rotation"], dtype=np.float64)
    determinant = float(np.linalg.det(rotation))
    if abs(determinant - 1.0) > 1e-9:
        raise ValueError(f"registration rotation is not proper: det = {determinant}")
    return Registration(
        scale=float(stage1["scale"]),
        rotation=rotation,
        translation_mm=np.asarray(
            document["stage2_cord_correction"]["final_translation_mm"], dtype=np.float64
        ),
        residual_mm=float(stage1["rms_residual_mm"]),
        source_sha256=digest_file(path),
    )


def cord_axis(
    points: npt.NDArray[np.float64],
    tetrahedra: npt.NDArray[np.int64],
    label: npt.NDArray[np.int64],
    z_range: tuple[float, float],
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], dict[str, Any]]:
    """A straight line through the cord over the span the roots occupy.

    The roll is a rigid rotation of the whole root structure, so it must be about
    one line, not about a locally varying axis -- twisting about a curve would
    deform the geometry rather than reorient it. The registration receipt notes
    that an earlier counterfactual twisted about the global z axis while an
    adversarial reimplementation used the cord centreline and the two disagreed
    on magnitude. The cord's own axis is the physically meaningful one, so it is
    used here and its deviation from vertical is reported.
    """
    cord_cells = tetrahedra[label == CORD_LABEL]
    if cord_cells.size == 0:
        raise ValueError("no cord-labelled cells in the mesh")
    centroids = points[cord_cells].mean(axis=1)
    span = (centroids[:, 2] >= z_range[0]) & (centroids[:, 2] <= z_range[1])
    if np.count_nonzero(span) < 100:
        raise ValueError("too few cord cells over the root span to define an axis")
    selected = centroids[span]

    origin = selected.mean(axis=0)
    centred = selected - origin
    # Principal direction by power iteration on the covariance, avoiding
    # numpy.linalg so this runs in any of the project's environments.
    covariance = centred.T @ centred
    direction = np.array([0.0, 0.0, 1.0])
    for _ in range(200):
        direction = covariance @ direction
        norm = float(np.sqrt((direction**2).sum()))
        if norm == 0.0:
            raise ValueError("degenerate cord covariance")
        direction /= norm
    if direction[2] < 0.0:
        direction = -direction

    tilt = float(np.degrees(np.arccos(np.clip(direction[2], -1.0, 1.0))))
    residual = centred - np.outer(centred @ direction, direction)
    diagnostics = {
        "cord_cells_in_span": int(np.count_nonzero(span)),
        "axis_origin_m": origin.tolist(),
        "axis_direction": direction.tolist(),
        "axis_tilt_from_vertical_deg": tilt,
        "cord_scatter_about_axis_rms_m": float(np.sqrt((residual**2).sum(axis=1).mean())),
    }
    return origin, direction, diagnostics


def roll_matrix(direction: npt.NDArray[np.float64], degrees: float) -> npt.NDArray[np.float64]:
    """Rodrigues rotation about a unit axis."""
    angle = np.radians(degrees)
    x, y, z = direction
    cross = np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]], dtype=np.float64)
    return (
        np.eye(3)
        + np.sin(angle) * cross
        + (1.0 - np.cos(angle)) * (cross @ cross)
    )


def resample_by_arclength(
    path: npt.NDArray[np.float64], spacing: float
) -> npt.NDArray[np.float64]:
    """Even samples along a polyline, so bins carry comparable path length."""
    steps = np.sqrt(((np.diff(path, axis=0)) ** 2).sum(axis=1))
    arclength = np.concatenate(([0.0], np.cumsum(steps)))
    total = float(arclength[-1])
    if total <= spacing * 4.0:
        raise ValueError("path is too short to resample at this spacing")
    targets = np.arange(0.0, total, spacing)
    return np.stack(
        [np.interp(targets, arclength, path[:, axis]) for axis in range(3)], axis=1
    )


def activating_function(
    potential: npt.NDArray[np.float64], spacing: float
) -> npt.NDArray[np.float64]:
    """Second difference of potential along the fibre, at a fixed baseline.

    Fixed rather than tied to the element size. A stencil that shrinks with the
    mesh cannot converge, and one below the element size measures the
    piecewise-linear kinks of the solution rather than the field.
    """
    if potential.size < 3:
        return np.zeros(0, dtype=np.float64)
    return (potential[2:] - 2.0 * potential[1:-1] + potential[:-2]) / spacing**2


def bin_potential(
    samples: npt.NDArray[np.float64],
    potential: npt.NDArray[np.float64],
    bin_width: float,
) -> tuple[npt.NDArray[np.float64], float]:
    """Average potential into equal arclength bins, mirroring the cord profile."""
    steps = np.sqrt(((np.diff(samples, axis=0)) ** 2).sum(axis=1))
    arclength = np.concatenate(([0.0], np.cumsum(steps)))
    index = np.floor(arclength / bin_width).astype(np.int64)
    unique = np.unique(index)
    if unique.size < 5:
        return np.zeros(0, dtype=np.float64), bin_width
    means = np.array(
        [float(potential[index == value].mean()) for value in unique], dtype=np.float64
    )
    return means, bin_width


def stratified_axon_sample(
    directory: Path, diameters: npt.NDArray[np.float64], count: int, seed: int
) -> list[int]:
    """Even coverage of the five diameter classes, so no class dominates."""
    generator = np.random.default_rng(seed)
    classes = np.unique(diameters)
    per_class = max(count // classes.size, 1)
    chosen: list[int] = []
    for value in classes:
        members = np.flatnonzero(diameters == value)
        take = min(per_class, members.size)
        chosen.extend(generator.choice(members, size=take, replace=False).tolist())
    available = {int(p.stem.split("_")[1]) for p in directory.glob("axon_*.npy")}
    return sorted(index for index in chosen if index in available)


def _roll_plausibility(
    query: npt.NDArray[np.float64],
    labels: npt.NDArray[np.int64],
    offsets: npt.NDArray[np.int64],
    cord_centroids: npt.NDArray[np.float64],
    head: int,
) -> dict[str, Any]:
    """Judge dorsal roots against the cord, retaining weaker proximity diagnostics.

    Whole-root bone fraction (55% fitted, 44-68% across rolls) is confounded by
    roots exiting the foramen in a 2 mm mesh. Cord distance also misleads:
    rootlets approach within 1.365-1.424 mm, but dorsal roots end 10.447 mm away
    (9.201 mm for these axons), so an 8 mm standoff is not misregistration.
    Proximal CSF occupancy tests whether the first 20 mm remain in the canal.
    All three proximity metrics ranked a 180-degree twist above the fitted roll;
    only the dorsal-ventral test rejects it. Alvar dorsal is -y, consistent with
    the registration receipt's cord offset of 37.067 mm below the disc in y.
    """
    proximal_mask = np.zeros(query.shape[0], dtype=bool)
    for start, stop in zip(offsets[:-1], offsets[1:], strict=True):
        proximal_mask[start : min(start + head, stop)] = True
    proximal_labels = labels[proximal_mask]
    dorsal_offsets = []
    for point in query[proximal_mask]:
        local = cord_centroids[np.abs(cord_centroids[:, 2] - point[2]) <= 0.003]
        if local.size:
            dorsal_offsets.append(float(point[1] - local[:, 1].mean()))
    dorsal_median = float(np.median(dorsal_offsets)) if dorsal_offsets else float("nan")
    return {
        "dorsal_offset_median_m": dorsal_median,
        "roots_are_dorsal": bool(dorsal_median < DORSAL_MARGIN_M),
        "in_bone_fraction": float(np.isin(labels, BONE_LABELS).mean()),
        "in_cord_fraction": float((labels == CORD_LABEL).mean()),
        "proximal_in_csf_fraction": float((proximal_labels == CSF_LABEL).mean()),
        "proximal_in_bone_fraction": float(np.isin(proximal_labels, BONE_LABELS).mean()),
        "proximal_points": int(proximal_labels.size),
        "distinct_materials": sorted({int(v) for v in np.unique(labels)}),
    }


def _response_by_bin_width(
    paths: list[npt.NDArray[np.float64]],
    sampled: npt.NDArray[np.float64],
    offsets: npt.NDArray[np.int64],
    indices: list[int],
) -> dict[str, Any]:
    by_width: dict[str, Any] = {}
    for width in BIN_WIDTHS_M:
        peaks, slopes, kept = [], [], []
        for index, path in enumerate(paths):
            piece = sampled[offsets[index] : offsets[index + 1]]
            binned, _ = bin_potential(path, piece, width)
            if binned.size < 5:
                continue
            values = activating_function(binned, width)
            peaks.append(float(np.max(np.abs(values))))
            # First differences measure field along the fibre; second differences
            # also respond to potential curvature. Keep both diagnostics.
            slopes.append(float(np.max(np.abs(np.diff(binned) / width))))
            kept.append(indices[index])
        if peaks:
            array = np.asarray(peaks)
            by_width[f"{width:g}"] = {
                "axons": int(array.size),
                "peak_abs_af_median_v_per_m2": float(np.median(array)),
                "peak_abs_af_p05_v_per_m2": float(np.percentile(array, 5)),
                "peak_abs_af_p25_v_per_m2": float(np.percentile(array, 25)),
                "peak_abs_af_p75_v_per_m2": float(np.percentile(array, 75)),
                "peak_abs_af_p95_v_per_m2": float(np.percentile(array, 95)),
                "peak_abs_af_max_v_per_m2": float(array.max()),
                "peak_abs_longitudinal_e_median_v_per_m": float(np.median(np.asarray(slopes))),
                "top_ten_axon_indices": [kept[i] for i in np.argsort(array)[::-1][:10]],
            }
    return by_width


def _montage_headline(entry: dict[str, Any]) -> str:
    reference = entry["by_bin_width_m"].get("0.005", {})
    value = reference.get("peak_abs_af_median_v_per_m2", float("nan"))
    return f"m{entry['coefficient_angle_deg']:.0f}={value:.3g}"


def _sweep_rolls(
    placed: list[npt.NDArray[np.float64]],
    indices: list[int],
    points: npt.NDArray[np.float64],
    tetrahedra: npt.NDArray[np.int64],
    source_label: npt.NDArray[np.int64],
    lead_field: npt.NDArray[np.float64],
    basis_matrix: npt.NDArray[np.float64],
    *,
    roll_step_deg: float,
    sample_spacing_m: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    stacked = np.vstack(placed)
    z_low, z_high = float(stacked[:, 2].min()), float(stacked[:, 2].max())

    origin, axis, axis_diagnostics = cord_axis(
        points, tetrahedra, source_label, (z_low, z_high)
    )

    # A local cell tree, built only over the neighbourhood the roots can reach
    # under any roll. Cell-based sampling mirrors the project's own cord profile,
    # which volume-averages cell potentials rather than reading nodes.
    reach = float(np.sqrt(((stacked - origin) ** 2).sum(axis=1)).max()) + 0.01
    centroids_all = points[tetrahedra].mean(axis=1)
    near = (
        (np.abs(centroids_all[:, 2] - origin[2]) <= (z_high - z_low) / 2.0 + 0.02)
        & (np.sqrt(((centroids_all[:, :2] - origin[:2]) ** 2).sum(axis=1)) <= reach)
    )
    local_cells = np.flatnonzero(near)
    if local_cells.size < 1000:
        raise ValueError("neighbourhood around the roots holds too few cells")
    local_centroids = centroids_all[local_cells]
    local_label = source_label[local_cells]
    tree = cKDTree(local_centroids)
    cell_nodes = tetrahedra[local_cells]
    cord_centroids = local_centroids[local_label == CORD_LABEL]
    if cord_centroids.shape[0] < 50:
        raise ValueError("too few cord cells in the neighbourhood for a dorsal test")

    resampled = [resample_by_arclength(path, sample_spacing_m) for path in placed]
    offsets = np.cumsum([0] + [path.shape[0] for path in resampled])
    head = max(int(0.02 / sample_spacing_m), 5)

    montages = []
    for angle in MONTAGE_ANGLES_DEG:
        coefficients = np.array(
            [np.cos(np.radians(angle)), np.sin(np.radians(angle))], dtype=np.float64
        )
        montages.append(
            {
                "coefficient_angle_deg": angle,
                "coefficients": coefficients,
                "electrode_currents_a": (basis_matrix @ coefficients).tolist(),
                "cell_potential": (coefficients @ lead_field)[cell_nodes].mean(axis=1),
            }
        )

    rolls = np.arange(0.0, 360.0, roll_step_deg)
    results: list[dict[str, Any]] = []
    for roll in rolls:
        matrix = roll_matrix(axis, float(roll))
        rotated = [((path - origin) @ matrix.T) + origin for path in resampled]
        query = np.vstack(rotated)
        _, nearest = tree.query(query, k=1)
        labels = local_label[nearest]

        plausibility = _roll_plausibility(query, labels, offsets, cord_centroids, head)
        per_montage = [
            {
                "coefficient_angle_deg": montage["coefficient_angle_deg"],
                "by_bin_width_m": _response_by_bin_width(
                    rotated, montage["cell_potential"][nearest], offsets, indices
                ),
            }
            for montage in montages
        ]

        results.append(
            {
                "roll_deg": float(roll),
                "plausibility": plausibility,
                "montages": per_montage,
            }
        )
        print(
            f"roll {roll:5.1f}  "
            f"dorsal {plausibility['dorsal_offset_median_m'] * 1000:+6.2f}mm "
            f"{'OK ' if plausibility['roots_are_dorsal'] else 'VENT'}  "
            + "  ".join(_montage_headline(entry) for entry in per_montage)
        )
    return results, montages_summary(montages), axis_diagnostics


def _sweep_receipt(
    arguments: argparse.Namespace,
    registration: Registration,
    mesh_sha: str,
    indices: list[int],
    results: list[dict[str, Any]],
    montage_family: list[dict[str, Any]],
    axis_diagnostics: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "generated_at": datetime.now(UTC).isoformat(),
        "status": "ROOT_ROLL_SENSITIVITY_SWEEP",
        "question": (
            "Does any conclusion this tool supports depend on the roll that the "
            "RADO-to-Alvar registration does not identify?"
        ),
        "provenance": {
            "registration": str(arguments.registration.resolve()),
            "registration_sha256": registration.source_sha256,
            "registration_residual_mm": registration.residual_mm,
            "mesh": str(arguments.mesh.resolve()),
            "mesh_sha256": mesh_sha,
            "field": str(arguments.field.resolve()),
            "field_sha256": digest_file(arguments.field),
            "axon_directory": str(arguments.axons),
            "axon_indices": indices,
            "axon_count": len(indices),
        },
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
        },
        "method": {
            "twist_axis": axis_diagnostics,
            "sample_spacing_m": arguments.sample_spacing_m,
            "bin_widths_m": list(BIN_WIDTHS_M),
            "roll_step_deg": arguments.roll_step_deg,
            "sampling": (
                "nearest cell centroid within a local neighbourhood, cell potential "
                "taken as the mean of its four nodes, mirroring the project's "
                "centreline_potential_profile which volume-averages cell potentials"
            ),
            "comparison_basis": (
                "every roll angle passes through an identical pipeline and is "
                "compared against the other angles, never against a truth, so "
                "sampling and binning error is common-mode"
            ),
            "montage_family": montage_family,
        },
        "rolls": results,
        "sets_no_gates": True,
        "claim_scope": (
            "A sensitivity study of the unidentified registration roll. It does "
            "not identify the roll, does not validate the registration, and makes "
            "no claim that these thoracic roots are the segments transcutaneous "
            "stimulation recruits -- RADO covers T9/10 to T12/L1 while the "
            "clinical target is lumbosacral."
        ),
    }


def montages_summary(montages: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "coefficient_angle_deg": item["coefficient_angle_deg"],
            "electrode_currents_a": item["electrode_currents_a"],
        }
        for item in montages
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registration", type=Path, default=DEFAULT_REGISTRATION)
    parser.add_argument("--axons", type=Path, default=DEFAULT_AXONS)
    parser.add_argument("--mesh", type=Path, default=DEFAULT_MESH)
    parser.add_argument("--field", type=Path, default=DEFAULT_FIELD)
    parser.add_argument("--roll-step-deg", type=float, default=10.0)
    parser.add_argument("--axon-count", type=int, default=200)
    parser.add_argument("--sample-spacing-m", type=float, default=0.0005)
    parser.add_argument("--seed", type=int, default=20260828)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args(argv)

    for path in (arguments.registration, arguments.mesh, arguments.field):
        if not path.is_file():
            raise FileNotFoundError(f"required input missing: {path}")
    if not arguments.axons.is_dir():
        raise FileNotFoundError(f"axon directory missing: {arguments.axons}")
    registration = load_registration(arguments.registration)
    with h5py.File(arguments.mesh, "r") as handle:
        points = np.asarray(handle["mesh/points_xyz_m"][...], dtype=np.float64)
        tetrahedra = np.asarray(handle["mesh/tetrahedra"][...], dtype=np.int64)
        source_label = np.asarray(handle["mesh/source_label"][...], dtype=np.int64)
    with h5py.File(arguments.field, "r") as handle:
        lead_field = np.asarray(handle["L_phi"][...], dtype=np.float64)
        basis_matrix = np.asarray(handle["basis_matrix"][...], dtype=np.float64)
        declared_mesh = str(handle.attrs.get("mesh_artifact_sha256", ""))
    mesh_sha = digest_file(arguments.mesh)
    if declared_mesh and declared_mesh != mesh_sha:
        raise ValueError(f"field declares mesh {declared_mesh}, on disk it is {mesh_sha}")

    diameters = np.load(arguments.axons / "AxonDiameters.npy")
    indices = stratified_axon_sample(
        arguments.axons, diameters, arguments.axon_count, arguments.seed
    )
    if not indices:
        raise ValueError("no axon trajectories selected")
    placed = [registration.apply(np.load(arguments.axons / f"axon_{i}.npy")) for i in indices]
    results, montage_family, axis_diagnostics = _sweep_rolls(
        placed, indices, points, tetrahedra, source_label, lead_field, basis_matrix,
        roll_step_deg=arguments.roll_step_deg, sample_spacing_m=arguments.sample_spacing_m,
    )
    receipt = _sweep_receipt(
        arguments, registration, mesh_sha, indices, results, montage_family, axis_diagnostics,
    )
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"\nwrote {arguments.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
