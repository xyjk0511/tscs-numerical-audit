"""Sample a small geometry ensemble of LSSM root complexes on a lead field.

This is an exploratory transfer calculation.  It deliberately keeps the named
electrode currents in the montage specification as the source of truth: the
generic lead field is first reduced to basis coefficients, then expanded to a
nodal potential before any root is sampled.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import sys
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import numpy.typing as npt
import scipy

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "fem") not in sys.path:
    sys.path.insert(0, str(ROOT / "fem"))

from fem.align_roots_by_level import fit_map, level_pairs  # noqa: E402
from fem.place_lumbosacral_roots import (  # noqa: E402
    DEFAULT_MARKERS,
    DEFAULT_MESH,
    SEGMENTS,
    digest_file,
    load_alvar,
    load_subject,
)
from fem.sample_level_anchored_root_fields import (  # noqa: E402
    MIN_RETAINED_POINTS,
    apply_local_direction_displacement,
    geometry_screen,
    local_cell_context,
    place_root,
)
from fem.sweep_root_roll import (  # noqa: E402
    BIN_WIDTHS_M,
    activating_function,
    bin_potential,
    resample_by_arclength,
)

DEFAULT_PRIOR_CONTRACT = Path(__file__).with_name("root_candidate_prior_contract.json")
SAMPLE_SPACING_M = 0.0005
BIN_WIDTHS_MM = (2.0, 5.0, 10.0)
DISPLACEMENTS_MM = (1.0, 1.5)
PHI_DEG = (45.0, 90.0, 135.0, 180.0, 225.0, 270.0, 315.0)
EXPECTED_SUBJECTS = tuple(f"sub-{index:02d}" for index in range(1, 15))
ROOT_KEYS = tuple((segment, side) for segment in SEGMENTS for side in ("L", "R"))


def _as_path(value: str | Path) -> Path:
    return value if isinstance(value, Path) else Path(value)


def _decode(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.bytes_):
        return bytes(value).decode("utf-8")
    return str(value)


def _finite_vector(values: Any, label: str) -> npt.NDArray[np.float64]:
    try:
        result = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must contain numeric values") from exc
    if result.ndim != 1 or result.size == 0 or not np.all(np.isfinite(result)):
        raise ValueError(f"{label} must be a non-empty finite vector")
    return result


def load_montage_spec(path_or_document: Path | str | Mapping[str, Any]) -> dict[str, Any]:
    """Load a JSON montage specification without copying unrelated fields."""
    if isinstance(path_or_document, Mapping):
        return dict(path_or_document)
    path = _as_path(path_or_document)
    document = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise ValueError("montage specification must be a JSON object")
    return document


def _spec_electrode_ids(spec: Mapping[str, Any]) -> list[str]:
    raw_ids = spec.get("electrode_ids")
    electrodes = spec.get("electrodes")
    if raw_ids is not None:
        if not isinstance(raw_ids, Sequence) or isinstance(raw_ids, str | bytes):
            raise ValueError("montage electrode_ids must be a sequence")
        ids = [str(value) for value in raw_ids]
    elif isinstance(electrodes, list):
        ids = [str(item.get("name")) for item in electrodes if isinstance(item, Mapping)]
    elif isinstance(electrodes, Mapping):
        definitions = electrodes.get("definitions") or electrodes.get("windows")
        if isinstance(definitions, Mapping):
            ids = [str(value) for value in definitions]
        else:
            ids = [str(value) for value in electrodes]
    else:
        ids = []
    if not ids or any(not value or value == "None" for value in ids):
        raise ValueError("montage specification has no electrode names")
    if len(set(ids)) != len(ids):
        raise ValueError("montage specification contains duplicate electrode names")
    return ids


def named_currents_from_spec(
    spec_or_path: Path | str | Mapping[str, Any], field_electrode_ids: Sequence[str]
) -> npt.NDArray[np.float64]:
    """Resolve currents by electrode name and return them in field order (A)."""
    spec = load_montage_spec(spec_or_path)
    field_ids = [str(value) for value in field_electrode_ids]
    if len(set(field_ids)) != len(field_ids) or not field_ids:
        raise ValueError("field electrode_ids must be unique and non-empty")
    spec_ids = _spec_electrode_ids(spec)
    if set(spec_ids) != set(field_ids):
        missing = sorted(set(field_ids) - set(spec_ids))
        extra = sorted(set(spec_ids) - set(field_ids))
        raise ValueError(f"montage/field electrode_ids mismatch; missing={missing}, extra={extra}")

    named: dict[str, float] = {}
    electrodes = spec.get("electrodes")
    if isinstance(electrodes, list):
        for item in electrodes:
            if not isinstance(item, Mapping) or "name" not in item:
                raise ValueError("each montage electrode needs a name")
            name = str(item["name"])
            if "current_A" in item:
                value = item["current_A"]
            elif "current_mA" in item:
                value = float(item["current_mA"]) * 1e-3
            elif "current" in item:
                value = item["current"]
            else:
                # A definitions list may carry a parallel currents_A/currents_mA
                # vector; defer to that branch below.
                continue
            try:
                named[name] = float(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"montage current for {name!r} is not numeric") from exc
    elif isinstance(electrodes, Mapping):
        # Current-only specs commonly carry definitions plus a parallel list.
        values = spec.get("currents_A", spec.get("currents_mA"))
        if values is not None:
            unit = "A" if "currents_A" in spec else "mA"
            if isinstance(values, Mapping):
                named = {str(name): float(value) * (1e-3 if unit == "mA" else 1.0)
                         for name, value in values.items()}
            else:
                vector = _finite_vector(values, f"currents_{unit}")
                if vector.size != len(spec_ids):
                    raise ValueError("montage current vector length does not match electrode_ids")
                scale = 1e-3 if unit == "mA" else 1.0
                named = dict(zip(spec_ids, (vector * scale).tolist(), strict=True))
        else:
            for name, item in electrodes.items():
                if isinstance(item, Mapping):
                    if "current_A" in item:
                        named[str(name)] = float(item["current_A"])
                    elif "current_mA" in item:
                        named[str(name)] = float(item["current_mA"]) * 1e-3
    if not named:
        for key, scale in (("currents_A", 1.0), ("currents_mA", 1e-3), ("montage_current_a", 1.0)):
            values = spec.get(key)
            if values is None:
                continue
            if isinstance(values, Mapping):
                named = {str(name): float(value) * scale for name, value in values.items()}
            else:
                vector = _finite_vector(values, key)
                if vector.size != len(spec_ids):
                    raise ValueError(f"{key} length does not match electrode_ids")
                named = dict(zip(spec_ids, (vector * scale).tolist(), strict=True))
            break
    if set(named) != set(field_ids):
        raise ValueError("montage specification does not provide one current per electrode")
    result = _finite_vector([named[name] for name in field_ids], "named currents")
    return result


def validate_zero_sum(currents_a: Sequence[float], *, atol: float = 1e-12) -> float:
    """Reject non-conserving montages and return the signed residual (A)."""
    values = _finite_vector(currents_a, "electrode currents")
    residual = float(values.sum())
    tolerance = atol + 1e-9 * float(np.abs(values).sum())
    if abs(residual) > tolerance:
        raise ValueError(f"electrode currents must be zero-sum; residual={residual:.6g} A")
    return residual


def _basis_reconstruction(
    L_phi: npt.NDArray[np.float64],
    basis_matrix: npt.NDArray[np.float64],
    electrode_ids: Sequence[str],
    montage_spec: Path | str | Mapping[str, Any],
    *,
    reconstruction_atol: float = 1e-12,
    reconstruction_rtol: float = 1e-9,
) -> dict[str, Any]:
    lead = np.asarray(L_phi, dtype=np.float64)
    basis = np.asarray(basis_matrix, dtype=np.float64)
    if lead.ndim != 2 or basis.ndim != 2:
        raise ValueError("L_phi and basis_matrix must be two-dimensional")
    if basis.shape[0] != len(electrode_ids) or basis.shape[1] != lead.shape[0]:
        raise ValueError(
            "basis_matrix shape must be (number of electrodes, number of L_phi bases)"
        )
    if not np.all(np.isfinite(lead)) or not np.all(np.isfinite(basis)):
        raise ValueError("lead field and basis matrix must be finite")
    currents = named_currents_from_spec(montage_spec, electrode_ids)
    zero_sum = validate_zero_sum(currents)
    coefficients, _, rank, _ = np.linalg.lstsq(basis, currents, rcond=None)
    reconstructed = basis @ coefficients
    error = reconstructed - currents
    max_error = float(np.max(np.abs(error)))
    relative_error = float(
        np.linalg.norm(error) / max(np.linalg.norm(currents), np.finfo(float).eps)
    )
    scale = max(float(np.max(np.abs(currents))), np.finfo(float).eps)
    if max_error > reconstruction_atol + reconstruction_rtol * scale:
        raise ValueError(
            "named currents cannot be reconstructed by basis_matrix; "
            f"max error={max_error:.6g} A"
        )
    drive_current = float(currents[currents > 0.0].sum())
    if not np.isfinite(drive_current) or drive_current <= 0.0:
        raise ValueError("positive drive current must be greater than zero")
    nodal = coefficients @ lead
    normalized_currents = currents / drive_current
    nodal_per_ampere = nodal / drive_current
    return {
        "nodal_potential": np.asarray(nodal, dtype=np.float64),
        "nodal_potential_v_per_a": np.asarray(nodal_per_ampere, dtype=np.float64),
        "electrode_ids": [str(value) for value in electrode_ids],
        "currents_a": currents,
        "normalized_currents_per_a": np.asarray(normalized_currents, dtype=np.float64),
        "drive_current_a": drive_current,
        "named_currents_a": {
            str(name): float(value)
            for name, value in zip(electrode_ids, currents, strict=True)
        },
        "basis_coefficients": np.asarray(coefficients, dtype=np.float64),
        "reconstructed_currents_a": np.asarray(reconstructed, dtype=np.float64),
        "current_sum_a": zero_sum,
        "max_abs_reconstruction_error_a": max_error,
        "relative_reconstruction_l2": relative_error,
        "basis_rank": int(rank),
    }


def reconstruct_named_current_montage(
    L_phi: npt.NDArray[np.float64],
    basis_matrix: npt.NDArray[np.float64],
    electrode_ids: Sequence[str],
    montage_spec: Path | str | Mapping[str, Any],
) -> dict[str, Any]:
    """Public named-current reconstruction returning diagnostics and potential."""
    return _basis_reconstruction(L_phi, basis_matrix, electrode_ids, montage_spec)


def reconstruct_montage_potential(
    L_phi: npt.NDArray[np.float64],
    basis_matrix: npt.NDArray[np.float64],
    electrode_ids: Sequence[str],
    montage_spec: Path | str | Mapping[str, Any],
) -> npt.NDArray[np.float64]:
    """Return only the reconstructed nodal potential (V/A)."""
    return reconstruct_named_current_montage(L_phi, basis_matrix, electrode_ids, montage_spec)[
        "nodal_potential"
    ]


# Short aliases make the operation easy to discover for callers of the analysis script.
reconstruct_named_montage = reconstruct_named_current_montage


def candidate_scenarios() -> list[dict[str, float | str | None]]:
    """Return baseline plus the prescribed local radial/tangential candidates."""
    scenarios: list[dict[str, float | str | None]] = [
        {"scenario_id": "baseline", "displacement_mm": 0.0, "phi_deg": None}
    ]
    for displacement in DISPLACEMENTS_MM:
        for phi in PHI_DEG:
            scenarios.append(
                {
                    "scenario_id": f"d{displacement:g}mm_phi{phi:g}deg",
                    "displacement_mm": float(displacement),
                    "phi_deg": float(phi),
                }
            )
    return scenarios


build_candidate_scenarios = candidate_scenarios


def validate_prior_contract(path_or_document: Path | str | Mapping[str, Any]) -> dict[str, Any]:
    """Fail closed when the frozen candidate-generation contract drifts."""
    if isinstance(path_or_document, Mapping):
        path = None
        document = dict(path_or_document)
    else:
        path = _as_path(path_or_document)
        if not path.is_file():
            raise FileNotFoundError(f"prior contract missing: {path}")
        document = json.loads(path.read_text(encoding="utf-8"))
    generation = document.get("candidate_generation")
    if not isinstance(generation, Mapping):
        raise ValueError("prior contract missing candidate_generation")
    expected: dict[str, Any] = {
        "segments": list(SEGMENTS),
        "anatomical_sides": ["L", "R"],
        "longitudinal_fit_sides": ["L", "R"],
        "sample_spacing_mm": SAMPLE_SPACING_M * 1000.0,
        "displacements_mm": [0.0, *DISPLACEMENTS_MM],
        "phi_deg_for_nonzero_displacements": list(PHI_DEG),
        "scenario_count_per_root": len(candidate_scenarios()),
        "bin_widths_mm": list(BIN_WIDTHS_MM),
    }
    for key, wanted in expected.items():
        actual = generation.get(key)
        if isinstance(wanted, int | float):
            try:
                matches = math.isclose(float(actual), float(wanted), rel_tol=0.0, abs_tol=1e-12)
            except (TypeError, ValueError):
                matches = False
        else:
            try:
                matches = list(actual) == list(wanted)
            except TypeError:
                matches = False
            if matches and (
                key.endswith("_mm") or key == "phi_deg_for_nonzero_displacements"
            ):
                try:
                    matches = bool(np.allclose(np.asarray(actual, dtype=float), wanted))
                except (TypeError, ValueError):
                    matches = False
        if not matches:
            raise ValueError(
                f"prior contract candidate_generation mismatch for {key}: "
                f"expected {wanted!r}, got {actual!r}"
            )
    result = {"candidate_generation": dict(generation)}
    if path is not None:
        result.update({"path": str(path.resolve()), "sha256": digest_file(path)})
    return result


def _stats(values: npt.NDArray[np.float64], prefix: str, unit: str) -> dict[str, float]:
    magnitude = np.abs(values)
    return {
        f"peak_abs_{prefix}_{unit}": float(np.max(magnitude)),
        f"median_abs_{prefix}_{unit}": float(np.median(magnitude)),
    }


def field_metrics_for_candidate(
    samples: npt.NDArray[np.float64],
    nearest: npt.NDArray[np.int64],
    cell_potential: npt.NDArray[np.float64],
) -> dict[str, Any]:
    """Sample a candidate and report AF/E statistics at 2, 5 and 10 mm bins."""
    sampled = np.asarray(cell_potential, dtype=np.float64)[nearest]
    by_bin: dict[str, Any] = {}
    for width_mm, width_m in zip(BIN_WIDTHS_MM, BIN_WIDTHS_M, strict=True):
        binned, _ = bin_potential(samples, sampled, width_m)
        key = f"{width_mm:g}"
        if binned.size < 5:
            by_bin[key] = None
            continue
        af = activating_function(binned, width_m)
        longitudinal = np.diff(binned) / width_m
        metrics = {"bins": int(binned.size)}
        metrics.update(_stats(af, "af", "v_per_a_m2"))
        metrics.update(_stats(longitudinal, "longitudinal_e", "v_per_a_m"))
        # Descriptive aliases retain the wording used in the ensemble protocol.
        metrics["peak_abs_activating_function_v_per_a_m2"] = metrics[
            "peak_abs_af_v_per_a_m2"
        ]
        metrics["median_abs_activating_function_v_per_a_m2"] = metrics[
            "median_abs_af_v_per_a_m2"
        ]
        by_bin[key] = metrics
    return {"by_bin_mm": by_bin}


def _directory_hash(path: Path) -> str:
    digest = hashlib.sha256()
    files = (p for p in path.rglob("*") if p.is_file())
    for item in sorted(files, key=lambda p: p.relative_to(path).as_posix()):
        digest.update(item.relative_to(path).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(bytes.fromhex(digest_file(item)))
    return digest.hexdigest()


def _decode_field_ids(values: Any) -> list[str]:
    return [_decode(value) for value in np.asarray(values).reshape(-1)]


def _load_field(
    field: Path,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    with h5py.File(field, "r") as handle:
        try:
            L_phi = np.asarray(handle["L_phi"][...], dtype=np.float64)
            basis = np.asarray(handle["basis_matrix"][...], dtype=np.float64)
        except KeyError as exc:
            raise ValueError(f"field artifact missing dataset: {exc.args[0]}") from exc
        if "electrode_ids" in handle:
            electrode_ids = _decode_field_ids(handle["electrode_ids"][...])
        else:
            raw = handle.attrs.get("electrode_ids")
            if raw is None:
                provenance = handle.attrs.get("provenance_json")
                if provenance:
                    raw = json.loads(_decode(provenance)).get("electrode_ids")
            if raw is None:
                raise ValueError("field artifact has no electrode_ids")
            raw_ids = raw if isinstance(raw, Sequence) else json.loads(_decode(raw))
            electrode_ids = [str(value) for value in raw_ids]
    if len(electrode_ids) != basis.shape[0]:
        raise ValueError("field electrode_ids count does not match basis_matrix rows")
    return L_phi, basis, electrode_ids


def _load_mesh_points_and_check_field_layout(
    mesh: Path, field: Path, *, chunk_size: int = 250_000
) -> np.ndarray:
    """Load only mesh nodes; compare field nodes in bounded chunks."""
    with h5py.File(mesh, "r") as mesh_handle:
        points = np.asarray(mesh_handle["mesh/points_xyz_m"][...], dtype=np.float64)
        tetrahedra_shape = tuple(mesh_handle["mesh/tetrahedra"].shape)
    with h5py.File(field, "r") as field_handle:
        field_points = field_handle["node_coordinates"]
        if tuple(field_points.shape) != tuple(points.shape):
            raise ValueError("field node_coordinates shape does not match the bound mesh")
        if tuple(field_handle["mesh/tetrahedra"].shape) != tetrahedra_shape:
            raise ValueError("field mesh/tetrahedra shape does not match mesh")
        for start in range(0, points.shape[0], chunk_size):
            stop = min(start + chunk_size, points.shape[0])
            if not np.allclose(
                field_points[start:stop], points[start:stop], rtol=0.0, atol=1e-12
            ):
                raise ValueError("field node_coordinates do not match the bound mesh")
    return points


def _root_name(key: tuple[str, str]) -> str:
    return f"{key[0]}_{key[1]}"


def _subject_paths(markers: Path) -> list[tuple[str, Path | None]]:
    found = {path.name: path for path in markers.iterdir() if path.is_dir()}
    names = list(EXPECTED_SUBJECTS)
    names.extend(sorted(set(found) - set(names)))
    return [(name, found.get(name)) for name in names]


def _empty_root(
    reason: str, key: tuple[str, str], source_points: int | None = None
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "root": _root_name(key),
        "segment": key[0],
        "side": key[1],
        "root_class": "root_complex",
        "status": "missing" if reason == "root marker missing" else "short",
        "reason": reason,
    }
    if source_points is not None:
        result["source_points"] = int(source_points)
    return result


def _alignment_entry(subject_name: str, fit_side: str, test_side: str) -> dict[str, Any]:
    return {
        "subject": subject_name,
        "fit_side": fit_side,
        "test_side": test_side,
        "root_class": "root_complex",
        "processed": False,
        "placed_roots": [],
        "missing_roots": [],
        "short_roots": [],
    }


def run_ensemble(
    *,
    mesh: Path,
    field: Path,
    montage_spec: Path,
    markers: Path,
    output: Path,
    prior_contract: Path = DEFAULT_PRIOR_CONTRACT,
) -> dict[str, Any]:
    """Run the ensemble and write one JSON receipt."""
    for path in (mesh, field, montage_spec, prior_contract):
        if not path.is_file():
            raise FileNotFoundError(f"required input missing: {path}")
    if not markers.is_dir():
        raise FileNotFoundError(f"markers directory missing: {markers}")
    prior_binding = validate_prior_contract(prior_contract)

    # This existing check binds the field to the exact mesh before any samples.
    from fem.sample_level_anchored_root_fields import validate_field_mesh_binding

    field_binding = validate_field_mesh_binding(field, mesh)
    L_phi, basis_matrix, electrode_ids = _load_field(field)

    mesh_sha = str(field_binding["mesh_sha256"])
    field_sha = digest_file(field)
    montage_sha = digest_file(montage_spec)
    markers_sha = _directory_hash(markers)
    reconstruction = reconstruct_named_current_montage(
        L_phi, basis_matrix, electrode_ids, montage_spec
    )
    del L_phi
    reconstruction.pop("nodal_potential")
    points = _load_mesh_points_and_check_field_layout(mesh, field)
    alvar = load_alvar(mesh)
    subjects: list[dict[str, Any]] = []
    contexts: list[
        tuple[dict[str, Any], dict[str, Any], tuple[str, str], np.ndarray, np.ndarray]
    ] = []
    baseline_lines: list[np.ndarray] = []
    subject_paths = _subject_paths(markers)
    for subject_name, subject_path in subject_paths:
        subject_entry: dict[str, Any] = {
            "subject": subject_name,
            "root_class": "root_complex",
            "fit_sides": {},
        }
        if subject_path is None:
            for fit_side in ("L", "R"):
                entry = _alignment_entry(
                    subject_name, fit_side, "R" if fit_side == "L" else "L"
                )
                entry["reason"] = "subject directory missing"
                entry["missing_roots"] = [
                    _empty_root("subject directory missing", key) for key in ROOT_KEYS
                ]
                for root in entry["missing_roots"]:
                    root["status"] = "missing"
                subject_entry["fit_sides"][fit_side] = entry
            subjects.append(subject_entry)
            continue
        try:
            subject = load_subject(subject_path)
        except Exception as exc:  # preserve every subject even when geometry is incomplete
            for fit_side in ("L", "R"):
                entry = _alignment_entry(
                    subject_name, fit_side, "R" if fit_side == "L" else "L"
                )
                entry["reason"] = "subject geometry missing"
                entry["error"] = str(exc)
                entry["missing_roots"] = [
                    _empty_root("subject geometry missing", key) for key in ROOT_KEYS
                ]
                for root in entry["missing_roots"]:
                    root["status"] = "missing"
                subject_entry["fit_sides"][fit_side] = entry
            subjects.append(subject_entry)
            continue

        for fit_side in ("L", "R"):
            test_side = "R" if fit_side == "L" else "L"
            entry = _alignment_entry(subject.name, fit_side, test_side)
            fit_rows = level_pairs(subject, alvar, fit_side)
            test_rows = level_pairs(subject, alvar, test_side)
            if len(fit_rows) < 3 or len(test_rows) < 3:
                entry["reason"] = "too few levels to fit a longitudinal map"
                subject_entry["fit_sides"][fit_side] = entry
                continue
            mapping = fit_map(
                [(row[1], row[2]) for row in fit_rows],
                tuple(row[0] for row in fit_rows),
            )
            fit_residual = [
                abs(mapping.apply(row[1]) - row[2]) * 1000.0 for row in fit_rows
            ]
            test_residual = [
                abs(mapping.apply(row[1]) - row[2]) * 1000.0 for row in test_rows
            ]
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
                }
            )
            for key in ROOT_KEYS:
                line = subject.roots.get(key)
                if line is None:
                    entry["missing_roots"].append(_empty_root("root marker missing", key))
                    continue
                line = np.asarray(line, dtype=np.float64)
                if line.ndim != 2 or line.shape[1] != 3 or line.shape[0] < 2:
                    entry["short_roots"].append(
                        _empty_root(
                            "root marker has fewer than two points",
                            key,
                            line.shape[0] if line.ndim else 0,
                        )
                    )
                    continue
                placed, dropped = place_root(subject, alvar, line, mapping.apply)
                if placed is None or placed.shape[0] < MIN_RETAINED_POINTS:
                    entry["short_roots"].append(
                        _empty_root(
                            "fewer than five points inside the mapped thecal sac",
                            key,
                            0 if placed is None else placed.shape[0],
                        )
                    )
                    continue
                try:
                    baseline = resample_by_arclength(placed, SAMPLE_SPACING_M)
                except ValueError as exc:
                    entry["short_roots"].append(
                        _empty_root(str(exc), key, placed.shape[0])
                    )
                    continue
                root_record = {
                    "root": _root_name(key),
                    "segment": key[0],
                    "side": key[1],
                    "root_class": "root_complex",
                    "status": "ready",
                    "source_points": int(line.shape[0]),
                    "retained_samples": int(placed.shape[0]),
                    "dropped_points": int(dropped),
                    "scenarios": [],
                }
                entry["placed_roots"].append(root_record)
                baseline_lines.append(placed)
                contexts.append((entry, root_record, key, baseline, placed))
            subject_entry["fit_sides"][fit_side] = entry
        subjects.append(subject_entry)

    tree = None
    local_label = None
    cell_nodes = None
    cell_potential = None
    if baseline_lines:
        stacked = np.vstack(baseline_lines)
        with h5py.File(mesh, "r") as handle:
            tree, local_label, cell_nodes = local_cell_context(
                points,
                handle["mesh/tetrahedra"],
                handle["mesh/source_label"],
                stacked,
            )
        nodal = reconstruction["nodal_potential_v_per_a"]
        cell_potential = nodal[cell_nodes].mean(axis=1)
        for _entry, root_record, _key, baseline, _placed in contexts:
            for scenario in candidate_scenarios():
                displacement = float(scenario["displacement_mm"])
                phi = scenario["phi_deg"]
                if displacement == 0.0:
                    samples = baseline
                    degenerate = 0
                else:
                    samples, degenerate = apply_local_direction_displacement(
                        baseline, alvar, displacement, float(phi)
                    )
                _, nearest = tree.query(samples, k=1)
                record = {
                    "scenario_id": scenario["scenario_id"],
                    "displacement_mm": displacement,
                    "phi_deg": phi,
                    "sample_count": int(samples.shape[0]),
                    "degenerate_radial_points": int(degenerate),
                    "geometry": geometry_screen(
                        _key, samples, local_label[nearest], alvar
                    ),
                    "field": field_metrics_for_candidate(samples, nearest, cell_potential),
                }
                root_record["scenarios"].append(record)

    montage_spec_document = load_montage_spec(montage_spec)
    montage_name = montage_spec_document.get("name", montage_spec.stem)
    montage_id = str(
        montage_spec_document.get("montage_id", montage_name)
    )
    receipt = {
        "schema_version": 1,
        "generated_at": datetime.now(UTC).isoformat(),
        "status": "TSS_UP_ROOT_CANDIDATE_ENSEMBLE",
        "source_hashes": {
            "mesh_sha256": mesh_sha,
            "field_sha256": field_sha,
            "montage_spec_sha256": montage_sha,
            "markers_sha256": markers_sha,
            "prior_contract_sha256": prior_binding["sha256"],
        },
        "provenance": {
            "mesh": {"path": str(mesh.resolve()), "sha256": mesh_sha},
            "field": {"path": str(field.resolve()), "sha256": field_sha},
            "montage_spec": {
                "path": str(montage_spec.resolve()),
                "sha256": montage_sha,
            },
            "markers": {"path": str(markers.resolve()), "sha256": markers_sha},
            "prior_contract": prior_binding,
            "field_binding": field_binding,
        },
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
        },
        "montage": {
            "montage_id": montage_id,
            "electrode_ids": reconstruction["electrode_ids"],
            "named_currents_a": reconstruction["named_currents_a"],
            "currents_a": reconstruction["currents_a"].tolist(),
            "drive_current_a": reconstruction["drive_current_a"],
            "normalized_currents_per_a": reconstruction["normalized_currents_per_a"].tolist(),
            "basis_coefficients": reconstruction["basis_coefficients"].tolist(),
            "reconstructed_currents_a": reconstruction["reconstructed_currents_a"].tolist(),
            "current_sum_a": reconstruction["current_sum_a"],
            "max_abs_reconstruction_error_a": reconstruction["max_abs_reconstruction_error_a"],
            "relative_reconstruction_l2": reconstruction["relative_reconstruction_l2"],
            "potential_units": {"nodal_potential": "V", "sampled_potential": "V/A"},
            "normalization": (
                "actual L_phi current reconstruction is retained as nodal_potential; "
                "sampling divides by positive drive_current_a"
            ),
        },
        "method": {
            "fit_sides": ["L", "R"],
            "sample_spacing_m": SAMPLE_SPACING_M,
            "bin_widths_mm": list(BIN_WIDTHS_MM),
            "scenario_count_per_root": len(candidate_scenarios()),
            "candidate_displacements_mm": [0.0, *DISPLACEMENTS_MM],
            "candidate_phi_deg": list(PHI_DEG),
            "displacement_frame": "per-point local radial/tangential frame; phi=0 radial outward",
            "sampling": "nearest cell centroid; cell potential is the mean of four node potentials",
        },
        "per_subject": subjects,
        "checks": {
            "named_current_reconstruction": True,
            "zero_sum_currents": abs(reconstruction["current_sum_a"]) <= 1e-12,
            "scenario_count_per_root": len(candidate_scenarios()),
            "fit_sides_run": ["L", "R"],
        },
        "sets_no_gates": True,
        "dorsal_ventral_identity": "unresolved",
        "root_class": "root_complex",
        "ex_vivo_geometry_hypothesis_warning": (
            "TSS-UP contact geometry and root-complex placement are hypotheses, "
            "not measured subject anatomy."
        ),
        "warnings": [
            (
                "ex-vivo/geometry hypothesis warning: TSS-UP contact geometry and "
                "root-complex placement are hypotheses, not measured subject anatomy."
            ),
            (
                "Field values are transfer metrics in V/A; no biological recruitment "
                "inference is made."
            ),
        ],
        "claim_scope": (
            "Exploratory TSS-UP lead-field sensitivity over LSSM root-complex "
            "candidates; no gate is set."
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return receipt


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mesh", type=Path, default=DEFAULT_MESH)
    parser.add_argument("--field", type=Path, required=True)
    parser.add_argument(
        "--montage-spec",
        "--montage",
        dest="montage_spec",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--prior-contract",
        type=Path,
        default=DEFAULT_PRIOR_CONTRACT,
    )
    parser.add_argument("--markers", type=Path, default=DEFAULT_MARKERS)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args(argv)
    run_ensemble(
        mesh=arguments.mesh,
        field=arguments.field,
        montage_spec=arguments.montage_spec,
        markers=arguments.markers,
        output=arguments.output,
        prior_contract=arguments.prior_contract,
    )
    print(f"wrote {arguments.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
