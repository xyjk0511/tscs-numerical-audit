"""Resample frozen fields and run the common-physical-support experiment.

This additive runner reads the immutable publication cache, writes only to a
separate output directory, and refuses to accept a newly sampled path unless its
native complete-bin AF peaks replay the corresponding frozen 0.125-mm rows.
It performs no FEM solve and writes no anatomical xyz coordinates.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import os
import platform
import sys
from collections import defaultdict
from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.no_new_mri_functional_calibration.common_support import (  # noqa: E402
    CENTRAL_MARGIN_M,
    COMMON_ENDPOINT_QUANTUM_M,
    MIN_COMPLETE_BINS,
    VARIANTS,
    WIDTHS_M,
    analyze_profile_variants,
    common_endpoint_m,
    normalize_profiles,
)
from paper.numerical_audit.matched_analysis import (  # noqa: E402
    SEGMENTS,
    geometry_mean,
    share,
)

SPACING_M = 0.000125
EXPECTED_OFFSETS = (0, 10, 20, 100)
REPLAY_RTOL = 5e-12
REPLAY_ATOL = 1e-8
SCHEMA = "common-physical-support-v1"
PROFILE_SCHEMA = "common-support-profile-v1"
PATH_SCHEMA = "common-support-path-v1"


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 << 20), b""):
            value.update(block)
    return value.hexdigest()


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def binding_digest(value: dict[str, Any]) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(value, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    os.replace(temporary, path)


def write_csv_atomic(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def _is_same_or_descendant(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def validate_output_location(audit_root: Path, output: Path) -> tuple[Path, Path]:
    """Resolve paths and refuse any write at or below the frozen cache."""
    frozen = audit_root.expanduser().resolve(strict=True)
    target = output.expanduser().resolve(strict=False)
    if _is_same_or_descendant(target, frozen):
        raise ValueError("--output must be outside --audit-root and its subdirectories")
    return frozen, target


def _relative(audit_root: Path, path: Path) -> str:
    return path.relative_to(audit_root).as_posix()


def _read_manifest(audit_root: Path, publication: dict[str, Any]) -> tuple[dict[str, str], str]:
    path = audit_root / "artifact-manifest.json"
    raw = path.read_bytes()
    actual = hashlib.sha256(raw).hexdigest()
    expected = publication.get("artifact_manifest_sha256")
    if actual != expected:
        raise ValueError("Frozen artifact manifest does not match publication_results.json")
    document = json.loads(raw)
    artifacts = document.get("artifacts")
    if not isinstance(artifacts, dict) or not artifacts:
        raise ValueError("artifact-manifest.json has no artifact hash mapping")
    if not all(isinstance(k, str) and isinstance(v, str) for k, v in artifacts.items()):
        raise ValueError("artifact-manifest.json artifact mapping is malformed")
    return artifacts, actual


def _input_record(audit_root: Path, path: Path, manifest: dict[str, str]) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    actual = digest(path)
    relative = _relative(audit_root, path)
    expected = manifest.get(relative)
    if expected is not None and actual != expected:
        raise ValueError(f"Frozen artifact hash mismatch: {relative}")
    return {"path": relative, "sha256": actual, "artifact_manifest_bound": expected is not None}


def _code_hashes() -> dict[str, str]:
    paths = [
        Path(__file__).resolve(),
        ROOT / "analysis/no_new_mri_functional_calibration/common_support.py",
        ROOT / "analysis/no_new_mri_functional_calibration/path_numerics_v2.py",
        ROOT / "analysis/no_new_mri_functional_calibration/publication_recompute.py",
        ROOT / "analysis/no_new_mri_functional_calibration/sample_root_ensemble.py",
        ROOT / "fem/place_lumbosacral_roots.py",
        ROOT / "fem/sample_level_anchored_root_fields.py",
        ROOT / "paper/numerical_audit/matched_analysis.py",
    ]
    return {path.relative_to(ROOT).as_posix(): digest(path) for path in paths}


def build_binding(audit_root: Path) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]], dict[str, str]]:
    publication_path = ROOT / "analysis/no_new_mri_functional_calibration/publication_results.json"
    publication = json.loads(publication_path.read_text(encoding="utf-8"))
    frozen_code = publication.get("code", {})
    for name in ("publication_recompute.py", "path_numerics_v2.py"):
        source = ROOT / "analysis/no_new_mri_functional_calibration" / name
        if frozen_code.get(name) != digest(source):
            raise ValueError(f"Current {name} differs from the code bound by publication_results.json")
    manifest, manifest_sha256 = _read_manifest(audit_root, publication)
    prepared_path = audit_root / "prepared.json"
    contexts_path = audit_root / "contexts.json"
    prepared = json.loads(prepared_path.read_text(encoding="utf-8"))
    contexts = json.loads(contexts_path.read_text(encoding="utf-8"))
    if prepared.get("schema") != "publication-recompute-v1":
        raise ValueError("Unsupported prepared.json schema")
    if not isinstance(contexts, list) or len(contexts) != prepared.get("n_paths"):
        raise ValueError("contexts.json does not match prepared path count")
    if len(contexts) != 392 or [row.get("index") for row in contexts] != list(range(392)):
        raise ValueError("Expected 392 consecutively indexed path contexts")
    fields = prepared.get("fields")
    if not isinstance(fields, list):
        raise ValueError("prepared.json has no field bindings")
    offsets = tuple(int(row["offset_mm"]) for row in fields)
    if offsets != EXPECTED_OFFSETS:
        raise ValueError(f"Expected field offsets {EXPECTED_OFFSETS}, got {offsets}")
    publication_fields = {int(row["offset_mm"]): row for row in publication["inputs"]["fields"]}
    if prepared.get("source_mesh_sha256") != publication["inputs"]["mesh_sha256"]:
        raise ValueError("Prepared mesh binding differs from publication_results.json")
    for field in fields:
        expected = publication_fields.get(int(field["offset_mm"]))
        if expected is None or any(
            field.get(name) != expected[name]
            for name in ("ensemble_sha256", "field_sha256", "spec_sha256")
        ):
            raise ValueError("Prepared field binding differs from publication_results.json")

    required = [
        prepared_path,
        contexts_path,
        audit_root / "baseline-paths.npz",
        audit_root / "points.npy",
        audit_root / "locator-cells.npy",
        audit_root / "geometry-only.h5",
        audit_root / "samples-0.125mm/run-meta.json",
    ]
    required.extend(audit_root / f"potential-{offset}.npy" for offset in offsets)
    inputs = [_input_record(audit_root, path, manifest) for path in required]
    binding = {
        "schema": SCHEMA,
        "source_artifact_manifest_sha256": manifest_sha256,
        "publication_results_sha256": digest(publication_path),
        "inputs": inputs,
        "code": _code_hashes(),
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": version("scipy"),
            "h5py": version("h5py"),
        },
        "parameters": {
            "spacing_m": SPACING_M,
            "widths_m": list(WIDTHS_M),
            "minimum_complete_bins_per_width_per_variant": MIN_COMPLETE_BINS,
            "common_endpoint": {
                "formula": "0.010 * floor(L / 0.010) with only 8-epsilon endpoint representation tolerance",
                "quantum_m": COMMON_ENDPOINT_QUANTUM_M,
                "never_extrapolate": True,
            },
            "central_search_interval_m": [CENTRAL_MARGIN_M, "L_common-0.015"],
            "variants": {
                "original_native": "full profile; complete bins per width; all native AF centres",
                "common_endpoint": "same-origin integration through L_common; all native AF centres",
                "common_endpoint_central": "same common bins; native centres within inclusive central interval",
            },
            "replay": {"rtol": REPLAY_RTOL, "atol_v_per_a_m2": REPLAY_ATOL},
            "normalization": "subtract the first potential sample independently for each field",
            "primary_discriminator": {
                "first_offset_mm": 0,
                "second_offset_mm": 100,
                "segment": "L4",
                "required_width_signs_2_5_10mm": ["negative", "negative", "positive"],
            },
        },
    }
    binding["binding_sha256"] = binding_digest(binding)
    return binding, prepared, contexts, manifest


def ensure_run_meta(output: Path, binding: dict[str, Any]) -> None:
    output.mkdir(parents=True, exist_ok=True)
    path = output / "run-meta.json"
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != binding:
            raise ValueError("Existing output has a different input/code/parameter binding")
    else:
        write_json_atomic(path, binding)


def _load_frozen_rows(
    audit_root: Path,
    path_index: int,
    context: dict[str, Any],
    artifact_manifest: dict[str, str],
) -> dict[tuple[str, int], dict[str, Any]]:
    path = audit_root / "samples-0.125mm" / f"{path_index:03d}.json"
    relative = _relative(audit_root, path)
    expected_hash = artifact_manifest.get(relative)
    if expected_hash is None:
        raise ValueError(f"Frozen sample is not bound by artifact-manifest.json: {relative}")
    if digest(path) != expected_hash:
        raise ValueError(f"Frozen sample hash mismatch: {relative}")
    records: dict[tuple[str, int], dict[str, Any]] = {}
    for row in json.loads(path.read_text(encoding="utf-8")):
        if row.get("method") != "p1_complete":
            raise ValueError(f"Unexpected frozen method in {relative}")
        for name in ("index", "subject", "fit", "segment", "side"):
            if row.get(name) != context.get(name):
                raise ValueError(f"Frozen row context mismatch for {name}: {relative}")
        key = (str(row["scenario"]), int(row["offset"]))
        if key in records:
            raise ValueError(f"Duplicate frozen scenario/offset row: {relative}")
        records[key] = row
    if len(records) != 15 * len(EXPECTED_OFFSETS):
        raise ValueError(f"Unexpected frozen row count in {relative}")
    return records


def _scalar_descriptor(descriptor: dict[str, Any]) -> dict[str, Any]:
    return {
        "valid": bool(descriptor["valid"]),
        "exclusion": descriptor["exclusion"],
        "n_bins": int(descriptor["n_bins"]),
        "candidate_count": int(descriptor["candidate_count"]),
        "retained_endpoint_m": float(descriptor["retained_endpoint_m"]),
        "discarded_from_full_length_m": float(descriptor["discarded_from_full_length_m"]),
        "peak_abs_af_v_per_a_m2": descriptor["peak_abs_af_v_per_a_m2"],
    }


def verify_original_replay(
    frozen_row: dict[str, Any],
    analyzed: dict[str, Any],
    *,
    rtol: float = REPLAY_RTOL,
    atol: float = REPLAY_ATOL,
) -> None:
    """Fail on any missing-mask, bin-count, support, or native-peak mismatch."""
    field = frozen_row.get("field")
    if not isinstance(field, dict):
        raise ValueError("Frozen P1 row has no field metrics")
    by_bin = field.get("by_bin_mm", {})
    diagnostics = field.get("diagnostics_by_bin_mm", {})
    native = analyzed["variants"]["original_native"]
    for width_m in WIDTHS_M:
        key = f"{1000.0 * width_m:g}"
        actual = native[key]
        expected_metric = by_bin.get(key)
        expected_diagnostic = diagnostics.get(key)
        if not isinstance(expected_diagnostic, dict):
            raise ValueError(f"Frozen row lacks diagnostics for {key} mm")
        expected_valid = expected_metric is not None
        if bool(actual["valid"]) != expected_valid:
            raise ValueError(f"Original replay missing-mask mismatch at {key} mm")
        if int(actual["n_bins"]) != int(expected_diagnostic["n_bins"]):
            raise ValueError(f"Original replay bin-count mismatch at {key} mm")
        if not math.isclose(
            float(actual["discarded_from_full_length_m"]),
            float(expected_diagnostic["discarded_tail_m"]),
            rel_tol=0.0,
            abs_tol=5e-15,
        ):
            raise ValueError(f"Original replay tail mismatch at {key} mm")
        if not expected_valid:
            continue
        expected_peak = float(expected_metric["peak_abs_activating_function_v_per_a_m2"])
        actual_peak = float(actual["peak_abs_af_v_per_a_m2"])
        if not math.isclose(actual_peak, expected_peak, rel_tol=rtol, abs_tol=atol):
            raise ValueError(
                f"Original replay peak mismatch at {key} mm: "
                f"new={actual_peak:.17g}, frozen={expected_peak:.17g}"
            )


def verify_frozen_sampling_metadata(
    frozen_row: dict[str, Any],
    scenario: dict[str, Any],
    *,
    offset: int,
    sample_count: int,
) -> None:
    """Verify that one frozen row describes the exact trajectory just sampled."""
    expected = {
        "scenario": str(scenario["scenario_id"]),
        "offset": int(offset),
        "sample_count": int(sample_count),
    }
    for name, value in expected.items():
        if frozen_row.get(name) != value:
            raise ValueError(f"Frozen sampling metadata mismatch for {name}")
    if not math.isclose(
        float(frozen_row.get("spacing_mm", math.nan)),
        1000.0 * SPACING_M,
        rel_tol=0.0,
        abs_tol=1e-15,
    ):
        raise ValueError("Frozen sampling metadata mismatch for spacing_mm")
    if not math.isclose(
        float(frozen_row.get("d_mm", math.nan)),
        float(scenario["displacement_mm"]),
        rel_tol=0.0,
        abs_tol=1e-15,
    ):
        raise ValueError("Frozen sampling metadata mismatch for d_mm")
    expected_phi = scenario["phi_deg"]
    actual_phi = frozen_row.get("phi_deg")
    if expected_phi is None:
        if actual_phi is not None:
            raise ValueError("Frozen sampling metadata mismatch for phi_deg")
    elif actual_phi is None or not math.isclose(
        float(actual_phi), float(expected_phi), rel_tol=0.0, abs_tol=1e-12
    ):
        raise ValueError("Frozen sampling metadata mismatch for phi_deg")


def _profile_payload(
    *,
    binding_sha256: str,
    context: dict[str, Any],
    scenario: dict[str, Any],
    offsets: tuple[int, ...],
    s_m: np.ndarray,
    normalized_profiles: np.ndarray,
) -> dict[str, np.ndarray]:
    phi = np.nan if scenario["phi_deg"] is None else float(scenario["phi_deg"])
    return {
        "schema": np.asarray(PROFILE_SCHEMA),
        "binding_sha256": np.asarray(binding_sha256),
        "path_index": np.asarray(int(context["index"]), dtype=np.int64),
        "subject": np.asarray(str(context["subject"])),
        "fit": np.asarray(str(context["fit"])),
        "segment": np.asarray(str(context["segment"])),
        "side": np.asarray(str(context["side"])),
        "scenario": np.asarray(str(scenario["scenario_id"])),
        "displacement_mm": np.asarray(float(scenario["displacement_mm"])),
        "phi_deg": np.asarray(phi),
        "spacing_m": np.asarray(SPACING_M),
        "offset_mm": np.asarray(offsets, dtype=np.int64),
        "s_m": np.asarray(s_m, dtype=np.float64),
        "potential_v_per_a": np.asarray(normalized_profiles, dtype=np.float64),
    }


def _validate_profile_file(path: Path, payload: dict[str, np.ndarray]) -> None:
    with np.load(path, allow_pickle=False) as stored:
        if set(stored.files) != set(payload):
            raise ValueError(f"Existing profile file has a different schema: {path}")
        for key, expected in payload.items():
            actual = stored[key]
            if actual.dtype.kind in "f":
                if not np.array_equal(actual, expected, equal_nan=True):
                    raise ValueError(f"Existing profile content differs for {key}: {path}")
            elif not np.array_equal(actual, expected):
                raise ValueError(f"Existing profile content differs for {key}: {path}")


def _write_profile_atomic(path: Path, payload: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        _validate_profile_file(path, payload)
        return
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **payload)
    os.replace(temporary, path)


def _profile_name(scenario_id: str) -> str:
    allowed = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.-"
    if (
        not scenario_id
        or scenario_id in {".", ".."}
        or scenario_id.startswith(".")
        or any(ch not in allowed for ch in scenario_id)
    ):
        raise ValueError(f"Unsafe scenario identifier: {scenario_id!r}")
    return f"{scenario_id}.npz"


def _validate_completed_path(
    path: Path,
    output: Path,
    binding_sha256: str,
    *,
    expected_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    document = json.loads(path.read_text(encoding="utf-8"))
    if document.get("schema") != PATH_SCHEMA or document.get("binding_sha256") != binding_sha256:
        raise ValueError(f"Completed path has a different binding: {path}")
    context = document.get("context")
    if not isinstance(context, dict) or int(context.get("index", -1)) != int(path.stem):
        raise ValueError(f"Completed path context does not match its filename: {path}")
    if expected_context is not None and context != expected_context:
        raise ValueError(f"Completed path context differs from contexts.json: {path}")
    profiles = document.get("profiles")
    if not isinstance(profiles, list) or len(profiles) != 15:
        raise ValueError(f"Completed path has an incomplete profile index: {path}")
    profile_scenarios = {row.get("scenario") for row in profiles}
    if len(profile_scenarios) != 15:
        raise ValueError(f"Completed path has duplicate profile scenarios: {path}")
    profile_metadata: dict[str, tuple[float, float | None]] = {}
    expected_descriptors: dict[tuple[str, int, str, int], dict[str, Any]] = {}
    expected_profile_keys = {
        "schema", "binding_sha256", "path_index", "subject", "fit", "segment",
        "side", "scenario", "displacement_mm", "phi_deg", "spacing_m",
        "offset_mm", "s_m", "potential_v_per_a",
    }
    for row in profiles:
        profile = (output / row["path"]).resolve(strict=False)
        if not _is_same_or_descendant(profile, output.resolve(strict=True)):
            raise ValueError(f"Completed path profile escapes the output directory: {path}")
        if not profile.is_file() or digest(profile) != row["sha256"]:
            raise ValueError(f"Completed path profile is missing or changed: {profile}")
        with np.load(profile, allow_pickle=False) as stored:
            if set(stored.files) != expected_profile_keys:
                raise ValueError(f"Completed profile has a different schema: {profile}")
            if str(stored["schema"].item()) != PROFILE_SCHEMA:
                raise ValueError(f"Completed profile has an unknown schema: {profile}")
            if str(stored["binding_sha256"].item()) != binding_sha256:
                raise ValueError(f"Completed profile has a different binding: {profile}")
            if int(stored["path_index"].item()) != int(context["index"]):
                raise ValueError(f"Completed profile has a different path index: {profile}")
            for name in ("subject", "fit", "segment", "side"):
                if str(stored[name].item()) != str(context[name]):
                    raise ValueError(f"Completed profile has a different {name}: {profile}")
            scenario = str(stored["scenario"].item())
            if scenario != str(row["scenario"]):
                raise ValueError(f"Completed profile has a different scenario: {profile}")
            if tuple(int(value) for value in stored["offset_mm"]) != EXPECTED_OFFSETS:
                raise ValueError(f"Completed profile has different field offsets: {profile}")
            if float(stored["spacing_m"].item()) != SPACING_M:
                raise ValueError(f"Completed profile has a different spacing: {profile}")
            s_m = np.asarray(stored["s_m"], dtype=np.float64)
            potential = np.asarray(stored["potential_v_per_a"], dtype=np.float64)
            if (
                s_m.ndim != 1 or len(s_m) < 2 or not np.isfinite(s_m).all()
                or s_m[0] != 0.0 or np.any(np.diff(s_m) <= 0)
                or potential.shape != (len(EXPECTED_OFFSETS), len(s_m))
                or not np.isfinite(potential).all()
                or not np.array_equal(potential[:, 0], np.zeros(len(EXPECTED_OFFSETS)))
            ):
                raise ValueError(f"Completed profile violates the numerical profile contract: {profile}")
            length = float(s_m[-1] - s_m[0])
            if int(row.get("sample_count", -1)) != len(s_m):
                raise ValueError(f"Completed profile sample count differs: {profile}")
            if float(row.get("profile_length_m", math.nan)) != length:
                raise ValueError(f"Completed profile length differs: {profile}")
            if float(row.get("common_endpoint_m", math.nan)) != common_endpoint_m(length):
                raise ValueError(f"Completed profile common endpoint differs: {profile}")
            locate = row.get("locate")
            if (
                not isinstance(locate, dict)
                or set(locate) != {"n_queries", "n_failed", "fallback_queries", "fallback_hits"}
                or int(locate["n_queries"]) != len(s_m)
                or int(locate["n_failed"]) != 0
                or any(int(locate[name]) < 0 for name in locate)
            ):
                raise ValueError(f"Completed profile locator diagnostics differ: {profile}")
            displacement = float(stored["displacement_mm"].item())
            raw_phi = float(stored["phi_deg"].item())
            phi = None if math.isnan(raw_phi) else raw_phi
            profile_metadata[scenario] = (displacement, phi)
            for field_index, offset in enumerate(EXPECTED_OFFSETS):
                analysis = analyze_profile_variants(s_m, potential[field_index])
                for variant in VARIANTS:
                    for width_m in WIDTHS_M:
                        width_key = f"{1000.0 * width_m:g}"
                        descriptor_key = (
                            scenario,
                            offset,
                            variant,
                            int(round(1000.0 * width_m)),
                        )
                        expected_descriptors[descriptor_key] = _scalar_descriptor(
                            analysis["variants"][variant][width_key]
                        )
    results = document.get("results")
    expected_rows = 15 * len(EXPECTED_OFFSETS) * len(VARIANTS) * len(WIDTHS_M)
    if not isinstance(results, list) or len(results) != expected_rows:
        raise ValueError(f"Completed path has an incomplete descriptor table: {path}")
    keys = set()
    for row in results:
        key = (row.get("scenario"), row.get("offset"), row.get("variant"), row.get("width_mm"))
        if key in keys:
            raise ValueError(f"Completed path has duplicate descriptors: {path}")
        keys.add(key)
        if row.get("scenario") not in profile_scenarios:
            raise ValueError(f"Completed path has a descriptor for an unknown scenario: {path}")
        displacement, phi = profile_metadata[str(row["scenario"])]
        if float(row.get("d_mm", math.nan)) != displacement:
            raise ValueError(f"Completed path displacement metadata differs: {path}")
        actual_phi = row.get("phi_deg")
        if (phi is None) != (actual_phi is None) or (
            phi is not None and float(actual_phi) != phi
        ):
            raise ValueError(f"Completed path angle metadata differs: {path}")
        if row.get("variant") not in VARIANTS or int(row.get("offset", -1)) not in EXPECTED_OFFSETS:
            raise ValueError(f"Completed path has an unknown descriptor key: {path}")
        if int(row.get("width_mm", -1)) not in (2, 5, 10):
            raise ValueError(f"Completed path has an unknown width: {path}")
        if bool(row.get("valid")) == (row.get("peak_abs_af_v_per_a_m2") is None):
            raise ValueError(f"Completed path validity/peak fields disagree: {path}")
        if bool(row.get("valid")) == (row.get("exclusion") is not None):
            raise ValueError(f"Completed path validity/exclusion fields disagree: {path}")
        expected = expected_descriptors[key]
        actual = {name: row.get(name) for name in expected}
        if actual != expected:
            raise ValueError(f"Completed path descriptor differs from its saved profile: {path}")
    return document


def verify_completed_original_replay(
    document: dict[str, Any],
    frozen: dict[tuple[str, int], dict[str, Any]],
) -> None:
    """Replay the frozen native-peak gate from saved profiles on resume."""
    grouped: dict[tuple[str, int], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in document["results"]:
        if row["variant"] != "original_native":
            continue
        key = (str(row["scenario"]), int(row["offset"]))
        grouped[key][str(int(row["width_mm"]))] = row
    if set(grouped) != set(frozen):
        raise ValueError("Completed path native replay support differs from frozen rows")
    for key, rows in grouped.items():
        if set(rows) != {"2", "5", "10"}:
            raise ValueError("Completed path native replay lacks a width")
        analysis = {"variants": {"original_native": rows}}
        verify_original_replay(frozen[key], analysis)


def _scenario_result_rows(
    analyses_by_offset: dict[int, dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for offset, analysis in sorted(analyses_by_offset.items()):
        for variant in VARIANTS:
            for width_m in WIDTHS_M:
                width_key = f"{1000.0 * width_m:g}"
                rows.append({
                    "offset": offset,
                    "variant": variant,
                    "width_mm": int(round(1000.0 * width_m)),
                    **_scalar_descriptor(analysis["variants"][variant][width_key]),
                })
    return rows


def process_path(
    *,
    audit_root: Path,
    output: Path,
    binding_sha256: str,
    context: dict[str, Any],
    artifact_manifest: dict[str, str],
    baseline_paths: Any,
    fields: np.ndarray,
    offsets: tuple[int, ...],
    locator: Any,
    alvar: Any,
    candidate_scenarios: Any,
    apply_displacement: Any,
    resample_polyline: Any,
) -> None:
    path_index = int(context["index"])
    result_path = output / "paths" / f"{path_index:03d}.json"
    scenarios = list(candidate_scenarios())
    scenario_ids = [str(row["scenario_id"]) for row in scenarios]
    if len(scenarios) != 15 or len(set(scenario_ids)) != 15:
        raise ValueError("Expected 15 unique frozen candidate scenarios")
    frozen = _load_frozen_rows(audit_root, path_index, context, artifact_manifest)
    expected_frozen = {(scenario, offset) for scenario in scenario_ids for offset in offsets}
    if set(frozen) != expected_frozen:
        raise ValueError(f"Frozen scenario/offset support differs for path {path_index}")
    if result_path.exists():
        document = _validate_completed_path(
            result_path, output, binding_sha256, expected_context=context
        )
        verify_completed_original_replay(document, frozen)
        return
    baseline = np.asarray(baseline_paths[str(path_index)], dtype=np.float64)
    pending: list[dict[str, Any]] = []
    for scenario in scenarios:
        displacement = float(scenario["displacement_mm"])
        if displacement == 0:
            fixed = baseline
        else:
            fixed, _ = apply_displacement(
                baseline, alvar, displacement, float(scenario["phi_deg"])
            )
        query, s_m = resample_polyline(fixed, SPACING_M)
        indices, weights, locate = locator.locate(query)
        if int(locate["n_failed"]) != 0:
            raise ValueError(
                f"P1 locator failed for path {path_index}, scenario {scenario['scenario_id']}"
            )
        sampled = locator.sample(fields, indices, weights)
        if sampled.shape != (len(offsets), len(s_m)) or not np.isfinite(sampled).all():
            raise ValueError("Sampled profile shape or finiteness contract failed")
        normalized = normalize_profiles(sampled)
        analyses: dict[int, dict[str, Any]] = {}
        for field_index, offset in enumerate(offsets):
            frozen_row = frozen[(str(scenario["scenario_id"]), offset)]
            verify_frozen_sampling_metadata(
                frozen_row,
                scenario,
                offset=offset,
                sample_count=len(s_m),
            )
            analysis = analyze_profile_variants(s_m, normalized[field_index])
            verify_original_replay(frozen_row, analysis)
            analyses[offset] = analysis
        pending.append({
            "scenario": scenario,
            "s_m": s_m,
            "normalized": normalized,
            "locate": {
                "n_queries": int(locate["n_queries"]),
                "n_failed": int(locate["n_failed"]),
                "fallback_queries": int(locate["fallback_queries"]),
                "fallback_hits": int(locate["fallback_hits"]),
            },
            "analyses": analyses,
        })

    profile_index: list[dict[str, Any]] = []
    results: list[dict[str, Any]] = []
    for item in pending:
        scenario = item["scenario"]
        relative = Path("profiles") / f"{path_index:03d}" / _profile_name(str(scenario["scenario_id"]))
        profile_path = output / relative
        payload = _profile_payload(
            binding_sha256=binding_sha256,
            context=context,
            scenario=scenario,
            offsets=offsets,
            s_m=item["s_m"],
            normalized_profiles=item["normalized"],
        )
        _write_profile_atomic(profile_path, payload)
        profile_index.append({
            "scenario": scenario["scenario_id"],
            "path": relative.as_posix(),
            "sha256": digest(profile_path),
            "sample_count": int(len(item["s_m"])),
            "profile_length_m": float(item["s_m"][-1] - item["s_m"][0]),
            "common_endpoint_m": float(next(iter(item["analyses"].values()))["common_endpoint_m"]),
            "locate": item["locate"],
        })
        for row in _scenario_result_rows(item["analyses"]):
            results.append({
                "scenario": scenario["scenario_id"],
                "d_mm": float(scenario["displacement_mm"]),
                "phi_deg": scenario["phi_deg"],
                **row,
            })
    write_json_atomic(result_path, {
        "schema": PATH_SCHEMA,
        "binding_sha256": binding_sha256,
        "context": context,
        "profiles": profile_index,
        "results": results,
    })


def load_path_documents(
    output: Path,
    binding_sha256: str,
    contexts: list[dict[str, Any]],
) -> list[dict[str, Any]] | None:
    paths = [output / "paths" / f"{index:03d}.json" for index in range(len(contexts))]
    if not all(path.is_file() for path in paths):
        return None
    documents = []
    for context, path in zip(contexts, paths, strict=True):
        document = _validate_completed_path(
            path, output, binding_sha256, expected_context=context
        )
        documents.append(document)
    return documents


def verify_complete_frozen_replay_set(
    *,
    audit_root: Path,
    contexts: list[dict[str, Any]],
    documents: list[dict[str, Any]],
    artifact_manifest: dict[str, str],
    scenario_ids: list[str],
    offsets: tuple[int, ...],
) -> None:
    """Re-hash all frozen sample files before final aggregation, including resumes."""
    expected = {(scenario, offset) for scenario in scenario_ids for offset in offsets}
    for context, document in zip(contexts, documents, strict=True):
        path_index = int(context["index"])
        frozen = _load_frozen_rows(
            audit_root, path_index, context, artifact_manifest
        )
        if set(frozen) != expected:
            raise ValueError(f"Frozen scenario/offset support differs for path {path_index}")
        verify_completed_original_replay(document, frozen)


def build_vectors(documents: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    cells: dict[tuple[Any, ...], dict[str, float]] = defaultdict(dict)
    exclusions: list[dict[str, Any]] = []
    for document in documents:
        context = document["context"]
        segment = str(context["segment"])
        for row in document["results"]:
            key = (
                row["variant"], int(row["offset"]), str(context["subject"]),
                str(context["fit"]), str(row["scenario"]), str(context["side"]),
                int(row["width_mm"]),
            )
            if not row["valid"]:
                exclusions.append({
                    "level": "descriptor",
                    "path_index": int(context["index"]),
                    "geometry": context["subject"],
                    "fit": context["fit"],
                    "segment": segment,
                    "side": context["side"],
                    "scenario": row["scenario"],
                    "offset": int(row["offset"]),
                    "width_mm": int(row["width_mm"]),
                    "variant": row["variant"],
                    "reason": row["exclusion"],
                    "n_bins": int(row["n_bins"]),
                    "candidate_count": int(row["candidate_count"]),
                })
                continue
            if segment in cells[key]:
                raise ValueError("Duplicate segment descriptor")
            cells[key][segment] = float(row["peak_abs_af_v_per_a_m2"])

    vectors: list[dict[str, Any]] = []
    for key, values in sorted(cells.items()):
        missing = [segment for segment in SEGMENTS if segment not in values]
        if missing:
            exclusions.append({
                "level": "vector",
                "variant": key[0], "offset": key[1], "geometry": key[2],
                "fit": key[3], "scenario": key[4], "side": key[5],
                "width_mm": key[6], "reason": "incomplete_seven_segment_vector",
                "missing_segments": missing,
            })
            continue
        array = np.array([values[segment] for segment in SEGMENTS], dtype=np.float64)
        if not np.isfinite(array).all() or np.any(array < 0):
            raise ValueError("Complete vector is not finite and nonnegative")
        total = float(array.sum())
        if total <= 0:
            exclusions.append({
                "level": "vector",
                "variant": key[0], "offset": key[1], "geometry": key[2],
                "fit": key[3], "scenario": key[4], "side": key[5],
                "width_mm": key[6], "reason": "nonpositive_seven_segment_sum",
            })
            continue
        vectors.append({
            "variant": key[0], "offset": key[1], "geometry": key[2],
            "fit": key[3], "scenario": key[4], "side": key[5],
            "width_mm": key[6],
            **{segment: float(value) for segment, value in zip(SEGMENTS, array, strict=True)},
            "sum_peak": total,
            "units": "V/(A*m^2)",
        })
    return vectors, exclusions


def _vector_table(vectors: list[dict[str, Any]]) -> dict[tuple[Any, ...], np.ndarray]:
    table = {}
    for row in vectors:
        key = (
            row["variant"], int(row["offset"]), row["geometry"], row["fit"],
            row["scenario"], row["side"], int(row["width_mm"]),
        )
        if key in table:
            raise ValueError("Duplicate complete vector")
        table[key] = np.array([row[segment] for segment in SEGMENTS], dtype=np.float64)
    return table


def _support_sets(
    table: dict[tuple[Any, ...], np.ndarray], variant: str, first: int, second: int
) -> dict[tuple[int, int], set[tuple[str, str, str, str]]]:
    sets: dict[tuple[int, int], set[tuple[str, str, str, str]]] = {}
    for offset in (first, second):
        for width in (2, 5, 10):
            sets[(offset, width)] = {
                (key[2], key[3], key[4], key[5])
                for key in table
                if key[0] == variant and key[1] == offset and key[6] == width
            }
    return sets


def summarize_comparisons(
    vectors: list[dict[str, Any]], offsets: tuple[int, ...] = EXPECTED_OFFSETS
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, Any],
    list[dict[str, Any]],
]:
    """Geometry-equal descriptive comparisons on native and common support sets."""
    table = _vector_table(vectors)
    comparisons: list[dict[str, Any]] = []
    geometry_rows: list[dict[str, Any]] = []
    support_exclusions: list[dict[str, Any]] = []
    support_summary: dict[str, Any] = {"pairs": []}
    for first, second in itertools.combinations(offsets, 2):
        component_sets = {
            variant: _support_sets(table, variant, first, second)
            for variant in VARIANTS
        }
        native = {
            variant: set.intersection(*component_sets[variant].values())
            for variant in VARIANTS
        }
        if any(not native[variant] for variant in VARIANTS):
            raise ValueError(
                f"No complete matched support for placement pair {first}->{second}"
            )
        for variant in VARIANTS:
            universe = set.union(*component_sets[variant].values())
            for geometry, fit, scenario, side in sorted(universe - native[variant]):
                missing = [
                    {"offset": offset, "width_mm": width}
                    for (offset, width), values in sorted(component_sets[variant].items())
                    if (geometry, fit, scenario, side) not in values
                ]
                support_exclusions.append({
                    "level": "pair_support",
                    "variant": variant,
                    "first": first,
                    "second": second,
                    "geometry": geometry,
                    "fit": fit,
                    "scenario": scenario,
                    "side": side,
                    "reason": "not_complete_across_both_offsets_and_all_widths",
                    "missing": missing,
                })
        all_variant = set.intersection(*(native[variant] for variant in VARIANTS))
        differs = any(native[variant] != all_variant for variant in VARIANTS)
        if differs:
            for variant in VARIANTS:
                for geometry, fit, scenario, side in sorted(native[variant] - all_variant):
                    support_exclusions.append({
                        "level": "cross_variant_support",
                        "variant": variant,
                        "first": first,
                        "second": second,
                        "geometry": geometry,
                        "fit": fit,
                        "scenario": scenario,
                        "side": side,
                        "reason": "excluded_from_all_variant_intersection",
                    })
        pair_summary = {
            "first": first,
            "second": second,
            "variant_native_matched": {variant: len(native[variant]) for variant in VARIANTS},
            "all_variant_intersection": len(all_variant),
            "support_differs_across_variants": differs,
        }
        support_summary["pairs"].append(pair_summary)
        support_modes = [("variant_native", native)]
        if differs:
            support_modes.append((
                "all_variant_intersection",
                {variant: all_variant for variant in VARIANTS},
            ))
        for support_mode, support_by_variant in support_modes:
            for variant in VARIANTS:
                base_keys = sorted(support_by_variant[variant])
                for width in (2, 5, 10):
                    keys = [(geometry, fit, scenario, side, width)
                            for geometry, fit, scenario, side in base_keys]
                    before = np.array([
                        table[(variant, first, *key)] for key in keys
                    ])
                    after = np.array([
                        table[(variant, second, *key)] for key in keys
                    ])
                    if len(keys) == 0:
                        continue
                    share_change = 100.0 * (share(after) - share(before))
                    raw_change = after - before
                    share_mean, share_geometries = geometry_mean(share_change, keys)
                    raw_mean, raw_geometries = geometry_mean(raw_change, keys)
                    counts = defaultdict(int)
                    for geometry, _, _, _, _ in keys:
                        counts[geometry] += 1
                    for index, segment in enumerate(SEGMENTS):
                        share_signs = [float(value[index]) for value in share_geometries.values()]
                        raw_signs = [float(value[index]) for value in raw_geometries.values()]
                        comparisons.append({
                            "support_mode": support_mode,
                            "variant": variant,
                            "first": first,
                            "second": second,
                            "width_mm": width,
                            "segment": segment,
                            "paired_realizations": len(keys),
                            "n_geometry": len(share_geometries),
                            "equal_geometry_share_change_pp": float(share_mean[index]),
                            "equal_geometry_raw_peak_change": float(raw_mean[index]),
                            "share_geometry_positive": sum(value > 0 for value in share_signs),
                            "share_geometry_negative": sum(value < 0 for value in share_signs),
                            "share_geometry_zero": sum(value == 0 for value in share_signs),
                            "raw_geometry_positive": sum(value > 0 for value in raw_signs),
                            "raw_geometry_negative": sum(value < 0 for value in raw_signs),
                            "raw_geometry_zero": sum(value == 0 for value in raw_signs),
                        })
                        for geometry in share_geometries:
                            geometry_rows.append({
                                "support_mode": support_mode,
                                "variant": variant,
                                "first": first,
                                "second": second,
                                "width_mm": width,
                                "segment": segment,
                                "geometry": geometry,
                                "n_realizations": counts[geometry],
                                "share_change_pp": float(share_geometries[geometry][index]),
                                "raw_peak_change": float(raw_geometries[geometry][index]),
                            })
    return comparisons, geometry_rows, support_summary, support_exclusions


def _sign(value: float) -> str:
    return "positive" if value > 0 else "negative" if value < 0 else "zero"


def primary_discriminator(
    comparisons: list[dict[str, Any]], support_summary: dict[str, Any]
) -> dict[str, Any]:
    pair = next(row for row in support_summary["pairs"] if row["first"] == 0 and row["second"] == 100)
    support_mode = (
        "all_variant_intersection"
        if pair["support_differs_across_variants"]
        else "variant_native"
    )
    variants = {}
    complete = True
    for variant in VARIANTS:
        rows = sorted(
            (
                row for row in comparisons
                if row["support_mode"] == support_mode
                and row["variant"] == variant
                and row["first"] == 0 and row["second"] == 100
                and row["segment"] == "L4"
            ),
            key=lambda row: row["width_mm"],
        )
        values = [row["equal_geometry_share_change_pp"] for row in rows]
        widths = [row["width_mm"] for row in rows]
        if widths != [2, 5, 10]:
            complete = False
        variants[variant] = {
            "widths_mm": widths,
            "equal_geometry_share_change_pp": values,
            "signs": [_sign(value) for value in values],
        }
    expected = ["negative", "negative", "positive"]
    survives = complete and all(
        variants[name]["signs"] == expected
        for name in ("common_endpoint", "common_endpoint_central")
    )
    condition = (
        "stop_insufficient_common_support"
        if not complete
        else "continue_bounded_scale_result"
        if survives
        else "reframe_as_joint_scale_and_support_sensitivity"
    )
    return {
        "support_mode": support_mode,
        "first": 0,
        "second": 100,
        "segment": "L4",
        "predeclared_sign_pattern": expected,
        "variants": variants,
        "support_complete": complete,
        "condition": condition,
        "no_alternative_segment_selection": True,
    }


def write_aggregate(output: Path, documents: list[dict[str, Any]], binding_sha256: str) -> None:
    vectors, exclusions = build_vectors(documents)
    comparisons, geometry_rows, support, support_exclusions = summarize_comparisons(vectors)
    exclusions.extend(support_exclusions)
    expected_native = math.comb(len(EXPECTED_OFFSETS), 2) * len(VARIANTS) * 3 * len(SEGMENTS)
    observed_native = sum(row["support_mode"] == "variant_native" for row in comparisons)
    if observed_native != expected_native:
        raise ValueError(
            f"Expected {expected_native} native descriptive rows, got {observed_native}"
        )
    discriminator = primary_discriminator(comparisons, support)
    vector_fields = [
        "variant", "offset", "geometry", "fit", "scenario", "side", "width_mm",
        *SEGMENTS, "sum_peak", "units",
    ]
    comparison_fields = [
        "support_mode", "variant", "first", "second", "width_mm", "segment",
        "paired_realizations", "n_geometry", "equal_geometry_share_change_pp",
        "equal_geometry_raw_peak_change", "share_geometry_positive",
        "share_geometry_negative", "share_geometry_zero", "raw_geometry_positive",
        "raw_geometry_negative", "raw_geometry_zero",
    ]
    geometry_fields = [
        "support_mode", "variant", "first", "second", "width_mm", "segment",
        "geometry", "n_realizations", "share_change_pp", "raw_peak_change",
    ]
    write_csv_atomic(output / "complete_vectors.csv", vectors, vector_fields)
    write_csv_atomic(output / "comparisons.csv", comparisons, comparison_fields)
    write_csv_atomic(output / "geometry_means.csv", geometry_rows, geometry_fields)
    exclusions_path = output / "exclusions.jsonl"
    temporary = exclusions_path.with_name(f".{exclusions_path.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        for row in exclusions:
            stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
    os.replace(temporary, exclusions_path)
    write_json_atomic(output / "common_support_results.json", {
        "schema": SCHEMA,
        "binding_sha256": binding_sha256,
        "scope": (
            "deterministic numerical descriptors from frozen paths and fields; "
            "path/scenario rows are not independent human observations"
        ),
        "units": {
            "raw_peak_and_change": "V/(A*m^2)",
            "normalized_share_change": "percentage points",
        },
        "complete_vector_count": len(vectors),
        "exclusion_count": len(exclusions),
        "support": support,
        "primary_discriminator": discriminator,
        "outputs": {
            "complete_vectors": "complete_vectors.csv",
            "comparisons": "comparisons.csv",
            "geometry_means": "geometry_means.csv",
            "exclusions": "exclusions.jsonl",
        },
    })


def run(args: argparse.Namespace) -> None:
    audit_root, output = validate_output_location(args.audit_root, args.output)
    binding, prepared, contexts, manifest = build_binding(audit_root)
    ensure_run_meta(output, binding)
    start = int(args.start_index)
    stop = len(contexts) if args.stop_index is None else int(args.stop_index)
    if start < 0 or stop < start or stop > len(contexts):
        raise ValueError(f"Path range must satisfy 0 <= start <= stop <= {len(contexts)}")

    # Empirical-only imports stay out of the pure common_support module.
    from analysis.no_new_mri_functional_calibration.path_numerics_v2 import (
        TetraP1Sampler,
        resample_polyline,
    )
    from analysis.no_new_mri_functional_calibration.sample_root_ensemble import (
        candidate_scenarios,
    )
    from fem.place_lumbosacral_roots import load_alvar
    from fem.sample_level_anchored_root_fields import apply_local_direction_displacement

    offsets = tuple(int(row["offset_mm"]) for row in prepared["fields"])
    baseline_paths = np.load(audit_root / "baseline-paths.npz")
    points = np.load(audit_root / "points.npy", mmap_mode="r")
    locator_cells = np.load(audit_root / "locator-cells.npy", mmap_mode="r")
    fields = np.stack([
        np.load(audit_root / f"potential-{offset}.npy", mmap_mode="r")
        for offset in offsets
    ])
    locator = TetraP1Sampler(points, locator_cells)
    alvar = load_alvar(audit_root / "geometry-only.h5")
    try:
        for context in contexts[start:stop]:
            process_path(
                audit_root=audit_root,
                output=output,
                binding_sha256=binding["binding_sha256"],
                context=context,
                artifact_manifest=manifest,
                baseline_paths=baseline_paths,
                fields=fields,
                offsets=offsets,
                locator=locator,
                alvar=alvar,
                candidate_scenarios=candidate_scenarios,
                apply_displacement=apply_local_direction_displacement,
                resample_polyline=resample_polyline,
            )
            print(f"path {int(context['index']) + 1}/{len(contexts)} accepted", flush=True)
    finally:
        baseline_paths.close()
    documents = load_path_documents(output, binding["binding_sha256"], contexts)
    if documents is None:
        completed = len(list((output / "paths").glob("[0-9][0-9][0-9].json")))
        print(f"partial run complete: {completed}/{len(contexts)} path files present", flush=True)
        return
    scenario_ids = [str(row["scenario_id"]) for row in candidate_scenarios()]
    verify_complete_frozen_replay_set(
        audit_root=audit_root,
        contexts=contexts,
        documents=documents,
        artifact_manifest=manifest,
        scenario_ids=scenario_ids,
        offsets=offsets,
    )
    write_aggregate(output, documents, binding["binding_sha256"])
    print("all paths complete; aggregate outputs written", flush=True)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--audit-root", type=Path, required=True)
    result.add_argument("--output", type=Path, required=True)
    result.add_argument("--start-index", type=int, default=0, help="zero-based inclusive path index")
    result.add_argument("--stop-index", type=int, help="zero-based exclusive path index")
    return result


def main() -> None:
    run(parser().parse_args())


if __name__ == "__main__":
    main()
