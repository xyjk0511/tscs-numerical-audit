"""Focused deterministic checks for the common-physical-support experiment."""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from analysis.no_new_mri_functional_calibration.common_support import (
    VARIANTS,
    af_descriptor,
    analyze_profile_variants,
    common_endpoint_m,
    normalize_profiles,
)
from paper.numerical_audit.matched_analysis import SEGMENTS
from paper.numerical_audit.run_common_support import (
    _profile_payload,
    _profile_name,
    _scalar_descriptor,
    _validate_completed_path,
    _write_profile_atomic,
    build_vectors,
    digest,
    primary_discriminator,
    summarize_comparisons,
    validate_output_location,
    verify_completed_original_replay,
    verify_frozen_sampling_metadata,
    verify_original_replay,
    write_aggregate,
    write_json_atomic,
)


def test_profile_operator_matches_exact_sine_bin_integrals():
    width = 0.005
    wavenumber = 83.0
    length = 0.050
    spacing = 0.00001
    s = np.arange(round(length / spacing) + 1, dtype=np.float64) * spacing
    potential = np.sin(wavenumber * s)
    result = af_descriptor(s, potential, width)

    edges = np.arange(11, dtype=np.float64) * width
    exact_means = (
        np.cos(wavenumber * edges[:-1]) - np.cos(wavenumber * edges[1:])
    ) / (wavenumber * width)
    exact_af = np.diff(exact_means, n=2) / width**2
    assert result["valid"]
    # Piecewise-linear sampling error is O((k*h)^2), independent of AF code.
    assert_allclose(result["af_v_per_a_m2"], exact_af, rtol=2e-6, atol=2e-6)


def test_constructed_terminal_feature_is_removed_by_common_endpoint():
    s = np.arange(0, 0.058 + 0.0000625, 0.000125)
    # Curvature exists only in the 50--58 mm tail.
    potential = np.where(s > 0.050, ((s - 0.050) / 0.008) ** 2, 0.0)
    result = analyze_profile_variants(s, potential)["variants"]

    assert result["original_native"]["2"]["peak_abs_af_v_per_a_m2"] > 0
    assert result["original_native"]["5"]["peak_abs_af_v_per_a_m2"] > 0
    assert result["original_native"]["10"]["peak_abs_af_v_per_a_m2"] == 0
    for width in ("2", "5", "10"):
        assert result["common_endpoint"][width]["peak_abs_af_v_per_a_m2"] == 0
        assert result["common_endpoint_central"][width]["peak_abs_af_v_per_a_m2"] == 0
        assert result["common_endpoint"][width]["retained_endpoint_m"] == pytest.approx(0.050)
        assert result["common_endpoint_central"][width]["retained_endpoint_m"] == pytest.approx(0.050)


def test_five_bin_gate_precedes_candidate_mask_and_true_short_bin_is_not_rounded_up():
    true_short = 0.050 - 1e-12
    roundoff_short = np.nextafter(0.050, 0.0)
    assert common_endpoint_m(true_short) == pytest.approx(0.040, abs=1e-18)
    assert common_endpoint_m(roundoff_short) == roundoff_short

    short = analyze_profile_variants([0.0, true_short], [0.0, 1.0])["variants"]
    central = short["common_endpoint_central"]["10"]
    assert not central["valid"]
    assert central["exclusion"] == "fewer_than_five_complete_bins"
    assert central["n_bins"] == 4

    represented = analyze_profile_variants(
        [0.0, roundoff_short], [0.0, 1.0]
    )["variants"]["common_endpoint_central"]["10"]
    assert represented["valid"]
    assert represented["n_bins"] == 5
    assert_allclose(represented["candidate_centers_m"], [0.015, 0.025, 0.035], atol=2e-17)


def test_positive_potential_gauge_shift_does_not_change_descriptors():
    s = np.linspace(0.0, 0.067, 537)
    potential = 0.2 * np.sin(91 * s) + 0.07 * np.cos(37 * s)
    baseline = analyze_profile_variants(s, potential)
    shifted = analyze_profile_variants(s, potential + 123.5)
    for variant in VARIANTS:
        for width in ("2", "5", "10"):
            a = baseline["variants"][variant][width]
            b = shifted["variants"][variant][width]
            assert a["valid"] == b["valid"]
            assert_allclose(a["af_v_per_a_m2"], b["af_v_per_a_m2"], rtol=0, atol=3e-9)
            assert a["peak_abs_af_v_per_a_m2"] == pytest.approx(
                b["peak_abs_af_v_per_a_m2"], rel=0, abs=3e-9
            )


def test_central_mask_uses_each_widths_native_centres_without_interpolation():
    s = np.linspace(0.0, 0.050, 401)
    potential = s**2
    result = analyze_profile_variants(s, potential)["variants"]["common_endpoint_central"]
    assert_allclose(result["10"]["candidate_centers_m"], [0.015, 0.025, 0.035], atol=2e-17)
    assert_allclose(
        result["5"]["candidate_centers_m"],
        [0.0175, 0.0225, 0.0275, 0.0325],
        atol=2e-17,
    )
    assert_allclose(np.diff(result["10"]["candidate_centers_m"]), 0.010, atol=2e-17)
    assert_allclose(np.diff(result["5"]["candidate_centers_m"]), 0.005, atol=2e-17)


def _frozen_row_from_analysis(analysis):
    by_bin, diagnostics = {}, {}
    for width in ("2", "5", "10"):
        descriptor = analysis["variants"]["original_native"][width]
        diagnostics[width] = {
            "n_bins": descriptor["n_bins"],
            "discarded_tail_m": descriptor["discarded_from_full_length_m"],
        }
        by_bin[width] = None if not descriptor["valid"] else {
            "peak_abs_activating_function_v_per_a_m2": descriptor[
                "peak_abs_af_v_per_a_m2"
            ]
        }
    return {"field": {"by_bin_mm": by_bin, "diagnostics_by_bin_mm": diagnostics}}


def test_original_peak_replay_is_enforced_not_reported_as_a_new_result():
    s = np.linspace(0.0, 0.061, 489)
    analysis = analyze_profile_variants(s, np.sin(120 * s))
    frozen = _frozen_row_from_analysis(analysis)
    verify_original_replay(frozen, analysis)

    frozen["field"]["by_bin_mm"]["5"][
        "peak_abs_activating_function_v_per_a_m2"
    ] += 1e-3
    with pytest.raises(ValueError, match="peak mismatch"):
        verify_original_replay(frozen, analysis)


def test_original_replay_rejects_missing_mask_and_support_mismatch():
    s = np.linspace(0.0, 0.061, 489)
    analysis = analyze_profile_variants(s, np.sin(75 * s))
    frozen = _frozen_row_from_analysis(analysis)
    frozen["field"]["by_bin_mm"]["10"] = None
    with pytest.raises(ValueError, match="missing-mask mismatch"):
        verify_original_replay(frozen, analysis)

    frozen = _frozen_row_from_analysis(analysis)
    frozen["field"]["diagnostics_by_bin_mm"]["2"]["n_bins"] += 1
    with pytest.raises(ValueError, match="bin-count mismatch"):
        verify_original_replay(frozen, analysis)


def test_resume_replays_saved_native_descriptors_against_frozen_rows():
    s = np.linspace(0.0, 0.061, 489)
    analysis = analyze_profile_variants(s, np.sin(80 * s))
    results = []
    for width in (2, 5, 10):
        results.append({
            "scenario": "baseline",
            "offset": 0,
            "variant": "original_native",
            "width_mm": width,
            **_scalar_descriptor(
                analysis["variants"]["original_native"][str(width)]
            ),
        })
    frozen = {("baseline", 0): _frozen_row_from_analysis(analysis)}
    verify_completed_original_replay({"results": results}, frozen)
    frozen[("baseline", 0)]["field"]["by_bin_mm"]["2"][
        "peak_abs_activating_function_v_per_a_m2"
    ] += 1e-3
    with pytest.raises(ValueError, match="peak mismatch"):
        verify_completed_original_replay({"results": results}, frozen)


def test_frozen_sampling_metadata_binds_scenario_spacing_and_sample_count():
    scenario = {"scenario_id": "d1mm_phi45deg", "displacement_mm": 1.0, "phi_deg": 45.0}
    row = {
        "scenario": "d1mm_phi45deg",
        "offset": 100,
        "sample_count": 401,
        "spacing_mm": 0.125,
        "d_mm": 1.0,
        "phi_deg": 45.0,
    }
    verify_frozen_sampling_metadata(row, scenario, offset=100, sample_count=401)
    row["sample_count"] = 400
    with pytest.raises(ValueError, match="sample_count"):
        verify_frozen_sampling_metadata(row, scenario, offset=100, sample_count=401)


def _vector(variant, offset, geometry, width, l4_value):
    values = {segment: 1.0 for segment in SEGMENTS}
    values["L4"] = l4_value
    return {
        "variant": variant,
        "offset": offset,
        "geometry": geometry,
        "fit": "L",
        "scenario": "baseline",
        "side": "L",
        "width_mm": width,
        **values,
        "sum_peak": sum(values.values()),
    }


def test_variant_native_and_all_variant_intersection_are_both_reported_when_support_differs():
    vectors = []
    for variant in VARIANTS:
        geometries = ("g1",) if variant == "common_endpoint_central" else ("g1", "g2")
        for geometry in geometries:
            for width in (2, 5, 10):
                vectors.append(_vector(variant, 0, geometry, width, 1.0))
                vectors.append(_vector(variant, 100, geometry, width, 1.2))
    comparisons, geometry_rows, support, support_exclusions = summarize_comparisons(
        vectors, offsets=(0, 100)
    )
    pair = support["pairs"][0]
    assert pair["variant_native_matched"] == {
        "original_native": 2,
        "common_endpoint": 2,
        "common_endpoint_central": 1,
    }
    assert pair["all_variant_intersection"] == 1
    assert pair["support_differs_across_variants"]
    assert {row["support_mode"] for row in comparisons} == {
        "variant_native", "all_variant_intersection"
    }
    l4 = [
        row for row in comparisons
        if row["variant"] == "original_native" and row["width_mm"] == 2
        and row["segment"] == "L4"
    ]
    assert {row["paired_realizations"] for row in l4} == {1, 2}
    assert geometry_rows
    assert support_exclusions


def test_all_six_pairs_and_primary_l4_condition_are_generated():
    vectors = []
    for variant in VARIANTS:
        for offset in (0, 10, 20, 100):
            for geometry in ("g1", "g2"):
                for width in (2, 5, 10):
                    l4 = 1.0
                    if offset == 100:
                        l4 = 0.8 if width in (2, 5) else 1.2
                    elif offset:
                        l4 = 1.0 + offset / 1000.0
                    vectors.append(_vector(variant, offset, geometry, width, l4))
    comparisons, _, support, support_exclusions = summarize_comparisons(vectors)
    assert len(support["pairs"]) == math.comb(4, 2)
    assert len(comparisons) == math.comb(4, 2) * len(VARIANTS) * 3 * len(SEGMENTS)
    assert not support_exclusions
    primary = primary_discriminator(comparisons, support)
    assert primary["support_mode"] == "variant_native"
    assert primary["condition"] == "continue_bounded_scale_result"
    assert primary["variants"]["common_endpoint"]["signs"] == [
        "negative", "negative", "positive"
    ]


def test_empty_all_variant_intersection_is_not_misreported_as_a_negative_result():
    vectors = []
    for variant, geometry in zip(VARIANTS, ("g1", "g2", "g3"), strict=True):
        for width in (2, 5, 10):
            vectors.append(_vector(variant, 0, geometry, width, 1.0))
            vectors.append(_vector(variant, 100, geometry, width, 1.1))
    comparisons, _, support, _ = summarize_comparisons(vectors, offsets=(0, 100))
    primary = primary_discriminator(comparisons, support)
    assert primary["support_mode"] == "all_variant_intersection"
    assert not primary["support_complete"]
    assert primary["condition"] == "stop_insufficient_common_support"


def test_aggregate_writer_emits_all_contract_files(tmp_path):
    documents = []
    for path_index, segment in enumerate(SEGMENTS):
        rows = []
        for variant in VARIANTS:
            for offset in (0, 10, 20, 100):
                for width in (2, 5, 10):
                    peak = 1.0
                    if segment == "L4" and offset == 100:
                        peak = 0.8 if width in (2, 5) else 1.2
                    rows.append({
                        "scenario": "baseline",
                        "d_mm": 0.0,
                        "phi_deg": None,
                        "offset": offset,
                        "variant": variant,
                        "width_mm": width,
                        "valid": True,
                        "exclusion": None,
                        "n_bins": 5,
                        "candidate_count": 3,
                        "retained_endpoint_m": 0.050,
                        "discarded_from_full_length_m": 0.0,
                        "peak_abs_af_v_per_a_m2": peak,
                    })
        documents.append({
            "context": {
                "index": path_index,
                "subject": "g1",
                "fit": "L",
                "segment": segment,
                "side": "L",
            },
            "results": rows,
        })
    write_aggregate(tmp_path, documents, "d" * 64)
    expected = {
        "complete_vectors.csv",
        "comparisons.csv",
        "geometry_means.csv",
        "exclusions.jsonl",
        "common_support_results.json",
    }
    assert {path.name for path in tmp_path.iterdir()} == expected
    summary = json.loads((tmp_path / "common_support_results.json").read_text())
    assert summary["complete_vector_count"] == len(VARIANTS) * 4 * 3
    assert len(summary["support"]["pairs"]) == math.comb(4, 2)
    assert summary["primary_discriminator"]["condition"] == "continue_bounded_scale_result"


def test_zero_sum_complete_vector_is_retained_as_an_exclusion_not_normalized():
    documents = []
    for path_index, segment in enumerate(SEGMENTS):
        documents.append({
            "context": {
                "index": path_index,
                "subject": "g1",
                "fit": "L",
                "segment": segment,
                "side": "L",
            },
            "results": [{
                "scenario": "baseline",
                "offset": 0,
                "variant": "original_native",
                "width_mm": 2,
                "valid": True,
                "peak_abs_af_v_per_a_m2": 0.0,
            }],
        })
    vectors, exclusions = build_vectors(documents)
    assert not vectors
    assert exclusions == [{
        "level": "vector",
        "variant": "original_native",
        "offset": 0,
        "geometry": "g1",
        "fit": "L",
        "scenario": "baseline",
        "side": "L",
        "width_mm": 2,
        "reason": "nonpositive_seven_segment_sum",
    }]


def test_profile_file_contains_no_xyz_and_is_resume_bound(tmp_path):
    s = np.linspace(0.0, 0.050, 11)
    profiles = normalize_profiles(np.vstack((s, 2*s + 4, -s + 8, s**2)))
    context = {"index": 3, "subject": "g", "fit": "L", "segment": "L4", "side": "R"}
    scenario = {"scenario_id": "baseline", "displacement_mm": 0.0, "phi_deg": None}
    payload = _profile_payload(
        binding_sha256="a" * 64,
        context=context,
        scenario=scenario,
        offsets=(0, 10, 20, 100),
        s_m=s,
        normalized_profiles=profiles,
    )
    target = tmp_path / "profile.npz"
    _write_profile_atomic(target, payload)
    _write_profile_atomic(target, payload)  # same binding/content is resumable
    with np.load(target, allow_pickle=False) as stored:
        assert "xyz" not in " ".join(stored.files).lower()
        assert_array_equal(stored["s_m"], s)
        assert stored["potential_v_per_a"].shape == (4, len(s))
        assert_array_equal(stored["potential_v_per_a"][:, 0], np.zeros(4))
        assert str(stored["binding_sha256"]) == "a" * 64
    changed = dict(payload)
    changed["binding_sha256"] = np.asarray("b" * 64)
    with pytest.raises(ValueError, match="content differs"):
        _write_profile_atomic(target, changed)


def test_profile_name_accepts_actual_decimal_displacement_but_rejects_traversal():
    assert _profile_name("d1.5mm_phi45deg") == "d1.5mm_phi45deg.npz"
    for unsafe in ("../escape", ".hidden", "a/b"):
        with pytest.raises(ValueError, match="Unsafe"):
            _profile_name(unsafe)


def test_completed_path_resume_recomputes_descriptors_from_saved_profiles(tmp_path):
    output = tmp_path / "output"
    binding = "c" * 64
    context = {
        "index": 0,
        "subject": "g",
        "fit": "L",
        "segment": "L4",
        "side": "R",
    }
    offsets = (0, 10, 20, 100)
    s = np.linspace(0.0, 0.050, 401)
    profiles = normalize_profiles(np.vstack([
        np.sin(40.0 * s),
        np.sin(50.0 * s),
        np.cos(60.0 * s),
        s**2,
    ]))
    profile_index = []
    results = []
    for index in range(15):
        scenario = {
            "scenario_id": f"scenario_{index:02d}",
            "displacement_mm": index / 10.0,
            "phi_deg": float(index),
        }
        relative = Path("profiles/000") / f"{scenario['scenario_id']}.npz"
        target = output / relative
        _write_profile_atomic(
            target,
            _profile_payload(
                binding_sha256=binding,
                context=context,
                scenario=scenario,
                offsets=offsets,
                s_m=s,
                normalized_profiles=profiles,
            ),
        )
        profile_index.append({
            "scenario": scenario["scenario_id"],
            "path": relative.as_posix(),
            "sha256": digest(target),
            "sample_count": len(s),
            "profile_length_m": float(s[-1] - s[0]),
            "common_endpoint_m": common_endpoint_m(float(s[-1] - s[0])),
            "locate": {
                "n_queries": len(s),
                "n_failed": 0,
                "fallback_queries": 0,
                "fallback_hits": 0,
            },
        })
        for field_index, offset in enumerate(offsets):
            analysis = analyze_profile_variants(s, profiles[field_index])
            for variant in VARIANTS:
                for width in (2, 5, 10):
                    results.append({
                        "scenario": scenario["scenario_id"],
                        "d_mm": scenario["displacement_mm"],
                        "phi_deg": scenario["phi_deg"],
                        "offset": offset,
                        "variant": variant,
                        "width_mm": width,
                        **_scalar_descriptor(
                            analysis["variants"][variant][str(width)]
                        ),
                    })
    document = {
        "schema": "common-support-path-v1",
        "binding_sha256": binding,
        "context": context,
        "profiles": profile_index,
        "results": results,
    }
    path = output / "paths/000.json"
    write_json_atomic(path, document)
    _validate_completed_path(path, output, binding, expected_context=context)

    document["results"][0]["peak_abs_af_v_per_a_m2"] += 1.0
    write_json_atomic(path, document)
    with pytest.raises(ValueError, match="differs from its saved profile"):
        _validate_completed_path(path, output, binding, expected_context=context)


def test_output_inside_frozen_cache_is_rejected(tmp_path):
    frozen = tmp_path / "frozen"
    frozen.mkdir()
    with pytest.raises(ValueError, match="outside"):
        validate_output_location(frozen, frozen)
    with pytest.raises(ValueError, match="outside"):
        validate_output_location(frozen, frozen / "new-results")
    resolved_frozen, resolved_output = validate_output_location(frozen, tmp_path / "results")
    assert resolved_frozen == frozen.resolve()
    assert resolved_output == (tmp_path / "results").resolve()
