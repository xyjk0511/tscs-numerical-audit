"""Pure profile analysis for the common-physical-support experiment.

The functions in this module operate only on arclength and scalar potential
profiles.  They do not import anatomy, mesh, locator, or field-loading code.
Distances are metres; potentials are V/A; AF peaks are V/(A*m**2).
"""
from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import numpy as np

from analysis.no_new_mri_functional_calibration.path_numerics_v2 import (
    complete_bin_potential,
)

WIDTHS_M = (0.002, 0.005, 0.010)
MIN_COMPLETE_BINS = 5
COMMON_ENDPOINT_QUANTUM_M = 0.010
CENTRAL_MARGIN_M = 0.015  # 1.5 * max(WIDTHS_M): full three-bin AF support.
VARIANTS = (
    "original_native",
    "common_endpoint",
    "common_endpoint_central",
)

_EPS = np.finfo(np.float64).eps
_GRID_ULPS = 8.0


def _positive_scalar(value: Any, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a finite positive scalar") from exc
    if not math.isfinite(result) or result <= 0:
        raise ValueError(f"{name} must be a finite positive scalar")
    return result


def _profile(arclength_m: Any, potential_v_per_a: Any) -> tuple[np.ndarray, np.ndarray]:
    raw_s = np.asarray(arclength_m)
    if np.iscomplexobj(raw_s):
        raise ValueError("arclength_m must be real")
    s = np.asarray(raw_s, dtype=np.float64)
    v = np.asarray(potential_v_per_a)
    if np.iscomplexobj(v):
        raise ValueError("potential_v_per_a must be real")
    v = np.asarray(v, dtype=np.float64)
    if s.ndim != 1 or len(s) < 2 or v.shape != s.shape:
        raise ValueError("arclength_m and potential_v_per_a must be matching 1D arrays")
    if not np.isfinite(s).all() or not np.isfinite(v).all() or np.any(np.diff(s) <= 0):
        raise ValueError("profile must be finite with strictly increasing arclength")
    return s, v


def _complete_interval_count(length_m: float, width_m: float) -> int:
    """Match path_numerics_v2's eight-epsilon complete-interval rule."""
    length = _positive_scalar(length_m, "length_m")
    width = _positive_scalar(width_m, "width_m")
    quotient = length / width
    if not math.isfinite(quotient) or quotient >= np.iinfo(np.intp).max - 1:
        raise ValueError("length/width requests an unrepresentable interval count")
    count = int(math.floor(quotient))
    near = count + 1
    edge = near * width
    if abs(edge - length) <= _GRID_ULPS * _EPS * max(length, edge):
        count = near
    return count


def common_endpoint_m(length_m: float, quantum_m: float = COMMON_ENDPOINT_QUANTUM_M) -> float:
    """Largest complete quantum endpoint, without rounding a true short bin up.

    A length infinitesimally below an exact multiple because of binary endpoint
    representation is treated like the existing numerical core: the interval
    count may advance only inside ``8*eps*scale``.  The returned physical endpoint
    is still capped at the available length, so this function never extrapolates.
    """
    length = _positive_scalar(length_m, "length_m")
    quantum = _positive_scalar(quantum_m, "quantum_m")
    count = _complete_interval_count(length, quantum)
    return min(length, count * quantum)


def normalize_profiles(potential_v_per_a: Any) -> np.ndarray:
    """Subtract each field's first value; accept one profile or fields x samples."""
    value = np.asarray(potential_v_per_a)
    if np.iscomplexobj(value):
        raise ValueError("potential_v_per_a must be real")
    value = np.asarray(value, dtype=np.float64)
    if value.ndim not in (1, 2) or value.shape[-1] < 2:
        raise ValueError("potential_v_per_a must have shape (n>=2,) or (fields,n>=2)")
    if value.ndim == 2 and value.shape[0] == 0:
        raise ValueError("potential_v_per_a must contain at least one field")
    if not np.isfinite(value).all():
        raise ValueError("potential_v_per_a must be finite")
    with np.errstate(over="ignore", invalid="ignore"):
        normalized = value - value[..., :1]
    if not np.isfinite(normalized).all():
        raise ValueError("gauge normalization overflowed")
    return normalized


def _truncate_profile(
    arclength_m: np.ndarray,
    potential_v_per_a: np.ndarray,
    relative_endpoint_m: float,
) -> tuple[np.ndarray, np.ndarray]:
    length = float(arclength_m[-1] - arclength_m[0])
    endpoint = float(relative_endpoint_m)
    if not math.isfinite(endpoint) or endpoint <= 0 or endpoint > length:
        raise ValueError("relative_endpoint_m must lie in (0, profile length]")
    target = float(arclength_m[0] + endpoint)
    scale = max(abs(target), abs(float(arclength_m[-1])), length)
    tolerance = _GRID_ULPS * _EPS * scale
    if abs(target - arclength_m[-1]) <= tolerance:
        return arclength_m, potential_v_per_a
    stop = int(np.searchsorted(arclength_m, target, side="left"))
    if stop < len(arclength_m) and abs(float(arclength_m[stop]) - target) <= tolerance:
        return arclength_m[: stop + 1], potential_v_per_a[: stop + 1]
    if stop == 0 or stop >= len(arclength_m):
        raise ValueError("common endpoint is outside the represented profile")
    endpoint_value = float(np.interp(target, arclength_m, potential_v_per_a))
    return (
        np.r_[arclength_m[:stop], target],
        np.r_[potential_v_per_a[:stop], endpoint_value],
    )


def af_descriptor(
    arclength_m: Any,
    potential_v_per_a: Any,
    width_m: float,
    *,
    endpoint_m: float | None = None,
    search_interval_m: tuple[float, float] | None = None,
    min_complete_bins: int = MIN_COMPLETE_BINS,
) -> dict[str, Any]:
    """Compute one finite-bin AF descriptor and expose its native centre lattice.

    ``endpoint_m`` is a distance from the profile origin.  Five complete bins are
    required before any search-mask logic, preserving the original descriptor
    contract.  ``search_interval_m`` is also relative to the profile origin and
    is inclusive up to a roundoff-only tolerance.  No cross-width interpolation
    of AF values or centres is performed.
    """
    s, raw = _profile(arclength_m, potential_v_per_a)
    width = _positive_scalar(width_m, "width_m")
    if isinstance(min_complete_bins, bool) or int(min_complete_bins) != min_complete_bins:
        raise ValueError("min_complete_bins must be a positive integer")
    minimum = int(min_complete_bins)
    if minimum <= 0:
        raise ValueError("min_complete_bins must be a positive integer")
    value = normalize_profiles(raw)
    full_length = float(s[-1] - s[0])
    working_s, working_v = s, value
    if endpoint_m is not None:
        endpoint = float(endpoint_m)
        if not math.isfinite(endpoint) or endpoint < 0 or endpoint > full_length:
            raise ValueError("endpoint_m must lie in [0, profile length]")
        if endpoint == 0:
            return {
                "valid": False,
                "exclusion": "fewer_than_five_complete_bins",
                "width_m": width,
                "n_bins": 0,
                "full_length_m": full_length,
                "analysis_length_m": 0.0,
                "retained_endpoint_m": 0.0,
                "discarded_from_full_length_m": full_length,
                "bin_centers_m": np.empty(0, dtype=np.float64),
                "af_centers_m": np.empty(0, dtype=np.float64),
                "candidate_centers_m": np.empty(0, dtype=np.float64),
                "af_v_per_a_m2": np.empty(0, dtype=np.float64),
                "candidate_mask": np.empty(0, dtype=bool),
                "candidate_count": 0,
                "peak_abs_af_v_per_a_m2": None,
            }
        working_s, working_v = _truncate_profile(s, value, endpoint)
    working_length = float(working_s[-1] - working_s[0])
    centers, means = complete_bin_potential(working_s, working_v, width)
    n_bins = int(len(centers))
    retained_endpoint = min(working_length, n_bins * width)
    base = {
        "valid": False,
        "exclusion": None,
        "width_m": width,
        "n_bins": n_bins,
        "full_length_m": full_length,
        "analysis_length_m": working_length,
        "retained_endpoint_m": retained_endpoint,
        "discarded_from_full_length_m": max(0.0, full_length - retained_endpoint),
        "bin_centers_m": centers - s[0],
        "af_centers_m": np.empty(0, dtype=np.float64),
        "candidate_centers_m": np.empty(0, dtype=np.float64),
        "af_v_per_a_m2": np.empty(0, dtype=np.float64),
        "candidate_mask": np.empty(0, dtype=bool),
        "candidate_count": 0,
        "peak_abs_af_v_per_a_m2": None,
    }
    if n_bins < minimum:
        base["exclusion"] = "fewer_than_five_complete_bins"
        return base
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        af = np.diff(means, n=2) / width**2
    if not np.isfinite(af).all():
        raise ValueError("AF calculation overflowed")
    af_centers = centers[1:-1] - s[0]
    mask = np.ones(len(af), dtype=bool)
    if search_interval_m is not None:
        lower, upper = map(float, search_interval_m)
        if not math.isfinite(lower) or not math.isfinite(upper):
            raise ValueError("search_interval_m must be finite")
        scale = max(abs(lower), abs(upper), working_length, width)
        tolerance = _GRID_ULPS * _EPS * scale
        mask = (af_centers >= lower - tolerance) & (af_centers <= upper + tolerance)
    base.update({
        "af_centers_m": af_centers,
        "af_v_per_a_m2": af,
        "candidate_mask": mask,
        "candidate_centers_m": af_centers[mask],
        "candidate_count": int(np.count_nonzero(mask)),
    })
    if not np.any(mask):
        base["exclusion"] = "no_af_center_in_search_interval"
        return base
    base["valid"] = True
    base["peak_abs_af_v_per_a_m2"] = float(np.max(np.abs(af[mask])))
    return base


def analyze_profile_variants(
    arclength_m: Any,
    potential_v_per_a: Any,
    widths_m: Sequence[float] = WIDTHS_M,
) -> dict[str, Any]:
    """Evaluate native, common-endpoint, and common-plus-central variants."""
    s, value = _profile(arclength_m, potential_v_per_a)
    widths = tuple(_positive_scalar(width, "widths_m entry") for width in widths_m)
    if not widths or len(set(widths)) != len(widths):
        raise ValueError("widths_m must contain unique positive widths")
    length = float(s[-1] - s[0])
    common = common_endpoint_m(length)
    variants: dict[str, dict[str, dict[str, Any]]] = {name: {} for name in VARIANTS}
    for width in widths:
        key = f"{1000.0 * width:g}"
        variants["original_native"][key] = af_descriptor(s, value, width)
        variants["common_endpoint"][key] = af_descriptor(
            s, value, width, endpoint_m=common
        )
        variants["common_endpoint_central"][key] = af_descriptor(
            s,
            value,
            width,
            endpoint_m=common,
            search_interval_m=(CENTRAL_MARGIN_M, common - CENTRAL_MARGIN_M),
        )
    return {
        "profile_length_m": length,
        "common_endpoint_m": common,
        "minimum_complete_bins": MIN_COMPLETE_BINS,
        "central_interval_m": [CENTRAL_MARGIN_M, common - CENTRAL_MARGIN_M],
        "variants": variants,
    }
