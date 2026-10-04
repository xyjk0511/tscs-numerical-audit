"""Independent root-path post-processing; no FEM solve or historical-file writes.

Distances are metres, input potentials V/A, longitudinal E V/(A*m), and AF
V/(A*m**2). AF is +d2V/ds2, matching the old scalar convention; E is -dV/ds.

Typical caller (the research runner, NOT wired into any old module here)::

    sampler = TetraP1Sampler(local_points_m, local_tets)
    xyz, s = resample_polyline(fixed_original_path_m, 0.00025)
    cells, bary, diagnostic = sampler.locate(xyz)
    if diagnostic["n_failed"]:
        raise ValueError("Reject this path: incomplete field coverage")
    values = sampler.sample(nodal_potential, cells, bary)
    # field_metrics_v2 rejects NaN/Inf, including failed field samples.
    field = field_metrics_v2(s, values)  # one field; loop rows for multiple fields

Always re-sample the ORIGINAL polyline when changing sampling spacing. Chord
lengths of returned points need not recover its arclength at corners. Changing
bin width changes the physical smoothing/stencil, NOT just numerical resolution.
Neither these functions nor their synthetic tests establish anatomical validity,
FEM mesh convergence, physiological recruitment, or human probabilities.
"""

from __future__ import annotations

import math
import operator
from collections.abc import Sequence
from typing import Any

import numpy as np
from scipy.spatial import cKDTree

__all__ = [
    "resample_polyline",
    "complete_bin_potential",
    "field_metrics_v2",
    "TetraP1Sampler",
]

_EPS = np.finfo(np.float64).eps
_GRID_ULPS = 8.0
_BARY_TOL = 1e-10  # dimensionless boundary tolerance; never used to fill failures
_GEOMETRY_CHUNK = 4096
_CANDIDATE_CHUNK = 4096
_MAX_NEAREST = 128
_MAX_QUERY_PAIRS = 32768


def _real_array(value: Any, name: str, *, finite: bool = True) -> np.ndarray:
    arr = np.asarray(value)
    if np.iscomplexobj(arr):
        raise ValueError(f"{name} must be real")
    try:
        arr = np.asarray(arr, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a real numeric array") from exc
    if finite and not np.isfinite(arr).all():
        raise ValueError(f"{name} must be finite")
    return arr


def _positive_float(value: Any, name: str) -> float:
    arr = _real_array(value, name)
    if arr.ndim != 0 or arr <= 0:
        raise ValueError(f"{name} must be a finite positive scalar")
    return float(arr)


def _positive_int(value: Any, name: str) -> int:
    try:
        result = operator.index(value)
    except TypeError as exc:
        raise ValueError(f"{name} must be a positive integer") from exc
    if isinstance(value, (bool, np.bool_)) or result <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return result


def _grid_count(length: float, width: float) -> int:
    """Count complete intervals, allowing only an 8-epsilon endpoint roundoff.

    A true short interval is never rounded up by decimal precision or a fraction
    of a bin. A terminal discrepancy <= 8*eps*max(length, n*width) is regarded as
    floating-point boundary representation, and integrations stop at real data.
    """
    quotient = length / width
    if not math.isfinite(quotient) or quotient >= np.iinfo(np.intp).max - 1:
        raise ValueError("spacing/width requests an unrepresentable number of intervals")
    n = int(math.floor(quotient))
    near = n + 1
    edge = near * width
    if near > 0 and abs(edge - length) <= _GRID_ULPS * _EPS * max(length, edge):
        n = near
    return n


def resample_polyline(path_xyz_m: Any, spacing_m: float) -> tuple[np.ndarray, np.ndarray]:
    """Sample an open, finite, non-repeating original polyline by arclength.

    Returns float64 ``(xyz[n,3], s[n])``, both in metres; ``s[0] == 0``.
    Interior steps are spacing_m; the true endpoint is retained once, so the
    final step may be shorter. A path shorter than spacing returns both ends.
    Repeated vertices (including a repeated closing vertex), zero segments and
    numerically unrepresentable segment lengths are rejected, not repaired.
    Self-intersections between distinct vertices are not a topology check here.
    """
    path = _real_array(path_xyz_m, "path_xyz_m")
    spacing = _positive_float(spacing_m, "spacing_m")
    if path.ndim != 2 or path.shape[1] != 3 or len(path) < 2:
        raise ValueError("path_xyz_m must have shape (n>=2, 3)")
    if len(np.unique(path, axis=0)) != len(path):
        raise ValueError("path_xyz_m contains repeated vertices")
    with np.errstate(over="ignore", invalid="ignore"):
        lengths = np.hypot.reduce(np.diff(path, axis=0), axis=1)
        source_s = np.r_[0.0, np.cumsum(lengths)]
    if not np.isfinite(source_s).all() or np.any(np.diff(source_s) <= 0):
        raise ValueError("path contains zero or unrepresentable segment lengths")
    total = float(source_s[-1])
    n = _grid_count(total, spacing)
    targets = np.arange(n + 1, dtype=np.float64) * spacing
    # Do not emit a duplicate/microscopic end interval due solely to rounding.
    tol = _GRID_ULPS * _EPS * max(total, float(targets[-1]))
    if n and abs(targets[-1] - total) <= tol:
        targets[-1] = total
    else:
        targets = np.r_[targets, total]
    if not np.isfinite(targets).all() or np.any(np.diff(targets) <= 0):
        raise ValueError("requested sample spacing is not representable")
    samples = np.column_stack([np.interp(targets, source_s, path[:, k]) for k in range(3)])
    samples[0], samples[-1] = path[0], path[-1]
    return samples, targets


def _path_values(arclength_m: Any, potential: Any) -> tuple[np.ndarray, np.ndarray]:
    s = _real_array(arclength_m, "arclength_m")
    v = _real_array(potential, "potential")
    if s.ndim != 1 or len(s) < 2:
        raise ValueError("arclength_m must have shape (n>=2,)")
    with np.errstate(over="ignore", invalid="ignore"):
        delta = np.diff(s)
        length = s[-1] - s[0]
    if not np.isfinite(delta).all() or not np.isfinite(length) or np.any(delta <= 0):
        raise ValueError("arclength_m must be strictly increasing with finite length")
    if v.ndim not in (1, 2) or v.shape[-1] != len(s):
        raise ValueError("potential must have shape (n_samples,) or (n_fields,n_samples)")
    if v.ndim == 2 and v.shape[0] == 0:
        raise ValueError("potential must contain at least one field")
    return s, v


def complete_bin_potential(
    arclength_m: Any, potential: Any, width_m: float
) -> tuple[np.ndarray, np.ndarray]:
    """Exact trapezoidal integral of the represented piecewise-linear V(s).

    Bins start at s[0], NOT at a global origin. Only complete width_m intervals
    are retained; no point-count averaging and no extrapolation into a short
    tail. Endpoint roundoff is handled as described in _grid_count. Integration
    splits at every sample and bin boundary, so nonuniform sampling and a short
    final sample step are valid. This is exact for the interpolant, not generally
    for an unknown continuous field between samples or unobserved P1 crossings.

    Returns ``centers_m[n_bins]`` and means of shape ``(n_bins,)`` or
    ``(n_fields,n_bins)``. Zero through four bins are allowed by this primitive.
    Temporary integration arrays are one field at a time, O(n_samples+n_bins).
    """
    s, v = _path_values(arclength_m, potential)
    width = _positive_float(width_m, "width_m")
    relative = s - s[0]
    n = _grid_count(float(relative[-1]), width)
    centers = s[0] + (np.arange(n, dtype=np.float64) + 0.5) * width
    result = np.empty(v.shape[:-1] + (n,), dtype=np.float64)
    if not n:
        return centers, result
    edges = np.arange(n + 1, dtype=np.float64) * width
    # Only a roundoff-sized overshoot is possible after _grid_count.
    edges[-1] = min(edges[-1], relative[-1])
    if np.any(np.diff(edges) <= 0) or (n > 1 and np.any(np.diff(centers) <= 0)):
        raise ValueError("bin grid is not representable at this arclength offset")
    knots = np.union1d(relative[(relative > 0) & (relative < edges[-1])], edges)
    starts = np.searchsorted(knots, edges[:-1])
    ds = np.diff(knots)
    bin_lengths = np.diff(edges)
    rows, out = np.atleast_2d(v), np.atleast_2d(result)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        for row, target in zip(rows, out, strict=True):
            baseline = row[0]
            vals = np.interp(knots, relative, row - baseline)
            area = (0.5 * vals[:-1] + 0.5 * vals[1:]) * ds
            target[:] = np.add.reduceat(area, starts) / bin_lengths + baseline
    if not np.isfinite(result).all():
        raise ValueError("bin integration overflowed; potential scale is unsupported")
    return centers, result


def field_metrics_v2(
    arclength_m: Any, potential: Any, widths_mm: Sequence[float] = (2.0, 5.0, 10.0)
) -> dict[str, Any]:
    """One V/A field -> old-compatible ``field['by_bin_mm']`` scalar metrics.

    Fewer than five complete bins give None, not zero, at that width. Diagnostics
    remain available under diagnostics_by_bin_mm even when a metric is None.
    AF = diff(diff(bin_means))/width**2; E = -diff(bin_means)/width. Peaks and
    medians are over the absolute derivative arrays. No small AF is forced to
    zero. A constant potential reference is removed before integration to avoid
    amplifying a large, physically irrelevant DC offset through differencing.

    Translation of s leaves metrics unchanged. Reversal changes E's sign; AF
    sign is unchanged on a reversed, matching bin support. If a tail is dropped,
    reversing the path drops the opposite physical tail: exact reversal
    invariance is NOT promised for different retained support/bin alignment.
    """
    s, v = _path_values(arclength_m, potential)
    if v.ndim != 1:
        raise ValueError("field_metrics_v2 requires one field of shape (n_samples,)")
    widths = [_positive_float(w, "widths_mm entry") for w in widths_mm]
    if not widths:
        raise ValueError("widths_mm must not be empty")
    keys = [f"{w:g}" for w in widths]
    if len(set(keys)) != len(keys):
        raise ValueError("widths_mm contains duplicate/ambiguous formatted keys")
    by_bin: dict[str, Any] = {}
    diagnostics: dict[str, Any] = {}
    with np.errstate(over="ignore", invalid="ignore"):
        referenced = v - v[0]
    for width_mm, key in zip(widths, keys, strict=True):
        width = width_mm / 1000.0
        centers, means = complete_bin_potential(s, referenced, width)
        n = len(centers)
        tail = max(0.0, float(s[-1] - s[0]) - n * width)
        diag = {"n_bins": n, "discarded_tail_m": tail, "width_m": width}
        diagnostics[key] = diag
        if n < 5:
            by_bin[key] = None
            continue
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            longitudinal_e = -np.diff(means) / width
            af = -np.diff(longitudinal_e) / width
        if not np.isfinite(af).all() or not np.isfinite(longitudinal_e).all():
            raise ValueError("derivative overflow; potential/width scale is unsupported")
        metrics: dict[str, Any] = {"bins": n, **diag}
        for name, values, unit in (
            ("af", af, "v_per_a_m2"),
            ("longitudinal_e", longitudinal_e, "v_per_a_m"),
        ):
            magnitude = np.abs(values)
            peak = float(magnitude.max())
            metrics[f"peak_abs_{name}_{unit}"] = peak
            metrics[f"median_abs_{name}_{unit}"] = (
                float(np.median(magnitude / peak) * peak) if peak else 0.0
            )
        for stat in ("peak", "median"):
            metrics[f"{stat}_abs_activating_function_v_per_a_m2"] = metrics[
                f"{stat}_abs_af_v_per_a_m2"
            ]
            metrics[f"{stat}_abs_e_v_per_a_m"] = metrics[
                f"{stat}_abs_longitudinal_e_v_per_a_m"
            ]
        e_scale = float(np.max(np.abs(longitudinal_e)))
        metrics["mean_longitudinal_e_v_per_a_m"] = (
            float(np.mean(longitudinal_e / e_scale) * e_scale) if e_scale else 0.0
        )
        by_bin[key] = metrics
    return {"by_bin_mm": by_bin, "diagnostics_by_bin_mm": diagnostics}


class TetraP1Sampler:
    """Reusable P1 locator on a caller-supplied LOCAL, conforming tetrahedral mesh.

    Initial centroid KD-tree candidates are an acceleration only. Every finite
    query not found there gets an exact (eps=0) radius search around its centroid
    neighborhood. For centroid c and radius R=max_i ||vertex_i-c||, convexity
    implies ||x-c||<=R for every point in that tetrahedron. A global max(R) search
    therefore contains every possible containing cell, even very long/thin ones.
    Radii are padded for the declared barycentric/float64 boundary tolerance.

    That fallback is one query at a time; its list can contain ALL local cells.
    It is then checked in bounded cell batches, never queries x all_tets. Worst
    case time is O(n_queries*n_tets), not a hidden fixed-k failure. Default
    nearest batches have <=32768 query/candidate pairs, and k is capped at 128.
    Geometry construction uses 4096-cell batches. Containment uses batched
    3x3 solves, not inverse-matrix dot products (less cancellation at thin faces).
    Stored arrays cost 160 bytes/tet (int64/float64), plus KD-tree storage; input
    arrays, conversions and returned arrays are additional. No nodal field is
    retained or converted in full by sample().

    Exact singular or float64-unresolvable cells raise ValueError at construction
    rather than being silently omitted. Barycentric acceptance is [-1e-10,
    1+1e-10], sum=1; weights are NOT clipped, so affine fields are not modified.
    This is tolerance-aware numerical containment, not exact-arithmetic geometry.
    A failed locate is not a proof that a root lies outside an anatomical domain.
    Shared faces in a conforming P1 mesh have the same interpolated potential;
    selected cell identity is repeatable for the same mesh/configuration, but
    need not be invariant to connectivity order or candidate_count. Overlapping
    cells, nonconforming interfaces and anatomical labels are caller concerns.
    """

    def __init__(
        self, points_xyz_m: Any, tetrahedra: Any, *,
        candidate_count: int = 32, query_chunk_size: int = 256,
    ) -> None:
        requested_k = _positive_int(candidate_count, "candidate_count")
        requested_chunk = _positive_int(query_chunk_size, "query_chunk_size")
        points = _real_array(points_xyz_m, "points_xyz_m")
        if points.ndim != 2 or points.shape[1] != 3 or len(points) < 4:
            raise ValueError("points_xyz_m must have shape (n_nodes>=4,3)")
        cells = np.asarray(tetrahedra)
        if cells.ndim != 2 or cells.shape[1] != 4 or not len(cells):
            raise ValueError("tetrahedra must have shape (n_tets>=1,4)")
        if cells.dtype.kind not in "iu":
            raise ValueError("tetrahedra must contain integer node indices")
        if cells.min() < 0 or cells.max() >= len(points):
            raise ValueError("tetrahedra node index out of bounds")
        self.n_nodes = len(points)
        self.n_tets = len(cells)
        self.candidate_count = min(requested_k, self.n_tets, _MAX_NEAREST)
        self.query_chunk_size = min(
            requested_chunk, max(1, _MAX_QUERY_PAIRS // self.candidate_count)
        )
        self._requested_k = requested_k
        self._tetrahedra = np.array(cells, dtype=np.int64, copy=True)
        self._origins = np.empty((self.n_tets, 3), dtype=np.float64)
        self._edges = np.empty((self.n_tets, 3, 3), dtype=np.float64)
        self._centroids = np.empty((self.n_tets, 3), dtype=np.float64)
        self._radii = np.empty(self.n_tets, dtype=np.float64)
        self._max_condition = 0.0
        for start in range(0, self.n_tets, _GEOMETRY_CHUNK):
            stop = min(start + _GEOMETRY_CHUNK, self.n_tets)
            vertices = points[self._tetrahedra[start:stop]]
            with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                edges = np.swapaxes(vertices[:, 1:] - vertices[:, :1], 1, 2)
                # Column equilibration avoids misclassifying orthogonal but
                # unequal edge lengths as a numerically singular system.
                scale = np.max(np.abs(edges), axis=1)
                if not np.isfinite(edges).all() or np.any(scale == 0):
                    raise ValueError(f"degenerate/unrepresentable tetrahedron in [{start},{stop})")
                normalized = edges / scale[:, None, :]
                sv = np.linalg.svd(normalized, compute_uv=False)
                bad = sv[:, -1] <= 64 * _EPS * sv[:, 0]
                if np.any(bad):
                    cell = start + int(np.flatnonzero(bad)[0])
                    raise ValueError(f"degenerate/numerically singular tetrahedron {cell}")
                condition = sv[:, 0] / sv[:, -1]
                centroid = (
                    vertices[:, 0]
                    + np.mean(vertices[:, 1:] - vertices[:, :1], axis=1) * 0.75
                )
                radius = np.hypot.reduce(vertices - centroid[:, None, :], axis=2).max(axis=1)
                padding = 64 * _EPS * np.max(np.abs(vertices), axis=(1, 2))
                radius = np.nextafter(radius * (1 + 8 * _BARY_TOL) + padding, np.inf)
            if not np.isfinite(centroid).all() or not np.isfinite(radius).all():
                raise ValueError("tetrahedral scale overflows float64")
            self._origins[start:stop] = vertices[:, 0]
            self._edges[start:stop] = edges
            self._centroids[start:stop] = centroid
            self._radii[start:stop] = radius
            self._max_condition = max(self._max_condition, float(condition.max()))
        self._max_radius = float(self._radii.max())
        self._tree = cKDTree(self._centroids, copy_data=False)
        for arr in (
            self._tetrahedra, self._origins, self._edges, self._centroids, self._radii
        ):
            arr.flags.writeable = False

    @property
    def array_storage_bytes(self) -> int:
        """Owned geometry array bytes only, excluding SciPy KD-tree internals."""
        return sum(arr.nbytes for arr in (
            self._tetrahedra, self._origins, self._edges, self._centroids, self._radii
        ))

    def _in_candidates(
        self, query: np.ndarray, candidates: np.ndarray, diagnostic: dict[str, Any]
    ) -> tuple[int, np.ndarray | None]:
        query_pad = 64 * _EPS * float(np.max(np.abs(query)))
        for start in range(0, len(candidates), _CANDIDATE_CHUNK):
            ids = candidates[start:start + _CANDIDATE_CHUNK]
            diagnostic["max_candidate_batch"] = max(diagnostic["max_candidate_batch"], len(ids))
            with np.errstate(over="ignore", invalid="ignore"):
                distance = np.hypot.reduce(query - self._centroids[ids], axis=1)
                ids = ids[distance <= self._radii[ids] + query_pad]
                if not len(ids):
                    continue
                tail = np.linalg.solve(
                    self._edges[ids], (query - self._origins[ids])[:, :, None]
                )[:, :, 0]
                bary = np.column_stack((1.0 - tail.sum(axis=1), tail))
            diagnostic["candidate_tests"] += len(ids)
            inside = np.isfinite(bary).all(axis=1) & (bary >= -_BARY_TOL).all(axis=1)
            inside &= (bary <= 1 + _BARY_TOL).all(axis=1)
            hit = np.flatnonzero(inside)
            if len(hit):
                first = int(hit[0])
                return int(ids[first]), bary[first]
        return -1, None

    def locate(self, queries_xyz_m: Any) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        """Return cell indices[n], weights[n,4], and JSON-serializable diagnostics.

        Nonfinite query rows fail individually; all unlocated rows are -1/NaN.
        Caller must block incomplete paths; no nearest-node/cell extrapolation.
        """
        queries = _real_array(queries_xyz_m, "queries_xyz_m", finite=False)
        if queries.ndim != 2 or queries.shape[1] != 3:
            raise ValueError("queries_xyz_m must have shape (n_queries,3)")
        indices = np.full(len(queries), -1, dtype=np.int64)
        weights = np.full((len(queries), 4), np.nan, dtype=np.float64)
        finite = np.isfinite(queries).all(axis=1)
        diagnostic: dict[str, Any] = {
            "n_queries": len(queries), "n_failed": 0, "n_nonfinite_queries": int((~finite).sum()),
            "n_unlocated_finite": 0, "initial_hits": 0, "fallback_queries": 0,
            "fallback_hits": 0, "candidate_tests": 0, "max_candidate_batch": 0,
            "max_radius_candidates": 0, "max_nearest_query_pairs": 0,
            "candidate_count_requested": self._requested_k,
            "candidate_count_effective": self.candidate_count,
            "query_chunk_size_effective": self.query_chunk_size,
            "barycentric_tolerance": _BARY_TOL, "coordinate_padding_eps_factor": 64,
            "max_padded_radius_m": self._max_radius,
            "max_column_scaled_condition_number": self._max_condition,
            "geometry_array_bytes_excluding_kdtree": self.array_storage_bytes,
            "search": "centroid_knn_then_complete_radius_eps0",
        }
        for start in range(0, len(queries), self.query_chunk_size):
            stop = min(start + self.query_chunk_size, len(queries))
            rows = np.flatnonzero(finite[start:stop]) + start
            if not len(rows):
                continue
            _, nearest = self._tree.query(queries[rows], k=self.candidate_count, eps=0, workers=1)
            nearest = np.asarray(nearest).reshape(len(rows), self.candidate_count)
            diagnostic["max_nearest_query_pairs"] = max(
                diagnostic["max_nearest_query_pairs"], int(nearest.size)
            )
            for row, candidate_ids in zip(rows, nearest, strict=True):
                cell, bary = self._in_candidates(queries[row], np.sort(candidate_ids), diagnostic)
                if cell >= 0:
                    diagnostic["initial_hits"] += 1
                else:
                    diagnostic["fallback_queries"] += 1
                    pad = 64 * _EPS * float(np.max(np.abs(queries[row])))
                    radius = np.nextafter(self._max_radius + pad, np.inf)
                    # Single-query list only: never an object array of all QxT candidates.
                    candidates = np.asarray(self._tree.query_ball_point(
                        queries[row], radius, eps=0, workers=1, return_sorted=True
                    ), dtype=np.int64)
                    diagnostic["max_radius_candidates"] = max(
                        diagnostic["max_radius_candidates"], len(candidates)
                    )
                    cell, bary = self._in_candidates(queries[row], candidates, diagnostic)
                    if cell >= 0:
                        diagnostic["fallback_hits"] += 1
                if cell >= 0:
                    indices[row], weights[row] = cell, bary
        diagnostic["n_failed"] = int(np.count_nonzero(indices < 0))
        diagnostic["n_unlocated_finite"] = (
            diagnostic["n_failed"] - diagnostic["n_nonfinite_queries"]
        )
        return indices, weights, diagnostic

    def sample(self, nodal_potential: Any, indices: Any, weights: Any) -> np.ndarray:
        """P1 interpolation of (n_nodes,) or (n_fields,n_nodes), field by field.

        Failed locations produce NaN. Any referenced nonfinite nodal value makes
        that field/sample NaN, even at a zero-weight vertex; unused nodes are not
        read or validated. Malformed indices/weights raise rather than fabricate
        samples. A float32/float64 array or memmap is not copied into a full
        float64 field; gathering/conversion is limited to one field/query chunk.
        """
        potential = np.asarray(nodal_potential)
        if potential.dtype.kind not in "biuf" or potential.ndim not in (1, 2):
            raise ValueError("nodal_potential must be a real numeric 1D or 2D array")
        if potential.shape[-1] != self.n_nodes or (potential.ndim == 2 and not len(potential)):
            raise ValueError("nodal_potential node dimension does not match the mesh")
        ids = np.asarray(indices)
        bary = _real_array(weights, "weights", finite=False)
        if ids.ndim != 1 or ids.dtype.kind not in "iu" or bary.shape != (len(ids), 4):
            raise ValueError("indices/weights must have shapes (n,) integer and (n,4)")
        if np.any(ids < -1) or np.any(ids >= self.n_tets):
            raise ValueError("cell index out of bounds (only -1 denotes failure)")
        found = ids >= 0
        good_weights = bary[found]
        if (not np.isfinite(good_weights).all()
                or np.any(good_weights < -_BARY_TOL)
                or np.any(good_weights > 1 + _BARY_TOL)
                or np.any(np.abs(good_weights.sum(axis=1) - 1) > 8 * _BARY_TOL)):
            raise ValueError("located samples have invalid barycentric weights")
        result = np.full(potential.shape[:-1] + (len(ids),), np.nan, dtype=np.float64)
        rows, out = np.atleast_2d(potential), np.atleast_2d(result)
        for start in range(0, len(ids), self.query_chunk_size):
            stop = min(start + self.query_chunk_size, len(ids))
            selected = np.flatnonzero(found[start:stop]) + start
            nodes = self._tetrahedra[ids[selected]]
            for row, target in zip(rows, out, strict=True):
                values = np.asarray(row[nodes], dtype=np.float64)
                valid = np.isfinite(values).all(axis=1)
                with np.errstate(over="ignore", invalid="ignore"):
                    # Reference form preserves a constant field exactly, without
                    # renormalizing or clipping the raw barycentric weights.
                    sampled = values[:, 0] + np.einsum(
                        "ni,ni->n", bary[selected, 1:], values[:, 1:] - values[:, :1]
                    )
                sampled[~valid | ~np.isfinite(sampled)] = np.nan
                target[selected] = sampled
        return result
