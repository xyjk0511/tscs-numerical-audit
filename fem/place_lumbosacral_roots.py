"""Place LSSM lumbosacral roots into Alvar by anatomy, then test whether it worked.

## Why placement rather than registration

The thoracic RADO registration could not determine rotation about the spinal
axis, because all four of its anchors -- intervertebral disc centroids -- lie on
that axis. A rotationally symmetric landmark cannot constrain a rotation. The
measured consequence was an angular spread with only 4.55% of 2000 perturbed
trials within five degrees of each other.

Fitting the lumbosacral case the same way would fail the same way, and for an
additional reason: Alvar's thecal sac is nearly circular in cross-section
(aspect 1.02 to 1.20 against an LSSM population 1.31), so matching sac shapes
would leave the roll just as free.

So this does not fit. It **constructs**:

- **Longitudinally**, the conus is an anatomical landmark both models carry. In
  LSSM it is where the cord annotation tapers to about 1 mm^2; in Alvar it is
  where the cord label ends.
- **Transversely**, each root is centred on Alvar's own sac at its own height.
- **Angularly**, the roots are set at the angle they occupy in LSSM, measured
  from the dorsal midline, mapped onto Alvar's dorsal direction, which needs no
  fitting: the cord sits 37.067 mm below the disc in y, so dorsal is -y.
- **Radially**, the offset is scaled by the square root of the sac area ratio at
  the matched height, so the roots sit at the same fractional position within a
  sac of a different size.

There is no free parameter to fit, so there is nothing under-determined.

## The tests, and why the fourth one matters most

1. Do the roots land in CSF rather than bone?
2. Are they dorsal?
3. Do their angles match the fourteen-subject population, within its own 7.5
   degree scatter?

Those three use quantities the construction was built from, so passing them
shows the construction is self-consistent, not that it is right.

4. **Do the root exit points land at Alvar's intervertebral discs?**

That one is different. Alvar's discs took no part in the placement -- the only
Alvar structures used are the cord tip, the CSF centroids and the CSF areas. If
LSSM's L4 ganglion lands at Alvar's L4-L5 disc without having been told where
that disc is, the longitudinal alignment is confirmed by a structure that did
not participate in it. If the levels come out shifted by one, the construction
is wrong in a way the first three tests could never reveal.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import re
import sys
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import numpy.typing as npt

DEFAULT_MARKERS = Path('<local-path-redacted>')
DEFAULT_MESH = Path(
    "analysis/model_next_steps_l9_resolution/raw_grid_candidates/"
    "pb-purity24p2-39p4-41p1-79p0p5h-v81-topologyfix-surf1p0d0p20-pad2-"
    "deterministic1-nolimit-run2-topology-repaired1-smooth130-pin1-polish1-cgal-mesh.h5"
)

SEGMENTS = ("L1", "L2", "L3", "L4", "L5", "S1", "S2")
CORD_LABEL = 66
CSF_LABEL = 18
BONE_LABELS = (11, 17)
DISC_LABEL = 44

# Population angles from analysis/lssm_population_stats.py, fourteen subjects,
# measured from the dorsal midline with positive toward the left.
POPULATION_ANGLE_DEG: dict[tuple[str, str], tuple[float, float]] = {
    ("L1", "L"): (69.5, 7.5), ("L1", "R"): (-72.1, 6.5),
    ("L2", "L"): (72.7, 7.4), ("L2", "R"): (-73.1, 5.7),
    ("L3", "L"): (70.4, 7.7), ("L3", "R"): (-69.1, 4.6),
    ("L4", "L"): (60.3, 8.6), ("L4", "R"): (-58.5, 5.0),
    ("L5", "L"): (51.0, 7.6), ("L5", "R"): (-48.0, 6.0),
    ("S1", "L"): (40.2, 8.0), ("S1", "R"): (-35.0, 6.2),
    ("S2", "L"): (31.9, 7.8), ("S2", "R"): (-24.1, 8.1),
}


def digest_file(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            value.update(chunk)
    return value.hexdigest()


def polygon_area(points: npt.NDArray[np.float64]) -> float:
    x, y = points[:, 0], points[:, 1]
    return 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def load_markups(path: Path) -> list[npt.NDArray[np.float64]]:
    document = json.loads(path.read_text(encoding="utf-8"))
    return [
        np.asarray([c["position"] for c in m["controlPoints"]], dtype=np.float64)
        for m in document["markups"]
    ]


def subject_files(subject: Path, kind: str) -> list[Path]:
    """sub-14 uses an underscore where the others use a hyphen, so allow both."""
    pattern = re.compile(rf"^sub[-_]\d+_{kind}[_-]", re.IGNORECASE)
    return sorted(p for p in subject.glob("*.json") if pattern.match(p.name))


@dataclass(frozen=True)
class LssmSubject:
    """One subject's lumbosacral geometry in its own LPS frame, millimetres."""

    name: str
    conus_z: float
    sac_centres: npt.NDArray[np.float64]
    sac_areas: npt.NDArray[np.float64]
    roots: dict[tuple[str, str], npt.NDArray[np.float64]]
    ganglia: dict[tuple[str, str], npt.NDArray[np.float64]]

    sac_boundaries: npt.NDArray[np.float64]

    def sac_at(
        self, z: float
    ) -> tuple[npt.NDArray[np.float64], float, npt.NDArray[np.float64]]:
        index = int(np.argmin(np.abs(self.sac_centres[:, 2] - z)))
        return (
            self.sac_centres[index, :2],
            float(self.sac_areas[index]),
            self.sac_boundaries[index],
        )


def load_subject(subject: Path) -> LssmSubject:
    dura = [loop for p in subject_files(subject, "dura") for loop in load_markups(p)]
    cord = [loop for p in subject_files(subject, "cord") for loop in load_markups(p)]
    if not dura or not cord:
        raise ValueError(f"{subject.name}: missing dura or cord loops")

    # The conus is where the cord annotation tapers away. Taking the smallest
    # loop rather than the lowest guards against a stray annotation below the tip.
    cord_areas = np.asarray([polygon_area(loop) for loop in cord])
    cord_z = np.asarray([loop[:, 2].mean() for loop in cord])
    conus_z = float(cord_z[int(np.argmin(cord_areas))])

    order = np.argsort([loop[:, 2].mean() for loop in dura])
    sac_centres = np.asarray([dura[i].mean(axis=0) for i in order])
    sac_areas = np.asarray([polygon_area(dura[i]) for i in order])
    sac_boundaries = np.asarray(
        [
            boundary_profile(
                densify_loop(dura[i][:, :2]) - dura[i][:, :2].mean(axis=0)
            )
            for i in order
        ]
    )

    roots: dict[tuple[str, str], npt.NDArray[np.float64]] = {}
    ganglia: dict[tuple[str, str], npt.NDArray[np.float64]] = {}
    for segment in SEGMENTS:
        for side in ("L", "R"):
            for kind, store in (("nerveroots", roots), ("ganglions", ganglia)):
                pattern = rf"^sub[-_]\d+_{kind}_{segment}_{side}\.json$"
                found = [p for p in subject.glob("*.json") if re.match(pattern, p.name)]
                if found:
                    store[(segment, side)] = load_markups(found[0])[0]

    return LssmSubject(
        subject.name, conus_z, sac_centres, sac_areas, roots, ganglia, sac_boundaries
    )


# Rays used to profile each sac boundary. Sixteen sectors is enough to follow an
# oval without chasing the ragged edge a 2 mm mesh leaves behind.
BOUNDARY_RAYS = 16
RAY_ANGLES = np.linspace(-np.pi, np.pi, BOUNDARY_RAYS, endpoint=False)


def densify_loop(
    loop: npt.NDArray[np.float64], target: int = 20 * BOUNDARY_RAYS
) -> npt.NDArray[np.float64]:
    """Resample a closed polygon evenly along its perimeter.

    A dura loop has about eighteen control points, too few to fill sixteen
    angular sectors, and without this the sector rule falls through to its mean
    and reports every sac as circular.
    """
    closed = np.vstack([loop, loop[:1]])
    steps = np.sqrt(((np.diff(closed, axis=0)) ** 2).sum(axis=1))
    perimeter = np.concatenate([[0.0], np.cumsum(steps)])
    if perimeter[-1] <= 0.0:
        raise ValueError("degenerate loop with zero perimeter")
    wanted = np.linspace(0.0, perimeter[-1], target, endpoint=False)
    return np.stack(
        [np.interp(wanted, perimeter, closed[:, axis]) for axis in range(2)], axis=1
    )


def boundary_profile(offsets: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Distance to the sac edge along each ray.

    Filled samples give the edge as a high quantile of the radii in each sector;
    a maximum would chase single stray cells. Sectors with too few samples take
    the profile mean, so a gap does not become a spike -- but that fallback is a
    last resort, and boundary polygons are densified first so it rarely fires.
    """
    radius = np.sqrt((offsets**2).sum(axis=1))
    angle = np.arctan2(offsets[:, 1], offsets[:, 0])
    width = 2 * np.pi / BOUNDARY_RAYS
    profile = np.full(BOUNDARY_RAYS, np.nan)
    for index, centre in enumerate(RAY_ANGLES):
        delta = np.abs(np.angle(np.exp(1j * (angle - centre))))
        sector = delta <= width
        if np.count_nonzero(sector) >= 3:
            profile[index] = float(np.percentile(radius[sector], 90))
    if np.all(np.isnan(profile)):
        raise ValueError("no sector held enough samples to profile the sac")
    profile[np.isnan(profile)] = float(np.nanmean(profile))
    return profile


def boundary_at(profile: npt.NDArray[np.float64], angle: float) -> float:
    """Interpolate the boundary distance at one angle, wrapping at pi."""
    extended = np.concatenate([RAY_ANGLES, [RAY_ANGLES[0] + 2 * np.pi]])
    values = np.concatenate([profile, [profile[0]]])
    wrapped = float(np.angle(np.exp(1j * angle)))
    if wrapped < RAY_ANGLES[0]:
        wrapped += 2 * np.pi
    return float(np.interp(wrapped, extended, values))


@dataclass(frozen=True)
class AlvarCanal:
    """Alvar's cord tip and its sac, sampled by height. Metres."""

    conus_z: float
    heights: npt.NDArray[np.float64]
    centres: npt.NDArray[np.float64]
    areas: npt.NDArray[np.float64]
    boundaries: npt.NDArray[np.float64]
    disc_heights: npt.NDArray[np.float64]

    def sac_at(
        self, z: float
    ) -> tuple[npt.NDArray[np.float64], float, npt.NDArray[np.float64]] | None:
        if z > self.heights.max() or z < self.heights.min():
            return None
        index = int(np.argmin(np.abs(self.heights - z)))
        return self.centres[index], float(self.areas[index]), self.boundaries[index]


def load_alvar(mesh: Path, slab: float = 0.006) -> AlvarCanal:
    with h5py.File(mesh, "r") as handle:
        points = np.asarray(handle["mesh/points_xyz_m"][...], dtype=np.float64)
        tetrahedra = np.asarray(handle["mesh/tetrahedra"][...], dtype=np.int64)
        label = np.asarray(handle["mesh/source_label"][...], dtype=np.int64)

    vertices = points[tetrahedra]
    volume = (
        np.abs(
            np.einsum(
                "ij,ij->i",
                vertices[:, 1] - vertices[:, 0],
                np.cross(vertices[:, 2] - vertices[:, 0], vertices[:, 3] - vertices[:, 0]),
            )
        )
        / 6.0
    )
    centroids = vertices.mean(axis=1)

    cord = centroids[label == CORD_LABEL]
    conus_z = float(cord[:, 2].min())
    axis = cord[cord[:, 2] < conus_z + 0.02][:, :2].mean(axis=0)

    # The thecal sac holds CSF and, above the conus, the cord as well. Sampling
    # CSF alone would put the sac centre in the annulus rather than the canal.
    #
    # The range runs well above the conus as well as below it. An earlier version
    # sampled only downward and silently truncated every root to the fraction
    # below the tip -- L1 vanished entirely and L2 kept six of its samples. The
    # LSSM roots span -28 to +90 mm about the conus, so the window must too.
    sac = (label == CSF_LABEL) | (label == CORD_LABEL)
    near_axis = np.sqrt(((centroids[:, :2] - axis) ** 2).sum(axis=1)) < 0.020
    heights, centres, areas, boundaries = [], [], [], []
    for z in np.arange(conus_z + 0.130, 0.980, -slab):
        selected = sac & near_axis & (np.abs(centroids[:, 2] - z) < slab / 2)
        if np.count_nonzero(selected) < 40:
            continue
        weights = volume[selected]
        centre = np.average(centroids[selected][:, :2], axis=0, weights=weights)
        try:
            profile = boundary_profile(centroids[selected][:, :2] - centre)
        except ValueError:
            continue
        heights.append(z)
        centres.append(centre)
        areas.append(float(weights.sum() * 1e9 / (slab * 1000.0)))
        boundaries.append(profile)

    disc = centroids[label == DISC_LABEL]
    disc = disc[np.sqrt(((disc[:, :2] - axis) ** 2).sum(axis=1)) < 0.045]
    ordered = np.sort(disc[:, 2])
    groups, current = [], [ordered[0]]
    for value in ordered[1:]:
        if value - current[-1] > 0.006:
            groups.append(current)
            current = [value]
        else:
            current.append(value)
    groups.append(current)
    disc_heights = np.asarray(
        sorted(float(np.mean(g)) for g in groups if len(g) >= 30), dtype=np.float64
    )

    return AlvarCanal(
        conus_z,
        np.asarray(heights),
        np.asarray(centres),
        np.asarray(areas),
        np.asarray(boundaries),
        disc_heights,
    )


def place(
    subject: LssmSubject,
    alvar: AlvarCanal,
    line_mm: npt.NDArray[np.float64],
    height_map: Callable[[float], float] | None = None,
) -> npt.NDArray[np.float64] | None:
    """Carry one LSSM polyline into Alvar metres by the construction above.

    Frames. LSSM is LPS: +x left, +y posterior, +z superior. Alvar's README gives
    x left-to-right and z inferior-to-superior, and its own sense check records
    the cord 37.067 mm below the disc in y, so +y is ventral. Both are therefore
    mapped by (x, y) -> (-x, -y), a proper rotation of 180 degrees about z that
    leaves the superior direction alone.

    When height_map is given it replaces the conus anchor: it maps an LSSM
    height in millimetres to an Alvar height in metres, so the longitudinal
    placement can be driven by the level-anchored map instead.
    """
    placed = []
    for point in line_mm:
        if height_map is not None:
            height = height_map(point[2])
        else:
            height = alvar.conus_z + (point[2] - subject.conus_z) / 1000.0
        target = alvar.sac_at(height)
        if target is None:
            continue
        centre, _, target_profile = target
        source_centre, _, source_profile = subject.sac_at(point[2])
        offset = (point[:2] - source_centre) / 1000.0
        offset = np.array([-offset[0], -offset[1]])  # LPS to Alvar
        # Scale along the root's own ray rather than by equivalent radius. The
        # two sacs differ in shape, not only in size -- LSSM runs at aspect 1.31
        # while Alvar's 2 mm mesh rounds its canal to 1.02-1.20 -- so an area
        # ratio pushes near-lateral roots outside a rounder sac of equal area.
        angle = float(np.arctan2(offset[1], offset[0]))
        source_edge = boundary_at(source_profile, angle) / 1000.0
        target_edge = boundary_at(target_profile, angle)
        if source_edge <= 0.0:
            continue
        offset *= target_edge / source_edge
        placed.append([centre[0] + offset[0], centre[1] + offset[1], height])
    return np.asarray(placed) if placed else None


def check_placement(
    placed: dict[tuple[str, str], npt.NDArray[np.float64]],
    alvar: AlvarCanal,
    mesh: Path,
) -> dict[str, Any]:
    """The three self-consistency tests, plus the disc test that is not."""
    from scipy.spatial import cKDTree

    with h5py.File(mesh, "r") as handle:
        points = np.asarray(handle["mesh/points_xyz_m"][...], dtype=np.float64)
        tetrahedra = np.asarray(handle["mesh/tetrahedra"][...], dtype=np.int64)
        label = np.asarray(handle["mesh/source_label"][...], dtype=np.int64)
    centroids = points[tetrahedra].mean(axis=1)
    window = (centroids[:, 2] >= alvar.heights.min() - 0.02) & (
        centroids[:, 2] <= alvar.heights.max() + 0.02
    )
    axis = alvar.centres.mean(axis=0)
    window &= np.sqrt(((centroids[:, :2] - axis) ** 2).sum(axis=1)) < 0.040
    local = np.flatnonzero(window)
    tree = cKDTree(centroids[local])
    local_label = label[local]

    rows = []
    for (segment, side), line in placed.items():
        _, nearest = tree.query(line, k=1)
        labels = local_label[nearest]
        offsets = []
        for point in line:
            index = int(np.argmin(np.abs(alvar.heights - point[2])))
            offsets.append(point[:2] - alvar.centres[index])
        offset = np.median(np.asarray(offsets), axis=0)
        # Alvar dorsal is -y, and its +x is to the subject's right, so a left-side
        # root has negative x. Recover the same convention the population used:
        # zero at the dorsal midline, positive toward the left.
        angle = float(np.degrees(np.arctan2(-offset[0], -offset[1])))
        expected, spread = POPULATION_ANGLE_DEG[(segment, side)]
        rows.append(
            {
                "segment": segment,
                "side": side,
                "samples": int(line.shape[0]),
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
        )
    return {"per_root": rows}


def check_levels(
    subject: LssmSubject, alvar: AlvarCanal
) -> dict[str, Any]:
    """Do the ganglion exit heights land on Alvar's discs?

    Alvar's discs took no part in the placement, so this either confirms the
    longitudinal alignment or exposes a level shift the other tests cannot see.
    """
    rows = []
    for segment in SEGMENTS:
        for side in ("L", "R"):
            ganglion = subject.ganglia.get((segment, side))
            if ganglion is None:
                continue
            height = alvar.conus_z + (float(ganglion[:, 2].mean()) - subject.conus_z) / 1000.0
            gap = alvar.disc_heights - height
            nearest = float(gap[int(np.argmin(np.abs(gap)))]) * 1000.0
            # An exit below the lowest disc Alvar carries has nothing to be
            # tested against: the adult sacrum is fused and the disc label stops.
            # Scoring those as misses scores the test, not the placement. It put
            # S1 and S2 at +26 to +49 mm purely because the only disc within
            # reach was the wrong way, and dragged the summary to chance level.
            testable = bool(height >= alvar.disc_heights.min())
            rows.append(
                {
                    "segment": segment,
                    "side": side,
                    "exit_height_m": height,
                    "nearest_disc_offset_mm": nearest,
                    "testable": testable,
                    "untestable_reason": (
                        None
                        if testable
                        else "exit lies below the lowest disc in Alvar; the sacrum is fused"
                    ),
                }
            )
    tested = [r for r in rows if r["testable"]]
    residuals = np.abs(np.asarray([r["nearest_disc_offset_mm"] for r in tested]))
    spacing = float(np.median(np.diff(alvar.disc_heights)) * 1000.0)
    return {
        "per_root": rows,
        "tested_roots": len(tested),
        "untestable_roots": len(rows) - len(tested),
        "median_abs_offset_mm": float(np.median(residuals)),
        "max_abs_offset_mm": float(residuals.max()),
        "alvar_disc_spacing_mm": spacing,
        # A random alignment would scatter uniformly over one disc interval, so
        # its expected absolute offset is a quarter of the spacing.
        "chance_level_mm": spacing / 4.0,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Place LSSM roots into Alvar and test.")
    parser.add_argument("--markers", type=Path, default=DEFAULT_MARKERS)
    parser.add_argument("--mesh", type=Path, default=DEFAULT_MESH)
    parser.add_argument("--subject", type=str, default="sub-03")
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args(argv)

    subject_dir = arguments.markers / arguments.subject
    if not subject_dir.is_dir():
        raise FileNotFoundError(f"subject not found: {subject_dir}")
    if not arguments.mesh.is_file():
        raise FileNotFoundError(f"mesh not found: {arguments.mesh}")

    subject = load_subject(subject_dir)
    alvar = load_alvar(arguments.mesh)
    print(f"LSSM {subject.name}: conus at z = {subject.conus_z:.1f} mm")
    print(f"Alvar: conus at z = {alvar.conus_z:.4f} m, {alvar.heights.size} sac slices")
    print(f"Alvar discs detected: {alvar.disc_heights.size}")

    placed = {}
    for key, line in subject.roots.items():
        result = place(subject, alvar, line)
        if result is not None and result.shape[0] >= 5:
            placed[key] = result
    print(f"placed {len(placed)} of {len(subject.roots)} roots\n")

    geometry = check_placement(placed, alvar, arguments.mesh)
    levels = check_levels(subject, alvar)

    print(f"{'root':8s} {'n':>4} {'in CSF':>8} {'in bone':>8} {'dorsal mm':>10} "
          f"{'angle':>8} {'expected':>9} {'resid':>7} {'ok':>4}")
    for row in geometry["per_root"]:
        print(
            f"{row['segment']}_{row['side']:5s} {row['samples']:4d} "
            f"{row['in_csf_fraction']:8.3f} {row['in_bone_fraction']:8.3f} "
            f"{row['dorsal_offset_mm']:10.2f} {row['angle_deg']:8.1f} "
            f"{row['population_angle_deg']:9.1f} {row['angle_residual_deg']:7.1f} "
            f"{'yes' if row['within_population'] else 'NO':>4}"
        )

    rows = geometry["per_root"]
    # Judged on the worst root, not the median. A median lets one root sit a
    # third of its length inside bone with nothing reporting it.
    verdict = {
        "every_root_mostly_in_csf": bool(min(r["in_csf_fraction"] for r in rows) >= 0.5),
        "every_root_avoids_bone": bool(max(r["in_bone_fraction"] for r in rows) <= 0.1),
        "roots_dorsal": bool(all(r["is_dorsal"] for r in rows)),
        "angles_within_population": bool(
            sum(r["within_population"] for r in rows) >= 0.8 * len(rows)
        ),
        "levels_beat_chance": bool(
            levels["median_abs_offset_mm"] < levels["chance_level_mm"]
        ),
    }

    print(
        f"\nlevel test on {levels['tested_roots']} roots, "
        f"{levels['untestable_roots']} below the lowest disc and untestable: "
        f"median |offset| {levels['median_abs_offset_mm']:.1f} mm, "
        f"max {levels['max_abs_offset_mm']:.1f} mm, "
        f"chance {levels['chance_level_mm']:.1f} mm, "
        f"spacing {levels['alvar_disc_spacing_mm']:.1f} mm"
    )
    worst = max(rows, key=lambda r: r["in_bone_fraction"])
    print(
        f"worst root for bone: {worst['segment']}_{worst['side']} at "
        f"{worst['in_bone_fraction']:.3f} of its length"
    )
    print("\nverdict:")
    for key, value in verdict.items():
        print(f"  {key:28s} {'PASS' if value else 'FAIL'}")

    receipt = {
        "schema_version": 1,
        "generated_at": datetime.now(UTC).isoformat(),
        "status": "LUMBOSACRAL_ROOT_PLACEMENT_TEST",
        "provenance": {
            "lssm_subject": subject.name,
            "lssm_markers": str(arguments.markers),
            "mesh": str(arguments.mesh.resolve()),
            "mesh_sha256": digest_file(arguments.mesh),
        },
        "environment": {"python": platform.python_version(), "numpy": np.__version__},
        "construction": {
            "longitudinal": "LSSM conus tip to Alvar cord tip, unit scale",
            "transverse": "centred on Alvar sac centroid at matched height",
            "angular": "LSSM angle from dorsal midline, LPS to Alvar by (x,y)->(-x,-y)",
            "radial": "scaled by sqrt of sac area ratio at matched height",
            "free_parameters": 0,
        },
        "geometry_tests": geometry,
        "level_test": levels,
        "checks": verdict,
        "passed": all(verdict.values()),
        "sets_no_gates": True,
        "claim_scope": (
            "A geometric placement of one LSSM subject's lumbosacral root "
            "centrelines into the Alvar frame, and tests of whether the result is "
            "anatomically possible. It is not a registration, carries no "
            "uncertainty estimate, and makes no claim that these lines are the "
            "posterior roots specifically -- LSSM does not separate dorsal from "
            "ventral, so each line is a root complex."
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
