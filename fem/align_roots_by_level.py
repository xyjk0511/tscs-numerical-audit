"""Align LSSM roots to Alvar by vertebral level, and hold out a side to test it.

## Why the previous anchor failed

`fem/place_lumbosacral_roots.py` anchored LSSM's conus to Alvar's conus at unit
scale. That assumes the conus-to-foramen distance is shared between two people.
It is not: the conus sits anywhere from T12 to L2 in healthy adults. Measured
here, **Alvar's conus is at the L2 vertebral body** -- z = 1.1318 m, between the
L1-L2 disc at 1.1432 and the L2-L3 disc at 1.1103. A subject whose conus is at L1
is a whole vertebra out, about 31 mm, and the observed level errors of 9 to 16 mm
across five of six subjects sit inside that.

## What replaces it

Alvar's disc levels are now named. Twelve discs were detected within 45 mm of the
cord axis; the intervertebral disc label stops at z = 0.9965 exactly where the
sacrum begins -- dense near-midline bone that curves dorsally from y = 70.5 mm at
z = 1.00 to y = 37.6 at z = 0.93 -- and the sacrum is fused, so it carries no
discs. The lowest disc is therefore L5-S1, and counting up names the rest.

That count is confirmed independently. The anchor inventory at
`analysis/model_next_steps_l12_roll_identifiability/anchor_inventory.json`
records four Alvar discs named when the RADO registration was built, by a
different route, and they agree to better than 0.4 mm: T9-T10 to 0.39, T10-T11
to 0.13, T11-T12 to 0.31, and T12-L1 to 0.32.

So the longitudinal map can be anchored on level identity rather than on a
distance. Each lumbar root exits below its own pedicle: the L4 root leaves
through the L4-L5 foramen. Anchoring L1 through L5 exits onto the L1-L2 through
L5-S1 discs fixes the map with no cross-subject distance assumption.

## The hold-out, which is the point

Anchoring on the exits consumes the disc test that caught the previous failure.
Left and right are separately annotated, so the map is **fitted on one side and
tested on the other**. Five levels fit, five levels test, and the test side takes
no part in the fit.

A weaker second check comes free: where the fitted map puts LSSM's conus. It
should land in the T12 to L2 range, since that is where a conus anatomically can
be. It cannot land on Alvar's conus exactly and should not be expected to --
these are different people, and that is the whole reason the old anchor failed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from place_lumbosacral_roots import (  # noqa: E402
    DEFAULT_MARKERS,
    DEFAULT_MESH,
    SEGMENTS,
    load_alvar,
    load_subject,
)

# A lumbar root exits below its own pedicle, so the L4 root leaves through the
# L4-L5 foramen. S1 and S2 leave through sacral foramina, which carry no disc.
ROOT_TO_DISC: dict[str, str] = {
    "L1": "L1-L2",
    "L2": "L2-L3",
    "L3": "L3-L4",
    "L4": "L4-L5",
    "L5": "L5-S1",
}

# Twelve discs, superior to inferior. The lowest is L5-S1 because the disc label
# stops where the fused sacrum begins; four of the middle ones are confirmed
# against anchor_inventory.json to better than 0.4 mm.
DISC_NAMES: tuple[str, ...] = (
    "T6-T7", "T7-T8", "T8-T9", "T9-T10", "T10-T11", "T11-T12",
    "T12-L1", "L1-L2", "L2-L3", "L3-L4", "L4-L5", "L5-S1",
)


def digest_file(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            value.update(chunk)
    return value.hexdigest()


@dataclass(frozen=True)
class LinearMap:
    """z_alvar = intercept + slope * z_lssm, both in metres after unit change."""

    slope: float
    intercept: float
    fitted_on: tuple[str, ...]

    def apply(self, z_lssm_mm: float) -> float:
        return self.intercept + self.slope * (z_lssm_mm / 1000.0)


def fit_map(pairs: list[tuple[float, float]], names: tuple[str, ...]) -> LinearMap:
    """Least squares through the level pairs, written out to avoid LAPACK."""
    if len(pairs) < 2:
        raise ValueError("need at least two levels to fit a longitudinal map")
    source = np.asarray([p[0] for p in pairs], dtype=np.float64) / 1000.0
    target = np.asarray([p[1] for p in pairs], dtype=np.float64)
    centred = source - source.mean()
    denominator = float((centred**2).sum())
    if denominator <= 0.0:
        raise ValueError("degenerate level spacing; cannot fit")
    slope = float((centred @ (target - target.mean())) / denominator)
    intercept = float(target.mean() - slope * source.mean())
    return LinearMap(slope, intercept, names)


def level_pairs(
    subject: Any, alvar: Any, side: str
) -> list[tuple[str, float, float]]:
    """Exit height in LSSM against the named Alvar disc, for one side."""
    disc_by_name = dict(zip(DISC_NAMES, sorted(alvar.disc_heights, reverse=True), strict=True))
    rows = []
    for segment in SEGMENTS:
        disc = ROOT_TO_DISC.get(segment)
        if disc is None:
            continue
        ganglion = subject.ganglia.get((segment, side))
        if ganglion is None:
            continue
        rows.append((segment, float(ganglion[:, 2].mean()), float(disc_by_name[disc])))
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--markers", type=Path, default=DEFAULT_MARKERS)
    parser.add_argument("--mesh", type=Path, default=DEFAULT_MESH)
    parser.add_argument("--fit-side", choices=("L", "R"), default="L")
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args(argv)

    alvar = load_alvar(arguments.mesh)
    if alvar.disc_heights.size != len(DISC_NAMES):
        raise ValueError(
            f"expected {len(DISC_NAMES)} discs, detected {alvar.disc_heights.size}; "
            "the level naming would be wrong"
        )
    test_side = "R" if arguments.fit_side == "L" else "L"

    subjects = sorted(p for p in arguments.markers.iterdir() if p.is_dir())
    print(f"fitting on the {arguments.fit_side} side, testing on the {test_side}\n")
    print(f"{'subject':10s} {'fit rms':>9} {'test median':>12} {'test max':>10} "
          f"{'conus level':>13} {'slope':>7}")

    results = []
    for path in subjects:
        subject = load_subject(path)
        fit_rows = level_pairs(subject, alvar, arguments.fit_side)
        test_rows = level_pairs(subject, alvar, test_side)
        if len(fit_rows) < 3 or len(test_rows) < 3:
            print(f"{path.name:10s} too few levels")
            continue

        mapping = fit_map(
            [(r[1], r[2]) for r in fit_rows], tuple(r[0] for r in fit_rows)
        )
        fit_residual = [abs(mapping.apply(r[1]) - r[2]) * 1000.0 for r in fit_rows]
        test_residual = [abs(mapping.apply(r[1]) - r[2]) * 1000.0 for r in test_rows]

        # Where the fitted map puts this subject's conus, named by the two discs
        # it falls between. Different people, so this is a plausibility check.
        conus = mapping.apply(subject.conus_z)
        ordered = sorted(alvar.disc_heights, reverse=True)
        above = [n for n, z in zip(DISC_NAMES, ordered, strict=True) if z > conus]
        below = [n for n, z in zip(DISC_NAMES, ordered, strict=True) if z <= conus]
        vertebra = (
            f"{above[-1].split('-')[1]}" if above and below else ("above" if below else "below")
        )

        results.append(
            {
                "subject": path.name,
                "fit_side": arguments.fit_side,
                "fit_levels": [r[0] for r in fit_rows],
                "test_levels": [r[0] for r in test_rows],
                "slope": mapping.slope,
                "intercept_m": mapping.intercept,
                "fit_rms_mm": float(np.sqrt(np.mean(np.square(fit_residual)))),
                "test_median_mm": float(np.median(test_residual)),
                "test_max_mm": float(np.max(test_residual)),
                "conus_height_m": conus,
                "conus_vertebra": vertebra,
            }
        )
        print(
            f"{path.name:10s} {results[-1]['fit_rms_mm']:9.2f} "
            f"{results[-1]['test_median_mm']:12.2f} {results[-1]['test_max_mm']:10.2f} "
            f"{vertebra:>13} {mapping.slope:7.3f}"
        )

    if not results:
        raise ValueError("no subject produced a fit")

    test_median = np.asarray([r["test_median_mm"] for r in results])
    test_max = np.asarray([r["test_max_mm"] for r in results])
    spacing = float(np.median(np.diff(sorted(alvar.disc_heights))) * 1000.0)
    conus_ok = [r for r in results if r["conus_vertebra"] in ("T12", "L1", "L2")]

    verdict = {
        # Chance for a random longitudinal alignment is a quarter of the spacing.
        "held_out_side_beats_chance": bool(np.median(test_median) < spacing / 4.0),
        "no_subject_off_by_half_a_level": bool(test_max.max() < spacing / 2.0),
        "every_subject_beats_chance": bool(test_median.max() < spacing / 4.0),
        "conus_lands_in_anatomical_range": bool(len(conus_ok) == len(results)),
        "slopes_near_unity": bool(
            all(0.85 <= r["slope"] <= 1.15 for r in results)
        ),
    }

    print(f"\nheld-out side: median of medians {np.median(test_median):.2f} mm, "
          f"worst {test_max.max():.2f} mm, chance {spacing / 4.0:.2f} mm, "
          f"spacing {spacing:.1f} mm")
    print(f"conus lands within T12-L2 for {len(conus_ok)}/{len(results)} subjects")
    print("\nverdict:")
    for key, value in verdict.items():
        print(f"  {key:36s} {'PASS' if value else 'FAIL'}")

    receipt = {
        "schema_version": 1,
        "generated_at": datetime.now(UTC).isoformat(),
        "status": "LEVEL_ANCHORED_LONGITUDINAL_ALIGNMENT",
        "provenance": {
            "markers": str(arguments.markers),
            "mesh": str(arguments.mesh.resolve()),
            "mesh_sha256": digest_file(arguments.mesh),
            "disc_naming_confirmed_against": (
                "analysis/model_next_steps_l12_roll_identifiability/anchor_inventory.json"
            ),
        },
        "environment": {"python": platform.python_version(), "numpy": np.__version__},
        "disc_levels": {
            name: float(z)
            for name, z in zip(DISC_NAMES, sorted(alvar.disc_heights, reverse=True), strict=True)
        },
        "alvar_conus_m": alvar.conus_z,
        "method": {
            "anchor": "root exit heights onto their own foramen levels",
            "fitted_side": arguments.fit_side,
            "tested_side": test_side,
            "held_out": "the opposite side takes no part in the fit",
            "chance_level_mm": spacing / 4.0,
        },
        "per_subject": results,
        "checks": verdict,
        "passed": all(verdict.values()),
        "sets_no_gates": True,
        "claim_scope": (
            "A longitudinal map from LSSM subject heights into Alvar, anchored on "
            "vertebral level and tested on a held-out side. It fixes only the "
            "rostro-caudal placement. It says nothing about the transverse or "
            "angular placement, nothing about whether these lines are posterior "
            "rather than root complexes, and nothing about any field computed on "
            "the result."
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
