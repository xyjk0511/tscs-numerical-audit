"""Independent local profile-only integration and geometry-weighted replay.

Uses per-bin trapezoidal integrals, not the delivered descriptor or aggregator.
Empirical profiles are read locally and never included in a submission bundle.
"""
import argparse
import csv
import hashlib
import itertools
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

SEGMENTS = ('L1', 'L2', 'L3', 'L4', 'L5', 'S1', 'S2')
VARIANTS = ('original_native', 'common_endpoint', 'common_endpoint_central')
WIDTHS = (2, 5, 10)
OFFSETS = (0, 10, 20, 100)
RTOL, ATOL = 5e-12, 1e-8


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def complete_count(length, width):
    count = int(np.floor(length / width))
    edge = (count + 1) * width
    if abs(edge - length) <= 8 * np.finfo(float).eps * max(length, edge):
        count += 1
    return count


def independent_peaks(s, potentials):
    length = float(s[-1])
    common = min(length, complete_count(length, .010) * .010)
    result = {}
    potentials = potentials - potentials[:, :1]
    for width_mm in WIDTHS:
        width = width_mm / 1000
        count = complete_count(length, width)
        means = np.empty((4, count))
        for bin_index in range(count):
            lo = bin_index * width
            hi = min((bin_index + 1) * width, length)
            begin = np.searchsorted(s, lo, side='right')
            end = np.searchsorted(s, hi, side='left')
            knots = np.r_[lo, s[begin:end], hi]
            for field_index in range(4):
                values = np.interp(knots, s, potentials[field_index])
                means[field_index, bin_index] = np.trapezoid(values, knots) / (hi - lo)
        common_count = complete_count(common, width) if common else 0
        for variant in VARIANTS:
            n = count if variant == 'original_native' else common_count
            if n < 5:
                result[(variant, width_mm)] = [None] * 4
                continue
            selected = means[:, :n]
            af = (selected[:, 2:] - 2 * selected[:, 1:-1] + selected[:, :-2]) / width**2
            if variant == 'common_endpoint_central':
                centres = (np.arange(1, n - 1) + .5) * width
                tolerance = 8 * np.finfo(float).eps * max(common, width, .015)
                af = af[:, (centres >= .015 - tolerance) & (centres <= common - .015 + tolerance)]
            result[(variant, width_mm)] = ([None] * 4 if af.shape[1] == 0
                                           else list(np.max(np.abs(af), axis=1)))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('output', type=Path)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    meta = json.loads((output / 'run-meta.json').read_text())
    summary = json.loads((output / 'common_support_results.json').read_text())
    assert summary['binding_sha256'] == meta['binding_sha256']
    paths = sorted((output / 'paths').glob('[0-9][0-9][0-9].json'))
    assert len(paths) == 392
    scalar_count = profile_count = 0
    max_abs = max_rel = 0.0
    cells = defaultdict(dict)
    for number, path in enumerate(paths):
        document = json.loads(path.read_text())
        assert document['context']['index'] == number
        assert document['binding_sha256'] == meta['binding_sha256']
        context = document['context']
        descriptors = {(row['scenario'], row['offset'], row['variant'], row['width_mm']): row
                       for row in document['results']}
        assert len(descriptors) == 540
        assert len(document['profiles']) == 15
        for row in document['profiles']:
            profile = output / row['path']
            assert digest(profile) == row['sha256']
            with np.load(profile, allow_pickle=False) as stored:
                assert str(stored['binding_sha256']) == meta['binding_sha256']
                assert str(stored['scenario']) == row['scenario']
                assert list(stored['offset_mm']) == list(OFFSETS)
                s, potentials = stored['s_m'], stored['potential_v_per_a']
                assert s[0] == 0 and np.all(np.diff(s) > 0)
                assert potentials.shape == (4, len(s)) and np.isfinite(potentials).all()
                expected = independent_peaks(s, potentials)
            for (variant, width), peaks in expected.items():
                for offset, peak in zip(OFFSETS, peaks, strict=True):
                    actual = descriptors[(row['scenario'], offset, variant, width)]['peak_abs_af_v_per_a_m2']
                    assert (actual is None) == (peak is None)
                    if peak is not None:
                        absolute = abs(actual - peak)
                        max_abs = max(max_abs, absolute)
                        max_rel = max(max_rel, absolute / max(abs(peak), np.finfo(float).tiny))
                        assert np.isclose(actual, peak, rtol=RTOL, atol=ATOL), (number, row['scenario'], variant, width, offset, actual, peak)
                    key = (variant, offset, context['subject'], context['fit'],
                           row['scenario'], context['side'], width)
                    assert context['segment'] not in cells[key]
                    cells[key][context['segment']] = peak
                    scalar_count += 1
            profile_count += 1
        if (number + 1) % 50 == 0:
            print(f'independent profiles: {number + 1}/392 paths', flush=True)
    assert scalar_count == 392 * 540 and profile_count == 5880
    table = {}
    for key, values in cells.items():
        assert set(values) == set(SEGMENTS)
        if any(values[segment] is None for segment in SEGMENTS):
            continue
        vector = np.array([values[segment] for segment in SEGMENTS])
        assert np.isfinite(vector).all() and np.all(vector >= 0)
        if vector.sum() > 0:
            table[key] = vector
    assert len(table) == summary['complete_vector_count']
    with (output / 'complete_vectors.csv').open(newline='') as stream:
        vectors = list(csv.DictReader(stream))
    exported = {(r['variant'], int(r['offset']), r['geometry'], r['fit'],
                 r['scenario'], r['side'], int(r['width_mm'])): r for r in vectors}
    assert len(exported) == len(vectors) and set(exported) == set(table)
    for key, expected in table.items():
        assert np.allclose([float(exported[key][segment]) for segment in SEGMENTS],
                           expected, rtol=RTOL, atol=ATOL)
    with (output / 'comparisons.csv').open(newline='') as stream:
        comparisons = list(csv.DictReader(stream))
    with (output / 'geometry_means.csv').open(newline='') as stream:
        geometry_rows = list(csv.DictReader(stream))
    geometry_lookup = {(r['support_mode'], r['variant'], int(r['first']), int(r['second']),
                        int(r['width_mm']), r['segment'], r['geometry']): r for r in geometry_rows}
    assert len(geometry_lookup) == len(geometry_rows)
    comparison_count = 0
    maximum_share_difference = 0.0
    maximum_raw_difference = 0.0
    verified_geometry_keys = set()
    for first, second in itertools.combinations(OFFSETS, 2):
        support = {}
        for variant in VARIANTS:
            sets = [{key[2:6] for key in table if key[0] == variant and key[1] == offset and key[6] == width}
                    for offset in (first, second) for width in WIDTHS]
            support[variant] = set.intersection(*sets)
        common = set.intersection(*support.values())
        for row in comparisons:
            if int(row['first']) != first or int(row['second']) != second:
                continue
            variant, width = row['variant'], int(row['width_mm'])
            selected = common if row['support_mode'] == 'all_variant_intersection' else support[variant]
            assert len(selected) == int(row['paired_realizations'])
            groups = defaultdict(list)
            raw_groups = defaultdict(list)
            for key in sorted(selected):
                before = table[(variant, first, *key, width)]
                after = table[(variant, second, *key, width)]
                groups[key[0]].append(100 * (after / after.sum() - before / before.sum()))
                raw_groups[key[0]].append(after - before)
            assert len(groups) == int(row['n_geometry'])
            segment_index = SEGMENTS.index(row['segment'])
            geometry_means = {geometry: np.mean(values, axis=0)[segment_index]
                              for geometry, values in sorted(groups.items())}
            raw_geometry_means = {geometry: np.mean(values, axis=0)[segment_index]
                                  for geometry, values in sorted(raw_groups.items())}
            mean = np.mean(list(geometry_means.values()))
            difference = abs(mean - float(row['equal_geometry_share_change_pp']))
            maximum_share_difference = max(maximum_share_difference, difference)
            assert difference < 1e-8, (row, mean)
            raw_mean = np.mean(list(raw_geometry_means.values()))
            maximum_raw_difference = max(maximum_raw_difference,
                                         abs(raw_mean - float(row['equal_geometry_raw_peak_change'])))
            assert np.isclose(raw_mean, float(row['equal_geometry_raw_peak_change']),
                              rtol=RTOL, atol=ATOL), (row, raw_mean)
            for prefix, means in (('share', geometry_means), ('raw', raw_geometry_means)):
                for sign, predicate in (('positive', lambda x: x > 0),
                                        ('negative', lambda x: x < 0), ('zero', lambda x: x == 0)):
                    assert sum(predicate(value) for value in means.values()) == int(row[f'{prefix}_geometry_{sign}'])
            for geometry, value in geometry_means.items():
                geometry_key = (row['support_mode'], variant, first, second, width,
                                row['segment'], geometry)
                stored = geometry_lookup[geometry_key]
                assert int(stored['n_realizations']) == len(groups[geometry])
                assert abs(value - float(stored['share_change_pp'])) < 1e-8
                assert np.isclose(raw_geometry_means[geometry], float(stored['raw_peak_change']),
                                  rtol=RTOL, atol=ATOL), (geometry_key, stored)
                verified_geometry_keys.add(geometry_key)
            comparison_count += 1
    assert comparison_count == len(comparisons)
    assert verified_geometry_keys == set(geometry_lookup)
    report = {'binding_sha256': meta['binding_sha256'], 'profile_files_verified': profile_count,
              'path_files_verified': len(paths), 'scalar_descriptors_verified': scalar_count,
              'maximum_absolute_peak_difference': max_abs,
              'maximum_relative_peak_difference': max_rel,
              'complete_vectors_verified': len(table), 'aggregate_comparisons_verified': comparison_count,
              'maximum_share_comparison_difference_pp': maximum_share_difference,
              'maximum_raw_comparison_difference': maximum_raw_difference,
              'geometry_rows_verified': len(verified_geometry_keys),
              'geometry_sign_counts_verified': True,
              'peak_rtol': RTOL, 'peak_atol_v_per_a_m2': ATOL,
              'share_atol_pp': 1e-8,
              'method': 'independent bin-by-bin trapezoidal integration; native bin prefixes for common endpoint; independent complete vectors, support intersection and arithmetic geometry weighting',
              'scope': 'local profile-only numerical replay; no anatomy, locator, FEM solve, physiological or population validation',
              'verifier_sha256': digest(Path(__file__)),
              'aggregate_sha256': {name: digest(output / name) for name in
                                   ('complete_vectors.csv', 'comparisons.csv', 'geometry_means.csv', 'exclusions.jsonl', 'common_support_results.json')}}
    args.report.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
