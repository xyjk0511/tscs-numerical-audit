"""Post-exposure matched-realization analyses of archived AF peaks.

The exported complete_peak_vectors.csv makes these analyses portable without
redistributing anatomy or rerunning FEM. Values are unnormalized V/(A*m^2).
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SEGMENTS = ('L1', 'L2', 'L3', 'L4', 'L5', 'S1', 'S2')
KEYS = ('offset', 'geometry', 'fit', 'scenario', 'side', 'width_mm')


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n',
                    encoding='utf-8', newline='\n')


def write_csv(path, rows):
    with path.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def export_vectors(audit_root):
    summary_file = HERE.parents[1] / 'analysis/no_new_mri_functional_calibration/publication_results.json'
    summary = json.loads(summary_file.read_text(encoding='utf-8'))
    manifest_bytes = (audit_root / 'artifact-manifest.json').read_bytes()
    if hashlib.sha256(manifest_bytes).hexdigest() != summary['artifact_manifest_sha256']:
        raise ValueError('Frozen artifact manifest mismatch')
    manifest = json.loads(manifest_bytes)['artifacts']
    cells = defaultdict(dict)
    paths = sorted((audit_root / 'samples-0.125mm').glob('[0-9][0-9][0-9].json'))
    if len(paths) != 392:
        raise ValueError('Expected 392 archived path records')
    for path in paths:
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != manifest[path.relative_to(audit_root).as_posix()]:
            raise ValueError(f'Archived sample mismatch: {path.name}')
        for record in json.loads(raw):
            if record['method'] != 'p1_complete':
                raise ValueError('Unexpected archived method')
            for width in (2, 5, 10):
                key = (record['offset'], record['subject'], record['fit'],
                       record['scenario'], record['side'], width)
                segment = record['segment']
                if segment in cells[key]:
                    raise ValueError('Duplicate segment')
                metric = (record['field'] or {}).get('by_bin_mm', {}).get(str(width))
                cells[key][segment] = None if metric is None else metric['peak_abs_activating_function_v_per_a_m2']
    output = []
    for key, peaks in sorted(cells.items()):
        if set(peaks) != set(SEGMENTS):
            raise ValueError('Missing segment record')
        if any(peaks[s] is None for s in SEGMENTS):
            continue
        values = np.array([peaks[s] for s in SEGMENTS])
        if not np.isfinite(values).all() or (values < 0).any() or values.sum() <= 0:
            raise ValueError('Invalid AF vector')
        output.append(dict(zip(KEYS, key)) | dict(zip(SEGMENTS, values)))
    if len(cells) != 10080 or len(output) != 9840:
        raise ValueError('Unexpected support counts')
    write_csv(HERE / 'complete_peak_vectors.csv', output)
    write_json(HERE / 'matched_input_provenance.json', {
        'source_manifest_sha256': summary['artifact_manifest_sha256'],
        'verified_sample_files': len(paths), 'planned_vectors': len(cells),
        'complete_vectors': len(output), 'units': 'V/(A*m^2)',
        'scope': 'derived AF values only; no anatomical coordinates or new FEM solve',
    })


def read_vectors(path):
    table = {}
    with path.open(encoding='utf-8', newline='') as stream:
        for row in csv.DictReader(stream):
            key = (int(row['offset']), row['geometry'], row['fit'], row['scenario'],
                   row['side'], int(row['width_mm']))
            if key in table:
                raise ValueError('Duplicate realization')
            values = np.array([float(row[s]) for s in SEGMENTS])
            if not np.isfinite(values).all() or (values < 0).any() or values.sum() <= 0:
                raise ValueError('Invalid AF vector')
            table[key] = values
    return table


def geometry_mean(values, keys):
    groups = defaultdict(list)
    for key, value in zip(keys, values, strict=True):
        groups[key[0]].append(value)
    means = {k: np.mean(v, axis=0) for k, v in sorted(groups.items())}
    return np.mean(list(means.values()), axis=0), means


def share(values):
    return values / values.sum(axis=-1, keepdims=True)


def symmetric_split(a0, b0, a1, b1):
    """Two-order exact attribution; no causal or independence interpretation."""
    a0, b0, a1, b1 = np.broadcast_arrays(a0, b0, a1, b1)
    if any(np.any(~np.isfinite(x)) or np.any(x < 0) for x in (a0, b0, a1, b1)):
        raise ValueError('Require finite nonnegative peaks')
    if any(np.any(x <= 0) for x in (a0+b0, a0+b1, a1+b0, a1+b1)):
        raise ValueError('Undefined crossed ratio')
    f00, f10 = a0/(a0+b0), a1/(a1+b0)
    f01, f11 = a0/(a0+b1), a1/(a1+b1)
    own = ((f10-f00)+(f11-f01))/2
    other = ((f01-f00)+(f11-f10))/2
    return own, other


def analyze(table):
    offsets = sorted({k[0] for k in table})
    widths = sorted({k[-1] for k in table})
    scale_rows, decomposition, baselines = [], [], []
    maximum_closure = 0.
    for first, second in itertools.combinations(offsets, 2):
        paired = sorted({k[1:] for k in table if k[0] == first}
                        & {k[1:] for k in table if k[0] == second})
        x = np.array([table[(first, *k)] for k in paired])
        y = np.array([table[(second, *k)] for k in paired])
        difference = 100*(share(y)-share(x))
        full_mean, full_geometry = geometry_mean(difference, paired)
        raw_mean, raw_geometry = geometry_mean(y-x, paired)
        # Compute the sum of other segments directly, avoiding subtraction loss.
        bx = np.stack([x[:, np.arange(7) != i].sum(axis=1) for i in range(7)], axis=1)
        by = np.stack([y[:, np.arange(7) != i].sum(axis=1) for i in range(7)], axis=1)
        own, other = symmetric_split(x, bx, y, by)
        maximum_closure = max(maximum_closure, float(np.max(np.abs(100*(own+other)-difference))))
        own_mean, own_geometry = geometry_mean(100*own, paired)
        other_mean, other_geometry = geometry_mean(100*other, paired)
        for i, segment in enumerate(SEGMENTS):
            baselines.append({'first': first, 'second': second, 'segment': segment,
                              'equal_geometry_pp': float(full_mean[i]), 'paired': len(paired)})
            decomposition.append({
                'first': first, 'second': second, 'segment': segment, 'paired': len(paired),
                'n_geometry': len(full_geometry), 'share_change_pp': float(full_mean[i]),
                'own_peak_contribution_pp': float(own_mean[i]),
                'other_peaks_contribution_pp': float(other_mean[i]),
                'raw_peak_change': float(raw_mean[i]),
                'raw_geometry_positive': sum(bool(v[i] > 0) for v in raw_geometry.values()),
                'share_geometry_positive': sum(bool(v[i] > 0) for v in full_geometry.values()),
                'geometry': [{
                    'id': g, 'share_change_pp': float(full_geometry[g][i]),
                    'raw_peak_change': float(raw_geometry[g][i]),
                    'own_peak_contribution_pp': float(own_geometry[g][i]),
                    'other_peaks_contribution_pp': float(other_geometry[g][i]),
                } for g in full_geometry],
            })
        by_width = {w: {k[:-1] for k in paired if k[-1] == w} for w in widths}
        common = sorted(set.intersection(*by_width.values()))
        for width in widths:
            keys = [(g, fit, scenario, side, width) for g, fit, scenario, side in common]
            before = np.array([table[(first, *k)] for k in keys])
            after = np.array([table[(second, *k)] for k in keys])
            mean, geometries = geometry_mean(100*(share(after)-share(before)), keys)
            for i, segment in enumerate(SEGMENTS):
                scale_rows.append({'first': first, 'second': second, 'segment': segment,
                                   'width_mm': width, 'paired': len(common),
                                   'n_geometry': len(geometries), 'equal_geometry_pp': float(mean[i]),
                                   'positive_geometries': sum(bool(v[i] > 0) for v in geometries.values()),
                                   'negative_geometries': sum(bool(v[i] < 0) for v in geometries.values())})
    return {'scope': 'additional post-exposure analysis; fixed P1 0.125-mm peaks',
            'common_width_comparisons': scale_rows,
            'symmetric_decomposition': decomposition,
            'baseline_replay': baselines,
            'max_decomposition_closure_pp': maximum_closure}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--audit-root', type=Path)
    args = parser.parse_args()
    if args.audit_root:
        export_vectors(args.audit_root)
    table = read_vectors(HERE / 'complete_peak_vectors.csv')
    result = analyze(table)
    if len(table) != 9840 or result['max_decomposition_closure_pp'] > 1e-10:
        raise ValueError('Data contract or decomposition identity failed')
    summary = json.loads((HERE.parents[1] / 'analysis/no_new_mri_functional_calibration/publication_results.json').read_text())
    expected = {(r['first'], r['second'], r['segment']): r for r in
                summary['methods']['p1_complete_0.125mm']['comparisons']}
    max_replay = max(abs(r['equal_geometry_pp'] - expected[(r['first'], r['second'], r['segment'])]['equal_geometry_pp'])
                     for r in result['baseline_replay'])
    if max_replay > 1e-10:
        raise ValueError('Portable values do not replay original comparisons')
    result['max_baseline_replay_difference_pp'] = max_replay
    result['input_sha256'] = hashlib.sha256((HERE / 'complete_peak_vectors.csv').read_bytes()).hexdigest()
    write_json(HERE / 'matched_analysis.json', result)
    write_csv(HERE / 'common_width_comparisons.csv', result['common_width_comparisons'])
    write_csv(HERE / 'normalization_decomposition.csv', [
        {k: v for k, v in r.items() if k != 'geometry'} for r in result['symmetric_decomposition']])
    print(json.dumps({k: v for k, v in result.items() if not isinstance(v, list)}, indent=2))
    for r in result['common_width_comparisons']:
        if r['first'] == 0 and r['second'] == 100 and r['segment'] in ('L4', 'S2'):
            print(r)
    for r in result['symmetric_decomposition']:
        if r['first'] == 0 and r['segment'] in ('L4', 'S2'):
            print({k: v for k, v in r.items() if k != 'geometry'})


if __name__ == '__main__':
    main()
