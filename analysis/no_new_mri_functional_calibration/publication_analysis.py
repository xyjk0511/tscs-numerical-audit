"""Audit versioned path results without changing any frozen scientific gate."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import itertools
import json
import platform
from collections import defaultdict
from pathlib import Path

import numpy as np

SEGMENTS = ('L1', 'L2', 'L3', 'L4', 'L5', 'S1', 'S2')
PEAK = 'peak_abs_activating_function_v_per_a_m2'


def normalized_table(rows):
    cells = defaultdict(dict)
    for row in rows:
        key = (row['offset'], row['subject'], row['fit'], row['scenario'],
               str(float(row['width_mm'])), row['side'])
        if row['segment'] in cells[key]:
            raise ValueError(f'duplicate segment: {key}')
        cells[key][row['segment']] = row['peak']
    valid, missing = {}, defaultdict(int)
    for key, scores in cells.items():
        if set(scores) != set(SEGMENTS) or any(v is None for v in scores.values()):
            missing[key[0]] += 1
            continue
        values = np.array([scores[s] for s in SEGMENTS])
        if not np.all(np.isfinite(values)) or np.any(values < 0) or values.sum() <= 0:
            raise ValueError(f'invalid scores: {key}')
        valid[key] = values / values.sum()
    return valid, {'planned': len(cells), 'complete': len(valid),
                   'missing_by_offset': dict(missing),
                   'closure_max_error': max(abs(v.sum()-1) for v in valid.values()) if valid else None}


def comparisons(table):
    result = []
    offsets = sorted({k[0] for k in table})
    planned_strata = 60*len({k[4] for k in table})
    for first, second in itertools.combinations(offsets, 2):
        paired = sorted({k[1:] for k in table if k[0] == first} &
                        {k[1:] for k in table if k[0] == second})
        if not paired:
            continue
        diffs = np.array([table[(second, *k)]-table[(first, *k)] for k in paired])
        geometries, strata = defaultdict(list), defaultdict(list)
        for key, diff in zip(paired, diffs, strict=True):
            geometries[key[0]].append(diff)
            strata[key[1:]].append(diff)
        means = np.array([np.mean(v, axis=0) for v in geometries.values()])
        complete = {k: np.mean(v, axis=0) for k, v in strata.items() if len(v) == 14}
        for i, segment in enumerate(SEGMENTS):
            full = np.array([v[i] for v in complete.values()])
            result.append({'first': first, 'second': second, 'segment': segment,
                           'paired': len(paired), 'n_geometry': len(geometries),
                           'pooled_pp': float(diffs[:, i].mean()*100),
                           'equal_geometry_pp': float(means[:, i].mean()*100),
                           'equal_complete_stratum_pp': float(full.mean()*100) if len(full) else None,
                           'geometry_positive_negative': [int((means[:, i]>0).sum()), int((means[:, i]<0).sum())],
                           'geometry_range_pp': [float(means[:, i].min()*100), float(means[:, i].max()*100)],
                           'realization_range_pp': [float(diffs[:, i].min()*100), float(diffs[:, i].max()*100)],
                           'strata_all_14': len(full), 'strata_planned':planned_strata,
                           'strata_not_all_14': planned_strata-len(full),
                           'strata_positive_negative': [int((full>0).sum()), int((full<0).sum())],
                           'strata': [{'key':list(k), 'mean_pp':float(v[i]*100)} for k,v in complete.items()]})
    return result


def required_multiplier(train, test):
    """Smallest median-centred expansion containing each bounded test value."""
    lo, med, hi = np.quantile(train, [.05, .5, .95], axis=0)
    width = np.where(test >= med, hi-med, med-lo)
    distance = np.abs(test-med)
    ratio = np.full_like(test, np.inf)
    np.divide(distance, width, out=ratio, where=width>0)
    ratio[distance == 0] = 0
    return ratio


def absolute_comparisons(rows):
    """Unnormalized AF comparisons, with their own explicit support counts."""
    indexed = {key_for_peak(r):r['peak'] for r in rows}
    result = []
    for first, second in itertools.combinations(sorted({r['offset'] for r in rows}), 2):
        for segment in SEGMENTS:
            paired = []
            by_geometry = defaultdict(list)
            for key, value in indexed.items():
                if key[0] != first or key[5] != segment or value is None:
                    continue
                other = indexed.get((second,*key[1:]))
                if other is not None:
                    paired.append((value,other))
                    by_geometry[key[1]].append(other-value)
            if not paired:
                continue
            values = np.array(paired)
            means = np.array([np.mean(v) for v in by_geometry.values()])
            result.append({'first':first,'second':second,'segment':segment,'paired':len(paired),
                           'units':'V/(A*m^2)', 'mean_first':float(values[:,0].mean()),
                           'mean_second':float(values[:,1].mean()),
                           'equal_geometry_change':float(means.mean()),
                           'geometry_positive_negative':[int((means>0).sum()),int((means<0).sum())]})
    return result


def compare_share_tables(first, second):
    keys = sorted(first.keys() & second.keys())
    difference = np.array([second[k]-first[k] for k in keys])*100
    before = {(r['first'],r['second'],r['segment']):r for r in comparisons(first)}
    after = {(r['first'],r['second'],r['segment']):r for r in comparisons(second)}
    shifts = [abs(after[k]['equal_geometry_pp']-before[k]['equal_geometry_pp']) for k in before.keys() & after.keys()]
    flips = [list(k) for k in before.keys() & after.keys() if after[k]['equal_geometry_pp']*before[k]['equal_geometry_pp'] < 0]
    return {'common_realizations':len(keys),
            'abs_share_change_pp_median_p95_max':np.quantile(np.abs(difference),[.5,.95,1]).tolist(),
            'max_equal_geometry_comparison_shift_pp':max(shifts),
            'equal_geometry_comparison_sign_flips':flips}


def select_multiplier(ratios):
    grid = np.arange(50, 300)/100
    coverage = np.array([np.mean(ratios <= w) for w in grid])
    passed = np.flatnonzero(coverage >= .9)
    index = int(passed[0]) if len(passed) else len(grid)-1
    return float(grid[index]), float(coverage[index]), bool(len(passed))


def nested_coverage(table, offsets=(0, 10, 20), grouped=True):
    """Outer leave-one-geometry-out; tuning uses inner geometry folds only.

    Target is a simulated realization's engineering share, not a subject mean
    or biological response. Fourteen geometries share one volume conductor.
    """
    grouped_rows = defaultdict(list)
    for key, value in sorted(table.items()):
        if key[0] in offsets:
            grouped_rows[(key[0], key[1])].append(
                np.array([value[0], value[1], value[2:].sum()]) if grouped else value)
    data = {k: np.array(v) for k, v in grouped_rows.items()}
    subjects = sorted({k[1] for k in data})
    if len(subjects) != 14 or any((o,s) not in data for o in offsets for s in subjects):
        return {'status': 'not_estimable_missing_geometry', 'n_geometry': len(subjects)}
    ratios = {}
    for outer in subjects:
        for offset in offsets:
            train = np.concatenate([data[(offset,s)] for s in subjects if s != outer])
            ratios[(offset,outer)] = required_multiplier(train, data[(offset,outer)])
    pooled = np.concatenate([r.ravel() for r in ratios.values()])
    naive_w, naive_cov, naive_ok = select_multiplier(pooled)
    folds, hits, total, group_hits = [], 0, 0, np.zeros(3 if grouped else 7, dtype=int)
    for outer in subjects:
        training_subjects = [s for s in subjects if s != outer]
        inner = []
        for inner_test in training_subjects:
            for offset in offsets:
                train = np.concatenate([data[(offset,s)] for s in training_subjects if s != inner_test])
                inner.append(required_multiplier(train, data[(offset,inner_test)]).ravel())
        w, inner_cov, achieved = select_multiplier(np.concatenate(inner))
        evaluated = np.concatenate([ratios[(offset,outer)] for offset in offsets])
        mask = evaluated <= w
        hit, n = int(mask.sum()), mask.size
        group_hits += mask.sum(axis=0)
        hits += hit
        total += n
        folds.append({'held_out_geometry':outer, 'training_geometries':training_subjects,
                      'multiplier':w, 'inner_coverage':inner_cov, 'inner_target_reached':achieved,
                      'outer_hits':hit, 'outer_total':n, 'outer_coverage':hit/n})
    return {'status':'computed_post_exposure_nested_internal_validation',
            'target':'simulated realization share; not human recruitment',
            'offsets_mm':list(offsets), 'groups':['L1','L2','L3-S2'] if grouped else list(SEGMENTS),
            'n_geometry':14, 'n_volume_conductor':1,
            'uncalibrated_hits':int((pooled<=1).sum()), 'total':total,
            'uncalibrated_coverage':float(np.mean(pooled<=1)),
            'same_data_selected_multiplier':naive_w, 'same_data_selected_coverage':naive_cov,
            'same_data_target_reached':naive_ok, 'nested_hits':hits, 'nested_coverage':hits/total,
            'geometry_equal_coverage':float(np.mean([f['outer_coverage'] for f in folds])),
            'per_group_hits':group_hits.tolist(), 'per_group_total':total//len(group_hits),
            'folds':folds}


def load_samples(output, spacing):
    rows, diagnostics = defaultdict(list), []
    files = sorted((output/f'samples-{spacing:g}mm').glob('[0-9][0-9][0-9].json'))
    if len(files) != 392:
        raise ValueError(f'incomplete sample run: {len(files)}/392 at {spacing} mm')
    for path in files:
        for r in json.loads(path.read_text()):
            if 'locate' in r:
                diagnostics.append({k:r[k] for k in ('index','offset','scenario','locate')})
            by_width = (r['field'] or {}).get('by_bin_mm', {})
            for width in (2.,5.,10.):
                metric = by_width.get(f'{width:g}')
                rows[r['method']].append({**{k:r[k] for k in ('offset','subject','fit','scenario','side','segment')},
                                         'width_mm':width,'peak':None if metric is None else metric[PEAK]})
    return rows, diagnostics


def key_for_peak(row):
    return tuple(row[k] for k in ('offset','subject','fit','scenario','side','segment','width_mm'))


def compare_peaks(first, second):
    a, b = ({key_for_peak(r):r['peak'] for r in rows} for rows in (first,second))
    keys = sorted(a.keys() & b.keys())
    pairs = [(a[k],b[k]) for k in keys if a[k] is not None and b[k] is not None]
    v = np.asarray(pairs)
    if not len(v):
        return {'paired_finite':0}
    absolute = np.abs(v[:,1]-v[:,0])
    relative = absolute/np.maximum(np.abs(v[:,0]), np.finfo(float).tiny)
    return {'paired_finite':len(pairs), 'unmatched_keys':len(a.keys() ^ b.keys()),
            'missing_mask_mismatches':sum((a[k] is None) != (b[k] is None) for k in keys),
            'missing_first':sum(a[k] is None for k in keys),
            'missing_second':sum(b[k] is None for k in keys), 'max_abs_difference':float(absolute.max()),
            'relative_error_median_p95_max':np.quantile(relative,[.5,.95,1]).tolist()}


def analyze(output, legacy_only=False):
    legacy = json.loads((output/'legacy-metrics.json').read_text())
    raw = {'frozen_legacy':legacy}
    diagnostics = {}
    if not legacy_only:
        for spacing in (.5,.25,.125):
            rows, diag = load_samples(output, spacing)
            raw.update({f'{method}_{spacing:g}mm':v for method,v in rows.items()})
            diagnostics[f'{spacing:g}mm'] = diag
    result = {'scope':'post-exposure numerical audit; frozen evidence unchanged', 'methods':{}}
    tables = {}
    for name, rows in raw.items():
        table, count = normalized_table(rows)
        tables[name] = table
        result['methods'][name] = {'counts':count, 'comparisons':comparisons(table),
                                   'unnormalized_comparisons':absolute_comparisons(rows),
                                   'comparisons_by_width_mm':{str(w):comparisons({k:v for k,v in table.items() if float(k[4]) == w}) for w in (2.,5.,10.)},
                                   'nested_three_group_0_20':nested_coverage(table),
                                   'nested_seven_segment_all':nested_coverage(table,(0,10,20,100),False)}
    if not legacy_only:
        common = set.intersection(*(set(t) for t in tables.values()))
        result['common_support'] = {'n_realizations_all_offsets':len(common),
                                    'methods':{name:comparisons({k:table[k] for k in common})
                                               for name,table in tables.items()}}
        result['share_sampling_sensitivity'] = {
            'spacing_0.5_to_0.25':compare_share_tables(tables['p1_complete_0.5mm'],tables['p1_complete_0.25mm']),
            'spacing_0.25_to_0.125':compare_share_tables(tables['p1_complete_0.25mm'],tables['p1_complete_0.125mm'])}
        result['peak_comparisons'] = {
            'legacy_replay':compare_peaks(legacy,raw['legacy_replay_0.5mm']),
            'complete_bins_effect':compare_peaks(raw['legacy_replay_0.5mm'],raw['nearest_complete_0.5mm']),
            'p1_effect':compare_peaks(raw['nearest_complete_0.5mm'],raw['p1_complete_0.5mm']),
            'spacing_0.5_to_0.25':compare_peaks(raw['p1_complete_0.5mm'],raw['p1_complete_0.25mm']),
            'spacing_0.25_to_0.125':compare_peaks(raw['p1_complete_0.25mm'],raw['p1_complete_0.125mm'])}
        check = result['peak_comparisons']['legacy_replay']
        if check['unmatched_keys'] or check['missing_mask_mismatches'] or check['relative_error_median_p95_max'][2] > 1e-8:
            raise ValueError(f'legacy replay failed: {check}')
        (output/'locator-diagnostics.json').write_text(json.dumps(diagnostics,indent=2)+'\n')
    target = output/('legacy-analysis.json' if legacy_only else 'analysis.json')
    target.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps({name:r['nested_three_group_0_20']['status'] for name,r in result['methods'].items()}))


def finalize(output):
    """Bind completed artifacts and emit a compact, repository-portable result."""
    def sha(path):
        digest = hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda:stream.read(4 << 20),b''):
                digest.update(block)
        return digest.hexdigest()

    result = json.loads((output/'analysis.json').read_text())
    prepared = json.loads((output/'prepared.json').read_text())
    sources = [Path(__file__), Path(__file__).with_name('publication_recompute.py'),
               Path(__file__).with_name('path_numerics_v2.py')]
    artifacts = {p.relative_to(output).as_posix():sha(p) for p in sorted(output.rglob('*'))
                 if p.is_file() and p.name != 'artifact-manifest.json'}
    manifest = {'python':platform.python_version(),
                'packages':{p:importlib.metadata.version(p) for p in ('numpy','scipy','h5py','pytest')},
                'code':{p.name:sha(p) for p in sources},'artifacts':artifacts}
    (output/'artifact-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
    concise = {k:v for k,v in result.items() if k not in ('methods','common_support')}
    concise['inputs'] = {'mesh_sha256':prepared['source_mesh_sha256'],
                        'fields':[{k:v for k,v in r.items() if k.endswith('sha256') or k=='offset_mm'}
                                  for r in prepared['fields']]}
    concise['artifact_manifest_sha256'] = sha(output/'artifact-manifest.json')
    concise['code'] = manifest['code']
    concise['methods'] = {}
    for name, method in result['methods'].items():
        concise['methods'][name] = {k:v for k,v in method.items() if k not in ('comparisons','comparisons_by_width_mm')}
        concise['methods'][name]['comparisons'] = [{k:v for k,v in r.items() if k!='strata'}
                                                  for r in method['comparisons']]
        concise['methods'][name]['comparisons_by_width_mm'] = {
            width:[{k:v for k,v in r.items() if k!='strata'} for r in comparisons_at_width]
            for width,comparisons_at_width in method['comparisons_by_width_mm'].items()}
    concise['common_support_count'] = result['common_support']['n_realizations_all_offsets']
    diag = json.loads((output/'locator-diagnostics.json').read_text())
    concise['locator'] = {spacing:{'queries':sum(r['locate']['n_queries'] for r in rows if r['offset']==0),
                                  'failed':sum(r['locate']['n_failed'] for r in rows if r['offset']==0)}
                          for spacing,rows in diag.items()}
    Path(__file__).with_name('publication_results.json').write_text(json.dumps(concise,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps({'artifacts_hashed':len(artifacts),'locator':concise['locator']}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--legacy-only',action='store_true')
    parser.add_argument('--finalize',action='store_true')
    args = parser.parse_args()
    if args.finalize:
        finalize(args.output)
    else:
        analyze(args.output,args.legacy_only)
