"""Versioned reanalysis of existing fields; never overwrites historical evidence.

Run prepare first, then sample after the numerical core is available. Original
root polylines/scenarios, field solutions and seven-segment denominator remain
fixed. No FEM, EMG, neural or human-validation gate is changed here.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import sys
import time
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'fem'))
from fem.align_roots_by_level import fit_map, level_pairs
from fem.place_lumbosacral_roots import load_alvar, load_subject
from fem.sample_level_anchored_root_fields import apply_local_direction_displacement, place_root
from fem.sweep_root_roll import resample_by_arclength, bin_potential, activating_function
from analysis.no_new_mri_functional_calibration.sample_root_ensemble import (
    candidate_scenarios, _load_field, _directory_hash, reconstruct_named_current_montage,
)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4 << 20), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def source_paths(source):
    base = source / 'work/no-new-mri-20260830'
    paired = source / 'work/method-search-20260830/novel-montage/paired-run-caudad-100mm-vs-baseline'
    return {
        0: paired / '02-gordineer-tss-up-bilateral-2x2-caudad-00mm-root-ensemble.json',
        10: base / 'root-ensemble-bilateral-2x2-caudad-10mm.json',
        20: base / 'root-ensemble-bilateral-2x2-caudad-20mm.json',
        100: paired / '01-novel-tss-up-bilateral-2x2-caudad-100mm-root-ensemble.json',
    }


def prepare(source, output):
    if (output / 'prepared.json').exists():
        raise FileExistsError('Prepared cache exists; use sample, do not overwrite it.')
    output.mkdir(parents=True, exist_ok=True)
    paths = source_paths(source)
    first = json.loads(paths[0].read_text(encoding='utf-8'))
    mesh = Path(first['provenance']['mesh']['path'])
    markers = Path(first['provenance']['markers']['path'])
    if _directory_hash(markers) != first['provenance']['markers']['sha256']:
        raise ValueError('Marker directory hash mismatch')
    if digest(mesh) != first['provenance']['mesh']['sha256']:
        raise ValueError('Mesh hash mismatch')
    print('mesh identity verified; extracting geometry-only tissue subset', flush=True)
    with h5py.File(mesh) as h:
        points = np.asarray(h['mesh/points_xyz_m'])
        np.save(output / 'points.npy', points)
        cells, labels = [], []
        for start in range(0, len(h['mesh/tetrahedra']), 25000):
            lab = h['mesh/source_label'][start:start+25000]
            keep = np.isin(lab, [18, 66, 44])
            cells.append(h['mesh/tetrahedra'][start:start+25000][keep])
            labels.append(lab[keep])
        selected = np.concatenate(cells)
        labels = np.concatenate(labels)
        used, inverse = np.unique(selected, return_inverse=True)
        with h5py.File(output / 'geometry-only.h5', 'w') as reduced:
            reduced['mesh/points_xyz_m'] = points[used]
            reduced['mesh/tetrahedra'] = inverse.reshape(-1, 4)
            reduced['mesh/source_label'] = labels
    # load_alvar consumes only these three tissue labels. This subset is not a FEM mesh.
    alvar = load_alvar(output / 'geometry-only.h5')
    del cells, labels, selected, used, inverse
    gc.collect()
    contexts, lines, archive = [], [], {}
    mapping_errors = []
    for subject_record in first['per_subject']:
        name = subject_record['subject']
        subject = load_subject(markers / name)
        for fit in ('L', 'R'):
            pairs = level_pairs(subject, alvar, fit)
            mapping = fit_map([(r[1], r[2]) for r in pairs], tuple(r[0] for r in pairs))
            old = subject_record['fit_sides'][fit]
            err = max(abs(mapping.slope-old['slope']), abs(mapping.intercept-old['intercept_m']))
            if err > 1e-10:
                raise ValueError(f'Original mapping replay failed: {name}/{fit}: {err}')
            mapping_errors.append(err)
            for root in old['placed_roots']:
                key = (root['segment'], root['side'])
                placed, dropped = place_root(subject, alvar, subject.roots[key], mapping.apply)
                if placed is None or dropped != root['dropped_points']:
                    raise ValueError('Original root support differs')
                baseline = resample_by_arclength(placed, 0.0005)
                if len(baseline) != root['scenarios'][0]['sample_count']:
                    raise ValueError('Original sample count differs')
                i = len(contexts)
                contexts.append({'subject': name, 'fit': fit, 'segment': key[0], 'side': key[1], 'index': i})
                lines.append(placed)
                archive[str(i)] = baseline
    np.savez_compressed(output / 'baseline-paths.npz', **archive)
    write_json(output / 'contexts.json', contexts)
    # Exactly reproduce the legacy centroid neighbourhood, but in small chunks.
    stacked = np.vstack(lines)
    origin = stacked.mean(axis=0)
    reach = np.linalg.norm(stacked[:, :2]-origin[:2], axis=1).max() + 0.02
    zlo, zhi = stacked[:, 2].min()-0.02, stacked[:, 2].max()+0.02
    local_cells = []
    with h5py.File(mesh) as h:
        for start in range(0, len(h['mesh/tetrahedra']), 25000):
            nodes = np.asarray(h['mesh/tetrahedra'][start:start+25000], dtype=np.int64)
            centroid = (points[nodes[:, 0]] + points[nodes[:, 1]] + points[nodes[:, 2]] + points[nodes[:, 3]]) * 0.25
            keep = (centroid[:, 2]>=zlo)&(centroid[:, 2]<=zhi)&(np.linalg.norm(centroid[:, :2]-origin[:2], axis=1)<=reach)
            local_cells.append(nodes[keep])
    local = np.concatenate(local_cells)
    np.save(output / 'local-cells.npy', local)
    print(f'{len(contexts)} paths; {len(local)} local tetrahedra; mappings replayed', flush=True)
    del local_cells, local, archive, lines, stacked
    gc.collect()
    bindings = []
    legacy = []
    for offset, path in paths.items():
        document = json.loads(path.read_text(encoding='utf-8'))
        prov = document['provenance']
        if prov['mesh']['sha256'] != first['provenance']['mesh']['sha256']:
            raise ValueError('Montages do not share a mesh')
        field, spec = Path(prov['field']['path']), Path(prov['montage_spec']['path'])
        for key, p in (('field', field), ('montage_spec', spec)):
            if digest(p) != prov[key]['sha256']:
                raise ValueError(f'{key} hash mismatch')
        with h5py.File(field) as h:
            if h['node_coordinates'].shape != points.shape:
                raise ValueError('Node layout mismatch')
            for start in range(0, len(points), 25000):
                if not np.allclose(h['node_coordinates'][start:start+25000], points[start:start+25000], rtol=0, atol=1e-12):
                    raise ValueError('Node coordinate mismatch')
        lead, basis, ids = _load_field(field)
        reconstructed = reconstruct_named_current_montage(lead, basis, ids, spec)
        np.save(output / f'potential-{offset}.npy', reconstructed['nodal_potential_v_per_a'])
        bindings.append({'offset_mm': offset, 'ensemble': str(path), 'ensemble_sha256': digest(path), 'field': str(field), 'field_sha256': prov['field']['sha256'], 'spec': str(spec), 'spec_sha256': prov['montage_spec']['sha256'], 'units':'V/A'})
        for sub in document['per_subject']:
            for fit, fitted in sub['fit_sides'].items():
                for root in fitted['placed_roots']:
                    for sc in root['scenarios']:
                        for width, m in sc['field']['by_bin_mm'].items():
                            legacy.append({'offset':offset,'subject':sub['subject'],'fit':fit,'segment':root['segment'],'side':root['side'],'scenario':sc['scenario_id'],'width_mm':float(width),'peak':None if m is None else m['peak_abs_activating_function_v_per_a_m2']})
        del lead, reconstructed, document
        gc.collect()
        print(f'field {offset} mm verified and normalized', flush=True)
    write_json(output / 'legacy-metrics.json', legacy)
    write_json(output / 'prepared.json', {'schema':'publication-recompute-v1','source_mesh':str(mesh),'source_mesh_sha256':first['provenance']['mesh']['sha256'],'mapping_replay_max_error':max(mapping_errors),'n_paths':len(contexts),'local_tetrahedra':int(np.load(output/'local-cells.npy', mmap_mode='r').shape[0]),'fields':bindings,'sets_no_gates':True,'geometry_subset_is_not_fem_mesh':True,'analysis_phase':'post_exposure_numerical_audit'})


def prepare_locator(output):
    """Retain every tetrahedron that could contain a displaced path point."""
    prepared = json.loads((output/'prepared.json').read_text())
    points = np.load(output/'points.npy', mmap_mode='r')
    with np.load(output/'baseline-paths.npz') as paths:
        stacked = np.vstack([paths[k] for k in paths.files])
    lower, upper = stacked.min(axis=0)-.002, stacked.max(axis=0)+.002
    selected = []
    with h5py.File(prepared['source_mesh']) as h:
        for start in range(0, len(h['mesh/tetrahedra']), 25000):
            nodes = h['mesh/tetrahedra'][start:start+25000]
            vertices = points[nodes]
            keep = np.all(vertices.max(axis=1)>=lower, axis=1) & np.all(vertices.min(axis=1)<=upper, axis=1)
            selected.append(nodes[keep])
    target = output/'locator-cells.npy'
    if target.exists():
        raise FileExistsError(target)
    np.save(target, np.concatenate(selected))
    write_json(output/'locator-domain.json', {'lower_m':lower.tolist(),'upper_m':upper.tolist(),'rule':'all full-mesh tetrahedron AABBs intersecting path AABB with 2 mm margin; maximum displacement 1.5 mm','sha256':digest(target)})
    print(f'locator domain: {sum(len(x) for x in selected)} tetrahedra', flush=True)


def sample(output, spacing_mm, start_index=0, stop_index=None):
    from scipy.spatial import cKDTree
    from analysis.no_new_mri_functional_calibration.path_numerics_v2 import (
        TetraP1Sampler, resample_polyline, field_metrics_v2,
    )
    prepared = json.loads((output/'prepared.json').read_text())
    contexts = json.loads((output/'contexts.json').read_text())
    original = json.loads(Path(prepared['fields'][0]['ensemble']).read_text())
    markers = original['provenance']['markers']
    if _directory_hash(Path(markers['path'])) != markers['sha256']:
        raise ValueError('Marker directory hash mismatch')
    paths = np.load(output/'baseline-paths.npz')
    points = np.load(output/'points.npy', mmap_mode='r')
    cells = np.load(output/'local-cells.npy', mmap_mode='r')
    fields = np.stack([np.load(output/f'potential-{r["offset_mm"]}.npy') for r in prepared['fields']])
    legacy_centers = (points[cells[:,0]]+points[cells[:,1]]+points[cells[:,2]]+points[cells[:,3]])*.25
    nearest = cKDTree(legacy_centers)
    cell_potential = fields[:, cells].mean(axis=2)
    locator = TetraP1Sampler(points, np.load(output/'locator-cells.npy', mmap_mode='r'))
    alvar = load_alvar(output/'geometry-only.h5')
    run = output/f'samples-{spacing_mm:g}mm'
    run.mkdir(exist_ok=True)
    bound = [output/'prepared.json', output/'contexts.json', output/'baseline-paths.npz',
             output/'points.npy', output/'local-cells.npy', output/'locator-cells.npy',
             Path(__file__), ROOT/'analysis/no_new_mri_functional_calibration/path_numerics_v2.py',
             ROOT/'fem/sample_level_anchored_root_fields.py', ROOT/'fem/sweep_root_roll.py']
    bound.extend(output/f'potential-{r["offset_mm"]}.npy' for r in prepared['fields'])
    manifest = {'spacing_mm':spacing_mm,'inputs':{str(p):digest(p) for p in bound}}
    manifest_path = run/'run-meta.json'
    if manifest_path.exists():
        if json.loads(manifest_path.read_text()) != manifest:
            raise ValueError('Resume inputs/code differ from bound sample run')
    else:
        write_json(manifest_path,manifest)
    start_time = time.monotonic()
    for context in contexts[start_index:stop_index]:
        index = context['index']
        target = run/f'{index:03d}.json'
        if target.exists():
            continue
        baseline = paths[str(index)]
        rows = []
        for scenario in candidate_scenarios():
            d, phi = scenario['displacement_mm'], scenario['phi_deg']
            fixed = baseline if d == 0 else apply_local_direction_displacement(baseline, alvar, d, phi)[0]
            query, arclength = resample_polyline(fixed, spacing_mm/1000)
            indices, weights, diag = locator.locate(query)
            sampled = locator.sample(fields, indices, weights)
            for j, field in enumerate(prepared['fields']):
                common = {**context,'offset':field['offset_mm'],'scenario':scenario['scenario_id'],'d_mm':d,'phi_deg':phi,'spacing_mm':spacing_mm,'sample_count':len(query)}
                metrics = field_metrics_v2(arclength, sampled[j]) if np.all(np.isfinite(sampled[j])) else None
                rows.append({**common,'method':'p1_complete','field':metrics,'locate':diag})
            if spacing_mm == 0.5:
                # Same original query points isolate tail/integration from interpolation changes.
                _, old_ids = nearest.query(fixed)
                s = np.r_[0., np.cumsum(np.linalg.norm(np.diff(fixed, axis=0), axis=1))]
                for j, field in enumerate(prepared['fields']):
                    common = {**context,'offset':field['offset_mm'],'scenario':scenario['scenario_id'],'d_mm':d,'phi_deg':phi,'spacing_mm':spacing_mm}
                    voltage = cell_potential[j,old_ids]
                    by_width = {}
                    for width in (2.,5.,10.):
                        binned, _ = bin_potential(fixed, voltage, width/1000)
                        by_width[f'{width:g}'] = None if len(binned)<5 else {'peak_abs_activating_function_v_per_a_m2':float(np.max(np.abs(activating_function(binned,width/1000))))}
                    rows.append({**common,'method':'legacy_replay','field':{'by_bin_mm':by_width}})
                    rows.append({**common,'method':'nearest_complete','field':field_metrics_v2(s,voltage)})
        write_json(target, rows)
        print(f'spacing={spacing_mm} path={index+1}/{len(contexts)} elapsed={time.monotonic()-start_time:.1f}s', flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=['prepare','locator','sample'])
    p.add_argument('--source-root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--spacing-mm',type=float,default=.5)
    p.add_argument('--start-index',type=int,default=0)
    p.add_argument('--stop-index',type=int)
    a=p.parse_args()
    if a.stage=='prepare':
        prepare(a.source_root,a.output)
    elif a.stage=='locator':
        prepare_locator(a.output)
    else:
        sample(a.output,a.spacing_mm,a.start_index,a.stop_index)


if __name__=='__main__':
    main()
