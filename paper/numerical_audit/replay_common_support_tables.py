"""Replay common-support comparisons from portable complete peak vectors.

This does not reconstruct potentials, locate tetrahedra, or replay exclusions
that require the locally held profiles. Use a separate directory for outputs.
"""
import argparse
import csv
import json
from pathlib import Path

from run_common_support import (
    SEGMENTS, digest, primary_discriminator, summarize_comparisons,
    write_csv_atomic, write_json_atomic,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=Path(__file__).resolve().parent / 'common_support')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    source, output = args.data_dir.resolve(), args.output.resolve()
    if output == source:
        raise ValueError('Replay outputs must not overwrite the supplied tables')
    validation = json.loads((source / 'profile_replay_validation.json').read_text())
    for name, expected in validation['aggregate_sha256'].items():
        if digest(source / name) != expected:
            raise ValueError(f'Bound aggregate file changed: {name}')
    with (source / 'complete_vectors.csv').open(encoding='utf-8', newline='') as stream:
        vectors = [dict(row, offset=int(row['offset']), width_mm=int(row['width_mm']),
                        **{segment: float(row[segment]) for segment in SEGMENTS})
                   for row in csv.DictReader(stream)]
    comparisons, geometries, support, _ = summarize_comparisons(vectors)
    output.mkdir(parents=True, exist_ok=True)
    for name, rows in (('comparisons.csv', comparisons), ('geometry_means.csv', geometries)):
        write_csv_atomic(output / name, rows, list(rows[0]))
        if digest(output / name) != digest(source / name):
            raise ValueError(f'Portable replay differs from supplied table: {name}')
    primary = primary_discriminator(comparisons, support)
    expected = json.loads((source / 'common_support_results.json').read_text())
    if support != expected['support'] or primary != expected['primary_discriminator']:
        raise ValueError('Portable support or primary result differs')
    write_json_atomic(output / 'portable_replay.json', {
        'complete_vectors': len(vectors), 'comparison_rows': len(comparisons),
        'geometry_rows': len(geometries), 'byte_identical_tables': ['comparisons.csv', 'geometry_means.csv'],
        'primary_discriminator': primary,
        'scope': 'derived-vector aggregation only; profile and field reconstruction not repeated',
    })
    print(f'Portable common-support replay: {len(comparisons)} comparisons; both tables byte-identical')


if __name__ == '__main__':
    main()
