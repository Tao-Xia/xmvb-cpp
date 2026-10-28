"""Analyze bounded actual CIAH snapshots on a Slurm compute node.

Spin-string compound matrices provide an independent physical overlap reference.
The molecular optimizer is neither called nor changed by this analysis.
"""
import csv
import itertools
import json
from pathlib import Path
import sys

import numpy as np
from scipy.linalg import cholesky, qr, solve_triangular


class Reference:
    def __init__(self, directory):
        self.directory = directory
        with (directory / 'prepared/basis.bin').open('rb') as f:
            version, self.nbf, self.nc, self.na, self.ns, self.ua, self.ub, roots = map(
                int, np.fromfile(f, '<u8', 8))
            assert version == roots == 1 and self.ua * self.ub <= 512
            masks = [np.fromfile(f, '<u8', n) for n in (self.ua, self.ub)]
            self.occupations = [np.array([[p for p in range(self.na) if int(m) & (1 << p)]
                                          for m in side]) for side in masks]
            self.s = np.fromfile(f, '<f8', self.nbf**2).reshape((self.nbf, self.nbf), order='F')
            self.w = np.fromfile(f, '<f8', self.ua*self.ub*self.ns).reshape(
                (self.ua, self.ub, self.ns), order='F').astype(np.longdouble)
        self.s_upper = cholesky(self.s, lower=False)
        self.cache = {}

    def compound(self, r, side):
        occ = self.occupations[side]
        minors = r[occ[:, None, :, None], occ[None, :, None, :]].astype(np.longdouble)
        k = occ.shape[1]
        result = np.zeros(minors.shape[:2], dtype=np.longdouble)
        for perm in itertools.permutations(range(k)):
            sign = (-1)**sum(perm[i] > perm[j] for i in range(k) for j in range(i+1, k))
            term = np.ones_like(result)
            for i in range(k):
                term *= minors[:, :, i, perm[i]]
            result += sign * term
        return result

    def load(self, name):
        if name in self.cache:
            return self.cache[name]
        data = np.fromfile(self.directory / 'prepared' / name, '<f8')
        p = self.nbf*self.nc
        raw_core = data[:p].reshape((self.nbf, self.nc), order='F')
        phi = data[p:p+self.nbf*self.na].reshape((self.nbf, self.na), order='F')
        c = data[p+self.nbf*self.na:].astype(np.longdouble)
        assert len(c) == self.ns
        # AO-weighted QR is an independent, stable reference, not the production frame.
        qc, _ = qr(self.s_upper @ raw_core, mode='economic')
        core = solve_triangular(self.s_upper, qc)
        qa, r = qr(self.s_upper @ phi, mode='economic')
        active = solve_triangular(self.s_upper, qa)
        a, b = self.compound(r, 0), self.compound(r, 1)
        factors = np.einsum('ai,ijs,bj->abs', a, self.w, b)
        f = factors.reshape((self.ua*self.ub, self.ns), order='F')
        state = (f @ c).reshape((self.ua, self.ub), order='F')
        norm = np.sqrt(np.sum(state*state))
        state /= norm
        diag = np.sqrt(np.sum(f*f, axis=0))
        scaled = c * diag / norm
        correlation = np.array((f.T @ f) / (diag[:, None]*diag[None, :]), dtype=float)
        spectrum = np.linalg.eigvalsh(correlation)
        point = dict(core=core, active=active, f=f, c=c, state=state,
                     norm=float(norm), cmax=float(max(abs(c))),
                     structure_norm_min=float(min(diag)), structure_norm_max=float(max(diag)),
                     normalized_cmax=float(max(abs(scaled))),
                     diagonal_sum=float(np.sum(scaled*scaled)),
                     offdiagonal_sum=float(1-np.sum(scaled*scaled)),
                     correlation_min=float(spectrum[0]), correlation_max=float(spectrum[-1]),
                     orbital_condition=float(np.linalg.cond(self.s_upper @ phi)))
        self.cache[name] = point
        return point

    def overlap(self, left, right):
        cross = left['active'].T @ self.s @ right['active']
        core_cross = left['core'].T @ self.s @ right['core']
        cross -= (left['active'].T @ self.s @ right['core'] @
                  np.linalg.solve(core_cross, left['core'].T @ self.s @ right['active']))
        a, b = self.compound(cross, 0), self.compound(cross, 1)
        image = a @ right['state'] @ b.T
        return np.longdouble(np.linalg.det(core_cross))**2 * np.sum(left['state']*image)

    def orbital_only(self, origin, trial):
        point = dict(trial)
        state = (trial['f'] @ origin['c']).reshape((self.ua, self.ub), order='F')
        point['state'] = state / np.sqrt(np.sum(state*state))
        return point


def events(filename):
    for line in filename.read_text().splitlines():
        if not line.startswith('CIAH_DIAG,'):
            continue
        fields = line.split(',')
        values = dict(item.split('=', 1) for item in fields[4:] if '=' in item)
        yield int(fields[1]), fields[3], values


def main(root):
    reference = Reference(root / 'path')
    snapshots, origins, pending = [], {}, {}
    for macro, kind, values in events(root / 'run.err'):
        if kind == 'ciah_state_snapshot':
            name = Path(values['path']).name
            entry = dict(macro=macro, stage=values['stage'], name=name,
                         energy=float(values['energy']))
            snapshots.append(entry)
            if entry['stage'] == 'origin':
                origins[macro] = entry
            pending[entry['stage']] = entry
        elif kind == 'ciah_keyframe' and 'keyframe' in pending:
            pending['keyframe'].update({k: float(values[k]) for k in
                                       ('actual', 'predicted', 'mismatch', 'accepted')})
        elif kind == 'ciah_macro_trial' and 'seed' in pending:
            pending['seed'].update({k: float(values[k]) for k in
                                   ('actual', 'predicted', 'rho', 'linear', 'quadratic')})
    rows, previous = [], {}
    maximum_self_error = 0.0
    for entry in snapshots:
        point = reference.load(entry['name'])
        row = {**entry, **{k: v for k, v in point.items() if isinstance(v, float)}}
        maximum_self_error = max(maximum_self_error, abs(float(reference.overlap(point, point))-1))
        if entry['stage'] == 'origin':
            previous[entry['macro']] = entry
        else:
            for prefix, source in [('', origins[entry['macro']]),
                                   ('local_', previous[entry['macro']])]:
                origin = reference.load(source['name'])
                orbital = reference.orbital_only(origin, point)
                so = np.sign(reference.overlap(origin, orbital))
                st = np.sign(reference.overlap(origin, point))
                a = abs(reference.overlap(origin, orbital))
                b = abs(reference.overlap(origin, point))
                c = so*st*reference.overlap(orbital, point)
                do2, dc2, dt2 = 2*(1-a), 2*(1-c), 2*(1-b)
                cross = c-1-b+a
                result = dict(orbital_change=float(np.sqrt(max(0, do2))),
                              ci_compensation=float(np.sqrt(max(0, dc2))),
                              physical_change=float(np.sqrt(max(0, dt2))),
                              interference=float(2*cross),
                              change_cancellation=float((do2+dc2)/dt2) if dt2 > 0 else None,
                              decomposition_error=float(do2+dc2+2*cross-dt2),
                              reference=source['name'])
                row.update({prefix+k: v for k, v in result.items()})
            if entry['stage'] == 'keyframe' and entry.get('accepted', 0):
                previous[entry['macro']] = entry
        rows.append(row)
    columns = list(dict.fromkeys(k for row in rows for k in row))
    with (root / 'wavefunction-path.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(dict(states=len(rows), max_normalized_coefficient=max(
        r['normalized_cmax'] for r in rows), max_diagonal_sum=max(r['diagonal_sum'] for r in rows),
        max_decomposition_error=max(abs(r.get('decomposition_error', 0)) for r in rows),
        maximum_self_overlap_error=maximum_self_error)))
    if maximum_self_error > 1e-9:
        raise RuntimeError('independent physical overlap reference lost normalization')


if __name__ == '__main__':
    main(Path(sys.argv[1]))
