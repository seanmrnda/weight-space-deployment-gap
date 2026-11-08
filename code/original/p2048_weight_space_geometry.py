"""Exact task-vector geometry for the P2048 lineage, written as a JSON receipt.

Task vectors are relative to L: tau_X = X - L. Norms and cosines use all
elements of each group in float64. The temperature buffer and the unused action
head are excluded. Reads checkpoints only.
"""
import argparse, json, math
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
import p2048_weight_space as wsp
import p2048_weight_space_soup as soup


def group(key):
    if key == 'temperature' or key.startswith('act_head.'):
        return None
    if not key.startswith('encoder.'):
        return 'decision'
    if '.router.' in key:
        return 'routers'
    if '.experts.' in key:
        return 'experts'
    return 'encoder_rest'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    import torch
    from safetensors import safe_open
    torch.set_num_threads(1)
    names = ('L', 'P', 'T', 'H')
    handles = {n: safe_open(str(soup.ROOT / soup.ANCHORS[n]), framework='pt', device='cpu') for n in names}
    keys = sorted(handles['L'].keys())
    totals = {}
    def add(g, field, value):
        totals.setdefault(g, {}).setdefault(field, []).append(value)
    for key in keys:
        g = group(key)
        if g is None:
            continue
        base = handles['L'].get_tensor(key).double()
        vectors = {n: handles[n].get_tensor(key).double() - base for n in ('P', 'T', 'H')}
        add(g, 'base2', float(base.pow(2).sum()))
        for n, v in vectors.items():
            add(g, n + '2', float(v.pow(2).sum()))
        add(g, 'P.T', float((vectors['P'] * vectors['T']).sum()))
        add(g, 'P.H', float((vectors['P'] * vectors['H']).sum()))
        add(g, 'T.H', float((vectors['T'] * vectors['H']).sum()))
    report = {}
    for g, f in totals.items():
        s = {k: math.fsum(v) for k, v in f.items()}
        report[g] = dict(base_l2=math.sqrt(s['base2']),
                         **{f'relative_percent_{n}': 100 * math.sqrt(s[n + '2'] / s['base2']) for n in ('P', 'T', 'H')},
                         cosine_P_T=s['P.T'] / math.sqrt(s['P2'] * s['T2']),
                         cosine_P_H=s['P.H'] / math.sqrt(s['P2'] * s['H2']),
                         cosine_T_H=s['T.H'] / math.sqrt(s['T2'] * s['H2']))
    wsp.write(args.output, dict(schema='p2048-weight-space-geometry-v1', task_vectors_relative_to='L',
                                anchors={n: wsp.sha(soup.ROOT / soup.ANCHORS[n]) for n in names},
                                excluded=['temperature', 'act_head.*'], groups=report))
    print(json.dumps(report, indent=1))


if __name__ == '__main__':
    main()
