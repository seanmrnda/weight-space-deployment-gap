"""Apply the pre-registered deployment rule to legacy guards and the external benchmark.

G1: repository legacy guards against L. Choice, Noul and Score accuracy may each
fall at most 2 points; workflow slices under 200 rows are reported, not gated;
security-incident Noul positives may have no more false negatives than L.
G2: external macro accuracy (DWIE, MATRES, AVeriTeC) at most 2 points below L.
Selection: among non-L candidates passing both, the highest lockbox macro.
Reads saved predictions only, and writes one decision receipt.
"""
import argparse, collections, json, random, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import p2048_weight_space as wsp
import p2048_weight_space_soup as soup

RUNS = soup.ROOT / 'research/weight-space/runs'
EXTERNAL = ('DWIE', 'MATRES', 'AVeriTeC')
WORKFLOWS = ('agent_trace_observability', 'customer_service', 'invoice_processing', 'security_incidents')



def rows_for(run, role):
    plan = json.loads((RUNS / run / 'plan.json').read_text())
    if wsp.sha(RUNS / run / 'data.jsonl') != plan['data_sha256']:
        raise ValueError(f'{run}: data digest differs from plan')
    data = {r['id']: r for r in map(json.loads, (RUNS / run / 'data.jsonl').open())}
    return [dict(data[i['id']], source_family=i['source'], connected_component=i['component']) for i in plan['roles'][role]]


def logits(run, recipe):
    soup.RUN = RUNS / run
    return soup.predictions(recipe)


def correct(row, values):
    return wsp.argmax(values) == wsp.gold_index(row)


def bootstrap(rows, a, b, sources, draws=4000, seed=2049):
    """Paired component bootstrap of accuracy deltas per source and their macro."""
    groups = {s: collections.defaultdict(list) for s in sources}
    for row in rows:
        if row['source_family'] in groups:
            groups[row['source_family']][row['connected_component']].append(
                int(correct(row, a[row['id']])) - int(correct(row, b[row['id']])))
    pools = {s: [(sum(v), len(v)) for v in g.values()] for s, g in groups.items()}
    rng = random.Random(seed)
    samples = {s: [] for s in sources}
    macro = []
    for _ in range(draws):
        values = []
        for s in sources:
            chosen = rng.choices(pools[s], k=len(pools[s]))
            values.append(sum(x for x, _ in chosen) / sum(n for _, n in chosen))
            samples[s].append(values[-1])
        macro.append(sum(values) / len(values))
    def interval(v):
        v = sorted(v)
        return [100 * v[int(.025 * draws)], 100 * v[int(.975 * draws) - 1]]
    return dict(**{s: interval(samples[s]) for s in sources}, macro=interval(macro))


def accuracy(rows, values, predicate):
    subset = [r for r in rows if predicate(r)]
    return (sum(correct(r, values[r['id']]) for r in subset) / len(subset) if subset else None), len(subset)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arms', default=str(RUNS / 'soup/recipes/lockbox.json'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    arms = {r['id']: r for r in json.loads(Path(args.arms).read_text())}
    lockbox = json.loads((RUNS / 'soup/lockbox-report.json').read_text())
    lockbox_macro = {item['id']: item['scores']['macro'] for item in lockbox['table']}

    legacy_rows = rows_for('legacy', 'legacy')
    legacy = {name: logits('legacy', recipe) for name, recipe in arms.items()}
    for name, values in legacy.items():
        missing = sum(r['id'] not in values for r in legacy_rows)
        if missing:
            raise SystemExit(f'legacy predictions incomplete for {name}: {missing} rows missing')
    security = [r for r in legacy_rows if r.get('source_family') == 'security_incidents' and r['type'] == 'noul'
                and int(r['hard_label_index']) == 1]
    report = dict(schema='p2048-weight-space-deploy-v1', arms=list(arms), legacy={}, external={}, decision={})
    base = legacy['L']
    for name, values in legacy.items():
        slices = {}
        for label, predicate in [*[(t, lambda r, t=t: r['type'] == t) for t in ('choice', 'noul', 'score')],
                                 *[(w, lambda r, w=w: r.get('source_family') == w) for w in WORKFLOWS]]:
            acc, n = accuracy(legacy_rows, values, predicate)
            ref, _ = accuracy(legacy_rows, base, predicate)
            delta = 100 * (acc - ref)
            slices[label] = dict(rows=n, accuracy=100 * acc, delta_vs_L=delta,
                                 gated=n >= 200, passes=(delta >= -2.0) if n >= 200 else None)
        fn = sum(not correct(r, values[r['id']]) for r in security)
        fn_l = sum(not correct(r, base[r['id']]) for r in security)
        g1 = all(s['passes'] is not False for s in slices.values()) and fn <= fn_l
        report['legacy'][name] = dict(slices=slices, security_positives=len(security), security_false_negatives=fn,
                                      security_false_negatives_L=fn_l, G1=g1)

    external_rows = rows_for('external', 'external')
    external = {name: logits('external', recipe) for name, recipe in arms.items()}
    for name, values in external.items():
        missing = sum(r['id'] not in values for r in external_rows)
        if missing:
            raise SystemExit(f'external predictions incomplete for {name}: {missing} rows missing')
    majority = {}
    for s in EXTERNAL:
        labels = collections.Counter(r['label'] for r in external_rows if r['source_family'] == s)
        majority[s] = 100 * labels.most_common(1)[0][1] / sum(labels.values())
    report['external_majority_baseline'] = majority
    for name, values in sorted(external.items(), key=lambda kv: kv[0] != 'L'):  # L first: others compare to it
        per = {s: 100 * accuracy(external_rows, values, lambda r, s=s: r['source_family'] == s)[0] for s in EXTERNAL}
        macro = sum(per.values()) / 3
        recall = {}
        for s in EXTERNAL:
            support, hit = collections.Counter(), collections.Counter()
            for r in external_rows:
                if r['source_family'] == s:
                    support[r['label']] += 1
                    hit[r['label']] += correct(r, values[r['id']])
            recall[s] = {k: [hit[k], support[k]] for k, _ in support.most_common()}
        item = dict(accuracy=per, macro=macro, recall=recall,
                    n={s: sum(r['source_family'] == s for r in external_rows) for s in EXTERNAL})
        if name != 'L':
            item['vs_L'] = dict(gain={s: per[s] - report['external']['L']['accuracy'][s] for s in EXTERNAL},
                                interval=bootstrap(external_rows, values, external['L'], EXTERNAL))
            item['vs_L']['gain']['macro'] = macro - report['external']['L']['macro']
        report['external'][name] = item

    eligible = []
    for name in arms:
        if name == 'L':
            continue
        g1 = report['legacy'][name]['G1']
        g2 = report['external'][name]['macro'] >= report['external']['L']['macro'] - 2.0
        report['decision'][name] = dict(G1=g1, G2=g2, lockbox_macro=100 * lockbox_macro[name])
        if g1 and g2:
            eligible.append(name)
    chosen = max(eligible, key=lambda n: lockbox_macro[n]) if eligible else 'L'
    report['selected'] = chosen
    report['rule'] = __doc__.split('\n\n')[1]
    wsp.write(args.output, report)

    print(f"{'arm':8}{'G1':>5}{'G2':>5}{'lockbox':>9}{'ext.macro':>10}" + ''.join(f'{s:>10}' for s in EXTERNAL) + '   security FN')
    for name in arms:
        d = report['decision'].get(name, dict(G1='-', G2='-', lockbox_macro=100 * lockbox_macro[name]))
        e = report['external'][name]
        print(f"{name:8}{str(d['G1']):>5}{str(d['G2']):>5}{d['lockbox_macro']:9.2f}{e['macro']:10.2f}"
              + ''.join(f"{e['accuracy'][s]:10.2f}" for s in EXTERNAL)
              + f"   {report['legacy'][name]['security_false_negatives']}/{len(security)}")
    print('majority baselines', {s: round(v, 2) for s, v in majority.items()})
    print('selected:', chosen)


if __name__ == '__main__':
    main()
