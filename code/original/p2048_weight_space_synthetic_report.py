"""Apply the pre-registered synthetic-benchmark rule to saved predictions; write one receipt.

Rule (docs/p2048-weight-space-synthetic-benchmark.md): a section is informative if some arm scores at least
15 points above chance; security is judged on misses of the 50 clear positives against L with a two-sided
exact sign test; gains count only when the 95% component-bootstrap interval excludes zero.
"""
import argparse, json, math, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import p2048_weight_space_deploy as d

SECTIONS = ('relations', 'temporal', 'verdict', 'security')
CHANCE = dict(relations=25.0, temporal=100 / 3, verdict=100 / 3, security=50.0)


def sign_test(fixed, broken):
    n, k = fixed + broken, min(fixed, broken)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n) if n else 1.0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arms', default=str(d.RUNS / 'soup/recipes/lockbox.json'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    arms = {r['id']: r for r in json.loads(Path(args.arms).read_text())}
    rows = d.rows_for('synthetic', 'synthetic')
    logits = {name: d.logits('synthetic', recipe) for name, recipe in arms.items()}
    for name, values in logits.items():
        missing = sum(r['id'] not in values for r in rows)
        if missing:
            raise SystemExit(f'synthetic predictions incomplete for {name}: {missing} rows missing')
    ok = {name: {r['id']: d.correct(r, values[r['id']]) for r in rows} for name, values in logits.items()}
    positives = [r for r in rows if r['source_family'] == 'security' and int(r['hard_label_index']) == 1]
    negatives = [r for r in rows if r['source_family'] == 'security' and int(r['hard_label_index']) == 0]
    report = dict(schema='p2048-weight-space-synthetic-report-v1', arms=list(arms), chance=CHANCE, results={})
    for name in arms:
        acc = {s: 100 * sum(ok[name][r['id']] for r in rows if r['source_family'] == s) /
               sum(r['source_family'] == s for r in rows) for s in SECTIONS}
        item = dict(accuracy=acc, macro=sum(acc.values()) / len(acc),
                    security_misses=sum(not ok[name][r['id']] for r in positives),
                    security_false_alarms=sum(not ok[name][r['id']] for r in negatives),
                    by_label={s: {} for s in SECTIONS})
        for r in rows:
            cell = item['by_label'][r['source_family']].setdefault(str(r['label']), [0, 0])
            cell[0] += ok[name][r['id']]
            cell[1] += 1
        temporal_rev = [r for r in rows if r['source_family'] == 'temporal' and r['provenance'].get('reversed_mention_order')]
        temporal_fwd = [r for r in rows if r['source_family'] == 'temporal' and r['label'] != 'SIMULTANEOUS'
                        and not r['provenance'].get('reversed_mention_order')]
        item['temporal_mention_order'] = dict(forward=[sum(ok[name][r['id']] for r in temporal_fwd), len(temporal_fwd)],
                                              reversed=[sum(ok[name][r['id']] for r in temporal_rev), len(temporal_rev)])
        if name != 'L':
            item['vs_L'] = dict(interval=d.bootstrap(rows, logits[name], logits['L'], SECTIONS), sections={})
            for s in SECTIONS + ('security_positives',):
                subset = positives if s == 'security_positives' else [r for r in rows if r['source_family'] == s]
                fixed = sum(ok[name][r['id']] and not ok['L'][r['id']] for r in subset)
                broken = sum(not ok[name][r['id']] and ok['L'][r['id']] for r in subset)
                item['vs_L']['sections'][s] = dict(fixed=fixed, broken=broken, p=sign_test(fixed, broken))
            item['vs_L']['gain'] = {s: acc[s] - 0 for s in SECTIONS}  # filled below once L is known
        report['results'][name] = item
    base = report['results']['L']
    informative = {s: max(report['results'][n]['accuracy'][s] for n in arms) >= CHANCE[s] + 15 for s in SECTIONS}
    report['informative'] = informative
    verdicts = {}
    for name in arms:
        if name == 'L':
            continue
        item = report['results'][name]
        item['vs_L']['gain'] = {s: item['accuracy'][s] - base['accuracy'][s] for s in SECTIONS}
        item['vs_L']['gain']['macro'] = item['macro'] - base['macro']
        pos = item['vs_L']['sections']['security_positives']
        if pos['broken'] > pos['fixed'] and pos['p'] < 0.05:
            verdicts[name] = 'confirmed: misses more clear incidents than L'
        elif item['security_misses'] <= base['security_misses']:
            verdicts[name] = 'no extra misses on clear incidents'
        else:
            verdicts[name] = 'unresolved'
    report['security_verdict'] = verdicts
    d.wsp.write(args.output, report)

    print(f"{'arm':8}" + ''.join(f'{s:>11}' for s in SECTIONS) + f"{'macro':>8}{'misses':>8}{'false+':>8}")
    for name in arms:
        r = report['results'][name]
        print(f'{name:8}' + ''.join(f"{r['accuracy'][s]:11.1f}" for s in SECTIONS)
              + f"{r['macro']:8.1f}{r['security_misses']:>5}/50{r['security_false_alarms']:>5}/50")
    print('informative', informative)
    for name, v in verdicts.items():
        iv = report['results'][name]['vs_L']['interval']
        print(name, 'security:', v, '| gains', {s: round(report['results'][name]['vs_L']['gain'][s], 1) for s in (*SECTIONS, 'macro')},
              '| CI', {s: [round(x, 1) for x in iv[s]] for s in iv})


if __name__ == '__main__':
    main()
