"""Apply the pre-registered L+c(T-L) blend rule (docs/p2048-weight-space-blend-LT.md); write one receipt.

G1 legacy guards vs L, G2 external macro >= L - 2, G3 synthetic security: no paired loss at p < 0.05 on the
100 security questions and clear-incident misses <= L + 2. Selection: highest lockbox macro among passing
blends, ties to smaller c; otherwise L stays. Reads saved predictions only.
"""
import argparse, json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import p2048_weight_space_deploy as d
from p2048_weight_space_synthetic_report import sign_test

LOCKBOX = ('Re-DocRED', 'MAVEN-ERE', 'FEVEROUS')
SYNTHETIC = ('relations', 'temporal', 'verdict', 'security')
SLICES = ('choice', 'noul', 'score') + d.WORKFLOWS


def complete(run, role, arms):
    rows = d.rows_for(run, role)
    logits = {name: d.logits(run, recipe) for name, recipe in arms.items()}
    for name, values in logits.items():
        missing = sum(r['id'] not in values for r in rows)
        if missing:
            raise SystemExit(f'{run}/{role}: {name} is missing {missing} rows')
    return rows, logits


def acc(rows, values, pred):
    sub = [r for r in rows if pred(r)]
    return 100 * sum(d.correct(r, values[r['id']]) for r in sub) / len(sub), len(sub)


def paired(rows, a, b, pred=lambda r: True):
    sub = [r for r in rows if pred(r)]
    fixed = sum(d.correct(r, a[r['id']]) and not d.correct(r, b[r['id']]) for r in sub)
    broken = sum(not d.correct(r, a[r['id']]) and d.correct(r, b[r['id']]) for r in sub)
    return dict(fixed=fixed, broken=broken, p=sign_test(fixed, broken))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arms', default=str(d.RUNS / 'soup/recipes/blend-LT.json'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    arms = {r['id']: r for r in json.loads(Path(args.arms).read_text())}
    coef = {name: (r['rules'][0]['coefs'].get('T', 0.0)) for name, r in arms.items()}
    order = sorted(arms, key=coef.get)
    report = dict(schema='p2048-weight-space-blend-report-v1', arms=order, c=coef, panels={}, gates={})

    rows, lg = complete('soup', 'lockbox', arms)
    report['panels']['lockbox'] = {}
    for n in order:
        per = {s: acc(rows, lg[n], lambda r, s=s: r['source_family'] == s)[0] for s in LOCKBOX}
        item = dict(accuracy=per, macro=sum(per.values()) / 3)
        if n != 'L':
            item['interval_vs_L'] = d.bootstrap(rows, lg[n], lg['L'], LOCKBOX)
        report['panels']['lockbox'][n] = item

    rows, lg = complete('legacy', 'legacy', arms)
    security = lambda r: r.get('source_family') == 'security_incidents' and r['type'] == 'noul' and int(r['hard_label_index']) == 1
    preds = {s: (lambda r, s=s: r['type'] == s) if s in ('choice', 'noul', 'score') else (lambda r, s=s: r.get('source_family') == s)
             for s in SLICES}
    report['panels']['legacy'] = {}
    base = {s: acc(rows, lg['L'], preds[s]) for s in SLICES}
    fn_l = sum(not d.correct(r, lg['L'][r['id']]) for r in rows if security(r))
    for n in order:
        slices = {}
        for s in SLICES:
            a, count = acc(rows, lg[n], preds[s])
            slices[s] = dict(rows=count, accuracy=a, delta=a - base[s][0], passes=(a - base[s][0] >= -2.0) if count >= 200 else None)
        fn = sum(not d.correct(r, lg[n][r['id']]) for r in rows if security(r))
        workflows = paired(rows, lg[n], lg['L'], lambda r: r.get('source_family') in d.WORKFLOWS)
        report['panels']['legacy'][n] = dict(slices=slices, security_fn=fn, security_fn_L=fn_l, workflows_vs_L=workflows,
                                             G1=all(v['passes'] is not False for v in slices.values()) and fn <= fn_l)

    rows, lg = complete('external', 'external', arms)
    report['panels']['external'] = {}
    for n in order:
        per = {s: acc(rows, lg[n], lambda r, s=s: r['source_family'] == s)[0] for s in d.EXTERNAL}
        item = dict(accuracy=per, macro=sum(per.values()) / 3)
        if n != 'L':
            item['interval_vs_L'] = d.bootstrap(rows, lg[n], lg['L'], d.EXTERNAL)
        report['panels']['external'][n] = item

    rows, lg = complete('synthetic', 'synthetic', arms)
    report['panels']['synthetic'] = {}
    pos = lambda r: r['source_family'] == 'security' and int(r['hard_label_index']) == 1
    sec = lambda r: r['source_family'] == 'security'
    misses_l = sum(not d.correct(r, lg['L'][r['id']]) for r in rows if pos(r))
    for n in order:
        per = {s: acc(rows, lg[n], lambda r, s=s: r['source_family'] == s)[0] for s in SYNTHETIC}
        misses = sum(not d.correct(r, lg[n][r['id']]) for r in rows if pos(r))
        false_alarms = sum(not d.correct(r, lg[n][r['id']]) for r in rows if sec(r) and not pos(r))
        test = paired(rows, lg[n], lg['L'], sec)
        report['panels']['synthetic'][n] = dict(accuracy=per, misses=misses, false_alarms=false_alarms, security_vs_L=test,
                                                G3=not (test['broken'] > test['fixed'] and test['p'] < 0.05) and misses <= misses_l + 2)

    ext_l = report['panels']['external']['L']['macro']
    eligible = []
    for n in order:
        if n == 'L':
            continue
        g = dict(G1=report['panels']['legacy'][n]['G1'], G2=report['panels']['external'][n]['macro'] >= ext_l - 2.0,
                 G3=report['panels']['synthetic'][n]['G3'], blend=n not in ('L', 'T'))
        report['gates'][n] = g
        if g['G1'] and g['G2'] and g['G3'] and g['blend']:
            eligible.append(n)
    chosen = max(eligible, key=lambda n: (report['panels']['lockbox'][n]['macro'], -coef[n])) if eligible else 'L'
    report['selected'] = chosen
    d.wsp.write(args.output, report)

    print(f"{'arm':13}{'c':>5}{'lockbox':>9}{'Re-Doc':>8}{'DWIE':>7}{'ext':>7}{'wf f/b':>9}{'secFN':>7}{'synSec':>8}{'miss':>6}  G1 G2 G3")
    for n in order:
        lb, lgc, ex, sy = (report['panels'][p][n] for p in ('lockbox', 'legacy', 'external', 'synthetic'))
        g = report['gates'].get(n, {})
        print(f"{n:13}{coef[n]:5.2f}{lb['macro']:9.2f}{lb['accuracy']['Re-DocRED']:8.2f}{ex['accuracy']['DWIE']:7.2f}{ex['macro']:7.2f}"
              f"{lgc['workflows_vs_L']['fixed']:>4}/{lgc['workflows_vs_L']['broken']:<4}{lgc['security_fn']:>5}{sy['accuracy']['security']:8.1f}"
              f"{sy['misses']:>6}  " + ' '.join(str(g.get(k, '-'))[0] for k in ('G1', 'G2', 'G3')))
    for n in order:
        failing = {s: round(v['delta'], 2) for s, v in report['panels']['legacy'][n]['slices'].items() if v['passes'] is False}
        if failing:
            print('  ', n, 'G1 failing slices', failing)
    print('selected:', chosen)


if __name__ == '__main__':
    main()
