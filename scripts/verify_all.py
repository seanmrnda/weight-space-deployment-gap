#!/usr/bin/env python3
"""Recompute the paper's main results (every table and figure of its results and deployment sections) from the
released logits and labels, and check each value against the paper.

Standard library only. Runs on a CPU in about a minute. Exit status 0 means every check matched the paper.
Bootstrap intervals use the study's exact procedure: paired component-cluster resampling, 4,000 draws, seed 2049,
with rows in frozen plan order, so the intervals reproduce digit for digit.
"""
import collections, hashlib, json, math, random, sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
LOCKBOX_SOURCES = ('Re-DocRED', 'MAVEN-ERE', 'FEVEROUS')
EXTERNAL_SOURCES = ('DWIE', 'MATRES', 'AVeriTeC')
WORKFLOWS = ('agent_trace_observability', 'customer_service', 'invoice_processing', 'security_incidents')
SCREEN_PER_SOURCE = 200
CHECKS, FAILS = [], []


def read(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def labels(name):
    return read(REPO / f'data/labels/{name}.jsonl')


INDEX = json.loads((REPO / 'results/logits/index.json').read_text())


def logits(run, recipe):
    return {r['id']: r['logits'] for r in read(REPO / f'results/logits/{run}/{INDEX[run][recipe]}.jsonl')}


def argmax(values):
    return max(range(len(values)), key=values.__getitem__)


def correct(row, values):
    return argmax(values) == row['gold_index']


def accuracy(rows, values, pred=lambda r: True):
    sub = [r for r in rows if pred(r)]
    return 100 * sum(correct(r, values[r['id']]) for r in sub) / len(sub)


def bootstrap(rows, a, b, sources, draws=4000, seed=2049):
    groups = {s: collections.defaultdict(list) for s in sources}
    for r in rows:
        if r['source'] in groups:
            groups[r['source']][r['component']].append(int(correct(r, a[r['id']])) - int(correct(r, b[r['id']])))
    pools = {s: [(sum(v), len(v)) for v in g.values()] for s, g in groups.items()}
    rng = random.Random(seed)
    samples, macro = {s: [] for s in sources}, []
    for _ in range(draws):
        values = []
        for s in sources:
            pick = rng.choices(pools[s], k=len(pools[s]))
            values.append(sum(x for x, _ in pick) / sum(n for _, n in pick))
            samples[s].append(values[-1])
        macro.append(sum(values) / len(values))
    ci = lambda v: [100 * sorted(v)[int(.025 * draws)], 100 * sorted(v)[int(.975 * draws) - 1]]
    return {**{s: ci(samples[s]) for s in sources}, 'macro': ci(macro)}


def sign_test(fixed, broken):
    n, k = fixed + broken, min(fixed, broken)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n) if n else 1.0


def check(label, got, expected, tol=0.006):
    ok = abs(got - expected) <= tol
    CHECKS.append((label, got, expected, ok))
    if not ok:
        FAILS.append(label)


def check_ci(label, ci, lo, hi):
    check(label + ' lower', ci[0], lo)
    check(label + ' upper', ci[1], hi)


def macro_gain(rows, a, b, sources):
    return sum(accuracy(rows, a, lambda r, s=s: r['source'] == s) - accuracy(rows, b, lambda r, s=s: r['source'] == s)
               for s in sources) / len(sources)


def screen_rows(search):
    out = []
    for s in LOCKBOX_SOURCES:
        pool = sorted((r for r in search if r['source'] == s), key=lambda r: hashlib.sha256(f"screen:{r['id']}".encode()).hexdigest())
        out += pool[:SCREEN_PER_SOURCE]
    return out


def main():
    lock = labels('lockbox')
    # Table: lockbox accuracy.
    expected = {'L': (87.16, 81.90, 61.58, 76.88), 'P': (88.11, 84.29, 62.93, 78.44), 'T': (91.76, 79.05, 64.09, 78.30),
                'A5': (88.27, 83.81, 63.51, 78.53), 'P+T-L': (90.65, 82.38, 64.29, 79.11)}
    lg = {arm: logits('soup', arm) for arm in expected}
    for arm, vals in expected.items():
        accs = [accuracy(lock, lg[arm], lambda r, s=s: r['source'] == s) for s in LOCKBOX_SOURCES]
        for s, got, exp in zip(LOCKBOX_SOURCES, accs, vals):
            check(f'lockbox {arm} {s}', got, exp)
        check(f'lockbox {arm} macro', sum(accs) / 3, vals[3])
    ci = bootstrap(lock, lg['P+T-L'], lg['L'], LOCKBOX_SOURCES)
    for s, (lo, hi) in zip(('Re-DocRED', 'MAVEN-ERE', 'FEVEROUS', 'macro'), ((1.26, 5.79), (-2.96, 4.27), (-0.58, 5.78), (0.40, 3.99))):
        check_ci(f'lockbox P+T-L vs L {s}', ci[s], lo, hi)
    for ref, exp in (('P', 0.66), ('T', 0.81), ('A5', 0.57)):
        check(f'lockbox P+T-L vs {ref} macro gain', macro_gain(lock, lg['P+T-L'], lg[ref], LOCKBOX_SOURCES), exp)
    check_ci('lockbox A5 vs L macro', bootstrap(lock, lg['A5'], lg['L'], LOCKBOX_SOURCES)['macro'], 0.53, 2.88)
    mav = [r for r in lock if r['source'] == 'MAVEN-ERE']
    def balanced(values):
        by = collections.defaultdict(list)
        for r in mav:
            by[r['label']].append(correct(r, values[r['id']]))
        return 100 * sum(sum(v) / len(v) for v in by.values()) / len(by)
    check('lockbox MAVEN balanced accuracy P+T-L', balanced(lg['P+T-L']), 55.55)
    check('lockbox MAVEN balanced accuracy L', balanced(lg['L']), 54.36)

    # Figure: gain shrinkage across panels.
    search, validation = labels('search'), labels('validation')
    for name, rows, exp in (('screen', screen_rows(search), 3.67), ('search', search, 3.03), ('validation', validation, 1.64),
                            ('lockbox', lock, 2.22)):
        check(f'shrinkage {name}', macro_gain(rows, lg['P+T-L'], lg['L'], LOCKBOX_SOURCES), exp)

    # Table: external benchmark.
    ext = labels('external')
    el = {arm: logits('external', arm) for arm in ('L', 'P', 'T', 'A5', 'P+T-L')}
    for s, exp in zip(EXTERNAL_SOURCES, (64.40, 47.10, 29.47)):
        check(f'external L {s}', accuracy(ext, el['L'], lambda r, s=s: r['source'] == s), exp)
    for s, exp in (('MATRES', 58.56), ('AVeriTeC', 53.03)):
        sub = [r for r in ext if r['source'] == s]
        check(f'external majority baseline {s}', 100 * collections.Counter(r['label'] for r in sub).most_common(1)[0][1] / len(sub), exp)
    table = {'P+T-L': ((9.73, 7.40, 12.07), (8.98, 4.73, 14.22), (-1.58, -3.95, 0.79), (5.71, 3.83, 7.81)),
             'T': ((10.88, 8.60, 13.21), (2.07, -0.19, 4.77), (-0.26, -1.58, 1.05), (4.23, 3.07, 5.49)),
             'P': ((1.15, -0.63, 2.93), (8.43, 4.35, 13.64), (-0.13, -2.24, 1.97), (3.15, 1.54, 5.11)),
             'A5': ((0.90, -0.51, 2.28), (4.56, 2.10, 7.56), (0.13, -1.58, 1.71), (1.86, 0.79, 3.10))}
    for arm, cells in table.items():
        ci = bootstrap(ext, el[arm], el['L'], EXTERNAL_SOURCES)
        for s, (gain, lo, hi) in zip(EXTERNAL_SOURCES + ('macro',), cells):
            got = macro_gain(ext, el[arm], el['L'], EXTERNAL_SOURCES) if s == 'macro' else \
                accuracy(ext, el[arm], lambda r, s=s: r['source'] == s) - accuracy(ext, el['L'], lambda r, s=s: r['source'] == s)
            check(f'external {arm} {s} gain', got, gain)
            check_ci(f'external {arm} {s}', ci[s], lo, hi)
    matres = [r for r in ext if r['source'] == 'MATRES']
    for arm, before, after in (('L', 166, 173), ('P+T-L', 340, 66)):
        check(f'MATRES {arm} BEFORE recall', sum(correct(r, el[arm][r['id']]) for r in matres if r['label'] == 'BEFORE'), before, 0)
        check(f'MATRES {arm} AFTER recall', sum(correct(r, el[arm][r['id']]) for r in matres if r['label'] == 'AFTER'), after, 0)
    for arm in el:
        acc = accuracy(ext, el[arm], lambda r: r['source'] == 'AVeriTeC')
        CHECKS.append((f'AVeriTeC {arm} within 27.9-29.6', acc, '27.9..29.6', 27.85 <= acc <= 29.65))
        if not 27.85 <= acc <= 29.65:
            FAILS.append(f'AVeriTeC {arm}')

    # Table: release guards.
    leg = labels('legacy')
    ll = {arm: logits('legacy', arm) for arm in ('L', 'P', 'T', 'A5', 'P+T-L')}
    slices = {'choice': lambda r: r['type'] == 'choice', 'noul': lambda r: r['type'] == 'noul', 'score': lambda r: r['type'] == 'score',
              **{w: (lambda r, w=w: r['source'] == w) for w in WORKFLOWS}}
    deltas = {'P+T-L': (0.17, -0.37, 0.00, -5.00, -5.00, -3.00, -7.50), 'P': (0.53, 0.31, 0.71, -2.50, -2.50, -3.50, -6.50),
              'T': (-0.12, 0.18, 0.81, -3.50, 0.00, 0.00, -3.00), 'A5': (-0.19, 0.67, 1.62, 0.00, -0.50, 1.50, -3.00)}
    sec = [r for r in leg if r['source'] == 'security_incidents' and r['type'] == 'noul' and r['gold_index'] == 1]
    check('legacy security positives', len(sec), 54, 0)
    check('legacy security positives with soft yes >= 0.8', sum(r['soft_yes'] >= 0.8 for r in sec), 9, 0)
    check('legacy security FN L', sum(not correct(r, ll['L'][r['id']]) for r in sec), 8, 0)
    wf = [r for r in leg if r['source'] in WORKFLOWS]
    workflow_expect = {'P+T-L': (32, 73, 0.00008), 'P': (33, 63, 0.003), 'T': (12, 25, 0.047), 'A5': (22, 26, 0.67)}
    fn_expect = {'P+T-L': 14, 'P': 13, 'T': 11, 'A5': 11}
    for arm, vals in deltas.items():
        for (name, pred), exp in zip(slices.items(), vals):
            check(f'legacy {arm} {name} delta', accuracy(leg, ll[arm], pred) - accuracy(leg, ll['L'], pred), exp)
        check(f'legacy {arm} security FN', sum(not correct(r, ll[arm][r['id']]) for r in sec), fn_expect[arm], 0)
        f = sum(correct(r, ll[arm][r['id']]) and not correct(r, ll['L'][r['id']]) for r in wf)
        b = sum(not correct(r, ll[arm][r['id']]) and correct(r, ll['L'][r['id']]) for r in wf)
        fx, bx, px = workflow_expect[arm]
        check(f'legacy {arm} workflows fixed', f, fx, 0)
        check(f'legacy {arm} workflows broken', b, bx, 0)
        p = sign_test(f, b)
        # Compare at the paper's printed precision: within half a unit of its last digit.
        check(f'legacy {arm} workflow sign-test p', p, px, 0.5 * 10 ** math.floor(math.log10(px)))

    # Case-level sensitivity for the workflow sign tests (post hoc; 160 cases of five questions).
    check('legacy workflow cases', len({r['component'] for r in wf}), 160, 0)
    check('legacy security positive cases', len({r['component'] for r in sec}), 28, 0)
    case_expect = {'P+T-L': (20, 53, 0.0001), 'P': (21, 44, 0.006), 'T': (11, 23, 0.058), 'A5': (19, 22, 0.76)}
    for arm, (ux, dx, px) in case_expect.items():
        net = collections.Counter()
        for r in wf:
            net[r['component']] += int(correct(r, ll[arm][r['id']])) - int(correct(r, ll['L'][r['id']]))
        up, down = sum(v > 0 for v in net.values()), sum(v < 0 for v in net.values())
        check(f'legacy {arm} cases improved', up, ux, 0)
        check(f'legacy {arm} cases worsened', down, dx, 0)
        check(f'legacy {arm} case-level sign-test p', sign_test(up, down), px, 0.5 * 10 ** math.floor(math.log10(px)))

    # Synthetic benchmark: clear-cut security and the temporal prior shift.
    syn = labels('synthetic')
    sl = {arm: logits('synthetic', arm) for arm in ('L', 'P', 'T', 'A5', 'P+T-L')}
    pos = [r for r in syn if r['source'] == 'security' and r['gold_index'] == 1]
    neg = [r for r in syn if r['source'] == 'security' and r['gold_index'] == 0]
    for arm, miss, fa in (('L', 3, 0), ('T', 4, 0), ('A5', 15, 0), ('P', 21, 10), ('P+T-L', 24, 6)):
        check(f'synthetic {arm} missed incidents', sum(not correct(r, sl[arm][r['id']]) for r in pos), miss, 0)
        check(f'synthetic {arm} false alarms', sum(not correct(r, sl[arm][r['id']]) for r in neg), fa, 0)
    for arm, broken in (('A5', 12), ('P', 18), ('P+T-L', 21)):
        f = sum(correct(r, sl[arm][r['id']]) and not correct(r, sl['L'][r['id']]) for r in pos)
        b = sum(not correct(r, sl[arm][r['id']]) and correct(r, sl['L'][r['id']]) for r in pos)
        check(f'synthetic {arm} positives fixed', f, 0, 0)
        check(f'synthetic {arm} positives broken', b, broken, 0)
    check('synthetic A5 sign-test p', sign_test(0, 12), 0.0005, 0.00002)
    check('synthetic positive judgments', len(pos), 50, 0)
    check('synthetic positive cases', len({r['component'] for r in pos}), 25, 0)
    syn_p = sl['P']
    check('synthetic P false negatives on credential questions',
          sum(not correct(r, syn_p[r['id']]) for r in pos if r['id'].endswith(':credential_compromise')), 21, 0)
    check('synthetic P false negatives on malicious-activity questions',
          sum(not correct(r, syn_p[r['id']]) for r in pos if r['id'].endswith(':true_positive')), 0, 0)
    f = sum(correct(r, sl['P+T-L'][r['id']]) and not correct(r, sl['L'][r['id']]) for r in pos)
    broken_cases = {r['component'] for r in pos if not correct(r, sl['P+T-L'][r['id']]) and correct(r, sl['L'][r['id']])}
    check('synthetic P+T-L broken positives span cases', len(broken_cases), 20, 0)
    temporal = [r for r in syn if r['source'] == 'temporal']
    for arm, counts in (('P+T-L', (30, 8, 0)), ('L', (9, 15, 6))):
        for lab, exp in zip(('BEFORE', 'AFTER', 'SIMULTANEOUS'), counts):
            check(f'synthetic temporal {arm} {lab}', sum(correct(r, sl[arm][r['id']]) for r in temporal if r['label'] == lab), exp, 0)

    # Table: interpolation L + c(T - L).
    blend = {'L+0.25(T-L)': ((1.43, 0.48, 2.48), (3.33, 2.02, 4.71), (1.80, 1.19, 2.40), -1.0, 9, 3),
             'L+0.5(T-L)': ((3.01, 1.56, 4.55), (6.15, 4.38, 8.05), (3.20, 2.37, 4.02), -1.0, 10, 3),
             'L+0.75(T-L)': ((3.65, 2.03, 5.30), (8.84, 6.78, 10.98), (3.83, 2.89, 4.77), -2.0, 11, 3),
             'T': ((4.60, 2.83, 6.43), (10.88, 8.60, 13.21), (4.23, 3.07, 5.49), -3.5, 11, 4)}
    bl_lock = {a: logits('soup', a) for a in blend}
    bl_ext = {a: logits('external', a) for a in blend}
    bl_leg = {a: logits('legacy', a) for a in blend}
    bl_syn = {a: logits('synthetic', a) for a in blend}
    for arm, (rd, dw, em, worst, fn, miss) in blend.items():
        c1 = bootstrap(lock, bl_lock[arm], lg['L'], LOCKBOX_SOURCES)
        c2 = bootstrap(ext, bl_ext[arm], el['L'], EXTERNAL_SOURCES)
        check(f'blend {arm} Re-DocRED gain', accuracy(lock, bl_lock[arm], lambda r: r['source'] == 'Re-DocRED') - accuracy(lock, lg['L'], lambda r: r['source'] == 'Re-DocRED'), rd[0])
        check_ci(f'blend {arm} Re-DocRED', c1['Re-DocRED'], rd[1], rd[2])
        check(f'blend {arm} DWIE gain', accuracy(ext, bl_ext[arm], lambda r: r['source'] == 'DWIE') - accuracy(ext, el['L'], lambda r: r['source'] == 'DWIE'), dw[0])
        check_ci(f'blend {arm} DWIE', c2['DWIE'], dw[1], dw[2])
        check(f'blend {arm} external macro gain', macro_gain(ext, bl_ext[arm], el['L'], EXTERNAL_SOURCES), em[0])
        check_ci(f'blend {arm} external macro', c2['macro'], em[1], em[2])
        check(f'blend {arm} worst legacy slice', min(accuracy(leg, bl_leg[arm], p) - accuracy(leg, ll['L'], p) for p in slices.values()), worst, 0.006)
        check(f'blend {arm} legacy FN', sum(not correct(r, bl_leg[arm][r['id']]) for r in sec), fn, 0)
        check(f'blend {arm} synthetic misses', sum(not correct(r, bl_syn[arm][r['id']]) for r in pos), miss, 0)

    # Analysis section: full-encoder distance of T from L, and decision-module substitutions (receipts).
    g = json.loads((REPO / 'results/receipts/soup/geometry.json').read_text())['groups']
    groups = ('encoder_rest', 'experts', 'routers')
    num = sum((g[k]['relative_percent_T'] / 100 * g[k]['base_l2']) ** 2 for k in groups)
    den = sum(g[k]['base_l2'] ** 2 for k in groups)
    check('T-L full-encoder relative distance (%)', 100 * math.sqrt(num / den), 0.051, 0.0006)
    check('T-L non-expert, non-router encoder distance (%)', g['encoder_rest']['relative_percent_T'], 0.037, 0.0006)
    mods = {i['id']: i['scores'] for i in json.loads((REPO / 'results/receipts/soup/modules-report.json').read_text())['table']}
    base = mods['P+T-L']
    worst = max(abs(mods[v][s]['correct'] - base[s]['correct']) for v in ('noT-decision', 'noP-decision', 'L-decision', 'half-decision')
                for s in LOCKBOX_SOURCES)
    check('decision-module substitutions: max net change in correct count per source', worst, 2, 0)

    width = max(len(c[0]) for c in CHECKS)
    for label, got, exp, ok in CHECKS:
        shown = (f'{got:.4f}' if abs(got) >= 0.01 or got == 0 else f'{got:.3g}') if isinstance(got, float) else str(got)
        print(f"{'PASS' if ok else 'FAIL'}  {label:<{width}}  computed {shown:>10}  paper {exp}")
    print(f'\n{len(CHECKS) - len(FAILS)} of {len(CHECKS)} checks match the paper.')
    sys.exit(1 if FAILS else 0)


if __name__ == '__main__':
    main()
