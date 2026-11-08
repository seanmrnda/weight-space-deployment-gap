"""Per-module affine weight recipes over same-lineage checkpoints.

A recipe assigns each parameter an affine combination of anchor checkpoints:
``theta[key] = sum(coef[a] * anchor[a][key])``. Rules are ordered regexes and
the first match wins. Unmatched parameters stay at L, as do all buffers and the
unused action head. There is no optimizer: every candidate is a closed-form
parameter edit scored with normal FP32 forwards.

Forwards append to one JSONL per recipe fingerprint, so a run can stop and
resume, and a larger panel only pays for rows it has not scored. Several workers
may share a recipe list; each recipe is claimed with an exclusive lock file.

The lockbox role is refused unless a freeze file names the recipe being scored.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
import p2048_weight_space as wsp

ROOT = wsp.ROOT
RUN = ROOT / 'research/weight-space/runs/soup'
SOUP_RUN = RUN   # the parity gate always uses this run's rows, whatever --run-dir selects
T_RESULTS = '.local/praxis-p2048-targeted-02/kaggle/downloaded/results/'
SNAPSHOT = T_RESULTS + 'snapshots/window-{:05d}.safetensors'
ANCHORS = dict(
    L='research/weight-space/weights/base-p2048/model.safetensors',
    P='.local/praxis-p-2048-01/kaggle/downloaded/results/best/model.safetensors',
    H='.local/praxis-next-run-03/recovered-output/H/best/model.safetensors',
    v3='.local/v3/kaggle/output/defense-praxis/model.safetensors',
    v4='.local/v4/kaggle/output/defense-praxis/selected_model/model.safetensors',
    pub='.local/publication/praxis/model.safetensors',
    A5='research/weight-space/runs/ancestry/selected-checkpoint/model.safetensors',
    **{f'T{w:04d}': SNAPSHOT.format(w) for w in range(250, 3001, 250)},
)
ANCHORS['T'] = ANCHORS['T2500']   # "best" differs from window 2500 only by temperature
SOURCES = ('Re-DocRED', 'MAVEN-ERE', 'FEVEROUS')
SCREEN_PER_SOURCE = 200
TEMPERATURE = 1.1094255447387695  # L Choice calibration; argmax does not depend on it


def fingerprint(recipe):
    return wsp.digest(dict(rules=recipe['rules']))[:16]


def cache_directory():
    """Logits are only meaningful for the exact row bytes that produced them."""
    return RUN / 'predictions' / wsp.sha(RUN / 'data.jsonl')[:12]


def single(anchor):
    return dict(id=anchor, rules=[dict(match='.*', coefs={anchor: 1.0})])


def panel_rows(role, data=None):
    plan = json.loads((RUN / 'plan.json').read_text())
    if wsp.sha(RUN / 'data.jsonl') != plan['data_sha256']:
        raise ValueError('Data digest differs from frozen plan')
    data = data or {r['id']: r for r in map(json.loads, (RUN / 'data.jsonl').open())}
    if role == 'screen':
        chosen = []
        for source in SOURCES:
            pool = [i for i in plan['roles']['search'] if i['source'] == source]
            pool.sort(key=lambda i: hashlib.sha256(f"screen:{i['id']}".encode()).hexdigest())
            chosen += pool[:SCREEN_PER_SOURCE]
    else:
        chosen = plan['roles'][role]
    return [dict(data[i['id']], source_family=i['source'], connected_component=i['component']) for i in chosen]


def require_freeze(role, recipes, freeze):
    if role != 'lockbox':
        return
    if freeze is None:
        raise SystemExit('The lockbox needs --freeze with the frozen recipe and reference arms')
    frozen = json.loads(Path(freeze).read_text())
    allowed = {frozen['selected_fingerprint'], *frozen['reference_fingerprints']}
    extra = {fingerprint(r) for r in recipes} - allowed
    if extra:
        raise SystemExit(f'Lockbox refuses recipes outside the freeze: {sorted(extra)}')


class Editor:
    """Applies recipes to a loaded model and always rebuilds from L."""

    def __init__(self, torch, model, order):
        from safetensors import safe_open
        self.torch, self.model = torch, model
        self.parameters = {k: v for k, v in model.named_parameters() if not k.startswith('act_head.')}
        self.handles = {}
        self.safe_open = safe_open
        self.order = order

    def tensor(self, anchor, key):
        if anchor not in self.handles:
            self.handles[anchor] = self.safe_open(str(ROOT / ANCHORS[anchor]), framework='pt', device='cpu')
        return self.handles[anchor].get_tensor(key).float()

    def apply(self, recipe):
        rules = [(re.compile(r['match']), r['coefs']) for r in recipe['rules']]
        with self.torch.no_grad():
            for key, parameter in self.parameters.items():
                coefs = next((c for pattern, c in rules if pattern.search(key)), {'L': 1.0})
                # Summed as L + sum(c_i (A_i - L)) so edits stay accurate at FP32 near L.
                base = self.tensor('L', key)
                total = sum(coefs.values())
                if abs(total - 1.0) > 1e-9:
                    raise ValueError(f'Affine coefficients must sum to 1, got {total}')
                result = base.clone()
                for anchor, coef in coefs.items():
                    if anchor != 'L' and coef:
                        result += coef * (self.tensor(anchor, key) - base)
                parameter.copy_(result)


def load_cache(path):
    cache = {}
    if path.exists():
        with path.open() as stream:
            for line in stream:
                if line.endswith('\n'):
                    item = json.loads(line)
                    cache[item['id']] = item['logits']
    return cache


def parity_gate(editor, collate, device, count=12, tolerance=1e-4):
    """Refuse to score unless L reproduces the previous round's cached logits.

    wsp.digest sorts keys, so request hashes cannot see an option reordering.
    Comparing real logits on real rows does, and covers any other input drift.
    """
    path = ROOT / 'research/weight-space/runs/ancestry/predictions/iteration1-baseline/baseline-L.json'
    reference = {r['id']: r['logits'] for r in json.loads(path.read_text())['rows']}
    data = {r['id']: r for r in map(json.loads, (SOUP_RUN / 'data.jsonl').open())}
    shared = sorted(i for i in reference if i in data)[:count]
    if len(shared) < count:
        raise ValueError('Too few shared rows for the parity gate')
    editor.apply(single('L'))
    worst = 0.
    for identifier in shared:
        request, _ = wsp.rotate(data[identifier], 0)
        batch = {k: v.to(device) for k, v in collate([request]).items()}
        with editor.torch.inference_mode():
            logits, _ = editor.model(**batch)
        values = logits[0, :len(data[identifier]['options'])].float().cpu().tolist()
        worst = max(worst, *(abs(a - b) for a, b in zip(values, reference[identifier])))
    if worst > tolerance:
        raise ValueError(f'Parity gate failed: L logits differ by {worst} from the cached baseline')
    return worst


def score_recipe(editor, collate, recipe, rows, device, log):
    torch, model = editor.torch, editor.model
    print_id = recipe['id']
    key = fingerprint(recipe)
    directory = cache_directory()
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f'{key}.jsonl'
    meta = directory / f'{key}.recipe.json'
    if not meta.exists():
        meta.write_text(json.dumps(recipe, indent=1) + '\n')
    cache = load_cache(path)
    missing = [r for r in rows if r['id'] not in cache]
    if not missing:
        return
    lock = directory / f'{key}.lock'
    try:
        descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        log(f'skip {print_id}: claimed by another worker')
        return
    try:
        os.write(descriptor, str(os.getpid()).encode())
        cache = load_cache(path)     # a peer may have finished before this claim
        missing = [r for r in rows if r['id'] not in cache]
        editor.apply(recipe)
        start = time.monotonic()
        with path.open('a') as stream:
            for index, row in enumerate(missing):
                request, mapping = wsp.rotate(row, 0)
                batch = {k: v.to(device) for k, v in collate([request]).items()}
                with torch.inference_mode():
                    logits, _ = model(**batch)
                values = logits[0, :len(mapping)].float().cpu().tolist()
                if not all(math.isfinite(v) for v in values):
                    raise FloatingPointError('Nonfinite logits: ' + row['id'])
                stream.write(json.dumps(dict(id=row['id'], logits=values)) + '\n')
                stream.flush()
                if (index + 1) % 200 == 0:
                    log(f'{print_id} {index+1}/{len(missing)} {(time.monotonic()-start)/(index+1):.3f}s/row')
        log(f'done {print_id} {key} rows={len(missing)} seconds={time.monotonic()-start:.0f}')
    finally:
        os.close(descriptor)
        lock.unlink(missing_ok=True)


def command_run(args):
    recipes = json.loads(Path(args.recipes).read_text())
    require_freeze(args.role, recipes, args.freeze)
    for recipe in recipes:
        for rule in recipe['rules']:
            unknown = set(rule['coefs']) - set(ANCHORS)
            if unknown:
                raise ValueError(f'Unknown anchors: {unknown}')
    rows = panel_rows(args.role)
    model_args = argparse.Namespace(model=wsp.DEFAULT_MODEL, expected_model_sha256=wsp.DEPLOYED_SHA256,
                                    threads=args.threads, device='cpu')
    torch, model, collate, _ = wsp.runtime(model_args)
    editor = Editor(torch, model, None)
    def log(message):
        print(json.dumps(dict(t=time.strftime('%H:%M:%S'), worker=args.worker, message=message)), flush=True)
    log(f'parity gate passed, worst logit error {parity_gate(editor, collate, "cpu"):.2e}')
    if args.shuffle:
        random.Random(args.worker).shuffle(recipes)
    for recipe in recipes:
        score_recipe(editor, collate, recipe, rows, 'cpu', log)
    log('worker finished')


def predictions(recipe):
    return load_cache(cache_directory() / f'{fingerprint(recipe)}.jsonl')


def nll(values, gold, temperature):
    scaled = [v / temperature for v in values]
    top = max(scaled)
    return top + math.log(sum(math.exp(v - top) for v in scaled)) - scaled[gold]


def best_temperature(rows, logits):
    """Golden-section NLL fit; positive scaling never changes the argmax."""
    def loss(t):
        return sum(nll(logits[r['id']], wsp.gold_index(r), t) for r in rows) / len(rows)
    low, high = .3, 5.
    ratio = (5 ** .5 - 1) / 2
    for _ in range(40):
        a, b = high - ratio * (high - low), low + ratio * (high - low)
        low, high = (low, b) if loss(a) < loss(b) else (a, high)
    return (low + high) / 2


def score_table(rows, logits, baseline=None, temperature=TEMPERATURE):
    """Source accuracy and NLL, plus paired fixes and regressions when baselined."""
    result = {}
    for source in SOURCES:
        subset = [r for r in rows if r['source_family'] == source and r['id'] in logits]
        if not subset:
            continue
        correct = [wsp.argmax(logits[r['id']]) == wsp.gold_index(r) for r in subset]
        item = dict(n=len(subset), correct=sum(correct), accuracy=sum(correct) / len(subset),
                    nll=sum(nll(logits[r['id']], wsp.gold_index(r), temperature) for r in subset) / len(subset))
        if baseline is not None:
            base = [wsp.argmax(baseline[r['id']]) == wsp.gold_index(r) for r in subset]
            item['fixes'] = sum(c and not b for c, b in zip(correct, base))
            item['regressions'] = sum(b and not c for c, b in zip(correct, base))
        result[source] = item
    if len(result) == 3:
        result['macro'] = sum(result[s]['accuracy'] for s in SOURCES) / 3
        result['macro_nll'] = sum(result[s]['nll'] for s in SOURCES) / 3
    return result


def paired_bootstrap(rows, logits, baseline, draws=4000, seed=2049):
    """Component-cluster bootstrap for paired accuracy deltas, by source and macro."""
    delta = {}
    for source in SOURCES:
        groups = collections.defaultdict(list)
        for row in rows:
            if row['source_family'] == source:
                gold = wsp.gold_index(row)
                groups[row['connected_component']].append(
                    int(wsp.argmax(logits[row['id']]) == gold) - int(wsp.argmax(baseline[row['id']]) == gold))
        delta[source] = [(sum(v), len(v)) for v in groups.values()]
    rng = random.Random(seed)
    samples = {s: [] for s in SOURCES}
    macro = []
    for _ in range(draws):
        values = []
        for source in SOURCES:
            pool = delta[source]
            chosen = rng.choices(pool, k=len(pool))
            value = sum(a for a, _ in chosen) / sum(b for _, b in chosen)
            samples[source].append(value)
            values.append(value)
        macro.append(sum(values) / 3)
    def interval(values):
        ordered = sorted(values)
        return [100 * ordered[int(.025 * draws)], 100 * ordered[int(.975 * draws) - 1]]
    return dict(**{s: interval(samples[s]) for s in SOURCES}, macro=interval(macro))


def command_report(args):
    recipes = json.loads(Path(args.recipes).read_text())
    rows = panel_rows(args.role)
    require_freeze(args.role, json.loads(Path(args.recipes).read_text()), args.freeze)
    baseline = predictions(single(args.baseline))
    ids = [r['id'] for r in rows]
    if not all(i in baseline for i in ids):
        raise SystemExit(f'Baseline {args.baseline} lacks rows for {args.role}')
    print(f"{'recipe':34}{'RD':>7}{'MV':>7}{'FV':>7}{'macro':>7}{'nll':>7}   fix/reg RD MV FV")
    table = []
    for recipe in recipes:
        logits = predictions(recipe)
        if not all(i in logits for i in ids):
            print(f"{recipe['id']:34} incomplete {sum(i in logits for i in ids)}/{len(ids)}")
            continue
        scores = score_table(rows, logits, baseline)
        table.append((scores['macro'], recipe['id'], scores))
        fr = ' '.join(f"+{scores[s]['fixes']}/-{scores[s]['regressions']}" for s in SOURCES)
        print(f"{recipe['id']:34}" + ''.join(f"{100*scores[s]['accuracy']:7.2f}" for s in SOURCES)
              + f"{100*scores['macro']:7.2f}{scores['macro_nll']:7.3f}   {fr}")
    if args.output:
        wsp.write(args.output, dict(role=args.role, baseline=args.baseline, table=[dict(id=i, scores=s) for _, i, s in table]))


def replay_audit(editor, collate, recipe, rows, device, sample, seed=2049):
    """Re-score saved rows at the original thread count and report any difference.

    The startup parity gate only proves a worker began correctly. Replay checks
    the saved evidence itself, so a transient fault during scoring cannot hide.
    """
    saved = predictions(recipe)
    scored = [r for r in rows if r['id'] in saved]
    chosen = random.Random(seed).sample(scored, min(sample, len(scored)))
    editor.apply(recipe)
    worst, flips = 0., 0
    for row in chosen:
        request, _ = wsp.rotate(row, 0)
        batch = {k: v.to(device) for k, v in collate([request]).items()}
        with editor.torch.inference_mode():
            logits, _ = editor.model(**batch)
        values = logits[0, :len(row['options'])].float().cpu().tolist()
        worst = max(worst, *(abs(a - b) for a, b in zip(values, saved[row['id']])))
        flips += wsp.argmax(values) != wsp.argmax(saved[row['id']])
    return dict(id=recipe['id'], fingerprint=fingerprint(recipe), rows=len(chosen), worst_logit_error=worst,
                answer_flips=flips)


def command_audit(args):
    rows = panel_rows(args.role)
    model_args = argparse.Namespace(model=wsp.DEFAULT_MODEL, expected_model_sha256=wsp.DEPLOYED_SHA256,
                                    threads=args.threads, device='cpu')
    torch, model, collate, _ = wsp.runtime(model_args)
    editor = Editor(torch, model, None)
    report = []
    for recipe in json.loads(Path(args.recipes).read_text()):
        item = replay_audit(editor, collate, recipe, rows, 'cpu', args.sample)
        report.append(item)
        print(json.dumps(item), flush=True)
    if args.output:
        wsp.write(args.output, dict(schema='p2048-weight-space-replay-audit-v1', role=args.role, threads=args.threads,
                                    sample=args.sample, seed=2049, recipes=report))


def tensor_sha(tensor):
    value = tensor.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(json.dumps(dict(shape=list(value.shape), dtype=str(value.dtype)), sort_keys=True).encode())
    digest.update(memoryview(value.numpy()))
    return digest.hexdigest()


def command_export(args):
    """Save the frozen recipe as a full checkpoint, then verify it by reload and replay."""
    import shutil
    freeze_path = Path(args.freeze)
    freeze = json.loads(freeze_path.read_text())
    recipe = freeze['selected']
    if fingerprint(recipe) != freeze['selected_fingerprint']:
        raise ValueError('Freeze fingerprint does not match its recipe')
    destination = Path(args.output_dir)
    wsp.guard_output(destination, model_paths=[wsp.DEFAULT_MODEL])
    model_args = argparse.Namespace(model=wsp.DEFAULT_MODEL, expected_model_sha256=wsp.DEPLOYED_SHA256,
                                    threads=args.threads, device='cpu')
    torch, model, collate, _ = wsp.runtime(model_args)
    editor = Editor(torch, model, None)
    editor.apply(recipe)
    expected = {k: tensor_sha(v) for k, v in model.state_dict().items()}
    destination.mkdir(parents=True)
    model.save_pretrained(destination)
    shutil.copytree(wsp.DEFAULT_MODEL / 'tokenizer', destination / 'tokenizer')
    shutil.copytree(wsp.DEFAULT_MODEL / 'licenses', destination / 'licenses')
    shutil.copy2(wsp.DEFAULT_MODEL / 'THIRD_PARTY_NOTICES.md', destination / 'THIRD_PARTY_NOTICES.md')
    from decision_moe import DecisionMoE
    reloaded = DecisionMoE.from_pretrained(destination).to(dtype=torch.float32).eval()
    reloaded.requires_grad_(False)
    actual = {k: tensor_sha(v) for k, v in reloaded.state_dict().items()}
    if actual != expected:
        raise ValueError('Reloaded state differs from the applied recipe')
    # Replay seeded lockbox rows through the reloaded checkpoint; exact match required.
    rows = panel_rows('lockbox')
    saved = predictions(recipe)
    chosen = random.Random(2049).sample([r for r in rows if r['id'] in saved], args.replay)
    worst = 0.
    for row in chosen:
        request, _ = wsp.rotate(row, 0)
        with torch.inference_mode():
            logits, _ = reloaded(**collate([request]))
        values = logits[0, :len(row['options'])].float().cpu().tolist()
        worst = max(worst, *(abs(a - b) for a, b in zip(values, saved[row['id']])))
    if worst > 1e-4:
        raise ValueError(f'Reloaded checkpoint differs from scored logits by {worst}')
    manifest = dict(schema='p2048-weight-space-soup-export-v1', research_only=True, recipe=recipe,
                    recipe_fingerprint=fingerprint(recipe), freeze_sha256=wsp.sha(freeze_path),
                    model_sha256=wsp.sha(destination / 'model.safetensors'), state_dict_sha256=wsp.digest(expected),
                    anchors={a: wsp.sha(ROOT / ANCHORS[a]) for a in sorted({a for r in recipe['rules'] for a in r['coefs']})},
                    buffers='Temperature buffer and act_head kept at L',
                    stored_choice_temperature=float(reloaded.temperature[0]),
                    search_fitted_choice_temperature=freeze['choice_temperature_fitted_on_search'],
                    reload_replay=dict(rows=len(chosen), seed=2049, worst_logit_error=worst),
                    unchecked=['Score and Noul primitives', 'legacy workflows and security false-negative guards',
                               'option-order robustness'])
    wsp.write(destination / 'research_manifest.json', manifest)
    print(json.dumps({k: manifest[k] for k in ('model_sha256', 'state_dict_sha256', 'reload_replay')}, indent=1))


def label_recall(rows, logits, source='MAVEN-ERE'):
    """Per-label recall and prediction counts; accuracy alone hides majority-class bias."""
    subset = [r for r in rows if r['source_family'] == source]
    support, correct, predicted = collections.Counter(), collections.Counter(), collections.Counter()
    for row in subset:
        choice = list(row['options'])[wsp.argmax(logits[row['id']])]
        support[row['label']] += 1
        predicted[choice] += 1
        correct[row['label']] += choice == row['label']
    top = support.most_common(1)[0]
    return dict(rows=len(subset), majority_label=top[0], majority_accuracy=top[1] / len(subset),
                recall={k: [correct[k], support[k]] for k, _ in support.most_common()},
                balanced_accuracy=sum(correct[k] / support[k] for k in support) / len(support),
                predicted=dict(predicted.most_common()))


def command_labels(args):
    rows = panel_rows(args.role)
    require_freeze(args.role, json.loads(Path(args.recipes).read_text()), args.freeze)
    print(f"{'recipe':22}{'bal.acc':>8}  recall by label (majority {args.source})")
    report = []
    for recipe in json.loads(Path(args.recipes).read_text()):
        logits = predictions(recipe)
        if not all(r['id'] in logits for r in rows):
            continue
        item = label_recall(rows, logits, args.source)
        report.append(dict(id=recipe['id'], **item))
        recall = ' '.join(f'{k}={a}/{b}' for k, (a, b) in item['recall'].items())
        print(f"{recipe['id']:22}{100*item['balanced_accuracy']:8.2f}  {recall}")
    if args.output:
        wsp.write(args.output, dict(role=args.role, source=args.source, table=report))


def command_compare(args):
    """Paired component-bootstrap intervals for each recipe against named baselines."""
    rows = panel_rows(args.role)
    require_freeze(args.role, json.loads(Path(args.recipes).read_text()), args.freeze)
    recipes = json.loads(Path(args.recipes).read_text())
    report = {}
    for name in args.baselines:
        baseline = predictions(single(name))
        for recipe in recipes:
            logits = predictions(recipe)
            if recipe['id'] == name or not all(r['id'] in logits and r['id'] in baseline for r in rows):
                continue
            interval = paired_bootstrap(rows, logits, baseline)
            scores = score_table(rows, logits, baseline)
            base = score_table(rows, baseline)
            gains = {s: 100 * (scores[s]['accuracy'] - base[s]['accuracy']) for s in SOURCES}
            gains['macro'] = 100 * (scores['macro'] - base['macro'])
            report[f"{recipe['id']} vs {name}"] = dict(gain=gains, interval=interval)
            print(f"{recipe['id']+' vs '+name:30}" + ' '.join(
                f"{k[:2]} {gains[k]:+.2f} [{interval[k][0]:+.2f},{interval[k][1]:+.2f}]" for k in (*SOURCES, 'macro')))
    if args.output:
        wsp.write(args.output, dict(role=args.role, draws=4000, comparisons=report))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    export = sub.add_parser('export')
    export.add_argument('--freeze', required=True)
    export.add_argument('--output-dir', required=True)
    export.add_argument('--replay', type=int, default=30)
    export.add_argument('--threads', type=int, default=2)
    export.set_defaults(handler=command_export)
    audit = sub.add_parser('audit')
    audit.add_argument('--recipes', required=True)
    audit.add_argument('--role', required=True, choices=('screen', 'search', 'validation'))
    audit.add_argument('--sample', type=int, default=100)
    audit.add_argument('--threads', type=int, default=2)
    audit.add_argument('--output')
    audit.set_defaults(handler=command_audit)
    labels = sub.add_parser('labels')
    labels.add_argument('--recipes', required=True)
    labels.add_argument('--role', required=True, choices=('screen', 'search', 'validation', 'lockbox'))
    labels.add_argument('--freeze')
    labels.add_argument('--source', default='MAVEN-ERE')
    labels.add_argument('--output')
    labels.set_defaults(handler=command_labels)
    compare = sub.add_parser('compare')
    compare.add_argument('--recipes', required=True)
    compare.add_argument('--role', required=True, choices=('screen', 'search', 'validation', 'lockbox'))
    compare.add_argument('--freeze')
    compare.add_argument('--baselines', nargs='+', default=['L'])
    compare.add_argument('--output')
    compare.set_defaults(handler=command_compare)
    run = sub.add_parser('run')
    run.add_argument('--recipes', required=True)
    run.add_argument('--role', required=True, choices=('screen', 'search', 'validation', 'lockbox', 'external', 'legacy', 'synthetic'))
    run.add_argument('--threads', type=int, default=2)
    run.add_argument('--worker', type=int, default=0)
    run.add_argument('--shuffle', action='store_true')
    run.add_argument('--freeze')
    run.set_defaults(handler=command_run)
    report = sub.add_parser('report')
    report.add_argument('--recipes', required=True)
    report.add_argument('--role', required=True, choices=('screen', 'search', 'validation', 'lockbox'))
    report.add_argument('--freeze')
    report.add_argument('--baseline', default='L')
    report.add_argument('--output')
    report.set_defaults(handler=command_report)
    parser.add_argument('--run-dir', help='Run folder holding plan.json and data.jsonl (default: runs/soup)')
    args = parser.parse_args()
    if args.run_dir:
        global RUN
        RUN = Path(args.run_dir).resolve()
    args.handler(args)


if __name__ == '__main__':
    main()
