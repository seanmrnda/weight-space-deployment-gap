"""Local, no-gradient scorer projection pilot. Never writes a model checkpoint.

Commands plan/fit/evaluate use only Python's standard library. collect/verify
need the repository's torch/transformers runtime. Every choice is evaluated
under all cyclic rotations; option keys are bookkeeping, never model input.
"""
from __future__ import annotations

import argparse
import collections
import copy
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import statistics
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MODEL = ROOT / 'research/weight-space/weights/base-p2048'
DEFAULT_DATA = ROOT / '.local/praxis-p2048-targeted-02/prepared/domain_development.jsonl'
DEPLOYED_SHA256 = '2a70d51d4b74dffaf1dff63c175019ac07f7fc131e21c912dd3f189968311dba'
SCHEMA = 'p2048-weight-space-v1'
GOLD_FIELDS = {'label', 'target_probs', 'hard_label_index', 'soft_label', 'gold', 'answer'}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def digest(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def guard_output(path, inputs=(), model_paths=(DEFAULT_MODEL,)):
    path = Path(path)
    resolved = path.resolve()
    if any(resolved.is_relative_to(Path(model).resolve()) for model in model_paths):
        raise ValueError('Output is inside a protected model directory: ' + str(path))
    if any(resolved == Path(item).resolve() for item in inputs):
        raise ValueError('Output aliases an input: ' + str(path))
    if os.path.lexists(path):
        raise FileExistsError('Refusing to replace an existing output: ' + str(path))


def publish_no_clobber(temporary, destination):
    """Atomic same-filesystem publication, with no existing-target replacement."""
    os.link(temporary, destination)
    Path(temporary).unlink()


def write(path, obj):
    path = Path(path)
    guard_output(path)
    serialized = json.dumps(obj, indent=2, allow_nan=False) + '\n'
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=path.parent,
                                         prefix='.' + path.name + '.', suffix='.partial', delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(serialized)
            stream.flush()
            os.fsync(stream.fileno())
        publish_no_clobber(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def read_rows(path):
    with Path(path).open(encoding='utf-8') as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    ids = [r['id'] for r in rows]
    if len(ids) != len(set(ids)):
        raise ValueError('Duplicate row IDs are not permitted')
    if any(not r.get('group_component') for r in rows):
        raise ValueError('Every row needs an explicit group_component')
    return rows


def source(row):
    return row.get('source_family', row.get('source', 'unknown'))


def order_key(seed, value):
    return hashlib.sha256(f'{seed}:{value}'.encode()).hexdigest()


def leakage_groups(rows):
    """Union explicit components connected by any raw split_groups key."""
    parents = {r['group_component']:r['group_component'] for r in rows}
    def find(key):
        while parents[key] != key:
            parents[key] = parents[parents[key]]
            key = parents[key]
        return key
    owner = {}
    for row in rows:
        for group in row.get('split_groups', []):
            if not isinstance(group, str):
                raise ValueError('split_groups must contain strings')
            component = row['group_component']
            if group in owner:
                a,b = find(component),find(owner[group])
                parents[max(a,b)] = min(a,b)
            else:
                owner[group] = component
    return {key:find(key) for key in parents}


def validate_plan(plan):
    roles = plan['roles']
    for field in ('id','component','split_component'):
        left = {r[field] for r in roles['discovery']}
        right = {r[field] for r in roles['evaluation']}
        if left & right:
            raise ValueError('Discovery/evaluation overlap: ' + field)
    left = {g for r in roles['discovery'] for g in r['raw_split_groups']}
    right = {g for r in roles['evaluation'] for g in r['raw_split_groups']}
    if left & right:
        raise ValueError('Discovery/evaluation raw split_groups overlap')


def make_plan(rows, seed, discovery_per_source, evaluation_per_source):
    """Partition leakage groups first; cap independent components without labels."""
    pools = {'discovery': collections.defaultdict(list), 'evaluation': collections.defaultdict(list)}
    connected = leakage_groups(rows)
    component_sources = collections.defaultdict(set)
    for row in rows:
        component_sources[connected[row['group_component']]].add(source(row))
    for row in rows:
        atom = connected[row['group_component']]
        if len(component_sources[atom]) != 1:
            continue
        role = 'discovery' if int(order_key(seed, atom), 16) % 2 == 0 else 'evaluation'
        if role == 'discovery' and row.get('type', 'choice') != 'choice':
            continue
        pools[role][source(row)].append(row)
    result = {}
    for role, by_source in pools.items():
        limit = discovery_per_source if role == 'discovery' else evaluation_per_source
        selected = []
        for name in sorted(by_source):
            ordered = sorted(by_source[name], key=lambda r: (order_key(seed, r['id']), r['id']))
            unique = {}
            for row in ordered:
                unique.setdefault(connected[row['group_component']], row)
            ordered = sorted(unique.values(), key=lambda r: (order_key(seed, connected[r['group_component']]), r['id']))
            selected.extend(ordered[:limit] if limit else ordered)
        result[role] = [dict(id=r['id'], source=source(r), component=r['group_component'],
                             split_component=connected[r['group_component']],
                             raw_split_groups=sorted(set(r.get('split_groups', []))),
                             type=r.get('type', 'choice')) for r in selected]
    if not all(result.values()):
        raise ValueError('Discovery and evaluation must both be nonempty')
    validate_plan({'roles':result})
    return result


def rotate(row, shift):
    """Return gold-free request and mapping displayed position -> original option."""
    clean = {k: copy.deepcopy(v) for k, v in row.items() if k not in GOLD_FIELDS}
    if row.get('type', 'choice') != 'choice':
        count = len(row['levels']) if row['type'] == 'score' else 2
        if shift:
            raise ValueError('Only choice options can be rotated')
        return clean, list(range(count))
    items = list(row['options'].items())
    permutation = [(i + shift) % len(items) for i in range(len(items))]
    clean['options'] = dict(items[i] for i in permutation)
    return clean, permutation


def dot(a, b):
    return math.fsum(x * y for x, y in zip(a, b))


def norm(a):
    return math.sqrt(dot(a, a))


def cosine(a, b):
    den = norm(a) * norm(b)
    return dot(a, b) / den if den else None


def mean_vector(vectors):
    vectors = list(vectors)
    if not vectors:
        raise ValueError('Cannot average an empty vector set')
    return [math.fsum(values) / len(vectors) for values in zip(*vectors)]


def question_contrast(record):
    """OLS slope h ~ normalized displayed position, same option across rotations.

    No gold or scores enter the contrast. Each option is content-centered and
    visits every position exactly once. Each question has equal final weight.
    """
    h, perms = record['activations'], record['permutations']
    count = len(h[0])
    if len(h) != count or any(sorted(p) != list(range(count)) for p in perms):
        raise ValueError('Expected complete cyclic rotations')
    width = len(h[0][0])
    values = [i / (count - 1) - .5 for i in range(count)]
    den = count * sum(x * x for x in values)
    result = [0.] * width
    for original in range(count):
        positions = [p.index(original) for p in perms]
        if sorted(positions) != list(range(count)):
            raise ValueError('Each option must occupy every position')
        for rotation, position in enumerate(positions):
            coefficient = values[position] / den
            for j, value in enumerate(h[rotation][original]):
                result[j] += coefficient * value
    return result


def projection(weight, direction, strength, maximum_relative_norm):
    if not 0 <= strength <= 1 or not 0 <= maximum_relative_norm <= 1:
        raise ValueError('strength and max-relative-norm must be in [0,1]')
    length = norm(direction)
    if length <= 1e-12:
        raise ValueError('Position contrast is zero; no identifiable edit')
    unit = [x / length for x in direction]
    coefficient = dot(weight, unit)
    alpha = min(strength, maximum_relative_norm * norm(weight) / abs(coefficient)) if coefficient else strength
    return [-alpha * coefficient * x for x in unit], unit, alpha


def cache_read(path):
    with gzip.open(path, 'rt') as stream:
        header = json.loads(next(stream))
        records = [json.loads(line) for line in stream if line.strip()]
    if header['schema'] != SCHEMA or header['kind'] != 'activation-cache':
        raise ValueError('Invalid activation cache')
    if [r['id'] for r in records] != header['row_ids']:
        raise ValueError('Cache rows differ from header')
    return header, records


def validate_bindings(header, plan, role):
    validate_plan(plan)
    if header['plan_sha256'] != digest(plan) or header['role'] != role:
        raise ValueError('Cache/plan role binding differs')
    if header['row_ids'] != [r['id'] for r in plan['roles'][role]]:
        raise ValueError('Cache row selection differs')


def command_plan(args):
    rows = read_rows(args.data)
    roles = make_plan(rows, args.seed, args.discovery_per_source, args.evaluation_per_source)
    plan = dict(schema=SCHEMA, data_sha256=sha(args.data), seed=args.seed, roles=roles,
                selection='SHA256 component parity; one row per union of component/raw split_groups; source-stratified component cap; multi-source unions excluded; no labels',
                inference=dict(max_length=2048, head_length=512, dtype='float32'),
                discovery_per_source=args.discovery_per_source, evaluation_per_source=args.evaluation_per_source)
    write(args.output, plan)
    print(json.dumps(dict(plan=str(args.output), counts={k:dict(collections.Counter(r['source'] for r in v)) for k,v in roles.items()})))


def runtime(args):
    # Do not resolve/download anything outside the explicitly supplied local model.
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    sys.path.insert(0, str(ROOT / 'src'))
    try:
        import torch
        from transformers import AutoTokenizer
        from decision_moe import DecisionMoE
        from decision_moe.data import DecisionCollator
    except ImportError as exc:
        raise SystemExit(f'Inference runtime unavailable ({exc}). Use a local compatible torch/transformers environment; no download was attempted.') from exc
    actual = sha(args.model / 'model.safetensors')
    if actual != args.expected_model_sha256:
        raise ValueError(f'Model digest differs: {actual}')
    manifest_path = args.model / 'deployment_manifest.json'
    manifest = None
    if args.expected_model_sha256 == DEPLOYED_SHA256:
        from decision_moe.serving import verify_bundle
        manifest = verify_bundle(args.model)
    elif manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        for name, expected in manifest['files'].items():
            if sha(args.model / name) != expected:
                raise ValueError('Deployment manifest mismatch: ' + name)
    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model = DecisionMoE.from_pretrained(args.model).to(device=args.device, dtype=torch.float32).eval()
    model.requires_grad_(False)
    model._weight_space_temperatures = ([manifest['temperatures'][k] for k in ('choice','score','noul')]
                                        if manifest else model.temperature.detach().cpu().tolist())
    model._weight_space_temperature_source = 'verified deployment manifest' if manifest else 'checkpoint buffer'
    tok = AutoTokenizer.from_pretrained(args.model / 'tokenizer', local_files_only=True)
    collate = DecisionCollator(tok, 2048, 512, labeled=False)
    return torch, model, collate, actual


def forward_rows(torch, model, collate, rows, device):
    captured = []
    hook = model.scorer[3].register_forward_pre_hook(lambda module, inputs: captured.append(inputs[0].detach().cpu()))
    try:
        batch = {k:v.to(device) for k,v in collate(rows).items()}
        with torch.inference_mode():
            logits, _ = model(**batch)
        if len(captured) != 1:
            raise ValueError('Expected one scorer invocation')
        return logits.detach().cpu(), captured[0]
    finally:
        hook.remove()


def selected_rows(args, role):
    plan = json.loads(args.plan.read_text())
    validate_plan(plan)
    if sha(args.data) != plan['data_sha256']:
        raise ValueError('Data digest differs from plan')
    by_id = {r['id']:r for r in read_rows(args.data)}
    return plan, [by_id[item['id']] for item in plan['roles'][role]]


def command_collect(args):
    plan, rows = selected_rows(args, args.role)
    torch, model, collate, actual = runtime(args)
    weight = model.scorer[3].weight.detach().cpu()[0].tolist()
    bias = float(model.scorer[3].bias.detach().cpu()[0])
    header = dict(schema=SCHEMA, kind='activation-cache', role=args.role, model_sha256=actual,
                  plan_sha256=digest(plan), data_sha256=plan['data_sha256'], row_ids=[r['id'] for r in rows],
                  scorer_weight=weight, scorer_bias=bias, dtype='float32', max_length=2048,
                  temperature=model._weight_space_temperatures, temperature_source=model._weight_space_temperature_source,
                  python=platform.python_version(),
                  torch=torch.__version__, device=args.device, marker_only_head=bool(model.marker_only_head),
                  script_sha256=sha(__file__))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=args.output.parent, prefix='.' + args.output.name + '.',
                                     suffix='.partial', delete=False) as temporary:
        partial = Path(temporary.name)
    maximum_error = 0.
    with gzip.open(partial, 'wt') as stream:
        stream.write(json.dumps(header, allow_nan=False) + '\n')
        for index, row in enumerate(rows):
            count = len(row['options']) if row.get('type', 'choice') == 'choice' else 1
            rotations = [rotate(row, shift) for shift in range(count)]
            scores, activations = [], []
            for start in range(0, count, args.batch_size):
                part = rotations[start:start + args.batch_size]
                logits, hidden = forward_rows(torch, model, collate, [r for r,p in part], args.device)
                for j, (_, permutation) in enumerate(part):
                    inverse = [permutation.index(k) for k in range(len(permutation))]
                    s = [float(logits[j,k]) for k in inverse]
                    h = [hidden[j,k].tolist() for k in inverse]
                    if not all(math.isfinite(x) for x in s) or not all(math.isfinite(x) for v in h for x in v):
                        raise FloatingPointError('Nonfinite scorer output')
                    maximum_error = max(maximum_error, max(abs(dot(weight,v) + bias - x) for v,x in zip(h,s)))
                    scores.append(s)
                    activations.append(h)
            record = dict(id=row['id'], source=source(row), component=row['group_component'],
                          type=row.get('type', 'choice'), permutations=[p for r,p in rotations],
                          logits=scores, activations=activations)
            stream.write(json.dumps(record, allow_nan=False) + '\n')
            if (index + 1) % 8 == 0 or index + 1 == len(rows):
                print(json.dumps(dict(role=args.role, completed=index + 1, total=len(rows))), flush=True)
    if maximum_error > args.parity_tolerance:
        raise ValueError(f'Scorer reconstruction error {maximum_error} exceeds tolerance; partial cache retained')
    publish_no_clobber(partial,args.output)
    write(str(args.output) + '.receipt.json', dict(cache_sha256=sha(args.output), rows=len(rows),
          model_sha256=actual, maximum_linear_reconstruction_error=maximum_error, source_unchanged=sha(args.model / 'model.safetensors') == actual))


def summary_cosines(vectors, direction):
    values = [c for v in vectors if (c := cosine(v, direction)) is not None]
    return dict(n=len(values), mean=statistics.mean(values) if values else None,
                median=statistics.median(values) if values else None,
                positive_fraction=sum(x > 0 for x in values) / len(values) if values else None)


def command_fit(args):
    plan = json.loads(args.plan.read_text())
    header, records = cache_read(args.discovery)
    validate_bindings(header, plan, 'discovery')
    slopes = [question_contrast(r) for r in records]
    direction = mean_vector(slopes)
    delta, unit, effective = projection(header['scorer_weight'], direction, args.strength, args.max_relative_norm)
    by_source = collections.defaultdict(list)
    # Split by component, not row, even for reproducibility diagnostics.
    halves = [[], []]
    pairs = collections.defaultdict(list)
    for record, slope in zip(records, slopes):
        by_source[record['source']].append(slope)
        halves[int(order_key('direction-half', record['component']), 16) % 2].append(slope)
        h, perms = record['activations'], record['permutations']
        count = len(perms)
        for a in range(count):
            for b in range(a + 1, count):
                contrast = []
                for original in range(count):
                    ra = next(i for i,p in enumerate(perms) if p[a] == original)
                    rb = next(i for i,p in enumerate(perms) if p[b] == original)
                    contrast.append([(y-x) / ((b-a)/(count-1)) for x,y in zip(h[ra][original], h[rb][original])])
                pairs[(count,a,b)].append(mean_vector(contrast))
    # Fraction of within-content shift energy in the single selected direction.
    total_energy = projected_energy = 0.
    for record in records:
        h = record['activations']
        for original in range(len(h[0])):
            center = mean_vector(rotation[original] for rotation in h)
            for rotation in h:
                difference = [a-b for a,b in zip(rotation[original], center)]
                total_energy += dot(difference,difference)
                projected_energy += dot(difference,unit)**2
    diagnostics = dict(question_cosines=summary_cosines(slopes, direction),
        component_half_cosine=cosine(mean_vector(halves[0]),mean_vector(halves[1])) if all(halves) else None,
        component_half_rows=list(map(len,halves)),
        source_cosines={s:dict(rows=len(v), cosine=cosine(mean_vector(v),direction)) for s,v in sorted(by_source.items())},
        position_pair_cosines={f'{k}:{a}->{b}':dict(rows=len(v), cosine=cosine(mean_vector(v),direction)) for (k,a,b),v in sorted(pairs.items())},
        rank_one_explained_within_content_shift_energy=projected_energy/total_energy if total_energy else None,
        scorer_position_slope_before=dot(header['scorer_weight'], direction),
        scorer_position_slope_after=dot([a+b for a,b in zip(header['scorer_weight'],delta)], direction))
    patch = dict(schema=SCHEMA, kind='rank-one-scorer-patch', parameter='scorer.3.weight',
        model_sha256=header['model_sha256'], plan_sha256=digest(plan), discovery_cache_sha256=sha(args.discovery),
        discovery_ids=header['row_ids'], discovery_components=sorted({r['component'] for r in records}),
        discovery_raw_split_groups=sorted({g for r in plan['roles']['discovery'] for g in r['raw_split_groups']}),
        base_weight=header['scorer_weight'], delta=delta, direction=unit,
        requested_strength=args.strength, effective_strength=effective, max_relative_norm=args.max_relative_norm,
        relative_delta_norm=norm(delta)/norm(header['scorer_weight']), diagnostics=diagnostics,
        derivation='Equal-question mean same-content normalized-position slopes; remove bounded scorer projection; no labels, gradients, optimizer, or evaluation rows.',
        fixed_before_evaluation=True, reversible='Subtract delta from the same base weight; checkpoint files never written.')
    patch['patch_sha256'] = digest(patch)
    write(args.output, patch)
    print(json.dumps(dict(patch=str(args.output), relative_delta_norm=patch['relative_delta_norm'], diagnostics=diagnostics)))


def load_patch(path):
    patch = json.loads(Path(path).read_text())
    body = dict(patch)
    expected = body.pop('patch_sha256')
    if digest(body) != expected or patch['parameter'] != 'scorer.3.weight':
        raise ValueError('Patch digest or parameter differs')
    if len(patch['delta']) != len(patch['base_weight']):
        raise ValueError('Patch shape differs')
    if norm(patch['delta']) > patch['max_relative_norm'] * norm(patch['base_weight']) + 1e-10:
        raise ValueError('Patch violates norm bound')
    return patch


def argmax(values):
    return max(range(len(values)), key=values.__getitem__)


def gold_index(row):
    if 'hard_label_index' in row:
        return int(row['hard_label_index'])
    if row.get('type', 'choice') == 'choice' and 'label' in row:
        return list(row['options']).index(row['label'])
    if 'target_probs' in row:
        return argmax(row['target_probs'])
    raise ValueError('Evaluation requires hard_label_index or target_probs: ' + row['id'])


def nll(logits, gold, temperature):
    values = [x/temperature for x in logits]
    maximum = max(values)
    return maximum + math.log(sum(math.exp(x-maximum) for x in values)) - values[gold]


def row_metrics(row, before, after, temperature):
    gold = gold_index(row)
    bp, ap = [argmax(v) for v in before], [argmax(v) for v in after]
    choice = row.get('type','choice') == 'choice'
    pairs = [(i,j) for i in range(len(bp)) for j in range(i+1,len(bp))]
    return dict(id=row['id'], source=source(row), component=row['group_component'], type=row.get('type','choice'),
        gold=gold, baseline_predictions=bp, edited_predictions=ap,
        original_baseline_correct=bp[0] == gold, original_edited_correct=ap[0] == gold,
        rotation_baseline_accuracy=sum(p == gold for p in bp)/len(bp),
        rotation_edited_accuracy=sum(p == gold for p in ap)/len(ap),
        baseline_all_rotations_correct=all(p == gold for p in bp), edited_all_rotations_correct=all(p == gold for p in ap),
        baseline_order_flip=any(p != bp[0] for p in bp) if choice else None,
        edited_order_flip=any(p != ap[0] for p in ap) if choice else None,
        baseline_pair_flip_rate=sum(bp[i] != bp[j] for i,j in pairs)/len(pairs) if pairs else None,
        edited_pair_flip_rate=sum(ap[i] != ap[j] for i,j in pairs)/len(pairs) if pairs else None,
        edited_prediction_change_fraction=sum(a != b for a,b in zip(bp,ap))/len(bp),
        baseline_rotation_nll=statistics.mean(nll(v,gold,temperature) for v in before),
        edited_rotation_nll=statistics.mean(nll(v,gold,temperature) for v in after),
        maximum_logit_change=max(abs(a-b) for x,y in zip(before,after) for a,b in zip(x,y)))


def aggregate(records):
    fields = ['original_baseline_correct','original_edited_correct','rotation_baseline_accuracy','rotation_edited_accuracy',
              'baseline_all_rotations_correct','edited_all_rotations_correct','baseline_order_flip','edited_order_flip',
              'baseline_pair_flip_rate','edited_pair_flip_rate','edited_prediction_change_fraction',
              'baseline_rotation_nll','edited_rotation_nll','maximum_logit_change']
    result = dict(rows=len(records), components=len({r['component'] for r in records}))
    for field in fields:
        values = [r[field] for r in records if r[field] is not None]
        result[field] = statistics.mean(values) if values else None
    result['original_fixed'] = sum(not r['original_baseline_correct'] and r['original_edited_correct'] for r in records)
    result['original_broken'] = sum(r['original_baseline_correct'] and not r['original_edited_correct'] for r in records)
    diffs = [r['rotation_edited_accuracy'] - r['rotation_baseline_accuracy'] for r in records]
    result['paired_rotation_accuracy_delta'] = statistics.mean(diffs)
    component_diffs = collections.defaultdict(list)
    for r,d in zip(records,diffs):
        component_diffs[r['component']].append(d)
    means = [statistics.mean(v) for v in component_diffs.values()]
    result['component_mean_rotation_accuracy_delta'] = statistics.mean(means)
    result['component_standard_error'] = statistics.stdev(means)/math.sqrt(len(means)) if len(means)>1 else None
    return result


def pilot_decision(records, diagnostics):
    """Apply the predeclared development gate; never use it to choose an edit."""
    domains = ('Re-DocRED', 'MAVEN-ERE', 'FEVEROUS')
    by_source = {name:[r for r in records if r['source'] == name] for name in domains}
    missing = [name for name,rows in by_source.items() if not rows]
    half_cosine = diagnostics.get('component_half_cosine')
    order_rows = [r for r in records if r['baseline_order_flip'] is not None]
    baseline_unstable = sum(r['baseline_order_flip'] for r in order_rows)
    edited_unstable = sum(r['edited_order_flip'] for r in order_rows)
    native_counts = {
        name:dict(rows=len(rows), baseline_correct=sum(r['original_baseline_correct'] for r in rows),
                  edited_correct=sum(r['original_edited_correct'] for r in rows))
        for name,rows in by_source.items()
    }
    deltas = {name:item['edited_correct']-item['baseline_correct'] for name,item in native_counts.items()}
    def condition(passed, supported=True):
        return dict(status=('pass' if passed else 'fail') if supported else 'hold',
                    passed=bool(passed) if supported else None)
    conditions = dict(
        positive_discovery_split_half_cosine=condition(half_cosine is not None and half_cosine > 0,
                                                       supported=half_cosine is not None),
        all_three_named_domains_present=condition(not missing,supported=not missing),
        at_least_one_unstable_baseline_question=condition(baseline_unstable > 0,supported=bool(order_rows)),
        fewer_unstable_edited_questions=condition(edited_unstable < baseline_unstable,supported=bool(order_rows)),
        native_accuracy_strictly_improves_in_at_least_one_domain=condition(any(v > 0 for v in deltas.values()),
                                                                         supported=not missing),
        native_accuracy_worsens_in_no_domain=condition(all(v >= 0 for v in deltas.values()),supported=not missing),
    )
    statuses = [item['status'] for item in conditions.values()]
    status = 'advance' if all(s == 'pass' for s in statuses) else 'hold' if 'hold' in statuses else 'stop'
    return dict(name='p2048-position-projection-development-gate-v1', status=status,
                rule_document='docs/p2048-weight-space-programming.md', conditions=conditions,
                evidence=dict(discovery_component_half_cosine=half_cosine, missing_domains=missing,
                              order_evaluable_questions=len(order_rows), baseline_unstable_questions=baseline_unstable,
                              edited_unstable_questions=edited_unstable, native_counts=native_counts),
                scope='Advance means larger study only; this gate does not authorize deployment or claim held-out gains.')


def command_evaluate(args):
    plan, rows = selected_rows(args, 'evaluation')
    header, records = cache_read(args.evaluation)
    validate_bindings(header, plan, 'evaluation')
    patch = load_patch(args.patch)
    if patch['plan_sha256'] != digest(plan) or patch['model_sha256'] != header['model_sha256'] or patch['base_weight'] != header['scorer_weight']:
        raise ValueError('Patch/cache model or plan binding differs')
    if set(patch['discovery_raw_split_groups']) & {g for r in rows for g in r.get('split_groups',[])}:
        raise ValueError('Discovery/evaluation raw split_groups overlap')
    if set(patch['discovery_components']) & {r['component'] for r in records}:
        raise ValueError('Discovery/evaluation component overlap')
    metrics = []
    by_id = {r['id']:r for r in rows}
    reconstruction_error = 0.
    no_op_decision_mismatches = 0
    for record in records:
        before = record['logits']
        after = [[s+dot(h,patch['delta']) for s,h in zip(scores,hidden)] for scores,hidden in zip(before,record['activations'])]
        for actual_scores, hidden_options in zip(before, record['activations']):
            reconstructed = [dot(h, header['scorer_weight']) + header['scorer_bias'] for h in hidden_options]
            no_op_decision_mismatches += argmax(reconstructed) != argmax(actual_scores)
            reconstruction_error = max(reconstruction_error, max(abs(a-b) for a,b in zip(reconstructed, actual_scores)))
        row = by_id[record['id']]
        temperature = header['temperature'][{'choice':0,'score':1,'noul':2}[record['type']]]
        metrics.append(row_metrics(row,before,after,temperature))
    groups = collections.defaultdict(list)
    for record in metrics:
        groups[record['source']].append(record)
    report = dict(schema=SCHEMA, kind='development-evaluation', plan_sha256=digest(plan),
        patch_sha256=patch['patch_sha256'], model_sha256=header['model_sha256'], evaluation_cache_sha256=sha(args.evaluation),
        no_op_maximum_error=reconstruction_error, no_op_decision_mismatches=no_op_decision_mismatches,
        maximum_linear_reconstruction_error=reconstruction_error,
        overall=aggregate(metrics), sources={s:aggregate(v) for s,v in sorted(groups.items())},
        pilot_decision=pilot_decision(metrics,patch['diagnostics']),
        safety=dict(original_regression_ids=[r['id'] for r in metrics if r['original_baseline_correct'] and not r['original_edited_correct']],
                    non_choice_rows=sum(r['type'] != 'choice' for r in metrics),
                    patch_relative_norm=patch['relative_delta_norm']), rows=metrics,
        limitations=['Development diagnostic, not held-out confirmation or a deployment decision.',
                     'Order permutations are cyclic rotations, not every factorial permutation.',
                     'One-dimensional position nuisance may also carry useful content information.',
                     'Source and row metrics are descriptive; rows sharing components are dependent.',
                     'Cached edited logits use exact final-linear counterfactual algebra in Python; verify checks real FP32 forwards.'])
    if reconstruction_error > args.parity_tolerance or no_op_decision_mismatches:
        raise ValueError('Linear reconstruction parity failed')
    write(args.output,report)
    print(json.dumps({k:v for k,v in report.items() if k not in {'rows','limitations'}},allow_nan=False))


def command_verify(args):
    plan, rows = selected_rows(args,'evaluation')
    if args.row_id:
        rows = [row for row in rows if row['id'] == args.row_id]
        if not rows:
            raise ValueError('Requested row is outside the evaluation selection')
    patch = load_patch(args.patch)
    if patch['plan_sha256'] != digest(plan):
        raise ValueError('Patch plan differs')
    torch, model, collate, actual = runtime(args)
    if patch['model_sha256'] != actual:
        raise ValueError('Patch model differs')
    baseline = model.scorer[3].weight.detach().clone()
    if baseline.cpu()[0].tolist() != patch['base_weight']:
        raise ValueError('Base scorer differs')
    row = rows[0]
    count = len(row['options']) if row.get('type','choice') == 'choice' else 1
    # One rotation per forward fixes padding/batch rounding for all three arms.
    maximum = restoration_error = 0.
    baseline_predictions, edited_predictions = [], []
    for shift in range(count):
        request, permutation = rotate(row, shift)
        before,h = forward_rows(torch,model,collate,[request],args.device)
        try:
            with torch.no_grad():
                model.scorer[3].weight.copy_(baseline + torch.tensor([patch['delta']],device=args.device,dtype=torch.float32))
            edited,_ = forward_rows(torch,model,collate,[request],args.device)
            expected = before + (h * torch.tensor(patch['delta'])).sum(-1)
            maximum = max(maximum,float((edited-expected).abs().max()))
            baseline_predictions.append(permutation[int(before[0].argmax())])
            edited_predictions.append(permutation[int(edited[0].argmax())])
        finally:
            with torch.no_grad():
                model.scorer[3].weight.copy_(baseline)
        restored,_ = forward_rows(torch,model,collate,[request],args.device)
        restoration_error = max(restoration_error,float((restored-before).abs().max()))
    source_unchanged = sha(args.model / 'model.safetensors') == actual
    receipt = dict(schema=SCHEMA, kind='actual-forward-patch-verification', id=row['id'], rotations=count,
                   baseline_predictions=baseline_predictions, edited_predictions=edited_predictions,
                   maximum_counterfactual_error=maximum, restoration_maximum_error=restoration_error,
                   source_unchanged=source_unchanged, model_sha256=actual,
                   patch_sha256=patch['patch_sha256'], passed=source_unchanged and maximum <= args.parity_tolerance and restoration_error == 0)
    write(args.output,receipt)
    print(json.dumps(receipt))
    if not receipt['passed']:
        raise ValueError('Actual-forward parity failed')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command',required=True)
    plan = sub.add_parser('plan')
    plan.add_argument('--data',type=Path,default=DEFAULT_DATA)
    plan.add_argument('--output',type=Path,required=True)
    plan.add_argument('--seed',default='2048')
    plan.add_argument('--discovery-per-source',type=int,default=8,help='0 selects all eligible rows')
    plan.add_argument('--evaluation-per-source',type=int,default=12,help='0 selects all eligible rows')
    plan.set_defaults(func=command_plan)
    for name in ('collect','fit','evaluate','verify'):
        p = sub.add_parser(name)
        p.add_argument('--plan',type=Path,required=True)
        p.add_argument('--output',type=Path,required=True)
        if name in ('collect','verify','evaluate'):
            p.add_argument('--data',type=Path,default=DEFAULT_DATA)
            p.add_argument('--parity-tolerance',type=float,default=2e-5)
        if name in ('collect','verify'):
            p.add_argument('--model',type=Path,default=DEFAULT_MODEL)
            p.add_argument('--expected-model-sha256',default=DEPLOYED_SHA256)
            p.add_argument('--device',default='cpu')
            p.add_argument('--threads',type=int,default=2)
        if name == 'collect':
            p.add_argument('--role',choices=('discovery','evaluation'),required=True)
            p.add_argument('--batch-size',type=int,default=1)
        if name == 'fit':
            p.add_argument('--discovery',type=Path,required=True)
            p.add_argument('--strength',type=float,default=.5)
            p.add_argument('--max-relative-norm',type=float,default=.1)
        if name in ('verify','evaluate'):
            p.add_argument('--patch',type=Path,required=True)
        if name == 'verify':
            p.add_argument('--row-id', help='Verify a specific row from the evaluation selection')
        if name == 'evaluate':
            p.add_argument('--evaluation',type=Path,required=True)
        p.set_defaults(func={'collect':command_collect,'fit':command_fit,'evaluate':command_evaluate,'verify':command_verify}[name])
    args = parser.parse_args()
    if hasattr(args,'batch_size') and args.batch_size < 1:
        parser.error('--batch-size must be positive')
    if args.command == 'plan' and (args.discovery_per_source < 0 or args.evaluation_per_source < 0):
        parser.error('source caps must be nonnegative')
    inputs = [getattr(args,name) for name in ('data','model','plan','discovery','evaluation','patch') if hasattr(args,name)]
    model_paths = [DEFAULT_MODEL] + ([args.model] if hasattr(args,'model') else [])
    guard_output(args.output, inputs=inputs, model_paths=model_paths)
    if args.command == 'collect':
        guard_output(str(args.output) + '.receipt.json', inputs=inputs, model_paths=model_paths)
    args.func(args)


if __name__ == '__main__':
    main()
