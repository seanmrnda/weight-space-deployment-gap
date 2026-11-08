"""Freeze search/validation/lockbox roles over all leak-free domain Choice rows.

Assignment uses connected document/page/claim components and a salted hash, and
never reads model predictions. Components touching any question inspected by an
earlier WSP round are forced into search, so validation and lockbox stay unseen
by weight-editing work.
"""
import argparse, collections, hashlib, json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import p2048_weight_space as wsp

PANELS = ('retention_validation', 'retention_test', 'retention_confirmation',
          'legacy_v3_fresh', 'confirmation', 'calibration', 'development')
SOURCES = {'domain:Re-DocRED': 'Re-DocRED', 'domain:MAVEN-ERE': 'MAVEN-ERE', 'domain:FEVEROUS': 'FEVEROUS'}
TRAINING = ('.local/praxis-p2048-targeted-02/prepared/train.jsonl', '.local/praxis-p-2048-01/prepared/train.jsonl',
            '.local/praxis-next-run-03/prepared/train.jsonl', '.local/v3/prepared/train.jsonl',
            '.local/v2/prepared/train.jsonl', '.local/prepared/train.jsonl')


def connected(rows):
    parent = {}
    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for row in rows:
        keys = [row['group_component'], *row.get('split_groups', [])]
        for key in keys[1:]:
            a, b = find(keys[0]), find(key)
            if a != b:
                parent[max(a, b)] = min(a, b)
    return {r['id']: find(r['group_component']) for r in rows}


def verify_round_trip(path, rows):
    """Written rows must keep source option order, state order and gold alignment."""
    with Path(path).open() as stream:
        written = [json.loads(line) for line in stream]
    if len(written) != len(rows):
        raise ValueError('Row count changed on write')
    for row in written:
        source = rows[row['id']]
        if list(row['options']) != list(source['options']) or json.dumps(row['state']) != json.dumps(source['state']):
            raise ValueError('Option or state order changed: ' + row['id'])
        if list(row['options'])[row['hard_label_index']] != row['label']:
            raise ValueError('hard_label_index does not point at label: ' + row['id'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepared', type=Path, default=Path('.local/praxis-p2048-targeted-02/prepared'))
    parser.add_argument('--prior-plan', type=Path, default=Path('research/weight-space/runs/ancestry/plan.json'))
    parser.add_argument('--salt', default='soup-20261107')
    parser.add_argument('--per-component-cap', type=int, default=4)
    parser.add_argument('--output-dir', type=Path, default=Path('research/weight-space/runs/soup'))
    args = parser.parse_args()

    rows, origin = {}, {}
    for panel in PANELS:
        for line in (args.prepared / f'{panel}.jsonl').open():
            row = json.loads(line)
            if row.get('source') in SOURCES and row.get('type') == 'choice':
                rows.setdefault(row['id'], row)
                origin.setdefault(row['id'], panel)
    training_groups = set()
    for path in TRAINING:
        for line in open(path):
            row = json.loads(line)
            training_groups.add(row.get('group_component'))
            training_groups.update(row.get('split_groups') or [])
    leaked = [i for i, r in rows.items() if ({r['group_component'], *r.get('split_groups', [])} & training_groups)]
    if leaked:
        raise ValueError(f'{len(leaked)} rows share a group with training data')

    ordered = [rows[i] for i in sorted(rows)]
    component = connected(ordered)
    prior = json.loads(args.prior_plan.read_text())
    inspected = set(prior['excluded_old_ids']) | {x['id'] for v in prior['roles'].values() for x in v}
    forced = {component[i] for i in inspected}

    def role(key):
        if key in forced:
            return 'search'
        bucket = int(hashlib.sha256(f'{args.salt}:{key}'.encode()).hexdigest(), 16) % 100
        return 'search' if bucket < 20 else 'validation' if bucket < 50 else 'lockbox'

    members = collections.defaultdict(list)
    for row in ordered:
        members[component[row['id']]].append(row)
    roles = collections.defaultdict(list)
    for key in sorted(members):
        # Hash-ordered cap: avoids one document dominating a panel, label-blind.
        kept = sorted(members[key], key=lambda r: hashlib.sha256(f"{args.salt}:{r['id']}".encode()).hexdigest())
        by_source = collections.Counter()
        for row in kept:
            name = SOURCES[row['source']]
            if by_source[name] >= args.per_component_cap:
                continue
            by_source[name] += 1
            roles[role(key)].append(dict(id=row['id'], source=name, component=key, origin=origin[row['id']],
                                         task=row.get('task'), inspected=row['id'] in inspected))
    data_path = args.output_dir / 'data.jsonl'
    with data_path.open('w') as stream:
        for row in ordered:
            # Key order is data here: options order defines hard_label_index and the model input.
            stream.write(json.dumps(dict(row, source_family=SOURCES[row['source']],
                                         connected_component=component[row['id']])) + '\n')
    verify_round_trip(data_path, rows)
    plan = dict(schema='p2048-weight-space-soup-plan-v1', salt=args.salt, data_sha256=wsp.sha(data_path),
                per_component_cap=args.per_component_cap, roles=roles, leaked_rows_removed=0,
                prior_inspected_rows=len(inspected), forced_search_components=len(forced),
                rule='Lockbox is opened once, for one frozen candidate plus pre-listed reference arms.')
    wsp.write(args.output_dir / 'plan.json', plan)
    for name, items in roles.items():
        count = collections.Counter(i['source'] for i in items)
        comps = {s: len({i['component'] for i in items if i['source'] == s}) for s in count}
        tasks = collections.Counter(i['task'] for i in items)
        print(name, len(items), dict(count), 'components', comps, dict(tasks))
    print('plan_sha256', wsp.sha(args.output_dir / 'plan.json'))


if __name__ == '__main__':
    main()
