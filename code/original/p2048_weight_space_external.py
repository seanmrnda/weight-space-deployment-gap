"""Build the independent external benchmark and the legacy guard set as one run.

Three published benchmarks are converted to the repository's typed Choice
format, mirroring the in-domain question structures:

* DWIE (Deutsche Welle news) -> Re-DocRED-style relation choice. Only DWIE
  relation types with a clear Wikidata counterpart are kept, with direction
  corrected where Wikidata's subject differs. Three distractors come from the
  Re-DocRED relation set, excluding gold relations and their implication family.
* MATRES Platinum (TempEval-3 news) -> event temporal relation. MATRES keeps
  real direction: BEFORE, AFTER, and EQUAL as SIMULTANEOUS. VAGUE is dropped.
* AVeriTeC (real-world fact-checks, web evidence) -> FEVEROUS-style verdict.
  Evidence is the annotated question-answer pairs, never the justification.
  Conflicting Evidence/Cherrypicking is dropped.

Every sampling choice is a salted hash, fixed before any model is scored. A
13-word shingle audit removes rows whose text overlaps any training file. The
legacy confirmation set is included unchanged for deployment guards.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import random
import re
import sys
from pathlib import Path
from urllib.parse import urlparse

sys.path.insert(0, str(Path(__file__).resolve().parent))
import p2048_weight_space as wsp

ROOT = wsp.ROOT
SALT = 'external-20261107'
TRAINING = ('.local/praxis-p2048-targeted-02/prepared/train.jsonl', '.local/praxis-p-2048-01/prepared/train.jsonl',
            '.local/praxis-next-run-03/prepared/train.jsonl', '.local/v3/prepared/train.jsonl',
            '.local/v2/prepared/train.jsonl', '.local/prepared/train.jsonl')

# DWIE type -> (Wikidata property, reverse direction?)
DWIE_MAP = {
    'in0': ('P17', False), 'based_in0': ('P17', False), 'event_in0': ('P17', False),
    'in2': ('P131', False), 'based_in2': ('P159', False), 'event_in2': ('P276', False),
    'citizen_of': ('P27', False), 'member_of': ('P463', False), 'player_of': ('P54', False),
    'head_of_state': ('P35', True), 'head_of_gov': ('P6', True), 'spouse_of': ('P26', False),
    'parent_of': ('P40', False), 'child_of': ('P40', True), 'sibling': ('P3373', False),
    'award_received': ('P166', False), 'directed_by': ('P57', False), 'founder_of': ('P112', True),
    'part_of': ('P361', False), 'character_in': ('P1441', False), 'product_of': ('P176', False),
    'publisher': ('P123', False),
}
# Relations that can co-hold with each other; never offered as distractors for one another.
FAMILIES = [
    {'P17', 'P131', 'P159', 'P276', 'P150', 'P495', 'P740', 'P551', 'P19', 'P20', 'P30', 'P706', 'P36', 'P1376',
     'P205', 'P206', 'P291', 'P937'},
    {'P463', 'P54', 'P102', 'P108', 'P488', 'P39', 'P1344', 'P710'},
    {'P361', 'P527', 'P355', 'P749', 'P127', 'P137', 'P176', 'P123', 'P112', 'P170', 'P178', 'P272', 'P264'},
    {'P26', 'P40', 'P3373', 'P22', 'P25'},
    {'P35', 'P6', 'P27', 'P1376', 'P36'},
    {'P57', 'P58', 'P86', 'P161', 'P175', 'P50', 'P170', 'P162', 'P676', 'P1441', 'P674'},
]
TEMPORAL = {'BEFORE': 'A occurs before B.', 'AFTER': 'A occurs after B.', 'SIMULTANEOUS': 'A and B are simultaneous.'}
MATRES_LABEL = {'BEFORE': 'BEFORE', 'AFTER': 'AFTER', 'EQUAL': 'SIMULTANEOUS'}
VERDICTS = {'SUPPORTS': 'The evidence supports the claim.', 'REFUTES': 'The evidence refutes the claim.',
            'NOT ENOUGH INFO': 'The evidence is insufficient to decide the claim.'}
AVERITEC_LABEL = {'Supported': 'SUPPORTS', 'Refuted': 'REFUTES', 'Not Enough Evidence': 'NOT ENOUGH INFO'}

# Possessives and hyphens split off, as in Re-DocRED's tokenization ("Iran 's", "Beaux - Arts").
TOKEN = re.compile(r"\w+|['’]\w+|[^\w\s]")
SENTENCE = re.compile(r'(?:(?<=[.!?])|(?<=[.!?]["”’)]))\s+(?=["“(]?[A-Z0-9])|\n+')


def h(*parts):
    return hashlib.sha256(':'.join([SALT, *map(str, parts)]).encode()).hexdigest()


def sentences_with_offsets(text):
    """Split into sentences and tokens, keeping character offsets for span mapping."""
    out, start = [], 0
    for match in [*SENTENCE.finditer(text), None]:
        end = match.start() if match else len(text)
        chunk = text[start:end]
        tokens = [(m.group(), start + m.start(), start + m.end()) for m in TOKEN.finditer(chunk)]
        if tokens:
            out.append(tokens)
        start = match.end() if match else len(text)
    return out


def locate(sentences, begin, end):
    """Map a character span to (sentence, [first token, last token + 1]) when inside one sentence."""
    for index, tokens in enumerate(sentences):
        inside = [i for i, (_, b, e) in enumerate(tokens) if b < end and e > begin]
        if inside:
            if tokens[inside[0]][1] > begin + 2 and inside[0] == 0:
                return None
            return index, [inside[0], inside[-1] + 1]
    return None


def render(sentences):
    return '\n'.join(f'{i}: ' + ' '.join(t for t, _, _ in tokens) for i, tokens in enumerate(sentences))


def family(relation):
    return set().union(*(f for f in FAMILIES if relation in f)) | {relation}


def dwie_rows(repo, content_dir, relation_names, per_doc, cap):
    rows, dropped = [], collections.Counter()
    relations = sorted(relation_names)
    for path in sorted((repo / 'data/annos').glob('*.json')):
        anno = json.loads(path.read_text())
        fetched = content_dir / path.name
        if not fetched.exists():
            dropped['no_content'] += 1
            continue
        record = json.loads(fetched.read_text())
        if not record['check'].get('hash_match'):
            dropped['hash_mismatch'] += 1
            continue
        text = record['content']
        sentences = sentences_with_offsets(text)
        mentions = collections.defaultdict(list)
        for m in anno['mentions']:
            if text[m['begin']:m['end']] != m['text']:
                dropped['mention_misaligned'] += 1
                continue
            where = locate(sentences, m['begin'], m['end'])
            if not where:
                dropped['mention_crosses_sentence'] += 1
                continue
            covered = ''.join(t for t, _, _ in sentences[where[0]][slice(*where[1])])
            if covered != m['text'].replace(' ', ''):
                dropped['mention_span_inexact'] += 1
                continue
            mentions[m['concept']].append(dict(name=m['text'], sentence=where[0], token_span=where[1]))
        pairs = collections.defaultdict(set)
        for r in anno['relations']:
            if r['p'] not in DWIE_MAP:
                continue
            prop, reverse = DWIE_MAP[r['p']]
            s, o = (r['o'], r['s']) if reverse else (r['s'], r['o'])
            if prop in relation_names and mentions.get(s) and mentions.get(o) and s != o:
                pairs[s, o].add(prop)
        rng = random.Random(h('dwie', anno['id']))
        chosen = sorted(pairs.items())
        rng.shuffle(chosen)
        for (s, o), golds in chosen[:per_doc]:
            gold = rng.choice(sorted(golds))
            blocked = set().union(*(family(g) for g in golds))
            keys = [gold, *rng.sample([k for k in relations if k not in blocked], 3)]
            rng.shuffle(keys)
            state = dict(document=render(sentences), subject_mentions=mentions[s], object_mentions=mentions[o])
            rows.append(dict(id=f"DWIE:{anno['id']}:{s}:{o}", type='choice', state=state,
                             question='Which listed relation is annotated from the subject entity to the object entity?',
                             options={k: relation_names[k] for k in keys}, label=gold,
                             hard_label_index=keys.index(gold), source='external:DWIE', source_family='DWIE',
                             task='entity_relation', group_component=f"DWIE:{anno['id']}",
                             provenance=dict(dataset='DWIE', article=anno['id'], tags=anno['tags'], golds=sorted(golds))))
    rows.sort(key=lambda r: h('dwie-cap', r['id']))
    return rows[:cap], dropped


def tml_document(path):
    raw = path.read_text(encoding='utf-8')
    title = re.search(r'<TITLE>(.*?)</TITLE>', raw, re.S)
    body = re.search(r'<TEXT>(.*?)</TEXT>', raw, re.S).group(1)
    instances = {m.group(1): m.group(2) for m in re.finditer(r'<MAKEINSTANCE[^>]*?eiid="ei(\d+)"[^>]*?eventID="(e\d+)"', raw)}
    instances.update({m.group(2): m.group(1) for m in re.finditer(r'<MAKEINSTANCE[^>]*?eventID="(e\d+)"[^>]*?eiid="ei(\d+)"', raw)})
    text, events, cursor = '', {}, 0
    prefix = (re.sub(r'<[^>]+>', '', title.group(1)).strip() + '\n') if title else ''
    text = prefix
    squash = lambda segment: re.sub(r'[ \t]+', ' ', segment)
    for m in re.finditer(r'<EVENT[^>]*?eid="(e\d+)"[^>]*>(.*?)</EVENT>|<[^>]+>', body, re.S):
        # Normalise each segment before recording offsets, so event spans stay exact.
        text += squash(body[cursor:m.start()])
        if m.group(1):
            begin = len(text)
            text += m.group(2)
            events[m.group(1)] = (begin, len(text), m.group(2))
        cursor = m.end()
    text += squash(body[cursor:])
    return text, events, instances


def matres_rows(matres, te3_dir):
    rows, dropped = [], collections.Counter()
    documents = {}
    for line in (matres / 'platinum.txt').read_text().splitlines():
        doc, verb1, verb2, ei1, ei2, relation = line.split('\t')
        relation = relation.strip()
        if relation not in MATRES_LABEL:
            dropped['vague'] += 1
            continue
        if doc not in documents:
            raw_path = te3_dir / f'{doc}.tml'
            documents[doc] = tml_document(raw_path)
        # The whitespace normalisation shifts offsets, so re-locate each event by recomputing.
        text, events, instances = documents[doc]
        spans = []
        for eiid, verb in ((ei1, verb1), (ei2, verb2)):
            eid = instances.get(eiid)
            if eid not in events or events[eid][2].strip() != verb:
                spans = None
                break
            spans.append(events[eid])
        if not spans:
            dropped['event_unresolved'] += 1
            continue
        sentences = sentences_with_offsets(text)
        state = {'document': render(sentences)}
        ok = True
        for name, (begin, end, word) in zip('AB', spans):
            where = locate(sentences, begin, end)
            if not where:
                ok = False
                break
            state[name] = [dict(trigger_word=word, sent_id=where[0], offset=where[1])]
        if not ok:
            dropped['span_unresolved'] += 1
            continue
        label = MATRES_LABEL[relation]
        rng = random.Random(h('matres', doc, ei1, ei2))
        keys = list(TEMPORAL)
        rng.shuffle(keys)
        rows.append(dict(id=f'MATRES:{doc}:{ei1}:{ei2}', type='choice', state=state,
                         question='Which temporal relation is annotated from A to B?',
                         options={k: TEMPORAL[k] for k in keys}, label=label, hard_label_index=keys.index(label),
                         source='external:MATRES', source_family='MATRES', task='temporal_relation',
                         group_component=f'MATRES:{doc}', provenance=dict(dataset='MATRES-platinum', doc=doc,
                                                                          original=relation)))
    return rows, dropped


def averitec_rows(repo, per_label_train):
    rows, dropped = [], collections.Counter()
    for split in ('dev', 'train'):
        claims = json.loads((repo / f'data/{split}.json').read_text())
        candidates = []
        for index, claim in enumerate(claims):
            if claim['label'] not in AVERITEC_LABEL:
                dropped[f'{split}:conflicting'] += 1
                continue
            evidence = []
            for q_index, question in enumerate(claim.get('questions') or []):
                for a_index, answer in enumerate(question.get('answers') or []):
                    text = (answer.get('answer') or '').strip()
                    if answer.get('boolean_explanation'):
                        text = f"{text}. {answer['boolean_explanation'].strip()}"
                    domain = urlparse(answer.get('source_url') or '').netloc or 'web'
                    evidence.append(dict(source=domain, element=f'qa_{q_index}_{a_index}',
                                         text=f"Q: {question['question'].strip()} A: {text}"))
            if not evidence:
                dropped[f'{split}:no_evidence'] += 1
                continue
            label = AVERITEC_LABEL[claim['label']]
            candidates.append((index, claim, evidence, label))
        if split == 'train':
            by_label = collections.defaultdict(list)
            for item in sorted(candidates, key=lambda c: h('averitec-train', c[0])):
                by_label[item[3]].append(item)
            candidates = [c for label in sorted(by_label) for c in by_label[label][:per_label_train]]
        for index, claim, evidence, label in candidates:
            rng = random.Random(h('averitec', split, index))
            keys = list(VERDICTS)
            rng.shuffle(keys)
            rows.append(dict(id=f'AVeriTeC:{split}:{index}', type='choice',
                             state=dict(claim=claim['claim'].strip(), evidence=evidence),
                             question='Which verdict is supported by this evidence for the claim?',
                             options={k: VERDICTS[k] for k in keys}, label=label, hard_label_index=keys.index(label),
                             source='external:AVeriTeC', source_family='AVeriTeC', task='evidence_verdict',
                             group_component=f'AVeriTeC:{split}:{index}',
                             provenance=dict(dataset='AVeriTeC', split=split, index=index, claim_date=claim.get('claim_date'))))
    return rows, dropped


def shingles(text, n=13):
    words = re.findall(r'\w+', text.lower())
    return {hash(' '.join(words[i:i + n])) for i in range(max(0, len(words) - n + 1))}


def source_text(row):
    state = row['state']
    if 'document' in state:
        return state['document']
    return state['claim'] + ' ' + ' '.join(e['text'] for e in state['evidence'])


def overlap_audit(rows):
    """Remove rows sharing any 13-word shingle with any training row."""
    index = collections.defaultdict(set)
    for row in rows:
        for value in shingles(source_text(row)):
            index[value].add(row['id'])
    hits = collections.Counter()
    for path in TRAINING:
        with open(ROOT / path) as stream:
            for line in stream:
                training = json.loads(line)
                state = training.get('state')
                text = (state if isinstance(state, str) else json.dumps(state, ensure_ascii=False)) + ' ' + training.get('question', '')
                for value in shingles(text):
                    for identifier in index.get(value, ()):
                        hits[identifier] += 1
    return hits


def fits(rows):
    """Keep rows that the 2,048-token collator accepts."""
    from transformers import AutoTokenizer
    from decision_moe.data import DecisionCollator
    tokenizer = AutoTokenizer.from_pretrained(wsp.DEFAULT_MODEL / 'tokenizer', local_files_only=True)
    collate = DecisionCollator(tokenizer, 2048, 512, labeled=False)
    kept, dropped = [], 0
    for row in rows:
        try:
            collate([wsp.rotate(row, 0)[0]])
            kept.append(row)
        except ValueError:
            dropped += 1
    return kept, dropped


def build_legacy(output_dir):
    """The repository's legacy confirmation set, unchanged, for the deployment guards."""
    source = ROOT / '.local/praxis-p2048-targeted-02/prepared/legacy_confirmation.jsonl'
    rows = [json.loads(line) for line in open(source)]
    for row in rows:
        row['group_component'] = row.get('group_id') or row['id']
    output_dir.mkdir(parents=True, exist_ok=True)
    data_path = output_dir / 'data.jsonl'
    with data_path.open('w') as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + '\n')
    plan = dict(schema='p2048-weight-space-legacy-plan-v1', source=str(source.relative_to(ROOT)), source_sha256=wsp.sha(source),
                data_sha256=wsp.sha(data_path),
                roles=dict(legacy=[dict(id=r['id'], source=r.get('source_family', 'legacy'), component=r['group_component'],
                                        type=r['type']) for r in rows]),
                guards='Repository release guards: per type and per workflow slice, accuracy delta >= -2 points vs L '
                       'when the slice has at least 200 rows; security_incidents Noul positives: no more false negatives than L.')
    wsp.write(output_dir / 'plan.json', plan)
    print(json.dumps(dict(rows=len(rows), types=collections.Counter(r['type'] for r in rows)), indent=1))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--external', type=Path, default=ROOT / '.local/external-benchmarks')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'research/weight-space/runs/external')
    parser.add_argument('--dwie-per-doc', type=int, default=2)
    parser.add_argument('--dwie-cap', type=int, default=800)
    parser.add_argument('--averitec-train-per-label', type=int, default=100)
    parser.add_argument('--part', choices=('external', 'legacy'), required=True,
                        help='Each part is its own run, so its prediction cache does not depend on the other')
    args = parser.parse_args()
    if args.part == 'legacy':
        return build_legacy(args.output_dir)
    relation_names = json.loads((ROOT / '.local/raw/rel_info.json').read_text())
    x = args.external
    dwie, dwie_drop = dwie_rows(x / 'DWIE', x / 'DWIE-content', relation_names, args.dwie_per_doc, args.dwie_cap)
    matres, matres_drop = matres_rows(x / 'MATRES', x / 'TE3/data/TempEval3/Evaluation/te3-platinum')
    averitec, averitec_drop = averitec_rows(x / 'AVeriTeC-repo', args.averitec_train_per_label)
    external = dwie + matres + averitec
    hits = overlap_audit(external)
    clean = [r for r in external if not hits[r['id']]]
    clean, too_long = fits(clean)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    data_path = args.output_dir / 'data.jsonl'
    with data_path.open('w') as stream:
        for row in clean:
            stream.write(json.dumps(row, ensure_ascii=False) + '\n')
    roles = dict(external=[dict(id=r['id'], source=r['source_family'], component=r['group_component']) for r in clean])
    inputs = {str(p.relative_to(ROOT)): wsp.sha(p) for p in [
        x / 'DWIE-content/manifest.json', x / 'MATRES/platinum.txt', x / 'AVeriTeC-repo/data/dev.json',
        x / 'AVeriTeC-repo/data/train.json', ROOT / '.local/raw/rel_info.json']}
    summary = dict(
        built=dict(DWIE=len(dwie), MATRES=len(matres), AVeriTeC=len(averitec)),
        dropped=dict(DWIE=dwie_drop, MATRES=matres_drop, AVeriTeC=averitec_drop),
        overlap_removed=collections.Counter(r['source_family'] for r in external if hits[r['id']]),
        over_token_limit=too_long,
        final=collections.Counter(r['source_family'] for r in clean),
        labels={s: collections.Counter(r['label'] for r in clean if r['source_family'] == s) for s in ('DWIE', 'MATRES', 'AVeriTeC')})
    plan = dict(schema='p2048-weight-space-external-plan-v1', salt=SALT, data_sha256=wsp.sha(data_path), inputs=inputs,
                dwie_map=DWIE_MAP, families=[sorted(f) for f in FAMILIES], summary=summary, roles=roles,
                rule='Frozen before any model is scored on these rows.')
    wsp.write(args.output_dir / 'plan.json', plan)
    print(json.dumps(summary, indent=1, default=dict))


if __name__ == '__main__':
    main()
