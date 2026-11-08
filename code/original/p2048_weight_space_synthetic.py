"""Build the synthetic benchmark: 400 template questions with labels known by construction.

Sections: relations (Re-DocRED Choice), temporal (MAVEN/MATRES Choice), verdict (FEVEROUS Choice)
and security (legacy workflow Noul). Every entity is invented. Deterministic from SALT.
Design and rule: docs/p2048-weight-space-synthetic-benchmark.md (written before generation).
"""
import argparse, collections, json, random, re, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import p2048_weight_space as wsp
import p2048_weight_space_external as ext

ROOT = wsp.ROOT
SALT = 'synthetic-20261108'


def rng_for(*parts):
    return random.Random(ext.h(SALT, *parts))


# ---------- invented names ----------
ONSETS = ['B', 'D', 'K', 'M', 'N', 'R', 'S', 'T', 'V', 'Z', 'Br', 'Dr', 'Gr', 'Kl', 'Tr', 'Th', 'Vel', 'Mar', 'Or', 'Il']
VOWELS = ['a', 'e', 'i', 'o', 'u', 'ae', 'ei', 'ou']
CODAS = ['n', 'r', 's', 'l', 'th', 'nd', 'rk', 'sk', 'v', 'm', '']


class Namer:
    """Unique invented words; a name never contains another as a token."""
    def __init__(self, rng):
        self.rng, self.used = rng, set()

    def word(self, syllables=2):
        while True:
            w = ''.join(self.rng.choice(ONSETS).lower() + self.rng.choice(VOWELS) + self.rng.choice(CODAS)
                        for _ in range(syllables)).capitalize()
            if w not in self.used and 4 <= len(w) <= 10:
                self.used.add(w)
                return w

    def person(self):
        return f'{self.word()} {self.word()}'

    def country(self):
        return self.word() + self.rng.choice(['ia', 'ania', 'esh', 'mark'])

    def place(self):
        return self.word()

    def org(self):
        return f"{self.word()} {self.rng.choice(['Systems', 'Holdings', 'Logistics', 'Foods', 'Energy', 'Textiles'])}"

    def university(self):
        return f'University of {self.word()}'


# ---------- relations ----------
RELATIONS = ['P17', 'P131', 'P27', 'P108', 'P26', 'P40', 'P112', 'P159', 'P69', 'P19']
DISTRACTOR_POOL = RELATIONS + ['P50', 'P57', 'P166', 'P463', 'P54', 'P361', 'P127', 'P36']
FACTS = {  # property: (subject slot, object slot, templates)
    'P17': ('city1', 'country', ['{s} is a city in {o}.', 'The city of {s} lies in the north of {o}.']),
    'P131': ('city2', 'province', ['{s} is a town in the province of {o}.', '{s} belongs to the {o} province.']),
    'P27': ('p1', 'country', ['{s} holds citizenship of {o}.', '{s} is a citizen of {o}.']),
    'P108': ('p1', 'org', ['{s} works as an engineer for {o}.', '{s} is employed by {o}.']),
    'P26': ('p1', 'p3', ['{s} is married to {o}.', '{s} and her husband {o} live together.']),
    'P40': ('p2', 'p4', ['{s} has a son named {o}.', '{o} is the son of {s}.']),
    'P112': ('org', 'p2', ['{s} was founded by {o}.', '{o} founded {s}.']),
    'P159': ('org', 'city2', ['{s} has its headquarters in {o}.', 'The head office of {s} is in {o}.']),
    'P69': ('p3', 'uni', ['{s} studied at the {o}.', '{s} graduated from the {o}.']),
    'P19': ('p2', 'city1', ['{s} was born in {o}.', '{s} was born in the city of {o}.']),
}


def relation_rows(relation_names, per_relation=10):
    rows = []
    for prop in RELATIONS:
        for k in range(per_relation):
            rng = rng_for('rel', prop, k)
            namer = Namer(rng)
            world = dict(p1=namer.person(), p2=namer.person(), p3=namer.person(), p4=namer.person(),
                         org=namer.org(), uni=namer.university(), city1=namer.place(), city2=namer.place(),
                         province=namer.place(), country=namer.country())
            world['p4'] = world['p4'].split()[0] + ' ' + world['p2'].split()[1]  # son shares the parent's surname
            facts = list(FACTS)
            rng.shuffle(facts)
            sentences, spans = [], collections.defaultdict(list)
            text = ''
            for f in facts:
                s_slot, o_slot, templates = FACTS[f]
                template = rng.choice(templates)
                sentence = template.format(s=world[s_slot], o=world[o_slot])
                start = len(text) + (1 if text else 0)
                text = (text + ' ' if text else '') + sentence
                for slot in (s_slot, o_slot):
                    for m in re.finditer(re.escape(world[slot]), sentence):
                        spans[slot].append((start + m.start(), start + m.end()))
            toks = ext.sentences_with_offsets(text)
            s_slot, o_slot, _ = FACTS[prop]

            def mentions(slot):
                out = []
                for b, e in sorted(set(spans[slot])):
                    where = ext.locate(toks, b, e)
                    covered = ''.join(t for t, _, _ in toks[where[0]][slice(*where[1])])
                    assert covered == world[slot].replace(' ', ''), (covered, world[slot])
                    out.append(dict(name=world[slot], sentence=where[0], token_span=where[1]))
                return out
            blocked = ext.family(prop)
            keys = [prop, *rng.sample([p for p in DISTRACTOR_POOL if p not in blocked], 3)]
            rng.shuffle(keys)
            state = dict(document=ext.render(toks), subject_mentions=mentions(s_slot), object_mentions=mentions(o_slot))
            rows.append(dict(id=f'SYN-REL:{prop}:{k}', type='choice', state=state,
                             question='Which listed relation is annotated from the subject entity to the object entity?',
                             options={p: relation_names[p] for p in keys}, label=prop, hard_label_index=keys.index(prop),
                             source='synthetic:relations', source_family='relations', task='entity_relation',
                             group_component=f'SYN-REL:{prop}:{k}', provenance=dict(world=world, order=facts)))
    return rows


# ---------- temporal ----------
EVENTS = [('the council', 'approved', 'the new budget'), ('the workers', 'repaired', 'the old bridge'),
          ('the museum', 'reopened', 'its east wing'), ('the mayor', 'announced', 'her resignation'),
          ('the court', 'rejected', 'the appeal'), ('the ferry', 'left', 'the harbour'),
          ('the union', 'called', 'a strike'), ('the company', 'recalled', 'its scooters'),
          ('the storm', 'flooded', 'the lower town'), ('the police', 'closed', 'the main road'),
          ('the school', 'cancelled', 'its classes'), ('the farmers', 'blocked', 'the highway'),
          ('the minister', 'signed', 'the treaty'), ('the orchestra', 'performed', 'a new symphony'),
          ('the airline', 'suspended', 'its flights'), ('the hospital', 'opened', 'a new ward'),
          ('the team', 'won', 'the final'), ('the factory', 'stopped', 'production')]
MONTHS = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October',
          'November', 'December']


def clause(event):
    return ' '.join(event)


def temporal_rows(counts=(('BEFORE', 34), ('AFTER', 33), ('SIMULTANEOUS', 33))):
    rows = []
    index = 0
    for label, n in counts:
        for k in range(n):
            rng = rng_for('tmp', label, k)
            namer = Namer(rng)
            town = namer.place()
            e1, e2 = rng.sample(EVENTS, 2)          # e1 happens first (or together, if simultaneous)
            x, y = clause(e1), clause(e2)
            month = rng.randrange(12)
            d1, d2 = rng.randrange(1, 12), rng.randrange(14, 28)
            if label == 'SIMULTANEOUS':
                reversed_order = False
                body = rng.choice([f'While {x}, {y}.', f'{x.capitalize()} at the same time as {y}.',
                                   f'On {d1} {MONTHS[month]}, {x}. At that very moment, {y}.'])
            else:
                reversed_order = k % 2 == 1          # half the items mention the later event first
                if not reversed_order:
                    body = rng.choice([f'On {d1} {MONTHS[month]}, {x}. On {d2} {MONTHS[month]}, {y}.',
                                       f'{x.capitalize()}. Two weeks later, {y}.', f'{x.capitalize()}, and then {y}.'])
                else:
                    body = rng.choice([f'On {d2} {MONTHS[month]}, {y}. Earlier, on {d1} {MONTHS[month]}, {x}.',
                                       f'{y.capitalize()} after {x}.', f'Before {y}, {x}.'])
            text = f'{town} had an eventful month. {body} Residents talked about little else.'
            # A/B assignment: BEFORE means A = earlier event; AFTER means A = later event.
            if label == 'BEFORE':
                a, b = e1, e2
            elif label == 'AFTER':
                a, b = e2, e1
            else:
                a, b = (e1, e2) if rng.random() < .5 else (e2, e1)
            toks = ext.sentences_with_offsets(text)
            state = {'document': ext.render(toks)}
            for name, event in zip('AB', (a, b)):
                start = text.lower().index(clause(event).lower()) + len(event[0]) + 1
                where = ext.locate(toks, start, start + len(event[1]))
                assert toks[where[0]][where[1][0]][0] == event[1]
                state[name] = [dict(trigger_word=event[1], sent_id=where[0], offset=where[1])]
            keys = list(ext.TEMPORAL)
            rng.shuffle(keys)
            rows.append(dict(id=f'SYN-TMP:{label}:{k}', type='choice', state=state,
                             question='Which temporal relation is annotated from A to B?',
                             options={key: ext.TEMPORAL[key] for key in keys}, label=label,
                             hard_label_index=keys.index(label), source='synthetic:temporal', source_family='temporal',
                             task='temporal_relation', group_component=f'SYN-TMP:{label}:{k}',
                             provenance=dict(reversed_mention_order=reversed_order, body=body)))
            index += 1
    return rows


# ---------- verdict ----------
INDUSTRIES = ['textiles', 'shipping', 'solar energy', 'frozen foods', 'medical devices', 'furniture']


def verdict_rows(counts=(('SUPPORTS', 34), ('REFUTES', 33), ('NOT ENOUGH INFO', 33))):
    rows = []
    for label, n in counts:
        for k in range(n):
            rng = rng_for('ver', label, k)
            namer = Namer(rng)
            company, founder, city, country = namer.org(), namer.person(), namer.place(), namer.country()
            year, staff = rng.randrange(1890, 2015), rng.randrange(2, 90) * 100
            industry = rng.choice(INDUSTRIES)
            evidence = [
                dict(source=company, element='title', text=company),
                dict(source=company, element='sentence_0',
                     text=f'{company} is a {industry} company based in {city}, {country}.'),
                dict(source=company, element='sentence_1', text=f'It was founded in {year} by {founder}.'),
                dict(source=company, element='cell_0_1_0', text='Founded:'),
                dict(source=company, element='cell_0_1_1', text=str(year)),
                dict(source=company, element='cell_0_2_0', text='Employees:'),
                dict(source=company, element='cell_0_2_1', text=f'{staff:,}'),
                dict(source=company, element='cell_0_3_0', text='Headquarters:'),
                dict(source=company, element='cell_0_3_1', text=f'{city}, {country}'),
            ]
            rng.shuffle(evidence)
            kind = k % 3
            if label == 'SUPPORTS':
                claim = [f'{company} was founded in {year}.', f'{company} was founded by {founder} and is based in {city}.',
                         f'{company} has {staff:,} employees and makes {industry}.'][kind]
            elif label == 'REFUTES':
                wrong_year = year + rng.choice([-7, -3, 4, 11])
                other = namer.person()
                claim = [f'{company} was founded in {wrong_year}.', f'{company} was founded by {other} and is based in {city}.',
                         f'{company} has {staff * 3:,} employees and makes {industry}.'][kind]
            else:
                claim = [f"{company}'s chief executive is {namer.person()}.",
                         f'{company} was founded by {founder} and opened a branch in {namer.place()} in {year + 20}.',
                         f'{company} reported record profits last year.'][kind]
            keys = list(ext.VERDICTS)
            rng.shuffle(keys)
            rows.append(dict(id=f'SYN-VER:{label}:{k}', type='choice', state=dict(claim=claim, evidence=evidence),
                             question='Which verdict is supported by this evidence for the claim?',
                             options={key: ext.VERDICTS[key] for key in keys}, label=label,
                             hard_label_index=keys.index(label), source='synthetic:verdict', source_family='verdict',
                             task='evidence_verdict', group_component=f'SYN-VER:{label}:{k}',
                             provenance=dict(kind=kind)))
    return rows


# ---------- security ----------
QUESTIONS = {
    'true_positive': ('This alert reflects genuinely malicious or unauthorised activity.',
                      'Benign activity, a misconfiguration, or a known false positive.',
                      'The underlying behaviour is malicious or unauthorised.'),
    'credential_compromise': ('The evidence indicates a credential or account has been compromised.', 'false', 'true'),
}
POSITIVE = [
    ('impossible_travel', 'sign-ins from two distant countries within an hour',
     'At {t} UTC the account {name} signed in from {ext_ip} in a country it had never used before, 38 minutes after '
     'signing in from its usual office. MFA had been switched off on the account two hours earlier from the same '
     'address. Within fifteen minutes the account was added to the domain administrators group and copied {gb} GB '
     'from the finance share to an external file-hosting service. No change ticket or maintenance window covers this.'),
    ('mass_download', 'an unusual volume of files read in a short window',
     'Between {t} and {t2} UTC the account {name} read {files} files from the HR records share and uploaded {gb} GB '
     'to {ext_ip}, an address on a threat-intelligence blocklist. The sign-in used a password that appeared in a '
     'breach dump last week, MFA was not enrolled, and the session originated from a country the account had never '
     'used. No change window was open.'),
    ('privilege_escalation', 'a new administrator role granted outside change control',
     'At {t} UTC the account {name} granted itself the global administrator role, created an access key, and '
     'disabled audit logging on the production database. The session came from {ext_ip}, which is not on the '
     'allowlist, using a token issued after a burst of {files} failed password attempts. No ticket requests these '
     'changes and no change window was open.'),
    ('suspicious_login', 'a sign-in after repeated MFA push rejections',
     'The account {name} rejected eleven MFA push prompts between {t} and {t2} UTC, then accepted the twelfth from '
     '{ext_ip} in a country it had never used. The new session immediately created a mailbox rule forwarding all '
     'mail to an outside address and downloaded {gb} GB of customer records. No change window was open.'),
    ('data_exfiltration', 'large outbound transfer to an unknown destination',
     'At {t} UTC the account {name}, whose password had not been rotated in {stale} days, signed in from {ext_ip} '
     'and compressed {files} files from the payroll server into an encrypted archive, then sent {gb} GB to an '
     'unregistered domain. The source address is not on the allowlist and no change window was open.'),
]
NEGATIVE = [
    ('mass_download', 'an unusual volume of files read in a short window',
     'Between {t} and {t2} UTC the backup service {name} read {files} files from the database dump directory and '
     'copied {gb} GB to the approved backup vault, as it does every night. The source address {int_ip} is on the '
     'allowlist, the job matches change ticket CHG-{ticket}, and the activity fell inside the scheduled maintenance '
     'window.'),
    ('port_scan', 'many connection attempts across hosts',
     'At {t} UTC the scanner account {name} probed {files} hosts on the internal network from {int_ip}. This is the '
     'approved weekly vulnerability scan: the address is on the allowlist, the scan matches change ticket '
     'CHG-{ticket}, and it ran inside the scheduled window. No host reported a successful login or a configuration '
     'change.'),
    ('privilege_escalation', 'a new administrator role granted',
     'At {t} UTC the account {name} received the database administrator role for four hours, as approved in change '
     'ticket CHG-{ticket} by two managers. The session came from {int_ip} in the head office, on the allowlist, '
     'with MFA verified, and the role was removed automatically at the end of the approved window.'),
    ('impossible_travel', 'sign-ins from two locations within an hour',
     'At {t} UTC the account {name} signed in from the head office and, 30 minutes later, from {int_ip}, the '
     'company VPN gateway in the same city. Both sessions passed MFA on the enrolled device. The VPN address is on '
     'the allowlist and the user regularly works through it.'),
    ('data_transfer', 'large outbound transfer',
     'At {t} UTC the sync service {name} sent {gb} GB of encrypted archives from {int_ip} to the company\'s '
     'contracted offsite storage provider, as scheduled in change ticket CHG-{ticket}. The destination is on the '
     'approved vendor list, the address is allowlisted, and the transfer ran inside the maintenance window.'),
]


def security_rows(per_class=25):
    rows = []
    for positive in (True, False):
        for k in range(per_class):
            rng = rng_for('sec', positive, k)
            rule, description, template = (POSITIVE if positive else NEGATIVE)[k % 5]
            hour = rng.randrange(0, 5) if positive else rng.randrange(1, 4)
            name = (f"{rng.choice(['j', 'm', 'a', 'r', 'k'])}.{Namer(rng).word().lower()}" if positive or k % 5 in (2, 3)
                    else f"svc-{rng.choice(['backup', 'scanner', 'sync'])}-{rng.randrange(100, 999)}")
            values = dict(t=f'{hour:02d}:{rng.randrange(60):02d}', t2=f'{hour + 1:02d}:{rng.randrange(60):02d}', name=name,
                          ext_ip=f'{rng.choice(["198.51.100", "203.0.113"])}.{rng.randrange(2, 254)}',
                          int_ip=f'10.{rng.randrange(1, 40)}.{rng.randrange(0, 255)}.{rng.randrange(2, 254)}',
                          gb=f'{rng.randrange(20, 400) / 10:.1f}', files=f'{rng.randrange(1200, 9000):,}',
                          ticket=rng.randrange(1000, 9999), stale=rng.randrange(200, 600))
            state = {
                'alert': {'description': description, 'evidence': template.format(**values), 'rule': rule},
                'context': {'asset_criticality': 'high' if positive else rng.choice(['low', 'medium']),
                            'change_window_active': not positive, 'source_on_allowlist': not positive},
                'history': {'credential_rotation_days_ago': values['stale'] if positive else rng.randrange(5, 40),
                            'distinct_countries_30d': rng.randrange(3, 6) if positive else 1,
                            'logins_30d': rng.randrange(20, 200), 'prior_alerts_90d': rng.randrange(2, 6) if positive else 0},
                'principal': {'mfa_enrolled': (not positive) or k % 5 == 3,  # the push-fatigue alert needs MFA enrolled
                              'name': name,
                              'privileges': ['finance:read', 'hr:read'] if positive else ['backup:write'],
                              'type': 'user' if positive or k % 5 in (2, 3) else 'service'},
            }
            label = int(positive)
            for qid, (question, false_c, true_c) in QUESTIONS.items():
                rows.append(dict(id=f"SYN-SEC:{'pos' if positive else 'neg'}:{k}:{qid}", type='noul', state=state,
                                 question=question, false_criteria=false_c, true_criteria=true_c, label=label,
                                 hard_label_index=label, source='synthetic:security', source_family='security',
                                 task='security_noul', group_component=f"SYN-SEC:{'pos' if positive else 'neg'}:{k}",
                                 provenance=dict(template=k % 5, positive=positive)))
    return rows


_external_source_text = ext.source_text


def source_text(row):
    """Audit text for each section; security alerts have neither a document nor a claim."""
    state = row['state']
    if 'alert' in state:
        return state['alert']['evidence'] + ' ' + row['question']
    return _external_source_text(row)


ext.source_text = source_text  # overlap_audit looks the function up at call time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'research/weight-space/runs/synthetic')
    parser.add_argument('--skip-audit', action='store_true', help='Skip the training-overlap audit (dry runs only)')
    args = parser.parse_args()
    relation_names = json.loads((ROOT / '.local/raw/rel_info.json').read_text())
    rows = relation_rows(relation_names) + temporal_rows() + verdict_rows() + security_rows()
    assert len({r['id'] for r in rows}) == len(rows) == 400
    hits = collections.Counter() if args.skip_audit else ext.overlap_audit(rows)
    clean = [r for r in rows if not hits[r['id']]]
    clean, too_long = ext.fits(clean)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    data_path = args.output_dir / 'data.jsonl'
    with data_path.open('w') as stream:
        for row in clean:
            stream.write(json.dumps(row, ensure_ascii=False) + '\n')
    sections = ('relations', 'temporal', 'verdict', 'security')
    summary = dict(built=collections.Counter(r['source_family'] for r in rows),
                   overlap_removed=collections.Counter(r['source_family'] for r in rows if hits[r['id']]),
                   overlap_audit_run=not args.skip_audit, over_token_limit=too_long,
                   final=collections.Counter(r['source_family'] for r in clean),
                   labels={s: collections.Counter(str(r['label']) for r in clean if r['source_family'] == s) for s in sections})
    plan = dict(schema='p2048-weight-space-synthetic-plan-v1', salt=SALT, data_sha256=wsp.sha(data_path),
                inputs={'.local/raw/rel_info.json': wsp.sha(ROOT / '.local/raw/rel_info.json')}, summary=summary,
                roles=dict(synthetic=[dict(id=r['id'], source=r['source_family'], component=r['group_component'])
                                      for r in clean]),
                rule='Frozen before any model is scored on these rows. See docs/p2048-weight-space-synthetic-benchmark.md.')
    wsp.write(args.output_dir / 'plan.json', plan)
    print(json.dumps(summary, indent=1, default=dict))


if __name__ == '__main__':
    main()
