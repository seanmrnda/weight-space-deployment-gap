#!/usr/bin/env python3
"""Level 2: rebuild the external benchmark (DWIE, MATRES, AVeriTeC) from the public sources and check it.

Every released question must be regenerated with the same gold answer and option position. The original build then
removed questions by a 13-gram overlap audit against private training files (1 removed) and a 2,048-token check with
the private tokenizer (20 removed); those steps are skipped here, so the rebuild contains those extra questions.

Sources (see README for exact commits): DWIE repository plus article text fetched with code/original/fetch_dwie.py,
MATRES platinum.txt with TempEval-3 platinum .tml files, and the AVeriTeC repository's data/ folder.
"""
import argparse, collections, json, sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'code/original'))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--dwie', type=Path, required=True, help='DWIE repository checkout')
    ap.add_argument('--dwie-content', type=Path, required=True, help='output folder of fetch_dwie.py')
    ap.add_argument('--matres', type=Path, required=True, help='folder with platinum.txt')
    ap.add_argument('--te3-platinum', type=Path, required=True, help='TempEval-3 te3-platinum folder of .tml files')
    ap.add_argument('--averitec', type=Path, required=True, help='AVeriTeC repository checkout')
    ap.add_argument('--rel-info', type=Path, required=True)
    args = ap.parse_args()
    import p2048_weight_space as wsp
    import p2048_weight_space_external as ext
    names = json.loads(args.rel_info.read_text())
    dwie, _ = ext.dwie_rows(args.dwie, args.dwie_content, names, 2, 800)
    matres, _ = ext.matres_rows(args.matres, args.te3_platinum)
    averitec, _ = ext.averitec_rows(args.averitec, 100)
    built = {r['id']: r for r in dwie + matres + averitec}
    released = [json.loads(l) for l in open(REPO / 'data/labels/external.jsonl')]
    missing = [r['id'] for r in released if r['id'] not in built]
    mismatched = [r['id'] for r in released if r['id'] in built and
                  (wsp.gold_index(built[r['id']]) != r['gold_index'] or str(built[r['id']].get('label')) != r['label'])]
    extra = collections.Counter(r['source_family'] for i, r in built.items() if i not in {x['id'] for x in released})
    report = {'released': len(released), 'rebuilt': len(built), 'missing': len(missing), 'mismatched': len(mismatched),
              'extra_by_source': dict(extra), 'expected_extra': 21}
    print(json.dumps(report))
    ok = not missing and not mismatched and sum(extra.values()) == 21
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
