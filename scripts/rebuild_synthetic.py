#!/usr/bin/env python3
"""Level 2: regenerate the synthetic benchmark and check it is byte-identical to data/synthetic/data.jsonl.

Needs only the standard library and DocRED's relation-name file `rel_info.json` (public, in the DocRED / Re-DocRED
data release). The original generator also ran a training-overlap audit and a 2,048-token check, which need private
files; neither removed any of the 400 questions, so they are skipped here.
"""
import argparse, hashlib, json, sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'code/original'))
REL_INFO_SHA256 = '5ecf4e5e55c179fc83a3a3d19baa01efffecb26ba5edc0b4ac5a54ddf61fe3de'


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--rel-info', type=Path, required=True)
    args = ap.parse_args()
    digest = hashlib.sha256(args.rel_info.read_bytes()).hexdigest()
    if digest != REL_INFO_SHA256:
        raise SystemExit(f'rel_info.json SHA-256 {digest} differs from the study input {REL_INFO_SHA256}')
    import p2048_weight_space_synthetic as syn
    names = json.loads(args.rel_info.read_text())
    rows = syn.relation_rows(names) + syn.temporal_rows() + syn.verdict_rows() + syn.security_rows()
    text = ''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows)
    built = hashlib.sha256(text.encode()).hexdigest()
    released = hashlib.sha256((REPO / 'data/synthetic/data.jsonl').read_bytes()).hexdigest()
    print(json.dumps({'rows': len(rows), 'rebuilt_sha256': built, 'released_sha256': released, 'identical': built == released}))
    sys.exit(0 if built == released else 1)


if __name__ == '__main__':
    main()
