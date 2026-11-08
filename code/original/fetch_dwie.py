"""Fetch DWIE article text from the DW API and verify each article's published hash.

Reimplements only the fetch-and-hash step of DWIE's own downloader, so no code
from the downloaded repository runs. Content is title, teaser and text joined by
newlines, with the repository's one documented text correction applied. Writes
one JSON per article with its content and a manifest of hash outcomes.
"""
import argparse, hashlib, json, time
from pathlib import Path
import urllib.request

FIXES = {'DW_40663341': [("starting with Sunday's", "starting Sunday's"),
                         ('$1 million (€840,000)', 'one million dollars (840,000 euros)'),
                         ('who kneel in protest during', 'to kneel in protest during')]}
RENAMED = {18525950: 19026607}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--delay', type=float, default=0.3)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    articles = json.loads((args.repo / 'data/article_id_to_url.json').read_text())
    manifest = []
    for item in articles:
        target = args.output / f"{item['id']}.json"
        if target.exists():
            manifest.append(json.loads(target.read_text())['check'])
            continue
        number = int(item['id'][3:])
        url = item['url'].replace(str(number), str(RENAMED[number])) if number in RENAMED else item['url']
        check = dict(id=item['id'], url=url, expected_sha1=item['hash'])
        try:
            with urllib.request.urlopen(url, timeout=30) as response:
                data = json.loads(response.read().decode('utf-8'))
            content = '{}\n{}\n{}'.format(data.get('name', ''), data.get('teaser', ''), data.get('text', ''))
            for old, new in FIXES.get(item['id'], []):
                content = content.replace(old, new)
            check['sha1'] = hashlib.sha1(content.encode('utf-8')).hexdigest()
            check['hash_match'] = check['sha1'] == item['hash']
            target.write_text(json.dumps(dict(id=item['id'], content=content, check=check)))
        except Exception as error:   # record and continue; failures are reported, not hidden
            check['error'] = repr(error)
        manifest.append(check)
        time.sleep(args.delay)
    (args.output / 'manifest.json').write_text(json.dumps(manifest, indent=1))
    ok = sum(bool(m.get('hash_match')) for m in manifest)
    print(json.dumps(dict(articles=len(manifest), hash_match=ok,
                          mismatch=sum(m.get('hash_match') is False for m in manifest),
                          errors=sum('error' in m for m in manifest))))


if __name__ == '__main__':
    main()
