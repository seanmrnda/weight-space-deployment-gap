# Weight-Space Editing Wins Benchmarks but Not Deployment: reproducibility artifacts

This repository holds the artifacts for the paper *Weight-Space Editing Wins Benchmarks but Not Deployment: A Pre-Registered Study on a Small Decision Model*.

The study edited the weights of a 188M-parameter decision model over seven pre-registered rounds. The edits included targeted repairs, ancestry restoration, task arithmetic and interpolation. Each round was evaluated on its benchmarks and against deployment-style release guards. The main result: the edits improve benchmark scores, including on an independent benchmark, yet none passes the release guards.

## What you can reproduce

| Level | What it checks | Needs | Time |
| --- | --- | --- | --- |
| **1. Verify** | The main results: every table and figure in the paper's results and deployment sections, including accuracies, paired bootstrap intervals, gate decisions and sign tests. They are recomputed from the released logits and compared with the published values. Rounds 1 to 5 are documented by their original receipts. | Python 3.9 or later, standard library only | about 10 seconds |
| **2. Rebuild data** | The synthetic benchmark (byte-identical) and the external benchmark (every released question and gold answer), rebuilt from public sources | Python standard library and the public source datasets | minutes |
| **3. Rescore models** | Re-running the forward passes | The model checkpoints, which are **not public** (see [Checkpoints](#checkpoints)) | — |

### Level 1: verify the main results

```bash
python3 scripts/verify_all.py
```

The script prints one line per check and ends with `219 of 219 checks match the paper.` It exits non-zero if any check fails. The intervals reproduce digit for digit because the script uses the study's exact procedure: paired component-cluster bootstrap, 4,000 draws, seed 2049, rows in frozen plan order.

### Level 2: rebuild the evaluation data

**Synthetic benchmark (400 questions, our own data, released in full).**

```bash
python3 scripts/rebuild_synthetic.py --rel-info <DocRED rel_info.json>
```

The script needs DocRED's relation-name file (SHA-256 checked by the script). It regenerates `data/synthetic/data.jsonl` byte for byte.

**External benchmark (2,265 questions).**

1. Fetch the sources at these commits:

   | Source | Location | Commit or date |
   | --- | --- | --- |
   | DWIE annotations | `github.com/klimzaporojets/DWIE` | `575e0be3e6f980b77152f587b26530082f94a6c1` |
   | DWIE article text | DW API, via `code/original/fetch_dwie.py --repo <DWIE> --output <content>` | fetched 2026-11-07; each article's published SHA-1 is checked |
   | MATRES | `github.com/qiangning/MATRES` | `ccac302b546401a1e55f2bd6097913083b5cd8d6` |
   | TempEval-3 Platinum text | `github.com/qiangning/StructTempRel-EMNLP17`, `data/TempEval3` | `d6518a5a79d940d749c95ee5b72830714b132238` |
   | AVeriTeC | `github.com/MichSchli/AVeriTeC`, `data/` | `7c62d1ec8df3fb560d6efe2b85fa191135636f81` |

2. Rebuild and check:

   ```bash
   python3 scripts/rebuild_external.py --dwie <DWIE> --dwie-content <content> --matres <MATRES> \
     --te3-platinum <TE3>/data/TempEval3/Evaluation/te3-platinum --averitec <AVeriTeC> --rel-info <rel_info.json>
   ```

Every released question is regenerated with the same gold answer and option position. Two steps of the original build need private files and are skipped: a training-overlap audit, which removed 1 question, and a 2,048-token check with the model's tokenizer, which removed 20. The rebuild therefore contains exactly 21 extra questions, and the script checks that too.

**Study panels and legacy guards.** The Re-DocRED, MAVEN-ERE and FEVEROUS panels and the legacy release-guard set were built by the project's internal data pipeline, which is not released. They ship as question IDs, connected components, gold indices and labels, without text. The frozen plans in `data/plans/` record their hashes.

## Repository layout

| Path | Contents |
| --- | --- |
| `scripts/verify_all.py` | Level 1 verification |
| `scripts/rebuild_synthetic.py`, `scripts/rebuild_external.py` | Level 2 rebuilds |
| `data/labels/` | Per-question ID, source, bootstrap component, type, gold index and label, in frozen plan order. The screen panel is derived from `search` (see `verify_all.py`). |
| `data/synthetic/data.jsonl` | The full synthetic benchmark (invented entities, labels known by construction) |
| `data/plans/` | Frozen data plans: role assignments, components and data hashes |
| `results/logits/<panel>/<fingerprint>.jsonl` | Saved logits for every evaluated recipe, each with a `.recipe.json` descriptor. `index.json` maps recipe names to fingerprints. |
| `results/recipes/`, `results/freezes/` | Recipe lists and the pre-registered freezes. The lockbox evaluator refused any recipe not named in a freeze. |
| `results/receipts/` | Original decision and report receipts (in three of them, `ancestry/results.json`, `repair/selection.json` and `search/selection.json`, absolute local paths were rewritten as repository-relative paths; nothing else changed), including rounds 2 to 5 of targeted editing. The round 1 pilot receipt contains question text and is omitted; its one changed answer is described in the paper. |
| `code/original/` | The study's original evaluation and data-building code, for transparency. Rescoring needs the private checkpoints. |
| `MANIFEST.sha256.json` | SHA-256 of every file |

## Checkpoints

The model checkpoints are **not publicly released**. **They are available to reviewers on request.** Their SHA-256 hashes are listed in [CHECKPOINTS.md](CHECKPOINTS.md), so anyone granted access can confirm they hold exactly the evaluated models. Every edited recipe is defined in `results/recipes/` as an affine combination of these anchors, so the edited weights can be rebuilt from them.

## Licenses

The code is under the MIT License (`LICENSE`). Data derived from public datasets keeps the original terms; see [DATA_LICENSES.md](DATA_LICENSES.md).

## Author

Sean Lester Miranda, ORCID [0009-0000-9782-4206](https://orcid.org/0009-0000-9782-4206), contact@seanmiranda.com

## Citation

```bibtex
@article{weightspace2026deploymentgap,
  title  = {Weight-Space Editing Wins Benchmarks but Not Deployment: A Pre-Registered Study on a Small Decision Model},
  author = {Miranda, Sean Lester},
  year   = {2026},
  note   = {arXiv identifier to be added}
}
```
