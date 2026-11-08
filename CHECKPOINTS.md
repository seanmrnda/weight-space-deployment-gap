# Checkpoints

The checkpoints are not publicly released. They are available to reviewers on request. The hashes below identify the exact files that were evaluated, so a recipient can verify a copy.

## Anchors

| Name | Role | `model.safetensors` SHA-256 |
| --- | --- | --- |
| L | Deployed checkpoint, the baseline for every comparison | `2a70d51d4b74dffaf1dff63c175019ac07f7fc131e21c912dd3f189968311dba` |
| P | Earlier accepted reference model, the parent of L | `ebc1b9b73a4e70469f7678d95320e577e1db572df0e016fd8c43e1cc10063fc8` |
| T | Targeted continuation of L, window 2,500 snapshot | `eaa3968255efd07b927e590575882b6523b9ef4f3a59fa161cdab47dccfd832e` |
| H | Sibling model | `1a9d8adfde970930a4357a12948a30fde0e6fe2f51f34f017fd81311e0a04903` |

Snapshot SHA-256 prefixes of T's run, used in the 12-snapshot average:

| Window | Prefix | Window | Prefix |
| ---: | --- | ---: | --- |
| 250 | `73f91f8e312d00b7` | 1,750 | `aaf4cc7373d04ebb` |
| 500 | `18c1afdcd25c3595` | 2,000 | `9ad33d69a00c10dc` |
| 750 | `bdeddc03e7530ffd` | 2,250 | `f1a0a434cd5000b7` |
| 1,000 | `cab260c588a58259` | 2,500 | `eaa3968255efd07b` |
| 1,250 | `a07edc2e1ed13994` | 2,750 | `093fe45e0cae4313` |
| 1,500 | `0cf7e2a55c31dc1c` | 3,000 | `7bc5f0357af974cd` |

## Exported edits

| Recipe | Definition | `model.safetensors` SHA-256 |
| --- | --- | --- |
| A5 (round 5) | L + 0.5(P − L) | `c76875d86d888e582bfe8465d32755908b755426f531c0c3007e14c9b8fd1ccc` |
| P+T−L (round 6) | P + (T − L) | `f00dd98a5193c24524cc509cc557804afebff5e84fbb54a5ddcfa35edad68143` |

Every other recipe is computed in memory from the anchors, following `results/recipes/`. The temperature buffer stays at L's value, and the unused action head is excluded. The recipe fingerprints in `results/logits/index.json` bind each set of saved logits to its definition.

## Pre-registration records

| Record | SHA-256 |
| --- | --- |
| Task-arithmetic freeze (`results/freezes/freeze.json`) | `33bc528d8ac3649aca8a09af2651d099f21f2105356c6be651df95d441ec9669` |
| Interpolation freeze (`results/freezes/freeze-LT.json`) | `aa9367f914df18d695025f7021f13ff6b6eecd27dd4fba4c6a22ee4314c4e48e` |
| Study panel data (private text; plan in `data/plans/soup.json`) | `27e7db82f2f95bd63b763b002c109b13d1c417e36b510074eb8e4af76c3b4172` |
| External benchmark data | `903c4c12986712274f09b93a931b4c0c4a9f8b098b562b2a71a0f55fb47679d9` |
| Synthetic benchmark data (`data/synthetic/data.jsonl`) | `78ef0f47909c6b7ddba08ad42bafc394ce5818c353b1556c9ea8016676f0c99d` |
