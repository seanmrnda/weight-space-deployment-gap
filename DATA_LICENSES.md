# Data and licenses

This repository contains no text from third-party datasets. For the study panels, the legacy set and the external benchmark, it releases only question identifiers, bootstrap components, gold indices and label strings, together with the model's saved logits. The synthetic benchmark was written for this study and is released in full under the repository's MIT License.

To rebuild the evaluation data, obtain each source yourself under its own terms:

| Dataset | Used for | Source | License, as distributed |
| --- | --- | --- | --- |
| Re-DocRED | Study panels (relations) | `github.com/tonytan48/Re-DocRED` | MIT |
| MAVEN-ERE | Study panels (event relations) | `github.com/THU-KEG/MAVEN-ERE` | GPL-3.0 |
| FEVEROUS | Study panels (claim verification) | `fever.ai` | See the dataset's terms |
| DWIE | External benchmark | `github.com/klimzaporojets/DWIE` | GPL-3.0; article text from Deutsche Welle, fetched by the user |
| MATRES, TempEval-3 Platinum | External benchmark | `github.com/qiangning/MATRES`, `github.com/qiangning/StructTempRel-EMNLP17` | See the dataset's terms |
| AVeriTeC | External benchmark | `github.com/MichSchli/AVeriTeC` | CC BY-NC 4.0 (per the FEVER site; the repository has no license file) |
| LocalLLaMA/typed-decisions | Legacy workflow guards | Hugging Face | See the dataset's terms |
| DocRED `rel_info.json` | Relation names | DocRED / Re-DocRED data release | See the dataset's terms |
