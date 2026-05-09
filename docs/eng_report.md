# TurPilot Review Intelligence: A Leakage-Free Multi-Output ML System for Tourism Reviews

## 1. Abstract

This project classifies Google reviews of tourism agencies across five service aspects: Transportation, Guide, Organization, Hotel, and Food. For each aspect, the model predicts one of three labels:

| Value | Meaning |
| --- | --- |
| 0 | Not mentioned or neutral |
| 1 | Praise |
| 2 | Complaint |

The main contribution of this version is not a cosmetic metric increase, but an academically honest evaluation. The MVP model previously reported near-100% test accuracy for some categories. After inspecting the feature pipeline, this was identified as data leakage. Rule-derived aspect signals used during relabeling were also fed into the supervised model as features, allowing the model to partially reconstruct the target labels.

The current version removes those target-proxy features. The supervised model now uses only cleaned review text (`CleanYorum`) and Google star rating (`Yildiz`).

## 2. Dataset

The initial dataset contained 532 raw Google reviews. New reviews were scraped from the Google links stored in `web_scraping/linkler.txt`. The scraping run produced 598 unique new reviews. After merging with the old raw dataset and removing duplicate review texts, the raw dataset reached 1088 unique rows.

| Stage | Rows |
| --- | ---: |
| Previous raw dataset | 532 |
| New unique scraped reviews | 598 |
| Raw dataset after merge and dedupe | 1088 |
| Rows used for model training | 1087 |

Raw star distribution:

| Star | Rows |
| --- | ---: |
| 1 | 288 |
| 2 | 164 |
| 3 | 169 |
| 4 | 187 |
| 5 | 280 |

One row was removed during model preparation because its cleaned text became empty.

## 3. Labeling and Cleaning

Labels were rebuilt with `scripts/relabel_dataset.py`. The script checks aspect vocabularies, positive and negative phrases near the aspect mention, and the star rating as a weak supporting signal. Star rating alone is not enough to label an aspect; the aspect must be mentioned or inferred through a domain-specific pattern.

Final label distribution:

| Category | 0 | 1 | 2 |
| --- | ---: | ---: | ---: |
| Ulasim | 583 | 294 | 211 |
| Rehber | 440 | 459 | 189 |
| Organizasyon | 93 | 658 | 337 |
| Otel | 822 | 148 | 118 |
| Yemek | 807 | 136 | 145 |

This distribution is plausible for tourism reviews. Organization is the most common aspect because many reviews describe the overall tour or agency process. Hotel and food are more sparse because they are only relevant when explicitly mentioned or included in the tour package.

## 4. Data Leakage Analysis

The old model used the following engineered columns:

```text
{Category}_mentioned
{Category}_pos_score
{Category}_neg_score
{Category}_star_pos
{Category}_star_neg
```

These columns were intermediate outputs of the same rule-based logic used for label generation. Therefore, the model was not purely learning from review text. It had access to features that were too close to the answer key, which explains the artificially high test scores.

The current supervised feature set is restricted to:

```text
CleanYorum
Yildiz
```

This makes the task harder, but the results are now valid.

## 5. Model Architecture

The problem is multiclass multi-output classification. Each review has five target columns, and each target can take one of three values: 0, 1, or 2.

Pipeline:

```text
CleanYorum -> Word TF-IDF, ngram_range=(1, 3)
CleanYorum -> Character TF-IDF, ngram_range=(3, 5)
Yildiz     -> StandardScaler
Classifier -> MultiOutputClassifier(LinearSVC(class_weight="balanced"))
```

Word n-grams capture expressions such as `very good`, `hotel was not clean`, and `the guide was helpful`. Character n-grams are useful for Turkish because suffixes create many surface forms of the same root. `class_weight="balanced"` is used to reduce majority-class dominance.

## 6. Evaluation

The model is evaluated with 5-Fold Cross Validation rather than relying only on a single train/test split. A reproducible 80/20 hold-out result is also generated for the Streamlit interface.

5-Fold Cross Validation:

| Metric | Value |
| --- | ---: |
| Strict Exact Match Accuracy | 41.12% ± 2.03% |
| Mean Category Accuracy | 81.67% ± 1.15% |
| Mean Macro F1 | 71.61% |
| Mean Weighted F1 | 80.88% |
| Hamming Loss | 18.33% |

Per-category 5-Fold accuracy:

| Category | Accuracy |
| --- | ---: |
| Ulasim | 77.74% |
| Rehber | 79.66% |
| Organizasyon | 77.73% |
| Otel | 87.30% |
| Yemek | 85.92% |

80/20 hold-out:

| Metric | Value |
| --- | ---: |
| Strict Exact Match Accuracy | 44.50% |
| Mean Category Accuracy | 82.11% |
| Mean Macro F1 | 71.28% |
| Mean Weighted F1 | 81.18% |
| Hamming Loss | 17.89% |

Strict exact-match is intentionally harsh: a row is counted as correct only if all five aspect labels are correct simultaneously. Therefore, product-level monitoring should consider mean category accuracy, macro F1, weighted F1, and Hamming loss together.

## 7. Conclusion

This version moves the project from a prototype toward a production-grade ML workflow. The previous near-perfect scores were attractive but misleading. After removing leakage and adding K-Fold Cross Validation, the model produces a realistic 80-88% range of per-category accuracy without seeing target-proxy features.

The dataset was expanded from 532 to 1088 raw unique rows, duplicate reviews were removed, labels were rebuilt, and the model pipeline was refactored into reusable training and evaluation modules. The result is lower than the leaked score, but much stronger as scientific evidence.

## 8. Future Work

1. Manually audit a stratified sample of labels.
2. Use two independent annotators and report Cohen's Kappa.
3. Compare LinearSVC with BERTurk or multilingual transformer baselines.
4. Persist the trained model as a `joblib` or `.pkl` artifact.
5. Monitor production drift by city, agency, star rating, and aspect.
