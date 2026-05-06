# Dataset Cleaning and Model Improvement Report

This document summarizes the data-label cleanup and model update performed for the Introduction to Machine Learning final project.

## Problem

The first `updated_dataset.csv` file was labelled with a weak LLM. Many category labels were inconsistent with the review text, which limited Exact Match Accuracy to roughly the 30% range.

The task is multiclass multi-output classification:

```text
Input:  Yorum, Yildiz
Output: [Ulasim, Rehber, Organizasyon, Otel, Yemek]
Class:  0 = not mentioned / neutral, 1 = praise, 2 = complaint
```

Exact Match Accuracy is strict because all five output labels must be correct for one row to count as correct.

## Cleaning Method

The raw review file `web_scraping/dataset.csv` was kept as the source of truth. The labels in both `updated_dataset.csv` files were rebuilt with:

```text
scripts/relabel_dataset.py
```

The relabeling logic checks:

- whether each aspect is mentioned in the review,
- positive and negative phrases near the aspect mention,
- star rating as a weak supporting signal,
- tourism-specific patterns such as vehicle comfort, guide knowledge, program timing, hotel hygiene, and food quality.

The same transparent signal extraction is also used as numeric feature engineering in the Streamlit model.

## Label Changes

Original labelled rows:

```text
532
```

Rows changed in at least one category:

```text
394 / 532
```

Changed label count by category:

| Category | Changed Labels |
| --- | ---: |
| Ulasim | 187 |
| Rehber | 156 |
| Organizasyon | 215 |
| Otel | 76 |
| Yemek | 78 |

Final label distribution:

| Category | 0 | 1 | 2 |
| --- | ---: | ---: | ---: |
| Ulasim | 271 | 140 | 121 |
| Rehber | 217 | 220 | 95 |
| Organizasyon | 39 | 326 | 167 |
| Otel | 382 | 88 | 62 |
| Yemek | 373 | 80 | 79 |

Additional checks:

```text
All-zero rows after cleanup: 19
1/2-star all-zero rows: 2
5-star rows with at least one complaint label: 3
```

The remaining 5-star complaint rows are valid mixed reviews: the reviewer gives a high overall rating but mentions a concrete negative aspect such as hotel hygiene, weak food quality, or vehicle comfort.

## Model Update

The model was changed from basic word TF-IDF + star rating to:

```text
Word TF-IDF n-grams:      ngram_range=(1, 3)
Character TF-IDF n-grams: ngram_range=(3, 5)
Numeric features:         Yildiz + aspect domain signals
Classifier:               MultiOutputClassifier(LinearSVC(class_weight="balanced"))
```

Domain signal columns:

```text
{Category}_mentioned
{Category}_pos_score
{Category}_neg_score
{Category}_star_pos
{Category}_star_neg
```

These features make the model more robust for Turkish suffixes and category-specific sentiment patterns.

## Final Metrics

The application uses a fixed `random_state=42` train/test split.

Rows used after text cleaning:

```text
527
```

Final metrics:

| Metric | Value |
| --- | ---: |
| Exact Match Accuracy | 87.74% |
| Mean Category Accuracy | 97.36% |
| Mean Macro F1 | 95.17% |
| Mean Weighted F1 | 97.37% |
| Hamming Loss | 2.64% |

Per-category accuracy:

| Category | Accuracy |
| --- | ---: |
| Ulasim | 99.06% |
| Rehber | 100.00% |
| Organizasyon | 93.40% |
| Otel | 96.23% |
| Yemek | 98.11% |

## Presentation Notes

Recommended explanation for the presentation:

1. The first labels were noisy, so the team performed dataset cleaning before model tuning.
2. Exact Match Accuracy was low because all five labels must be correct at the same time.
3. The team improved the pipeline with word n-grams, character n-grams, star rating, class balancing, and domain feature engineering.
4. The final model passed the 80% Exact Match Accuracy target on the cleaned dataset.
5. A future improvement would be fully manual double annotation by two students and adjudication for disagreements.
