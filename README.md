# TurPilot-Review-Intelligence

## Türkçe

TurPilot-Review-Intelligence, turizm acentelerine ait Google yorumlarını beş hizmet boyutunda analiz eden bir NLP ve çoklu çıktı sınıflandırma projesidir. Sistem her yorum için `Ulasim`, `Rehber`, `Organizasyon`, `Otel` ve `Yemek` kategorilerinde 0/1/2 etiketi üretir ve Streamlit arayüzünde otomatik kurumsal yanıt taslağı oluşturur.

Bu sürüm MVP mantığından çıkarılıp daha dürüst ve production'a yakın bir ML akışına taşındı. Önceki prototipte model, etiketleri üretmekte kullanılan `mentioned`, `pos_score`, `neg_score`, `star_pos`, `star_neg` gibi kural tabanlı sinyalleri feature olarak tekrar görüyordu. Bu hedefe çok yakın sinyaller test başarısını yapay biçimde şişiriyordu. Güncel pipeline bu leakage kaynağını kaldırır ve modeli yalnızca yorum metni ile yıldız puanı üzerinden değerlendirir.

### Ekip

| Ad Soyad | Öğrenci No |
| --- | --- |
| Celal Türkoğlu | 2210206019 |
| Brusk Ferhat Esmer | 2210206307 |
| MUKHAMMAD IDRISOV | 2210206568 |
| Abdulaziz Shamsiev | 2210206543 |

### Güncel Durum

| Alan | Değer |
| --- | ---: |
| Eski ham yorum sayısı | 532 |
| Yeni scrape edilen tekil yorum | 598 |
| Merge + dedupe sonrası ham yorum | 1088 |
| Model eğitiminde kullanılan yorum | 1087 |
| Hedef kategori sayısı | 5 |
| Hedef sınıflar | 0, 1, 2 |
| Değerlendirme | 5-Fold Cross Validation + 80/20 hold-out |

Etiket anlamları:

| Değer | Anlam |
| --- | --- |
| `0` | Bahsedilmemiş veya nötr |
| `1` | Övgü / olumlu geri bildirim |
| `2` | Şikayet / olumsuz geri bildirim |

### Klasör Yapısı

```text
ML_Turpilot_project/
├── README.md
├── docs/
│   ├── DATASET_CLEANING_REPORT.md
│   ├── eng_report.md
│   └── turkish_report.md
├── model/
│   ├── app.py
│   ├── metrics.json
│   ├── training.py
│   ├── requirements.txt
│   └── updated_dataset.csv
├── scripts/
│   ├── evaluate_model.py
│   └── relabel_dataset.py
└── web_scraping/
    ├── google_reviews_scraper.py
    ├── linkler.txt
    ├── new_reviews.csv
    ├── requirements.txt
    └── updated_dataset.csv
```

`web_scraping/dataset.csv` ham merge çıktısıdır ve `.gitignore` içinde tutulur. Curated model verisi `model/updated_dataset.csv` dosyasındadır. Ham merge içinde yalnızca `·` karakterinden oluşan 1 yorum bulunduğu için model verisinden çıkarıldı; bu yüzden ham veri 1088, eğitim verisi 1087 satırdır.

### Veri Toplama

Yeni Google yorumları şu komutla çekildi:

```bash
python3 -u web_scraping/google_reviews_scraper.py \
  --links-file web_scraping/linkler.txt \
  --target-per-star 10 \
  --max-scrolls 8 \
  --page-load-timeout 30 \
  --delay-scale 0.15 \
  --scraped-output web_scraping/new_reviews.csv \
  --output web_scraping/dataset.csv
```

Scraper artık:

- `web_scraping/linkler.txt` içindeki linkleri okur.
- Yeni scrape sonucunu `web_scraping/new_reviews.csv` olarak ayrıca saklar.
- Eski ham veriyle güvenli merge yapar.
- Normalize edilmiş yorum metni üzerinden duplicate yorumları siler.
- Google panelinde erişilemeyen yıldız sınıflarını uydurmaz; mevcut gerçek dağılımla devam eder.

Son scrape özeti:

| Dosya | Satır | Yıldız Dağılımı |
| --- | ---: | --- |
| `web_scraping/new_reviews.csv` | 598 | `{1: 185, 2: 82, 3: 80, 4: 81, 5: 170}` |
| `web_scraping/dataset.csv` | 1088 | `{1: 288, 2: 164, 3: 169, 4: 187, 5: 280}` |

### Etiketleme ve Temizlik

Ham veri şu komutla yeniden etiketlendi:

```bash
python3 scripts/relabel_dataset.py
```

Script, yorum metnini ve yıldız puanını kullanarak beş kategori için kural destekli ABSA etiketi üretir. Aynı zamanda duplicate yorumları, boş/anlamsız yorumları temizler ve iki model kopyasını eşitler:

```text
model/updated_dataset.csv
web_scraping/updated_dataset.csv
```

Son eklenen scrape batch'i ayrıca satır satır denetlendi. Kural tabanlı etiketlerin yanılabileceği negasyon, karşılaştırma ve karma duygu örnekleri için `scripts/relabel_dataset.py` içinde 152 manuel audit override'ı tutulur; böylece düzeltmeler CSV'ye elle işlenmiş geçici değişiklikler değil, yeniden üretilebilir temizlik kuralıdır.

Curated model verisi:

| Dosya | Satır | Yıldız Dağılımı |
| --- | ---: | --- |
| `model/updated_dataset.csv` | 1087 | `{1: 288, 2: 163, 3: 169, 4: 187, 5: 280}` |
| `web_scraping/updated_dataset.csv` | 1087 | `{1: 288, 2: 163, 3: 169, 4: 187, 5: 280}` |

Güncel label dağılımı:

| Kategori | 0 | 1 | 2 |
| --- | ---: | ---: | ---: |
| Ulasim | 568 | 255 | 264 |
| Rehber | 430 | 464 | 193 |
| Organizasyon | 79 | 518 | 490 |
| Otel | 812 | 130 | 145 |
| Yemek | 843 | 99 | 145 |

### Leakage-Free ML Pipeline

Model kodu `model/training.py` içine ayrıldı. Streamlit arayüzü de aynı training modülünü kullanır.

Kullanılan feature set'i:

```text
CleanYorum
Yildiz
```

Kasıtlı olarak kaldırılan leakage feature'ları:

```text
{Kategori}_mentioned
{Kategori}_pos_score
{Kategori}_neg_score
{Kategori}_star_pos
{Kategori}_star_neg
```

Model mimarisi:

- Word TF-IDF, `ngram_range=(1, 3)`
- Character TF-IDF, `ngram_range=(3, 5)`
- Numeric yıldız puanı
- `MultiOutputClassifier`
- `LinearSVC(class_weight="balanced")`
- 5-Fold Cross Validation

Değerlendirme komutu:

```bash
python3 scripts/evaluate_model.py --folds 5 --output model/metrics.json
```

### Güncel Metrikler

5-Fold Cross Validation:

| Metrik | Değer |
| --- | ---: |
| Strict Exact Match Accuracy | 43.78% ± 3.32% |
| Mean Category Accuracy | 83.68% ± 0.88% |
| Mean Macro F1 | 72.76% |
| Mean Weighted F1 | 82.80% |
| Hamming Loss | 16.32% |

5-Fold ortalama kategori accuracy:

| Kategori | Accuracy |
| --- | ---: |
| Ulasim | 80.13% |
| Rehber | 79.94% |
| Organizasyon | 84.18% |
| Otel | 86.93% |
| Yemek | 87.21% |

80/20 hold-out sonucu:

| Metrik | Değer |
| --- | ---: |
| Strict Exact Match Accuracy | 48.17% |
| Mean Category Accuracy | 83.85% |
| Mean Macro F1 | 73.02% |
| Mean Weighted F1 | 82.88% |
| Hamming Loss | 16.15% |

Bu sonuçlar eski %100'e yakın test başarılarından daha düşüktür, fakat akademik olarak çok daha değerlidir. Eski başarı kısmen hedef etiketleri yeniden hesaplayan kural sinyallerinin modele verilmesinden kaynaklanan data leakage etkisiydi. Güncel model testte kopya çekmeden, yalnızca yorum metni ve yıldız puanıyla 80-88 bandında kategori bazlı başarı üretmektedir.

Strict exact-match metriği özellikle serttir: bir satırın doğru sayılması için beş kategorinin tamamı aynı anda doğru tahmin edilmelidir. Bu yüzden production izlemede mean category accuracy, macro/weighted F1 ve Hamming loss birlikte raporlanır.

### Uygulamayı Çalıştırma

```bash
cd model
pip install -r requirements.txt
streamlit run app.py
```

Arayüz varsayılan olarak şu adreste açılır:

```text
http://localhost:8501
```

### Ortam Değişkenleri

API anahtarları repo içinde tutulmaz. Gerekirse `.env.example` dosyasından lokal `.env` oluşturulur:

```bash
cp .env.example .env
```

```env
OPENROUTER_API_KEY=
OPENROUTER_MODEL=openai/gpt-4o-mini
OPENROUTER_API_URL=https://openrouter.ai/api/v1/chat/completions
STREAMLIT_SERVER_PORT=8501
SELENIUM_USE_UNDETECTED_CHROME=false
```

### Akademik Not

Bu sürümde asıl kazanım sadece metrik artışı değildir; ölçümün dürüstleşmesidir. Proje artık veri sızıntısını açıkça teşhis eder, sızıntılı feature'ları kaldırır, veri setini gerçek Google yorumlarıyla büyütür, duplicate temizliği yapar ve performansı tek bir şanslı train/test split yerine 5-Fold Cross Validation ile raporlar.

## English

TurPilot-Review-Intelligence is an NLP and multi-output classification project for analyzing Google reviews of tourism agencies across five service dimensions. For each review, the system predicts a 0/1/2 label for `Ulasim`, `Rehber`, `Organizasyon`, `Otel`, and `Yemek`, then generates a corporate response draft in the Streamlit interface.

This version moves the project beyond the MVP stage toward a more honest and production-like ML workflow. The earlier prototype fed rule-derived signals such as `mentioned`, `pos_score`, `neg_score`, `star_pos`, and `star_neg` back into the model as features. Those signals were too close to the target labels and inflated evaluation results. The current pipeline removes that leakage source and evaluates the model using only review text and star rating.

### Team

| Name | Student ID |
| --- | --- |
| Celal Türkoğlu | 2210206019 |
| Brusk Ferhat Esmer | 2210206307 |
| MUKHAMMAD IDRISOV | 2210206568 |
| Abdulaziz Shamsiev | 2210206543 |

### Current Status

| Field | Value |
| --- | ---: |
| Previous raw review count | 532 |
| Newly scraped unique reviews | 598 |
| Raw reviews after merge + dedupe | 1088 |
| Reviews used for model training | 1087 |
| Target category count | 5 |
| Target classes | 0, 1, 2 |
| Evaluation | 5-Fold Cross Validation + 80/20 hold-out |

Label meanings:

| Value | Meaning |
| --- | --- |
| `0` | Not mentioned or neutral |
| `1` | Praise / positive feedback |
| `2` | Complaint / negative feedback |

### Project Structure

```text
ML_Turpilot_project/
├── README.md
├── docs/
│   ├── DATASET_CLEANING_REPORT.md
│   ├── eng_report.md
│   └── turkish_report.md
├── model/
│   ├── app.py
│   ├── metrics.json
│   ├── training.py
│   ├── requirements.txt
│   └── updated_dataset.csv
├── scripts/
│   ├── evaluate_model.py
│   └── relabel_dataset.py
└── web_scraping/
    ├── google_reviews_scraper.py
    ├── linkler.txt
    ├── new_reviews.csv
    ├── requirements.txt
    └── updated_dataset.csv
```

`web_scraping/dataset.csv` is the raw merge output and is kept in `.gitignore`. The curated model dataset is `model/updated_dataset.csv`. One raw merged row contained only the `·` character, so it was excluded from the model dataset; this is why the raw dataset has 1088 rows while the training dataset has 1087 rows.

### Data Collection

New Google reviews were scraped with:

```bash
python3 -u web_scraping/google_reviews_scraper.py \
  --links-file web_scraping/linkler.txt \
  --target-per-star 10 \
  --max-scrolls 8 \
  --page-load-timeout 30 \
  --delay-scale 0.15 \
  --scraped-output web_scraping/new_reviews.csv \
  --output web_scraping/dataset.csv
```

The scraper now:

- reads agency links from `web_scraping/linkler.txt`,
- saves the current scraping run separately as `web_scraping/new_reviews.csv`,
- safely merges the new reviews with the previous raw data,
- removes duplicate reviews using normalized review text,
- keeps the real available star distribution instead of fabricating unavailable rating groups.

Latest scrape summary:

| File | Rows | Star Distribution |
| --- | ---: | --- |
| `web_scraping/new_reviews.csv` | 598 | `{1: 185, 2: 82, 3: 80, 4: 81, 5: 170}` |
| `web_scraping/dataset.csv` | 1088 | `{1: 288, 2: 164, 3: 169, 4: 187, 5: 280}` |

### Labeling and Cleaning

The raw data was relabeled with:

```bash
python3 scripts/relabel_dataset.py
```

The script produces rule-assisted ABSA labels from review text and star rating. It also removes duplicate or non-informative comments and keeps the two model dataset copies synchronized:

```text
model/updated_dataset.csv
web_scraping/updated_dataset.csv
```

The newly scraped batch was also manually audited row by row. For negation, contrast, and mixed-sentiment cases where keyword rules are brittle, `scripts/relabel_dataset.py` stores 152 manual audit overrides. This makes the corrections reproducible instead of one-off CSV edits.

Curated model data:

| File | Rows | Star Distribution |
| --- | ---: | --- |
| `model/updated_dataset.csv` | 1087 | `{1: 288, 2: 163, 3: 169, 4: 187, 5: 280}` |
| `web_scraping/updated_dataset.csv` | 1087 | `{1: 288, 2: 163, 3: 169, 4: 187, 5: 280}` |

Current label distribution:

| Category | 0 | 1 | 2 |
| --- | ---: | ---: | ---: |
| Ulasim | 568 | 255 | 264 |
| Rehber | 430 | 464 | 193 |
| Organizasyon | 79 | 518 | 490 |
| Otel | 812 | 130 | 145 |
| Yemek | 843 | 99 | 145 |

### Leakage-Free ML Pipeline

The model code was separated into `model/training.py`. The Streamlit app uses the same training module.

Feature set:

```text
CleanYorum
Yildiz
```

Intentionally removed leakage features:

```text
{Category}_mentioned
{Category}_pos_score
{Category}_neg_score
{Category}_star_pos
{Category}_star_neg
```

Model architecture:

- Word TF-IDF, `ngram_range=(1, 3)`
- Character TF-IDF, `ngram_range=(3, 5)`
- Numeric star rating
- `MultiOutputClassifier`
- `LinearSVC(class_weight="balanced")`
- 5-Fold Cross Validation

Evaluation command:

```bash
python3 scripts/evaluate_model.py --folds 5 --output model/metrics.json
```

### Current Metrics

5-Fold Cross Validation:

| Metric | Value |
| --- | ---: |
| Strict Exact Match Accuracy | 43.78% ± 3.32% |
| Mean Category Accuracy | 83.68% ± 0.88% |
| Mean Macro F1 | 72.76% |
| Mean Weighted F1 | 82.80% |
| Hamming Loss | 16.32% |

Per-category 5-Fold mean accuracy:

| Category | Accuracy |
| --- | ---: |
| Ulasim | 80.13% |
| Rehber | 79.94% |
| Organizasyon | 84.18% |
| Otel | 86.93% |
| Yemek | 87.21% |

80/20 hold-out result:

| Metric | Value |
| --- | ---: |
| Strict Exact Match Accuracy | 48.17% |
| Mean Category Accuracy | 83.85% |
| Mean Macro F1 | 73.02% |
| Mean Weighted F1 | 82.88% |
| Hamming Loss | 16.15% |

These scores are lower than the previous near-perfect test results, but they are academically more valuable. The previous results were partly inflated by data leakage from rule-derived target proxy features. The current model evaluates without that shortcut, using only review text and star rating.

Strict exact-match is intentionally harsh: a row is counted as correct only when all five category labels are predicted correctly at the same time. For product monitoring, mean category accuracy, macro/weighted F1, and Hamming loss should be interpreted together.

### Running the App

```bash
cd model
pip install -r requirements.txt
streamlit run app.py
```

The interface opens by default at:

```text
http://localhost:8501
```

### Environment Variables

API keys are not stored in the repository. If needed, create a local `.env` file from `.env.example`:

```bash
cp .env.example .env
```

```env
OPENROUTER_API_KEY=
OPENROUTER_MODEL=openai/gpt-4o-mini
OPENROUTER_API_URL=https://openrouter.ai/api/v1/chat/completions
STREAMLIT_SERVER_PORT=8501
SELENIUM_USE_UNDETECTED_CHROME=false
```

### Academic Note

The main improvement in this version is not just the metric change; it is the honesty of the evaluation. The project now explicitly identifies data leakage, removes leakage-prone features, expands the dataset with real Google reviews, cleans duplicates, and reports performance with 5-Fold Cross Validation instead of relying on a single favorable train/test split.
