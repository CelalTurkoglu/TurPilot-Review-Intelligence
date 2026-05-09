# TurPilot-Review-Intelligence

TurPilot-Review-Intelligence, turizm acentelerine ait Google yorumlarını beş hizmet boyutunda analiz eden bir NLP ve çoklu çıktı sınıflandırma projesidir. Sistem her yorum için `Ulasim`, `Rehber`, `Organizasyon`, `Otel` ve `Yemek` kategorilerinde 0/1/2 etiketi üretir ve Streamlit arayüzünde otomatik kurumsal yanıt taslağı oluşturur.

Bu sürüm MVP mantığından çıkarılıp daha dürüst ve production'a yakın bir ML akışına taşındı. Önceki prototipte model, etiketleri üretmekte kullanılan `mentioned`, `pos_score`, `neg_score`, `star_pos`, `star_neg` gibi kural tabanlı sinyalleri feature olarak tekrar görüyordu. Bu hedefe çok yakın sinyaller test başarısını yapay biçimde şişiriyordu. Güncel pipeline bu leakage kaynağını kaldırır ve modeli yalnızca yorum metni ile yıldız puanı üzerinden değerlendirir.

## Güncel Durum

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

## Klasör Yapısı

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

`web_scraping/dataset.csv` ham merge çıktısıdır ve `.gitignore` içinde tutulur. Curated model verisi `model/updated_dataset.csv` dosyasındadır.
Ham merge içinde yalnızca `·` karakterinden oluşan 1 yorum bulunduğu için model verisinden çıkarıldı; bu yüzden ham veri 1088, eğitim verisi 1087 satırdır.

## Veri Toplama

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

## Etiketleme ve Temizlik

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

## Leakage-Free ML Pipeline

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

## Güncel Metrikler

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

## Uygulamayı Çalıştırma

```bash
cd model
pip install -r requirements.txt
streamlit run app.py
```

Arayüz varsayılan olarak şu adreste açılır:

```text
http://localhost:8501
```

## Ortam Değişkenleri

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

## Akademik Not

Bu sürümde asıl kazanım sadece metrik artışı değildir; ölçümün dürüstleşmesidir. Proje artık veri sızıntısını açıkça teşhis eder, sızıntılı feature'ları kaldırır, veri setini gerçek Google yorumlarıyla büyütür, duplicate temizliği yapar ve performansı tek bir şanslı train/test split yerine 5-Fold Cross Validation ile raporlar.
