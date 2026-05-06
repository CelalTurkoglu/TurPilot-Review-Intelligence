# Turizm Acente Yorum Analizi ve Otonom Yanıt Sistemi

Bu proje, Introduction to Machine Learning dersi dönem projesi için hazırlanmış bir NLP ve çoklu çıktı sınıflandırma uygulamasıdır. Turizm acentelerine ait müşteri yorumları; ulaşım, rehber, organizasyon, otel ve yemek boyutlarında analiz edilir ve Streamlit arayüzü üzerinden otomatik kurumsal yanıt üretilir.

## Proje Özeti

Sistem müşteri yorumlarını 5 hizmet boyutunda analiz eder:

| Kategori | Açıklama |
| --- | --- |
| `Ulasim` | Araç, otobüs, transfer, şoför, yolculuk konforu |
| `Rehber` | Tur rehberi, personel ilgisi, bilgi seviyesi, iletişim |
| `Organizasyon` | Program akışı, zaman yönetimi, planlama, tur düzeni |
| `Otel` | Konaklama, oda kalitesi, temizlik, otel konumu |
| `Yemek` | Kahvaltı, akşam yemeği, restoran, menü ve lezzet |

Her kategori için model şu değerlerden birini tahmin eder:

| Değer | Anlam |
| --- | --- |
| `0` | Bahsedilmemiş veya nötr |
| `1` | Övgü / olumlu geri bildirim |
| `2` | Şikayet / olumsuz geri bildirim |

Model sonucu kullanılarak kurumsal bir otomatik yanıt metni üretilir.

## Klasör Yapısı

```text
ML_Turpilot_project/
├── .env.example
├── .gitignore
├── README.md
├── docs/
│   └── DATASET_CLEANING_REPORT.md
├── model/
│   ├── app.py
│   ├── requirements.txt
│   └── updated_dataset.csv
├── scripts/
│   ├── __init__.py
│   └── relabel_dataset.py
└── web_scraping/
    ├── absa_analysis.py
    ├── google_reviews_scraper.py
    ├── requirements.txt
    └── updated_dataset.csv
```

Not: `web_scraping` klasör adı projede mevcut haliyle korunmuştur.

## Ana Bileşenler

### 1. Google Yorum Scraper

Dosya:

```text
web_scraping/google_reviews_scraper.py
```

Bu script Selenium ile Google Search / Google Maps yerel yorum panelinden yorumları ve yıldız puanlarını toplamaya çalışır.

Öne çıkan özellikler:

- 1, 2, 3, 4 ve 5 yıldızlı yorumları ayrı ayrı toplar.
- Google yorum panelini dinamik olarak kaydırır.
- `Daha fazla` / `Diğer` butonlarına basarak uzun yorumları açar.
- Mutlak XPath kullanmaz; daha esnek selector ve DOM tarama yaklaşımı kullanır.
- Çıktıyı CSV olarak kaydeder.

Çalıştırma:

```bash
cd web_scraping
pip install -r requirements.txt
python3 google_reviews_scraper.py
```

Üretilen ham dosya:

```text
web_scraping/dataset.csv
```

Bu dosya `.gitignore` içinde ignore edilmiştir, çünkü scraping çıktısı tekrar üretilebilir ve kişisel/ham veri içerebilir.

### 2. ABSA Etiketleme Scripti

Dosya:

```text
web_scraping/absa_analysis.py
```

Bu script ham `dataset.csv` dosyasını okuyup OpenRouter API üzerinden Aspect-Based Sentiment Analysis etiketi üretir.

Girdi:

```text
Yorum,Yildiz
```

Çıktı:

```text
Yorum,Yildiz,Ulasim,Rehber,Organizasyon,Otel,Yemek
```

Çalıştırma:

```bash
cd web_scraping
python3 absa_analysis.py
```

Önemli güvenlik notu:

API key artık kod içinde tutulmaz. `OPENROUTER_API_KEY` değeri proje kökündeki `.env` dosyasından okunur.

### 3. Veri Etiketi Temizleme Scripti

Dosya:

```text
scripts/relabel_dataset.py
```

İlk etiketli veri setinde zayıf LLM kaynaklı ciddi 0/1/2 hataları bulunduğu için ham `web_scraping/dataset.csv` yeniden işlenmiştir. Bu script yorumları baştan tarar, her kategori için aspect mention ve yakın bağlamdaki olumlu/olumsuz sinyalleri çıkarır, ardından iki `updated_dataset.csv` kopyasını eşitler.

Çalıştırma:

```bash
python3 scripts/relabel_dataset.py
```

Üretilen dosyalar:

```text
model/updated_dataset.csv
web_scraping/updated_dataset.csv
```

### 4. Streamlit ML Uygulaması

Dosya:

```text
model/app.py
```

Bu uygulama `model/updated_dataset.csv` veri seti üzerinden modeli eğitir ve web arayüzü sunar.

Kullanılan ML yaklaşımı:

- `clean_text` fonksiyonu ile Türkçe metin temizleme
- `TfidfVectorizer` ile word n-gram ve character n-gram özellikleri
- `ColumnTransformer` ile metin, yıldız puanı ve domain signal feature'larını birleştirme
- `StandardScaler` ile numeric feature'ları ölçekleme
- `MultiOutputClassifier`
- `LinearSVC(class_weight="balanced")`

Çalıştırma:

```bash
cd model
pip install -r requirements.txt
streamlit run app.py
```

Tarayıcı adresi:

```text
http://localhost:8501
```

## Veri Seti Şeması

Model uygulaması aşağıdaki kolonları bekler:

| Kolon | Tip | Açıklama |
| --- | --- | --- |
| `Yorum` | Metin | Müşteri yorumu |
| `Yildiz` | 1-5 tam sayı | Google yıldız puanı |
| `Ulasim` | 0, 1, 2 | Ulaşım etiketi |
| `Rehber` | 0, 1, 2 | Rehber etiketi |
| `Organizasyon` | 0, 1, 2 | Organizasyon etiketi |
| `Otel` | 0, 1, 2 | Otel etiketi |
| `Yemek` | 0, 1, 2 | Yemek etiketi |

Güncel veri seti:

```text
Ham yorum sayısı: 532
Model eğitiminde kullanılan yorum sayısı: 527
```

Beş satır çok kısa veya sadece sembol içerdiği için temiz metin aşamasında model eğitiminden çıkarılır.

Model uygulaması şu dosya isimlerini sırasıyla arar:

```text
model/dataset.csv
model/updated_dataset.csv
```

Bu repoda varsayılan veri dosyası:

```text
model/updated_dataset.csv
```

## Kurulum

Önerilen kurulum:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r model/requirements.txt
pip install -r web_scraping/requirements.txt
```

Sadece Streamlit uygulamasını çalıştırmak için:

```bash
cd model
pip install -r requirements.txt
streamlit run app.py
```

Sadece scraping tarafını çalıştırmak için:

```bash
cd web_scraping
pip install -r requirements.txt
python3 google_reviews_scraper.py
```

## Ortam Değişkenleri

Repo içinde gerçek `.env` dosyası paylaşılmamalıdır. Ekip arkadaşları şu komutla örnek dosyadan kendi lokal dosyasını oluşturabilir:

```bash
cp .env.example .env
```

Sonra `.env` içindeki değerleri doldururlar:

```env
OPENROUTER_API_KEY=
OPENROUTER_MODEL=openai/gpt-4o-mini
OPENROUTER_API_URL=https://openrouter.ai/api/v1/chat/completions
STREAMLIT_SERVER_PORT=8501
SELENIUM_USE_UNDETECTED_CHROME=false
```

Güvenlik:

- `.env` dosyası `.gitignore` içinde ignore edilir.
- Gerçek API key hiçbir zaman commit edilmemelidir.
- Daha önce local dosyada hardcoded API key bulunduysa, GitHub'a yüklemeden önce ilgili sağlayıcı panelinden key rotate edilmelidir.

## Etiket Temizleme Özeti

İlk `updated_dataset.csv` dosyasında zayıf LLM kaynaklı tutarsız etiketler vardı. Bu yüzden `scripts/relabel_dataset.py` ile tüm yorumlar yeniden etiketlendi.

Özet:

```text
En az bir kategorisi değişen satır: 394 / 532
Ulasim değişen label: 187
Rehber değişen label: 156
Organizasyon değişen label: 215
Otel değişen label: 76
Yemek değişen label: 78
```

Güncel label dağılımı:

| Kategori | 0 | 1 | 2 |
| --- | ---: | ---: | ---: |
| Ulasim | 271 | 140 | 121 |
| Rehber | 217 | 220 | 95 |
| Organizasyon | 39 | 326 | 167 |
| Otel | 382 | 88 | 62 |
| Yemek | 373 | 80 | 79 |

Detaylı rapor:

```text
docs/DATASET_CLEANING_REPORT.md
```

## Model Eğitim Akışı

`model/app.py` içinde model her Streamlit çalışmasında cache ile eğitilir.

Adımlar:

1. CSV dosyası okunur.
2. Zorunlu kolonlar kontrol edilir.
3. `Yildiz` ve hedef kolonlar numeric tipe çevrilir.
4. `Yorum` metni temizlenir ve `CleanYorum` kolonu oluşturulur.
5. Veri train/test olarak ayrılır.
6. Domain feature kolonları oluşturulur:
   - Her kategori için `mentioned`
   - Her kategori için `pos_score`
   - Her kategori için `neg_score`
   - Her kategori için yıldız destekli `star_pos` ve `star_neg`
7. `ColumnTransformer` üç feature bloğu üretir:
   - `CleanYorum` -> word `TfidfVectorizer(ngram_range=(1, 3))`
   - `CleanYorum` -> character `TfidfVectorizer(ngram_range=(3, 5))`
   - `Yildiz` + domain feature'ları -> `StandardScaler`
8. `MultiOutputClassifier(LinearSVC(...))` modeli eğitilir.
9. Sidebar için metrikler hesaplanır:
   - Exact Match Accuracy
   - Kategori bazlı accuracy
   - Macro F1, Weighted F1, Precision, Recall
   - Hamming Loss

Güncel test sonucu:

```text
Exact Match Accuracy: 87.74%
Ortalama Kategori Accuracy: 97.36%
Ortalama Macro F1: 95.17%
Hamming Loss: 2.64%
```

## Otomatik Yanıt Mantığı

Kullanıcı yorum yazıp yıldız puanı seçtikten sonra `Analiz Et` butonuna basar.

Uygulama:

1. Yorumu temizler.
2. Temiz metin ve yıldız puanını modele gönderir.
3. 5 kategori için `0`, `1`, `2` tahminlerini üretir.
4. Tahminleri renkli etiketler olarak gösterir.
5. `generate_auto_reply` fonksiyonuyla kurumsal yanıt üretir.

Örnek:

```text
Otel = 2, Rehber = 1
```

Bu durumda sistem hem otel şikayeti için özür metni hem de rehber övgüsü için teşekkür metni üretir.

## GitHub'a Yüklemeden Önce Kontrol Listesi

GitHub'a manuel yüklemeden önce:

```bash
find . -name "__pycache__" -type d
find . -name ".DS_Store" -type f
```

Bu dosyalar `.gitignore` ile ignore edilir, fakat daha temiz bir repo için lokalden silinebilir.

Git durumunu kontrol etmek için repo başlatıldıktan sonra:

```bash
git init
git status
```

Commit öncesi özellikle şunlara dikkat edin:

- `.env` commit edilmemeli.
- `web_scraping/debug/` commit edilmemeli.
- `web_scraping/dataset.csv` commit edilmemeli.
- `model/updated_dataset.csv` uygulamanın demo veri seti olarak commit edilebilir.
- API key içeren hiçbir dosya commit edilmemeli.

Örnek Git akışı:

```bash
git init
git add .
git status
git commit -m "Initial tourism NLP analysis project"
git branch -M main
git remote add origin <GITHUB_REPO_URL>
git push -u origin main
```

## Sorun Giderme

### Streamlit bulunamıyor

```bash
pip install -r model/requirements.txt
```

### Dataset bulunamadı

`model` klasörü altında şu dosyalardan biri olmalı:

```text
dataset.csv
updated_dataset.csv
```

### API key hatası

`.env` içinde `OPENROUTER_API_KEY` dolu olmalı:

```env
OPENROUTER_API_KEY=your_key_here
```

### Selenium ChromeDriver hatası

Chrome sürümünü güncelleyin veya standart Selenium modu ile çalıştırın. Projede varsayılan ayar standart Selenium kullanımına yakındır.

## Geliştirme Notları

Gelecekte eklenebilecek geliştirmeler:

- Modeli `.pkl` olarak kaydedip her açılışta yeniden eğitmemek
- Daha büyük ve dengeli etiketli veri seti oluşturmak
- BERTurk veya multilingual transformer tabanlı model denemek
- Kullanıcı yanıtlarını veritabanına kaydetmek
- Dashboard'a kategori bazlı şikayet trendleri eklemek
- Yanıt üretimini kurallardan LLM destekli güvenli şablonlara taşımak

# TurPilot-Review-Intelligence
