# TurPilot Review Intelligence: Turizm Yorumları İçin Leakage-Free Çoklu Çıktı ML Sistemi

## 1. Özet

Bu proje, turizm acentelerine ait Google yorumlarını beş hizmet boyutunda sınıflandıran bir doğal dil işleme çalışmasıdır: Ulaşım, Rehber, Organizasyon, Otel ve Yemek. Her kategori için model üç sınıftan birini tahmin eder:

| Değer | Anlam |
| --- | --- |
| 0 | Bahsedilmemiş veya nötr |
| 1 | Övgü |
| 2 | Şikayet |

Bu sürümde temel amaç yalnızca daha yüksek skor almak değil, skorları akademik olarak savunulabilir hale getirmektir. Önceki prototipte bazı kategorilerde %100'e yakın test başarısı görülmüştü. İnceleme sonucunda bunun gerçek genelleme başarısı değil, data leakage etkisi olduğu belirlendi. Etiket üretiminde kullanılan kural sinyallerinin modele feature olarak verilmesi, modelin test setinde kopya çekmesine neden oluyordu.

Güncel sistemde bu sızıntı kapatıldı. Model artık yalnızca temizlenmiş yorum metni (`CleanYorum`) ve yıldız puanı (`Yildiz`) ile eğitilir.

## 2. Veri Seti

Başlangıçta veri setinde 532 ham yorum vardı. `web_scraping/linkler.txt` içindeki yeni Google yorum linkleriyle scraping yapıldı ve 598 yeni tekil yorum elde edildi. Eski veriyle merge ve duplicate temizliği sonrasında ham veri 1088 tekil yoruma ulaştı.

| Aşama | Satır |
| --- | ---: |
| Eski ham veri | 532 |
| Yeni scrape edilen tekil veri | 598 |
| Merge + dedupe sonrası ham veri | 1088 |
| Model eğitiminde kullanılan veri | 1087 |

Ham yıldız dağılımı:

| Yıldız | Satır |
| --- | ---: |
| 1 | 288 |
| 2 | 164 |
| 3 | 169 |
| 4 | 187 |
| 5 | 280 |

Model eğitiminde bir yorum, temiz metin aşamasından sonra boş kaldığı için çıkarıldı.

## 3. Etiketleme ve Temizleme

Etiketler `scripts/relabel_dataset.py` ile yeniden üretildi. Script her yorumda kategori kelimelerini, kategoriye yakın olumlu/olumsuz ifadeleri ve yıldız puanını birlikte değerlendirir. Yıldız puanı tek başına etiket belirlemek için kullanılmaz; yalnızca ilgili kategori gerçekten anılmışsa destekleyici sinyal olarak kullanılır.

Güncel label dağılımı:

| Kategori | 0 | 1 | 2 |
| --- | ---: | ---: | ---: |
| Ulasim | 583 | 294 | 211 |
| Rehber | 440 | 459 | 189 |
| Organizasyon | 93 | 658 | 337 |
| Otel | 822 | 148 | 118 |
| Yemek | 807 | 136 | 145 |

Bu dağılım turizm yorumlarının doğasına uygundur: organizasyon genel yorumlarda en sık geçen boyuttur; otel ve yemek ise yalnızca tur paketinde veya yorumda açıkça bahsedildiğinde etiketlenir.

## 4. Data Leakage Analizi

Önceki modelde şu feature'lar kullanılıyordu:

```text
{Kategori}_mentioned
{Kategori}_pos_score
{Kategori}_neg_score
{Kategori}_star_pos
{Kategori}_star_neg
```

Bu feature'lar, etiketleri üreten kural tabanlı mantığın ara çıktılarıydı. Dolayısıyla model, test verisinde yorum metninden öğrenmek yerine hedef etikete çok yakın sinyalleri kullanabiliyordu. Bu nedenle bazı kategorilerde %100 test accuracy görülmesi güvenilir değildi.

Güncel modelde bu feature'lar tamamen kaldırıldı. Kullanılan tek girişler:

```text
CleanYorum
Yildiz
```

Bu karar skorları düşürdü; fakat skorların anlamını yükseltti.

## 5. Model Mimarisi

Problem çoklu çıktı sınıflandırmasıdır. Her yorum için beş hedef kolon vardır ve her hedef 0, 1 veya 2 değerini alır.

Kullanılan pipeline:

```text
CleanYorum -> Word TF-IDF, ngram_range=(1, 3)
CleanYorum -> Character TF-IDF, ngram_range=(3, 5)
Yildiz     -> StandardScaler
Classifier -> MultiOutputClassifier(LinearSVC(class_weight="balanced"))
```

Word n-gram yapısı `çok iyi`, `otel temiz değildi`, `rehber ilgiliydi` gibi ifadeleri yakalar. Character n-gram yapısı Türkçedeki eklerden kaynaklanan varyasyonları azaltır. `class_weight="balanced"` sınıf dengesizliğine karşı kullanılır.

## 6. Değerlendirme

Model tek bir train/test split ile değil, 5-Fold Cross Validation ile değerlendirildi. Ayrıca Streamlit arayüzü için 80/20 hold-out metrikleri de üretilmektedir.

5-Fold Cross Validation:

| Metrik | Değer |
| --- | ---: |
| Strict Exact Match Accuracy | 41.12% ± 2.03% |
| Mean Category Accuracy | 81.67% ± 1.15% |
| Mean Macro F1 | 71.61% |
| Mean Weighted F1 | 80.88% |
| Hamming Loss | 18.33% |

Kategori bazlı 5-Fold accuracy:

| Kategori | Accuracy |
| --- | ---: |
| Ulasim | 77.74% |
| Rehber | 79.66% |
| Organizasyon | 77.73% |
| Otel | 87.30% |
| Yemek | 85.92% |

80/20 hold-out:

| Metrik | Değer |
| --- | ---: |
| Strict Exact Match Accuracy | 44.50% |
| Mean Category Accuracy | 82.11% |
| Mean Macro F1 | 71.28% |
| Mean Weighted F1 | 81.18% |
| Hamming Loss | 17.89% |

Strict exact-match metriği çok serttir; bir satırın doğru sayılması için beş kategorinin tamamının aynı anda doğru olması gerekir. Bu nedenle ürün ve akademik yorumda mean category accuracy, macro F1, weighted F1 ve Hamming loss birlikte değerlendirilmelidir.

## 7. Sonuç

Bu çalışma MVP'den production seviyesine geçiş için kritik bir temizlik adımıdır. Eski %100'e yakın skorların cazip ama yanıltıcı olduğu gösterildi. Data leakage kapatıldıktan ve K-Fold Cross Validation eklendikten sonra model 80-88 bandında gerçekçi kategori bazlı başarı üretmektedir.

Bu, daha olgun bir akademik sonuçtur: model artık hedef etiketleri taklit eden feature'larla değil, yorum metni ve yıldız puanı üzerinden öğrenmektedir. Veri seti 532 satırdan 1088 ham satıra çıkarılmış, duplicate yorumlar temizlenmiş ve tüm raporlanabilir metrikler yeniden üretilmiştir.

## 8. Gelecek Çalışmalar

1. Stratified manuel etiket kontrolü yapılabilir.
2. İki bağımsız annotator ile Cohen's Kappa raporlanabilir.
3. BERTurk veya multilingual transformer modelleriyle karşılaştırma yapılabilir.
4. Eğitim sonrası model `.pkl` veya `joblib` artifact olarak saklanabilir.
5. Production dashboard'da şehir, acente, yıldız ve kategori bazlı drift izlenebilir.
