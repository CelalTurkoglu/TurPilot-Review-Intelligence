import pandas as pd
import requests
import json
import time
import os
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent


def load_env_file():
    """Load simple KEY=VALUE pairs from the project .env file if it exists."""
    env_path = PROJECT_ROOT / ".env"
    if not env_path.exists():
        return

    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


load_env_file()

# API credentials are intentionally read from .env instead of being hardcoded.
API_KEY = os.getenv("OPENROUTER_API_KEY", "")

# Using gpt-4o-mini as it is cheap and highly capable in Turkish JSON outputs.
MODEL = os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini")
API_URL = os.getenv("OPENROUTER_API_URL", "https://openrouter.ai/api/v1/chat/completions")

# Prompt template
SYSTEM_PROMPT = """Sen uzman bir Veri Bilimci ve Doğal Dil İşleme (NLP) analistisin.
Görevin, turizm acentelerine ait müşteri yorumlarında "Varlık Tabanlı Duygu Analizi (Aspect-Based Sentiment Analysis)" yapmaktır.

Aşağıdaki 5 kategori için puanlama yapmalısın:
1. Ulasim
2. Rehber
3. Organizasyon
4. Otel
5. Yemek

PUANLAMA KURALLARI:
* [ 0 ] : Yorumda bu kategoriden HİÇ BAHSEDİLMEMİŞ veya tamamen nötr.
* [ 1 ] : Yorumda bu kategoriyle ilgili ÖVGÜ, MEMNUNİYET veya POZİTİF bir durum var.
* [ 2 ] : Yorumda bu kategoriyle ilgili ŞİKAYET, SORUN veya NEGATİF bir durum var.

KATEGORİ KAPSAMLARI:
- Ulasim: Otobüs, araç, şoför, kaptan, yolculuk konforu, klima arızası, motor arızası, şoförün tavrı.
- Rehber: Tur rehberi, hostes, personelin ilgisi, rehberin bilgi seviyesi, personelin güler yüzü.
- Organizasyon: Tur programına uyulması, zaman yönetimi, sınır kapısında bekleme, geç kalma, turların birleştirilmesi.
- Otel: Konaklama, oda sıcaklığı, otelin temizliği, otelin konumu.
- Yemek: Sabah kahvaltısı, akşam yemeği, molalarda gidilen restoranlar, yöresel lezzetler.

ETİKETLEME KURALLARI:
1. Genel bir memnuniyet (örn: "Her şey çok güzeldi") varsa: Ulasim:0, Rehber:0, Organizasyon:1, Otel:0, Yemek:0
2. Hem şikayet hem övgü olan kısımları bağımsız değerlendir.

Çıktıyı SADECE AŞAĞIDAKİ JSON FORMATINDA ver. Asla ek metin yazma:
{"Ulasim": 0, "Rehber": 0, "Organizasyon": 0, "Otel": 0, "Yemek": 0}
"""

def analyze_review(review_text):
    if not API_KEY:
        raise RuntimeError("OPENROUTER_API_KEY .env dosyasinda tanimli degil.")

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Yorum: {review_text}"}
        ],
        "response_format": {"type": "json_object"}
    }
    
    try:
        response = requests.post(API_URL, headers=headers, json=payload, timeout=20)
        response.raise_for_status()
        result = response.json()
        content = result['choices'][0]['message']['content']
        return json.loads(content)
    except Exception as e:
        print(f"Hata oluştu: {e}")
        return None

def main():
    print("Veri seti yükleniyor...")
    df = pd.read_csv("dataset.csv", header=None, names=["Yorum", "Yildiz"])
    
    # Sütunları hazırla
    for col in ["Ulasim", "Rehber", "Organizasyon", "Otel", "Yemek"]:
        df[col] = 0
        
    print(f"Toplam yorum sayısı: {len(df)}")
    
    for index, row in df.iterrows():
        review_text = str(row['Yorum'])
        if pd.isna(review_text) or review_text.strip() == "":
            continue
            
        print(f"Analiz ediliyor ({index+1}/{len(df)})...")
        
        analysis = analyze_review(review_text)
        if analysis:
            for key in ["Ulasim", "Rehber", "Organizasyon", "Otel", "Yemek"]:
                df.at[index, key] = analysis.get(key, 0)
        
        # Rate limitlere (API hız limitleri) takılmamak için kısa bir bekleme (opsiyonel)
        # OpenRouter genelde iyidir ancak garanti olması için:
        time.sleep(0.3)
        
        # Her 50 satırda bir ara kayıt yap (güvenlik için)
        if (index + 1) % 50 == 0:
            df.to_csv("updated_dataset.csv", index=False)
            print(f"--- {index+1} satır kaydedildi ---")
            
    df.to_csv("updated_dataset.csv", index=False)
    print("İşlem tamamlandı! Sonuçlar updated_dataset.csv dosyasına kaydedildi.")

if __name__ == "__main__":
    main()
