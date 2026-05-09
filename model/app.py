from pathlib import Path

import pandas as pd
import streamlit as st

from training import LABEL_NAMES, TARGET_COLUMNS, clean_text, load_dataset as load_training_dataset
from training import train_model as train_leakage_free_model


# -----------------------------------------------------------------------------
# Streamlit page configuration
# This block controls the browser tab title, layout width, and page icon.
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Turizm Yorum Analizi",
    page_icon="T",
    layout="wide",
)


# -----------------------------------------------------------------------------
# Global constants
# These variables keep the application easy to maintain if the dataset schema
# changes in the future.
# -----------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
LABEL_CLASSES = {
    0: "neutral-badge",
    1: "positive-badge",
    2: "negative-badge",
}


@st.cache_data(show_spinner=False)
def load_dataset() -> pd.DataFrame:
    """Load and validate the labelled tourism review dataset."""
    try:
        return load_training_dataset(BASE_DIR)
    except (FileNotFoundError, ValueError) as exc:
        st.error(str(exc))
        st.stop()


@st.cache_resource(show_spinner=False)
def train_model(df: pd.DataFrame):
    """Train a leakage-free text + star-rating MultiOutput classifier."""
    return train_leakage_free_model(df)


def build_badge(category: str, value: int) -> str:
    """Create a colored HTML badge for a predicted aspect label."""
    label = LABEL_NAMES.get(value, "Bilinmiyor")
    css_class = LABEL_CLASSES.get(value, "neutral-badge")
    return f"<span class='badge {css_class}'>{category}: {label}</span>"


def generate_auto_reply(predictions: dict, star_rating: int) -> str:
    """Generate a corporate reply by combining aspect-specific templates."""
    opening = "Sayın misafirimiz, değerli yorumunuz için teşekkür ederiz."
    reply_parts = []
    praise_parts = []

    # Transportation-related response logic.
    if predictions["Ulasim"] == 2:
        reply_parts.append(
            "Araç, transfer veya ulaşım sürecinde yaşadığınız aksaklıklar için özür dileriz. "
            "Geri bildiriminiz operasyon ekibimize iletilmiş olup rota, araç planlama ve mola süreçleri yeniden değerlendirilecektir."
        )
    elif predictions["Ulasim"] == 1:
        praise_parts.append(
            "Ulaşım ve transfer sürecinden memnun kalmanız bizi ayrıca mutlu etti."
        )

    # Guide-related response logic.
    if predictions["Rehber"] == 2:
        reply_parts.append(
            "Rehberlik hizmetimizle ilgili yaşadığınız memnuniyetsizliği dikkate alıyoruz. "
            "Rehber performansı ve misafir iletişimi süreçleri ilgili birimimiz tarafından incelenecektir."
        )
    elif predictions["Rehber"] == 1:
        praise_parts.append(
            "Rehberimiz hakkındaki güzel sözleriniz ekibimiz için çok kıymetli."
        )

    # Organization-related response logic.
    if predictions["Organizasyon"] == 2:
        reply_parts.append(
            "Tur programı ve organizasyon akışında yaşanan sorunlar için üzgünüz. "
            "Planlama, zaman yönetimi ve bilgilendirme süreçlerimizi iyileştirmek adına yorumunuzu kayıt altına aldık."
        )
    elif predictions["Organizasyon"] == 1:
        praise_parts.append(
            "Organizasyon akışından memnun kalmanız doğru planlama yaptığımızı gösteriyor."
        )

    # Hotel-related response logic.
    if predictions["Otel"] == 2 and predictions["Rehber"] == 1:
        reply_parts.append(
            "Konaklama tesisimizde yaşadığınız sorunlar için özür dileriz. "
            "Şikayetiniz otel yönetimine iletilmiştir. Öte yandan rehberimiz hakkındaki güzel sözleriniz bizi mutlu etti."
        )
    elif predictions["Otel"] == 2:
        reply_parts.append(
            "Konaklama deneyiminizde beklentinizi karşılayamayan noktalar için üzgünüz. "
            "Otel seçimi, oda standardı ve tesis iletişimi konularında gerekli kontroller yapılacaktır."
        )
    elif predictions["Otel"] == 1:
        praise_parts.append(
            "Konaklama deneyiminizden memnun kalmanız bizi sevindirdi."
        )

    # Food-related response logic.
    if predictions["Yemek"] == 2:
        reply_parts.append(
            "Yemek hizmetleriyle ilgili geri bildiriminizi önemsiyoruz. "
            "Restoran seçimi, menü kalitesi ve hijyen standartları operasyon ekibimiz tarafından değerlendirilecektir."
        )
    elif predictions["Yemek"] == 1:
        praise_parts.append(
            "Yemek ve lezzet deneyiminizle ilgili olumlu yorumunuz için teşekkür ederiz."
        )

    # Star rating can strengthen the final tone even though the ML model uses text.
    if star_rating <= 2:
        closing = (
            "Yaşadığınız olumsuz deneyimi telafi edebilmek adına bizimle rezervasyon bilgileriniz üzerinden iletişime geçmenizi rica ederiz."
        )
    elif star_rating == 3:
        closing = (
            "Deneyiminizi daha iyi bir seviyeye taşımak için belirttiğiniz noktaları dikkatle inceleyeceğiz."
        )
    else:
        closing = (
            "Memnuniyetinizi paylaşmanız bizi motive etti; sizi yeni rotalarımızda tekrar ağırlamaktan mutluluk duyarız."
        )

    if reply_parts:
        return " ".join([opening, *reply_parts, *praise_parts, closing])

    if praise_parts:
        return " ".join([opening, *praise_parts, closing])

    return (
        f"{opening} Yorumunuzda belirgin bir hizmet kategorisine ait şikayet veya övgü tespit edilmedi. "
        f"{closing}"
    )


def render_sidebar(metrics: dict, df: pd.DataFrame) -> None:
    """Render academic model metrics in the Streamlit sidebar."""
    st.sidebar.header("Model Performansı")
    st.sidebar.caption("Leakage-free Word/Char TF-IDF + Yildiz + Balanced LinearSVC")

    st.sidebar.metric("Veri Sayısı", len(df))
    st.sidebar.metric("Train/Test", f"{metrics['train_size']} / {metrics['test_size']}")
    st.sidebar.metric("Exact Match Accuracy", f"{metrics['exact_match_accuracy']:.2%}")
    st.sidebar.metric("Ortalama Kategori Accuracy", f"{metrics['mean_category_accuracy']:.2%}")
    st.sidebar.metric("Ortalama Macro F1", f"{metrics['mean_macro_f1']:.2%}")
    st.sidebar.metric("Hamming Loss", f"{metrics['hamming_loss']:.2%}")

    st.sidebar.subheader("Kategori Accuracy")
    for category, score in metrics["per_category_accuracy"].items():
        st.sidebar.progress(score, text=f"{category}: {score:.2%}")

    with st.sidebar.expander("Detaylı Rapor"):
        rows = []
        for category, report in metrics["reports"].items():
            rows.append(
                {
                    "Kategori": category,
                    "Macro F1": report["macro avg"]["f1-score"],
                    "Weighted F1": report["weighted avg"]["f1-score"],
                    "Precision": report["weighted avg"]["precision"],
                    "Recall": report["weighted avg"]["recall"],
                }
            )
        report_df = pd.DataFrame(rows)
        st.dataframe(report_df, width="stretch", hide_index=True)


def render_style() -> None:
    """Inject small CSS rules for a cleaner Streamlit prototype UI."""
    st.markdown(
        """
        <style>
        .main-title {
            font-size: 2.35rem;
            font-weight: 800;
            margin-bottom: 0.25rem;
        }
        .subtitle {
            color: #5d6472;
            font-size: 1.05rem;
            margin-bottom: 1.4rem;
        }
        .badge-wrapper {
            display: flex;
            flex-wrap: wrap;
            gap: 0.65rem;
            margin: 0.5rem 0 1rem 0;
        }
        .badge {
            border-radius: 999px;
            display: inline-flex;
            align-items: center;
            padding: 0.45rem 0.8rem;
            font-size: 0.92rem;
            font-weight: 700;
            border: 1px solid transparent;
        }
        .neutral-badge {
            background: #f1f5f9;
            color: #475569;
            border-color: #cbd5e1;
        }
        .positive-badge {
            background: #dcfce7;
            color: #166534;
            border-color: #86efac;
        }
        .negative-badge {
            background: #fee2e2;
            color: #991b1b;
            border-color: #fecaca;
        }
        .reply-box {
            background: #f8fafc;
            border: 1px solid #dbe4ee;
            border-radius: 14px;
            padding: 1rem 1.1rem;
            line-height: 1.65;
            color: #1f2937;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    """Run the Streamlit application."""
    render_style()

    st.markdown(
        "<div class='main-title'>Turizm Acente Yorum Analizi ve Otonom Yanıt Sistemi</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<div class='subtitle'>NLP tabanlı çoklu çıktı sınıflandırma modeli ile tur yorumlarında hizmet boyutu analizi.</div>",
        unsafe_allow_html=True,
    )

    df = load_dataset()
    model, metrics = train_model(df)
    render_sidebar(metrics, df)

    left_column, right_column = st.columns([3, 1])

    with left_column:
        review_text = st.text_area(
            "Yeni turizm yorumunu girin",
            height=190,
            placeholder=(
                "Örnek: Rehberimiz çok ilgiliydi fakat otel temiz değildi ve transfer süreci gecikti..."
            ),
        )

    with right_column:
        star_rating = st.selectbox(
            "Yıldız Puanı",
            options=[1, 2, 3, 4, 5],
            index=4,
        )
        st.info("0: Bahsedilmemiş\n\n1: Övgü\n\n2: Şikayet")

    analyze_clicked = st.button("Analiz Et", type="primary", width="stretch")

    if analyze_clicked:
        if not review_text.strip():
            st.warning("Lütfen analiz etmek için bir yorum girin.")
            return

        # The model sees only the cleaned review text and the selected star rating.
        prediction_input = pd.DataFrame(
            [
                {
                    "CleanYorum": clean_text(review_text),
                    "Yildiz": float(star_rating),
                }
            ]
        )
        prediction = model.predict(prediction_input)[0]
        predictions = {
            category: int(prediction[index])
            for index, category in enumerate(TARGET_COLUMNS)
        }
        cleaned_review = prediction_input.loc[0, "CleanYorum"]

        st.subheader("Tahmin Edilen Kategori Etiketleri")
        badges = [build_badge(category, value) for category, value in predictions.items()]
        st.markdown(
            f"<div class='badge-wrapper'>{''.join(badges)}</div>",
            unsafe_allow_html=True,
        )

        complaint_count = sum(value == 2 for value in predictions.values())
        praise_count = sum(value == 1 for value in predictions.values())

        if complaint_count > 0:
            st.error(f"{complaint_count} hizmet kategorisinde şikayet sinyali tespit edildi.")
        elif praise_count > 0:
            st.success(f"{praise_count} hizmet kategorisinde olumlu geri bildirim tespit edildi.")
        else:
            st.warning("Yorumda belirgin bir hizmet kategorisi sinyali tespit edilmedi.")

        auto_reply = generate_auto_reply(predictions, star_rating)

        st.subheader("Otomatik Kurumsal Yanıt")
        st.markdown(
            f"<div class='reply-box'>{auto_reply}</div>",
            unsafe_allow_html=True,
        )

        with st.expander("Model Girdisi ve Ham Tahmin"):
            st.write("Temizlenmiş metin:")
            st.code(cleaned_review or "(boş)", language="text")
            st.json(predictions)


if __name__ == "__main__":
    main()
