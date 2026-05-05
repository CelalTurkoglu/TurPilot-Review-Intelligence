import re
import string
from pathlib import Path

import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler


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
DATASET_FILENAMES = ["dataset.csv", "updated_dataset.csv"]
TARGET_COLUMNS = ["Ulasim", "Rehber", "Organizasyon", "Otel", "Yemek"]
LABEL_NAMES = {
    0: "Bahsedilmemiş",
    1: "Övgü",
    2: "Şikayet",
}
LABEL_CLASSES = {
    0: "neutral-badge",
    1: "positive-badge",
    2: "negative-badge",
}


# -----------------------------------------------------------------------------
# Turkish stop-word list
# The project must run without downloading external NLP resources, so the most
# common Turkish stop-words are embedded directly in the application.
# -----------------------------------------------------------------------------
TURKISH_STOPWORDS = {
    "acaba", "ama", "aslında", "az", "bazı", "belki", "biri", "birkaç",
    "birşey", "biz", "bu", "çok", "çünkü", "da", "daha", "de", "defa",
    "diye", "eğer", "en", "gibi", "hem", "hep", "hepsi", "her", "hiç",
    "için", "ile", "ise", "kez", "ki", "kim", "mı", "mu", "mü", "nasıl",
    "ne", "neden", "nerde", "nerede", "nereye", "niçin", "niye", "o",
    "sanki", "şey", "siz", "şu", "tüm", "ve", "veya", "ya", "yani",
    "bir", "olarak", "olan", "oldu", "olduk", "olur", "oluyor", "ben",
    "bana", "beni", "bizim", "sizin", "onlar", "onların", "kadar", "sonra",
    "önce", "var", "yok", "şöyle", "böyle", "ancak", "fakat", "lakin",
}


# These words look like stop-words, but they are important for sentiment context.
# Keeping them allows bigrams such as "çok iyiydi" and "hiç memnun" to survive.
CONTEXT_WORDS_TO_KEEP = {"çok", "hiç", "değil", "değildi", "kötü", "iyi"}


def clean_text(text: str) -> str:
    """Clean Turkish review text before TF-IDF vectorization."""
    # Convert possible missing values or non-string inputs into safe strings.
    text = "" if pd.isna(text) else str(text)

    # Lowercasing reduces duplicate vocabulary entries such as "Otel" and "otel".
    text = text.lower()

    # Remove punctuation while preserving Turkish alphabet characters.
    punctuation_pattern = f"[{re.escape(string.punctuation)}“”‘’…]"
    text = re.sub(punctuation_pattern, " ", text)

    # Remove digits because star rating is handled separately by the interface.
    text = re.sub(r"\d+", " ", text)

    # Tokenize by whitespace and remove Turkish stop-words.
    tokens = [
        token
        for token in text.split()
        if (token not in TURKISH_STOPWORDS or token in CONTEXT_WORDS_TO_KEEP)
        and len(token) > 1
    ]

    return " ".join(tokens)


@st.cache_data(show_spinner=False)
def load_dataset() -> pd.DataFrame:
    """Load and validate the labelled tourism review dataset."""
    # Prefer dataset.csv for the final project, but keep updated_dataset.csv as
    # a backwards-compatible fallback for the current workspace.
    data_path = next(
        (BASE_DIR / filename for filename in DATASET_FILENAMES if (BASE_DIR / filename).exists()),
        None,
    )

    if data_path is None:
        expected_files = ", ".join(DATASET_FILENAMES)
        st.error(f"Dataset bulunamadı. Beklenen dosyalardan biri gerekli: {expected_files}")
        st.stop()

    # utf-8-sig handles CSV files exported from Excel with a UTF-8 BOM marker.
    df = pd.read_csv(data_path, encoding="utf-8-sig")

    required_columns = ["Yorum", "Yildiz", *TARGET_COLUMNS]
    missing_columns = [column for column in required_columns if column not in df.columns]
    if missing_columns:
        st.error(f"Eksik kolonlar: {', '.join(missing_columns)}")
        st.stop()

    # Drop rows that cannot be used for supervised training.
    df = df.dropna(subset=["Yorum", "Yildiz", *TARGET_COLUMNS]).copy()

    # The star rating is now used as an independent numeric feature in the
    # ColumnTransformer, so it must be cleaned before the train/test split.
    df["Yildiz"] = pd.to_numeric(df["Yildiz"], errors="coerce").fillna(3).astype(float)
    df["Yildiz"] = df["Yildiz"].clip(lower=1, upper=5)

    # Force target columns into integer classes: 0 = none, 1 = praise, 2 = complaint.
    for column in TARGET_COLUMNS:
        df[column] = pd.to_numeric(df[column], errors="coerce").fillna(0).astype(int)
        df[column] = df[column].clip(lower=0, upper=2)

    df["CleanYorum"] = df["Yorum"].apply(clean_text)
    df = df[df["CleanYorum"].str.len() > 0].copy()

    return df


@st.cache_resource(show_spinner=False)
def train_model(df: pd.DataFrame):
    """Train a text + star-rating MultiOutput Logistic Regression classifier."""
    # Refactor note: X is now a two-column feature table. CleanYorum captures
    # the NLP signal, while Yildiz provides an explicit numeric satisfaction cue.
    X = df[["CleanYorum", "Yildiz"]]
    y = df[TARGET_COLUMNS]

    # A fixed random_state makes the academic demo reproducible.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        shuffle=True,
    )

    # Refactor note: ColumnTransformer builds a feature union. The text column is
    # converted with TF-IDF using unigrams+bigrams, and the star-rating column is
    # scaled into the 0-1 range before both feature blocks are concatenated.
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "text_tfidf",
                TfidfVectorizer(
                    max_features=15000,
                    ngram_range=(1, 2),
                    min_df=1,
                    sublinear_tf=True,
                ),
                "CleanYorum",
            ),
            (
                "star_rating",
                MinMaxScaler(),
                ["Yildiz"],
            ),
        ],
        remainder="drop",
    )

    # Refactor note: class_weight='balanced' reduces the impact of class
    # imbalance for minority labels such as praise or rare complaints. C=50.0
    # was selected after a small validation check because it improved exact-match
    # accuracy while keeping the balanced class weighting requirement.
    model = Pipeline(
        steps=[
            (
                "features",
                preprocessor,
            ),
            (
                "classifier",
                MultiOutputClassifier(
                    LogisticRegression(
                        max_iter=4000,
                        class_weight="balanced",
                        C=50.0,
                        solver="lbfgs",
                    )
                ),
            ),
        ]
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Exact match accuracy is strict: all five labels must be correct at once.
    # sklearn.accuracy_score does not support multiclass-multioutput directly,
    # so the row-wise equality is calculated manually.
    exact_match_accuracy = float((y_test.to_numpy() == y_pred).all(axis=1).mean())

    # Per-category accuracy is easier to interpret in the sidebar.
    per_category_accuracy = {
        column: accuracy_score(y_test[column], y_pred[:, index])
        for index, column in enumerate(TARGET_COLUMNS)
    }

    # Classification reports are stored as dictionaries for a compact UI table.
    reports = {
        column: classification_report(
            y_test[column],
            y_pred[:, index],
            labels=[0, 1, 2],
            target_names=[LABEL_NAMES[0], LABEL_NAMES[1], LABEL_NAMES[2]],
            output_dict=True,
            zero_division=0,
        )
        for index, column in enumerate(TARGET_COLUMNS)
    }

    metrics = {
        "train_size": len(X_train),
        "test_size": len(X_test),
        "exact_match_accuracy": exact_match_accuracy,
        "per_category_accuracy": per_category_accuracy,
        "reports": reports,
    }

    return model, metrics


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
    st.sidebar.caption("TF-IDF N-Gram + Yıldız Feature + Balanced Logistic Regression")

    st.sidebar.metric("Veri Sayısı", len(df))
    st.sidebar.metric("Train/Test", f"{metrics['train_size']} / {metrics['test_size']}")
    st.sidebar.metric("Exact Match Accuracy", f"{metrics['exact_match_accuracy']:.2%}")

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

        # The model expects the same two-feature schema used during training:
        # cleaned review text plus the user-selected star rating.
        cleaned_review = clean_text(review_text)
        prediction_input = pd.DataFrame(
            [
                {
                    "CleanYorum": cleaned_review,
                    "Yildiz": float(star_rating),
                }
            ]
        )
        prediction = model.predict(prediction_input)[0]
        predictions = {
            category: int(prediction[index])
            for index, category in enumerate(TARGET_COLUMNS)
        }

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
