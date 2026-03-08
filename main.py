import streamlit as st
import numpy as np
import pandas as pd
import joblib
import os
import re
from nltk.tokenize import WordPunctTokenizer
from nltk.stem.wordnet import WordNetLemmatizer

# ── Page Config ─────────────────────────────────────────────
st.set_page_config(
    page_title="Sentiment Analyzer",
    page_icon="💬",
    layout="centered"
)

# ── Custom CSS ───────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');

/* ── Base ── */
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: #0f0f0f;
    color: #f0ede6;
}

.stApp {
    background-color: #0f0f0f;
}

/* ── Header ── */
.hero {
    text-align: center;
    padding: 3rem 0 2rem 0;
}
.hero-title {
    font-family: 'Syne', sans-serif;
    font-size: 3.2rem;
    font-weight: 800;
    letter-spacing: -1px;
    line-height: 1.1;
    color: #f0ede6;
    margin-bottom: 0.4rem;
}
.hero-title span {
    color: #c8f060;
}
.hero-sub {
    font-size: 1rem;
    color: #888;
    font-weight: 300;
    letter-spacing: 0.5px;
}

/* ── Card ── */
.card {
    background: #1a1a1a;
    border: 1px solid #2a2a2a;
    border-radius: 16px;
    padding: 2rem;
    margin-bottom: 1.5rem;
}

/* ── Label ── */
.section-label {
    font-family: 'Syne', sans-serif;
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 3px;
    text-transform: uppercase;
    color: #555;
    margin-bottom: 0.6rem;
}

/* ── Result Box ── */
.result-box {
    border-radius: 14px;
    padding: 2rem;
    text-align: center;
    margin-top: 1.5rem;
    border: 1px solid transparent;
}
.result-positive {
    background: linear-gradient(135deg, #0d2b0a, #122e0a);
    border-color: #2d6e1f;
}
.result-negative {
    background: linear-gradient(135deg, #2b0a0a, #2e0f0a);
    border-color: #6e1f1f;
}
.result-neutral {
    background: linear-gradient(135deg, #1a1a0d, #1e1e0d);
    border-color: #5e5e1f;
}
.result-emoji {
    font-size: 3rem;
    margin-bottom: 0.5rem;
}
.result-label {
    font-family: 'Syne', sans-serif;
    font-size: 1.8rem;
    font-weight: 800;
    letter-spacing: -0.5px;
}
.result-positive .result-label  { color: #c8f060; }
.result-negative .result-label  { color: #f06060; }
.result-neutral  .result-label  { color: #f0d060; }
.result-confidence {
    font-size: 0.85rem;
    color: #888;
    margin-top: 0.4rem;
    font-weight: 300;
}

/* ── Stat Pills ── */
.stat-row {
    display: flex;
    gap: 0.8rem;
    justify-content: center;
    margin-top: 1.2rem;
    flex-wrap: wrap;
}
.stat-pill {
    background: #222;
    border: 1px solid #333;
    border-radius: 20px;
    padding: 0.4rem 1rem;
    font-size: 0.8rem;
    color: #aaa;
}
.stat-pill strong {
    color: #f0ede6;
}

/* ── Batch Table ── */
.batch-header {
    font-family: 'Syne', sans-serif;
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 3px;
    text-transform: uppercase;
    color: #555;
    padding: 0.8rem 1rem;
    border-bottom: 1px solid #2a2a2a;
    display: grid;
    grid-template-columns: 1fr 130px;
}
.batch-row {
    padding: 0.9rem 1rem;
    border-bottom: 1px solid #1e1e1e;
    display: grid;
    grid-template-columns: 1fr 130px;
    align-items: center;
    transition: background 0.15s;
}
.batch-row:hover { background: #1e1e1e; }
.batch-text {
    font-size: 0.88rem;
    color: #ccc;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    padding-right: 1rem;
}
.badge {
    display: inline-block;
    padding: 0.25rem 0.75rem;
    border-radius: 20px;
    font-size: 0.75rem;
    font-weight: 600;
    font-family: 'Syne', sans-serif;
    letter-spacing: 0.5px;
}
.badge-positive { background: #1a2e0a; color: #c8f060; border: 1px solid #2d6e1f; }
.badge-negative { background: #2e0a0a; color: #f06060; border: 1px solid #6e1f1f; }
.badge-neutral  { background: #2e2e0a; color: #f0d060; border: 1px solid #5e5e1f; }

/* ── Textarea & Inputs ── */
textarea, .stTextArea textarea {
    background: #141414 !important;
    border: 1px solid #2a2a2a !important;
    border-radius: 10px !important;
    color: #f0ede6 !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.95rem !important;
}
textarea:focus, .stTextArea textarea:focus {
    border-color: #c8f060 !important;
    box-shadow: 0 0 0 2px rgba(200,240,96,0.1) !important;
}

/* ── Button ── */
.stButton > button {
    background: #c8f060 !important;
    color: #0f0f0f !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    font-size: 0.9rem !important;
    letter-spacing: 1px !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 0.7rem 2rem !important;
    width: 100% !important;
    transition: opacity 0.2s !important;
}
.stButton > button:hover {
    opacity: 0.85 !important;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background: #1a1a1a;
    border-radius: 10px;
    padding: 4px;
    gap: 4px;
    border: 1px solid #2a2a2a;
}
.stTabs [data-baseweb="tab"] {
    font-family: 'Syne', sans-serif !important;
    font-size: 0.8rem !important;
    font-weight: 600 !important;
    letter-spacing: 1px !important;
    color: #555 !important;
    border-radius: 8px !important;
    padding: 0.5rem 1.2rem !important;
}
.stTabs [aria-selected="true"] {
    background: #c8f060 !important;
    color: #0f0f0f !important;
}

/* ── Divider ── */
hr { border-color: #2a2a2a; }

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: #141414;
    border-right: 1px solid #2a2a2a;
}
.sidebar-title {
    font-family: 'Syne', sans-serif;
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 3px;
    text-transform: uppercase;
    color: #555;
    margin-bottom: 1rem;
}
.model-badge {
    background: #1e1e1e;
    border: 1px solid #2d6e1f;
    border-radius: 8px;
    padding: 0.6rem 0.8rem;
    font-size: 0.82rem;
    color: #c8f060;
    font-family: 'Syne', sans-serif;
    font-weight: 600;
    margin-bottom: 0.5rem;
}
.info-item {
    font-size: 0.82rem;
    color: #666;
    padding: 0.4rem 0;
    border-bottom: 1px solid #1e1e1e;
}
.info-item strong { color: #aaa; }

/* ── Hide Streamlit branding ── */
#MainMenu, footer, header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ── Load Models ──────────────────────────────────────────────
@st.cache_resource
def load_models():
    base = 'saved_models'
    tvec1    = joblib.load(f'{base}/tfidf_text.pkl')
    tvec2    = joblib.load(f'{base}/tfidf_title.pkl')
    le_senti = joblib.load(f'{base}/label_encoder_sentiment.pkl')
    model    = joblib.load(f'{base}/best_model.pkl')
    return tvec1, tvec2, le_senti, model

try:
    tvec1, tvec2, le_senti, best_model = load_models()
    models_loaded = True
except Exception as e:
    models_loaded = False
    load_error = str(e)


# ── Text Cleaning (same as training) ────────────────────────
@st.cache_resource
def get_nlp_tools():
    import nltk
    nltk.download('wordnet', quiet=True)
    nltk.download('punkt',   quiet=True)
    tok = WordPunctTokenizer()
    wnl = WordNetLemmatizer()
    return tok, wnl

tok, wnl = get_nlp_tools()

negations_dic = {
    "isn't":"is not","aren't":"are not","wasn't":"was not","weren't":"were not",
    "haven't":"have not","hasn't":"has not","hadn't":"had not","won't":"will not",
    "wouldn't":"would not","don't":"do not","doesn't":"does not","didn't":"did not",
    "can't":"can not","couldn't":"could not","shouldn't":"should not",
    "mightn't":"might not","mustn't":"must not"
}
neg_pattern = re.compile(r'\b(' + '|'.join(negations_dic.keys()) + r')\b')

def clean_text(t):
    t = str(t).lower()
    t = neg_pattern.sub(lambda x: negations_dic[x.group()], t)
    t = re.sub("[^a-zA-Z0-9]", " ", t)
    tokens = [w for w in tok.tokenize(t) if len(w) > 3]
    return " ".join([wnl.lemmatize(w) for w in tokens]).strip()


# ── Predict Function ─────────────────────────────────────────
label_map  = {0:'Negative', 1:'Neutral', 2:'Positive'}
emoji_map  = {'Positive':'😊', 'Negative':'😞', 'Neutral':'😐'}
class_map  = {'Positive':'positive', 'Negative':'negative', 'Neutral':'neutral'}

def predict(review_text, review_title=''):
    cleaned_text  = clean_text(review_text)
    cleaned_title = clean_text(review_title)

    n_tfidf_text  = tvec1.get_feature_names_out().shape[0]
    n_tfidf_title = tvec2.get_feature_names_out().shape[0]
    n_total       = best_model.n_features_in_
    n_structured  = n_total - n_tfidf_text - n_tfidf_title

    structured  = np.zeros((1, n_structured))
    text_tfidf  = tvec1.transform([cleaned_text]).toarray()
    title_tfidf = tvec2.transform([cleaned_title]).toarray()
    features    = np.hstack([structured, text_tfidf, title_tfidf])

    pred  = best_model.predict(features)[0]
    label = label_map[pred]

    # Word counts for stats
    word_count = len(review_text.split())
    char_count = len(review_text)

    return label, word_count, char_count


# ── Sidebar ──────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="sidebar-title">Model Info</div>', unsafe_allow_html=True)

    if models_loaded:
        model_name = type(best_model).__name__
        st.markdown(f'<div class="model-badge">✦ {model_name}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="info-item"><strong>Features</strong> &nbsp; {best_model.n_features_in_:,}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="info-item"><strong>Classes</strong> &nbsp; Negative · Neutral · Positive</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="info-item"><strong>Vectorizer</strong> &nbsp; TF-IDF</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="info-item"><strong>Status</strong> &nbsp; ✅ Ready</div>', unsafe_allow_html=True)
    else:
        st.error("Models not loaded")
        st.code(load_error)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="sidebar-title">About</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="info-item">
    Analyses product review text and classifies sentiment as
    <strong>Positive</strong>, <strong>Neutral</strong>, or <strong>Negative</strong>
    using a trained ML model with TF-IDF features.
    </div>
    """, unsafe_allow_html=True)


# ── Hero Header ──────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-title">Sentiment <span>Analyzer</span></div>
    <div class="hero-sub">Product Review Intelligence · ML-Powered</div>
</div>
""", unsafe_allow_html=True)

if not models_loaded:
    st.error(f"⚠️ Could not load models from `saved_models/` folder. Make sure all `.pkl` files are present.")
    st.stop()


# ── Tabs ─────────────────────────────────────────────────────
tab1, tab2 = st.tabs(["SINGLE REVIEW", "BATCH ANALYSIS"])


# ════════════════════════════════════════════════════════════
# TAB 1 — Single Review
# ════════════════════════════════════════════════════════════
with tab1:
    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown('<div class="section-label">Review Text</div>', unsafe_allow_html=True)
    review_text = st.text_area(
        label="review_text",
        label_visibility="collapsed",
        placeholder="Paste or type a product review here...",
        height=140
    )

    st.markdown('<div class="section-label" style="margin-top:1rem;">Review Title (optional)</div>', unsafe_allow_html=True)
    review_title = st.text_area(
        label="review_title",
        label_visibility="collapsed",
        placeholder="e.g. Great value for money",
        height=60
    )

    st.markdown("<br>", unsafe_allow_html=True)
    analyze_btn = st.button("ANALYZE SENTIMENT", key="single")

    if analyze_btn:
        if not review_text.strip():
            st.warning("Please enter a review to analyze.")
        else:
            with st.spinner("Analyzing..."):
                label, word_count, char_count = predict(review_text, review_title)

            css_class = class_map[label]
            emoji     = emoji_map[label]

            st.markdown(f"""
            <div class="result-box result-{css_class}">
                <div class="result-emoji">{emoji}</div>
                <div class="result-label">{label}</div>
                <div class="result-confidence">Sentiment detected in your review</div>
                <div class="stat-row">
                    <div class="stat-pill"><strong>{word_count}</strong> words</div>
                    <div class="stat-pill"><strong>{char_count}</strong> characters</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    # ── Example Reviews ──────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-label">Try an Example</div>', unsafe_allow_html=True)

    examples = [
        ("Absolutely love this product! Works perfectly and arrived fast.", "Amazing purchase"),
        ("Terrible quality. Broke after two days. Total waste of money.",   "Disappointing"),
        ("It does the job. Nothing special but gets the work done.",        "Okay product"),
    ]

    cols = st.columns(3)
    for i, (ex_text, ex_title) in enumerate(examples):
        with cols[i]:
            if st.button(f'"{ex_text[:28]}..."', key=f'ex_{i}'):
                label, wc, cc = predict(ex_text, ex_title)
                css_class = class_map[label]
                emoji     = emoji_map[label]
                st.markdown(f"""
                <div class="result-box result-{css_class}" style="padding:1.2rem;">
                    <div style="font-size:1.8rem;">{emoji}</div>
                    <div class="result-label" style="font-size:1.2rem;">{label}</div>
                    <div style="font-size:0.75rem;color:#666;margin-top:0.3rem;">
                        {ex_text[:50]}...
                    </div>
                </div>
                """, unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════
# TAB 2 — Batch Analysis
# ════════════════════════════════════════════════════════════
with tab2:
    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown('<div class="section-label">Paste Multiple Reviews (one per line)</div>', unsafe_allow_html=True)
    batch_input = st.text_area(
        label="batch_reviews",
        label_visibility="collapsed",
        placeholder="This product is amazing!\nTerrible, broke immediately.\nDecent quality for the price.\nWould not recommend to anyone.",
        height=200
    )

    st.markdown("<br>", unsafe_allow_html=True)

    col1, col2 = st.columns([2, 1])
    with col1:
        batch_btn = st.button("ANALYZE ALL REVIEWS", key="batch")
    with col2:
        uploaded = st.file_uploader("or upload CSV", type=['csv'],
                                    label_visibility="collapsed")

    # ── CSV Upload ───────────────────────────────────────────
    if uploaded is not None:
        df_upload = pd.read_csv(uploaded)
        st.markdown(f'<div class="section-label">Preview — {len(df_upload)} rows</div>',
                    unsafe_allow_html=True)
        st.dataframe(df_upload.head(), use_container_width=True)

        text_col = st.selectbox("Select the review text column", df_upload.columns)

        if st.button("ANALYZE CSV", key="csv_btn"):
            with st.spinner(f"Analyzing {len(df_upload)} reviews..."):
                preds = []
                for _, row in df_upload.iterrows():
                    lbl, _, _ = predict(str(row[text_col]))
                    preds.append(lbl)
                df_upload['Predicted Sentiment'] = preds

            # Summary donut-style counts
            counts = pd.Series(preds).value_counts()
            total  = len(preds)

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown('<div class="section-label">Results Summary</div>', unsafe_allow_html=True)

            m1, m2, m3 = st.columns(3)
            pos_count = counts.get('Positive', 0)
            neu_count = counts.get('Neutral',  0)
            neg_count = counts.get('Negative', 0)

            m1.markdown(f"""
            <div class="card" style="text-align:center;padding:1.2rem;">
                <div style="font-size:1.8rem;color:#c8f060;font-family:Syne,sans-serif;
                            font-weight:800;">{pos_count}</div>
                <div style="font-size:0.75rem;color:#555;letter-spacing:2px;">POSITIVE</div>
                <div style="font-size:0.8rem;color:#888;">{pos_count/total*100:.1f}%</div>
            </div>""", unsafe_allow_html=True)

            m2.markdown(f"""
            <div class="card" style="text-align:center;padding:1.2rem;">
                <div style="font-size:1.8rem;color:#f0d060;font-family:Syne,sans-serif;
                            font-weight:800;">{neu_count}</div>
                <div style="font-size:0.75rem;color:#555;letter-spacing:2px;">NEUTRAL</div>
                <div style="font-size:0.8rem;color:#888;">{neu_count/total*100:.1f}%</div>
            </div>""", unsafe_allow_html=True)

            m3.markdown(f"""
            <div class="card" style="text-align:center;padding:1.2rem;">
                <div style="font-size:1.8rem;color:#f06060;font-family:Syne,sans-serif;
                            font-weight:800;">{neg_count}</div>
                <div style="font-size:0.75rem;color:#555;letter-spacing:2px;">NEGATIVE</div>
                <div style="font-size:0.8rem;color:#888;">{neg_count/total*100:.1f}%</div>
            </div>""", unsafe_allow_html=True)

            # Full results table
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown('<div class="section-label">Full Results</div>', unsafe_allow_html=True)
            st.dataframe(df_upload, use_container_width=True)

            # Download button
            csv_out = df_upload.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="⬇  DOWNLOAD RESULTS CSV",
                data=csv_out,
                file_name='sentiment_results.csv',
                mime='text/csv'
            )

    # ── Paste Batch ──────────────────────────────────────────
    if batch_btn:
        if not batch_input.strip():
            st.warning("Please enter at least one review.")
        else:
            lines = [l.strip() for l in batch_input.strip().split('\n') if l.strip()]

            with st.spinner(f"Analyzing {len(lines)} reviews..."):
                results = []
                for line in lines:
                    lbl, wc, _ = predict(line)
                    results.append({'text': line, 'label': lbl, 'words': wc})

            # Summary
            counts = pd.Series([r['label'] for r in results]).value_counts()
            total  = len(results)

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown('<div class="section-label">Summary</div>', unsafe_allow_html=True)

            pos_c = counts.get('Positive', 0)
            neu_c = counts.get('Neutral',  0)
            neg_c = counts.get('Negative', 0)

            c1, c2, c3 = st.columns(3)
            c1.markdown(f"""
            <div class="card" style="text-align:center;padding:1rem;">
                <div style="font-size:1.6rem;color:#c8f060;font-family:Syne,sans-serif;font-weight:800;">{pos_c}</div>
                <div style="font-size:0.7rem;color:#555;letter-spacing:2px;">POSITIVE</div>
            </div>""", unsafe_allow_html=True)
            c2.markdown(f"""
            <div class="card" style="text-align:center;padding:1rem;">
                <div style="font-size:1.6rem;color:#f0d060;font-family:Syne,sans-serif;font-weight:800;">{neu_c}</div>
                <div style="font-size:0.7rem;color:#555;letter-spacing:2px;">NEUTRAL</div>
            </div>""", unsafe_allow_html=True)
            c3.markdown(f"""
            <div class="card" style="text-align:center;padding:1rem;">
                <div style="font-size:1.6rem;color:#f06060;font-family:Syne,sans-serif;font-weight:800;">{neg_c}</div>
                <div style="font-size:0.7rem;color:#555;letter-spacing:2px;">NEGATIVE</div>
            </div>""", unsafe_allow_html=True)

            # Results list
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown('<div class="section-label">Results</div>', unsafe_allow_html=True)

            st.markdown('<div class="card" style="padding:0;">', unsafe_allow_html=True)
            st.markdown("""
            <div class="batch-header">
                <span>REVIEW</span>
                <span>SENTIMENT</span>
            </div>""", unsafe_allow_html=True)

            for r in results:
                badge_cls = f"badge-{r['label'].lower()}"
                em = emoji_map[r['label']]
                st.markdown(f"""
                <div class="batch-row">
                    <div class="batch-text" title="{r['text']}">{r['text']}</div>
                    <div><span class="badge {badge_cls}">{em} {r['label']}</span></div>
                </div>""", unsafe_allow_html=True)

            st.markdown('</div>', unsafe_allow_html=True)

            # Download
            df_results = pd.DataFrame(results)[['text','label']]
            df_results.columns = ['Review','Sentiment']
            csv_out = df_results.to_csv(index=False).encode('utf-8')
            st.markdown("<br>", unsafe_allow_html=True)
            st.download_button(
                label="⬇  DOWNLOAD RESULTS CSV",
                data=csv_out,
                file_name='batch_sentiment_results.csv',
                mime='text/csv'
            )
