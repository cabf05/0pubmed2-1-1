import streamlit as st
import pandas as pd
import numpy as np
import re
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, ENGLISH_STOP_WORDS
import altair as alt
import io
import requests

st.set_page_config(page_title="Text Trends Analyzer with MeSH", layout="wide")

# -------------------------
# Vocabulário histórico e stopwords médicas
# -------------------------
DEFAULT_VOCAB = {"study","patient","patients","results","conclusion","method","methods","background",
                "objective","aim","participants","randomized","control","group","groups","significant",
                "analysis","data","clinical","treatment","age","years","percent","increase","decrease",
                "observed","reported","compared","model","measure","followup","outcome","sample"}

MEDICAL_STOPWORDS = set([
    "trial", "control", "conclusion", "treatment", "significant", "data",
    "baseline", "adverse", "event", "study", "patients", "results",
    "randomized", "placebo", "clinical", "methods", "analysis", "groups",
    "follow", "period", "risk", "safety"
])

# -------------------------
# Helpers de pré-processamento
# -------------------------
def simple_tokenize(text):
    if pd.isna(text): return []
    s = str(text).lower()
    s = re.sub(r'http\S+', ' ', s)
    s = re.sub(r'\[[^\]]*\]', ' ', s)
    s = re.sub(r'\d+', ' ', s)
    s = re.sub(r'[^\w\s]', ' ', s)
    s = re.sub(r'\s+', ' ', s).strip()
    tokens = re.findall(r'\b[a-z]{2,}\b', s)
    return tokens

@st.cache_data
def preprocess_texts(texts, extra_stopwords):
    stopset = set(ENGLISH_STOP_WORDS) | set([w.lower() for w in extra_stopwords])
    cleaned_docs, token_lists = [], []
    for t in texts:
        toks = [tok for tok in simple_tokenize(t) if tok not in stopset]
        token_lists.append(toks)
        cleaned_docs.append(" ".join(toks))
    return cleaned_docs, token_lists

@st.cache_data
def compute_ngrams_counts(cleaned_docs, ngram_range=(1,3), top_k=50, apply_filters=False):
    if len(cleaned_docs)==0: return pd.DataFrame(columns=["term","count"])
    vec = CountVectorizer(ngram_range=ngram_range, token_pattern=r'(?u)\b[a-z]{2,}\b')
    X = vec.fit_transform(cleaned_docs)
    sums = np.asarray(X.sum(axis=0)).ravel()
    terms = np.array(vec.get_feature_names_out())
    df = pd.DataFrame({"term":terms, "count":sums})

    if apply_filters:
        df = df[~df['term'].str.split().str[0].isin(MEDICAL_STOPWORDS)]
        df = df[df['term'].str.contains(r'[A-Z0-9]|mab$|ib$|ine$|ol$|gene|receptor', regex=True, case=False)]

    df = df.sort_values("count", ascending=False).reset_index(drop=True)
    return df.head(top_k)

@st.cache_data
def compute_tfidf(cleaned_docs, ngram_range=(1,1), top_k=25):
    if len(cleaned_docs)==0: return pd.DataFrame(columns=["term","tfidf"]), None, None
    tf = TfidfVectorizer(ngram_range=ngram_range, token_pattern=r'(?u)\b[a-z]{2,}\b')
    X = tf.fit_transform(cleaned_docs)
    mean_tfidf = np.asarray(X.mean(axis=0)).ravel()
    terms = np.array(tf.get_feature_names_out())
    df = pd.DataFrame({"term":terms,"tfidf":mean_tfidf})
    df = df.sort_values("tfidf", ascending=False).reset_index(drop=True)
    return df.head(top_k), tf, X

@st.cache_data
def novelty_detection(token_lists, historical_vocab, top_k=50):
    historical_vocab = set([w.lower() for w in historical_vocab])
    rows, all_new_tokens = [], []
    for i, toks in enumerate(token_lists):
        toks_set = set(toks)
        new_tokens = sorted([t for t in toks_set if t not in historical_vocab])
        novelty_count = len(new_tokens)
        total_unique = len(toks_set) if len(toks_set)>0 else 1
        novelty_ratio = novelty_count / total_unique
        rows.append({"doc_index":i,"new_tokens":new_tokens,"novelty_count":novelty_count,
                     "novelty_ratio":novelty_ratio,"total_unique_tokens":total_unique})
        all_new_tokens.extend([t for t in toks if t not in historical_vocab])
    counter = Counter(all_new_tokens)
    df_top_new = pd.DataFrame(counter.most_common(top_k), columns=["token","count"])
    df_docs = pd.DataFrame(rows)
    return df_docs, df_top_new

# -------------------------
# MeSH API integration
# -------------------------
@st.cache_data
def check_mesh(term):
    url = f"https://id.nlm.nih.gov/mesh/lookup/descriptor?label={term}&match=exact"
    try:
        resp = requests.get(url, timeout=5)
        if resp.status_code == 200 and resp.json():
            return True
    except:
        pass
    return False

@st.cache_data
def validate_terms_mesh(tokens):
    results = {}
    for t in tokens:
        results[t] = check_mesh(t)
    return results

# -------------------------
# UI
# -------------------------
st.title("🔎 Text Trends Analyzer with MeSH")
uploaded_file = st.file_uploader("Carregue CSV", type=["csv"])
if uploaded_file is None: st.stop()

try:
    df = pd.read_csv(uploaded_file, encoding="utf-8", engine="python")
except: st.stop()

col1,col2,col3 = st.columns([1,1,1])
with col1: title_col = st.selectbox("Coluna Title", df.columns, index=0)
with col2: abstract_col = st.selectbox("Coluna Abstract", df.columns, index=min(1,len(df.columns)-1))
with col3: date_col = st.selectbox("Coluna Date (opcional)", [None]+list(df.columns), index=0)
combine_texts = st.checkbox("Combinar Title + Abstract", True)
extra_stop_text = st.text_area("Stopwords adicionais (vírgula)", value="study,patients")
extra_stopwords = [w.strip().lower() for w in extra_stop_text.split(",") if w.strip()]

vocab_file = st.file_uploader("Vocabulário histórico opcional", type=["txt"])
if vocab_file: historical_vocab = {line.strip().lower() for line in vocab_file.read().decode("utf-8").splitlines() if line.strip()}
else: historical_vocab = DEFAULT_VOCAB.copy()

ngram_choice = st.selectbox("N-grams", ["1-gram","1-2 grams","1-2-3 grams"], index=1)
tfidf_ngram_choice = st.selectbox("TF-IDF n-grams", ["1-gram","1-2 grams","1-3 grams"], index=0)
top_k = st.slider("Top termos", 10,200,50,10)
run = st.button("▶️ Rodar análise")
if not run: st.stop()

# Prepare texts
if combine_texts: df['text_raw'] = df[title_col].fillna('').astype(str) + ". " + df[abstract_col].fillna('').astype(str)
else: df['text_raw'] = df[abstract_col].fillna('').astype(str)
cleaned_docs, token_lists = preprocess_texts(df['text_raw'].tolist(), extra_stopwords)

# NGRAMS com filtro
if ngram_choice=="1-gram": ngram_range=(1,1)
elif ngram_choice=="1-2 grams": ngram_range=(1,2)
else: ngram_range=(1,3)
df_ngrams = compute_ngrams_counts(cleaned_docs, ngram_range=ngram_range, top_k=top_k, apply_filters=True)
st.subheader("N-grams — Frequências")
st.dataframe(df_ngrams, use_container_width=True)

# TF-IDF
