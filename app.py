# app.py
"""
Streamlit app - Text Trends Analyzer (with optional spaCy lemmatization)
Designed to work on Streamlit Community Cloud when requirements.txt
includes spacy and the en_core_web_sm wheel.
"""

import streamlit as st
import pandas as pd
import numpy as np
import re
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, ENGLISH_STOP_WORDS
import altair as alt
import io

st.set_page_config(page_title="Text Trends Analyzer (spaCy)", layout="wide")

# -----------------------
# Try import spaCy (may be installed via requirements.txt)
# -----------------------
SPACY_AVAILABLE = False
nlp = None
try:
    import spacy
    SPACY_AVAILABLE = True
except Exception:
    SPACY_AVAILABLE = False

# helper to attempt loading the model (with a runtime fallback attempt)
def try_load_spacy_model(show_messages=True):
    global nlp, SPACY_AVAILABLE
    if not SPACY_AVAILABLE:
        return False, "spaCy package not installed."
    try:
        # disable heavy components we don't need for lemmatization
        nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
        return True, None
    except Exception as e:
        # try to download at runtime as fallback (may fail in some environments)
        try:
            if show_messages:
                st.info("spaCy model não encontrado — tentando baixar em runtime (pode demorar).")
            import spacy.cli
            spacy.cli.download("en_core_web_sm")
            nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
            return True, None
        except Exception as e2:
            return False, str(e2)

# -----------------------
# Tokenizers / preprocess
# -----------------------
def simple_tokenize(text):
    """Lowercase, remove urls/numbers/punct, return tokens (len>=2)."""
    if pd.isna(text):
        return []
    s = str(text).lower()
    s = re.sub(r'http\S+', ' ', s)
    s = re.sub(r'\[[^\]]*\]', ' ', s)
    s = re.sub(r'\d+', ' ', s)
    s = re.sub(r'[^\w\s]', ' ', s)
    s = re.sub(r'\s+', ' ', s).strip()
    toks = re.findall(r'\b[a-z]{2,}\b', s)
    return toks

@st.cache_data
def preprocess_texts_simple(texts, extra_stopwords):
    stopset = set(ENGLISH_STOP_WORDS) | set([w.lower().strip() for w in extra_stopwords if w.strip()])
    cleaned_docs = []
    token_lists = []
    for t in texts:
        toks = [tok for tok in simple_tokenize(t) if tok not in stopset]
        token_lists.append(toks)
        cleaned_docs.append(" ".join(toks))
    return cleaned_docs, token_lists

@st.cache_data
def preprocess_texts_spacy(texts, extra_stopwords):
    """Use spaCy to lemmatize + filter. Returns cleaned strings and token lists."""
    stopset = set(ENGLISH_STOP_WORDS) | set([w.lower().strip() for w in extra_stopwords if w.strip()])
    cleaned_docs = []
    token_lists = []
    # Ensure nlp exists
    global nlp
    if nlp is None:
        return preprocess_texts_simple(texts, extra_stopwords)
    # adjust max_length if necessary (some long abstracts)
    try:
        maxlen = max(len(t) for t in texts) if texts else 0
        if maxlen > getattr(nlp, "max_length", 0):
            nlp.max_length = maxlen + 1000
    except Exception:
        pass

    for doc in nlp.pipe(texts, batch_size=32):
        toks = []
        for token in doc:
            # keep only alpha tokens; use lemma; exclude stopwords & short
            if not token.is_alpha:
                continue
            lemma = token.lemma_.lower().strip()
            if not lemma or len(lemma) < 2:
                continue
            # spaCy's token.is_stop is useful; combine with our stopset
            if getattr(token, "is_stop", False):
                continue
            if lemma in stopset:
                continue
            toks.append(lemma)
        token_lists.append(toks)
        cleaned_docs.append(" ".join(toks))
    return cleaned_docs, token_lists

# -----------------------
# N-grams / TF-IDF / novelty
# -----------------------
@st.cache_data
def compute_ngrams_counts(cleaned_docs, ngram_range=(1,3), top_k=50):
    if len(cleaned_docs) == 0:
        return pd.DataFrame(columns=["term","count"])
    vec = CountVectorizer(ngram_range=ngram_range, token_pattern=r'(?u)\b[a-z]{2,}\b')
    X = vec.fit_transform(cleaned_docs)
    sums = np.asarray(X.sum(axis=0)).ravel()
    terms = np.array(vec.get_feature_names_out())
    df = pd.DataFrame({"term": terms, "count": sums})
    df = df.sort_values("count", ascending=False).reset_index(drop=True)
    return df.head(top_k)

@st.cache_data
def compute_tfidf(cleaned_docs, ngram_range=(1,1), top_k=25):
    if len(cleaned_docs) == 0:
        return pd.DataFrame(columns=["term","tfidf"]), None, None
    tf = TfidfVectorizer(ngram_range=ngram_range, token_pattern=r'(?u)\b[a-z]{2,}\b')
    X = tf.fit_transform(cleaned_docs)
    mean_tfidf = np.asarray(X.mean(axis=0)).ravel()
    terms = np.array(tf.get_feature_names_out())
    df = pd.DataFrame({"term": terms, "tfidf": mean_tfidf})
    df = df.sort_values("tfidf", ascending=False).reset_index(drop=True)
    return df.head(top_k), tf, X

def novelty_detection(token_lists, historical_vocab, top_k=50):
    historical_vocab = set([w.lower() for w in historical_vocab])
    rows = []
    all_new_tokens = []
    for i, toks in enumerate(token_lists):
        toks_set = set(toks)
        new_tokens = sorted([t for t in toks_set if t not in historical_vocab])
        novelty_count = len(new_tokens)
        total_unique = len(toks_set) if len(toks_set)>0 else 1
        novelty_ratio = novelty_count / total_unique
        rows.append({
            "doc_index": i,
            "new_tokens": new_tokens,
            "novelty_count": novelty_count,
            "novelty_ratio": novelty_ratio,
            "total_unique_tokens": total_unique
        })
        all_new_tokens.extend([t for t in toks if t not in historical_vocab])
    counter = Counter(all_new_tokens)
    df_top_new = pd.DataFrame(counter.most_common(top_k), columns=["token", "count"])
    df_docs = pd.DataFrame(rows)
    return df_docs, df_top_new

# -----------------------
# UI
# -----------------------
st.title("🔎 Text Trends Analyzer — spaCy opcional")
st.write("Upload CSV (colunas: Title, Abstract, Date...) → limpeza, n-grams, TF-IDF, novelty. Lematização opcional com spaCy.")

uploaded_file = st.file_uploader("1) Carregue o CSV", type=["csv"])
if uploaded_file is None:
    st.info("Carregue um CSV para começar.")
    st.stop()

# read CSV (robust)
try:
    df = pd.read_csv(uploaded_file, encoding="utf-8", engine="python")
except Exception as e:
    st.error(f"Erro ao ler CSV: {e}")
    st.stop()

st.write(f"Colunas: {', '.join(df.columns)}")
title_col = st.selectbox("Coluna Title", options=list(df.columns), index=0)
abstract_col = st.selectbox("Coluna Abstract", options=list(df.columns), index=min(1, len(df.columns)-1))
date_col = st.selectbox("Coluna Date (opcional)", options=[None]+list(df.columns), index=0)

combine_texts = st.checkbox("Combinar Title + Abstract", value=True)

st.markdown("**Stopwords adicionais (separe por vírgula)**")
extra_stop_text = st.text_area("ex.: study,patients", value="study,patients")
extra_stopwords = [w.strip().lower() for w in extra_stop_text.split(",") if w.strip()]

# spaCy option
st.markdown("**Lematização (opcional, usa spaCy)**")
use_spacy = st.checkbox("Ativar lematização com spaCy (requer spaCy + modelo instalado)", value=False)
if use_spacy:
    if not SPACY_AVAILABLE:
        st.warning("spaCy (pacote) não está instalado no ambiente. O requirements.txt do repositório inclui spaCy e o modelo — verifique o deploy logs.")
    else:
        # try load model (if not already)
        loaded, err = try_load_spacy_model(show_messages=True)
        if loaded:
            st.success("spaCy e modelo carregados — lematização habilitada (parser/NER desativados para economizar memória).")
        else:
            st.error(f"Falha ao carregar modelo spaCy: {err}. O app seguirá sem lematização.")
            use_spacy = False

# historical vocab para novelty
st.markdown("**Vocabulário histórico (opcional) — .txt**")
vocab_file = st.file_uploader("Upload .txt (1 token por linha)", type=["txt"])
if vocab_file is not None:
    historical_vocab = {line.strip().lower() for line in vocab_file.read().decode("utf-8").splitlines() if line.strip()}
    st.success(f"Vocabulário histórico carregado: {len(historical_vocab)} tokens")
else:
    historical_vocab = {"study","patient","patients","result","method","methods","data","analysis","model"}
    st.info("Nenhum vocabulário carregado: usando um vocab exemplo pequeno (recomendo carregar seu vocab histórico real).")

# n-gram choices
ngram_choice = st.selectbox("N-grams (frequência)", options=["1-gram","1-2-3 grams","1-2 grams"], index=1)
if ngram_choice == "1-gram":
    ngram_range = (1,1)
elif ngram_choice == "1-2 grams":
    ngram_range = (1,2)
else:
    ngram_range = (1,3)

tfidf_choice = st.selectbox("TF-IDF n-grams", options=["1-gram","1-2 grams","1-3 grams"], index=0)
tfidf_ngram = (1,1) if tfidf_choice=="1-gram" else ((1,2) if tfidf_choice=="1-2 grams" else (1,3))

top_k = st.slider("Top termos", min_value=10, max_value=200, value=50, step=10)

run = st.button("▶️ Rodar análise")
if not run:
    st.stop()

# prepare texts
if combine_texts:
    df['text_raw'] = df[title_col].fillna('').astype(str) + ". " + df[abstract_col].fillna('').astype(str)
else:
    df['text_raw'] = df[abstract_col].fillna('').astype(str)

texts = df['text_raw'].tolist()

# preprocess (spaCy or simple)
if use_spacy and SPACY_AVAILABLE and nlp is not None:
    cleaned_docs, token_lists = preprocess_texts_spacy(texts, extra_stopwords)
else:
    cleaned_docs, token_lists = preprocess_texts_simple(texts, extra_stopwords)

# stats
st.subheader("Estatísticas iniciais")
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Documentos", len(df))
with col2:
    avg_tokens = np.mean([len(toks) for toks in token_lists]) if len(token_lists)>0 else 0
    st.metric("Tokens médios por doc", f"{avg_tokens:.1f}")
with col3:
    vocab_size = len(set([t for toks in token_lists for t in toks]))
    st.metric("Vocabulário único (corpus atual)", vocab_size)

# ngrams freq
st.subheader("N-grams — frequências")
df_ngrams = compute_ngrams_counts(cleaned_docs, ngram_range=ngram_range, top_k=top_k)
st.dataframe(df_ngrams, use_container_width=True)
if not df_ngrams.empty:
    chart = alt.Chart(df_ngrams.reset_index()).mark_bar().encode(
        x=alt.X('count:Q'),
        y=alt.Y('term:N', sort='-x')
    ).properties(height=min(600, 25*len(df_ngrams)))
    st.altair_chart(chart, use_container_width=True)

# TF-IDF
st.subheader("TF-IDF")
df_tfidf, tf_vectorizer, tfidf_matrix = compute_tfidf(cleaned_docs, ngram_range=tfidf_ngram, top_k=top_k)
st.dataframe(df_tfidf, use_container_width=True)

# novelty
st.subheader("Detecção de novelty")
df_docs_novel, df_top_new = novelty_detection(token_lists, historical_vocab, top_k=top_k)
st.markdown("Top tokens novos (fora do vocab histórico):")
st.dataframe(df_top_new, use_container_width=True)
st.markdown("Amostra novelty por documento:")
st.dataframe(df_docs_novel.head(20), use_container_width=True)

# export results
st.subheader("Exportar resultados")
if st.button("Gerar arquivo Excel com resultados"):
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="xlsxwriter") as writer:
        df_ngrams.to_excel(writer, sheet_name="ngrams", index=False)
        df_tfidf.to_excel(writer, sheet_name="tfidf", index=False)
        df_top_new.to_excel(writer, sheet_name="novelty_top_tokens", index=False)
        df_docs_novel.to_excel(writer, sheet_name="novelty_per_doc", index=False)
    st.download_button("Download results.xlsx", data=buf.getvalue(), file_name="text_analysis_results.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

st.markdown("---")
st.write("Observações: se você for usar com muitos documentos no Streamlit Cloud, prefira desabilitar componentes pesados e use nlp.pipe() (como já fiz aqui).")
