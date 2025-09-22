# app.py
import streamlit as st
import pandas as pd
import numpy as np
import re
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, ENGLISH_STOP_WORDS
import altair as alt
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import io
from itertools import combinations
import datetime

st.set_page_config(page_title="Text Trends Analyzer (Full)", layout="wide")
st.title("🔎 Text Trends Analyzer — Enhanced MVP")

# -------------------------
# Default historical vocab
# -------------------------
DEFAULT_VOCAB = {
    "study","patient","patients","results","conclusion","method","methods","background",
    "objective","aim","participants","randomized","control","group","groups","significant",
    "analysis","data","clinical","treatment","age","years","percent","increase","decrease",
    "observed","reported","compared","model","measure","followup","outcome","sample"
}

# -------------------------
# Helpers (preprocess / ngrams / tfidf / novelty / cooccurrence / bursts)
# -------------------------
def simple_tokenize(text):
    if pd.isna(text):
        return []
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
    cleaned_docs = []
    token_lists = []
    for t in texts:
        toks = [tok for tok in simple_tokenize(t) if tok not in stopset]
        token_lists.append(toks)
        cleaned_docs.append(" ".join(toks))
    return cleaned_docs, token_lists

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

def cooccurrence_matrix(token_lists, top_terms, window=None):
    # builds cooccurrence counts across docs (document co-occurrence)
    term_idx = {t:i for i,t in enumerate(top_terms)}
    mat = np.zeros((len(top_terms), len(top_terms)), dtype=int)
    for toks in token_lists:
        toks_set = set(toks)
        present = [t for t in top_terms if t in toks_set]
        for a,b in combinations(present, 2):
            i = term_idx[a]; j = term_idx[b]
            mat[i,j] += 1
            mat[j,i] += 1
    return pd.DataFrame(mat, index=top_terms, columns=top_terms)

def burst_detection(df_time, cleaned_docs, terms, freq_window='M'):
    # df_time: dataframe with parsed_date column (datetime) and cleaned (string)
    # terms: list of terms to monitor
    # returns dataframe with term, last_count, prev_mean, growth_ratio
    rows = []
    if df_time['parsed_date'].isna().all():
        return pd.DataFrame(columns=["term","last_count","prev_mean","growth_ratio"])
    df_time = df_time.copy()
    df_time['period'] = df_time['parsed_date'].dt.to_period(freq_window).dt.to_timestamp()
    for t in terms:
        g = df_time[df_time['cleaned'].str.contains(rf'\b{re.escape(t)}\b')].groupby('period').size().rename('count')
        if g.empty:
            continue
        g = g.reindex(pd.period_range(df_time['period'].min().to_period(freq_window), df_time['period'].max().to_period(freq_window)).to_timestamp(), fill_value=0)
        # last period vs previous mean
        last = int(g.iloc[-1])
        prev_mean = float(g.iloc[:-1].mean()) if len(g)>1 else 0.0
        growth = (last / prev_mean) if prev_mean>0 else (float('inf') if last>0 else 0.0)
        rows.append({"term": t, "last_count": last, "prev_mean": prev_mean, "growth_ratio": growth})
    df_bursts = pd.DataFrame(rows).sort_values("growth_ratio", ascending=False)
    return df_bursts

# -------------------------
# UI: Tabs & Upload
# -------------------------
tabs = st.tabs(["Upload & Config","Frequencies","TF-IDF","Novelty","Temporal","Co-occurrence","Summary & Export"])
tab_upload, tab_freq, tab_tfidf, tab_novel, tab_time, tab_cooc, tab_sum = tabs

with tab_upload:
    st.header("Upload & Configuration")
    st.write("Carregue um CSV com colunas `Title`, `Abstract`, `Date` (opcional).")
    uploaded_file = st.file_uploader("CSV file", type=["csv"])
    if uploaded_file is None:
        st.info("Você pode carregar um CSV ou usar o exemplo. Clique em 'Load example dataset' para testar rapidamente.")
        if st.button("Load example dataset"):
            # small example
            data = {
                "Title":[
                    "Semaglutide improves weight and glycemic control in type 2 diabetes",
                    "New CAR-T approach shows promise in lymphoma patients",
                    "SGLT2 inhibitors reduce cardiovascular events in heart failure"
                ],
                "Abstract":[
                    "Randomized trial shows ...",
                    "Phase 1/2 study reports ...",
                    "Meta-analysis of trials indicates ..."
                ],
                "Date":["2024-06-01","2024-05-20","2024-04-10"]
            }
            df = pd.DataFrame(data)
            st.success("Example dataset loaded.")
        else:
            st.stop()
    else:
        try:
            df = pd.read_csv(uploaded_file, encoding="utf-8", engine="python")
            st.success(f"CSV loaded: {len(df)} rows")
        except Exception as e:
            st.error(f"Erro ao ler CSV: {e}")
            st.stop()

    st.markdown("### Columns")
    cols = list(df.columns)
    title_col = st.selectbox("Title column", options=cols, index=0)
    abstract_col = st.selectbox("Abstract column", options=cols, index=min(1, len(cols)-1))
    date_col = st.selectbox("Date column (optional)", options=[None]+cols, index=0)
    combine_texts = st.checkbox("Combine Title + Abstract into single text", value=True)

    st.markdown("### Stopwords / historical vocab")
    extra_stop_text = st.text_area("Extra stopwords (comma-separated)", value="study,patients,trial")
    extra_stopwords = [w.strip().lower() for w in extra_stop_text.split(",") if w.strip()]
    vocab_file = st.file_uploader("Optional: upload historical vocab (.txt one token per line)", type=["txt"])
    if vocab_file is not None:
        historical_vocab = {line.strip().lower() for line in vocab_file.read().decode("utf-8").splitlines() if line.strip()}
        st.success(f"Historical vocab loaded: {len(historical_vocab)} tokens")
    else:
        historical_vocab = DEFAULT_VOCAB.copy()
        st.info("No vocab uploaded — using example small vocab. You can upload a larger vocab to improve novelty detection.")

    st.markdown("### Performance controls")
    max_docs = st.number_input("Max documents to process (sample for speed)", min_value=10, max_value=10000, value=min(500, len(df)), step=10)
    run_btn = st.button("▶️ Run full analysis")

# stop until run pressed
if 'df' not in locals():
    st.stop()

if not run_btn:
    st.info("Configure parameters and press 'Run full analysis' to compute results.")
    st.stop()

# Slice / prepare rows
df = df.head(int(max_docs)).copy()
if combine_texts:
    df['text_raw'] = df[title_col].fillna('').astype(str) + ". " + df[abstract_col].fillna('').astype(str)
else:
    df['text_raw'] = df[abstract_col].fillna('').astype(str)

# Preprocess
with st.spinner("Preprocessing texts..."):
    cleaned_docs, token_lists = preprocess_texts(df['text_raw'].tolist(), extra_stopwords)

# Basic stats
with st.container():
    st.subheader("Dataset summary")
    c1,c2,c3 = st.columns(3)
    c1.metric("Documents", len(df))
    avg_tokens = np.mean([len(t) for t in token_lists]) if token_lists else 0
    c2.metric("Avg tokens/doc", f"{avg_tokens:.1f}")
    c3.metric("Unique vocab (corpus)", len(set([t for toks in token_lists for t in toks])))

# -------------------------
# Frequencies tab
# -------------------------
with tab_freq:
    st.header("N-grams / Frequencies")
    ngram_choice = st.selectbox("N-gram range", options=["1","1-2","1-3"], index=2)
    if ngram_choice=="1":
        ngram_range=(1,1)
    elif ngram_choice=="1-2":
        ngram_range=(1,2)
    else:
        ngram_range=(1,3)
    top_k = st.slider("Top K terms", min_value=10, max_value=200, value=50, step=10)
    df_ngrams = compute_ngrams_counts(cleaned_docs, ngram_range=ngram_range, top_k=top_k)
    st.dataframe(df_ngrams, use_container_width=True)
    if not df_ngrams.empty:
        chart = alt.Chart(df_ngrams.reset_index()).mark_bar().encode(
            x=alt.X('count:Q'),
            y=alt.Y('term:N', sort='-x')
        ).properties(height=min(600, 25*len(df_ngrams)))
        st.altair_chart(chart, use_container_width=True)
    # wordcloud of top unigrams
    st.subheader("Wordcloud (top unigrams)")
    uni_list = []
    for toks in token_lists:
        uni_list.extend([t for t in toks if len(t.split())==1])
    if uni_list:
        wc = WordCloud(width=800, height=400, background_color='white').generate(" ".join(uni_list))
        fig, ax = plt.subplots(figsize=(10,4.5))
        ax.imshow(wc, interpolation='bilinear'); ax.axis('off')
        st.pyplot(fig)
    else:
        st.info("No unigram tokens to show.")

# -------------------------
# TF-IDF tab
# -------------------------
with tab_tfidf:
    st.header("TF-IDF Analysis")
    tfidf_choice = st.selectbox("TF-IDF ngram range", options=["1","1-2","1-3"], index=0)
    tfidf_ngram = (1,1) if tfidf_choice=="1" else ((1,2) if tfidf_choice=="1-2" else (1,3))
    df_tfidf, tf_vectorizer, tfidf_matrix = compute_tfidf(cleaned_docs, ngram_range=tfidf_ngram, top_k=top_k)
    st.dataframe(df_tfidf, use_container_width=True)
    # top TF-IDF terms bar
    if not df_tfidf.empty:
        chart = alt.Chart(df_tfidf.reset_index()).mark_bar().encode(
            x='tfidf:Q',
            y=alt.Y('term:N', sort='-x')
        ).properties(height=min(600, 25*len(df_tfidf)))
        st.altair_chart(chart, use_container_width=True)

# -------------------------
# Novelty tab
# -------------------------
with tab_novel:
    st.header("Novelty Detection (out-of-historical-vocab)")
    df_docs_novel, df_top_new = novelty_detection(token_lists, historical_vocab, top_k=top_k)
    st.subheader("Top new tokens in corpus (outside historical vocab)")
    st.dataframe(df_top_new, use_container_width=True)
    st.subheader("Per-document novelty sample (first 30 docs)")
    st.dataframe(df_docs_novel.head(30), use_container_width=True)
    # allow merge new tokens into historical vocab on demand
    st.markdown("Add selected new tokens to historical vocab:")
    to_add = st.multiselect("Select tokens to add", options=df_top_new['token'].tolist(), default=[])
    if st.button("Merge selected tokens into historical vocab"):
        historical_vocab = set(list(historical_vocab) + list(to_add))
        st.success(f"Added {len(to_add)} tokens. New historical vocab size: {len(historical_vocab)}")

# -------------------------
# Temporal tab
# -------------------------
with tab_time:
    st.header("Temporal analysis & Burst detection")
    if date_col is None:
        st.info("No date column configured. Configure it in Upload tab to use temporal analysis.")
    else:
        df['parsed_date'] = pd.to_datetime(df[date_col], errors='coerce')
        if df['parsed_date'].notna().sum()==0:
            st.info("Date column could not be parsed; check formats (e.g., YYYY-MM-DD).")
        else:
            st.subheader("Top terms time series")
            df_temp = df.copy()
            df_temp['cleaned'] = cleaned_docs
            # choose top terms from ngrams
            top_terms = df_ngrams['term'].tolist()[:50]
            picked = st.multiselect("Pick terms for time series", options=top_terms, default=top_terms[:5])
            if picked:
                rows = []
                df_temp['month'] = df_temp['parsed_date'].dt.to_period('M').dt.to_timestamp()
                for t in picked:
                    g = df_temp[df_temp['cleaned'].str.contains(rf'\b{re.escape(t)}\b', na=False)].groupby('month').size().rename('count')
                    tmp = g.reset_index()
                    tmp['term'] = t
                    rows.append(tmp)
                if rows:
                    ts_df = pd.concat(rows, ignore_index=True).fillna(0)
                    chart = alt.Chart(ts_df).mark_line(point=True).encode(
                        x='month:T', y='count:Q', color='term:N'
                    ).properties(height=300).interactive()
                    st.altair_chart(chart, use_container_width=True)
            st.subheader("Burst detection (growth in last period vs previous mean)")
            freq_window = st.selectbox("Frequency window", options=["M","W"], index=0)
            # evaluate bursts on top TF-IDF terms
            monitor_terms = df_tfidf['term'].tolist()[:50] if not df_tfidf.empty else df_ngrams['term'].tolist()[:50]
            bursts = burst_detection(df_temp.assign(parsed_date=df_temp['parsed_date']), cleaned_docs, monitor_terms, freq_window)
            if not bursts.empty:
                st.dataframe(bursts.head(50), use_container_width=True)
            else:
                st.info("No bursty terms detected in current data.")

# -------------------------
# Co-occurrence tab
# -------------------------
with tab_cooc:
    st.header("Co-occurrence (terms that appear together by document)")
    top_n = st.slider("Top N terms to include in matrix", min_value=10, max_value=100, value=30, step=5)
    top_terms = df_ngrams['term'].tolist()[:top_n]
    if not top_terms:
        st.info("No terms to compute co-occurrence.")
    else:
        cooc_df = cooccurrence_matrix(token_lists, top_terms)
        st.subheader("Co-occurrence matrix (document co-occurrence counts)")
        st.dataframe(cooc_df, use_container_width=True)
        # heatmap with altair
        cooc_melt = cooc_df.reset_index().melt(id_vars='index')
        cooc_melt.columns = ['term_x','term_y','count']
        chart = alt.Chart(cooc_melt).mark_rect().encode(
            x=alt.X('term_x:N', sort=top_terms),
            y=alt.Y('term_y:N', sort=top_terms),
            color='count:Q',
            tooltip=['term_x','term_y','count']
        ).properties(height=600, width=600)
        st.altair_chart(chart, use_container_width=True)

# -------------------------
# Summary & Export tab
# -------------------------
with tab_sum:
    st.header("Executive Summary & Export")
    # Executive summary components
    st.subheader("Top insights")
    top_freq = df_ngrams.head(10).to_dict(orient='records')
    top_tfidf = df_tfidf.head(10).to_dict(orient='records') if not df_tfidf.empty else []
    top_new = df_top_new.head(10).to_dict(orient='records') if not df_top_new.empty else []
    st.markdown("**Top frequency terms (top 10)**")
    st.write(pd.DataFrame(top_freq))
    st.markdown("**Top TF-IDF terms (top 10)**")
    st.write(pd.DataFrame(top_tfidf))
    st.markdown("**Top novel tokens (top 10)**")
    st.write(pd.DataFrame(top_new))

    # create downloadable excel
    if st.button("Generate export (Excel + summary.md)"):
        buf = io.BytesIO()
        with pd.ExcelWriter(buf, engine="xlsxwriter") as writer:
            df_ngrams.to_excel(writer, sheet_name="ngrams", index=False)
            df_tfidf.to_excel(writer, sheet_name="tfidf", index=False)
            df_top_new.to_excel(writer, sheet_name="novelty_top", index=False)
            df_docs_novel.to_excel(writer, sheet_name="novelty_per_doc", index=False)
            df[['text_raw']].to_excel(writer, sheet_name="raw_text", index=False)
        excel_data = buf.getvalue()
        st.download_button("⬇️ Download Excel results", data=excel_data, file_name="text_trends_results.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

        # markdown summary
        md = []
        md.append(f"# Executive Summary — generated {datetime.datetime.utcnow().isoformat()} UTC")
        md.append("## Top frequency terms")
        for r in top_freq:
            md.append(f"- {r['term']}: {r['count']}")
        md.append("## Top TF-IDF terms")
        for r in top_tfidf:
            md.append(f"- {r['term']}: {r['tfidf']:.4f}")
        md.append("## Top novel tokens")
        for r in top_new:
            md.append(f"- {r['token']}: {r['count']}")
        md_text = "\n".join(md)
        st.download_button("⬇️ Download summary (MD)", data=md_text, file_name="summary.md", mime="text/markdown")

st.markdown("---")
st.write("End of analysis. Suggestions: run on a monthly snapshot, export vocab, iterate pipeline.")
