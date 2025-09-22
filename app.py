import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.decomposition import NMF

st.set_page_config(page_title="Topic Explorer", layout="wide")

st.title("🔍 Topic Modeling Explorer")

st.markdown("Faça upload de um arquivo CSV com uma coluna de texto (ex.: abstracts).")

# Upload
uploaded_file = st.file_uploader("Escolha um arquivo CSV", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    # Selecionar coluna de texto
    text_col = st.selectbox("Selecione a coluna de texto", df.columns)

    if st.button("Analisar"):
        texts = df[text_col].dropna().astype(str)

        # Vetorização com ajustes compatíveis
        vectorizer = TfidfVectorizer(
            stop_words=ENGLISH_STOP_WORDS,
            ngram_range=(1, 2),   # unigrams + bigrams
            min_df=0.01,          # ignora termos que aparecem em menos de 1% dos docs
            max_df=0.85           # ignora termos muito comuns
        )
        X = vectorizer.fit_transform(texts)

        # NMF para tópicos
        n_topics = st.slider("Número de tópicos", 2, 15, 5)
        nmf = NMF(n_components=n_topics, random_state=42)
        W = nmf.fit_transform(X)
        H = nmf.components_

        terms = vectorizer.get_feature_names_out()

        st.subheader("📌 Principais tópicos")
        for topic_idx, topic in enumerate(H):
            top_terms = [terms[i] for i in topic.argsort()[:-11:-1]]
            st.markdown(f"**Tópico {topic_idx+1}:** {', '.join(top_terms)}")

        # Mostrar dataframe com tópicos atribuídos
        df["Tópico Principal"] = W.argmax(axis=1)
        st.subheader("📊 Textos classificados por tópico")
        st.dataframe(df[[text_col, "Tópico Principal"]])

        # Download
        csv_download = df.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Baixar resultados CSV", csv_download, "resultados.csv", "text/csv")
