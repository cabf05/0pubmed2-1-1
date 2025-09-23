import streamlit as st
import pandas as pd
import re
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer

# -------------------------
# Configuração inicial
# -------------------------
st.set_page_config(page_title="Analisador Biomédico de n-grams", layout="wide")
st.title("🔬 Analisador Biomédico de n-grams com Filtro Inteligente")

# -------------------------
# Upload de dados
# -------------------------
file = st.file_uploader("Carregue um CSV com abstracts (coluna: 'abstract')", type=["csv"])
if file:
    df = pd.read_csv(file)
    if "abstract" not in df.columns:
        st.error("O CSV precisa ter uma coluna chamada 'abstract'")
        st.stop()
else:
    st.info("Faça upload de um CSV para começar.")
    st.stop()

# -------------------------
# Pré-processamento de textos
# -------------------------
texts = df["abstract"].fillna("").astype(str).tolist()

# Stopwords médicas genéricas para filtrar
medical_stopwords = set([
    "trial", "control", "conclusion", "treatment", "significant", "data",
    "baseline", "adverse", "event", "study", "patients", "results",
    "randomized", "placebo", "clinical", "methods", "analysis", "groups",
    "follow", "period", "risk", "safety"
])

# Vocabulário histórico (simulação — pode ser carregado de arquivo externo)
historical_vocab = set(["insulin", "glucose", "metformin"])

# -------------------------
# Função para gerar n-grams com filtro inteligente
# -------------------------
def generate_ngrams(texts, n=2, top_k=100):
    vectorizer = TfidfVectorizer(ngram_range=(n, n), stop_words="english", min_df=2)
    X = vectorizer.fit_transform(texts)
    terms = vectorizer.get_feature_names_out()

    # Frequências brutas
    counts = Counter()
    for text in texts:
        tokens = re.findall(r"[a-zA-Z][a-zA-Z0-9\-]+", text.lower())
        ngrams = zip(*[tokens[i:] for i in range(n)])
        for ng in ngrams:
            term = " ".join(ng)
            counts[term] += 1

    # Filtrar n-grams
    filtered = {}
    for term, freq in counts.items():
        first_word = term.split()[0]
        if first_word in medical_stopwords:
            continue  # remove n-grams que começam com stopwords médicas
        if not re.search(r"[A-Za-z0-9]", term):
            continue
        # Heurística: manter termos técnicos (se contiver letras maiúsculas típicas de genes ou se parecer droga)
        if not re.search(r"[A-Z0-9]|mab$|ib$|ine$|ol$", term, re.IGNORECASE):
            continue
        filtered[term] = freq

    df_out = pd.DataFrame(sorted(filtered.items(), key=lambda x: x[1], reverse=True), columns=["term", "freq"])
    return df_out.head(top_k)

# -------------------------
# Execução: gerar n-grams
# -------------------------
with st.spinner("Gerando n-grams filtrados..."):
    df_ngrams = generate_ngrams(texts, n=2, top_k=200)

st.subheader("📊 Top n-grams filtrados")
st.dataframe(df_ngrams, use_container_width=True)

# -------------------------
# Upload de dicionário biomédico
# -------------------------
biomed_file = st.file_uploader("Dicionário biomédico (MeSH/UMLS/genes/drogas) em TXT", type=["txt"])
if biomed_file is not None:
    biomed_dict = {line.strip().lower() for line in biomed_file.read().decode("utf-8").splitlines() if line.strip()}
    st.success(f"Dicionário biomédico carregado: {len(biomed_dict)} termos")

    # ✅ Termos biomédicos validados
    df_known_terms = df_ngrams[df_ngrams['term'].isin(biomed_dict)].reset_index(drop=True)

    # 🌱 Novos candidatos (fora do dicionário e fora do histórico)
    df_novel_candidates = df_ngrams[
        (~df_ngrams['term'].isin(biomed_dict)) &
        (~df_ngrams['term'].isin(historical_vocab))
    ].reset_index(drop=True)

    st.subheader("✅ Termos biomédicos validados")
    if df_known_terms.empty:
        st.info("Nenhum termo biomédico conhecido encontrado.")
    else:
        st.dataframe(df_known_terms, use_container_width=True)

    st.subheader("🌱 Candidatos novos (fora do dicionário e histórico)")
    if df_novel_candidates.empty:
        st.info("Nenhum candidato novo detectado.")
    else:
        st.dataframe(df_novel_candidates, use_container_width=True)

else:
    st.info("Nenhum dicionário biomédico carregado — mostrando apenas análise padrão.")
