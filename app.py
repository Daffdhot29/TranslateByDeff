from pathlib import Path

import streamlit as st

from model.inference import load_translator, translate_beam


BASE_DIR = Path(__file__).resolve().parent
CHECKPOINT = BASE_DIR / "model" / "best_seq2seq.pt"
TOKENIZER = BASE_DIR / "tokenizer" / "spm.model"


st.set_page_config(
    page_title="TranslateByDeff",
    page_icon="🌐",
    layout="wide",
)

st.markdown("""
<style>
.block-container {
    max-width: 1100px;
    padding-top: 3rem;
}

.hero {
    text-align: center;
    margin-bottom: 2rem;
}

.hero h1 {
    font-size: 2.5rem;
    margin-bottom: .3rem;
}

.hero p {
    opacity: .72;
}

.output-box {
    border: 1px solid rgba(128, 128, 128, .25);
    border-radius: 10px;
    padding: 1rem;
    min-height: 170px;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
    <h1>TranslateByDeff</h1>
    <p>Indonesian → English LSTMCell Machine Translation</p>
</div>
""", unsafe_allow_html=True)


@st.cache_resource
def get_translator():
    return load_translator(CHECKPOINT, TOKENIZER)


try:
    model, sp, device = get_translator()
except Exception as exc:
    st.error(f"Model gagal dimuat: {exc}")
    st.stop()


left, right = st.columns(2, gap="large")

with left:
    st.subheader("🇮🇩 Indonesian")
    source = st.text_area(
        "Teks sumber",
        placeholder="Contoh: saya ingin belajar machine learning",
        height=180,
        label_visibility="collapsed",
    )

with right:
    st.subheader("🇬🇧 English")
    result_placeholder = st.empty()


if st.button("Translate", type="primary", use_container_width=True):
    if not source.strip():
        st.warning("Masukkan kalimat Bahasa Indonesia terlebih dahulu.")
    else:
        with st.spinner("Translating..."):
            result = translate_beam(model, source, sp, device)

        result_placeholder.markdown(
            f'<div class="output-box">{result}</div>',
            unsafe_allow_html=True,
        )
else:
    result_placeholder.markdown(
        '<div class="output-box" style="opacity:.55">'
        'Translation will appear here.'
        '</div>',
        unsafe_allow_html=True,
    )


st.divider()
st.caption("Translate Machine by Deff | Keyword : SentencePiece • LSTM • Attention • Beam Search")