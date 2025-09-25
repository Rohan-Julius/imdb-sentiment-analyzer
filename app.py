import re
import streamlit as st
from transformers import pipeline, AutoTokenizer

st.set_page_config(page_title="IMDb Sentiment + Sarcasm + Summary", page_icon="🎬")

@st.cache_resource  # cache heavy resources across reruns [web:218][web:219]
def load_models():
    sentiment = pipeline(
        "text-classification",
        model="rohan10juli/roBERT-imdb-sentiment-tuned",
        device_map="auto"
    )  # [web:83]

    sarcasm_t5 = pipeline(
        "text2text-generation",
        model="mrm8488/t5-base-finetuned-sarcasm-twitter",
        device_map="auto"
    )  # [web:141]

    summarizer = pipeline(
        "summarization",
        model="facebook/bart-large-cnn",
        device_map="auto"
    )  # [web:101]
    tok_sum = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")  # [web:101]
    return sentiment, sarcasm_t5, summarizer, tok_sum

sentiment_model, sarcasm_t5, summarizer, tok_sum = load_models()

def clean_text(t: str) -> str:
    t = re.sub(r"<br\s*/?>", " ", str(t), flags=re.IGNORECASE)
    t = re.sub(r"\s+", " ", t).strip()
    return t

def _smart_lengths(text: str, ratio=(0.30, 0.60), caps=(20, 120)):
    n = len(tok_sum(text)["input_ids"])
    min_len = max(caps[0], int(ratio[0] * n))
    max_len = max(min_len + 5, min(caps[1], int(ratio[1] * n)))
    return min_len, max_len  # [web:101]

def _tidy_summary(s: str) -> str:
    s = re.sub(r"\s+", " ", s).strip()
    s = re.sub(r"(^\s*[a-z])", lambda m: m.group(0).upper(), s)
    s = re.sub(r"([.!?]\s+)([a-z])", lambda m: m.group(1) + m.group(2).upper(), s)
    if not re.search(r"[.!?]$", s):
        s += "."
    return s

def summarize_clean(text: str, ratio=(0.30, 0.60), caps=(20, 120)) -> str:
    min_len, max_len = _smart_lengths(text, ratio=ratio, caps=caps)
    out = summarizer(
        text,
        min_length=min_len,
        max_length=max_len,
        do_sample=False,
        num_beams=4,
        length_penalty=2.0,
        no_repeat_ngram_size=3
    )  # [web:83][web:101]
    return _tidy_summary(out[0]["summary_text"])

SARC_CUES = [
    r"\byeah right\b", r"\bas if\b", r"\bthanks for nothing\b",
    r"\bjust what i needed\b", r"\bso great\b.*\bnot\b", r"\bof course\b.*\bnot\b"
]  # [web:324]

def _rule_sarcasm(text: str) -> bool:
    t = text.lower()
    return any(re.search(p, t) for p in SARC_CUES)  # [web:324]

def detect_sarcasm_sentence_vote(text: str, frac_threshold: float = 0.25) -> str:
    sents = re.split(r'(?<=[.!?])\s+', text.strip())
    sents = [s for s in sents if s]
    if not sents:
        return "NOT_SARCASTIC"
    outs = sarcasm_t5(
        sents, max_length=3, do_sample=False, num_beams=1, temperature=0.01, top_k=1, top_p=1.0
    )  # [web:83][web:141]
    sarcastic = 0
    for o in outs:
        raw = o["generated_text"].strip().lower()
        if raw.startswith("derison") or raw == "derision":
            sarcastic += 1
    frac = sarcastic / max(1, len(sents))
    return "SARCASTIC" if frac >= frac_threshold else "NOT_SARCASTIC"

def detect_sarcasm_label(text: str) -> str:
    t = re.sub(r"\s+", " ", text).strip()
    label = detect_sarcasm_sentence_vote(t, frac_threshold=0.25)  # [web:141]
    if label == "NOT_SARCASTIC" and _rule_sarcasm(t):
        label = "SARCASTIC"  # [web:324]
    return label

st.title("🎬 IMDb Review Sentiment App")
st.write("Checks sarcasm, predicts sentiment (fine‑tuned), and generates a concise summary of reviews.")  # [web:219]

user_input = st.text_area("✍️ Enter your movie review:", height=220, placeholder="Paste an IMDb-style review here...")

col1, col2 = st.columns([1, 1])
with col1:
    run_btn = st.button("Analyze", type="primary")
with col2:
    clear_btn = st.button("Clear")

if clear_btn:
    st.experimental_rerun()

if run_btn and user_input.strip():
    text = clean_text(user_input)

    st.subheader("🔹 Analysis")

    with st.spinner("Checking sarcasm..."):
        sarcasm_label = detect_sarcasm_label(text)  # [web:141][web:324]
    if sarcasm_label == "SARCASTIC":
        st.error("⚠️ Sarcasm detected. Sentiment may be unreliable on sarcastic text.")

    with st.spinner("Predicting sentiment..."):
        s_out = sentiment_model(text)[0]  # {'label','score'} [web:83]
        sent_label = s_out["label"]
        sent_score = s_out.get("score")

    if "pos" in sent_label.lower():
        st.success(f"✅ Sentiment: {sent_label}")  
    elif "neg" in sent_label.lower():
        st.error(f"❌ Sentiment: {sent_label}")   
    else:
        st.info(f"ℹ️ Sentiment: {sent_label}")    


    with st.spinner("Generating summary..."):
        summary = summarize_clean(text)  # [web:101]

    st.subheader("📌 Summary")
    st.write(summary)
