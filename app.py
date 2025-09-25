import re
import streamlit as st
from transformers import pipeline, AutoTokenizer
import time

st.set_page_config(
    page_title="🎬 IMDb Sentiment Analyzer",
    page_icon="🎭",
    layout="wide"
)

# Custom CSS for IMDb theme
st.markdown("""
<style>
/* ... your existing CSS unchanged ... */
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;500;700&display=swap');
    .main-header { font-family: 'Roboto', sans-serif; font-size: 2.5rem; font-weight: 700; color: #F5C518; background-color: #000000; text-align: center; padding: 1rem; border-radius: 8px; margin-bottom: 2rem; border: 2px solid #F5C518; }
    .imdb-container { background-color: #1a1a1a; color: #ffffff; padding: 1.5rem; border-radius: 8px; border: 1px solid #333333; margin: 1rem 0; }
    .result-card { background-color: #2a2a2a; border: 1px solid #404040; border-radius: 8px; padding: 1.5rem; margin: 1rem 0; color: #ffffff; }
    .positive-result { border-left: 4px solid #5cb85c; background-color: #1e3a1e; }
    .negative-result { border-left: 4px solid #d9534f; background-color: #3a1e1e; }
    .sarcastic-result { border-left: 4px solid #f0ad4e; background-color: #3a2e1e; }
    .stButton > button { background-color: #F5C518; color: #000000; border: none; border-radius: 4px; padding: 0.5rem 1.5rem; font-weight: 500; font-family: 'Roboto', sans-serif; }
    .stButton > button:hover { background-color: #e6b800; }
    .stTextArea > div > div > textarea { background-color: #2a2a2a; color: #ffffff; border: 1px solid #404040; border-radius: 4px; font-family: 'Roboto', sans-serif; }
    .sample-button { background-color: #404040; color: #ffffff; border: 1px solid #666666; border-radius: 4px; padding: 0.5rem 1rem; margin: 0.25rem; font-family: 'Roboto', sans-serif; cursor: pointer; }
    .sample-button:hover { background-color: #555555; }
    .quick-tabs { display: flex; gap: 1rem; margin: 1rem 0; flex-wrap: wrap; }
    .tab-button { background-color: #404040; color: #ffffff; border: 1px solid #666666; border-radius: 4px; padding: 0.5rem 1rem; font-family: 'Roboto', sans-serif; cursor: pointer; transition: all 0.3s ease; }
    .tab-button:hover { background-color: #F5C518; color: #000000; }
    .summary-box { background-color: #2a2a2a; border: 1px solid #F5C518; border-radius: 8px; padding: 1rem; margin: 1rem 0; color: #ffffff; }
    .mixed-badge { display:inline-block; margin-left:8px; padding:2px 8px; border-radius:10px; background:#F5C518; color:#000; font-weight:600; font-size:0.85rem; }
</style>
""", unsafe_allow_html=True)

if 'selected_sample' not in st.session_state:
    st.session_state.selected_sample = None

@st.cache_resource
def load_models():
    """Load all ML models"""
    with st.spinner("🎬 Loading IMDb AI models..."):
        sentiment = pipeline(
            "text-classification",
            model="rohan10juli/roBERT-imdb-sentiment-tuned",
            device_map="auto"
        )
        sarcasm_t5 = pipeline(
            "text2text-generation",
            model="mrm8488/t5-base-finetuned-sarcasm-twitter",
            device_map="auto"
        )
        summarizer = pipeline(
            "summarization",
            model="facebook/bart-large-cnn",
            device_map="auto"
        )
        tok_sum = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")
    return sentiment, sarcasm_t5, summarizer, tok_sum

# Helper functions
def clean_text(t: str) -> str:
    t = re.sub(r"<br\s*/?>", " ", str(t), flags=re.IGNORECASE)
    t = re.sub(r"\s+", " ", t).strip()
    return t

def _smart_lengths(text: str, ratio=(0.30, 0.60), caps=(20, 120)):
    n = len(tok_sum(text)["input_ids"])
    min_len = max(caps[0], int(ratio[0] * n))
    max_len = max(min_len + 5, min(caps[1], int(ratio[1] * n)))
    return min_len, max_len

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
    )
    return _tidy_summary(out[0]["summary_text"])

SARC_CUES = [
    r"\byeah right\b", r"\bas if\b", r"\bthanks for nothing\b",
    r"\bjust what i needed\b", r"\bso great\b.*\bnot\b", r"\bof course\b.*\bnot\b"
]

def _rule_sarcasm(text: str) -> bool:
    t = text.lower()
    return any(re.search(p, t) for p in SARC_CUES)

def detect_sarcasm_sentence_vote(text: str, frac_threshold: float = 0.25) -> tuple:
    sents = re.split(r'(?<=[.!?])\s+', text.strip())
    sents = [s for s in sents if s]
    if not sents:
        return "NOT_SARCASTIC", 0.0
    outs = sarcasm_t5(
        sents, max_length=3, do_sample=False, num_beams=1, temperature=0.01, top_k=1, top_p=1.0
    )
    sarcastic = 0
    for o in outs:
        raw = o["generated_text"].strip().lower()
        if raw.startswith("derison") or raw == "derision":
            sarcastic += 1
    frac = sarcastic / max(1, len(sents))
    label = "SARCASTIC" if frac >= frac_threshold else "NOT_SARCASTIC"
    return label, frac

def detect_sarcasm_label(text: str) -> tuple:
    t = re.sub(r"\s+", " ", text).strip()
    label, confidence = detect_sarcasm_sentence_vote(t, frac_threshold=0.25)
    if label == "NOT_SARCASTIC" and _rule_sarcasm(t):
        label = "SARCASTIC"
        confidence = 0.8
    return label, confidence

# Mixed-review detection 

def _sentence_split(t: str):
    return [s for s in re.split(r'(?<=[.!?])\s+', t.strip()) if s]

def review_sentiment_mixed(text: str, pos_thresh=0.20, neg_thresh=0.20, neutral_margin=0.10) -> tuple:
    """
    Returns ('MIXED'|'POSITIVE'|'NEGATIVE', details_dict)
    details_dict = {'fpos': float, 'fneg': float, 'pos': int, 'neg': int, 'n_used': int}
    """
    sents = _sentence_split(text)
    if not sents:
        return "NEGATIVE", {'fpos': 0.0, 'fneg': 0.0, 'pos': 0, 'neg': 0, 'n_used': 0}
    preds = sentiment_model(sents)
    pos = neg = 0
    used = 0
    for p in preds:
        label = p["label"].lower()
        score = p.get("score", 0.5)
        if (0.5 - neutral_margin) < score < (0.5 + neutral_margin):
            continue
        used += 1
        if "pos" in label:
            pos += 1
        elif "neg" in label:
            neg += 1
    n = max(1, pos + neg)
    fpos, fneg = pos / n, neg / n
    if fpos >= pos_thresh and fneg >= neg_thresh:
        overall = "MIXED"
    else:
        overall = "POSITIVE" if fpos > fneg else "NEGATIVE"
    return overall, {'fpos': fpos, 'fneg': fneg, 'pos': pos, 'neg': neg, 'n_used': used}

st.markdown('<h1 class="main-header">🎬 IMDb Sentiment Analyzer</h1>', unsafe_allow_html=True)

sentiment_model, sarcasm_t5, summarizer, tok_sum = load_models()

st.markdown('<div class="imdb-container">', unsafe_allow_html=True)
st.markdown("### Quick Sample Reviews")

sample_reviews = {
    "Positive Blockbuster": "The opening twenty minutes glide by on confident filmmaking—clean geography, generous wide shots, and a score that swells without smothering. The lead performance anchors every turn; small gestures pay off later as the character’s choices tighten the plot. Jokes land because they grow out of behavior, not mugging, and the action escalates with purpose rather than noise. By the finale, stakes feel personal and the resolution is cathartic without tipping into syrup. It’s mainstream entertainment built with craft and heart.",
    "Negative Disaster": "The film never finds its rhythm. Key scenes are rushed, character motivations are underdeveloped, and the middle act wanders through subplots that don’t affect the outcome. Action sequences are loud but incoherent, with choppy editing that obscures basic geography. By the finale, conflicts are resolved through convenient twists rather than earned choices, leaving a flat, unsatisfying experience.",
    "Sarcastic Review": "Oh yes, a true masterpiece—endless exposition, recycled twists, and action scenes that repeat like a broken GIF. The characters posture while the plot sprints in circles, and by the time the villain explains his plan for the third time, the only suspense is whether the credits will save us.",
    "Mixed Feelings": "I honestly don't understand why Test (2025) faced so much rejection. Sure, the storyline had its weak moments, but overall, it was engaging enough to hold my attention. What truly stood out were the fantastic performances by R. Madhavan and Siddharth—they brought depth and emotion that really elevated the film. Yes, the plot could have been tighter, and at times it felt like it lacked direction. But the lead actors carried the movie with such conviction that it still worked for me. All in all, I'd say Test is a good one-time watch, especially if you appreciate solid performances. Don't go in expecting a masterpiece—but it's definitely not as bad as some are making it out to be."
}

cols = st.columns(len(sample_reviews))
for i, (title, review) in enumerate(sample_reviews.items()):
    with cols[i]:
        if st.button(title, key=f"sample_{i}", use_container_width=True):
            st.session_state.selected_sample = review

st.markdown('</div>', unsafe_allow_html=True)

# Input section
st.markdown('<div class="imdb-container">', unsafe_allow_html=True)
st.markdown("### ✍️ Enter Your Movie Review")

default_text = st.session_state.selected_sample if st.session_state.selected_sample else ""

user_input = st.text_area(
    "Movie Review:",
    height=200,
    placeholder="Enter your movie review here... (or click a sample above)",
    value=default_text,
    help="Paste any movie review to analyze its sentiment and detect sarcasm"
)

# Action buttons
col1, col2 = st.columns([3, 1])
with col1:
    analyze_btn = st.button(" Analyze Review", type="primary", use_container_width=True)
with col2:
    if st.button(" Clear", use_container_width=True):
        st.session_state.selected_sample = None
        st.rerun()

st.markdown('</div>', unsafe_allow_html=True)

# Analysis section
if analyze_btn and user_input.strip():
    text = clean_text(user_input)
    with st.spinner(" Analyzing your review..."):
        sarcasm_label, sarcasm_confidence = detect_sarcasm_label(text)
        s_out = sentiment_model(text)[0]
        sent_label = s_out["label"]
        sent_score = s_out.get("score", 0)
        summary = summarize_clean(text)
        mixed_label, mixed_meta = review_sentiment_mixed(text)  # NEW

    # Results display
    st.markdown("## Analysis Results")
    col1, col2 = st.columns(2)

     
    with col1:
        badge = f'<span class="mixed-badge">MIXED</span>' if mixed_label == "MIXED" else ""
        if "pos" in sent_label.lower():
            st.markdown(f'''
            <div class="result-card positive-result">
                <h3>😊 Sentiment: POSITIVE {badge}</h3>
                <p>This review expresses positive feelings about the movie.</p>
            </div>
            ''', unsafe_allow_html=True)
        else:
            st.markdown(f'''
            <div class="result-card negative-result">
                <h3>😞 Sentiment: NEGATIVE {badge}</h3>
                <p>This review expresses negative feelings about the movie.</p>
            </div>
            ''', unsafe_allow_html=True)

    # Sarcasm result 
    with col2:
      THRESHOLD = 0.50  

      if sarcasm_confidence >= THRESHOLD:
          st.markdown(f'''
          <div class="result-card sarcastic-result">
              <h3>🎭 Sarcasm: DETECTED</h3>
              <p>⚠️ This review contains sarcasm. The sentiment analysis may not reflect the true opinion.</p>
          </div>
          ''', unsafe_allow_html=True)
      else:
          st.markdown(f'''
          <div class="result-card">
              <h3>✅ Sarcasm: NOT DETECTED</h3>
              <p>The review appears to be straightforward without sarcasm.</p>
          </div>
          ''', unsafe_allow_html=True)


    st.markdown(f'''
    <div class="summary-box">
        <h3>📝 Review Summary</h3>
        <p style="font-size: 1.1em; line-height: 1.6;">{summary}</p>
    </div>
    ''', unsafe_allow_html=True)

st.markdown("---")
st.markdown('''
<div style="text-align: center; color: #888; padding: 1rem; font-family: 'Roboto', sans-serif;">
    <p><strong> Powered by Advanced AI Models</strong></p>
    <p>Sentiment: RoBERTa-IMDb | Sarcasm: T5-Sarcasm | Summary: BART-CNN</p>
</div>
''', unsafe_allow_html=True)
