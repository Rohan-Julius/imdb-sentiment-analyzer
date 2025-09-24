# imdb-sentiment-checker

This repo provides an IMDb review sentiment system with a fine‑tuned RoBERTa classifier, plus two production features: sarcasm awareness and concise summarization within a Streamlit UI loaded via Transformers pipelines.

Method
- Sentiment: the notebook fine‑tunes a RoBERTa‑family sequence classifier on IMDb and exports it via save_pretrained, ensuring a valid config.json and tokenizer for AutoModel/AutoTokenizer loading in the app.
- Also performed fine-tuned distillBERT model whose accuracy is slightly lesser.

Additonal Features:  
- Sarcasm: the app uses a T5 text‑to‑text sarcasm model; to stabilize outputs on long reviews, decoding is deterministic and predictions are aggregated over sentences with lightweight lexical backstops.
- Summarization: BART‑CNN generates summaries with beam search and tokenizer‑aware min/max lengths to avoid clipped or rambling outputs; summaries are lightly tidied for casing and final punctuation.

Why sarcasm and summary:  
Sarcasm can invert sentiment cues, so a detector helps contextualize confidence and warn when sentiment may mislead, while a short summary improves readability for long reviews in the UI.

Streamlit app :  
The app caches all pipelines with st.cache_resource for fast reruns and loads the pushed model repo id directly via pipeline("text-classification"), displaying color‑coded labels and confidence, a sarcasm warning when triggered, and a cleaned BART summary.

Model results
- The fine‑tuned roBERTa achieves the best validation performance with Accuracy ≈ 0.946 and weighted F1 ≈ 0.946, indicating strong, balanced classification across classes.
- DistilBERT trails slightly at Accuracy ≈ 0.920 and F1 ≈ 0.920, offering a favorable accuracy‑efficiency tradeoff for lighter deployments despite the small gap to roBERTa.
- Classical baselines perform competitively but below Transformers: TF‑IDF + Logistic Regression reaches Accuracy/F1 ≈ 0.914, while TF‑IDF + Linear SVM reaches ≈ 0.920, confirming feature‑engineered linear models are solid but outperformed by contextual encoders.
- Recommendation: use roBERTa for highest quality, use DistilBERT when latency or memory is constrained, and keep TF‑IDF baselines as interpretable references or low‑resource fallbacks
  
Links
- Google colab: https://colab.research.google.com/drive/1Elb_iCFLgrwIcANbZwZihPfNyeBQDqgx#scrollTo=oVl0yNIiRQNd 
- Hugging face model: https://huggingface.co/rohan10juli/roBERT-imdb-sentiment-tuned
