import os
import re
from typing import List, Tuple, Dict
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import gradio as gr
import evaluate
import math

MODEL_NAME = os.environ.get("SUMM_MODEL", "facebook/bart-large-cnn")
MAX_NEW_TOKENS = 180
MIN_NEW_TOKENS = 40
CHUNK_TOKEN_TARGET = 900
CHUNK_OVERLAP = 120
TARGET_SENTENCES = (3, 5)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
device = 0 if torch.cuda.is_available() else -1
summarizer = pipeline("summarization", model=model, tokenizer=tokenizer, device=device)

_raw_model_max = getattr(tokenizer, "model_max_length", None)
if not _raw_model_max or _raw_model_max > 100000:
    SAFE_MAX_INPUT_LEN = min(CHUNK_TOKEN_TARGET, 4096)
else:
    SAFE_MAX_INPUT_LEN = min(CHUNK_TOKEN_TARGET, _raw_model_max)

rouge_metric = evaluate.load("rouge")


def sent_split(text: str) -> List[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]


def enforce_sentence_count(text: str, lo: int, hi: int) -> str:
    sents = sent_split(text)
    if len(sents) == 0:
        return text.strip()
    if len(sents) <= hi:
        return " ".join(sents)
    return " ".join(sents[:max(lo, min(hi, len(sents)))])


def chunk_by_tokens(text: str, token_budget: int, overlap: int) -> List[str]:
    input_ids = tokenizer.encode(text, add_special_tokens=False)
    chunks = []
    i = 0
    n = len(input_ids)
    while i < n:
        j = min(i + token_budget, n)
        chunk_ids = input_ids[i:j]
        chunks.append(tokenizer.decode(chunk_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True))
        if j >= n:
            break
        i = j - overlap
        if i < 0:
            i = 0
    return chunks


def safe_truncate_text(text: str, max_tokens: int) -> str:
    ids = tokenizer.encode(text, add_special_tokens=False)
    if len(ids) <= max_tokens:
        return text
    return tokenizer.decode(ids[:max_tokens], skip_special_tokens=True, clean_up_tokenization_spaces=True)


def safe_summarize_single_input(text: str) -> str:
    try:
        res = summarizer(
            text,
            max_new_tokens=MAX_NEW_TOKENS,
            min_new_tokens=MIN_NEW_TOKENS,
            do_sample=False,
        )
        if not res:
            return ""
        return res[0].get("summary_text", "") or ""
    except Exception as e:
        return f"[Summarization error: {type(e).__name__}: {str(e)}]"


def summarize_long(text: str) -> str:
    text = text.strip()
    if not text:
        return ""
    try:
        total_tokens = len(tokenizer.encode(text, add_special_tokens=False))
    except Exception:
        total_tokens = math.inf
    if total_tokens <= CHUNK_TOKEN_TARGET:
        small_input = safe_truncate_text(text, SAFE_MAX_INPUT_LEN)
        out = safe_summarize_single_input(small_input)
        return enforce_sentence_count(out, *TARGET_SENTENCES)
    chunks = chunk_by_tokens(text, CHUNK_TOKEN_TARGET, CHUNK_OVERLAP)
    sub_summaries: List[str] = []
    for c in chunks:
        c_trunc = safe_truncate_text(c, SAFE_MAX_INPUT_LEN)
        s = safe_summarize_single_input(c_trunc)
        sub_summaries.append(s)
    fused = "\n".join([s for s in sub_summaries if s])
    fused_trunc = safe_truncate_text(fused, SAFE_MAX_INPUT_LEN)
    fused_sum = safe_summarize_single_input(fused_trunc)
    return enforce_sentence_count(fused_sum, *TARGET_SENTENCES)


def summarize_batch(texts: List[str]) -> List[str]:
    results = []
    for t in texts:
        results.append(summarize_long(t))
    return results


def summarize_multi(input_blob: str) -> str:
    parts = [p.strip() for p in re.split(r"\n-{3,}\n", input_blob.strip()) if p.strip()]
    outs = summarize_batch(parts)
    return "\n\n".join(outs)


with gr.Blocks(title="AI Text Summarizer with ROUGE") as demo:
    gr.Markdown(
        "# AI Text Summarizer\nSummarizes long documents "
    )
    with gr.Tab("Single"):
        inp = gr.Textbox(label="Input text to summarize", lines=12, placeholder="Paste an article or paper section…")
        out = gr.Textbox(label="Summarized text (3–5 sentences)", lines=6)
        btn = gr.Button("Summarize")
        btn.click(fn=lambda x: summarize_long(x), inputs=inp, outputs=out)
    with gr.Tab("Batch"):
        gr.Markdown("Paste multiple articles separated by a line with `---` on its own.")
        inp_b = gr.Textbox(label="Multiple articles", lines=18, placeholder="Article 1...\n---\nArticle 2...")
        out_b = gr.Textbox(label="Summaries (each 3–5 sentences)", lines=18)
        btn_b = gr.Button("Summarize Batch")
        btn_b.click(fn=summarize_multi, inputs=inp_b, outputs=out_b)
    
    with gr.Accordion("Debug / Warnings", open=False):
        gr.Markdown(
        f"- SAFE_MAX_INPUT_LEN={SAFE_MAX_INPUT_LEN}\n"
        f"- CHUNK_TOKEN_TARGET={CHUNK_TOKEN_TARGET}\n"
        f"- CHUNK_OVERLAP={CHUNK_OVERLAP}\n"
        "- Truncation warning: inputs longer than SAFE_MAX_INPUT_LEN are pre-truncated using the model tokenizer to avoid ambiguous pipeline truncation behavior.\n"
        "- Potential runtime issues: large inputs can still cause high memory usage; reduce CHUNK_TOKEN_TARGET or batch size, or run on a GPU with sufficient memory.\n\n"
        "---\n"
        "© **Vivek Reddy**  \n"
        "🔗 [GitHub](https://github.com/vivekreddy1105) | [LinkedIn](https://linkedin.com/in/vivekreddy1105)"
    )


if __name__ == "__main__":
    demo.launch()
