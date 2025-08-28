

#  AI-Powered Text Summarizer

## **Objective**

This project implements an **AI-powered text summarizer** using transformer-based models. The application can summarize large texts into concise **3–5 sentence summaries** and supports **batch summarization**. It also includes **runtime warnings, truncation safeguards, and an interactive Gradio web interface**.

---

## **Features**

* ✅ Summarizes **long documents** (splitting into chunks when necessary).
* ✅ Ensures summaries are between **3–5 sentences**.
* ✅ Handles **batch summarization** (multiple articles separated by `---`).
* ✅ Includes **truncation safeguards** and **runtime warnings**.
* ✅ Interactive **Gradio interface** with tabs for single and batch input.
* ✅ Debug accordion showing runtime configuration and copyright.
* ✅ Integrated **ROUGE evaluation support** (if dependencies are installed).

---

## **Requirements**

### Python Version

* Python **3.8+**

### Install Dependencies

Create a virtual environment and install dependencies:

```bash
pip install torch transformers gradio evaluate
```

> ⚠️ If ROUGE evaluation fails, install missing metrics:

```bash
pip install absl-py rouge-score
```

---

## **Usage**

### Run the application

```bash
python app.py
```

This launches a **Gradio web app** at `http://127.0.0.1:7860`.

---

### Web Interface

* **Single Tab**: Paste a long article or paper → get a 3–5 sentence summary.
* **Batch Tab**: Paste multiple articles separated by `---` → get separate summaries.

**Debug / Warnings Accordion**:

* Displays truncation and runtime warnings.
* Shows current token settings (`SAFE_MAX_INPUT_LEN`, `CHUNK_TOKEN_TARGET`, `CHUNK_OVERLAP`).
* Includes copyright and author links.

---

## **Implementation Details**

### Model

* Uses Hugging Face’s **facebook/bart-large-cnn** transformer for **abstractive summarization**.
* Loaded with `transformers.pipeline("summarization")`.
* Automatically uses **GPU if available**.

### Preprocessing

1. **Sentence Splitting** → Ensure well-formed summaries.
2. **Chunking** → Long documents split by tokens with overlap.
3. **Safe Truncation** → Ensures inputs never exceed model’s safe max length.

### Summarization Logic

* For short texts → summarize directly.
* For long texts →

  1. Split into chunks.
  2. Summarize each chunk.
  3. Fuse summaries and generate a final short summary.

### Batch Processing

* Multiple texts handled via `summarize_batch()` and `summarize_multi()`.
* Input documents must be separated by a line with `---`.

---

## **Optional: ROUGE Evaluation**

The code integrates Hugging Face’s `evaluate` library to load **ROUGE metrics** for measuring summary quality (requires `rouge-score` package).

Example:

```python
from evaluate import load
rouge = load("rouge")
results = rouge.compute(predictions=[summary], references=[reference])
```

---

## **Deliverables**

* **Code** (`app.py`): Implements the summarizer with Gradio UI.
* **requirements.txt**: Lists dependencies.
* **Documentation**: Explains setup, usage, and implementation.

---

## **Evaluation Criteria**

* ✔️ **Correct functionality**: Summaries constrained to 3–5 sentences.
* ✔️ **NLP model usage**: Transformer-based abstractive summarization.
* ✔️ **Batch support**: Multiple articles handled seamlessly.
* ✔️ **Code quality**: Modular, error-handled, GPU support.
* ✔️ **Documentation clarity**: Setup + usage explained.
* ⭐ Optional: Web interface, runtime warnings, ROUGE integration.

---

## **Author**

* **© Vivek Reddy**
* 🔗 [GitHub](https://github.com/vivekreddy1105)
* 🔗 [LinkedIn](https://linkedin.com/in/vivekreddy1105)

---

Do you want me to also generate a **`requirements.txt`** file for this project so you can drop it into your repo?
