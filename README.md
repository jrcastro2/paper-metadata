# Paper Metadata

Two small LLM modules for processing scientific paper abstracts, built as portfolio pieces alongside a [RAG pipeline](https://github.com/jrcastro2/rag-paper-assistant) and a [tool-calling agent](https://github.com/jrcastro2/paper-assistant).

**Stack:** Python · Claude (Anthropic API)

---

## Modules

### Module 1 — Metadata Extractor (`extract.py`)

Takes raw paper text (title + abstract, or an unstructured block) and extracts structured metadata as JSON: title, authors, affiliations, year, keywords, and identifiers (DOI, arXiv ID).

Demonstrates how to make an LLM return reliable structured output:
- Schema embedded in the prompt as an explicit contract
- Rules against guessing or inventing values (`null` for missing fields)
- Markdown-fence stripping for the most common output failure mode
- Post-parse type validation without external libraries
- One retry on JSON parse failure

Works on clean, partial, and messy inputs — including informal text with no explicit field labels.

```
$ python extract.py

Sample: MESSY
{
  "title": "Communication-Efficient Learning of Deep Networks",
  "authors": ["McMahan et al."],
  "affiliations": ["Google"],
  "year": 2017,
  "keywords": ["federated learning", "FedAvg", "privacy", ...],
  "identifiers": { "doi": null, "arxiv_id": null }
}
```

### Module 2 — Paper Classifier (`classify.py`)

Classifies a paper abstract into a scientific category (cs.LG, cs.CL, quant-ph, q-bio, etc.) and returns keywords and a confidence level.

Demonstrates two prompting strategies side by side:

| Strategy | How it works | When to use |
|---|---|---|
| **Zero-shot** | Give the model the category list only | Categories are self-explanatory; fast baseline |
| **Few-shot** | Add labelled examples to the prompt | Ambiguous boundaries between similar categories |

```
$ python classify.py

Sample: FEDERATED LEARNING
  [ZERO-SHOT]  cs.LG  (high)  — proposes a machine learning training algorithm
  [FEW-SHOT]   cs.LG  (high)  — core contribution is a learning methodology
```

---

## Key concepts

**Structured output with LLMs** — LLMs return free-form text by default. To get predictable JSON, embed the schema directly in the prompt as a contract, add explicit rules (no extra fields, `null` for unknowns), and validate the parsed output. This is the standard approach before constrained decoding or tool use.

**Zero-shot vs few-shot** — Zero-shot relies on the model's pre-trained knowledge of category semantics. Few-shot lets you define your own boundary between similar categories by showing examples. The right choice: zero-shot first, add few-shot only where you observe systematic misclassification.

**Confidence routing** — The classifier returns `high / medium / low` confidence. This isn't a calibrated probability — it's a useful signal for routing: auto-accept high-confidence results, send low-confidence ones to human review. This gives a practical human-in-the-loop system without building an active learning pipeline.

---

## Setup

```bash
git clone <repo>
cd paper-metadata
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# add your ANTHROPIC_API_KEY to .env
```

## Run

```bash
python extract.py    # metadata extraction on 3 sample inputs (clean / partial / messy)
python classify.py   # classification with zero-shot vs few-shot comparison
```

---

## Failure modes worth knowing

| Module | Failure | Cause | Guard |
|---|---|---|---|
| Extractor | Malformed JSON | Model wraps in markdown, truncates | Strip fences, retry, raise `max_tokens` |
| Extractor | Wrong types | Year as `"2017"` instead of `2017` | Post-parse type check |
| Extractor | Hallucinated values | Model invents a DOI | Explicit "use null, don't guess" in prompt |
| Classifier | Out-of-set category | Model returns `"cs.NE"` not in your list | Validate against fixed category list |
| Classifier | Boundary misclassification | cs.LG vs cs.CL ambiguity | Add few-shot examples for that pair |
