"""
Module 2 — Paper Classifier

Classifies a paper abstract into a scientific category using Claude.
Demonstrates two approaches side by side:
  - zero_shot: model uses its own knowledge of category definitions
  - few_shot:  model is given labelled examples to anchor its definitions

Run:
    python classify.py
"""

import json
import os
import anthropic
from dotenv import load_dotenv

load_dotenv()

MODEL = "claude-haiku-4-5-20251001"

# Fixed set of categories (loosely based on arXiv taxonomy)
CATEGORIES = [
    "cs.AI",       # Artificial Intelligence (reasoning, knowledge representation)
    "cs.LG",       # Machine Learning (learning algorithms, neural networks)
    "cs.CL",       # Computation and Language (NLP, text processing)
    "cs.CV",       # Computer Vision
    "quant-ph",    # Quantum Physics
    "hep-ph",      # High Energy Physics - Phenomenology
    "astro-ph",    # Astrophysics
    "cond-mat",    # Condensed Matter Physics
    "q-bio",       # Quantitative Biology
    "eess.SP",     # Signal Processing
]

CATEGORY_LIST_STR = "\n".join(f"  - {c}" for c in CATEGORIES)

# Few-shot examples: one per ambiguous boundary (cs.LG vs cs.CL is the classic one)
FEW_SHOT_EXAMPLES = [
    {
        "abstract": "We propose a new method for training deep neural networks on image classification tasks using residual connections. Our approach achieves state-of-the-art accuracy on ImageNet.",
        "category": "cs.LG",
        "keywords": ["deep learning", "residual networks", "image classification"],
        "reasoning": "Core contribution is a learning algorithm/architecture for neural networks.",
    },
    {
        "abstract": "We present a transformer-based model for machine translation that achieves human parity on WMT benchmarks. The model is trained end-to-end on parallel corpora.",
        "category": "cs.CL",
        "keywords": ["machine translation", "transformer", "NLP"],
        "reasoning": "Primary task is language — translation between natural languages.",
    },
    {
        "abstract": "Using observations from the Hubble Space Telescope, we measure the expansion rate of the universe and constrain dark energy models.",
        "category": "astro-ph",
        "keywords": ["cosmology", "dark energy", "Hubble constant"],
        "reasoning": "Observational astronomy and cosmology.",
    },
    {
        "abstract": "We demonstrate quantum advantage on a 53-qubit superconducting processor by sampling from a random quantum circuit distribution.",
        "category": "quant-ph",
        "keywords": ["quantum computing", "quantum supremacy", "superconducting qubits"],
        "reasoning": "Quantum hardware and computation.",
    },
]

OUTPUT_SCHEMA = """{
  "category": "one of the categories above",
  "confidence": "high | medium | low",
  "keywords": ["2-4 topic keywords"],
  "reasoning": "one sentence explaining the classification"
}"""


def build_zero_shot_prompt(abstract: str) -> str:
    return f"""Classify the following scientific paper abstract into exactly one of these categories:

{CATEGORY_LIST_STR}

Return a JSON object with this schema:
{OUTPUT_SCHEMA}

Return ONLY the JSON object. No explanation, no markdown.

Abstract:
{abstract}"""


def build_few_shot_prompt(abstract: str) -> str:
    examples_str = ""
    for ex in FEW_SHOT_EXAMPLES:
        example_output = json.dumps({
            "category": ex["category"],
            "keywords": ex.get("keywords", []),
            "reasoning": ex["reasoning"],
        }, indent=2)
        examples_str += f"\nAbstract: {ex['abstract']}\nClassification: {example_output}\n---"

    return f"""Classify the following scientific paper abstract into exactly one of these categories:

{CATEGORY_LIST_STR}

Here are examples of correct classifications:
{examples_str}

Now classify this abstract. Return a JSON object with this schema:
{OUTPUT_SCHEMA}

Return ONLY the JSON object. No explanation, no markdown.

Abstract:
{abstract}"""


def classify(abstract: str, mode: str, client: anthropic.Anthropic) -> dict | None:
    prompt = build_zero_shot_prompt(abstract) if mode == "zero_shot" else build_few_shot_prompt(abstract)

    response = client.messages.create(
        model=MODEL,
        max_tokens=300,
        messages=[{"role": "user", "content": prompt}],
    )
    raw = response.content[0].text.strip()

    if raw.startswith("```"):
        lines = raw.splitlines()
        raw = "\n".join(lines[1:-1] if lines[-1] == "```" else lines[1:])

    try:
        result = json.loads(raw)
        if result.get("category") not in CATEGORIES:
            result["_warning"] = f"Category '{result.get('category')}' not in approved list"
        return result
    except json.JSONDecodeError:
        print(f"    [error] Could not parse response: {raw[:200]}")
        return None


def print_comparison(label: str, zero: dict | None, few: dict | None) -> None:
    print(f"\n{'=' * 60}")
    print(f"Sample: {label.upper()}")
    print(f"{'=' * 60}")

    for mode_name, result in [("ZERO-SHOT", zero), ("FEW-SHOT", few)]:
        print(f"\n  [{mode_name}]")
        if result is None:
            print("    Failed to parse")
            continue
        if "_warning" in result:
            print(f"    WARNING: {result['_warning']}")
        print(f"    Category   : {result.get('category')}")
        print(f"    Confidence : {result.get('confidence')}")
        print(f"    Keywords   : {result.get('keywords')}")
        print(f"    Reasoning  : {result.get('reasoning')}")

    if zero and few and zero.get("category") != few.get("category"):
        print(f"\n  *** DISAGREEMENT: zero-shot={zero.get('category')} vs few-shot={few.get('category')}")


if __name__ == "__main__":
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise EnvironmentError("ANTHROPIC_API_KEY not set — add it to your .env file")

    client = anthropic.Anthropic(api_key=api_key)

    abstracts = {
        "transformer (attention)": (
            "The dominant sequence transduction models are based on complex recurrent or "
            "convolutional neural networks. We propose the Transformer, based solely on "
            "attention mechanisms, dispensing with recurrence entirely. Experiments on "
            "machine translation show these models to be superior in quality."
        ),
        "BERT (language model)": (
            "We introduce BERT, a language representation model designed to pre-train deep "
            "bidirectional representations from unlabeled text. The pre-trained BERT model "
            "can be fine-tuned for question answering, language inference, and other NLP tasks."
        ),
        "federated learning": (
            "We study federated optimization for training ML models across many devices "
            "without sharing raw data. We propose the FedAvg algorithm and show it works "
            "well even with non-iid data across clients. Privacy preservation is the primary motivation."
        ),
        "quantum supremacy": (
            "We use a processor with programmable superconducting qubits to create quantum "
            "states on 53 qubits. Our Sycamore processor samples a quantum circuit in 200 "
            "seconds — a task that would take a classical supercomputer approximately 10,000 years."
        ),
    }

    for label, abstract in abstracts.items():
        zero = classify(abstract, "zero_shot", client)
        few = classify(abstract, "few_shot", client)
        print_comparison(label, zero, few)
