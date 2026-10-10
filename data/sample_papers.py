"""
Sample paper texts for testing the extractor and classifier.
Includes clean, partial, and messy inputs.
"""

# Clean: well-structured with all fields present
CLEAN = """
Title: Attention Is All You Need

Authors: Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones,
Aidan N. Gomez, Łukasz Kaiser, Illia Polosukhin

Affiliations: Google Brain, Google Research, University of Toronto

Abstract: The dominant sequence transduction models are based on complex recurrent or
convolutional neural networks that include an encoder and a decoder. The best performing
models also connect the encoder and decoder through an attention mechanism. We propose a
new simple network architecture, the Transformer, based solely on attention mechanisms,
dispensing with recurrence and convolutions entirely. Experiments on two machine
translation tasks show these models to be superior in quality while being more
parallelizable and requiring significantly less time to train.

Published: 2017
arXiv ID: 1706.03762
DOI: 10.48550/arXiv.1706.03762
Keywords: transformer, attention mechanism, neural machine translation, sequence transduction
"""

# Partial: no explicit labels, identifiers embedded in prose
PARTIAL = """
BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding

Jacob Devlin, Ming-Wei Chang, Kenton Lee, Kristina Toutanova (Google AI Language)

We introduce a new language representation model called BERT, which stands for
Bidirectional Encoder Representations from Transformers. Unlike recent language
representation models, BERT is designed to pre-train deep bidirectional representations
from unlabeled text by jointly conditioning on both left and right context in all layers.
The pre-trained BERT model can be fine-tuned with just one additional output layer.
We achieved state-of-the-art results on eleven NLP tasks. See arxiv.org/abs/1810.04805
for the full paper. This work appeared at NAACL 2019.
"""

# Messy: unstructured, inconsistent formatting, partial info, noise
MESSY = """
hey so this is the abstract for our paper on federated learning

basically we look at training ML models across a bunch of devices without sharing raw
data - privacy is the main motivation. we call it federated optimization and propose
the FedAvg algorithm. works pretty well even with non-iid data distribution across
clients. main challenge is communication efficiency.

its by mcmahan et al., google. came out around 2017 i think? might be on arxiv
somewhere, search "communication-efficient learning of deep networks"
no doi that i know of
"""

SAMPLES = [
    ("clean", CLEAN),
    ("partial", PARTIAL),
    ("messy", MESSY),
]
