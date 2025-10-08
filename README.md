# VeritasFi
An Adaptable, Multi-tiered RAG Framework for Multi-modal Financial Question Answering

## Abstract
Retrieval-Augmented Generation (RAG) is becoming increasingly
essential for Question Answering (QA) in the financial sector, where
accurate and contextually grounded insights from complex public
disclosures are crucial. However, existing financial RAG systems
face two significant challenges:  they struggle to process hetero-
geneous data formats, such as text, tables, and figures; and  they
encounter difficulties in balancing general-domain applicability
with company-specific adaptation. To overcome these challenges,
we present VeritasFi, an innovative hybrid RAG framework that
incorporates a multi-modal preprocessing pipeline alongside a
cutting-edge two-stage training strategy for its re-ranking compo-
nent. VeritasFi enhances financial QA through three key innova-
tions:  A multi-modal preprocessing pipeline that seamlessly
transforms heterogeneous data into a coherent, machine-readable
format. A tripartite hybrid retrieval engine that operates in
parallel, combining deep multi-path retrieval over a semantically
indexed document corpus, real-time data acquisition through tool
utilization, and an expert-curated memory bank for high-frequency
questions, ensuring comprehensive scope, accuracy, and efficiency.
 A two-stage training strategy for the document re-ranker,
which initially constructs a general, domain-specific model us-
ing anonymized data, followed by rapid fine-tuning on company-
specific data for targeted applications. By integrating our proposed
designs, VeritasFi presents a groundbreaking framework that
greatly enhances the adaptability and robustness of financial RAG
systems, providing a scalable solution for both general-domain and
company-specific QA tasks.

[](https://pic1.imgdb.cn/item/68e632fbc5157e1a885cfba0.png)
