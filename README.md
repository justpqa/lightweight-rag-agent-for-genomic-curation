# 🧬 Lightweight RAG / Agent for Genomic Curation

## 📘 Overview
This project implements a **lightweight Retrieval-Augmented Generation (RAG)** system to answer biomedical questions from a **local corpus of text snippets**.  
It retrieves relevant evidence, generates concise summaries, and cites sources — all using **CPU-only, open-source models** with **zero cost**.

---

## ⚙️ Setup Instructions

### 1. Clone Repository

```bash
git clone https://github.com/justpqa/lightweight_rag_agent_for_genomic_curation.git
cd lightweight_rag_agent_for_genomic_curation
```

### 1'. Create virtual environments (to make sure install packages were separate from local machines)

```bash
python3 -m venv venv
```

(or python or py depends on your python version)

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Running demo of the application

```bash
python3 main.py
```

(or python or py depends on your python version)
Note: 

- chroma_db folder should not be deleted since it will hold the vector database for the RAG application
- corpus folder should not be deleted since it will hold the corpus of documents (but you can also add new documents into this and when we run the application, you can choose option 1 to reindexing the vector db). 
- In addition to this, for the first use, it will download several models (which should take at most 1GB for all 3 models, and you can update models in `.env` if you prefer more lightweight models). => you can also download models from HuggingFace to local machine and change the path of model in `.env` into the folder path

## 🧩 Model + embedding choices

In this RAG applications, we will need 3 main components: 

- Embeddings Model: all-MiniLM-L6-v2 => lightweight, easy to use with CPU, but also one of the best among lightweight embeddings models space
- LLM: ibm-granite/granite-4.0-h-350m => lightweight model to use on CPU, modern model and from a bigger family of model (where distillation from bigger model and same training process as bigger model would allow better generation) allow better generation, and the h version of the granite-4.0 allow better handling of long context, which is crucial when we incorporate text citations into LLM prompt
- Reranker: cross-encoder/ms-marco-MiniLM-L-6-v2 => lightweight, one of best overall trade-off between accuracy and speed. Excellent for re-ranking top-k results in RAG.

However, with the design of the application to allow plugging of different models (in HuggingFace for examples), we can try to plugging different models choice at each of the 3 steps to allow for better result

## 🔧 Retrieval settings (k, chunk size, threshold)

In this RAG application, the retrieval settings is set to be as follow:

- top_k (Number of documents from initial retrieval): 20, high number of initial retrieval for more rerank candidate
- top_k_rerank (Number of documents after reranked from similarity search, often smaller than top_k): 5, less candidate after reranking to prevent the model from handling too much unnecessary context
- chunk_size: 500, for small enough chunks for prompt of smaller size but still have enough information
- chunk_size_overlap: 100, for decent amount but not too much overlap 
- similarity score threshold: 0.1 - a decent amount that allow for finding many possible choice while still keep things relevant before applying later reranking
- Temperature: 0.1 -> small enough for generation that require consistency like in biomedical research

## 💬 Example Query and Output

```
>Q: APOE variant and Alzheimer's disease
>A: The provided citations discuss the role of the APOE variant in Alzheimer's disease, highlighting its association with the disease and its genetic heterogeneity. The study by koran et al. (2017) and the meta-analysis by leekhochosun et al. (2017) further support this finding, showing that the apolipoprotein E (APOE) gene, particularly the rs405509 variant, is a significant genetic risk factor for Alzheimer's disease. The study also highlights that the tt allele of rs405509 synergizes with APOE epsilon4 in the impairment of cognition and its underlying default mode network in non-demented elderly.

Sources:
Citation 1 (nihms-1579247.pdf):
d, bryden l 2007 a high-density whole-genome association study reveals that apoe is the major susceptibility gene for sporadic late-onset alzheimers disease. the journal of clinical psychiatry 68 613618. pubmed 17474819 cornes bk, brody ja, nikpoor n, morrison ac, dang hcp, ahn bs, wang s, dauriz m, barzilay ji, dupuis j 2014 association of levels of fasting glucose and insulin with rare variants at the

Citation 2 (nihms-1051989.pdf):
using alzheimers disease genetics consortium data. we showed a moderate genetic correlation rg 0.64 between the two age groups, supporting genetic heterogeneity. heritability explained by variants on chromosome 19 harboring apoe was significantly larger in younger than in older onset group p 0.05. apoe region, bin1, or2s2, ms4a4e and picalm were identified at the gene-based genome-wide significance p 2.73106 with larger effects at younger age

Citation 3 (awz206.pdf):
2005 koran et al., 2017. notably, the apolipoprotein e apoe gene, which is the strongest genetic risk factor for alzheimers disease shows a stronger association among females compared to males, particularly between ages 65 and 75 years neu et al., 2017. despite growing evidence of sex differences in the genetic drivers of alzheimers dis- ease deming et al., 2018, limited work has systematically explored sex-specic genetic associations with alzheimers

Citation 4 (jcm-08-01236.pdf):
xiao, h. gao, y. liu, l. li, y. association between polymorphisms in the promoter region of the apolipoprotein e apoe gene and alzheimers disease a meta-analysis. excli j. 2017, 16, 921938. crossref 69. ma, c. zhang, y. li, x. chen, y. zhang, j. liu, z. chen, k. zhang, z. the tt allele of rs405509 synergizes with apoe epsilon4 in the impairment of cognition and its underlying default mode network in non-demented elderly. curr. alzheimer res. 2016, 13, 708717. crossref 70.

Citation 5 (jcm-08-01236.pdf):
department of neural development and disease, korea brain research institute, daegu 41062, korea correspondence leekhochosun.ac.kr tel. 82-62-230-6246 received 5 july 2019 accepted 12 august 2019 published 16 august 2019 abstract variants in the apoe gene region may explain ethnic dierences in the association of alzheimers disease ad with 4. ethnic dierences in allele frequencies for three apoe region snps
```

## 📊 Evaluation and analysis

I have test 4 different configurations of (chunk_size, top_k, top_k_rerank) across 10 different questions in queries.jsonl and the results are as follow (here, I measured based on question-answer cosine similarity, as well as max/min similarity of citation with answer (to make sure if answer aligned with citation)):

| chunk_size | top_k | top_k_rerank | question_answer_similarity | citation_answer_max_similarity | citation_answer_min_similarity |
|------------|--------------|----------|------------|--------------|----------|
| 500 | 20 | 5 | 0.822992 | 0.805708 | 0.582623 | 
| 500 | 50 | 5 | 0.790218 | 0.834320 | 0.517048 | 
| 500 | 50 | 10 | 0.778366 | 0.812251 | 0.520168 | 
| 1000 | 20 | 5 | 0.777831	 | 0.783236 | 0.539775 | 

=> A good amount of top_k and top_k_rerank but not too much would work well in many cases.

Some example score on best configuration (only show first 5 questions)
| chunk_size | top_k | top_k_rerank | query | answer |
|------------|--------------|----------|------------|--------------|
| 500 | 20 | 5 | What is known about APOE and Alzheimer’s disease? | The provided citations discuss the role of APOE in Alzheimer's disease, highlighting its association with the disease and its impact on the genetic landscape. |
| 500 | 20 | 5 | How do TREM2 variants influence the risk of Alzheimer’s disease? | "The provided citations highlight the role of TREM2 variants in the risk of Alzheimer's disease. A comprehensive study identified five new genome-wide associations for load and showed that genetic pleiotropy or co-morbidity of these traits with pathology of load. The study also highlights the enrichment of rare variants in pathways associated with Alzheimer's disease, indicating that additional rare variants remain to be identified, and larger" |
| 500 | 20 | 5 | What evidence links ABCA7 to Alzheimer’s pathology? | The provided citations discuss the role of the ABCA7 gene in Alzheimer's disease pathology. Specifically, they highlight its expression in Alzheimer's disease versus control brains, evaluation of transcriptomic expression in Alzheimer's disease tissues, and gene cluster pathway analyses. The study also identifies rare abca7 variants in two German families with Alzheimer's disease and provides evidence linking abca7 expression to increased cerebral amyloid-beta accumulation in the J20 mouse model of Alzheimer's disease.| 
| 500 | 20 | 5 | Are there genetic loci associated with early-onset Alzheimer’s disease? | A genome-wide association study (GWAS) has identified several genetic loci associated with the risk of late-onset Alzheimer's disease. These loci, including abca7, ms4a6ams4a4e, epha1, cd33, and cd2ap, have been found to be associated with the disease. Additionally, the hla-dqa10301hla-dqb10302 haplotype has also been found to be not associated with Alzheimer's disease. These findings suggest that multiple genetic loci may contribute to the development of Alzheimer's disease, highlighting the complexity of the disease and the need for further research to better understand its pathophysiology |
| 500 | 20 | 5 | Which biological pathways are most commonly implicated in Alzheimer’s according to recent genomic studies? | Recent genomic studies have identified several biological pathways most commonly implicated in Alzheimer's disease (AD). These pathways include the neuronal and synaptic function pathways, as well as cell types such as monocytes, macrophages, and brain-derived neurotrophic factor (BDNF). The study by Chung et al. (2019) highlights the role of genes like cr1, abca7, bin1, spi1, ms4a2, ms4a6a, and ms4a4a in AD, with these genes being expressed in brain regions of AD patients. The findings suggest that these genes are involved in neuronal and synaptic function, as well as in the development and progression of AD. | 

## 💰 Cost & Efficiency Notes

- With our current model choice and everything can be handled and download to used locally, the cost of our RAG pipeline is 0

## 💡 Future direction

- Incorporate separate pipeline for automated download of corpus of papers => allow easier extraction of richer metadata
- Experiments with different models (with a high focus on lightweight model, especially sub-1B model on LLM) and configuration of generation