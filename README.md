# Week 4: Advanced Analysis and Reporting of Language Trends using NLP

This project investigates how news vocabulary and thematic emphasis changed over time in the HuffPost News Category Dataset (2012–2022).

## Project files

- `REPORT.md` — complete methodology, findings framework, limitations, and recommendations.
- `src/analyze_language_trends.py` — reproducible temporal NLP pipeline.
- `requirements.txt` — Python dependencies.
- `submission_description.txt` — 200-word internship portal description.
- The formatted Word report is supplied separately as `Week4_Language_Trends_NLP_Report.docx`.

## Research question

How did the vocabulary and thematic emphasis of HuffPost headlines and short descriptions change between 2012 and 2022?

## Dataset

- Hugging Face mirror: [gohelrakesh/news-category](https://huggingface.co/datasets/gohelrakesh/news-category)
- Original dataset: [Kaggle News Category Dataset](https://www.kaggle.com/datasets/rmisra/news-category-dataset)
- Research paper: [Misra (2022)](https://arxiv.org/abs/2209.11429)

**Sampling limitation:** the dataset card reports roughly 200,000 records up to May 2018 and about 10,000 from May 2018 through 2022. Annual phrase and category rates are normalized by document volume, but this cannot fully remove changes in archive coverage or editorial composition.

## Methods

1. Date validation, cleaning, and text normalization.
2. Tokenization and unigram/bigram frequency analysis.
3. Tracked phrase prevalence per 1,000 documents, linear slopes, and Spearman correlation.
4. Latent Dirichlet Allocation (LDA) topic modeling.
5. TF-IDF document representation and MiniBatchKMeans clustering.
6. Annual visualization of article counts, categories, phrases, topics, clusters, headline length, and n-grams.

## Run locally

Python 3.9+ is recommended. Dataset download requires an internet connection.

```bash
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python src/analyze_language_trends.py --out_dir outputs
```

For a quicker functional run:

```bash
python src/analyze_language_trends.py --max_documents 5000 --topic_sample_size 3000 --cluster_sample_size 3000 --out_dir smoke_outputs
```

A local CSV, JSON, or JSONL dataset can be supplied with `--input_file`.

## Generated outputs

The pipeline exports annual profiles, category shares, top n-grams, tracked phrase trends and statistics, LDA topic terms and prevalence, cluster terms and prevalence, run metadata, and seven charts.

Topic and cluster identifiers are unsupervised. Inspect their top terms and representative headlines before assigning semantic labels. The script calculates results at runtime; no precomputed or invented trend values are inserted into the report.

## Reproducibility

All sample-based models use a fixed random seed. The full corpus is used for yearly counts, phrase rates, and categories unless a document cap is specified. LDA and clustering are fitted on reproducible samples and applied to all documents in batches. Keep generated outputs with the report for reproducibility.

## Attribution

Misra, R. (2022). *News Category Dataset*. arXiv:2209.11429. https://arxiv.org/abs/2209.11429. Dataset is identified as CC BY 4.0 in the source card; retain attribution when reusing it.
