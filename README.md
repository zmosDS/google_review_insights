# Aspect-Based Sentiment Analysis of Google Maps Reviews

Compares traditional NLP (TF-IDF), transformer-based (RoBERTa), and generative LLM (GPT-4.1-mini) approaches for extracting aspect-sentiment pairs from large-scale customer reviews. Labels are generated via weak supervision using star ratings, then validated against a manually labeled holdout set.

---

## Goal
The goal of this project is to evaluate how well classical and modern NLP methods can identify aspect-level sentiment from real-world review data at scale. The pipeline covers data processing, exploratory analysis, weak supervision label generation, and model training across three approaches — TF-IDF + Logistic Regression, RoBERTa fine-tuning, and LLM prompting (zero-shot and few-shot) — benchmarked against a manually labeled ABSA validation set.

---

## Results

| Model | Weighted F1 | Macro F1 | Test Accuracy |
|-------|-------------|----------|---------------|
| RoBERTa Fine-Tuned (100K) | 0.927 | 0.872 | 0.909 |
| TF-IDF + Logistic Regression | 0.882 | 0.785 | 0.783 |
| GPT-4.1-mini Zero-Shot | TBD | TBD | TBD |
| GPT-4.1-mini Few-Shot | TBD | TBD | TBD |

- RoBERTa at 250K samples achieved Macro F1 0.884 — marginal gains at 3x training time
- Best TF-IDF config: 3x negative class penalty + negation handling
- Best RoBERTa config: 100K sample, baseline (no class weights)
- LLM evaluation run on 10K balanced sample across all star ratings

---

## Highlights
- Weak supervision pipeline scales to 22M reviews using star ratings as proxy labels
- RoBERTa outperforms TF-IDF baseline by ~9 points Macro F1
- Scaling experiments (10K–250K) show diminishing returns beyond 100K training samples
- Atmosphere is the hardest aspect across all models — semantic breadth limits keyword-based detection
- Results validated against a manually labeled ABSA holdout set

---

## Built With
- Python (Pandas, NumPy, Matplotlib, Seaborn, Scikit-Learn)
- Transformers (HuggingFace RoBERTa)
- OpenAI API (GPT-4.1-mini)
- Jupyter Notebooks
- Google Local Reviews dataset (UC San Diego McAuley Lab)

---

## Labels

**7 aspects × 2 sentiments = 14 binary labels**

Aspects: `product_quality`, `service`, `wait_time`, `price_value`, `cleanliness`, `atmosphere`, `general`

Sentiments: `positive`, `negative`

---

## Figures

| Figure | Description |
|--------|-------------|
| `figure1_tfidf_comparison.png` | TF-IDF model configs compared across Weighted F1, Macro F1, accuracy, aspect F1 by sentiment, and atmosphere penalty zoom |
| `figure2_tfidf_confusion.png` | Confusion matrices for best, average, and worst performing TF-IDF classes |
| `figure3_atmosphere_features.png` | Top 20 TF-IDF features driving atmosphere negative predictions |
| `figure4_roberta_scaling.png` | RoBERTa scaling results across 10K–250K samples with training time |
| `figure5_roberta_aspect_f1.png` | RoBERTa aspect F1 by sentiment for best run (100K) |
| `figure6_roberta_confusion.png` | Confusion matrices for best, average, and worst performing RoBERTa classes |

---

## File Structure
```
google_review_insights/
├── figures/
│   ├── figure1_tfidf_comparison.png
│   ├── figure2_tfidf_confusion.png
│   ├── figure3_atmosphere_features.png
│   ├── figure4_roberta_scaling.png
│   ├── figure5_roberta_aspect_f1.png
│   └── figure6_roberta_confusion.png
├── data/                                # Data files (stored on UCSD DataHub)
│   └── .gitkeep
├── notebooks/
│   ├── 01_process_data.ipynb            # Data cleaning and preprocessing
│   ├── 02_eda.ipynb                     # Exploratory data analysis
│   ├── 03_feature_engineering.ipynb     # Feature extraction and engineering
│   ├── 04_absa_training_set.ipynb       # Weak supervision label generation
│   ├── 05_model_tfidf.ipynb             # TF-IDF + Logistic Regression
│   ├── 06a_model_roberta_scaling.ipynb  # RoBERTa scaling experiments (10K–250K)
│   ├── 06b_model_roberta_tuning.ipynb   # RoBERTa hyperparameter tuning at 100K
│   └── 07_model_llm_prompt.ipynb        # GPT-4.1-mini zero-shot and few-shot
├── models/
│   ├── tfidf_vectorizer.pkl             # Fitted TF-IDF vectorizer
│   ├── tfidf_logreg_final.pkl           # Best TF-IDF LogReg model (3x penalty)
│   └── roberta_final/                   # Fine-tuned RoBERTa model (HuggingFace format)
├── .gitignore
├── requirements.txt
└── README.md
```

---

## Data Setup

**Google Local Data (2021) — UC San Diego McAuley Lab**

Raw data (~25GB) is not stored due to storage constraints. Processed files ready for modeling are available on UCSD DataHub.

To download the raw data directly:
[Google Local Data (2021)](https://mcauleylab.ucsd.edu/public_datasets/gdrive/googlelocal/)

---

## Contributors
Zack M. • Jillian O.

---

## Citations

Li, J., Shang, J., & McAuley, J. (2022). UCTopic: Unsupervised Contrastive Learning for Phrase Representations and Topic Mining. *ACL.*

Yan, A., He, Z., Li, J., Zhang, T., & McAuley, J. (2023). Personalized Showcases: Generating Multi-Modal Explanations for Recommendations. *SIGIR.*
