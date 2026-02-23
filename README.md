# Aspect-Based Sentiment Analysis of Google Maps Reviews

Compares traditional NLP (TF-IDF) and transformer-based (RoBERTa) approaches for extracting aspect-sentiment pairs from customer reviews.

## Models

### TF-IDF + Logistic Regression 
(`05_model_tfidf.ipynb`) <br>
Classical NLP baseline trained on the full 22M review dataset. Uses keyword-based aspect detection and star rating weak supervision for labels.

**Best Result:** 2x negative penalty + negation handling
| Metric | Score |
|---|---|
| Weighted F1 | 0.880 |
| Macro F1 | 0.778 |
| Test Accuracy | 0.794 |

### RoBERTa Fine-Tuning 
(`06a_model_roberta_scaling.ipynb`, `06b_model_roberta_tuning.ipynb`) <br>
Transformer-based model fine-tuned on weak supervision labels. Scaling notebook explores 10K–250K training samples. Tuning notebook applies class weights at 100K — the sweet spot of performance and training time.

**Best Result:** 100K sample (baseline, no class weights)
| Metric | Score |
|---|---|
| Weighted F1 | 0.930 |
| Macro F1 | 0.879 |
| Test Accuracy | 0.915 |

250K achieved marginal gains (Macro F1 0.884) at 3x the training time.

## Labels
7 aspects × 2 sentiments = 14 binary labels

Aspects: `product_quality`, `service`, `wait_time`, `price_value`, `cleanliness`, `atmosphere`, `general`
Sentiments: `positive`, `negative`

## Notes
- TF-IDF trained locally (64GB RAM), RoBERTa trained on UCSD's DataHub GPU
- Manual labeling validation set excluded from training via `review_id` filter

## Data 
[Google Local Data (2021) — UC San Diego McAuley Lab](https://mcauleylab.ucsd.edu/public_datasets/gdrive/googlelocal/)

**Citation:**
Li, J., Shang, J., & McAuley, J. (2022). UCTopic: Unsupervised Contrastive Learning for Phrase Representations and Topic Mining. ACL.

Yan, A., He, Z., Li, J., Zhang, T., & McAuley, J. (2023). Personalized Showcases: Generating Multi-Modal Explanations for Recommendations. SIGIR.

## Contributors
Zack M. • Jillian O.
