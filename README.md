# Interview_Prep

## 3-Week Prep Tracker — SQL + ML/Stats + Product Sense + GenAI
**Goal:** Interview-ready fundamentals + senior-level tradeoff thinking + GenAI fluency  
**Daily time target:** 2.5–3.5 hours  
- SQL: 30–45m  
- ML/Stats: 60–75m  
- Product Sense: 30–45m  
- GenAI: 20–40m  
- Notes/flashcards: 10m

---

## How to use this tracker
- Each day has:
  1) **Study** checklist  
  2) **Resources** (exact links)  
  3) **Deliverable** (proof you learned it)
- If time is tight: do **SQL + ML/Stats + 1 deliverable**.

---

# WEEK 1 — Probability, Metrics, Errors, Linear Models + Kaggle GenAI (Days 1–5)

## Day 1 — Probability Foundations + GenAI Day 1
### ✅ Checklist
**SQL**
- [ ] 1 LeetCode SQL (easy/medium)
- [ ] 1 window-function drill (ROW_NUMBER / SUM OVER / LAG)

**ML/Stats**
- [ ] Random variables (discrete vs continuous)
- [ ] PMF vs PDF vs CDF
- [ ] Expectation + linearity of expectation
- [ ] Variance + intuition
- [ ] Covariance vs correlation
- [ ] Distributions: Bernoulli, Binomial, Poisson, Normal
- [ ] LLN vs CLT (intuition + when each applies)

**Product Sense**
- [ ] Why averages can be misleading (distribution + tails)
- [ ] Example: “avg delivery time” vs p95/p99

**GenAI (Kaggle 5-Day GenAI — Day 1)**
- [ ] Foundational models overview
- [ ] What an LLM predicts (next-token probability)
- [ ] Prompt basics (instruction + context)

### 📚 Resources (exact)
- StatQuest video index (use relevant videos): https://statquest.org/video_index.html
- StatQuest: Central Limit Theorem (video): https://www.youtube.com/watch?v=YAlJCEDH2uY
- Kaggle 5-Day GenAI guide: https://www.kaggle.com/learn-guide/5-day-genai

### 🧾 Deliverable
- [ ] Write 8–10 bullet notes: “LLN vs CLT + when Poisson makes sense”
- [ ] One example: *what distribution might model X?*

---

## Day 2 — Hypothesis Testing + Errors (FP/FN) + GenAI Day 2
### ✅ Checklist
**SQL**
- [ ] 1 LeetCode SQL (medium)
- [ ] Practice JOINs + filtering pitfalls (LEFT JOIN + WHERE vs ON)

**ML/Stats**
- [ ] Population vs sample
- [ ] Sampling bias vs random sampling
- [ ] Null vs alternative hypothesis
- [ ] p-value (what it is / isn’t)
- [ ] Type I error = False Positive
- [ ] Type II error = False Negative
- [ ] Power (what increases power?)
- [ ] Confidence intervals (interpretation)

**Product Sense**
- [ ] Map FP vs FN cost to 3 domains: fraud, churn, medical
- [ ] Explain: “What changes when we lower alpha?”

**GenAI (Kaggle Day 2)**
- [ ] Embeddings: what they represent (high-level)
- [ ] Vector stores / why retrieval helps

### 📚 Resources (exact)
- StatQuest video index (p-values / hypothesis tests / CI): https://statquest.org/video_index.html
- Kaggle 5-Day GenAI guide: https://www.kaggle.com/learn-guide/5-day-genai

### 🧾 Deliverable
- [ ] 5-sentence explanation: p-value + Type I/II to a PM
- [ ] 1 table: FP vs FN consequences for fraud/churn/medical

---

## Day 3 — Confusion Matrix + Core Classification Metrics + GenAI Day 3
### ✅ Checklist
**SQL**
- [ ] 1 LeetCode SQL (medium)
- [ ] Build a cohort query (DATE_TRUNC + retention style)

**ML/Stats**
- [ ] Confusion matrix (TP/FP/TN/FN)
- [ ] Accuracy + why misleading
- [ ] Precision (PPV)
- [ ] Recall (Sensitivity / TPR)
- [ ] Specificity (TNR)
- [ ] FPR, FNR
- [ ] F1 (when it helps / when it doesn’t)
- [ ] Threshold moving: how metrics change

**Product Sense**
- [ ] Decide metric for: spam filter, fraud, medical, churn
- [ ] Explain threshold tuning in business terms

**GenAI (Kaggle Day 3)**
- [ ] Prompt engineering: instructions, few-shot examples
- [ ] Output control basics (tone/format constraints)

### 📚 Resources (exact)
- Google ML Crash Course — Classification module: https://developers.google.com/machine-learning/crash-course/classification
- Kaggle 5-Day GenAI guide: https://www.kaggle.com/learn-guide/5-day-genai

### 🧾 Deliverable
- [ ] Create 1 confusion matrix example and compute every metric by hand
- [ ] 2-min spoken answer (record yourself): “Why not accuracy?”

---

## Day 4 — ROC/AUC + Precision-Recall + Imbalance + GenAI Day 4
### ✅ Checklist
**SQL**
- [ ] 1 LeetCode SQL (medium)
- [ ] Practice anti-joins / set logic (NOT EXISTS / EXCEPT)

**ML/Stats**
- [ ] ROC curve construction (threshold sweep)
- [ ] AUC meaning (probabilistic interpretation)
- [ ] PR curve vs ROC (when PR matters)
- [ ] Class imbalance strategies (at least 3): weighting, resampling, thresholding
- [ ] Calibration concept (why scores can be miscalibrated)

**Product Sense**
- [ ] When ROC can be misleading for business
- [ ] Choose metric for imbalanced scenario + defend

**GenAI (Kaggle Day 4)**
- [ ] Model evaluation intuition (why “accuracy” isn’t enough)
- [ ] Common failure modes (hallucination, missing context)

### 📚 Resources (exact)
- Google MLCC — ROC & AUC: https://developers.google.com/machine-learning/crash-course/classification/roc-and-auc
- Kaggle 5-Day GenAI guide: https://www.kaggle.com/learn-guide/5-day-genai

### 🧾 Deliverable
- [ ] Write: “When PR > ROC” with one real-world example
- [ ] Explain AUC in 2 sentences without jargon

---

## Day 5 — Linear Regression Deep + GenAI Day 5
### ✅ Checklist
**SQL**
- [ ] 1 LeetCode SQL (medium)
- [ ] Aggregation drill (GROUP BY with conditional sums)

**ML/Stats**
- [ ] OLS objective (minimizing squared error)
- [ ] Assumptions: linearity, independence, homoscedasticity, normal errors
- [ ] Coefficient interpretation (unit change)
- [ ] R² vs Adjusted R²
- [ ] Residual diagnostics (what to look for)
- [ ] Multicollinearity + VIF concept

**Product Sense**
- [ ] Translate coefficients into “product impact”
- [ ] Identify when regression is not appropriate

**GenAI (Kaggle Day 5)**
- [ ] End-to-end workflow recap + mini capstone (if included)
- [ ] Where GenAI fits vs classical ML

### 📚 Resources (exact)
- StatQuest video index (linear regression topics): https://statquest.org/video_index.html
- Kaggle 5-Day GenAI guide: https://www.kaggle.com/learn-guide/5-day-genai

### 🧾 Deliverable
- [ ] One-page “Linear Regression Pitfalls” note
- [ ] 3 bullets: where GenAI helps your DS work at Verizon (safe, realistic)

---

## Day 6 — Logistic Regression + Kaggle Review
### ✅ Checklist
**SQL**
- [ ] 1 LeetCode SQL (medium)
- [ ] Practice CASE WHEN + percent metrics

**ML/Stats**
- [ ] Log-odds intuition
- [ ] Sigmoid & probability output
- [ ] Odds ratios interpretation
- [ ] Decision threshold selection
- [ ] Regularization intro (L1 vs L2 — why)

**Product Sense**
- [ ] Why logistic is “trusted” in industry
- [ ] Explain threshold choice + business costs

**GenAI**
- [ ] Review Days 1–5: write 3 bullets each day (“what I learned”)

### 📚 Resources (exact)
- StatQuest video index (logistic regression): https://statquest.org/video_index.html
- Kaggle 5-Day GenAI guide: https://www.kaggle.com/learn-guide/5-day-genai

### 🧾 Deliverable
- [ ] 2-min spoken answer: “Explain logistic regression + thresholds”

---

## Day 7 — Week 1 Review + Mock
### ✅ Checklist
- [ ] Recompute all classification metrics from a confusion matrix (no notes)
- [ ] Explain Type I/II + power + CI to a PM
- [ ] Explain ROC vs PR and when each is used
- [ ] Write a “Metric selection framework” (10 bullets)

### 🧾 Deliverable
- [ ] 1-page cheat sheet: metrics + errors + threshold rules

---

# WEEK 2 — Bias/Variance, Regularization, Trees/Ensembles, Evaluation, A/B Testing

## Day 8 — Bias/Variance + Learning Curves
### ✅ Checklist
**SQL**
- [ ] 1 LeetCode SQL (medium)

**ML/Stats**
- [ ] Bias vs variance (definitions + examples)
- [ ] Underfit vs overfit
- [ ] Learning curves (what shapes mean)
- [ ] Train/val/test split best practices

**Product Sense**
- [ ] Why offline wins can fail online
- [ ] Identify “overfitting to a metric” in product teams

**GenAI**
- [ ] Prompting vs fine-tuning vs RAG (when each is appropriate)

### 📚 Resources
- Google MLCC — Overfitting module: https://developers.google.com/machine-learning/crash-course/overfitting
- StatQuest video index (bias-variance): https://statquest.org/video_index.html

### 🧾 Deliverable
- [ ] 8 bullets: diagnosing overfitting + what to do

---

## Day 9 — Regularization + Cross-Validation + Leakage
### ✅ Checklist
**ML/Stats**
- [ ] L1 vs L2: effect on coefficients (sparsity vs shrinkage)
- [ ] Elastic Net: when useful
- [ ] k-fold CV vs time-based CV
- [ ] Leakage: 5 examples and how to prevent

**Product Sense**
- [ ] “Leakage” in product metrics (post-treatment bias)

**GenAI**
- [ ] LLM evaluation: why hard, and 3 practical evaluation strategies

### 📚 Resources
- StatQuest video index (regularization): https://statquest.org/video_index.html
- Google MLCC — validation basics (classification pages): https://developers.google.com/machine-learning/crash-course/classification

### 🧾 Deliverable
- [ ] “Leakage checklist” you’ll use at work (10 items)

---

## Day 10 — Decision Trees
### ✅ Checklist
**ML/Stats**
- [ ] How splits work (greedy)
- [ ] Gini vs entropy intuition
- [ ] Depth/leaf constraints to reduce overfit
- [ ] Feature importance pitfalls

**Product Sense**
- [ ] Interpretability tradeoffs (why leaders like trees)

**GenAI**
- [ ] RAG: when retrieval improves factuality + 3 failure modes

### 📚 Resources
- StatQuest video index (
