# Business Case Analysis — Promotion Effectiveness at a Fashion Retail Chain

---

## B1. Problem Formulation

---

### B1(a) — Machine Learning Problem Formulation

**Target Variable**

The target variable is `items_sold` — the number of items sold at a given store in a given month under a given promotion type.

**Candidate Input Features**

| Category | Features |
|---|---|
| Store attributes | `store_id`, `store_size`, `location_type` (urban / semi-urban / rural), `monthly_footfall`, `competition_density` |
| Promotion | `promotion_type` (Flat Discount, BOGO, Free Gift, Category Offer, Loyalty Points) |
| Temporal | `month`, `year`, `is_weekend`, `is_festival`, `is_month_end` |
| Customer demographics | `avg_customer_age`, `income_band`, `loyalty_membership_rate` (if available) |
| Historical performance | `items_sold_lag_1month`, `items_sold_same_month_last_year`, `avg_items_sold_last_3months` |

**Type of ML Problem and Justification**

This is a **supervised regression problem**. The target variable (`items_sold`) is a continuous integer count, and we have labelled historical observations pairing store-promotion-context combinations with their resulting sales volumes.

The ultimate business goal — *which promotion to run in each store each month* — is technically a recommendation or optimisation task. However, the correct approach is to first build a regression model that predicts `items_sold` for each of the five promotion types under a given store-month context, then select the promotion with the highest predicted count. This is preferable to framing it as a 5-class classification problem (predict the "best" promotion directly) because:

1. **Regression preserves magnitude** — we learn *how much better* one promotion is than another, not just which is best, enabling cost-benefit trade-offs.
2. **Classification discards information** — labelling the historically used promotion as the "correct" class ignores the counterfactual: a different promotion might have performed even better in the same context.
3. **Regression enables simulation** — at inference time, we score all five promotions for each store-month and rank them, giving the marketing team a full recommendation with predicted uplifts for each option.

---

### B1(b) — Items Sold vs Revenue as Target Variable

**Why items sold is more reliable than total sales revenue**

Using revenue as the target variable introduces several confounds that make the model harder to train, less stable, and less actionable:

1. **Price variation inflates revenue independent of promotion effectiveness.** A store that runs a Flat Discount on high-ticket items will generate high revenue even with low unit volume. The model would conflate price-level effects with promotion effectiveness.

2. **Revenue is non-stationary across time.** Seasonal price changes, inflation, and product mix shifts cause systematic drift in revenue that has nothing to do with promotion performance. `items_sold` is a more stable, inflation-agnostic signal.

3. **Promotions change the price directly** — a Flat Discount reduces the per-item revenue by design. If the target is revenue, the model will systematically undervalue discounting promotions even when they drive strong volume uplift.

4. **Operational relevance.** The key marketing question is *which promotion drives footfall and basket size*, not which maximises short-term revenue. Volume is the proximate output of promotion choice; revenue is a downstream transformation involving price which adds noise to the causal chain.

**Broader principle: target variable alignment**

This illustrates the principle of **target variable alignment with the causal mechanism you are modelling**. In real-world ML projects, the most easily measured metric (revenue, clicks, page views) is often not the best target because it conflates the signal of interest with confounders. Best practice is to choose a target that is:

- **Directly influenced** by the decision being optimised (here: promotion type → volume)
- **Stable** across the time horizon of interest
- **Not self-fulfilling** — revenue could lead to circular incentives where high-price stores always appear to benefit most regardless of promotion type

---

### B1(c) — Modelling Strategy Across Heterogeneous Stores

**Problem with a single global model**

A single global model trained across all 50 stores will learn average relationships. Since stores in urban, semi-urban, and rural locations respond very differently to the same promotion (e.g., BOGO may drive strong uplift in urban stores with high footfall but negligible uplift in rural stores where customers buy infrequently), the global model will be mis-specified for most individual stores and systematically wrong for stores at the extremes.

**Proposed strategy: Hierarchical / Grouped Modelling**

The recommended alternative is a **segmented modelling approach with shared structure**, implemented as follows:

1. **Segment stores into cohorts** based on meaningful characteristics: `location_type` (urban / semi-urban / rural) and `store_size` (small / medium / large), giving up to 9 natural segments. Train one model per segment so each model learns promotion response patterns from stores with similar structural profiles.

2. **Use store-level fixed effects or embeddings** within each segment model to capture residual store-specific behaviour (e.g., a particular urban store in a tourist area may have an unusually strong festival effect). This can be achieved via:
   - Including `store_id` as a categorical feature with target encoding (encoding each store by its historical mean `items_sold`)
   - Or using a mixed-effects model framework that separates population-level (segment) effects from store-level random effects

3. **Alternatively, use a single model with rich interaction features** — interaction terms between `promotion_type` and `location_type`, and between `promotion_type` and `store_size`. Tree-based ensemble models (Random Forest, Gradient Boosting) can learn these interactions automatically if given sufficient data.

**Justification:** This strategy preserves the benefit of pooling data (segment models see more training examples than individual store models) while respecting the structural heterogeneity that makes a fully global model unreliable. It also degrades gracefully for new stores — a new urban medium store is immediately assigned to the urban/medium segment model without needing its own training history.

---

## B2. Data and EDA Strategy

---

### B2(a) — Joining Tables and Dataset Grain

**Table Structure and Join Logic**

The four source tables and their natural keys are:

| Table | Grain | Key Columns |
|---|---|---|
| `transactions` | One row per transaction | `transaction_id`, `store_id`, `transaction_date`, `items_sold` |
| `store_attributes` | One row per store | `store_id`, `store_size`, `location_type`, `footfall`, `competition_density` |
| `promotion_details` | One row per store-month | `store_id`, `month`, `year`, `promotion_type` |
| `calendar` | One row per calendar date | `date`, `is_weekend`, `is_festival`, `month`, `year` |

**Join sequence:**

```sql
SELECT *
FROM transactions t
  LEFT JOIN store_attributes s   ON t.store_id = s.store_id
  LEFT JOIN promotion_details p  ON t.store_id = p.store_id
                                AND YEAR(t.transaction_date)  = p.year
                                AND MONTH(t.transaction_date) = p.month
  LEFT JOIN calendar c           ON t.transaction_date = c.date
```

Use LEFT JOINs throughout to preserve all transactions even if store attributes or promotion details are temporarily missing (and flag those as NULL for investigation before modelling).

**Grain of the final modelling dataset**

The grain is: **one row = one store × one month × one promotion type**

This is the natural decision unit — the marketing team assigns one promotion per store per month, so the model must predict at this level.

**Aggregations before modelling**

Raw transaction-level data must be aggregated to the store-month grain before modelling:

| Aggregation | Source | Purpose |
|---|---|---|
| `SUM(items_sold)` | transactions | Target variable: total monthly items sold |
| `COUNT(transaction_id)` | transactions | Feature: footfall proxy (transaction volume) |
| `AVG(basket_size)` | transactions | Feature: average items per visit |
| `promotion_type` (already monthly) | promotion_details | Primary categorical feature |
| `MAX(is_festival)` over month | calendar | 1 if any festival day occurred in the month |
| `COUNT(is_weekend = 1)` over month | calendar | Number of weekend days — varies slightly by month |

---

### B2(b) — EDA Strategy

**1. Target Variable Distribution (`items_sold`)**

*Chart:* Histogram of `items_sold` with separate overlays by `location_type` and `store_size`.

*What to look for:* Skewness, outliers, multi-modality. If the distribution is strongly right-skewed, a log transformation of the target (`log(items_sold + 1)`) may improve model fit and reduce the influence of extreme observations. Multi-modality confirms that store segments have fundamentally different baseline volumes, supporting the segmented modelling strategy from B1(c).

*Modelling implication:* Skewness → consider log-transforming the target. Heavy outliers → consider Huber loss instead of MSE. Multi-modal distribution → confirms segmented models are necessary.

**2. Promotion Type vs Items Sold — Grouped Box Plot by Location Type**

*Chart:* Grouped box plot — x-axis: `promotion_type`, y-axis: `items_sold`, hue: `location_type`.

*What to look for:* Whether the same promotion performs differently across location types. Does BOGO show high variance in urban stores but low variance in rural? Are there interaction effects where one promotion dominates only in a specific segment?

*Modelling implication:* Clear interactions → engineer `promotion_type × location_type` interaction features or confirm segmented models are necessary. If one promotion dominates universally, the problem may be simpler than expected.

**3. Monthly Seasonality Analysis — Line Plot**

*Chart:* Line plot of mean `items_sold` by `month`, faceted by `promotion_type`.

*What to look for:* Seasonal peaks (e.g., December festival season, summer sales). Whether different promotions amplify seasonal effects — e.g., does Flat Discount have an especially strong December effect while Loyalty Points dominates off-peak months?

*Modelling implication:* Strong seasonality → `month` and `is_festival` are important features. Consider adding `items_sold_same_month_last_year` as a lag feature to capture year-over-year cycles explicitly.

**4. Competition Density vs Promotion Effectiveness — Scatter Plot**

*Chart:* Scatter plot of `competition_density` vs `items_sold`, coloured by `promotion_type`, with a regression line per promotion.

*What to look for:* Whether promotion effectiveness changes as a function of local competition. BOGO might be more effective in high-competition environments (where customers choose between nearby stores) than in low-competition areas.

*Modelling implication:* Clear interaction → engineer a `competition_density × promotion_type` feature. Strong main effect independent of promotion → `competition_density` should rank among top features.

**5. Correlation Heatmap and Collinearity Check**

*Chart:* Correlation heatmap of all numerical features.

*What to look for:* High collinearity between features (e.g., `footfall` and `store_size` may correlate > 0.8). Multicollinearity inflates variance in Linear Regression and splits feature importance in Random Forest across correlated features.

*Modelling implication:* High collinearity → drop one of a collinear pair for linear models; interpret importance of correlated features jointly for tree-based models.

---

### B2(c) — Addressing Promotion Imbalance (80% No-Promotion Transactions)

**How the imbalance affects the model**

With 80% of records having no promotion, the model is trained predominantly on the no-promotion baseline. This creates two problems:

1. **Biased feature importance** — `promotion_type` will appear less important than it actually is because most training examples share the same no-promotion value. The model learns the baseline well but has insufficient examples of each promotion type to learn their differential effects reliably.

2. **Underestimation of promotion uplift** — if promotion-period records are underrepresented, the model's predicted uplift for promotion periods will be attenuated toward the no-promotion baseline, causing systematic under-recommendation of promotions.

**Steps to address the imbalance**

1. **Stratified sampling at training time.** Oversample promotion-period records so that each of the five promotion types is adequately represented alongside the no-promotion baseline. Use `sklearn.utils.resample` with temporal care — oversample within time blocks, never randomly across the full dataset, to avoid introducing future data into earlier folds.

2. **Uplift modelling.** Train a separate model to predict promotion *uplift* — the incremental `items_sold` above each store's no-promotion baseline — rather than raw `items_sold`. The no-promotion records then serve as calibration points to establish the baseline, and the uplift model is trained only on promotion-period records. This cleanly separates the baseline signal from the promotion signal.

3. **Weighted loss function.** Assign higher sample weights to promotion-period records during training via the `sample_weight` parameter in scikit-learn. This penalises errors on promotion records more heavily without resampling, preserving temporal integrity of the dataset.

4. **Feature engineering: relative target.** Engineer the target as `items_sold / store_monthly_baseline` — normalising by each store's rolling no-promotion average. This transforms the prediction problem into forecasting a promotion uplift ratio, making all records comparably informative regardless of promotion presence.

---

## B3. Model Evaluation and Deployment

---

### B3(a) — Train-Test Split, Evaluation Metrics

**Train-Test Split Strategy**

With three years of monthly data across 50 stores (approximately 1,800 store-month observations), the appropriate split is a **temporal walk-forward split**, not a random split.

*Single holdout implementation:*
- **Training set:** Months 1–30 (Years 1 and 2 plus first half of Year 3)
- **Test set:** Months 31–36 (second half of Year 3 — the most recent six months)

*Preferred: rolling walk-forward cross-validation:*
- Fold 1: Train on months 1–18, validate on months 19–24
- Fold 2: Train on months 1–24, validate on months 25–30
- Fold 3: Train on months 1–30, validate on months 31–36

This simulates real deployment — the model always predicts future months it has never seen, trained strictly on historical data. Performance metrics are averaged across folds to reduce variance in the estimate.

**Why a random split is inappropriate**

A random split violates the temporal ordering of the data in two critical ways:

1. **Temporal leakage (look-ahead bias):** If a record from Month 30 (December Year 3) is in the training set, the model learns that December drives high sales. When evaluated on a December record randomly placed in the test set, it faces a scenario whose temporal context it has already seen — producing unrealistically optimistic metrics that will not hold in production.

2. **Concept drift is hidden:** Retail data evolves over time — customer behaviour changes, new competitors enter, and store demographics shift. A temporal split exposes whether the model generalises across time, which is the only meaningful test. A random split masks this drift entirely because train and test are sampled from the same overall time distribution.

**Evaluation Metrics**

| Metric | Formula | Business Interpretation |
|---|---|---|
| **RMSE** | √(mean((ŷ − y)²)) | Average prediction error in item units, penalising large individual errors more heavily. A RMSE of 15 means the typical prediction error is ~15 items — use this to judge operational acceptability against store baselines. |
| **MAE** | mean(|ŷ − y|) | Average absolute error in item units — robust to outliers and easy to communicate. "Our forecasts are off by X items on average." |
| **MAPE** | mean(|ŷ − y| / y) × 100 | Percentage error — essential for comparing performance fairly across stores with very different baseline volumes. A small store (50 items/month) and a large store (500 items/month) should not be compared via MAE alone. |
| **Promotion recommendation accuracy** | % of months where the model's top-1 recommended promotion matches the actual best-performing promotion (in hindsight) | Business-level metric — directly measures whether the model would have given correct advice. Requires counterfactual evaluation on held-out data where multiple promotion types were tested in similar contexts. |

---

### B3(b) — Investigating Different Recommendations for the Same Store

**Why the model recommends differently for Store 12 in December vs March**

The model predicts `items_sold` for each of the five promotions separately under each store-month context, then recommends the highest. Different recommendations arise because the feature values that encode December context (high `is_festival`, `month=12`, more weekend days) produce a different promotion ranking than the March context — not because of any inconsistency in the model.

**Investigation process**

**Step 1: Model-level feature importance**

Extract global feature importances from the Random Forest. If `is_festival`, `month`, and `is_weekend` are top-ranked features (as found empirically in Q3), this immediately explains the December vs March difference — the model has learned that December's festival and holiday context favours Loyalty Points (customers are in a gift-buying, relationship-building mindset and value future reward accumulation across multiple visits), while March's lower seasonal engagement makes price-sensitive Flat Discount more effective at motivating otherwise-infrequent purchases.

**Step 2: SHAP value decomposition**

Use SHAP (SHapley Additive exPlanations) to decompose the model's prediction for Store 12 in each month into per-feature contributions:

```python
import shap
explainer   = shap.TreeExplainer(rf_pipeline.named_steps['model'])
shap_dec    = explainer.shap_values(X_store12_december_loyalty)
shap_mar    = explainer.shap_values(X_store12_march_flat_discount)
shap.waterfall_plot(shap.Explanation(values=shap_dec[0], ...))
```

The waterfall chart shows, feature by feature, how much each input pushed the prediction above or below the model's average baseline. In December, `is_festival=1` and `month=12` will show large positive SHAP values for Loyalty Points; in March, lower temporal features push Flat Discount above the others.

**Step 3: Present a full promotion ranking to the marketing team**

Rather than presenting a single recommendation, show predicted `items_sold` for all five promotions in each month, so the marketing team can see the full competitive landscape:

| Promotion | Dec Predicted | Mar Predicted | Dec Rank | Mar Rank |
|---|---|---|---|---|
| Loyalty Points Bonus | 412 | 287 | **#1** | #3 |
| Flat Discount | 378 | 318 | #2 | **#1** |
| BOGO | 355 | 301 | #3 | #2 |
| Free Gift | 340 | 271 | #4 | #4 |
| Category Offer | 321 | 259 | #5 | #5 |

Accompany this with plain-language narrative grounded in the SHAP values: *"In December, the festival effect and end-of-year shopping mindset make customers more receptive to relationship-building promotions — Loyalty Points capitalise on multiple-visit purchasing. In March, with no seasonal uplift, customers need a direct price incentive to purchase, which is exactly what Flat Discount provides."*

This makes the model's reasoning transparent, auditable, and actionable — the marketing team can challenge, override, or accept recommendations with full visibility into the reasoning.

---

### B3(c) — End-to-End Deployment and Monitoring

**1. Saving the Model**

After final training on all available data, serialise the complete scikit-learn Pipeline — which includes the fitted `ColumnTransformer` (with fitted scaler statistics and OHE vocabulary) and the trained model — using `joblib`:

```python
import joblib
joblib.dump(pipeline, 'models/promotion_recommender_v1_2024Q4.pkl')
```

Saving the full pipeline (not just the model weights) is essential because the `StandardScaler` must apply the same mean and variance statistics it was fitted on during training. Re-fitting on new data would introduce leakage. Version the saved file with a date stamp so rollbacks are straightforward.

**2. Preparing and Feeding Monthly Data**

At the start of each month, an automated data preparation job runs the following steps:

1. **Construct the feature matrix** for each of the 50 stores × 5 promotion types = 250 rows, representing every possible store-promotion combination for the upcoming month.
2. **Populate features** from live sources: store attributes (static, refreshed quarterly), the proposed promotion schedule from the marketing team (if available, else score all five options), and calendar features (generated deterministically from the month).
3. **Load the serialised pipeline** and call `pipeline.predict(X_new)` — the fitted preprocessor and model execute automatically, outputting 250 predicted `items_sold` values.
4. **Rank and select** — for each of the 50 stores, select the `promotion_type` with the highest predicted value. Generate a ranked recommendation report and publish to the marketing team's dashboard.

The entire workflow should be containerised (Docker), version-controlled, and scheduled via an orchestration tool (Apache Airflow, GitHub Actions, or a cloud scheduler such as AWS EventBridge) to run automatically at month start without manual intervention.

**3. Monitoring for Model Degradation**

Once deployed, monitor three layers continuously:

| Layer | Metric | Alert Threshold | Action |
|---|---|---|---|
| **Input data drift** | Population Stability Index (PSI) on key features (`competition_density`, `footfall`, `promotion_type` frequency distribution) between current month and training distribution | PSI > 0.2 on any feature | Investigate data pipeline; flag for potential retraining |
| **Prediction drift** | Distribution of predicted `items_sold` values month-over-month (mean and variance) | Mean prediction shifts > 2 standard deviations from the 6-month rolling average | Review input features; consider retraining |
| **Performance degradation** | After each month's actuals are available (~2 weeks post-month-end), compute RMSE and MAE against real outcomes | RMSE exceeds training-period RMSE by > 20% for two consecutive months | Trigger full retraining pipeline |

**Retraining cadence:** Retrain quarterly by appending the three most recent months of actuals to the training corpus and re-running the full pipeline fit. Before promoting the new model to production, validate it on the last two quarters of held-out data using the walk-forward evaluation described in B3(a). The new model must outperform the incumbent on both RMSE and MAE to replace it — this is a **champion/challenger framework** that prevents accidental regression.

**Concept drift watchlist:** Track whether the historical ranking of promotions by effectiveness is stable over time using Spearman rank correlation between current and historical promotion performance rankings per store segment. A sudden reversal — e.g., Flat Discount replacing Loyalty Points as the top December performer — signals a genuine shift in customer behaviour that retraining alone cannot fix and requires a business review of the feature set and potential collection of new signals (e.g., macroeconomic indicators, competitor promotion data).
