# 🧠 Challenge Overview: Multimodal Market Forecasting with News

In today's data-driven financial world, market prices are not influenced by numbers alone — news headlines, sentiment, and macroeconomic narratives play a pivotal role in shaping investor behavior and driving price movements.

This challenge simulates a realistic financial forecasting scenario by combining quantitative market data with qualitative news information, offering a rich and complex environment for applying advanced machine learning techniques.

---

## 🚀 Why This Challenge Matters

- 📉 **Traditional forecasting models** often rely solely on historical price data, missing the broader context behind market movements.
- 📰 **Financial news headlines** provide critical signals about investor sentiment, economic conditions, and global events — all of which can influence asset prices within minutes or hours.
- 🤖 **Transformer-based architectures**, originally built for language modeling, offer a powerful framework to integrate both temporal sequences and textual inputs, enabling context-aware, informed predictions.

---

## 🎯 Key Goals

This challenge encourages participants to:

- 📊 Explore **multimodal learning** by combining time series data with natural language inputs.
- 🧠 Design **custom Transformer-based models** adapted to financial forecasting.
- 🎯 Tackle a **multi-target regression problem** by predicting prices of multiple assets (`price1`, `price2`, `price3`) simultaneously.
- 🧩 Develop insights into **feature engineering**, **temporal alignment**, and **model generalization** in the context of real-world financial markets.

> By integrating news headlines into your forecasting pipeline, you go beyond static historical data and begin modeling the real-world drivers of price behavior. This not only enhances predictive performance but also bridges the gap between machine learning and practical finance, offering valuable insights and tools for building AI-powered trading systems and intelligent market analysis.

---

## 🎯 Your Task

Participants must:

- Design a **custom Transformer-based architecture** adapted for **multivariate time series forecasting**.
- Incorporate **news headlines** as an auxiliary input to improve price prediction.
- Predict three target prices: `price1`, `price2`, and `price3` for each test date.

> 🧠 This is a **multi-label regression problem**.

## 📊 Evaluation

### 🔢 Scoring Description

Your predictions will be evaluated using the **R² score (coefficient of determination)** for each target: `price1`, `price2`, and `price3`.

---

### 📐 R² Formula:

The formula for the R² score is:

\[
R^2 = 1 - \frac{SSR}{SST}
\]

Where:

- **SSR** (Sum of Squared Residuals):  
  \[
  SSR = \sum_{i=1}^{n} (y_i - \hat{y}_i)^2
  \]
  - \( y_i \): actual observed value  
  - \( \hat{y}_i \): predicted value

- **SST** (Total Sum of Squares):  
  \[
  SST = \sum_{i=1}^{n} (y_i - \bar{y})^2
  \]
  - \( \bar{y} \): mean of observed values

The closer the R² score is to **1**, the better your model explains the variance in the target values.

---

### 🏆 Final Score Calculation

The final competition score will be computed as a **weighted average** of the following components:

- **📈 Kaggle Score (65%)**  
  - Mean **R²** across all three predicted prices (`price1`, `price2`, `price3`).

- **📓 Notebook Quality (20%)**  
  - Emphasis on:
    - Innovative modeling
    - Reproducibility
    - Code clarity

- **📝 Short Report (15%)**  
  - **Maximum 10 pages**
  - Should explain:
    - Your approaches
    - Results and key findings
    - How you overcame major challenges
  - **Tip**: The more innovative and insightful you are, the more points you'll gain here.

---

> 🎯 **Your goal:** Maximize the R² score while demonstrating strong reasoning, model design, and communication in your report and notebook.

## 💡 Tips

- 🔄 Consider using **cross-modal attention** to effectively fuse **time series** and **textual data**.
- 🕒 Be mindful of **news timing** — headlines **before or after market hours** may have different impacts on prices.
- 🧠 Experiment with:
  - **Custom temporal encodings**
  - **Multi-task learning** strategies to model shared patterns across price targets.
- ✍️ You may also **process the textual data independently**, then apply **traditional time series analysis (TSA)** techniques as you normally would.

---

🚀 **Good luck**, and may your **Transformer** be as sharp as the **market swings**!
