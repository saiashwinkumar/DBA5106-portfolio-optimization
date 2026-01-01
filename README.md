# Portfolio Optimization: Mitigating Overfitting with Regularization (Ridge/LASSO)

This repository contains the code and experiments for a portfolio optimization project that studies **overfitting in mean–variance optimization** and improves out-of-sample performance using **regularized, regression-based portfolio construction**.

## Objective
The objective is to use a **portfolio optimization (mean–variance) framework** to study and mitigate **overfitting** by reformulating minimum-variance portfolio construction as a **linear regression problem**, then improving out-of-sample performance using **regularized portfolios (LASSO and Ridge)** evaluated via **rolling-window backtesting** and **Sharpe ratio**. 

## Dataset
- Daily value-weighted returns for **100 stock portfolios** (10×10 sorts by Market Equity (ME) and Operating Profits (OP))
- Source: CRSP-based portfolio series (as of June 2025)
- Date range: **1963-07-01 to 2025-06-30** (15,603 trading days)
- Missing values encoded as **-99.99 / -999** 

## Problem Setup
- Rolling backtest with **126-day estimation window**
- Strict out-of-sample validation period: **2025-01-02 to 2025-06-30**
- Baselines:
  - **Equal-Weight (EW)**
  - **Minimum-Variance (MinVar)** under full-investment constraint  
MinVar exhibits strong overfitting in short samples due to noisy/ill-conditioned covariance inversion, producing unstable/extreme weights and poor OOS performance. 

## Methodology
We apply the **Data + Loss + Structure + Constraint** framework to diagnose overfitting and design robust alternatives:

### 1) Regression Reformulation (EW-anchored)
- Anchor on EW portfolio return: `y = R w_EW`
- Regress `y = Xβ + ε` with `X = R N` (N spans the null space of the budget constraint)
- Convert back to portfolio weights: `w = w_EW − Nβ`
This replaces covariance inversion with a more stable regression-based estimation. 

### 2) Models Implemented
- **Baselines:** EW, MinVar
- **EW-anchored regularization:** Ridge, LASSO, ElasticNet, Adaptive LASSO
- **MinVar with weight penalties:** MinVar+Ridge(γ), MinVar+LASSO(γ)
- **Robust losses:** Huber, Pinball/Quantile regression
- **Factor structure:** PCA/PCR + Ridge/LASSO
- **Noise reduction:** PCA-denoised returns + Ridge
- **Meta-learning:** Stacking (PCA+Ridge/LASSO) with and without denoising
- **Alternative selection:** asymmetric utility for downside-aware tuning 

### 3) Evaluation
- Primary metric: **Out-of-sample Sharpe ratio**
- Daily rebalancing with strict no-leakage forecasting
- Additional tests: June 2025 “unseen” holdout evaluation
  
## Key Results (OOS Sharpe: 2025-01-02 to 2025-06-30)
- EW: **0.016**
- MinVar: **-0.132** (overfits)
- EW + Ridge: **0.051**
- EW + LASSO: **0.072**
- PCA + Ridge: **0.083**
- PCA + LASSO: **0.087**
- PCA + LASSO + asymmetric utilities: **0.097**
- PCA + Ridge/LASSO + stacking: **0.095**
- **PCA + Ridge/LASSO + stacking + denoising: 0.202 (best)** 

Main insight: Overfitting is driven primarily by **estimation variance/noise**, not outliers—robust losses (Huber/Pinball) did not help, while **factor structure + denoising + shrinkage** delivered the strongest gains. 

## June 2025 Unseen Test (Holdout)
Using the same pipeline trained on the last 126 trading days ending 2025-05-31:
- Selected PCA+Ridge via inner validation (validation Sharpe ≈ 0.205)
- Achieved **June OOS Sharpe 0.116 (annualized Sharpe 1.842; annualized return 29.35%)**
A refit-daily June variant achieved higher Sharpe but with high turnover (~0.60/day), implying sensitivity to trading costs. 

## Reproducibility
Run the notebooks/scripts end-to-end to reproduce:
- rolling-window backtests (126-day estimation)
- baseline EW and MinVar portfolios
- EW-anchored regression portfolios (Ridge/LASSO/ElasticNet/Adaptive LASSO)
- PCA/PCR factor models, SVD denoising, stacking meta-learner
- OOS Sharpe comparison tables and wealth curves

*Footnote:* This report extends the portfolio-optimization study by **retesting the Project 1 chosen strategy (Stacked + Denoised PCA ensemble, k=10)** against **LASSOCV and RidgeCV** on a **strict unseen OOS window (1 Aug–30 Sep 2025)** under the **same daily-rolling 126-day backtest** with **inner-window tuning for fairness**. 
It finds the chosen model achieves the **highest daily Sharpe (0.308)**, outperforming **EW (0.146)**, **LASSOCV (0.273)** and **RidgeCV (0.275)**, confirming the ensemble’s robustness on new data.

