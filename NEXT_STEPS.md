# Next Development Plan

This plan focuses on turning the dashboard from a CV-ready prototype into a stronger portfolio analytics product.

## Phase 1: Reliability

1. Done: Add app-level smoke tests for the main calculations used in `app.py`.
2. Done: Add validation for impossible or suspicious inputs, such as too-short date ranges and very concentrated portfolios.
3. Done: Add graceful messages for one-asset portfolios, invalid tickers, and partial Yahoo Finance downloads.
4. Done: Add a small sample dataset so the app can run fully offline without generated prices.

## Phase 2: Better User Decisions

1. Done: Add a plain-language explanation panel for Sharpe, volatility, drawdown, VaR, and beta.
2. Done: Add portfolio concentration checks, such as top holding weight and effective number of assets.
3. Done: Add benchmark comparison against a selected market index.
4. Done: Add risk contribution by asset so users can see which position creates the most portfolio risk.

## Phase 3: Stronger Analytics

1. Done: Add rebalancing analysis: monthly, quarterly, and yearly.
2. Done: Add rolling return and rolling Sharpe charts.
3. Done: Add stress tests for selected crisis windows.
4. Done: Add transaction cost assumptions to optimization outputs.

## Phase 4: Product Polish

1. Done: Add export buttons for summary tables.
2. Done: Add an HTML report generator.
3. Done: Add preset portfolios such as balanced, growth, defensive, and inflation hedge.
4. Done: Add a short landing-free onboarding state for first-time users.

## Phase 5: Deployment

1. Done: Add environment and dependency notes for Streamlit Cloud or Render.
2. Done: Add CI that runs tests on every push.
3. Done: Add screenshots to the README.
4. Done: Add a short project demo video or GIF plan for GitHub and CV use.
