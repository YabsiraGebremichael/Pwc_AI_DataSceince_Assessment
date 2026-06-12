# GameVault Publishing — Acquisition Analysis

## Overview
This project analyses Steam player review data to help GameVault Publishing 
identify games with strong licensing and acquisition potential. The analysis 
moves from raw review data through cleaning, feature engineering, sentiment 
analysis, and acquisition scoring to produce a ranked shortlist of candidate 
titles.

## The Problem
GameVault currently relies on surface-level metrics, manual browsing, and has 
no centralised view for comparing acquisition targets. This project addresses 
all three pain points through a data-driven scoring system and interactive 
dashboard.

## Approach
1. **Data Cleaning** — Handled missing values, removed duplicates, converted 
data types, and capped outliers in hours played
2. **Feature Engineering** — Aggregated review-level data to game level, 
creating engagement and longevity metrics including recommendation rate, 
average hours played, review period days, and reviews per month
3. **Sentiment Analysis** — Applied TextBlob to extract average sentiment 
polarity per game from review text
4. **Acquisition Scoring** — Designed a composite 0–100 score combining 
recommendation rate, engagement depth, review volume, sentiment, and longevity 
using min-max scaling
5. **EDA** — Visualised distributions, engagement patterns, and correlations 
to validate scoring approach
6. **Dashboard** — Built an interactive Streamlit app for GameVault to explore 
and filter candidates

## Key Findings
- **Terraria** and **Left 4 Dead 2** are the top acquisition candidates, 
scoring 73.9 and 72.8 respectively, combining near-perfect recommendation 
rates, 400+ average hours played, and review periods spanning over 7 years
- Recommendation rate alone is a misleading signal — several highly rated 
games show low average playtime, indicating approval without deep engagement
- Games with review periods exceeding 365 days represent lower acquisition 
risk due to established, sustained communities

## Recommendations
1. Prioritise Terraria and Left 4 Dead 2 as anchor titles for the catalogue
2. Flag any game where sentiment score lags significantly behind recommendation 
rate — this divergence warrants manual review before deal finalisation
3. Weight acquisition budget toward titles with review periods over 365 days 
to minimise engagement risk

## Repository Structure
