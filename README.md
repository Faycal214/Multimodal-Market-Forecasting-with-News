# NewsAware Market Forecasting

Multimodal machine learning system for financial forecasting using:
- historical time series
- financial news headlines
- a PyTorch-based fusion model
- FastAPI for inference
- Docker and GitHub Actions for production-style delivery

This project is based on a Kaggle challenge where the task was to combine market data and news headlines to predict `price1`, `price2`, and `price3`. The original notebook used pre/post-market tagging, daily news aggregation, lag features, a BERT-based textual branch, a temporal branch, and R² evaluation. :contentReference[oaicite:1]{index=1}

## Project idea

The model learns from:
- past prices
- news headlines
- engineered calendar and text statistics

It then predicts the next target prices as a multi-output regression problem.

## Why this project is useful

This repository is designed to show:
- practical NLP + time series modeling
- multimodal feature engineering
- reproducible training
- deployment with FastAPI
- containerization with Docker
- automated testing with GitHub Actions

## Repository structure

```text
project/
├── api/
├── data/
├── models/
├── notebooks/
├── reports/
├── src/
├── tests/
├── Dockerfile
├── requirements.txt
└── README.md