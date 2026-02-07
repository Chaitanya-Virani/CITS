# Customer Insight Triage System (CITS) 🚀

> **"Voice of Bharat" Edition** — An AI-powered triage system designed to handle the chaos of Indian E-commerce reviews (Hinglish, slang, and mixed languages).

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.95%2B-green)
![Scikit-Learn](https://img.shields.io/badge/Sklearn-1.2%2B-orange)
![Status](https://img.shields.io/badge/Status-Deployed-success)

## 📖 Overview
In the high-volume world of Indian e-commerce (Flipkart/Amazon), human agents cannot read every review. 
**CITS** acts as an intelligent first line of defense. It automatically:
1.  **Analyzes Sentiment:** Determines if a customer is Happy (Positive) or Angry (Negative).
2.  **Detects Urgency:** Uses a specialized keyword dictionary to flag high-risk cases like "Scam", "Fake", "Broken", or "Refund" immediately.
3.  **Prioritizes:** Sorts reviews into `High Priority` (Escalate to Human) vs. `Normal` (Automated Reply).

## 🛠️ Tech Stack
* **Core Logic:** Python 3.x
* **Machine Learning:** Scikit-Learn (LinearSVC + TF-IDF Vectorization)
* **API Framework:** FastAPI + Uvicorn
* **Data Processing:** Pandas & NumPy
* **Validation:** Pydantic

## 📂 Project Structure
```text
Customer Insight Triage System/
├── app/
│   ├── __init__.py
│   ├── main.py          # The FastAPI Server (Entry Point)
│   ├── schemas.py       # Pydantic Models (Input/Output definitions)
│   └── utils.py         # Cleaning & Urgency Detection Logic
├── models/
│   └── cits.pkl         # The Trained Model (LinearSVC)
├── requirements.txt     # Dependency List
├── .gitignore           # Files to ignore (datasets, venv)
└── README.md            # Documentation