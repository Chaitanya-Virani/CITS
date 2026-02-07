# Customer Insight Triage System (CITS) 🚀

> **"Voice of Bharat" Edition** — An AI-powered triage system designed to handle the chaos of Indian E-commerce reviews (Hinglish, slang, and mixed languages).

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.95%2B-green)
![Scikit-Learn](https://img.shields.io/badge/Sklearn-1.2%2B-orange)
![Status](https://img.shields.io/badge/Status-Deployed-success)

## 📖 Overview

In the high-volume world of Indian e-commerce (Flipkart, Amazon, Meesho), millions of customer reviews are generated daily. Relying on human agents to manually read and sort these reviews is slow, expensive, and error-prone. Critical complaints often get buried under thousands of generic "nice product" reviews, leading to poor customer retention and fraud risks.

**CITS (Customer Insight Triage System)** solves this by acting as an intelligent, automated "First Responder." It does not just read text; it understands intent and urgency in real-time.

### How It Works:
The system uses a **Hybrid Intelligence** approach, combining Machine Learning with Rule-Based Logic to process reviews in three stages:

1.  **Sentiment Analysis (The AI Brain):**
    * It uses a **Linear Support Vector Classifier (LinearSVC)** trained on thousands of reviews to instantly determine if a customer is Happy (Positive) or Angry (Negative).
    * It effectively handles "Hinglish" (e.g., *"Product thik hai but delivery late thi"*) by using TF-IDF vectorization to capture context beyond simple keywords.

2.  **Urgency Detection (The Safety Net):**
    * AI models can sometimes be "confused" by sarcasm or complex sentences. To prevent critical failures, CITS implements a **Deterministic Logic Layer**.
    * It scans for high-risk keywords specifically used in the Indian market (e.g., *"fraud", "fake", "tuta hua", "scam", "return"*).

3.  **Automated Triage (The Decision):**
    * Based on the analysis, every review is assigned a **Priority Level**:
        * **🚨 HIGH:** Requires immediate human intervention (Fraud/Damage).
        * **✅ NORMAL:** Can be handled by automated chatbots or standard workflows.

---

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