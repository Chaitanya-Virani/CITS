# Customer Insight Triage System (CITS) 🚀

> **"Voice of Bharat" Edition** — A full-stack AI-powered triage system for Indian e-commerce reviews, featuring real-time sentiment analysis, fake review detection, and an admin analytics dashboard.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.95%2B-green)
![Scikit-Learn](https://img.shields.io/badge/Sklearn-1.8%2B-orange)
![SQLite](https://img.shields.io/badge/SQLite-3-lightgrey)
![Status](https://img.shields.io/badge/Status-Production_Ready-success)

---

## 📖 Overview

In the high-volume world of Indian e-commerce (Flipkart, Amazon, Meesho), millions of customer reviews are generated daily. Critical complaints often get buried under thousands of generic "nice product" reviews.

**CITS** acts as an intelligent, automated **"First Responder"** — it doesn't just read text; it understands intent, urgency, and authenticity in real-time.

### How It Works

The system uses a **Hybrid Intelligence** approach across three stages:

1. **Sentiment Analysis (The AI Brain):**
   - Uses a **TF-IDF + Logistic Regression** pipeline trained on thousands of reviews
   - Handles Hinglish (e.g., *"Product thik hai but delivery late thi"*) via TF-IDF n-gram vectorization

2. **Urgency Detection (The Safety Net):**
   - Deterministic rule-based layer scanning for high-risk keywords (*"fraud", "fake", "tuta hua", "scam", "return"*)
   - Catches cases AI models might miss (sarcasm, complex sentences)

3. **Automated Triage (The Decision):**
   - 🚨 **HIGH** — Requires immediate human intervention (fraud/damage)
   - ✅ **NORMAL** — Can be handled by automated workflows

---

## 🖥️ Features

### User Page (`/user`)
- **Product selector** — browse 145+ seeded products
- **Trust score** — computed from sentiment ratio, review volume, and anomaly detection
- **Rating breakdown** — visual 1–5 star distribution
- **AI summary** — rule-based analysis with keyword extraction
- **Review submission** — real-time ML inference (sentiment + fake probability)
- **Scrollable review list** — with sentiment badges and urgency flags

### Admin Dashboard (`/admin`)
- **Model metrics** — accuracy, F1, precision, recall (real evaluation on DB data)
- **Dataset health** — percentage of clean vs rejected reviews
- **Sentiment distribution chart** — review count per star rating
- **Fake probability chart** — histogram of ML-predicted fake scores
- **Drift detection** — compares recent vs. historical sentiment ratios
- **Performance comparison table** — current model vs. candidate
- **One-click retraining** — rebuilds pipeline from all DB reviews

### Admin Reviews Page (`/admin/reviews`)
- **Product-wise breakdown** — positive/negative counts per product
- **Visual sentiment bars** — green/red ratio indicators
- **Expandable negative reviews** — click to see full customer feedback
- **Sort controls** — sort by most/least positive, most/least negative, total reviews
- **Search** — filter products by name or ASIN

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| Backend | FastAPI + Uvicorn |
| Database | SQLite + SQLAlchemy ORM |
| ML | Scikit-Learn (TF-IDF + LogisticRegression) |
| Data | Pandas, NumPy |
| Validation | Pydantic v2 |
| Frontend | Vanilla HTML/CSS/JS (no frameworks) |
| Theme | Dark/light mode with localStorage persistence |

---

## 📂 Project Structure

```text
Customer Insight Triage System/
├── app/
│   ├── __init__.py
│   ├── main.py              # FastAPI app — routes, static files, templates
│   ├── database.py          # SQLite + SQLAlchemy engine, sessions
│   ├── models.py            # ORM models (Product, Review)
│   ├── schemas.py           # Pydantic request/response models
│   ├── utils.py             # Trust score, summary, drift, text cleaning
│   ├── seed.py              # One-time CSV → DB importer
│   └── routes/
│       ├── __init__.py
│       ├── products.py      # GET /api/products, /api/product/{id}
│       ├── reviews.py       # GET /api/reviews/{id}, POST /api/reviews
│       ├── summary.py       # GET /api/summary/{id}
│       └── admin.py         # GET /api/model-metrics, /api/drift, POST /api/retrain
├── model/
│   └── cits.pkl             # Trained ML model (auto-updated on retrain)
├── data/
│   └── cits.db              # SQLite database (created by seed.py)
├── templates/
│   ├── user.html            # Public product analysis page
│   ├── admin.html           # Admin dashboard
│   └── reviews.html         # Admin reviews breakdown page
├── static/
│   ├── css/styles.css       # Full design system (dark/light themes)
│   └── js/
│       ├── main.js          # User page logic
│       ├── dashboard.js     # Admin dashboard logic
│       └── reviews.js       # Admin reviews page logic
├── amazon_vfl_reviews.csv   # Source dataset (used only for seeding)
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 🚀 Quick Start

### 1. Setup

```bash
# Clone the repo
git clone https://github.com/your-username/CITS.git
cd "Customer Insight Triage System"

# Create virtual environment
python -m venv .venv
source .venv/bin/activate   # Linux/Mac
# .venv\Scripts\activate    # Windows

# Install dependencies
pip install -r requirements.txt
```

### 2. Seed the Database

```bash
python -m app.seed
```

This reads `amazon_vfl_reviews.csv`, runs each review through the ML model, and populates the SQLite database with **145 products** and **~2,800 reviews**.

### 3. Run the Server

```bash
uvicorn app.main:app --reload
```

### 4. Open in Browser

| Page | URL |
|------|-----|
| User (Product Analysis) | [http://localhost:8000/user](http://localhost:8000/user) |
| Admin Dashboard | [http://localhost:8000/admin](http://localhost:8000/admin) |
| Admin Reviews | [http://localhost:8000/admin/reviews](http://localhost:8000/admin/reviews) |

---

## 📡 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/products` | List all products (id, name, asin) |
| `GET` | `/api/product/{id}` | Product details + trust score |
| `GET` | `/api/reviews/{id}` | Reviews for a product + rating breakdown |
| `POST` | `/api/reviews` | Submit a review (returns sentiment + priority) |
| `GET` | `/api/summary/{id}` | AI-generated summary for a product |
| `GET` | `/api/model-metrics` | Model accuracy, F1, distributions |
| `GET` | `/api/drift` | Drift detection + dataset health |
| `POST` | `/api/retrain` | Retrain model on all DB reviews |
| `GET` | `/api/admin/reviews` | Product-wise +ve/-ve breakdown |

---

## 📄 License

This project is for educational and portfolio purposes.