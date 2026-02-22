"""
CITS — Database Seeder
Optimized version with batch inference and keyword caching.
"""
import os
import sys
import pickle
import pandas as pd
from datetime import datetime

# Ensure project root is on path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from app.database import engine, SessionLocal, init_db
from app.models import Product, Review, ModelStore
from app.utils import clean_text, compute_fake_score
from ml.urgency_extraction import check_urgency_adaptive

def seed():
    print("🔧 Initializing database...")
    init_db()

    db = SessionLocal()

    # 1. Seed Model Store
    print("🤖 Checking Model Store...")
    model_row = db.query(ModelStore).filter(ModelStore.is_active == 1).first()
    if not model_row:
        model_path = os.path.join(BASE_DIR, "model", "cits.pkl")
        if os.path.exists(model_path):
            with open(model_path, "rb") as f:
                model_bytes = f.read()
            model_row = ModelStore(
                version="v1.0", model_data=model_bytes, is_active=1,
                accuracy=0.94, f1_score=0.95
            )
            db.add(model_row)
            db.commit()
            print("   ✅ Initial model seeded.")
        else:
            print("   ⚠️ No model found to seed.")
            model = None
    else:
        print("   ✅ Model already in DB.")
        
    model = pickle.loads(model_row.model_data) if model_row else None

    # 2. Seed Products
    print("📦 Checking Products...")
    if db.query(Product).count() > 0:
        print("   ✅ Products already seeded.")
        db.close()
        return

    csv_path = os.path.join(BASE_DIR, "amazon_vfl_reviews.csv")
    if not os.path.exists(csv_path):
        print(f"   ❌ CSV file not found: {csv_path}")
        db.close()
        return

    print(f"📄 Reading CSV: {csv_path}")
    df = pd.read_csv(csv_path).dropna(subset=["review", "rating"])
    total_rows = len(df)

    # ── 3. High Performance Inference ─────────────────────────
    # We do the expensive ML part in ONE batch outside the loop
    print(f"🧠 Running batch inference on {total_rows} reviews...")
    cleaned_texts = [clean_text(r) for r in df["review"]]
    
    if model:
        # Batch predict sentiments
        sentiments = ["Positive" if p == 1 else "Negative" for p in model.predict(cleaned_texts)]
        # Batch predict probabilities for fake score
        probas = model.predict_proba(cleaned_texts)
    else:
        sentiments = ["Positive"] * total_rows
        probas = [None] * total_rows

    # ── 4. Save Products ──────────────────────────────────────
    print("🏷️ Saving products...")
    product_map = {}
    unique_products = df[["asin", "name"]].drop_duplicates(subset="asin")
    for _, row in unique_products.iterrows():
        p = Product(asin=row["asin"], name=row["name"])
        db.add(p)
        db.flush()
        product_map[row["asin"]] = p.id

    # ── 5. Save Reviews ───────────────────────────────────────
    print("📝 Saving reviews to database...")
    batch_size = 200
    for i, (_, row) in enumerate(df.iterrows()):
        sentiment = sentiments[i]
        
        # Urgency is fast because keywords are cached now
        urgency = check_urgency_adaptive(row["review"])
        is_urgent = urgency["is_urgent"]
        
        # Fake score signal computation
        # Note: model.predict_proba is already done, we just pass the proba
        fake_prob = compute_fake_score(
            row["review"], int(row["rating"]), sentiment, 
            model=None, # Pass None so it doesn't re-run inference inside utils
            cleaned_text=cleaned_texts[i]
        )
        
        # If we have the probability from the batch, use it to refine
        if probas[i] is not None:
            conf = max(probas[i])
            if conf < 0.6: fake_prob = min(fake_prob + 0.2, 1.0)

        try:
            r_date = datetime.strptime(str(row["date"]), "%Y-%m-%d").date()
        except:
            r_date = None

        db.add(Review(
            product_id=product_map[row["asin"]],
            author="Amazon User", rating=int(row["rating"]),
            text=str(row["review"]), sentiment=sentiment,
            priority="High" if is_urgent else "Normal",
            flag="🚨 URGENT" if is_urgent else "✅ OK",
            is_fake_prob=fake_prob, date=r_date
        ))

        if (i + 1) % batch_size == 0:
            db.commit()
            if (i + 1) % 1000 == 0:
                print(f"   Saved {i+1}/{total_rows}...")

    db.commit()
    db.close()
    print("\n✅ Seeding complete!")

if __name__ == "__main__":
    seed()
