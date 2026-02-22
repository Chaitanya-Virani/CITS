"""
CITS — Database Seeder
Imports amazon_vfl_reviews.csv and the initial ML model into the database.
Supports both PostgreSQL and SQLite.
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

    # 1. Seed Model Store if empty
    print("🤖 Checking Model Store...")
    if db.query(ModelStore).count() == 0:
        model_path = os.path.join(BASE_DIR, "model", "cits.pkl")
        if os.path.exists(model_path):
            print(f"   Loading model from disk: {model_path}")
            with open(model_path, "rb") as f:
                model_bytes = f.read()
                model = pickle.loads(model_bytes)
            
            initial_model = ModelStore(
                version="v1.0",
                model_data=model_bytes,
                is_active=1,
                accuracy=0.94,  # Placeholder for initial seed
                f1_score=0.95
            )
            db.add(initial_model)
            db.commit()
            print("   ✅ Initial model (v1.0) seeded to DB.")
        else:
            print("   ⚠️ No cits.pkl found on disk to seed.")
    else:
        print("   ✅ Model Store already has models.")

    # 2. Seed Products
    print("📦 Checking Products...")
    if db.query(Product).count() > 0:
        print("   ✅ Products already seeded. Skipping CSV import.")
        db.close()
        return

    csv_path = os.path.join(BASE_DIR, "amazon_vfl_reviews.csv")
    if not os.path.exists(csv_path):
        print(f"   ❌ CSV file not found: {csv_path}")
        db.close()
        return

    print(f"📄 Reading CSV: {csv_path}")
    df = pd.read_csv(csv_path).dropna(subset=["review", "rating"])
    
    # Load model for inference during seeding
    active_model_row = db.query(ModelStore).filter(ModelStore.is_active == 1).first()
    model = pickle.loads(active_model_row.model_data) if active_model_row else None

    product_map = {}
    unique_products = df[["asin", "name"]].drop_duplicates(subset="asin")

    for _, row in unique_products.iterrows():
        product = Product(asin=row["asin"], name=row["name"])
        db.add(product)
        db.flush()
        product_map[row["asin"]] = product.id
    
    # 3. Seed Reviews
    print("📝 Processing reviews...")
    batch_size = 100
    for i, (_, row) in enumerate(df.iterrows()):
        cleaned = clean_text(row["review"])
        sentiment = "Positive"
        fake_prob = 0.0
        
        if model:
            sentiment = "Positive" if model.predict([cleaned])[0] == 1 else "Negative"
            fake_prob = compute_fake_score(row["review"], int(row["rating"]), sentiment, model, cleaned)

        urgency = check_urgency_adaptive(row["review"])
        is_urgent = urgency["is_urgent"]
        
        try:
            r_date = datetime.strptime(str(row["date"]), "%Y-%m-%d").date()
        except:
            r_date = None

        db.add(Review(
            product_id=product_map[row["asin"]],
            author="Amazon User",
            rating=int(row["rating"]),
            text=str(row["review"]),
            sentiment=sentiment,
            priority="High" if is_urgent else "Normal",
            flag="🚨 URGENT" if is_urgent else "✅ OK",
            is_fake_prob=fake_prob,
            date=r_date
        ))

        if (i + 1) % batch_size == 0:
            db.commit()
            print(f"   Processed {i+1}/{len(df)}...")

    db.commit()
    db.close()
    print("\n✅ Seeding complete!")

if __name__ == "__main__":
    seed()
