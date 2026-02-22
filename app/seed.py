"""
CITS — Database Seeder
Imports amazon_vfl_reviews.csv into the SQLite database.
Run once: python -m app.seed
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
from app.models import Product, Review
from app.utils import clean_text, check_urgency


def seed():
    print("🔧 Initializing database...")
    init_db()

    db = SessionLocal()

    # Check if already seeded
    existing = db.query(Product).count()
    if existing > 0:
        print(f"⚠️  Database already has {existing} products. Skipping seed.")
        print("   To re-seed, delete data/cits.db and run again.")
        db.close()
        return

    # Load CSV
    csv_path = os.path.join(BASE_DIR, "amazon_vfl_reviews.csv")
    print(f"📄 Reading CSV: {csv_path}")
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=["review", "rating"])
    print(f"   {len(df)} reviews loaded")

    # Load ML model
    model_path = os.path.join(BASE_DIR, "model", "cits.pkl")
    print(f"🤖 Loading model: {model_path}")
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    print("   Model loaded")

    # --- Create Products ---
    print("📦 Creating products...")
    product_map = {}  # asin -> Product.id
    unique_products = df[["asin", "name"]].drop_duplicates(subset="asin")

    for _, row in unique_products.iterrows():
        product = Product(
            asin=row["asin"],
            name=row["name"],
            category="General",
        )
        db.add(product)
        db.flush()  # Get the ID
        product_map[row["asin"]] = product.id

    print(f"   {len(product_map)} products created")

    # --- Create Reviews ---
    print("📝 Processing reviews through ML model...")
    batch_size = 100
    total = len(df)

    for i, (_, row) in enumerate(df.iterrows()):
        cleaned = clean_text(row["review"])
        prediction = model.predict([cleaned])[0]
        sentiment = "Positive" if prediction == 1 else "Negative"

        # Fake probability
        try:
            proba = model.predict_proba([cleaned])[0]
            # proba[1] = probability of positive class
            # Use the lower class probability as "fake" indicator
            fake_prob = round(float(min(proba)), 4)
        except Exception:
            fake_prob = 0.0

        urgency = check_urgency(row["review"])
        is_urgent = urgency["is_urgent"]
        priority = "High" if is_urgent else "Normal"
        flag = "🚨 URGENT" if is_urgent else "✅ OK"

        # Parse date
        try:
            review_date = datetime.strptime(str(row["date"]), "%Y-%m-%d").date()
        except Exception:
            review_date = None

        review = Review(
            product_id=product_map[row["asin"]],
            author="Amazon User",
            rating=int(row["rating"]),
            text=str(row["review"]),
            sentiment=sentiment,
            priority=priority,
            flag=flag,
            is_fake_prob=fake_prob,
            date=review_date,
        )
        db.add(review)

        if (i + 1) % batch_size == 0:
            db.commit()
            print(f"   Processed {i + 1}/{total} reviews...")

    db.commit()
    db.close()
    print(f"\n✅ Seeding complete! {total} reviews across {len(product_map)} products.")


if __name__ == "__main__":
    seed()
