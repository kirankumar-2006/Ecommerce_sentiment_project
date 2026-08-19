

import random
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

random.seed(42)
np.random.seed(42)

CATEGORIES = ["Electronics", "Fashion", "Home & Kitchen", "Beauty",
              "Sports & Outdoors", "Books", "Toys & Games", "Grocery"]

PRODUCTS = {
    "Electronics": ["Wireless Earbuds", "Smartwatch", "Bluetooth Speaker", "Laptop Stand", "Power Bank"],
    "Fashion": ["Running Shoes", "Denim Jacket", "Cotton T-Shirt", "Leather Wallet", "Sunglasses"],
    "Home & Kitchen": ["Air Fryer", "Non-stick Pan Set", "Vacuum Cleaner", "LED Desk Lamp", "Coffee Maker"],
    "Beauty": ["Face Serum", "Hair Dryer", "Lipstick Set", "Sunscreen SPF50", "Electric Trimmer"],
    "Sports & Outdoors": ["Yoga Mat", "Dumbbell Set", "Camping Tent", "Cycling Helmet", "Water Bottle"],
    "Books": ["Mystery Novel", "Self-Help Guide", "Cookbook", "Fantasy Trilogy", "Biography"],
    "Toys & Games": ["Building Blocks Set", "Remote Control Car", "Puzzle 1000pc", "Board Game", "Action Figure"],
    "Grocery": ["Organic Almonds", "Green Tea Pack", "Protein Powder", "Olive Oil", "Breakfast Cereal"],
}

# --- Sentence banks used to compose reviews with a known ground-truth sentiment ---
POSITIVE_SNIPPETS = [
    "Absolutely love this product, exceeded my expectations!",
    "Great quality for the price, would buy again.",
    "Fast delivery and excellent packaging.",
    "Works perfectly, exactly as described.",
    "Amazing customer service when I had a question.",
    "Best purchase I've made this year.",
    "The build quality is fantastic and it feels premium.",
    "Highly recommend this to anyone looking for a reliable option.",
    "Super easy to use and set up.",
    "Value for money, very satisfied with this purchase.",
    "Five stars, no complaints at all.",
    "My family loves it, we use it every day now.",
]

NEGATIVE_SNIPPETS = [
    "Very disappointed, it broke after just two days.",
    "Terrible quality, not worth the money at all.",
    "Arrived late and the packaging was damaged.",
    "Does not match the description on the website.",
    "Customer support was unhelpful and slow to respond.",
    "Worst purchase I've made in a long time.",
    "Feels cheap and flimsy, expected much better.",
    "Would not recommend this to anyone.",
    "Difficult to set up and the instructions were unclear.",
    "Overpriced for what you actually get.",
    "One star, complete waste of money.",
    "Stopped working within a week of normal use.",
]

NEUTRAL_SNIPPETS = [
    "It's okay, does the job but nothing special.",
    "Average product, matches the price point.",
    "Delivery was on time, product is as expected.",
    "Not bad, but there are probably better alternatives.",
    "It works fine, though the design could be improved.",
    "Decent for casual use, wouldn't use it professionally.",
    "Met basic expectations, no major issues so far.",
    "Fair quality, would consider buying similar items.",
    "It's fine for the price, nothing to complain about.",
    "Reasonable product, packaging could be better.",
]


FILLER_SNIPPETS = [
    "Ordered this for a gift.",
    "Bought it during a sale.",
    "This was my first time buying from this brand.",
    "Saw a lot of reviews before purchasing.",
    "Using it mostly on weekends.",
    "Compared a few options before choosing this one.",
]


def compose_review(sentiment):
    if sentiment == "positive":
        bank = POSITIVE_SNIPPETS
        n = random.choice([1, 2])
    elif sentiment == "negative":
        bank = NEGATIVE_SNIPPETS
        n = random.choice([1, 2])
    else:
        bank = NEUTRAL_SNIPPETS
        n = 1

    parts = random.sample(bank, n)

    # Occasionally add a sentiment-neutral filler sentence (real reviews
    # ramble) and, less often, a mildly conflicting opposite-sentiment
    # remark ("mixed" reviews) so the classification task isn't trivially
    # separable, mirroring real-world review noise.
    if random.random() < 0.35:
        parts.append(random.choice(FILLER_SNIPPETS))
    if random.random() < 0.12:
        other_bank = NEGATIVE_SNIPPETS if sentiment == "positive" else (
            POSITIVE_SNIPPETS if sentiment == "negative" else random.choice([POSITIVE_SNIPPETS, NEGATIVE_SNIPPETS]))
        parts.append(random.choice(other_bank))

    random.shuffle(parts)
    text = " ".join(parts)

    # Light typo/casing noise to mimic real user-generated text
    if random.random() < 0.15:
        text = text.lower()
    if random.random() < 0.1:
        text = text.replace("great", "gr8").replace("okay", "ok")

    return text


def sentiment_to_rating(sentiment):
    if sentiment == "positive":
        return random.choices([4, 5], weights=[0.3, 0.7])[0]
    elif sentiment == "negative":
        return random.choices([1, 2], weights=[0.6, 0.4])[0]
    else:
        return 3


def generate_dataset(n_rows=3000, start_date="2025-01-01", end_date="2026-07-01"):
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")
    span_days = (end - start).days

    rows = []
    # Slight class imbalance to mimic real-world review distributions
    sentiment_choices = ["positive"] * 55 + ["negative"] * 25 + ["neutral"] * 20

    for i in range(n_rows):
        sentiment = random.choice(sentiment_choices)
        category = random.choice(CATEGORIES)
        product = random.choice(PRODUCTS[category])
        review_text = compose_review(sentiment)
        rating = sentiment_to_rating(sentiment)
        review_date = start + timedelta(days=random.randint(0, span_days))

        rows.append({
            "review_id": f"REV{i+10000}",
            "date": review_date.strftime("%Y-%m-%d"),
            "category": category,
            "product": product,
            "review_text": review_text,
            "rating": rating,
            "sentiment": sentiment,  # ground truth label (used for training)
        })

    df = pd.DataFrame(rows)
    df = df.sort_values("date").reset_index(drop=True)
    return df


if __name__ == "__main__":

    import os

    # Generate dataset
    df = generate_dataset()

    # Get project root directory
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # Create data folder if it doesn't exist
    DATA_DIR = os.path.join(PROJECT_ROOT, "data")
    os.makedirs(DATA_DIR, exist_ok=True)

    # Output CSV path
    out_path = os.path.join(DATA_DIR, "reviews.csv")

    # Save dataset
    df.to_csv(out_path, index=False)

    # Print information
    print(f"Generated {len(df)} reviews -> {out_path}")
    print()
    print("Sentiment Distribution:")
    print(df["sentiment"].value_counts())
    print()
    print("First 5 Records:")
    print(df.head())
