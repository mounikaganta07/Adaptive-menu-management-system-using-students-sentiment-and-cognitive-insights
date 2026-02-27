import random
from datetime import datetime, timedelta
import pandas as pd


MENU_ITEMS = [
    "Idli", "Dosa", "Pongal", "Sambar Rice", "Curd Rice",
    "Veg Curry", "Chicken Biryani", "Veg Biryani", "Roti",
    "Paneer Curry", "Samosa", "Gulab Jamun", "Ice Cream",
    "Uggani", "Bonda", "Vada", "Fish Curry", "Mutton Curry"
]

MEAL_TIMES = ["Breakfast", "Lunch", "Snacks", "Dinner"]

POSITIVE = [
    "The food tasted great today.",
    "Really enjoyed the meal, it was delicious.",
    "Good quality and well cooked.",
    "Fresh and tasty, good job by the kitchen.",
    "Portion size was perfect and taste was nice.",
    "Amazing flavors, I loved the taste.",
    "The meal was light and refreshing.",
]

NEUTRAL = [
    "The meal was okay.",
    "Taste was average, nothing special.",
    "It was fine, could be better.",
    "Neither good nor bad, just normal.",
    "Decent meal overall.",
]

NEGATIVE = [
    # Temperature
    "The food was served too cold.",
    "The meal was cold and unpleasant.",
    "The food was too hot and overcooked.",
    "The temperature of the food was not good.",

    # Taste / seasoning
    "The meal was too salty.",
    "The meal was not salty enough.",
    "The food was too spicy.",
    "The flavors were bland and needed more seasoning.",
    "The food tasted strange today.",

    # Oil / digestion
    "The food was too oily.",
    "The curry had too much oil floating on top.",
    "The food felt heavy and greasy.",

    # Texture / cooking
    "The food was not cooked properly.",
    "The rice was undercooked.",
    "The vegetables were overcooked and mushy.",
    "The texture of the dish was not good.",
    "The roti was too hard.",

    # Quality / freshness
    "The overall quality was poor today.",
    "The taste felt stale.",
    "The food smelled bad and felt old.",
    "The quality was inconsistent this week.",

    # Portion / quantity
    "The portion size was too small.",
    "The quantity was not enough.",
    "The serving was not sufficient.",

    # Missing items (few, not dominant)
    "The meal was missing an important ingredient.",
    "The side dish mentioned in the menu was missing.",
    "The gravy was missing today.",

    # Service / timing
    "The service was delayed and the food got cold.",
    "The food was served very late.",
]


def generate_dataset(
    n: int = 3000,
    pos_ratio: float = 0.70,
    neu_ratio: float = 0.25,
    seed: int = 42,
) -> pd.DataFrame:
    random.seed(seed)

    pos_n = int(n * pos_ratio)
    neu_n = int(n * neu_ratio)
    neg_n = n - pos_n - neu_n

    labels = (["Positive"] * pos_n) + (["Neutral"] * neu_n) + (["Negative"] * neg_n)
    random.shuffle(labels)

    start = datetime(2024, 1, 1)
    end = datetime(2026, 2, 27)
    days = (end - start).days

    rows = []
    for i in range(n):
        menu_item = random.choice(MENU_ITEMS)
        meal_time = random.choice(MEAL_TIMES)

        lab = labels[i]
        if lab == "Positive":
            feedback_text = random.choice(POSITIVE)
        elif lab == "Neutral":
            feedback_text = random.choice(NEUTRAL)
        else:
            feedback_text = random.choice(NEGATIVE)

        ts = start + timedelta(days=random.randint(0, days))

        rows.append({
            "feedback_text": feedback_text,
            "feedback_timestamp": ts.isoformat(),
            "menu_item": menu_item,
            "meal_time": meal_time,
        })

    return pd.DataFrame(rows)


if __name__ == "__main__":
    df = generate_dataset(n=3000)
    out_path = "data/raw/synthetic_feedback_data.csv"
    df.to_csv(out_path, index=False)
    print(f"✅ Generated {out_path} with {len(df)} rows")