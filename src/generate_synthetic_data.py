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
    "okish.",
    "meh.",
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

    # Missing items
    "The meal was missing an important ingredient.",
    "The side dish mentioned in the menu was missing.",
    "The gravy was missing today.",

    # Service / timing
    "The service was delayed and the food got cold.",
    "The food was served very late.",
]

# --- NEW: hard/ambiguous patterns to avoid perfect separability ---

MIXED = [
    # mixed sentiment (harder)
    "The food tasted good but it was served too cold.",
    "Good flavors, but the portion was too small.",
    "Nice taste, but it felt too oily.",
    "Fresh meal, but service was delayed.",
    "Quality was okay, but the roti was too hard.",
]

NEGATION_TRAPS = [
    "Not bad, but could be better.",
    "Not good today.",
    "It wasn't great.",
    "It wasn't terrible, just average.",
    "Not spicy enough, not salty enough.",
]

MILD_NEG = [
    "A bit salty.",
    "Slightly oily.",
    "A little too spicy.",
    "Portion felt slightly small.",
    "Texture was a bit off.",
]

TYPO_VARIANTS = {
    "spicy": ["spcy", "spicyy"],
    "salty": ["salti", "saltyy"],
    "greasy": ["greesy", "greasyy"],
    "oily": ["oil", "oilly"],
}

def _maybe_add_item_context(text: str, menu_item: str) -> str:
    # 30% chance: add menu item mention for realism
    if random.random() < 0.30:
        return f"{menu_item} - {text}"
    return text

def _maybe_add_mild_intensity(text: str) -> str:
    # 15% chance: add natural intensifiers
    if random.random() < 0.15:
        return text.replace("too", "a bit too").replace("was", "was a bit")
    return text

def _maybe_add_typos(text: str) -> str:
    # very small chance: add typos to simulate real feedback
    if random.random() < 0.05:
        for word, variants in TYPO_VARIANTS.items():
            if word in text.lower() and random.random() < 0.5:
                text = re_sub_word(text, word, random.choice(variants))
    return text

def re_sub_word(text: str, word: str, repl: str) -> str:
    # simple word replacement (case-insensitive-ish)
    return " ".join([repl if w.lower().strip(".,!?") == word else w for w in text.split()])

def _sample_feedback(label: str) -> str:
    # introduce controlled ambiguity so ML isn't perfect
    if label == "Positive":
        # 10% mixed/negation to create edge cases
        r = random.random()
        if r < 0.06:
            return random.choice(MIXED)
        if r < 0.10:
            return random.choice(NEGATION_TRAPS)
        return random.choice(POSITIVE)

    if label == "Neutral":
        # 15% can look slightly positive/negative
        r = random.random()
        if r < 0.07:
            return random.choice(POSITIVE)
        if r < 0.15:
            return random.choice(MILD_NEG)
        return random.choice(NEUTRAL)

    # Negative
    r = random.random()
    if r < 0.15:
        return random.choice(MIXED)
    if r < 0.25:
        return random.choice(NEGATION_TRAPS)
    if r < 0.35:
        return random.choice(MILD_NEG)
    return random.choice(NEGATIVE)

def generate_dataset(
    n: int = 3000,
    pos_ratio: float = 0.50,
    neu_ratio: float = 0.30,
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
        feedback_text = _sample_feedback(lab)

        feedback_text = _maybe_add_item_context(feedback_text, menu_item)
        feedback_text = _maybe_add_mild_intensity(feedback_text)
        feedback_text = _maybe_add_typos(feedback_text)

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