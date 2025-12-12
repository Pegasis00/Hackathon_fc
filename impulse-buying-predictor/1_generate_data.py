"""
Improved Data Generation Script
Generates realistic e-commerce impulse buying dataset
"""

import numpy as np
import pandas as pd
import os
from datetime import datetime, timedelta

np.random.seed(42)

# Configuration
N_USERS = 50000
N_DAYS = 90
RAW_DIR = "data/raw"

# User Personas
PERSONA_NAMES = np.array([
    "impulse_buyer", "deal_hunter", "steady_buyer",
    "window_shopper", "premium_user", "cautious_saver"
])

PERSONA_PCT = np.array([0.18, 0.22, 0.20, 0.25, 0.10, 0.05])

INCOME_CLUSTERS = {
    "impulse_buyer":        (50000, 12000),
    "deal_hunter":          (35000, 8000),
    "steady_buyer":         (45000, 10000),
    "window_shopper":       (32000, 7000),
    "premium_user":         (120000, 30000),
    "cautious_saver":       (20000, 5000)
}

NIGHT_RATIOS = {
    "impulse_buyer": 0.45,
    "deal_hunter": 0.30,
    "steady_buyer": 0.20,
    "window_shopper": 0.25,
    "premium_user": 0.15,
    "cautious_saver": 0.35
}

STRESS_MEAN = {
    "impulse_buyer": 7.5,
    "deal_hunter": 5.0,
    "steady_buyer": 4.0,
    "window_shopper": 6.0,
    "premium_user": 3.5,
    "cautious_saver": 7.0
}


def generate_user_profile():
    """Generate user demographic data"""
    print("Generating user profiles...")
    
    personas = np.random.choice(PERSONA_NAMES, size=N_USERS, p=PERSONA_PCT)
    ages = np.random.randint(18, 65, N_USERS)
    genders = np.random.choice(["Male", "Female", "Other"], size=N_USERS, p=[0.48, 0.48, 0.04])
    cities = np.random.choice(
        ["Mumbai", "Pune", "Delhi", "Bengaluru", "Hyderabad", "Chennai", "Kolkata"], 
        size=N_USERS
    )
    payment = np.random.choice(
        ["UPI", "Card", "COD", "Wallet"], 
        size=N_USERS, 
        p=[0.6, 0.25, 0.1, 0.05]
    )
    acc_age = np.random.randint(30, 2000, N_USERS)

    # Income based on persona
    income = np.zeros(N_USERS)
    for persona in PERSONA_NAMES:
        idx = (personas == persona)
        mean, sd = INCOME_CLUSTERS[persona]
        income[idx] = np.random.normal(mean, sd, idx.sum())

    df = pd.DataFrame({
        "user_id": np.arange(1, N_USERS + 1),
        "persona": personas,
        "age": ages,
        "monthly_income": np.clip(income, 10000, 300000),
        "account_age_days": acc_age,
        "gender": genders,
        "default_payment_method": payment,
        "city": cities
    })

    return df


def generate_browsing_logs(user_df):
    """Generate browsing behavior data"""
    print("Generating browsing logs...")

    base_date = pd.Timestamp.today().normalize()
    user_ids = user_df["user_id"].values
    personas = user_df["persona"].values

    sessions_per_user_day = np.random.poisson(lam=1.2, size=(N_USERS, N_DAYS))
    sessions_per_user_day = np.clip(sessions_per_user_day, 0, 3)

    total_sessions = sessions_per_user_day.sum()
    print(f"  Total browsing sessions: {total_sessions:,}")

    user_repeat = np.repeat(user_ids, sessions_per_user_day.sum(axis=1))
    persona_repeat = np.repeat(personas, sessions_per_user_day.sum(axis=1))

    # Generate timestamps
    day_offsets = np.random.randint(0, N_DAYS, len(user_repeat))
    ts = base_date - pd.to_timedelta(N_DAYS - day_offsets, unit="D")

    # Time of day based on persona
    night_ratio_map = {p: NIGHT_RATIOS[p] for p in PERSONA_NAMES}
    hours = []
    for persona in persona_repeat:
        if np.random.rand() < night_ratio_map[persona]:
            hours.append(np.random.choice([22, 23, 0, 1, 2, 3]))
        else:
            hours.append(np.random.randint(8, 22))
    hours = np.array(hours)

    timestamps = ts + pd.to_timedelta(hours, unit="h") + \
                 pd.to_timedelta(np.random.randint(0, 60, len(hours)), unit="m")

    page_types = np.random.choice(
        ["home", "product", "cart", "checkout"],
        p=[0.35, 0.45, 0.12, 0.08],
        size=len(hours)
    )

    time_spent = np.clip(np.random.exponential(scale=60, size=len(hours)), 5, 600)
    devices = np.random.choice(
        ["mobile", "desktop", "tablet"], 
        size=len(hours), 
        p=[0.75, 0.20, 0.05]
    )

    df = pd.DataFrame({
        "session_id": np.arange(1, len(hours) + 1),
        "user_id": user_repeat,
        "timestamp": timestamps,
        "page_type": page_types,
        "time_spent_seconds": time_spent,
        "device_type": devices
    })

    return df


def generate_psychology_survey(user_df):
    """Generate psychological factors"""
    print("Generating psychology data...")
    
    personas = user_df["persona"].values
    stress = np.zeros(N_USERS)
    
    for p in PERSONA_NAMES:
        idx = personas == p
        stress[idx] = np.random.normal(STRESS_MEAN[p], 1.5, idx.sum())

    stress = np.clip(stress, 1, 10)

    mood = np.random.choice(
        ["Happy", "Neutral", "Sad", "Anxious"], 
        size=N_USERS, 
        p=[0.35, 0.35, 0.15, 0.15]
    )
    
    saving = np.random.randint(1, 6, N_USERS)

    return pd.DataFrame({
        "user_id": user_df["user_id"],
        "stress_level": stress,
        "mood_last_week": mood,
        "saving_habit_score": saving
    })


def generate_transactions(user_df):
    """Generate purchase transactions"""
    print("Generating transactions...")

    personas = user_df["persona"].values
    income = user_df["monthly_income"].values

    base = np.random.poisson(lam=1.5, size=N_USERS)
    n_purchase_total = base.sum()
    print(f"  Total transactions: {n_purchase_total:,}")

    user_ids_expanded = np.repeat(user_df["user_id"].values, base)
    personas_expanded = np.repeat(personas, base)
    income_expanded = np.repeat(income, base)

    base_date = pd.Timestamp.today().normalize()
    day_offsets = np.random.randint(0, N_DAYS, size=n_purchase_total)

    ts = base_date - pd.to_timedelta(N_DAYS - day_offsets, unit="D")
    first_view = ts - pd.to_timedelta(np.random.randint(2, 300, n_purchase_total), unit="m")

    discounts = np.random.choice(
        [0, 5, 10, 15, 20, 30, 40, 50], 
        size=n_purchase_total, 
        p=[0.4, 0.1, 0.1, 0.1, 0.1, 0.1, 0.05, 0.05]
    )

    # Improved impulse calculation
    stress_map = {p: STRESS_MEAN[p] for p in PERSONA_NAMES}
    persona_stress = np.array([stress_map[p] for p in personas_expanded])
    
    impulse_prob = (persona_stress / 10) * 0.6 + (discounts / 100) * 0.4
    impulse_prob = np.clip(impulse_prob, 0, 0.85)

    impulse_flag = (np.random.rand(n_purchase_total) < impulse_prob).astype(int)

    # Amount based on income and discount
    amount = np.random.lognormal(mean=7.5, sigma=0.6, size=n_purchase_total)
    amount *= (income_expanded / 50000)
    amount *= (1 - discounts / 100)
    amount = np.clip(amount, 100, 150000)

    categories = np.random.choice(
        ["Electronics", "Fashion", "Beauty", "Gaming", "Home", "Groceries"],
        size=n_purchase_total
    )

    df = pd.DataFrame({
        "transaction_id": np.arange(1, n_purchase_total + 1),
        "user_id": user_ids_expanded,
        "timestamp": ts,
        "amount": amount,
        "discount_applied": discounts,
        "product_category": categories,
        "first_view_time": first_view,
        "is_impulse_purchase": impulse_flag
    })

    return df


def main():
    """Generate all datasets"""
    os.makedirs(RAW_DIR, exist_ok=True)

    print("=" * 60)
    print("GENERATING SYNTHETIC E-COMMERCE DATASET")
    print("=" * 60)

    user_df = generate_user_profile()
    user_df.to_csv(f"{RAW_DIR}/user_profile.csv", index=False)

    browsing_df = generate_browsing_logs(user_df)
    browsing_df.to_csv(f"{RAW_DIR}/browsing_logs.csv", index=False)

    trans_df = generate_transactions(user_df)
    trans_df.to_csv(f"{RAW_DIR}/transactions.csv", index=False)

    psy_df = generate_psychology_survey(user_df)
    psy_df.to_csv(f"{RAW_DIR}/psychology_survey.csv", index=False)

    print("\n" + "=" * 60)
    print("✅ DATASET GENERATION COMPLETE!")
    print("=" * 60)
    print(f"Files saved in: {RAW_DIR}/")
    print(f"  - user_profile.csv: {len(user_df):,} rows")
    print(f"  - browsing_logs.csv: {len(browsing_df):,} rows")
    print(f"  - transactions.csv: {len(trans_df):,} rows")
    print(f"  - psychology_survey.csv: {len(psy_df):,} rows")


if __name__ == "__main__":
    main()