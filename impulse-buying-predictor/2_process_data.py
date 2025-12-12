"""
Data Processing & Feature Engineering
Creates final dataset with improved target calculation
"""

import os
import numpy as np
import pandas as pd

# Paths
RAW_DIR = "data/raw"
OUTPUT_DIR = "data"
FINAL_OUTPUT_PATH = f"{OUTPUT_DIR}/data.csv"


def load_raw():
    """Load all raw datasets"""
    print("Loading raw data...")
    user_df = pd.read_csv(f"{RAW_DIR}/user_profile.csv")
    browsing_df = pd.read_csv(f"{RAW_DIR}/browsing_logs.csv")
    trans_df = pd.read_csv(f"{RAW_DIR}/transactions.csv")
    survey_df = pd.read_csv(f"{RAW_DIR}/psychology_survey.csv")
    
    print(f"  Users: {len(user_df):,}")
    print(f"  Browsing sessions: {len(browsing_df):,}")
    print(f"  Transactions: {len(trans_df):,}")
    
    return user_df, browsing_df, trans_df, survey_df


def aggregate_browsing_logs(df: pd.DataFrame):
    """Aggregate browsing behavior per user"""
    print("Aggregating browsing logs...")
    
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["hour"] = df["timestamp"].dt.hour
    df["is_late_night"] = df["hour"].isin([22, 23, 0, 1, 2, 3]).astype(int)

    agg = df.groupby("user_id").agg(
        total_sessions=("session_id", "count"),
        num_product_page_visits=("page_type", lambda x: (x == "product").sum()),
        num_cart_visits=("page_type", lambda x: (x == "cart").sum()),
        num_checkout_visits=("page_type", lambda x: (x == "checkout").sum()),
        avg_time_on_product=("time_spent_seconds", "mean"),
        late_night_session_ratio=("is_late_night", "mean"),
        device_preference=("device_type", lambda x: x.mode()[0] if len(x.mode()) else "mobile"),
    )

    agg["avg_time_on_product"] = agg["avg_time_on_product"].fillna(0)

    return agg.reset_index()


def aggregate_transactions(df: pd.DataFrame):
    """Aggregate transaction data per user"""
    print("Aggregating transactions...")
    
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["first_view_time"] = pd.to_datetime(df["first_view_time"])

    df["minutes_to_purchase"] = (
        (df["timestamp"] - df["first_view_time"]).dt.total_seconds() / 60
    ).clip(lower=0)

    agg = df.groupby("user_id").agg(
        total_purchases=("transaction_id", "count"),
        total_spent=("amount", "sum"),
        avg_purchase_value=("amount", "mean"),
        avg_discount_used=("discount_applied", "mean"),
        impulse_purchase_ratio=("is_impulse_purchase", "mean"),
        past_impulse_purchases=("is_impulse_purchase", "sum"),
        avg_minutes_to_purchase=("minutes_to_purchase", "mean"),
    )

    return agg.reset_index()


def merge_all(user_df, browsing_agg, trans_agg, survey_df):
    """Merge all datasets"""
    print("Merging all datasets...")
    
    merged = (
        user_df
        .merge(browsing_agg, on="user_id", how="left")
        .merge(trans_agg, on="user_id", how="left")
        .merge(survey_df, on="user_id", how="left")
    )

    # Fill missing values for users with no transactions
    trans_cols = [
        "total_purchases", "total_spent", "avg_purchase_value",
        "avg_discount_used", "impulse_purchase_ratio",
        "past_impulse_purchases", "avg_minutes_to_purchase"
    ]
    merged[trans_cols] = merged[trans_cols].fillna(0)

    # Fill missing browsing data
    merged["avg_time_on_product"] = merged["avg_time_on_product"].fillna(0)
    merged["device_preference"] = merged["device_preference"].fillna("mobile")

    return merged


def compute_impulse_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute realistic impulse buying score (0-100)
    
    Score is based on:
    - Past impulse behavior (40%)
    - Discount sensitivity (20%)
    - Psychological factors (20%)
    - Browsing patterns (15%)
    - Time pressure (5%)
    """
    print("Computing impulse buy score...")
    
    # Normalize core signals
    impulse_ratio = df["impulse_purchase_ratio"].clip(0, 1)
    discount_norm = (df["avg_discount_used"] / 50).clip(0, 1)
    late_night = df["late_night_session_ratio"].clip(0, 1)
    stress = (df["stress_level"] / 10).clip(0, 1)
    
    # Browsing intensity
    revisit_ratio = (
        df["num_product_page_visits"] / df["total_sessions"].replace(0, 1)
    ).clip(0, 1)
    
    # Time pressure (faster purchase = more impulsive)
    time_pressure = np.exp(-df["avg_minutes_to_purchase"].clip(1, 500) / 200)
    
    # Persona effect
    persona_map = {
        "impulse_buyer": 1.3,
        "deal_hunter": 1.15,
        "steady_buyer": 0.85,
        "window_shopper": 0.70,
        "premium_user": 0.75,
        "cautious_saver": 1.0
    }
    persona_factor = df["persona"].map(persona_map).fillna(1.0)
    
    # Mood effect
    mood_map = {
        "Happy": -0.05,
        "Neutral": 0.0,
        "Sad": 0.10,
        "Anxious": 0.15
    }
    mood_adjustment = df["mood_last_week"].map(mood_map).fillna(0)
    
    # Saving habit (inverse relationship)
    saving_penalty = (6 - df["saving_habit_score"]) / 10
    
    # Compute weighted score
    score = (
        35 * (impulse_ratio ** 1.5) +           # Past behavior is strongest
        18 * np.sqrt(discount_norm) +            # Discount sensitivity
        12 * (stress ** 1.8) +                   # Stress amplifies impulse
        10 * (late_night ** 1.3) +               # Late night browsing
        8 * revisit_ratio +                      # Product page revisits
        5 * time_pressure +                      # Quick purchases
        6 * persona_factor +                     # Persona multiplier
        4 * mood_adjustment +                    # Mood adjustment
        2 * saving_penalty +                     # Saving habits
        np.random.normal(0, 3, len(df))          # Natural variance
    )
    
    df["impulse_buy_score"] = np.clip(score, 0, 100).round(2)
    
    print(f"  Score range: {df['impulse_buy_score'].min():.2f} - {df['impulse_buy_score'].max():.2f}")
    print(f"  Mean score: {df['impulse_buy_score'].mean():.2f}")
    print(f"  Median score: {df['impulse_buy_score'].median():.2f}")
    
    return df


def main():
    """Main processing pipeline"""
    print("=" * 60)
    print("PROCESSING DATA FOR MODEL TRAINING")
    print("=" * 60 + "\n")

    # Load data
    user_df, browsing_df, trans_df, survey_df = load_raw()

    # Aggregate
    browsing_agg = aggregate_browsing_logs(browsing_df)
    trans_agg = aggregate_transactions(trans_df)

    # Merge
    merged = merge_all(user_df, browsing_agg, trans_agg, survey_df)

    # Compute target
    final_df = compute_impulse_score(merged)

    # Drop persona column (was only for generation)
    final_df = final_df.drop(columns=["persona"])

    # Save
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    final_df.to_csv(FINAL_OUTPUT_PATH, index=False)

    print("\n" + "=" * 60)
    print("✅ DATA PROCESSING COMPLETE!")
    print("=" * 60)
    print(f"Final dataset: {FINAL_OUTPUT_PATH}")
    print(f"Shape: {final_df.shape}")
    print(f"Features: {final_df.shape[1] - 1}")
    print(f"\nFirst few rows:")
    print(final_df.head())


if __name__ == "__main__":
    main()