"""
Restaurant Customer Satisfaction Prediction - Streamlit UI
Uses trained models and SQLite backend for logging predictions.
"""

import os
import json
import pandas as pd
import joblib
import streamlit as st

# Compatibility shim: models saved with sklearn 1.5+ reference _RemainderColsList,
# which doesn't exist in sklearn 1.4.x. Inject it so joblib.load() can unpickle.
def _patch_sklearn_for_pickle():
    from collections import UserList
    import sklearn.compose._column_transformer as ct
    if not hasattr(ct, "_RemainderColsList"):
        class _RemainderColsList(UserList):
            pass
        ct._RemainderColsList = _RemainderColsList

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
ARTIFACT_DIR = os.path.join(PROJECT_ROOT, "model_artifacts", "individual_models")

CHOICE_MAP = {
    "Gender": ["Male", "Female"],
    "VisitFrequency": ["Daily", "Weekly", "Monthly", "Rarely"],
    "PreferredCuisine": ["American", "Chinese", "Indian", "Italian", "Mexican"],
    "TimeOfVisit": ["Breakfast", "Lunch", "Dinner"],
    "DiningOccasion": ["Casual", "Business", "Celebration"],
    "MealType": ["Dine-in", "Takeaway"],
}

MODEL_OPTIONS = {
    "logistic_regression": "Logistic Regression (Human Model)",
    "random_forest": "Random Forest (AI Model)",
}


@st.cache_resource
def load_model_bundle(model_key: str):
    """Load model, meta, and threshold from artifacts."""
    model_path = os.path.join(ARTIFACT_DIR, f"{model_key}.pkl")
    meta_path = os.path.join(ARTIFACT_DIR, f"{model_key}_meta.json")
    threshold_path = os.path.join(ARTIFACT_DIR, f"{model_key}_threshold.json")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Meta not found: {meta_path}")
    if not os.path.exists(threshold_path):
        raise FileNotFoundError(f"Threshold not found: {threshold_path}")

    _patch_sklearn_for_pickle()
    model = joblib.load(model_path)
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    with open(threshold_path, "r", encoding="utf-8") as f:
        threshold = float(json.load(f).get("threshold", 0.5))
    return model, meta, threshold


def order_input_df(model, meta, row_dict: dict) -> pd.DataFrame:
    """Build input DataFrame with correct column order."""
    df = pd.DataFrame([row_dict])
    expected_cols = meta.get("num_features", []) + meta.get("cat_features", [])
    for col in expected_cols:
        if col not in df.columns:
            df[col] = None
    return df[expected_cols]


def run_prediction(model_key: str, row: dict) -> tuple:
    """Run prediction and return (probability, predicted_class, threshold)."""
    model, meta, threshold = load_model_bundle(model_key)
    input_df = order_input_df(model, meta, row)
    prob = float(model.predict_proba(input_df)[0, 1])
    pred = int(prob >= threshold)
    return prob, pred, threshold


def main():
    st.set_page_config(
        page_title="Restaurant Satisfaction Predictor",
        page_icon="🍽️",
        layout="wide",
    )
    st.title("🍽️ Restaurant Customer Satisfaction Predictor")
    st.markdown(
        "Predict whether a customer is likely to be **highly satisfied** based on their profile and visit details."
    )

    # Sidebar: model selection
    st.sidebar.header("Model Selection")
    model_choice = st.sidebar.selectbox(
        "Choose model",
        options=list(MODEL_OPTIONS.keys()),
        format_func=lambda k: MODEL_OPTIONS[k],
    )

    # Initialize database (optional - graceful if DB not configured)
    db_available = False
    try:
        import database
        database.init_schema()
        db_available = True
    except Exception as e:
        st.sidebar.warning(f"Database not connected: {e}. Predictions won't be logged.")

    # Main form
    with st.form("prediction_form"):
        st.subheader("Customer & Visit Details")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("#### Demographics")
            age = st.number_input("Age", min_value=18, max_value=100, value=35)
            gender = st.selectbox("Gender", CHOICE_MAP["Gender"])
            income = st.number_input("Income", min_value=0.0, value=50000.0, step=1000.0)

        with col2:
            st.markdown("#### Visit Pattern")
            visit_freq = st.selectbox("Visit Frequency", CHOICE_MAP["VisitFrequency"])
            avg_spend = st.number_input("Average Spend", min_value=0.0, value=25.0, step=5.0)
            preferred_cuisine = st.selectbox("Preferred Cuisine", CHOICE_MAP["PreferredCuisine"])
            time_of_visit = st.selectbox("Time of Visit", CHOICE_MAP["TimeOfVisit"])
            group_size = st.number_input("Group Size", min_value=1, max_value=20, value=2)

        with col3:
            st.markdown("#### Experience")
            dining_occasion = st.selectbox("Dining Occasion", CHOICE_MAP["DiningOccasion"])
            meal_type = st.selectbox("Meal Type", CHOICE_MAP["MealType"])
            online_reservation = st.selectbox("Online Reservation", [0, 1], format_func=lambda x: "Yes" if x else "No")
            delivery_order = st.selectbox("Delivery Order", [0, 1], format_func=lambda x: "Yes" if x else "No")
            loyalty_member = st.selectbox("Loyalty Program Member", [0, 1], format_func=lambda x: "Yes" if x else "No")
            wait_time = st.slider("Wait Time (minutes)", 0, 60, 15)
            service_rating = st.slider("Service Rating (1-5)", 1.0, 5.0, 4.0, 0.1)
            food_rating = st.slider("Food Rating (1-5)", 1.0, 5.0, 4.0, 0.1)
            ambiance_rating = st.slider("Ambiance Rating (1-5)", 1.0, 5.0, 4.0, 0.1)

        submitted = st.form_submit_button("Predict Satisfaction")

    if submitted:
        row = {
            "Age": float(age),
            "Gender": gender,
            "Income": float(income),
            "VisitFrequency": visit_freq,
            "AverageSpend": float(avg_spend),
            "PreferredCuisine": preferred_cuisine,
            "TimeOfVisit": time_of_visit,
            "GroupSize": float(group_size),
            "DiningOccasion": dining_occasion,
            "MealType": meal_type,
            "OnlineReservation": int(online_reservation),
            "DeliveryOrder": int(delivery_order),
            "LoyaltyProgramMember": int(loyalty_member),
            "WaitTime": float(wait_time),
            "ServiceRating": float(service_rating),
            "FoodRating": float(food_rating),
            "AmbianceRating": float(ambiance_rating),
        }

        try:
            prob, pred, threshold = run_prediction(model_choice, row)
            display_name = MODEL_OPTIONS[model_choice]

            st.success("Prediction complete!")
            st.metric("Predicted Probability (High Satisfaction)", f"{prob:.2%}")
            st.metric("Predicted Class", "Highly Satisfied ✓" if pred == 1 else "Not Highly Satisfied")
            st.caption(f"Threshold used: {threshold:.4f} | Model: {display_name}")

            if db_available:
                pred_id = database.log_prediction(
                    model_name=model_choice,
                    input_data=row,
                    predicted_probability=prob,
                    predicted_class=pred,
                    threshold_used=threshold,
                )
                st.sidebar.success(f"Logged to database (ID: {pred_id})")

        except FileNotFoundError as e:
            st.error(f"Model files not found. Run training in Colab and place artifacts in `model_artifacts/individual_models/`. {e}")
        except Exception as e:
            st.error(f"Prediction failed: {e}")
            raise

    # History section (if DB available)
    if db_available:
        with st.expander("📊 Prediction History & Stats", expanded=False):
            try:
                stats = database.get_prediction_stats()
                recent = database.get_recent_predictions(limit=20)
                if stats and stats.get("total_predictions", 0) > 0:
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Total Predictions", stats.get("total_predictions", 0))
                    c2.metric("High Satisfaction", stats.get("high_satisfaction_count", 0))
                    c3.metric("Low Satisfaction", stats.get("low_satisfaction_count", 0))
                    avg_p = stats.get("avg_probability")
                    c4.metric("Avg Probability", f"{avg_p:.2%}" if avg_p is not None else "N/A")
                if recent:
                    st.dataframe(
                        pd.DataFrame(recent).drop(columns=["input_data"], errors="ignore"),
                        use_container_width=True,
                    )
                else:
                    st.info("No predictions logged yet.")
            except Exception as e:
                st.error(f"Could not load history: {e}")


if __name__ == "__main__":
    main()
