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


def render_dashboard(db_available: bool):
    """Dashboard overview with key metrics."""
    st.subheader("📊 Dashboard Overview")

    if db_available:
        try:
            import database
            stats = database.get_prediction_stats()
            total = stats.get("total_predictions", 0) or 0
            high = stats.get("high_satisfaction_count", 0) or 0
            low = stats.get("low_satisfaction_count", 0) or 0
            avg_p = stats.get("avg_probability")

            if total > 0:
                rate = (high / total * 100) if total else 0
                c1, c2, c3 = st.columns(3)
                c1.metric("Total Predictions", f"{total:,}", help="Total predictions made so far")
                c2.metric("Highly Satisfied", f"{high:,}", f"{rate:.1f}%", help="Customers predicted as highly satisfied")
                c3.metric("Not Highly Satisfied", f"{low:,}", f"{100-rate:.1f}%", help="Customers predicted as not highly satisfied")

                st.markdown("---")
                st.markdown("#### Class Distribution")
                dist_df = pd.DataFrame({"Status": ["Highly Satisfied", "Not Highly Satisfied"], "Count": [high, low]})
                st.bar_chart(dist_df.set_index("Status"), use_container_width=True)
            else:
                st.info("No predictions yet. Go to **Predict** to make your first prediction.")
        except Exception as e:
            st.warning(f"Could not load stats: {e}")
    else:
        st.info("Database not connected. Predictions will not be logged. Go to **Predict** to try the model.")

def render_predict(db_available: bool, model_choice: str):
    """Prediction form and results."""
    st.subheader("🔮 Single Prediction")

    st.markdown("Enter customer and visit details below. The model will predict the likelihood of **high satisfaction**.")

    with st.container(key="predict_form"):
        with st.form("prediction_form"):
            col1, col2, col3 = st.columns(3)

            with col1:
                st.markdown("##### Demographics")
                age = st.number_input("Age", min_value=18, max_value=100, value=35, help="Customer age in years")
                gender = st.selectbox("Gender", CHOICE_MAP["Gender"])
                income = st.number_input("Income", min_value=0.0, value=50000.0, step=1000.0, help="Annual income")

            with col2:
                st.markdown("##### Visit Pattern")
                visit_freq = st.selectbox("Visit Frequency", CHOICE_MAP["VisitFrequency"])
                avg_spend = st.number_input("Average Spend", min_value=0.0, value=25.0, step=5.0, help="Typical spend per visit")
                preferred_cuisine = st.selectbox("Preferred Cuisine", CHOICE_MAP["PreferredCuisine"])
                time_of_visit = st.selectbox("Time of Visit", CHOICE_MAP["TimeOfVisit"])
                group_size = st.number_input("Group Size", min_value=1, max_value=20, value=2)

            with col3:
                st.markdown("##### Experience")
                dining_occasion = st.selectbox("Dining Occasion", CHOICE_MAP["DiningOccasion"])
                meal_type = st.selectbox("Meal Type", CHOICE_MAP["MealType"])
                online_reservation = st.selectbox("Online Reservation", [0, 1], format_func=lambda x: "Yes" if x else "No")
                delivery_order = st.selectbox("Delivery Order", [0, 1], format_func=lambda x: "Yes" if x else "No")
                loyalty_member = st.selectbox("Loyalty Program Member", [0, 1], format_func=lambda x: "Yes" if x else "No")
                wait_time = st.slider("Wait Time (minutes)", 0, 60, 15, help="Time waited for service")
                service_rating = st.slider("Service Rating (1-5)", 1.0, 5.0, 4.0, 0.1)
                food_rating = st.slider("Food Rating (1-5)", 1.0, 5.0, 4.0, 0.1)
                ambiance_rating = st.slider("Ambiance Rating (1-5)", 1.0, 5.0, 4.0, 0.1)

            submitted = st.form_submit_button("PREDICT SATISFACTION STATUS", type="primary", key="predict_btn", width="stretch")

    if submitted:
        row = {
            "Age": float(age), "Gender": gender, "Income": float(income),
            "VisitFrequency": visit_freq, "AverageSpend": float(avg_spend),
            "PreferredCuisine": preferred_cuisine, "TimeOfVisit": time_of_visit,
            "GroupSize": float(group_size), "DiningOccasion": dining_occasion,
            "MealType": meal_type, "OnlineReservation": int(online_reservation),
            "DeliveryOrder": int(delivery_order), "LoyaltyProgramMember": int(loyalty_member),
            "WaitTime": float(wait_time), "ServiceRating": float(service_rating),
            "FoodRating": float(food_rating), "AmbianceRating": float(ambiance_rating),
        }

        try:
            with st.spinner("Running prediction..."):
                prob, pred, threshold = run_prediction(model_choice, row)
            display_name = MODEL_OPTIONS[model_choice]

            # Result box - teal border, bright teal text
            status_text = "✓ HIGHLY SATISFIED" if pred == 1 else "NOT HIGHLY SATISFIED"
            model_short = display_name.split(" (")[0]
            st.markdown(f"""
            <div class="prediction-result-box">
                <p class="result-main">{status_text}</p>
                <p class="result-sub">Predicted using {model_short}</p>
            </div>
            """, unsafe_allow_html=True)

            # Probability displays - side by side
            not_prob = 1 - prob
            r1, r2 = st.columns(2)
            with r1:
                st.markdown(f"""
                <div class="prob-display">
                    <div class="prob-label">High Satisfaction Probability</div>
                    <div class="prob-value">{prob:.2%}</div>
                </div>
                """, unsafe_allow_html=True)
            with r2:
                st.markdown(f"""
                <div class="prob-display">
                    <div class="prob-label">Not High Satisfaction Probability</div>
                    <div class="prob-value">{not_prob:.2%}</div>
                </div>
                """, unsafe_allow_html=True)

            # Confidence gauge - horizontal bar
            pct = min(100, int(prob * 100))
            st.markdown(f"""
            <div class="confidence-gauge">
                <div class="gauge-label">Satisfaction Confidence</div>
                <div style="width:100%; height:14px; background:#374151; border-radius:7px; overflow:hidden; margin:0.5rem 0;">
                    <div style="width:{pct}%; height:100%; background:linear-gradient(90deg, #17a2b8, #28a745); border-radius:7px; transition:width 0.3s;"></div>
                </div>
                <div class="gauge-value">{prob:.1%}</div>
            </div>
            """, unsafe_allow_html=True)

            if db_available:
                import database
                pred_id = database.log_prediction(
                    model_name=model_choice, input_data=row,
                    predicted_probability=prob, predicted_class=pred, threshold_used=threshold,
                )
                st.sidebar.success(f"Logged (ID: {pred_id})")

        except FileNotFoundError as e:
            st.error(f"Model files not found. {e}")
        except Exception as e:
            st.error(f"Prediction failed: {e}")


def render_history(db_available: bool):
    """Prediction history table."""
    st.subheader("📋 Prediction History")

    if db_available:
        try:
            import database
            recent = database.get_recent_predictions(limit=50)
            if recent:
                df = pd.DataFrame(recent).drop(columns=["input_data"], errors="ignore")
                df["created_at"] = pd.to_datetime(df["created_at"]).dt.strftime("%Y-%m-%d %H:%M")
                df["predicted_class"] = df["predicted_class"].map({1: "Highly Satisfied", 0: "Not Satisfied"})
                st.dataframe(df, use_container_width=True, hide_index=True)
            else:
                st.info("No predictions logged yet. Go to **Predict** to make predictions.")
        except Exception as e:
            st.error(f"Could not load history: {e}")
    else:
        st.warning("Database not connected. Predictions are not being saved.")


def render_about():
    """About and how it works."""
    st.subheader("ℹ️ About This App")

    st.markdown("""
    ### Restaurant Customer Satisfaction Predictor

    This app uses **machine learning** to predict whether a restaurant customer is likely to be **highly satisfied** 
    based on their profile, visit patterns, and experience ratings.

    #### How It Works

    1. **Input**: You provide 17 features about the customer and their visit.
    2. **Model**: A trained classifier (Logistic Regression or Random Forest) processes the input.
    3. **Output**: The model returns a **probability** (0–100%) and a **binary prediction** (Highly Satisfied / Not).

    #### Features Used

    | Category | Features |
    |----------|----------|
    | **Demographics** | Age, Gender, Income |
    | **Visit Pattern** | Visit Frequency, Average Spend, Preferred Cuisine, Time of Visit, Group Size |
    | **Experience** | Dining Occasion, Meal Type, Online Reservation, Delivery Order, Loyalty Member |
    | **Ratings** | Wait Time, Service Rating, Food Rating, Ambiance Rating |
    """)

    # Model performance table if available
    comp_path = os.path.join(PROJECT_ROOT, "model_artifacts", "model_comparison_results.csv")
    if os.path.exists(comp_path):
        try:
            comp_df = pd.read_csv(comp_path)
            # Filter to our two models
            our_models = ["Human Model - Logistic Regression", "AI Model - Random Forest"]
            comp_df = comp_df[comp_df["Model"].isin(our_models)][["Model", "Accuracy", "F1", "ROC_AUC"]]
            comp_df["Model"] = comp_df["Model"].replace({
                "Human Model - Logistic Regression": "Logistic Regression",
                "AI Model - Random Forest": "Random Forest",
            })
            for c in ["Accuracy", "F1", "ROC_AUC"]:
                comp_df[c] = comp_df[c].apply(lambda x: f"{x:.1%}")
            st.markdown("#### Model Performance (Test Set)")
            st.dataframe(comp_df, use_container_width=True, hide_index=True)
        except Exception:
            pass


def inject_custom_css():
    """Inject CSS for anchor-style nav and prominent predict button."""
    st.markdown("""
    <style>
    /* Sidebar - dark navy background */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1e3a5f 0%, #0f2744 100%) !important;
    }
    [data-testid="stSidebar"] .stMarkdown {
        color: #94a3b8 !important;
    }
    /* Nav items - clean link style */
    [data-testid="stSidebar"] button[kind="secondary"] {
        width: 100% !important;
        justify-content: flex-start !important;
        background: transparent !important;
        border: none !important;
        box-shadow: none !important;
        font-weight: 400 !important;
        padding: 0.6rem 1rem !important;
        border-radius: 0.5rem !important;
        color: #7dd3fc !important;
    }
    [data-testid="stSidebar"] button[kind="secondary"]:hover {
        background: rgba(125, 211, 252, 0.1) !important;
        color: #bae6fd !important;
    }
    /* Active nav item - light blue outline, brighter text */
    [data-testid="stSidebar"] button[kind="primary"] {
        width: 100% !important;
        justify-content: flex-start !important;
        background: rgba(56, 189, 248, 0.15) !important;
        border: 1px solid #38bdf8 !important;
        box-shadow: none !important;
        font-weight: 500 !important;
        padding: 0.6rem 1rem !important;
        border-radius: 0.5rem !important;
        color: #7dd3fc !important;
        margin-top:0px;}
    [data-testid="stSidebar"] button[kind="primary"]:hover {
        background: rgba(56, 189, 248, 0.15) !important;
        border: 1px solid #38bdf8 !important;
        color: #bae6fd !important;
    }
    /* Predict button only - sleek red bar, 3D embossed (excludes +/- and help buttons) */
    [class*="st-key-predict-btn"] {
        width: 100% !important;
        font-size: 1rem !important;
        font-weight: 600 !important;
        letter-spacing: 0.04em !important;
        padding: 0.85rem 2rem !important;
        background: linear-gradient(180deg, #e74c3c 0%, #c0392b 50%, #a93226 100%) !important;
        border: none !important;
        border-radius: 0.6rem !important;
        color: rgba(255, 255, 255, 0.95) !important;
        text-transform: uppercase !important;
        box-shadow: 0 4px 0 #922b21, 0 6px 12px rgba(0,0,0,0.3) !important;
    }
    [class*="st-key-predict-btn"]:hover {
        background: linear-gradient(180deg, #ec7063 0%, #e74c3c 50%, #c0392b 100%) !important;
        color: white !important;
        box-shadow: 0 2px 0 #922b21, 0 4px 8px rgba(0,0,0,0.3) !important;
    }
    /* Result box - teal border, dark bg */
    .prediction-result-box {
        padding: 2rem;
        border-radius: 0.5rem;
        margin: 1.5rem 0;
        text-align: center;
        background: rgba(30, 30, 30, 0.5);
        border: 2px solid #17a2b8;
    }
    .prediction-result-box .result-main {
        font-size: 1.75rem;
        font-weight: 700;
        color: #17a2b8;
        margin: 0;
        letter-spacing: 0.02em;
    }
    .prediction-result-box .result-sub {
        font-size: 0.9rem;
        color: #9ca3af;
        margin: 0.5rem 0 0 0;
    }
    /* Probability displays */
    .prob-display {
        background: rgba(30, 30, 30, 0.4);
        padding: 1rem 1.5rem;
        border-radius: 0.4rem;
        margin: 0.5rem 0;
    }
    .prob-display .prob-label { font-size: 0.85rem; color: #9ca3af; margin-bottom: 0.25rem; }
    .prob-display .prob-value { font-size: 1.5rem; font-weight: 700; color: white; }
    .prob-display .prob-delta { font-size: 0.8rem; color: #2ecc71; }
    /* Confidence gauge */
    .confidence-gauge {
        margin: 1.5rem 0;
        text-align: center;
    }
    .confidence-gauge .gauge-label { font-size: 0.85rem; color: #9ca3af; margin-bottom: 0.5rem; }
    .confidence-gauge .gauge-value { font-size: 1.25rem; font-weight: 700; color: white; margin-top: 0.5rem; }
    </style>
    """, unsafe_allow_html=True)


def main():
    st.set_page_config(
        page_title="Restaurant Satisfaction Predictor",
        page_icon="🍽️",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    inject_custom_css()

    # Initialize page in session state
    if "page" not in st.session_state:
        st.session_state.page = "Dashboard"

    # Sidebar - navigation at top
    nav_items = [
        ("Dashboard", "Dashboard"),
        ("Predict", "Single Prediction"),
        ("History", "Prediction History"),
        ("About", "About"),
    ]
    for page_key, display_label in nav_items:
        is_active = st.session_state.page == page_key
        if st.sidebar.button(
            display_label,
            key=f"nav_{page_key}",
            use_container_width=True,
            type="primary" if is_active else "secondary",
        ):
            st.session_state.page = page_key
            st.rerun()

    st.sidebar.markdown("---")
    st.sidebar.header("Model Selection")
    model_choice = st.sidebar.selectbox(
        "Choose model",
        options=list(MODEL_OPTIONS.keys()),
        format_func=lambda k: MODEL_OPTIONS[k],
        label_visibility="collapsed",
    )

    # Database init
    db_available = False
    try:
        import database
        database.init_schema()
        db_available = True
    except Exception as e:
        st.sidebar.warning(f"DB: {str(e)[:50]}...")

    # Main content
    st.title("Restaurant Customer Satisfaction Intelligence")
    st.markdown("Predict high-satisfaction customers from demographics, visit patterns, and experience ratings.")
    st.markdown("---")

    if st.session_state.page == "Dashboard":
        render_dashboard(db_available)
    elif st.session_state.page == "Predict":
        render_predict(db_available, model_choice)
    elif st.session_state.page == "History":
        render_history(db_available)
    else:
        render_about()

    st.sidebar.markdown("---")
    st.sidebar.caption("© Restaurant Satisfaction Predictor")


if __name__ == "__main__":
    main()
