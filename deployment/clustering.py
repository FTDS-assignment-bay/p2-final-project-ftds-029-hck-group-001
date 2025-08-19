import streamlit as st
import pickle
import pandas as pd
from equinox_inference import load_bandit, EQuinoxPredictor

bandit = load_bandit('equinox_model_v1.1')

def show():
    # Session state to control whether we're showing results or form
    if "show_results" not in st.session_state:
        st.session_state.show_results = False
        st.session_state.last_mode = None
        st.session_state.last_user_features = {}
        st.session_state.last_session_context = {}
        st.session_state.last_result = None

    # "Pick Existing User"
    mode = st.selectbox("", ["How It Works", "Create New User"],
                        label_visibility="collapsed")

    if mode == 'How It Works':
        st.markdown("""
        ### How It Works

        This tool uses **EQuinox AI's contextual bandit model** to decide the most effective **marketing action** for a specific customer in real time.

        **Step-by-step:**
        1. **Input Customer Data**  
        - Create a **new customer profile** or pick an **existing customer**.

        2. **Provide Session Context**  
        - Add details about the current browsing session.

        3. **Model Prediction**  
        - The model chooses the most effective action from:
            - 📧 **Email (No Discount)**  
            - 💸 **Email (10% Discount)**  
            - 🏷 **Banner (Limited Time Offer)**  
            - 🔔 **Popup (Abandoned Cart Reminder)**  

        **Why This Works:**  
        The **contextual bandit algorithm** learns which action works best for each situation.
        """)

    elif st.session_state.show_results and mode == st.session_state.last_mode:
        # Show results instead of form
        result = st.session_state.last_result
        recommendation = result["recommended_action"]
        lift = f"{result['expected_conversion_rate'] - result['baseline_rate']:.1%}"
        breakdown = result["action_breakdown"]

        st.markdown(f"""
        ### 🎯 Recommended Action: **{recommendation.replace('_', ' ').title()}**
        **Predicted Conversion Lift:** {lift}
        """)

        breakdown_df = pd.DataFrame(
            list(breakdown.items()),
            columns=["Action", "Predicted Score"]
        )
        breakdown_df["Predicted Score"] = breakdown_df["Predicted Score"].apply(lambda x: f"{x:.2%}")

        st.subheader("📊 Action Score Breakdown")
        st.table(breakdown_df)

        if st.button("Go Back"):
            st.session_state.show_results = False

    elif mode == 'Create New User':
        with st.form('Riwayat Calon Nasabah'):
            st.write('## Customer Behaviour')
            col1, col2 = st.columns(2)
            col3, col4 = st.columns(2)

            with col1:
                total_views = st.number_input('Total Views', min_value=0, value=80)
                events_per_day = st.number_input('Events Per Day', min_value=0.0, format="%.2f", value=8.5)
                conversion_rate = st.number_input('Conversion Rate', min_value=0.0, max_value=1.0, format="%.2f", value=0.4)
                avg_price_viewed = st.number_input('Average Price Viewed', min_value=0, value=750000)
                max_price_viewed = st.number_input('Maximum Price Viewed', min_value=0, value=2500000)

            with col2:
                unique_categories = st.number_input('Unique Categories', min_value=0, value=10)
                user_tenure_days = st.number_input('User Tenure (days)', min_value=0, value=365)
                has_converted = st.selectbox('Has Converted', [0, 1])
                avg_availability = st.number_input('Average Availability', min_value=0.0, max_value=1.0, format="%.2f",value=0.8)

            with col3:
                hour = st.number_input('Hour', min_value=0, max_value=23, step=1,value=23)
                is_weekend = st.selectbox('Is Weekend', [False, True])

            with col4:
                item_price = st.number_input('Item Price', min_value=0, value= 1200000)
                item_category = st.number_input('Item Category', min_value=0, step=1,value= 990)

            submit = st.form_submit_button()

        if submit:
            user_features = {
                'total_views': total_views,
                'events_per_day': events_per_day,
                'conversion_rate': conversion_rate,
                'avg_price_viewed': avg_price_viewed,
                'max_price_viewed': max_price_viewed,
                'unique_categories': unique_categories,
                'user_tenure_days': user_tenure_days,
                'has_converted': has_converted,
                'avg_availability': avg_availability
            }
            session_context = {
                'hour': hour,
                'is_weekend': is_weekend,
                'item_price': item_price,
                'item_category': item_category
            }

            predictor = EQuinoxPredictor(bandit)
            result = predictor.predict(user_features, session_context)

            st.session_state.show_results = True
            st.session_state.last_mode = mode
            st.session_state.last_user_features = user_features
            st.session_state.last_session_context = session_context
            st.session_state.last_result = result
            st.rerun()

    # elif mode == "Pick Existing User":
    #     with st.form('Riwayat Calon Nasabah'):
    #         user_id = st.number_input("Select User ID", min_value=0, max_value=10000)
    #         submit = st.form_submit_button()

    #     if submit:
    #         user_features = {"user_id": user_id}  # Adjust as needed for your inference
    #         session_context = {}

    #         predictor = EQuinoxPredictor(bandit)
    #         result = predictor.predict(user_features, session_context)

    #         st.session_state.show_results = True
    #         st.session_state.last_mode = mode
    #         st.session_state.last_user_features = user_features
    #         st.session_state.last_session_context = session_context
    #         st.session_state.last_result = result
    #         st.experimental_rerun()
