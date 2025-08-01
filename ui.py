import streamlit as st
import pandas as pd
import numpy as np
import joblib
import base64
import datetime
import requests
import matplotlib.pyplot as plt
import seaborn as sns

# Background

def get_base64_from_url(image_url):
    try:
        response = requests.get(image_url)
        if response.status_code == 200:
            return base64.b64encode(response.content).decode()
    except:
        return ""
    return ""

bg_url = "https://images.unsplash.com/photo-1507525428034-b723cf961d3e?auto=format&fit=crop&w=1353&q=80"
bg_image_base64 = get_base64_from_url(bg_url)

if bg_image_base64:
    st.markdown(f"""
        <style>
        .stApp {{
            background-image: url("data:image/jpg;base64,{bg_image_base64}");
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
        }}
        .block-container {{
            background-color: rgba(255,255,255,0);
        }}
        h1, h2, h3, h4, p {{
            color: white;
            text-shadow: 1px 1px 3px black;
        }}
        </style>
    """, unsafe_allow_html=True)

# Config
st.set_page_config(page_title="Home Energy App", layout="centered")

# Navigation Bar
page = st.selectbox(
    "🔍 Navigate",
    ["🏠 Home", "📊 Visual Insights", "📘 Energy Tips", "📂 About Project"]
)

# Model Columns
model_columns = [
    'num_occupants', 'house_size_sqft', 'monthly_income', 'outside_temp_celsius',
    'year', 'month', 'day', 'season',
    'heating_type_Electric', 'heating_type_Gas', 'heating_type_None',
    'cooling_type_AC', 'cooling_type_Fan', 'cooling_type_None',
    'manual_override_Y', 'manual_override_N',
    'is_weekend', 'temp_above_avg', 'income_per_person', 'square_feet_per_person',
    'high_income_flag', 'low_temp_flag',
    'season_spring', 'season_summer', 'season_fall', 'season_winter',
    'day_of_week_0', 'day_of_week_6', 'energy_star_home'
]

# # Model selection
# model_choice = st.selectbox("Choose Model", ["Random Forest", "Decision Tree","Linear"])
# if model_choice == "Random Forest":
#     model = joblib.load("Random-Forest-model.pkl")
# elif model_choice == "Decision Tree":
#     model = joblib.load("DecisionTree-model.pkl")
# elif model_choice == "Linear":
#     model = joblib.load("Linear-model.pkl")


# Page: Home
if page == "🏠 Home":
    st.markdown("<h1 style='text-align:center'>🏡 Energy Consumption Predictor</h1>", unsafe_allow_html=True)
    # st.markdown("<p style='text-align:center;font-size:18px;'>Estimate home energy usage based on your lifestyle</p>", unsafe_allow_html=True)
    st.markdown("<hr>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        num_occupants = st.number_input("👨‍👩‍👧 Number of Occupants", min_value=1, value=4)
        house_size_sqft = st.number_input("🏠 House Size (sqft)", min_value=100.0, value=1500.0)
        monthly_income = st.number_input("💵 Monthly Income ($)", min_value=0, value=20000)
        year = st.number_input("📆 Year", min_value=1970, max_value=2099, value=2025)
    with col2:
        outside_temp_celsius = st.slider("🌡️ Outside Temp (°C)", -20.0, 50.0, 25.0)
        month = st.selectbox("📅 Month", list(range(1, 13)))
        day = st.number_input("📅 Day", min_value=1, max_value=31, value=1)

    # Date validation
    try:
        obj = datetime.date(int(year), int(month), int(day))
        day_of_week = obj.weekday()
    except:
        st.error("❗ Invalid date.")
        st.stop()

    season_label = (
        "winter" if month in [12, 1, 2] else
        "summer" if month in [3, 4, 5] else
        "fall" if month in [6, 7, 8] else
        "spring"
    )

    # Preferences
    # st.subheader("🌬️ Heating/Cooling Preferences")
    col3, col4, col5 = st.columns(3)
    with col3:
        heating_type = st.selectbox("🔥 Heating Type", ["Electric", "Gas", "None"])
    with col4:
        cooling_type = st.selectbox("❄️ Cooling Type", ["AC", "Fan", "None"])
    with col5:
        manual_override = st.radio("⚙️ Manual Override", ["Yes", "No"])
    energy_star_home = st.checkbox("🏅 Energy Star Certified Home", value=False)

    # Derived features
    is_weekend = int(day_of_week >= 5)
    temp_above_avg = int(outside_temp_celsius > 28)
    income_per_person = monthly_income / num_occupants
    square_feet_per_person = house_size_sqft / num_occupants
    high_income_flag = int(monthly_income > 40000)
    low_temp_flag = int(outside_temp_celsius < 15)

    input_data = {
        'num_occupants': num_occupants,
        'house_size_sqft': house_size_sqft,
        'monthly_income': monthly_income,
        'outside_temp_celsius': outside_temp_celsius,
        'year': year,
        'month': month,
        'day': day,
        'season': {'winter': 1, 'summer': 2, 'fall': 3, 'spring': 4}[season_label],
        'heating_type_Electric': int(heating_type == "Electric"),
        'heating_type_Gas': int(heating_type == "Gas"),
        'heating_type_None': int(heating_type == "None"),
        'cooling_type_AC': int(cooling_type == "AC"),
        'cooling_type_Fan': int(cooling_type == "Fan"),
        'cooling_type_None': int(cooling_type == "None"),
        'manual_override_Y': int(manual_override == "Yes"),
        'manual_override_N': int(manual_override == "No"),
        'is_weekend': is_weekend,
        'temp_above_avg': temp_above_avg,
        'income_per_person': income_per_person,
        'square_feet_per_person': square_feet_per_person,
        'high_income_flag': high_income_flag,
        'low_temp_flag': low_temp_flag,
        'season_spring': int(season_label == "spring"),
        'season_summer': int(season_label == "summer"),
        'season_fall': int(season_label == "fall"),
        'season_winter': int(season_label == "winter"),
        'day_of_week_0': int(day_of_week == 0),
        'day_of_week_6': int(day_of_week == 6),
        'energy_star_home': int(energy_star_home)
    }
    input_df = pd.DataFrame([input_data])[model_columns]

# Model selection
model_choice = st.selectbox("Choose Model", ["Random Forest", "Decision Tree","Linear"])
if model_choice == "Random Forest":
    model = joblib.load("Random-Forest-model.pkl")
elif model_choice == "Decision Tree":
    model = joblib.load("DecisionTree-model.pkl")
elif model_choice == "Linear":
    model = joblib.load("Linear-model.pkl")
    
if st.button("🔍 Predict Energy Consumption"):
    try:
        prediction = model.predict(input_df)[0]
        st.success(f"💡 Estimated Energy Usage: {prediction:.2f} kWh")
        st.session_state.prediction = prediction
        st.session_state.input_df = input_df
    except Exception as e:
        st.error(f"Prediction Error: {e}")

# Page: Visual Insights
elif page == "📊 Visual Insights":
    st.header("📊 Visual Analysis")

    if "prediction" in st.session_state:
        prediction = st.session_state.prediction
        input_df = st.session_state.input_df

        # Histogram
        st.subheader("📈 Energy Usage Distribution")
        sim_data = np.random.normal(loc=350, scale=50, size=500)
        fig1, ax1 = plt.subplots()
        sns.histplot(sim_data, bins=30, kde=True, ax=ax1)
        ax1.axvline(prediction, color='red', linestyle='--', label="Your Prediction")
        ax1.legend()
        st.pyplot(fig1)

        # Feature importance
        st.subheader("📌 Feature Importance")
        try:
            importances = model.feature_importances_
            df = pd.DataFrame({
                'Feature': model_columns,
                'Importance': importances
            }).sort_values(by='Importance', ascending=False).head(10)
            fig2, ax2 = plt.subplots()
            sns.barplot(data=df, x='Importance', y='Feature', ax=ax2, palette='coolwarm')
            st.pyplot(fig2)
        except:
            st.info("Feature importance not available for this model.")
    else:
        st.info("🔍 Please make a prediction first on the Home page.")

# Page: Energy Tips
elif page == "📘 Energy Tips":
    st.header("🌱 Personalized Energy Tips")

    if "input_df" in st.session_state:
        row = st.session_state.input_df.iloc[0]
        tips = []
        if row["energy_star_home"] == 0:
            tips.append("✅ Upgrade to an Energy Star Certified Home.")
        if row["cooling_type_AC"] == 1:
            tips.append("✅ Optimize AC with smart thermostats.")
        if row["outside_temp_celsius"] > 30:
            tips.append("✅ Use ceiling fans or blackout curtains.")
        if row["monthly_income"] < 10000:
            tips.append("✅ Look for energy rebate programs.")

        for tip in tips:
            st.markdown(f"- {tip}")
    else:
        st.info("⏳ Make a prediction to view tips.")

# Page: About Project
elif page == "📂 About Project":
    st.header("📂 About This App")
    st.markdown("""
    - 🔧 **Model**: Trained with Random Forest on simulated household data  
    - 📊 **Inputs**: Number of occupants, income, temperature, house size, etc.  
    - 📉 **Outputs**: Estimated electricity consumption in kWh  
    - 🎯 **Goal**: Help users understand and manage their energy usage  
    """)




# import streamlit as st
# import pandas as pd
# import numpy as np
# import joblib
# import base64
# import datetime
# import requests

# # Load model
# model = joblib.load("Random_forest_model (2).pkl")

# # Convert image URL to base64
# def get_base64_from_url(image_url):
#     try:
#         response = requests.get(image_url)
#         if response.status_code == 200:
#             return base64.b64encode(response.content).decode()
#         else:
#             st.warning("⚠️ Could not fetch image.")
#             return ""
#     except Exception as e:
#         st.error(f"Image fetch error: {e}")
#         return ""

# # Set background
# bg_url = "https://images.unsplash.com/photo-1507525428034-b723cf961d3e?auto=format&fit=crop&w=1353&q=80"
# bg_image_base64 = get_base64_from_url(bg_url)

# if bg_image_base64:
#     st.markdown(
#         f"""
#         <style>
#         .stApp {{
#             background-image: url("data:image/jpg;base64,{bg_image_base64}");
#             background-size: cover;
#             background-position: center;
#             background-repeat: no-repeat;
#             background-attachment: fixed;
#         }}
#         .block-container {{
#             background-color: rgba(255, 255, 255, 0); /* Transparent container */
#             padding: 1rem 2rem;
#         }}
#         .main {{
#             background-color: rgba(255, 255, 255, 0); /* Remove default white */
#         }}
#         header, footer, .stSidebar {{
#             background-color: rgba(255,255,255,0); 
#         }}
#         h1, h2, h3, h4, p {{
#             color: white;
#             text-shadow: 1px 1px 3px black;
#         }}
#         </style>
#         """,
#         unsafe_allow_html=True
#     )

# # Streamlit config
# st.set_page_config(page_title="Home Energy Predictor", page_icon="🏡", layout="centered")

# # Expected model input columns
# model_columns = [
#     'num_occupants', 'house_size_sqft', 'monthly_income', 'outside_temp_celsius',
#     'year', 'month', 'day', 'season',
#     'heating_type_Electric', 'heating_type_Gas', 'heating_type_None',
#     'cooling_type_AC', 'cooling_type_Fan', 'cooling_type_None',
#     'manual_override_Y', 'manual_override_N',
#     'is_weekend', 'temp_above_avg', 'income_per_person', 'square_feet_per_person',
#     'high_income_flag', 'low_temp_flag',
#     'season_spring', 'season_summer', 'season_fall', 'season_winter',
#     'day_of_week_0', 'day_of_week_6', 'energy_star_home'
# ]

# # Title
# st.markdown("""<h1 style="text-align:center">🏡 Residential Energy Consumption Predictor</h1>""", unsafe_allow_html=True)
# st.markdown("<p style='text-align: center; font-size:18px;'>Estimate how much energy your home consumes based on lifestyle and home structure 🔍</p>", unsafe_allow_html=True)
# st.markdown("<hr style='border-top: 2px solid #ccc;'>", unsafe_allow_html=True)

# # Input fields
# st.header("📋 Household Information")

# col1, col2 = st.columns(2)
# with col1:
#     num_occupants = st.number_input("👨‍👩‍👧 Number of Occupants", min_value=1, value=4)
#     house_size_sqft = st.number_input("🏠 House Size (sqft)", min_value=100.0, value=1500.0)
#     monthly_income = st.number_input("💵 Monthly Income ($)", min_value=0, value=20000)
#     year = st.number_input("📆 Year", min_value=1970, max_value=2099, value=2025)
# with col2:
#     outside_temp_celsius = st.slider("🌡️ Outside Temperature (°C)", -20.0, 50.0, 25.0)
#     month = st.selectbox("📅 Month", list(range(1, 13)))
#     day = st.number_input("📅 Day", min_value=1, max_value=31, value=1)

# # Validate date
# try:
#     obj = datetime.date(int(year), int(month), int(day))
#     day_of_week = obj.weekday()
# except:
#     st.error("❗ Invalid date.")
#     st.stop()

# # Determine season
# if month in [12, 1, 2]:
#     season_label = "winter"
# elif month in [3, 4, 5]:
#     season_label = "summer"
# elif month in [6, 7, 8]:
#     season_label = "fall"
# else:
#     season_label = "spring"

# # Preferences
# st.subheader("🌬️ Heating and Cooling Preferences")
# col3, col4, col5 = st.columns(3)
# with col3:
#     heating_type = st.selectbox("🔥 Heating Type", ["Electric", "Gas", "None"])
# with col4:
#     cooling_type = st.selectbox("❄️ Cooling Type", ["AC", "Fan", "None"])
# with col5:
#     manual_override = st.radio("⚙️ Manual Override", ["Yes", "No"])
# energy_star_home = st.checkbox("🏅 Energy Star Certified Home", value=False)

# # Derived features
# is_weekend = int(day_of_week >= 5)
# temp_above_avg = int(outside_temp_celsius > 28)
# income_per_person = monthly_income / num_occupants
# square_feet_per_person = house_size_sqft / num_occupants
# high_income_flag = int(monthly_income > 40000)
# low_temp_flag = int(outside_temp_celsius < 15)

# # Prepare input
# input_data = {
#     'num_occupants': num_occupants,
#     'house_size_sqft': house_size_sqft,
#     'monthly_income': monthly_income,
#     'outside_temp_celsius': outside_temp_celsius,
#     'year': year,
#     'month': month,
#     'day': day,
#     'season': {'winter': 1, 'summer': 2, 'fall': 3, 'spring': 4}[season_label],
#     'heating_type_Electric': int(heating_type == "Electric"),
#     'heating_type_Gas': int(heating_type == "Gas"),
#     'heating_type_None': int(heating_type == "None"),
#     'cooling_type_AC': int(cooling_type == "AC"),
#     'cooling_type_Fan': int(cooling_type == "Fan"),
#     'cooling_type_None': int(cooling_type == "None"),
#     'manual_override_Y': int(manual_override == "Yes"),
#     'manual_override_N': int(manual_override == "No"),
#     'is_weekend': is_weekend,
#     'temp_above_avg': temp_above_avg,
#     'income_per_person': income_per_person,
#     'square_feet_per_person': square_feet_per_person,
#     'high_income_flag': high_income_flag,
#     'low_temp_flag': low_temp_flag,
#     'season_spring': int(season_label == "spring"),
#     'season_summer': int(season_label == "summer"),
#     'season_fall': int(season_label == "fall"),
#     'season_winter': int(season_label == "winter"),
#     'day_of_week_0': int(day_of_week == 0),
#     'day_of_week_6': int(day_of_week == 6),
#     'energy_star_home': int(energy_star_home)
# }
# input_df = pd.DataFrame([input_data])[model_columns]

# # Predict
# st.markdown("<hr style='border-top: 1px solid #ccc;'>", unsafe_allow_html=True)
# if st.button("🔍 Predict Energy Consumption"):
#     try:
#         prediction = model.predict(input_df)[0]
#         st.markdown(f"""
#             <div style='background-color:#f0f9f5; padding:20px; border-radius:10px; text-align:center'>
#                 <h2 style='color:#2E8B57;'>💡 Estimated Energy Usage: <span style='color:#006400'>{prediction:.2f} kWh</span></h2>
#             </div>
#         """, unsafe_allow_html=True)
#     except Exception as e:
#         st.error(f"⚠️ Prediction failed: {e}")

# # Show input summary
# with st.expander("📊 View Input Summary"):
#     st.dataframe(input_df.style.highlight_max(axis=1, color='lightgreen'))

# st.markdown("<hr style='border-top: 1px dashed #ccc;'>", unsafe_allow_html=True)      