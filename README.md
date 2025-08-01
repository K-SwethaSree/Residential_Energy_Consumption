# 🏠 Residential Energy Consumption Prediction Dashboard

This project is an interactive Streamlit web application that predicts residential energy usage based on lifestyle, weather, and housing characteristics. It uses multiple machine learning models and provides actionable energy-saving tips with visual insights.

## 🔍 Features

🔢 Input Customization: Number of occupants, house size, income, heating/cooling types, and more.

⚙️ ML Model Switching: Choose from Random Forest, Decision Tree, and Linear Regression.

📊 Visual Analytics: Histogram with prediction overlay, feature importance plot.

🌱 Personalized Tips: Energy-saving recommendations based on your input.

🖼️ Custom UI: Clean design with a dynamic background and styled components.

📁 Multiple Pages: Home, Visual Insights, Energy Tips, and Project Info.

## 📂 Project Structure

├── app.py # Main Streamlit app

├── Random-Forest-model.pkl # Trained Random Forest model

├── DecisionTree-model.pkl # Trained Decision Tree model

├── Linear-model.pkl # Trained Linear Regression model

├── requirements.txt # Python dependencies

## Models Used

Models: Random Forest, Decision Tree, Linear Regression (joblib-serialized).

Prediction Output: Energy usage in kWh.

## 🧾 Sample Inputs

| Feature               | Sample Value | Description                            |
| --------------------- | ------------ | -------------------------------------- |
| Number of Occupants   | 4            | Total people living in the house       |
| House Size (sqft)     | 1500         | Total square footage of the house      |
| Monthly Income (\$)   | 20000        | Household's monthly income             |
| Outside Temp (°C)     | 25.0         | Temperature outside the house          |
| Year                  | 2025         | Year of prediction                     |
| Month                 | 4            | Month (1–12)                           |
| Day                   | 15           | Day of the month                       |
| Heating Type          | Electric     | Heating method: Electric, Gas, or None |
| Cooling Type          | AC           | Cooling method: AC, Fan, or None       |
| Manual Override       | No           | User overrides automation: Yes or No   |
| Energy Star Certified | ❌            | Is the home Energy Star certified?     |

## 📊 Example Output

💡 Estimated Energy Usage: 342.15 kWh

(Based on selected model and input parameters)

## 📌 Pages Overview

🏠 Home: Enter inputs and get real-time energy prediction.

📊 Visual Insights: Histogram & top feature importance plot.

📘 Energy Tips: Smart suggestions tailored to your profile.

📂 About Project: Overview and usage information.

### Contact
K Swetha Sree

📧 [swethasreekongoti23@gmail.com]

🔗 www.linkedin.com/in/swetha-sree-55ab14317 | https://github.com/K-SwethaSree
