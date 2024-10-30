import streamlit as st  # Import the Streamlit library for creating web applications
import pandas as pd  # Import Pandas for data manipulation
import pickle  # Import Pickle for loading saved models
import numpy as np  # Import NumPy for numerical operations
import os  # Import OS for interacting with the operating system
from openai import OpenAI  # Import OpenAI for interacting with OpenAI's API
from utils import create_gauge_chart, create_model_probability_chart
import utils as ut
import uuid
import xgboost as xgb 

# Get the Groq API key from environment variables
groq_api_key = os.environ.get('GROQ_API_KEY')

# Initialize OpenAI client using Groq API
client = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=groq_api_key
)

# Load the XGBoost model from the JSON file
xgboost_model = xgb.Booster()
xgboost_model.load_model('xgb_model.json')

random_forest_model = pickle.load(open('rf_model.pkl', "rb"))
knn_model = pickle.load(open('knn_model.pkl', "rb"))

# Function to prepare input data for the machine learning models
def prepare_input(credit_score, location, gender, age, tenure, balance, num_products, has_credit_card, is_active_member, estimated_salary):
    input_dict = {
        'CreditScore': credit_score,
        'Age': age,
        'Tenure': tenure,
        'Balance': balance,
        'NumOfProducts': num_products,
        'HasCrCard': int(has_credit_card),
        'IsActiveMember': int(is_active_member),
        'EstimatedSalary': estimated_salary,
        'Geography_France': 1 if location == "France" else 0,
        'Geography_Germany': 1 if location == "Germany" else 0,
        'Geography_Spain': 1 if location == "Spain" else 0,
        'Gender_Male': 1 if gender == "Male" else 0,
        'Gender_Female': 1 if gender == "Female" else 0
    }
    input_df = pd.DataFrame([input_dict])
    return input_df, input_dict

# Function to make predictions using the selected models
def make_predictions(input_df, input_dict):
    probabilities = {
        'XGBoost': xgboost_model.predict_proba(input_df)[0][1],
        'Random Forest': random_forest_model.predict_proba(input_df)[0][1],
        'K-Nearest Neighbors': knn_model.predict_proba(input_df)[0][1]
    }

    avg_probability = np.mean(list(probabilities.values()))

    col1, col2 = st.columns(2)

    with col1:
        # Generate a truly unique key using uuid
        unique_key_gauge = str(uuid.uuid4())
        fig = ut.create_gauge_chart(avg_probability)
        st.plotly_chart(fig, use_container_width=True, key=f"gauge_chart_{unique_key_gauge}")
        st.write(f"The customer has a {avg_probability:.2%} probability of churning.")

    with col2:
        # Generate a truly unique key for the second chart
        unique_key_probs = str(uuid.uuid4())
        fig_probs = ut.create_model_probability_chart(probabilities)
        st.plotly_chart(fig_probs, use_container_width=True, key=f"probability_chart_{unique_key_probs}")

    return avg_probability, probabilities  # Return avg_probability and probabilities


def generate_personalized_email(surname, probability):
    email_body = f"""
    Dear {surname},

    We hope this message finds you well. At our bank, your satisfaction is our top priority, and we greatly value your relationship with us.

    We understand that there are many options available when it comes to financial services, but we wanted to personally reach out and let you know how much we appreciate your continued trust in us. Whether it’s personalized financial advice, competitive rates, or exclusive benefits tailored to your needs, we are committed to offering you the best possible banking experience.

    To ensure we continue meeting your expectations, we invite you to explore our latest offerings, from enhanced digital services to a dedicated support team ready to assist with any concerns you might have.

    If there’s anything we can do to improve your experience or if you have any questions, please don't hesitate to reach out. We’re here for you, and we’re confident we can continue to be your trusted financial partner for many years to come.

    Thank you once again for being part of the bank family.

    Best regards,
    Your Personal Banking Team
    """
    return email_body

# Function to explain the churn prediction using a language model
def explain_prediction(probability, input_dict, surname):
    prompt = f"""
    You are an expert data scientist at a bank, where you specialize in interpreting and explaining predictions of machine learning models.

    Your machine learning model has predicted that a customer named {surname} has a 
    {round(probability * 100, 1)}% probability of churning, based on the information provided below.

    Here is the customer's information:
    {input_dict}
    """
    print("EXPLANATION PROMPT:", prompt)
    raw_response = client.chat.completions.create(
        model="llama-3.2-3b-preview",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )
    return raw_response.choices[0].message.content

# Set the title of the Streamlit application
st.title("Customer Churn Prediction")

# Load customer data from a CSV file
df = pd.read_csv("churn.csv")

# Create a dropdown menu to select a customer
customers = [f"{row['CustomerId']} - {row['Surname']}" for _, row in df.iterrows()]
selected_customer_option = st.selectbox("Select a customer", customers)

# Extract customer ID and surname if a customer is selected
if selected_customer_option:
  selected_customer_id = int(selected_customer_option.split(" - ")[0])
  selected_surname = selected_customer_option.split(" - ")[1]
  selected_customer  = df.loc[df['CustomerId'] == selected_customer_id]

# Create input fields for the customer data
col1, col2 = st.columns(2)
with col1:
  credit_score = st.number_input("Credit Score", min_value=300, max_value=850, value=int(selected_customer['CreditScore'].iloc[0]))
  location = st.selectbox("Location", ["Spain", "France", "Germany"], index=["Spain", "France", "Germany"].index(selected_customer['Geography'].iloc[0]))
  gender = st.radio("Gender", ["Male", "Female"], index=0 if selected_customer['Gender'].iloc[0] == 'Male' else 1)
  age = st.number_input("Age", min_value=18, max_value=100, value=int(selected_customer['Age'].iloc[0]))
  tenure = st.number_input("Tenure (years)", min_value=0, max_value=50, value=int(selected_customer['Tenure'].iloc[0]))

with col2:
  balance = st.number_input("Balance", min_value=0.0, value=float(selected_customer['Balance'].iloc[0]))
  num_products = st.number_input("Number of Products", min_value=1, max_value=10, value=int(selected_customer['NumOfProducts'].iloc[0]))
  has_credit_card = st.checkbox("Has Credit Card", value=bool(selected_customer['HasCrCard'].iloc[0]))
  is_active_member = st.checkbox("Is Active Member", value=bool(selected_customer['IsActiveMember'].iloc[0]))
  estimated_salary = st.number_input("Estimated Salary", min_value=0.0, value=float(selected_customer['EstimatedSalary'].iloc[0]))

# Prepare the input data based on user inputs
input_df, input_dict = prepare_input(credit_score, location, gender, age, tenure, balance, num_products, has_credit_card, is_active_member, estimated_salary)

# Make predictions using the selected models
avg_probability, probabilities = make_predictions(input_df, input_dict)

# Generate an explanation for the churn prediction
explanation = explain_prediction(avg_probability, input_dict, selected_surname)

# Display the explanation in the Streamlit app
st.markdown("---")
st.subheader("Explanation of Prediction")
st.markdown(explanation)


