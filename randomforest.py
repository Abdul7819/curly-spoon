import streamlit as st
import pandas as pd
import pickle
import folium
from folium import IFrame
from folium.plugins import MarkerCluster
from PIL import Image
from io import BytesIO

# Load the model and label encoders
with open('trained_model.pkl', 'rb') as file:
    data = pickle.load(file)

model = data["model"]
label_encoders = data["label_encoders"]

# Define a function to preprocess the input data
def preprocess_input(example_data, encoders):
    for col in example_data.columns:
        if col in encoders:
            le = encoders[col]
            # Handle unseen labels by mapping to an integer (e.g., -1 for unknown)
            example_data[col] = [le.transform([value])[0] if value in le.classes_ else -1 for value in example_data[col]]
    return example_data

# Function to get latitude and longitude for a given location
def get_location_coordinates(location):
    location_coordinates = {
        "Dubai": [25.276987, 55.296249],
        "Singapore": [1.352083, 103.819839],
        "Berlin": [52.520008, 13.404954],
        "Tokyo": [35.682839, 139.759455],
        "San Francisco": [37.774929, -122.419418],
        "London": [51.507351, -0.127758],
        "Paris": [48.856613, 2.352222],
        "Sydney": [-33.868820, 151.209296],
        "Toronto": [43.651070, -79.347015],
        "New York": [40.712776, -74.005974]
    }
    return location_coordinates.get(location, [40.712776, -74.005974])  # Default to New York if location not found

# Streamlit app
st.title("Salary Prediction Web App")

# Create dropdown menus for categorical variables
job_title_options = list(label_encoders["Job_Title"].classes_)
industry_options = list(label_encoders["Industry"].classes_)
company_size_options = list(label_encoders["Company_Size"].classes_)
location_options = list(label_encoders["Location"].classes_)
ai_adoption_level_options = list(label_encoders["AI_Adoption_Level"].classes_)
automation_risk_options = list(label_encoders["Automation_Risk"].classes_)
required_skills_options = list(label_encoders["Required_Skills"].classes_)
remote_friendly_options = list(label_encoders["Remote_Friendly"].classes_)
job_growth_projection_options = list(label_encoders["Job_Growth_Projection"].classes_)

# Collect user inputs
job_title = st.selectbox("Job Title", job_title_options)
industry = st.selectbox("Industry", industry_options)
company_size = st.selectbox("Company Size", company_size_options)
location = st.selectbox("Location", location_options)
ai_adoption_level = st.selectbox("AI Adoption Level", ai_adoption_level_options)
automation_risk = st.selectbox("Automation Risk", automation_risk_options)
required_skills = st.selectbox("Required Skills", required_skills_options)
remote_friendly = st.selectbox("Remote Friendly", remote_friendly_options)
job_growth_projection = st.selectbox("Job Growth Projection", job_growth_projection_options)

# Create a DataFrame with the user inputs
input_data = pd.DataFrame({
    "Job_Title": [job_title],
    "Industry": [industry],
    "Company_Size": [company_size],
    "Location": [location],
    "AI_Adoption_Level": [ai_adoption_level],
    "Automation_Risk": [automation_risk],
    "Required_Skills": [required_skills],
    "Remote_Friendly": [remote_friendly],
    "Job_Growth_Projection": [job_growth_projection]
})

# Preprocess input data and make prediction
input_data = preprocess_input(input_data, label_encoders)
input_data = input_data.astype(float)

if st.button('Predict Salary'):
    prediction = model.predict(input_data)
    st.write(f"Predicted Salary: ${prediction[0]:,.2f}")

    # Get latitude and longitude based on the selected location
    latitude, longitude = get_location_coordinates(location)

    # Create a base map
    m = folium.Map(location=[latitude, longitude], zoom_start=12)

    # Define the icon for the industry
    industry_icon = folium.CustomIcon(icon_image="https://img.icons8.com/ios-filled/50/000000/business.png", icon_size=(30, 30))

    # Define the icon for the salary (dollar sign)
    salary_icon = folium.CustomIcon(icon_image="https://img.icons8.com/ios-filled/50/000000/coins.png", icon_size=(30, 30))

    # Add markers to the map
    folium.Marker([latitude, longitude],
                  popup=f"Industry: {industry}<br>Predicted Salary: ${prediction[0]:,.2f}",
                  icon=industry_icon).add_to(m)

    folium.Marker([latitude, longitude],
                  popup=f"Predicted Salary: ${prediction[0]:,.2f}",
                  icon=salary_icon).add_to(m)

    # Save map to HTML file
    map_html = m._repr_html_()

    # Render map in Streamlit
    st.write("Location Map:")
    st.components.v1.html(map_html, height=500, width=800)

    # Show selected attributes
    st.subheader("Selected Attributes")
    attributes_df = pd.DataFrame({
        "Attribute": [
            "Job Title", "Industry", "Company Size", "Location", 
            "AI Adoption Level", "Automation Risk", "Required Skills", 
            "Remote Friendly", "Job Growth Projection"
        ],
        "Value": [
            job_title, industry, company_size, location,
            ai_adoption_level, automation_risk, required_skills,
            remote_friendly, job_growth_projection
        ]
    })

    st.write(attributes_df)

    # Plot selected attributes as a bar chart
    st.subheader("Selected Attributes Overview")

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(attributes_df["Attribute"], range(len(attributes_df)), color='skyblue')
    ax.set_yticks(range(len(attributes_df)))
    ax.set_yticklabels(attributes_df["Attribute"])
    ax.set_xlabel('Value')
    ax.set_title('Selected Attributes Overview')

    for index, value in enumerate(attributes_df["Value"]):
        ax.text(0, index, f' {value}', color='black', va='center')

    st.pyplot(fig)
