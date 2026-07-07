import gardio as gr
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder
loaded_model = joblib.load('Car_Price_model.pkl')

# Re-initialize LabelEncoders for consistent preprocessing if needed.
# In this specific case, the original LabelEncoders were applied directly to the DataFrame columns.
# If you need to transform new input values, you'll need the *original* fitted encoders.
# For demonstration, I'll create a dummy encoder for new inputs based on the original data.
# In a real-world scenario, you would save these encoders with the model or fit them on the full dataset.

# This is a simplified approach assuming the app will use similar data to the training.
# For robust deployment, you'd save and load the fitted LabelEncoders as well.
fuel_type_le = LabelEncoder()
fuel_type_le.fit(['Petrol', 'Diesel', 'CNG'])

seller_type_le = LabelEncoder()
seller_type_le.fit(['Dealer', 'Individual'])

transmission_le = LabelEncoder()
transmission_le.fit(['Manual', 'Automatic'])

print("Model and LabelEncoders loaded/initialized successfully!")
def predict_car_price(Present_Price, Kms_Driven, Fuel_Type, Seller_Type, Transmission, Owner, Year):
    # Calculate Car_Age
    Car_Age = 2026 - Year

    # Encode categorical features
    fuel_type_encoded = fuel_type_le.transform([Fuel_Type])[0]
    seller_type_encoded = seller_type_le.transform([Seller_Type])[0]
    transmission_encoded = transmission_le.transform([Transmission])[0]

    # Create a DataFrame for the input features, ensuring column order matches training data
    input_data = pd.DataFrame([[Present_Price, Kms_Driven, fuel_type_encoded, seller_type_encoded, transmission_encoded, Owner, Car_Age]],
                              columns=['Present_Price', 'Kms_Driven', 'Fuel_Type', 'Seller_Type', 'Transmission', 'Owner', 'Car_Age'])

    # Make prediction
    print(input_data)
    prediction = loaded_model.predict(input_data)[0]
    print(prediction)
    return round(prediction, 2)
inputs = [
    gr.Number(label="Present Price (in Lakhs)", minimum=0, maximum=100.0, step=0.1), # Changed minimum from 0.1 to 0
    gr.Number(label="Kilometers Driven", minimum=0, maximum=500000, step=100),
    gr.Dropdown(label="Fuel Type", choices=fuel_type_le.classes_.tolist()),
    gr.Dropdown(label="Seller Type", choices=seller_type_le.classes_.tolist()),
    gr.Dropdown(label="Transmission Type", choices=transmission_le.classes_.tolist()),
    gr.Number(label="Number of Owners (0, 1, 2, etc.)", minimum=0, maximum=5, step=1),
    gr.Number(label="Manufacturing Year", minimum=1990, maximum=2025, step=1)
]

# Define Gradio Interface output
output = gr.Textbox(label="Estimated Selling Price (in Lakhs)")

# Create and launch the Gradio interface
app = gr.Interface(
    fn=predict_car_price,
    inputs=inputs,
    outputs=output,
    title="Car Price Prediction App",
    description="Enter car details to get an estimated selling price."
)

# The messages about 'Keyboard interruption' are not errors, but indicate the server was stopped.
# The app itself launched successfully.
# Setting debug=False can make the output less verbose, but the core functionality is correct.
app.launch(debug=True, share=True)
