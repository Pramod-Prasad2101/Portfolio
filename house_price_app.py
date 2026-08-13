import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import pandas as pd
import joblib
import numpy as np
import os

# Get the directory where this script is located
base_dir = os.path.dirname(os.path.abspath(__file__))

# Load your cleaned data (using relative paths)
df12 = pd.read_pickle(os.path.join(base_dir, 'df12.pkl'))

# Load the trained model
lr_clf = joblib.load(os.path.join(base_dir, 'house_price_model.pkl'))

# Extract location columns
locations = df12.columns[4:]  # Assuming the first four columns are 'total_sqft', 'bath', 'price', and 'bhk'

# Define the predict_price function with corrections
def predict_price(location, sqft, bath, bhk):
    try:
        # Create the input vector with the correct number of features
        x = np.zeros(len(df12.columns) - 1)
        x[0] = sqft
        x[1] = bath
        x[2] = bhk

        if location in df12.columns:
            loc_index = df12.columns.get_loc(location)
            x[loc_index - 1] = 1  # Adjust the location index since we skipped the 'price' column

        print("Input vector:", x)  # Debugging line
        prediction = lr_clf.predict([x])[0]
        print("Prediction:", prediction)  # Debugging line
        return prediction
    except Exception as e:
        print("Error in predict_price:", str(e))  # Debugging line
        raise

# Initialize the Dash app
app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1("Home Sweet Home Price Prediction"),
    
    html.Label("Total Square Feet:"),
    dcc.Input(id='total_sqft', type='number', value=1000, min=1),
    
    html.Label(" Number of Bathrooms:"),
    dcc.Input(id='bath', type='number', value=2, min=1),
    
    html.Label(" Number of Bedrooms (BHK):"),
    dcc.Input(id='bhk', type='number', value=2, min=1),
    
    html.Label("Location:"),
    dcc.Dropdown(id='location', options=[{'label': loc, 'value': loc} for loc in locations], value=locations[0]),
    
    html.Button('Predict', id='predict-button', n_clicks=0),
    html.Div(id='prediction-output', children='Enter values and press predict'),
])

@app.callback(
    Output('prediction-output', 'children'),
    [Input('total_sqft', 'value'),
     Input('bath', 'value'),
     Input('bhk', 'value'),
     Input('location', 'value'),
     Input('predict-button', 'n_clicks')]
)
def update_prediction(total_sqft, bath, bhk, location, n_clicks):
    if n_clicks > 0:
        try:
            # Use the predict_price function to get the predicted price
            predicted_price = predict_price(location, total_sqft, bath, bhk)
            return f"Predicted House Price: ${predicted_price:,.2f}"
        except Exception as e:
            return f"Error in prediction: {str(e)}"
    return "Enter values and press predict"

if __name__ == '__main__':
    app.run(debug=True)