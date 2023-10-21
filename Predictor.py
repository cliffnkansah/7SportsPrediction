import pickle
from sklearn.preprocessing import StandardScaler
import numpy as np
import streamlit as st

# Load the trained model
model = pickle.load(open("bestmodel.pkl", "rb"))

# Load the StandardScaler
scaler = pickle.load(open("scaler.pkl", "rb"))

# Define the Streamlit app
def main():
    st.title("FIFA Player Overall Rating Predictor")

#creating sliders for the features
    st.write("Enter feature values for prediction:")
    movement_reactions =  st.slider("movement_reactions", min_value=0, max_value=100, value=50)
    skill_dribbling = st.slider("skill_dribbling", min_value=0, max_value=100, value=50)
    passing = st.slider("passing", min_value=0, max_value=100, value=50)
    potential = st.slider("potential", min_value=0, max_value=100, value=50)
    dribbling = st.slider("dribbling", min_value=0, max_value=100, value=50)
    attacking_short_passing = st.slider("attacking_short_passing", min_value=0, max_value=100, value=50)
    physic = st.slider("physic", min_value=0, max_value=100, value=50)
    skill_long_passing = st.slider("skill_long_passing", min_value=0, max_value=100, value=50)
    movement_agility = st.slider("movement_agility", min_value=0, max_value=100, value=50)
    skill_moves = st.slider("skill_moves", min_value=0, max_value=100, value=50)
    shooting = st.slider("shooting", min_value=0, max_value=100, value=50)
    skill_ball_control = st.slider("skill_ball_control", min_value=0, max_value=100, value=50)
    mentality_vision = st.slider("mentality_vision", min_value=0, max_value=100, value=50)
    weight_kg = st.slider("weight_kg", min_value=0, max_value=100, value=50)
    attacking_crossing = st.slider("attacking_crossing", min_value=0, max_value=100, value=50)
    
    
    
    try:
        input_features = np.array([
            movement_reactions,skill_dribbling, passing, potential, dribbling, attacking_short_passing,
            physic, skill_long_passing, movement_agility, skill_moves, shooting, skill_ball_control,
            mentality_vision, weight_kg, attacking_crossing
        ]).astype(float).reshape(1, -1)

        scaled_input = scaler.transform(input_features)

        # Ensure the input features are in the correct data type (float)
        if scaled_input.dtype != np.float64:
            st.write("Warning: Input features are not in the correct data type (float).")

        if st.button("Predict Overall"):
            prediction = model.predict(scaled_input)
            st.write(f"Predicted Overall Rating: {prediction[0]:.2f}")
    except Exception as e:
        st.write(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()

