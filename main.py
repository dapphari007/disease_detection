import streamlit as st
import tensorflow as tf
import numpy as np


#Tensorflow Model Prediction
def model_prediction(test_image):
    model = tf.keras.models.load_model("trained_model.keras")
    image = tf.keras.preprocessing.image.load_img(test_image,target_size=(128,128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr]) #convert single image to batch
    predictions = model.predict(input_arr)
    return np.argmax(predictions) #return index of max element

#Sidebar
st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select Page", ["Home", "About", "Disease Recognition"])

#Main Page
if(app_mode=="Home"):
    st.header("PLANT DISEASE RECOGNITION SYSTEM")
    image_path = "home_page.jpeg"
    st.image(image_path,use_column_width=True)
    st.markdown("""
    Welcome to the Plant Disease Recognition System! 🌿🔍
    
    Our mission is to help in identifying plant diseases efficiently. Upload an image of a plant, and our system will analyze it to detect any signs of diseases. Together, let's protect our crops and ensure a healthier harvest!

    ### How It Works
    1. **Upload Image:** Go to the **Disease Recognition** page and upload an image of a plant with suspected diseases.
    2. **Analysis:** Our system will process the image using advanced algorithms to identify potential diseases.
    3. **Results:** View the results and recommendations for further action.

    ### Why Choose Us?
    - **Accuracy:** Our system utilizes state-of-the-art machine learning techniques for accurate disease detection.
    - **User-Friendly:** Simple and intuitive interface for seamless user experience.
    - **Fast and Efficient:** Receive results in seconds, allowing for quick decision-making.

    ### Get Started
    Click on the **Disease Recognition** page in the sidebar to upload an image and experience the power of our Plant Disease Recognition System!

    ### About Us
    Learn more about the project, our team, and our goals on the **About** page.
    """)

#About Project
elif(app_mode=="About"):
    st.header("About")
    st.markdown("""
                #### About Dataset
                This dataset is recreated using offline augmentation from the original dataset.The original dataset can be found on this github repo.
                This dataset consists of about 87K rgb images of healthy and diseased crop leaves which is categorized into 38 different classes.The total dataset is divided into 80/20 ratio of training and validation set preserving the directory structure.
                A new directory containing 33 test images is created later for prediction purpose.
                #### Content
                1. train (70295 images)
                2. test (33 images)
                3. validation (17572 images)

                """)

# Prediction Page
elif app_mode == "Disease Recognition":
    st.header("Disease Recognition")
    test_image = st.file_uploader("Choose an Image:")
    if st.button("Show Image"):
        st.image(test_image, use_column_width=True)
    # Predict button
    if st.button("Predict"):
        st.write("Our Prediction")
        result_index = model_prediction(test_image)
        # Reading Labels
        class_name = [
            'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
            'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 
            'Cherry_(including_sour)___healthy', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 
            'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 
            'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 
            'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot',
            'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 
            'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy', 
            'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew', 
            'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot', 
            'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold', 
            'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 
            'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
            'Tomato___healthy'
        ]
        predicted_disease = class_name[result_index]
        st.success(f"Model is Predicting it's a {predicted_disease}")

# Dictionary containing causes, remedies, and prevention methods
disease_info = {
    "Apple___Apple_scab": {
        "cause": "Fungal infection caused by Venturia inaequalis.",
        "remedy": "Apply fungicides like Captan or Mancozeb. Remove and destroy infected leaves.",
        "climatic_condition": "Cool and moist conditions favor the spread."
    },
    "Apple___Black_rot": {
        "cause": "Fungal infection caused by Botryosphaeria obtusa.",
        "remedy": "Prune out infected branches and apply fungicides such as Thiophanate-methyl.",
        "climatic_condition": "Warm and wet conditions promote disease development."
    },
    "Apple___Cedar_apple_rust": {
        "cause": "Fungal infection caused by Gymnosporangium juniperi-virginianae.",
        "remedy": "Remove nearby juniper plants and apply fungicides like Myclobutanil.",
        "climatic_condition": "High humidity and moderate temperatures."
    },
    "Apple___healthy": {
        "cause": "No disease detected.",
        "remedy": "Maintain regular care and monitor for potential issues.",
        "climatic_condition": "Suitable care ensures health."
    },
    "Blueberry___healthy": {
        "cause": "No disease detected.",
        "remedy": "Continue providing proper care and monitoring.",
        "climatic_condition": "Well-drained soil and adequate sunlight are essential."
    },
    "Cherry_(including_sour)___Powdery_mildew": {
        "cause": "Fungal infection caused by Podosphaera clandestina.",
        "remedy": "Apply sulfur-based fungicides and ensure proper air circulation.",
        "climatic_condition": "Dry weather with high humidity."
    },
    "Cherry_(including_sour)___healthy": {
        "cause": "No disease detected.",
        "remedy": "Maintain care and monitor for potential signs of disease.",
        "climatic_condition": "Healthy growing conditions."
    },
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": {
        "cause": "Fungal infection caused by Cercospora zeae-maydis.",
        "remedy": "Use resistant varieties and apply fungicides like Azoxystrobin.",
        "climatic_condition": "Warm and humid conditions favor the disease."
    },
    "Corn_(maize)___Common_rust_": {
        "cause": "Fungal infection caused by Puccinia sorghi.",
        "remedy": "Use rust-resistant varieties and apply fungicides such as Mancozeb.",
        "climatic_condition": "Cool and moist conditions."
    },
    "Corn_(maize)___Northern_Leaf_Blight": {
        "cause": "Fungal infection caused by Exserohilum turcicum.",
        "remedy": "Apply fungicides and practice crop rotation.",
        "climatic_condition": "Moderate temperatures and high humidity."
    },
    "Corn_(maize)___healthy": {
        "cause": "No disease detected.",
        "remedy": "Ensure consistent care and monitor regularly.",
        "climatic_condition": "Healthy conditions maintained."
    },
    "Grape___Black_rot": {
        "cause": "Fungal infection caused by Guignardia bidwellii.",
        "remedy": "Apply fungicides like Mancozeb and remove infected parts.",
        "climatic_condition": "Warm, wet weather promotes spread."
    },
    "Grape___Esca_(Black_Measles)": {
        "cause": "Complex fungal infection involving several species.",
        "remedy": "Prune infected areas and apply fungicides.",
        "climatic_condition": "Hot and dry conditions with periodic rain."
    },
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": {
        "cause": "Fungal infection caused by Pseudocercospora vitis.",
        "remedy": "Improve air circulation and apply fungicides.",
        "climatic_condition": "Warm and humid environments."
    },
    "Grape___healthy": {
        "cause": "No disease detected.",
        "remedy": "Maintain proper care and monitor for signs of disease.",
        "climatic_condition": "Healthy growing conditions."
    },
    "Orange___Haunglongbing_(Citrus_greening)": {
        "cause": "Bacterial infection transmitted by psyllid insects.",
        "remedy": "Use insecticides to control psyllids and remove infected trees.",
        "climatic_condition": "Warm climates with insect presence."
    },
    "Peach___Bacterial_spot": {
        "cause": "Bacterial infection caused by Xanthomonas arboricola.",
        "remedy": "Apply copper-based bactericides and avoid overhead watering.",
        "climatic_condition": "Warm and wet conditions promote spread."
    },
    "Peach___healthy": {
        "cause": "No disease detected.",
        "remedy": "Maintain regular care and monitor for potential issues.",
        "climatic_condition": "Healthy growing conditions."
    },
    "Pepper,_bell___Bacterial_spot": {
        "cause": "Bacterial infection caused by Xanthomonas campestris.",
        "remedy": "Apply copper-based sprays and practice crop rotation.",
        "climatic_condition": "Warm and moist conditions favor the disease."
    },
    "Pepper,_bell___healthy": {
        "cause": "No disease detected.",
        "remedy": "Ensure proper care and monitor regularly.",
        "climatic_condition": "Healthy growing conditions."
    },
    "Potato___Early_blight": {
        "cause": "Fungal infection caused by Alternaria solani.",
        "remedy": "Use fungicides like Mancozeb and remove infected plants.",
        "climatic_condition": "Warm and wet conditions favor spread."
    },
    "Potato___Late_blight": {
        "cause": "Fungal infection caused by Phytophthora infestans.",
        "remedy": "Apply fungicides like Chlorothalonil and destroy infected debris.",
        "climatic_condition": "Cool and moist conditions promote disease."
    },
    "Potato___healthy": {
        "cause": "No disease detected.",
        "remedy": "Maintain proper care and monitor for potential signs of disease.",
        "climatic_condition": "Healthy growing conditions."
    },
    "Raspberry___healthy": {
        "cause": "No disease detected.",
        "remedy": "Continue regular care and monitor plants.",
        "climatic_condition": "Suitable care ensures health."
    },
    "Soybean___healthy": {
        "cause": "No disease detected.",
        "remedy": "Ensure proper nutrition and monitor regularly.",
        "climatic_condition": "Healthy growing conditions maintained."
    },
    "Squash___Powdery_mildew": {
        "cause": "Fungal infection caused by Erysiphales family.",
        "remedy": "Apply sulfur-based fungicides and ensure good air circulation.",
        "climatic_condition": "Dry weather with high humidity."
    },
    "Strawberry___Leaf_scorch": {
        "cause": "Fungal infection caused by Diplocarpon earlianum.",
        "remedy": "Remove infected leaves and apply fungicides.",
        "climatic_condition": "Warm and wet conditions favor the spread."
    },
    "Strawberry___healthy": {
        "cause": "No disease detected.",
        "remedy": "Maintain proper care and monitor regularly.",
        "climatic_condition": "Healthy growing conditions maintained."
    },
    "Tomato___Bacterial_spot": {
        "cause": "Bacterial infection caused by Xanthomonas vesicatoria.",
        "remedy": "Apply copper-based bactericides and remove infected leaves.",
        "climatic_condition": "Warm and wet conditions promote disease spread."
    },
    "Tomato___Early_blight": {
        "cause": "Fungal infection caused by Alternaria solani.",
        "remedy": "Use fungicides like Mancozeb and ensure crop rotation.",
        "climatic_condition": "Warm and humid conditions favor spread."
    },
    "Tomato___Late_blight": {
        "cause": "Fungal infection caused by Phytophthora infestans.",
        "remedy": "Apply fungicides like Chlorothalonil and improve drainage.",
        "climatic_condition": "Cool and moist conditions promote spread."
    },
    "Tomato___Leaf_Mold": {
        "cause": "Fungal infection caused by Passalora fulva.",
        "remedy": "Apply fungicides and ensure proper ventilation.",
        "climatic_condition": "High humidity and moderate temperatures."
    },
    "Tomato___Septoria_leaf_spot": {
        "cause": "Fungal infection caused by Septoria lycopersici.",
        "remedy": "Remove infected leaves and apply fungicides like Mancozeb.",
        "climatic_condition": "Warm and wet conditions favor the disease."
    },
    "Tomato___Spider_mites Two-spotted_spider_mite": {
        "cause": "Infestation by Tetranychus urticae.",
        "remedy": "Use miticides and encourage natural predators like ladybugs.",
        "climatic_condition": "Hot and dry conditions promote infestation."
    },
    "Tomato___Target_Spot": {
        "cause": "Fungal infection caused by Corynespora cassiicola.",
        "remedy": "Apply fungicides and ensure good air circulation.",
        "climatic_condition": "Warm and humid conditions favor spread."
    },
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": {
        "cause": "Viral infection transmitted by whiteflies.",
        "remedy": "Use insecticides to control whiteflies and remove infected plants.",
        "climatic_condition": "Warm climates with high whitefly populations."
    },
    "Tomato___Tomato_mosaic_virus": {
        "cause": "Viral infection spread through contaminated tools or seeds.",
        "remedy": "Use virus-free seeds and disinfect tools.",
        "climatic_condition": "Varies with environmental conditions."
    },
    "Tomato___healthy": {
        "cause": "No disease detected.",
        "remedy": "Maintain proper care and monitor regularly.",
        "climatic_condition": "Healthy growing conditions maintained."
    }
}

# Fetch additional information
if predicted_disease in disease_info:
    info = disease_info[predicted_disease]
    st.markdown(f"### Cause:\n{info['cause']}")
    st.markdown(f"### Remedies:\n{info['remedy']}")
    st.markdown(f"### Climatic_condition:\n{info['climatic_condition']}")
else:
    st.warning("No additional information available for this disease.")