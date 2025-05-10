import streamlit as st
import tensorflow as tf
import numpy as np
import librosa
from matplotlib import pyplot
from tensorflow.image import resize
from tensorflow.keras.models import load_model as tf_load_model

st.cache_resource()
def load_Trained_model():
    model = tf_load_model("./Trained_model.h5")
    return model

def load_and_preprocess_data(file_path, target_shape=(150, 150)):
    data = []
    audio_data, sample_rate = librosa.load(file_path, sr=None)
    # Perform preprocessing (e.g., convert to Mel spectrogram and resize)
    # Define the duration of each chunk and overlap
    chunk_duration = 4  # seconds
    overlap_duration = 2  # seconds
                
    # Convert durations to samples
    chunk_samples = chunk_duration * sample_rate
    overlap_samples = overlap_duration * sample_rate
                
    # Calculate the number of chunks
    num_chunks = int(np.ceil((len(audio_data) - chunk_samples) / (chunk_samples - overlap_samples))) + 1
                
    # Iterate over each chunk
    for i in range(num_chunks):
                    # Calculate start and end indices of the chunk
        start = i * (chunk_samples - overlap_samples)
        end = start + chunk_samples
                    
                    # Extract the chunk of audio
        chunk = audio_data[start:end]
                    
                    # Compute the Mel spectrogram for the chunk
        mel_spectrogram = librosa.feature.melspectrogram(y=chunk, sr=sample_rate)
                    
                #mel_spectrogram = librosa.feature.melspectrogram(y=audio_data, sr=sample_rate)
        mel_spectrogram = resize(np.expand_dims(mel_spectrogram, axis=-1), target_shape)
        data.append(mel_spectrogram)
    
    return np.array(data)

def model_prediction(X_test):
    model = load_Trained_model()
    y_pred = model.predict(X_test)
    predicted_categories = np.argmax(y_pred,axis=1)
    unique_elements, counts = np.unique(predicted_categories, return_counts=True)
    #print(unique_elements, counts)
    max_count = np.max(counts)
    max_elements = unique_elements[counts == max_count]
    return max_elements[0]

#UI/UX STARTS HERE ONWARDS
st.sidebar.title("Dashboard")

app_mode = st.sidebar.selectbox("Select Page",["Home","Project Description", "Music Prediction"])

#Landing page

if(app_mode =="Home"):
    st.markdown("""
<style>
/* Full-page gradient background */
[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #2c003e, #8a007f, #ff006e, #ff4d6d);
    background-size: 400% 400%;
    animation: pulseBG 20s ease infinite;
    overflow-y: auto;
}

/* Animated gradient movement */
@keyframes pulseBG {
    0% {background-position: 0% 50%;}
    50% {background-position: 100% 50%;}
    100% {background-position: 0% 50%;}
}

/* Main content block styling */
.block-container {
    max-width: 950px;
    margin: 0 auto;
    background: rgba(0, 0, 0, 0.2);  /* translucent dark glass effect */
    padding: 2rem;
    border-radius: 12px;
    box-shadow: 0 0 20px rgba(255, 0, 102, 0.4);
    color: #fff;
}

/* Sidebar theme - red dashboard */
[data-testid="stSidebar"] {
    background-color: #1a001f;
}

h1, h2, h3, h4 {
    color: #9B4DFF;
    text-shadow: 0 0 10px #ff3399;
}

strong {
    color: #ffffff;
}

a {
    color: #ffd6f6;
    font-weight: bold;
    text-decoration: underline;
}
</style>
""", unsafe_allow_html=True)




    st.markdown(''' ## Welcome To Music Genre Classification''')
    image_path = "Music_Homepage.jpg"
    st.image(image_path, use_column_width=True)

    st.markdown("""
**The aim of this project is to help identify the genre of music from audio tracks uploaded by users. Upload an audio file for the deep learning algorithm to analyse**

### How It Works
1. **Upload Audio:** Go to the **Music Prediction** page and upload an audio file.
2. **Analysis:** The Application will process the audio using the deep learning algorithms to classify it into one of the predefined genres.
3. **Results:** The predicted genre will be displayed along with related information.

### Get Started
Click on the **Music Prediction** page in the sidebar to upload an audio file and Discover the magic of AI tools in the world of music!

### About Us
Learn more about the project on the **About** page.
""")

#Dscrpn pg

elif(app_mode=="Project Description"):
     st.markdown("""
                ### About Project
                Music. Experts have been trying for a long time to understand sound and what differenciates one song from another. How to visualize sound. What makes a tone different from another.

                This dataset will hopefully give the opportunity to do just that.

                ### About Dataset
                #### Content
                1. **genres original** - A collection of 10 genres with 100 audio files each, all having a length of 30 seconds (the famous GTZAN dataset, the MNIST of sounds)
                2. **List of Genres** - blues, classical, country, disco, hiphop, jazz, metal, pop, reggae, rock
                3. **images original** - A visual representation for each audio file. One way to classify data is through neural networks. Because NNs (like CNN, what we will be using today) usually take in some sort of image representation, the audio files were converted to Mel Spectrograms to make this possible.
                4. **2 CSV files** - Containing features of the audio files. One file has for each song (30 seconds long) a mean and variance computed over multiple features that can be extracted from an audio file. The other file has the same structure, but the songs were split before into 3 seconds audio files (this way increasing 10 times the amount of data we fuel into our classification models). With data, more is always better.
                """)
     st.markdown("""
<style>
/* Full-page gradient background */
[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #2c003e, #8a007f, #ff006e, #ff4d6d);
    background-size: 400% 400%;
    animation: pulseBG 20s ease infinite;
    overflow-y: auto;
}

/* Animated gradient movement */
@keyframes pulseBG {
    0% {background-position: 0% 50%;}
    50% {background-position: 100% 50%;}
    100% {background-position: 0% 50%;}
}


/* Sidebar theme - red dashboard */
[data-testid="stSidebar"] {
    background-color: #1a001f;
}

h1, h2, h3, h4 {
    color: #9B4DFF;
    text-shadow: 0 0 10px #ff3399;
}

strong {
    color: #ffffff;
}

a {
    color: #ffd6f6;
    font-weight: bold;
    text-decoration: underline;
}
</style>
""", unsafe_allow_html=True)
     
elif(app_mode=="Music Prediction"):
    st.markdown("""
<style>
/* Full-page gradient background */
[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #2c003e, #8a007f, #ff006e, #ff4d6d);
    background-size: 400% 400%;
    animation: pulseBG 20s ease infinite;
    overflow-y: auto;
}

/* Animated gradient movement */
@keyframes pulseBG {
    0% {background-position: 0% 50%;}
    50% {background-position: 100% 50%;}
    100% {background-position: 0% 50%;}
}


/* Sidebar theme - red dashboard */
[data-testid="stSidebar"] {
    background-color: #1a001f;
}

h1, h2, h3, h4 {
    color: #9B4DFF;
    text-shadow: 0 0 10px #ff3399;
}

strong {
    color: #ffffff;
}

a {
    color: #ffd6f6;
    font-weight: bold;
    text-decoration: underline;
}
</style>
""", unsafe_allow_html=True)
    
    st.header("Model Prediction")
    test_mp3 = st.file_uploader("Upload audio file", type=["mp3"])

    if test_mp3 is not None:
        filepath = "Test_Music/"+test_mp3.name

    #Test audio btn
    if(st.button("Play Audiofile")):
        st.audio(test_mp3)
    
    #ai part

    if(st.button("Predict")):
        with st.spinner("Please wait..."):
            X_test = load_and_preprocess_data(filepath)
            result_index = model_prediction(X_test)
            label = ['blues', 'classical','country','disco','hiphop','jazz','metal','pop','reggae','rock']
        
            st.markdown("Model Prediction: {} music!!".format(label[result_index]))
