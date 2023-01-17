from PIL import Image, ImageOps
import streamlit as st
from streamlit_drawable_canvas import st_canvas
import numpy as np
from numpy import argmax
from tensorflow.keras.utils import load_img
from tensorflow.keras.utils import img_to_array
from keras.models import load_model
from gtts import gTTS
st.set_page_config(
    page_title="Handwritten Numeric Recognizer using CNN",
    page_icon="🔤",
    layout = "wide",
    initial_sidebar_state="expanded")  

with st.sidebar:
    st.subheader("Breif Description :")
    st.markdown("""
            - Based on **:blue[CNN - Convolutional Neural Networks]**, a type of Deep-Learning model .
            - Trained on a MNIST Dataset with **:green[Tensorflow]**  , consists of 60000 train , 10000 test images .
            - Text to Speech done using **:red[Pyttsx3]** , a Python library (custom TS coming soon...) .
            - More details in **:green[GitHub]** Repo :- .
            """)
    st.markdown("""---""")
    st.subheader("Upcoming Fixes :-")
    st.markdown("""
            - This version only supports **:green[Numbers / Digits]**  , letters and characters support coming soon...
            - Still **:blue[Glitchs]**, with some numbers having a unique pattern . """)
            # - Text to Speech done using **:red[Pyttsx3]** , a Python library (custom TS coming soon...) .
            # - More details in **:green[GitHub]** Repo :- .
            # """)
    st.markdown("""---""")
    sidebar_text = st.markdown("Result :- " + " Start to see the **:red[result.....]** ")
    
    st.image("assests/example1.png")
    st.image("assests/example2.png")



st.markdown("<h1 style='font-family: monaco, monospace;text-align: center;font-size: 70px;background: linear-gradient(to left,violet,indigo,blue,green, yellow, orange,red);-webkit-background-clip: text;color: transparent;'>Handwritten Numeric Recognizer</h1> ", unsafe_allow_html=True)
# st.markdown("<h1 style='font-family: monaco, monospace;text-align: center;font-size: 70px;background: linear-gradient(to left,violet,indigo,blue,green, yellow, orange,red);-webkit-background-clip: text;color: transparent;'>with sound </h1> ", unsafe_allow_html=True)
st.markdown("""
        <style>
            .sidebar .sidebar-content {
                width: 375px;
            }
            .big-font {
                font-size:80px;
                font-weight : 1000;
            }
            .small-font {
                font-size:30px;
                font-weight : 500;
            }
            .MuiGrid-item{
                font-size:19px;
            }
            .css-1yy6isu p{
                font-size:25px;
            }
            .st-dx{
                font-size :18px;
            }
            .css-1fv8s86 p{
                font-size:18px;
            }
        </style>""", unsafe_allow_html=True)
st.markdown('<p class="big-font"><center class="small-font">Made by :- Srivatsa Gorti</center></p>', unsafe_allow_html=True)
st.markdown("""---""")
# st.markdown('<p> My GitHub 📖: https://github.com/srivatsacool </p>', unsafe_allow_html=True)
col1, col2 = st.columns([1, 1])
# Create a canvas componentI
with col1:
    st.text("Draw here ....")
    canvas_result = st_canvas(
        fill_color="rgba(255, 165, 0, 0.3)",  # Fixed fill color with some opacity
        stroke_width=15,
        stroke_color="white",
        background_color='#000000',
        update_streamlit=True,
        height=400,
        width=400, 
        background_image = None ,#Image.open("number_5.png"),
        drawing_mode="freedraw",
        key="canvas",
    )
    

# Do something interesting with the image data and paths
if canvas_result.image_data is not None:
    col2.text("The drawn digit (Input)")
    col2.image(canvas_result.image_data)
    print(canvas_result.image_data)
    print(canvas_result.image_data.shape)
    img = Image.fromarray(canvas_result.image_data)
    img = ImageOps.grayscale(img)
    img.save("assests/pil.png")


def load_image(filename):
        # load the image
        img = load_img(filename, color_mode = "grayscale", target_size=(28, 28))
        # convert to array
        img = img_to_array(img)
        print(img.shape)
        # reshape into a single sample with 1 channel
        img = img.reshape(1, 28, 28, 1)
        # prepare pixel data
        img = img.astype('float32')
        img = img / 255.0
        return img

# load an image and predict the class
def run_main():
        # load the image
        img = load_image('assests/pil.png')
        # load model
        model = load_model('final_model.h5')
        # predict the class
        predict_value = model.predict(img)
        digit = argmax(predict_value)
        print(predict_value)
        print(digit)
        return digit

st.markdown("""---""")
columns = st.columns((2, 1, 2))

button_pressed = columns[1].button('Start !!')
col = st.columns((1, 1, 10))
col[0].markdown("Result :-")
result_txt = col[2].markdown(" Start to see the **:red[result.....]** ")
# st.markdown(f"Result :-  Start to see the **:red[result]** ")

if button_pressed:
    res = run_main()
    result_txt.markdown(f"The predicted number is **:red[{res}]**")
    sidebar_text.markdown("Result :- "+f"The predicted number is **:red[{res}]**")
    speech = gTTS(text = f"The predicted result is the Number {res}", lang = 'en', slow = False)
    speech.save('assests/res_trans.mp3') 
    audio_file = open('assests/res_trans.mp3', 'rb')    
    audio_bytes = audio_file.read()    
    st.audio(audio_bytes, format='audio/ogg',start_time=0)

st.markdown("""---""")
st.subheader("Examples :- ")
st.image("assests/example1.png")
<<<<<<< HEAD
st.image("assests/example2.png")
=======
st.image("assests/example2.png")
>>>>>>> a719d8943fc88f7fff403305d6abff9436ad292c
