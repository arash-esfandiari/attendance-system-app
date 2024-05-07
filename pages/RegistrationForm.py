import streamlit as st
from Home import face_rec
import cv2
import numpy as np
from streamlit_webrtc import webrtc_streamer
import av


# st.set_page_config(page_title="ReportLog")
st.subheader("Registration Form")

# Initialize registration form
registration_form = face_rec.RegistrationForm()


# Step 1: Collect person name and role
# form
full_name = st.text_input(label="Name", placeholder="Full Name")
role = st.selectbox(label="Select Your Role", options=("Student", "Teacher"))


# Step 2: collect facial embedding of the user
def video_callback_func(frame):
    img = frame.to_ndarray(format="bgr24")  # 3d array, b,g,r
    reg_img, embedding = registration_form.get_embeddings(img)

    # two step process to save the data
    # step 1: save the data into local computer int .txt
    if embedding is not None:
        with open("face_embedding.txt", mode="w+") as f:
            np.savetxt(f, embedding)
    # step 2:

    return av.VideoFrame.from_ndarray(reg_img, format="bgr24")


webrtc_streamer(
    key="registration",
    video_frame_callback=video_callback_func,
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
)

# Step 3: save the data in Redis Database

if st.button("Submit"):
    return_val = registration_form.save_data_in_redis_db(full_name, role)
    if return_val == True:
        st.success(f"{full_name} registered successfully")
    elif return_val == "name_false":
        st.error("Name cannot be empty")
    elif return_val == "file_false":
        st.error("Facial data not found, please refresh the page and excecute again")
