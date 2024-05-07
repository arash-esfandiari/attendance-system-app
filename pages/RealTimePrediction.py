from Home import face_rec
import streamlit as st
from streamlit_webrtc import webrtc_streamer
import av
import time

# Configuration
PAGE_TITLE = "Real-Time Prediction"
DATA_KEY = "academy:register"
WAIT_TIME = 30  # Time in seconds between logs saving


# st.set_page_config(page_title=PAGE_TITLE)
st.subheader("Real-Time Attendance System")


# Retrieve the data from Redis Database
# Data Retrieval with Error Handling
try:
    with st.spinner("Retrieving Data from RedisDB"):
        redis_face_db = face_rec.retrieve_data(name=DATA_KEY)
        st.dataframe(redis_face_db)
    st.success("Data successfully retrieved from RedisDB")
except Exception as e:
    st.error(f"Failed to retrieve data: {str(e)}")

# time
waiTtime = 30  # time in seconds
setTime = time.time()
realtimepred = face_rec.RealTimePred()

# Real Time Prediction
# streamlit webrtc


# callback function
def video_frame_callback(frame):
    global setTime
    img = frame.to_ndarray(format="bgr24")  # bgr24 is a three dimention numpy array
    # operation that you can perform on the array
    pred_img = realtimepred.face_prediction(
        img, redis_face_db, "facial_features", ["Name", "Role"], thresh=0.5
    )
    if time.time() - setTime >= waiTtime:
        realtimepred.save_logs_redis()
        setTime = time.time()  # reset time
        print("Save Data to redis database")
    return av.VideoFrame.from_ndarray(pred_img, format="bgr24")


webrtc_streamer(
    key="realTimePrediction",
    video_frame_callback=video_frame_callback,
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
)
