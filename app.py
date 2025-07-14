# app.py
import streamlit as st
import numpy as np
import cv2
from region_utils import get_face_regions
from analysis_utils import (
    score_redness, score_shine, score_texture, score_pores, score_dark_circle
)

st.title("🧴 MVP Skin Analyzer")

uploaded_file = st.file_uploader("Upload a clear face photo", type=["jpg", "jpeg", "png"])
if uploaded_file:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)

    st.image(image, caption="Uploaded Image", channels="BGR")

    with st.spinner("Analyzing face..."):
        regions = get_face_regions(image)

    if not regions:
        st.error("No face detected. Try another photo.")
    else:
        st.subheader("🔍 Skin Trait Scores")

        redness = score_redness(regions['cheek'] if 'cheek' in regions else regions['left_cheek'])
        shine = score_shine(regions['forehead'])
        # texture = score_texture(regions['left_cheek'])
        # pores = score_pores(regions['right_cheek'])
        dark_circle = score_dark_circle(regions['under_eyes'])

        st.write(f" **Redness:** {redness}/10")
        st.write(f" **Shine:** {shine}/10")
        # st.write(f"🌾 **Texture:** {texture}/10")
        # st.write(f" **Pores:** {pores}/10")
        st.write(f" **Dark Circles:** {dark_circle}/10")
                
                # Texture
        left_texture = score_texture(regions['left_cheek'])
        right_texture = score_texture(regions['right_cheek'])

        # Pores
        left_pores = score_pores(regions['left_cheek'])
        right_pores = score_pores(regions['right_cheek'])

        # Display side-by-side
        col1, col2 = st.columns(2)
        with col1:
            st.subheader(" Left Cheek")
            st.write(f" **Texture:** {left_texture}/10")
            st.write(f" **Pores:** {left_pores}/10")
        with col2:
            st.subheader(" Right Cheek")
            st.write(f" **Texture:** {right_texture}/10")
            st.write(f" **Pores:** {right_pores}/10")

