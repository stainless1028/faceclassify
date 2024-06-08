from tensorflow import keras
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np
import streamlit as st

# Define and register the custom layer before loading the model
from tensorflow.keras.layers import DepthwiseConv2D


class CustomDepthwiseConv2D(DepthwiseConv2D):
    def __init__(self, **kwargs):
        # Remove 'groups' from kwargs if it exists
        kwargs.pop('groups', None)
        super().__init__(**kwargs)


# Register the custom layer
tf.keras.utils.get_custom_objects().update({'DepthwiseConv2D': CustomDepthwiseConv2D})

with open("style.css", encoding="UTF8") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)  # CSS 적용

st.markdown("<h1>동물상 테스트</h1>", unsafe_allow_html=True)
st.markdown("<h2>인공지능이 여러분의 동물상을 알려줍니다!</h2>", unsafe_allow_html=True)

mode = st.radio("placeholder", ["이미지 업로드", "웹캠"], horizontal=True, label_visibility="hidden")

np.set_printoptions(suppress=True)
model = keras.models.load_model("keras_model.h5", compile=False)
class_names = open("labels.txt", "r").readlines()


if mode == "이미지 업로드":

    uploaded_file = st.file_uploader(label="placeholder", label_visibility="hidden", type=['jpeg', 'png', 'jpg', 'webp'])

    # Create the array of the right shape to feed into the keras model
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

    col1, col2, col3 = st.columns([1, 2, 1])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        with col2:
            st.image(image, use_column_width=True)

        image = image.convert('RGB')
        size = (224, 224)
        image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
        image_array = np.asarray(image)
        normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
        data[0] = normalized_image_array

        # Predict the model
        prediction = model.predict(data)
        result = {
            "강아지상": f"{prediction[0][0] * 100:.0f}%",
            "고양이상": f"{prediction[0][1] * 100:.0f}%",
            "곰상": f"{prediction[0][2] * 100:.0f}%",
            "공룡상": f"{prediction[0][3] * 100:.0f}%",
            "토끼상": f"{prediction[0][4] * 100:.0f}%",
        }
        sorted_result = dict(sorted(result.items(), key=lambda item: int(item[1][:-1]), reverse=True))  # 퍼센트순 정렬
        bar_colors = {"강아지상": "#FFF79D", "고양이상": "#1BAFEA", "곰상": "#C38C66", "공룡상": "#4CAF50", "토끼상": "#FFB6C1"}
        index = np.argmax(prediction)
        class_name = class_names[index].strip()
        confidence_score = prediction[0][index]


        def render_thick_progress_bar(percentage, bar_id, label: str, color):
            progress_bar_html = f"""
            <style>
            .progress-bar-wrapper-{bar_id} {{
                display: flex;
                align-items: center;
                margin-bottom: 10px;
            }}
            .label-container-{bar_id} {{
                flex: 0 0 80px;
                text-align: left;
                font-size: 16px;
                white-space: nowrap;
            }}
            .progress-bar-container-{bar_id} {{
                flex-grow: 1;
                background-color: rgb(240, 242, 246);
                border-radius: 10px;
                position: relative;
            }}
            .progress-bar-{bar_id} {{
                width: {percentage}%;
                height: 2rem;
                background-color: {color};
                border-radius: 10px;
                position: relative;
            }}
            .progress-text-{bar_id} {{
                position: absolute;
                width: 100%;
                top: 0;
                left: 0;
                height: 2rem;
                line-height: 2rem;
                text-align: center;
                color: black;
            }}
            </style>
            <div class="progress-bar-wrapper-{bar_id}">
                <div class="label-container-{bar_id}">{label}</div>
                <div class="progress-bar-container-{bar_id}">
                    <div class="progress-bar-{bar_id}"></div>
                    <div class="progress-text-{bar_id}">{percentage}%</div>
                </div>
            </div>
            """
            st.markdown(progress_bar_html, unsafe_allow_html=True)

        face_types = {"0 dog": "강아지상", "1 cat": "고양이상", "2 bear": "곰상", "3 dinosaur": "공룡상", "4 rabbit": "토끼상"}
        st.markdown("<h1>"f"{face_types[class_name]}""</h1>", unsafe_allow_html=True)

        bar_id = 0
        for x in sorted_result:
            bar_id += 1
            render_thick_progress_bar(percentage=int(sorted_result[x][:-1]), bar_id=bar_id, label=x, color=bar_colors[x])

else:  # 웹캠일때
    st.markdown("<h1>wait.</h1>", unsafe_allow_html=True)
