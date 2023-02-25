import streamlit as st 
import numpy as np 
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

import base64
import json
import os
import re
import time
import uuid
from io import BytesIO
from pathlib import Path

import pandas as pd
from PIL import Image
from streamlit_drawable_canvas import st_canvas
from svgpathtools import parse_path

def main():
    if "button_id" not in st.session_state:
        st.session_state["button_id"] = ""
    if "color_to_label" not in st.session_state:
        st.session_state["color_to_label"] = {}
    PAGES = {
        "Basic Drawing": full_app,
        "Deawing and Download PNG": png_export,
        "streamlit scikitlearn":streamlit_scikitlearn
    }
    page = st.sidebar.selectbox("Page:", options=list(PAGES.keys()))
    PAGES[page]()

def full_app():
    st.sidebar.header("Configuration")
    st.markdown(
        """
    วาดอะไรก็ได้ แล้วดูผลลัพธ์
    * 👈ตั้งค่าได้ที่แถบ
    * ในโหมดรูปหลายเหลี่ยม คลิกซ้ายเพื่อเพิ่มจุด คลิกขวาเพื่อปิดรูปหลายเหลี่ยม ดับเบิลคลิกเพื่อลบจุดล่าสุด
    """
    )

    with st.echo("below"):
        # Specify canvas parameters in application
        drawing_mode = st.sidebar.selectbox(
            "ประเภทของเส้น:",
            ("freedraw", "line", "rect", "circle", "transform", "polygon", "point"),
        )
        stroke_width = st.sidebar.slider("ขนาดของเส้น: ", 1, 25, 3)
        if drawing_mode == 'point':
            point_display_radius = st.sidebar.slider("Point display radius: ", 1, 25, 3)
        stroke_color = st.sidebar.color_picker("เลือกสีของเส้น: ")
        bg_color = st.sidebar.color_picker("เลือกสีของ Background: ", "#eee")
        bg_image = st.sidebar.file_uploader("เลือกรูปเป็น Background:", type=["png", "jpg"])
        realtime_update = st.sidebar.checkbox("Update in realtime", True)

        # Create a canvas component
        canvas_result = st_canvas(
            fill_color="rgba(255, 165, 0, 0.3)",  # Fixed fill color with some opacity
            stroke_width=stroke_width,
            stroke_color=stroke_color,
            background_color=bg_color,
            background_image=Image.open(bg_image) if bg_image else None,
            update_streamlit=realtime_update,
            height=150,
            drawing_mode=drawing_mode,
            point_display_radius=point_display_radius if drawing_mode == 'point' else 0,
            display_toolbar=st.sidebar.checkbox("Display toolbar", True),
            key="full_app",
        )

        # Do something interesting with the image data and paths
        if canvas_result.image_data is not None:
            st.image(canvas_result.image_data)
        if canvas_result.json_data is not None:
            objects = pd.json_normalize(canvas_result.json_data["objects"])
            for col in objects.select_dtypes(include=["object"]).columns:
                objects[col] = objects[col].astype("str")
            st.dataframe(objects)

def png_export():
    st.markdown(
        """
    Realtime update is disabled for this demo. 
    Press the 'Download' button at the bottom of canvas to update exported image.
    """
    )
    try:
        Path("tmp/").mkdir()
    except FileExistsError:
        pass

    # Regular deletion of tmp files
    # Hopefully callback makes this better
    now = time.time()
    N_HOURS_BEFORE_DELETION = 1
    for f in Path("tmp/").glob("*.png"):
        st.write(f, os.stat(f).st_mtime, now)
        if os.stat(f).st_mtime < now - N_HOURS_BEFORE_DELETION * 3600:
            Path.unlink(f)

    if st.session_state["button_id"] == "":
        st.session_state["button_id"] = re.sub(
            "\d+", "", str(uuid.uuid4()).replace("-", "")
        )

    button_id = st.session_state["button_id"]
    file_path = f"tmp/{button_id}.png"


    custom_css = f""" 
        <style>
            #{button_id} {{
                display: inline-flex;
                align-items: center;
                justify-content: center;
                background-color: rgb(255, 255, 255);
                color: rgb(38, 39, 48);
                padding: .25rem .75rem;
                position: relative;
                text-decoration: none;
                border-radius: 4px;
                border-width: 1px;
                border-style: solid;
                border-color: rgb(230, 234, 241);
                border-image: initial;
            }} 
            #{button_id}:hover {{
                border-color: rgb(246, 51, 102);
                color: rgb(246, 51, 102);
            }}
            #{button_id}:active {{
                box-shadow: none;
                background-color: rgb(246, 51, 102);
                color: white;
                }}
        </style> """

    data = st_canvas(update_streamlit=False, key="png_export")
    if data is not None and data.image_data is not None:
        img_data = data.image_data
        im = Image.fromarray(img_data.astype("uint8"), mode="RGBA")
        im.save(file_path, "PNG")

        buffered = BytesIO()
        im.save(buffered, format="PNG")
        img_data = buffered.getvalue()
        try:
            # some strings <-> bytes conversions necessary here
            b64 = base64.b64encode(img_data.encode()).decode()
        except AttributeError:
            b64 = base64.b64encode(img_data).decode()

        dl_link = (
            custom_css
            + f'<a download="{file_path}" id="{button_id}" href="data:file/txt;base64,{b64}">Export PNG</a><br></br>'
        )
        st.markdown(dl_link, unsafe_allow_html=True)

def streamlit_scikitlearn():
    with st.echo("below"):

        dataset_name = st.sidebar.selectbox(
            'เลือก Dataset',
            ('Iris', 'Breast Cancer', 'Wine')
        )

        st.write(f"## {dataset_name} Dataset")

        classifier_name = st.sidebar.selectbox(
            'เลือก classifier',
            ('KNN', 'SVM', 'Random Forest')
        )
        def get_dataset(name):
            data = None
            if name == 'Iris':
                data = datasets.load_iris()
            elif name == 'Wine':
                data = datasets.load_wine()
            else:
                data = datasets.load_breast_cancer()
            X = data.data
            y = data.target
            return X, y

        X, y = get_dataset(dataset_name)
        st.write('สัดส่วนของ dataset:', X.shape)
        st.write('จำนวนของ classes:', len(np.unique(y)))

        def add_parameter_ui(clf_name):
            params = dict()
            if clf_name == 'SVM':
                C = st.sidebar.slider('C', 0.01, 10.0)
                params['C'] = C
            elif clf_name == 'KNN':
                K = st.sidebar.slider('K', 1, 15)
                params['K'] = K
            else:
                max_depth = st.sidebar.slider('max_depth', 2, 15)
                params['max_depth'] = max_depth
                n_estimators = st.sidebar.slider('n_estimators', 1, 100)
                params['n_estimators'] = n_estimators
            return params

        params = add_parameter_ui(classifier_name)

        def get_classifier(clf_name, params):
            clf = None
            if clf_name == 'SVM':
                clf = SVC(C=params['C'])
            elif clf_name == 'KNN':
                clf = KNeighborsClassifier(n_neighbors=params['K'])
            else:
                clf = clf = RandomForestClassifier(n_estimators=params['n_estimators'], 
                    max_depth=params['max_depth'], random_state=1234)
            return clf

        clf = get_classifier(classifier_name, params)
        #### CLASSIFICATION ####

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        acc = accuracy_score(y_test, y_pred)

        st.write(f'ประเภท = {classifier_name}')
        st.write(f'ความแม่นยำ =', acc)

        #### PLOT DATASET ####
        # Project the data onto the 2 primary principal components
        pca = PCA(2)
        X_projected = pca.fit_transform(X)

        x1 = X_projected[:, 0]
        x2 = X_projected[:, 1]

        fig = plt.figure()
        plt.scatter(x1, x2,
                c=y, alpha=0.8,
                cmap='viridis')

        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.colorbar()

        #plt.show()
        st.pyplot(fig)


if __name__ == "__main__":
    st.set_page_config(
        page_title="Streamlit Drawable Canvas Demo", page_icon=":pencil2:"
    )
    st.title("Streamlit Drawing✏️ and Scikitlearn🤖")
    st.sidebar.subheader("Configuration")
    main()