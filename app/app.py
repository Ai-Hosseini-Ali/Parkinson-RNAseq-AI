import streamlit as st
import pandas as pd
import sys
import os


# اضافه کردن مسیر پروژه
BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

sys.path.append(BASE_DIR)


from preprocessing.predictor import predict_patient



st.set_page_config(
    page_title="Parkinson AI",
    page_icon="🧠"
)


st.title("🧠 Parkinson AI Predictor")

st.write(
    "Upload RNA expression CSV file"
)



uploaded_file = st.file_uploader(
    "Choose CSV file",
    type=["csv"]
)



if uploaded_file is not None:


    df = pd.read_csv(
        uploaded_file
    )


    st.subheader(
        "Uploaded Data"
    )

    st.write(
        df.head()
    )


    if st.button(
        "Analyze"
    ):


        try:

            result = predict_patient(
                df
            )


            st.success(
                "Analysis Completed"
            )


            st.metric(
                "Parkinson Risk (%)",
                result["Risk"]
            )


            st.write(
                "Prediction:",
                result["Prediction"]
            )


            st.write(
                "Genes detected:",
                result["Genes found"]
            )


        except Exception as e:

            st.error(
                str(e)
            )