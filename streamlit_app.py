import json
import numpy as np
import pandas as pd
import streamlit as st
from steps.prediction_service_loader import mlflow_prediction_service_loader
from run_pipeline import deploy


def main():
    st.title("End to End Stroke Prediction Pipeline with ZenML")
    st.markdown(
        """ 
    #### Problem Statement 
     The objective here is to predict wether the person will have a stroke or not    """
    )
    age = st.sidebar.slider("Select Age", 0, 100)

    selected_hypertension = st.sidebar.radio("Hypertension", ["No", "Yes"])
    hypertension = 1 if selected_hypertension == 'Yes' else 0


    selected_heart_disease = st.sidebar.radio("Heart Disease", ["No", "Yes"])
    heart_disease = 1 if selected_heart_disease == 'Yes' else 0


    glucose_level = st.number_input("Glucose Level", step=0.1, format="%.2f")

    bmi = st.number_input("BMI", step=0.1, format="%.2f")

    selected_gender = st.sidebar.radio("Select Gender", ['Female', 'Male', 'Other'])
    gender_Male = 1 if selected_gender == 'Male' else 0
    gender_Female = 1 if selected_gender == 'Female' else 0
    gender_Other = 1 if selected_gender == 'Other' else 0

    selected_married = st.sidebar.radio("married?", ['No', 'Yes'])
    married_No = 1 if selected_married == 'No' else 0
    married_Yes = 1 if selected_married == 'Yes' else 0


    selected_work_type = st.sidebar.radio("Select Work Type", ['Govt_job', 'Never_worked', 'Private', 'Self-employed', 'Children'])
    work_type_Govt_job = 1 if selected_work_type == 'Govt_job' else 0
    work_type_Never_worked = 1 if selected_work_type == 'Never_worked' else 0
    work_type_Private = 1 if selected_work_type == 'Private' else 0
    work_type_Self_employed = 1 if selected_work_type == 'Self-employed' else 0
    work_type_children = 1 if selected_work_type == 'Children' else 0
    
    selected_residence = st.sidebar.radio("Select Residence Type", ['Rural', 'Urban'])
    residence_Rural = 1 if selected_residence == 'Rural' else 0
    residence_Urban = 1 if selected_residence == 'Urban' else 0
    
    selected_smoking = st.sidebar.radio("Select Smoking Status", ['Unknown', 'formerly smoked', 'never smoked', 'smokes'])
    smoking_Unknown = 1 if selected_smoking == 'Unknown' else 0
    smoking_formerly_smoked = 1 if selected_smoking == 'formerly smoked' else 0
    smoking_never_smoked = 1 if selected_smoking == 'never smoked' else 0
    smoking_smokes = 1 if selected_smoking == 'smokes' else 0


    df = pd.DataFrame(
        {
            "age": [age],
            "hypertension": [hypertension],
            "heart_disease": [heart_disease],
            "glucose_level": [glucose_level],
            "bmi": [bmi],
            "gender_Female": [gender_Female],
            "gender_Male": [gender_Male],
            "gender_Other": [gender_Other],
            "married_No": [married_No],
            "married_Yes": [married_Yes],
            "work_type_Govt_job": [work_type_Govt_job],
            "work_type_Never_worked": [work_type_Never_worked],
            "work_type_Private": [work_type_Private],
            "work_type_Self_employed": [work_type_Self_employed],
            "work_type_children": [work_type_children],
            "residence_Rural": [residence_Rural],
            "residence_Urban": [residence_Urban],
            "smoking_Unknown": [smoking_Unknown],
            "smoking_formerly_smoked": [smoking_formerly_smoked],
            "smoking_never_smoked": [smoking_never_smoked],
            "smoking_smokes": [smoking_smokes],
        }
    )
    df = df.round(1)
    # print(df.dtypes)
    # df = df.astype(float)
    # print(df.head())

    if st.button("Predict"):
        service = mlflow_prediction_service_loader(
            pipeline_name="continuous_deployment_pipeline",
            step_name="mlflow_model_deployer_step",
            running=False,
        )
        if service is None:
            st.write(
                "No service could be found. The pipeline will be run first to create a service."
            )
            deploy()
        json_list = json.loads(json.dumps(list(df.T.to_dict().values())))
        data = np.array(json_list)
        pred = service.predict(data)
        if pred == 1 :
            st.success("The person might be suffer a stroke")
        if pred == 0 :
            st.success("The person is healthy and won't suffer from stroke")

if __name__ == "__main__":
    main()
