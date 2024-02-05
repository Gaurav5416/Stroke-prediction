
# 🩺 Stroke Prediction System 
Predict whether a person is prone to a heart stroke based on Age, BMI(Body Mass Index), Glucose level etc . . .

## Table of Contents
- [Introduction](#Introduction)
- [Project Overview](#project-overview)
- [Prerequisites](#prerequisites)
- [Running the project](#Running_the_project)
- [ZenML](#Zenml)
- [MLFlow Experiment Tracking](#Experiment)
- [Streamlit App](#Streamlit_app)

## Introduction 🚩

This project aims to predict wether a person is prone to a heart stroke or not using machine learning. 

Heart strokes have become common concern for people. This ML model warns a person about their potential risk of stroke based on various health indicators, allowing for early intervention and prevention measures to be implemented through the utilization of machine learning algorithm.

## Project Overview 👁️

Our project consists of several components:

### Data Collection 📁
We used data of patients who suffered stroke, the data contain various useful information such as BMI(Body Mass Index), Glucose level, Hypertension, heart disease of patients which might be indicators for the health.

### Data Handling 📊
We used ORM (Object relation Mapping) to handle the data. We shifted the data to a Postgresql server and fetched it from the database using ORM. 

### Exploratory Data Analysis 🔍
In EDA, we perfomed initial exploration of the data, familiarity with the features, Inspected the nature of the data through visualizations and various other statistical techniques. please visit EDA.ipynb in notebooks folder.

### Data Preprocessing 🧹
After initial exploration and identifying features that needed to be dealt with. We moved towards data preprocessing where transformed the features into a format suitable for machine learning. We also handle missing data and encode categorical variables.

### Model Building and Evaluation 🏗️
We used decision tree as the ML model and evaluated the performance using metrics like R2 score (Cofficient of determination) and Mean Squared Error (MSE). The model will only be deployed if it satisfies the minimum accuracy.

### Deployment 🚀
We create a user-friendly web application using Streamlit. Users can input patient details, and the model predicts wether the person has a risk of heart stroke or not. This application might help for pre-screening,enabling early identification of potential stroke risks and facilitating timely medical intervention. 

***

## Prerequisites 📋
Before starting, ensure you have the following prerequisites:
- **Python 3.10**
- **Wsl terminal**
- **Postgresql server**
- **pgadmin4**
---

## Running the project 🏃‍♂️

Please follow following steps to run the project on your machine locally . . . 

1. Clone this repository to your local machine.

    ```
    git clone https://github.com/Gaurav5416/Stroke-prediction
    cd Stroke-prediction
    ```

2. Install necessary requirements.
    ```
    pip install -r requirements.txt
    ```
3. Wake up zenml
    ```
    zenml init
    zenml up
    ```
4. Installing zenml integrations
    ```
    zenml integration install mlflow -y
    ```

5. Registering necessary stack components
    ```
    zenml experiment-tracker register mlflow_tracker --flavor=mlflow
    zenml model-deployer register mlflow_deployer --flavor=mlflow
    zenml stack register local_mlflow_stack -a default -o default -d mlflow_deployer -e mlflow_tracker
    zenml stack set local_mlflow_stack
    ```

6. Create a .env file specifying datapath of table and database path eg..
    ```
    DB_URL = "postgresql://postgres:1234@localhost:5432/stroke"
    datapath = "/mnt/c/name/stoke_system/data/stroke_capped.csv"
    ```

7. Create SQL table and push CSV data into postgreSQL database using ORM
    ```
    cd data/management
    python index.py
    python fill_table.py
    ```

8. Deploying the ML Model
    ```
    python run_pipeline.py
    ```

9. Running Streamlit app
    ```
    streamlit run streamlit_app.py
    ```

***


## ZenML
Zenml provides easy to use MLOPS framework to manage your End to End ML model. It provides various components such as **Experiment trackers** and **Model deployers** which we used in this project. 

- Zenml offers user friendly dashboard locally to manage pipelines and components.

![Zenml Dashboard](/assets/pipelines.png)

---
## MLFlow Experiment Tracking 🧪
We used Mlflow experiment tracker to monitor and record key metrics, parameters, and results throughout the machine learning pipeline. Mlflow provides a user friendly Dashboard for managing experiments.

![MLFLow](/assets/mlflow.png)
---
## Streamlit App ✨
Here is the user friendly web application developed using Streamlit. It offers an intuitive interface for users to interact with the machine learning model, allowing them to input patient details and receive predictions regarding their risk of stroke.

![Streamlit app](/assets/streamlit_app.png)


## Contributing 🤝
We welcome contributions from the community! Whether it's fixing a bug, adding a feature, or improving documentation, your input is valuable. Fork the project, submit pull requests, and collaborate with us to make this project even better


## License 📜
This project is open-source and is licensed under the [MIT License](LICENSE). You can find the full license in the [LICENSE](LICENSE) file. 

