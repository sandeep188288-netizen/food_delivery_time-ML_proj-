
# 🍔 Food Delivery Time Prediction

### Machine Learning | XGBoost | FastAPI | Docker | Render

A machine learning-based web application that predicts the estimated delivery time of food orders using delivery-person details, order information, traffic conditions, weather, vehicle condition and delivery distance.

The project takes a complete machine learning workflow from data analysis and preprocessing to model training, hyperparameter tuning and deployment. The final tuned XGBoost model is integrated with a FastAPI backend and an interactive web interface.

🚀 **Live Demo:** https://food-delivery-time-ml-proj.onrender.com/

---

## 🌐 Live Application

The application is deployed on Render and can be accessed here:

**[Open QuickBite AI – Food Delivery Time Prediction](https://food-delivery-time-ml-proj.onrender.com/)**

Users can enter delivery-related information and get an estimated delivery time in minutes.

---

## 📌 Project Overview

Food delivery time depends on several factors, such as the distance between the restaurant and customer, traffic conditions, weather, vehicle condition and delivery-person ratings.

The objective of this project is to develop a regression model that estimates food delivery time using these factors.

Different regression algorithms were trained and evaluated. After comparing their performance, a tuned **XGBoost Regressor** was selected as the final model.

The trained model is deployed through a FastAPI application with an interactive frontend.

## ✨ Key Features

- Exploratory Data Analysis (EDA)
- Data cleaning and preprocessing
- Feature engineering using geographical coordinates
- Training and comparison of multiple regression models
- Hyperparameter tuning using RandomizedSearchCV
- Optimized XGBoost regression model
- Interactive web-based prediction interface
- FastAPI backend for model inference
- Distance calculation using the Haversine formula
- Docker support
- Live deployment on Render

---

## 📊 Dataset Information

The project uses a food delivery dataset containing order details, delivery-person information, geographical coordinates and environmental conditions.

| Dataset Information | Details |
|---|---|
| Original Records | 45,593 |
| Records After Cleaning | 45,162 |
| Target Variable | `Time_taken(min)` |
| Problem Type | Regression |
| Removed Records | 431 |

Records with impossible delivery distances (greater than 100 km) were removed during data cleaning.

### Target Variable

`Time_taken(min)` represents the time required to deliver an order, measured in minutes.

### Features Used

| Feature | Description |
|---|---|
| Delivery_person_Age | Age of the delivery person |
| Delivery_person_Ratings | Rating of the delivery person |
| distance_km | Distance between restaurant and delivery location |
| order_hour | Hour at which the order was placed |
| Road_traffic_density | Traffic condition during delivery |
| Vehicle_condition | Condition of the delivery vehicle |
| multiple_deliveries | Number of deliveries handled together |
| Type_of_order | Category of the food order |
| Type_of_vehicle | Vehicle used for delivery |
| Festival | Festival-related information |
| City | City category |
| Weatherconditions | Weather conditions |

---

## 🔍 Data Cleaning and Preprocessing

The following preprocessing steps were performed:

- Removed records with impossible geographical distances.
- Cleaned inconsistent text values in categorical columns.
- Handled missing values using median and mode imputation.
- Capped delivery-person ratings at the maximum rating of 5.
- Converted categorical variables into numerical representations.
- Applied feature scaling using StandardScaler.

## ⚙️ Feature Engineering

Feature engineering was an important part of this project.

### 1. Distance Calculation

Restaurant and delivery-location coordinates were used to calculate the geographical distance in kilometres using the Haversine formula.

### 2. Order Hour Extraction

The order time was converted into an hour feature (`order_hour`) to capture time-related delivery patterns.

### 3. Traffic Encoding

Road traffic density was encoded as follows:

| Traffic Condition | Encoded Value |
|---|---:|
| Low | 0 |
| Medium | 1 |
| High | 2 |
| Jam | 3 |

### 4. Categorical Encoding

Categorical features such as order type, vehicle type, city, festival and weather conditions were converted using one-hot encoding.

### 5. Feature Scaling

StandardScaler was used to scale the model input features.

---

## 🤖 Model Training and Comparison

Multiple regression algorithms were trained and evaluated to identify the most suitable model.

The following models were compared:

- Linear Regression
- Decision Tree Regressor
- Random Forest Regressor
- XGBoost Regressor

### Model Performance

| Model | Test R² Score | Test RMSE | Train R² Score |
|---|---:|---:|---:|
| Linear Regression | 0.57 | 6.21 min | 0.58 |
| Decision Tree | 0.67 | 5.47 min | 1.00 |
| Random Forest | 0.82 | 4.04 min | 0.97 |
| **Tuned XGBoost** | **0.83** | **3.92 min** | **0.85** |

### Final Model: XGBoost Regressor

The tuned XGBoost model achieved the best test performance among the evaluated models.

- **Test R² Score:** 0.83
- **Test RMSE:** 3.92 minutes
- **Train R² Score:** 0.85

The difference between training and testing performance is smaller than that of the Decision Tree and Random Forest models, indicating better generalisation on the test data.

### Hyperparameter Tuning

RandomizedSearchCV was used to find suitable XGBoost hyperparameters.

Search configuration:

```python
RandomizedSearchCV(
    estimator=xgb,
    param_distributions=param_grid,
    n_iter=40,
    cv=3,
    scoring='r2',
    random_state=42,
    n_jobs=-1
)
```

Best model configuration:

| Parameter | Selected Value |
|---|---:|
| n_estimators | 500 |
| max_depth | 8 |
| learning_rate | 0.01 |
| subsample | 0.9 |

---

## 🏗️ Application Architecture

The application connects the trained machine learning model with a web interface using FastAPI.

```text
             User
               |
               v
       HTML/CSS/JavaScript
               |
               v
          FastAPI API
               |
               v
       Input Preprocessing
               |
               v
       Distance Calculation
        (Haversine Formula)
               |
               v
       Feature Preparation
               |
               v
         StandardScaler
               |
               v
        Tuned XGBoost
               |
               v
       Delivery Time Prediction
               |
               v
         Result in Minutes
```

---

## 🛠️ Tech Stack

| Category | Technologies |
|---|---|
| Programming Language | Python |
| Data Analysis | Pandas, NumPy |
| Data Visualization | Matplotlib, Seaborn |
| Machine Learning | Scikit-learn |
| Boosting Algorithm | XGBoost |
| Hyperparameter Tuning | RandomizedSearchCV |
| Model Serialization | Joblib |
| Backend | FastAPI |
| API Server | Uvicorn |
| Frontend | HTML, CSS, JavaScript |
| Containerization | Docker |
| Deployment | Render |
| Development Environment | Jupyter Notebook, VS Code |

---

## 📁 Project Structure

```text
food_delivery_time-ML_proj-/
│
├── Food_Delivery_Time_Prediction.ipynb
├── Food delivery.csv
│
├── app.py
├── run_cors.py
├── index.html
│
├── xgb_best.pkl
├── standard_scaler.pkl
├── model_columns.pkl
│
├── requirements.txt
├── Dockerfile
├── .gitignore
└── README.md
```

---

## 🚀 Run the Project Locally

### 1. Clone the Repository

```bash
git clone https://github.com/sandeep188288-netizen/food_delivery_time-ML_proj-.git

cd food_delivery_time-ML_proj-
```

### 2. Create a Virtual Environment

For macOS/Linux:

```bash
python3 -m venv .venv

source .venv/bin/activate
```

For Windows:

```bash
python -m venv .venv

.venv\Scripts\activate
```

### 3. Install Dependencies

```bash
python -m pip install --upgrade pip

pip install -r requirements.txt
```

### 4. Start FastAPI

```bash
python -m uvicorn app:app --reload
```

If you want to run the optional CORS wrapper:

```bash
python -m uvicorn run_cors:app --reload
```

### 5. Open the Application

Frontend:

http://127.0.0.1:8000/

API Documentation:

http://127.0.0.1:8000/docs

---

## 🐳 Run with Docker

Docker is included to make it easier to run the application in a containerized environment.

### Build Docker Image

```bash
docker build -t food-delivery-api .
```

### Run Docker Container

```bash
docker run -p 8000:8000 food-delivery-api
```

Open:

http://127.0.0.1:8000/

---

## 📦 Saved Model Artifacts

The following files are used by the FastAPI backend:

| File | Purpose |
|---|---|
| `xgb_best.pkl` | Trained and tuned XGBoost model |
| `standard_scaler.pkl` | Saved StandardScaler |
| `model_columns.pkl` | Expected feature-column order |

These artifacts are loaded by the backend to preprocess input data and generate predictions.

---

## 🔌 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | Serves the frontend application |
| POST | `/predict` | Predicts estimated food delivery time |
| GET | `/docs` | Interactive API documentation |

The prediction endpoint accepts delivery details and returns the estimated delivery time in minutes.

---

## 🔮 Future Improvements

- Experiment with additional feature engineering techniques.
- Improve prediction performance with more delivery data.
- Add model explainability using SHAP.
- Introduce prediction history and analytics.
- Improve frontend visualisation and user experience.
- Add automated model monitoring.

---

## 👨‍💻 Author

**Sandeep Kushwaha**

B.Tech – Artificial Intelligence and Machine Learning  
Lakshmi Narain College of Technology, Bhopal

- GitHub: [sandeep188288-netizen](https://github.com/sandeep188288-netizen)
- LinkedIn: [Sandeep Kushwaha](https://www.linkedin.com/in/sandeep-kushwaha-266447379/)

---

⭐ If you find this project interesting, feel free to explore the repository and try the live application.

**Live Demo:** https://food-delivery-time-ml-proj.onrender.com/
