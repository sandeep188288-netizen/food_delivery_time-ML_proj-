
# Food Delivery Time Prediction

A Machine Learning project that predicts the estimated time required to deliver a food order based on delivery distance, traffic conditions, weather, order details, vehicle information and delivery-person attributes.

The project covers the complete ML workflow, starting from data analysis and preprocessing to model training, hyperparameter tuning and deployment through a FastAPI web application.

## Project Overview

Food delivery time depends on several factors, including distance between the restaurant and customer, traffic conditions, weather, vehicle condition and delivery-person details.

The objective of this project is to build a regression model that can estimate delivery time using these factors.

After experimenting with multiple regression algorithms, a tuned **XGBoost Regressor** was selected for the prediction system.

The trained model is integrated with a FastAPI backend and an interactive frontend where users can enter delivery details and get an estimated delivery time in minutes.

## Key Features

- Exploratory Data Analysis (EDA)
- Data cleaning and preprocessing
- Feature engineering
- Training and comparison of regression models
- Hyperparameter tuning using RandomizedSearchCV
- Tuned XGBoost regression model
- Model and preprocessing artifact serialization using Joblib
- FastAPI-based prediction API
- Interactive HTML, CSS and JavaScript frontend
- Distance calculation using the Haversine formula
- Docker support for containerization

## Machine Learning Workflow

The project follows the workflow below:

1. Dataset loading and inspection
2. Data cleaning and handling missing values
3. Exploratory Data Analysis
4. Feature engineering
5. Categorical feature encoding
6. Feature scaling
7. Train-test split
8. Regression model training
9. Model evaluation and comparison
10. XGBoost hyperparameter tuning
11. Saving the trained model and preprocessing artifacts
12. Integrating the model with FastAPI

## Dataset Information

**Dataset:** Food Delivery Dataset

**Target Variable:** `Time_taken(min)`

The target variable represents the time taken to deliver an order, measured in minutes.

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
| Type_of_order | Type of food order |
| Type_of_vehicle | Vehicle used for delivery |
| Festival | Festival-related information |
| City | City category |
| Weatherconditions | Weather during delivery |

## Exploratory Data Analysis

EDA was performed to understand the dataset and identify patterns that may affect delivery time.

The analysis includes:

- Distribution of delivery time
- Distribution of numerical features
- Relationship between delivery distance and delivery time
- Impact of traffic conditions on delivery time
- Impact of weather conditions
- Vehicle condition analysis
- Delivery-person ratings analysis
- Correlation heatmap
- Analysis of categorical features

## Feature Engineering and Preprocessing

Several preprocessing steps were performed before training the machine learning models.

- Extracted the order hour from order time.
- Calculated delivery distance using restaurant and delivery coordinates.
- Encoded categorical features.
- Prepared numerical and categorical features for model training.
- Applied feature scaling using StandardScaler.
- Maintained the feature-column order required by the trained model.

The same preprocessing logic must be maintained during prediction to ensure consistency between training and inference.

## Model Training and Evaluation

Multiple regression algorithms were explored and compared to identify a suitable model for predicting delivery time.

The models include:

- Linear Regression
- Decision Tree Regressor
- Random Forest Regressor
- XGBoost Regressor

The models were evaluated using regression evaluation metrics such as:

- R² Score
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)

### XGBoost Hyperparameter Tuning

XGBoost was further optimized using `RandomizedSearchCV`.

The search configuration included:

- `n_iter = 40`
- `cv = 3`
- `scoring = 'r2'`
- `random_state = 42`
- `n_jobs = -1`

The hyperparameter search space included:

| Parameter | Search Values |
|---|---|
| n_estimators | 300, 500, 800 |
| max_depth | 3, 4, 5, 6, 8 |
| learning_rate | 0.01, 0.03, 0.05, 0.1 |
| subsample | 0.7, 0.8, 0.9, 1.0 |
| colsample_bytree | 0.7, 0.8, 0.9, 1.0 |
| reg_alpha | 0, 0.1, 0.5, 1 |
| reg_lambda | 1, 1.5, 2 |

The best estimator obtained from the search was saved as the final prediction model.

## Application Architecture

The trained model is integrated into a web application using FastAPI.

The application works as follows:

1. The user enters delivery details through the frontend.
2. Restaurant and delivery-location coordinates are used to calculate distance.
3. Order time is converted into an hour value.
4. Input data is prepared according to the model's expected features.
5. The saved scaler and model are loaded by the backend.
6. The model generates a delivery-time prediction.
7. The predicted delivery time is returned to the frontend in minutes.

## Tech Stack

| Category | Technology |
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
| Development Environment | Jupyter Notebook, VS Code |

## Project Structure

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
│
└── README.md
```

## Installation and Setup

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

### 4. Run the FastAPI Application

```bash
python -m uvicorn app:app --reload
```

If you need the optional CORS wrapper:

```bash
python -m uvicorn run_cors:app --reload
```

### 5. Open the Application

Frontend:

http://127.0.0.1:8000/

FastAPI Swagger Documentation:

http://127.0.0.1:8000/docs

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | Serves the frontend application |
| POST | `/predict` | Predicts estimated food delivery time |
| GET | `/docs` | Interactive FastAPI documentation |

The `/predict` endpoint accepts delivery-related input and returns the estimated delivery time in minutes.

## Docker Deployment

The project also includes a Dockerfile to run the FastAPI application inside a Docker container.

### Build Docker Image

```bash
docker build -t food-delivery-api .
```

### Run Docker Container

```bash
docker run -p 8000:8000 food-delivery-api
```

Open the application:

http://127.0.0.1:8000/

## Saved Model Files

The following files are required by the backend:

| File | Purpose |
|---|---|
| xgb_best.pkl | Trained and tuned XGBoost model |
| standard_scaler.pkl | Saved StandardScaler |
| model_columns.pkl | Expected feature-column order |

These files are loaded when the FastAPI application starts.

## Future Improvements

- Improve prediction accuracy through additional feature engineering.
- Experiment with additional regression algorithms.
- Add more detailed model evaluation and explainability.
- Deploy the application on a cloud platform.
- Improve frontend responsiveness and user experience.
- Add prediction history and monitoring.

