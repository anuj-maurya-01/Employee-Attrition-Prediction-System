# Employee Attrition Prediction System

This repository implements an end-to-end Machine Learning pipeline to predict whether an employee is likely to leave an organization. The project follows the complete machine learning lifecycle: data collection, preprocessing, exploratory data analysis (EDA), data balancing, model training, evaluation, and interactive dashboard deployment.

---

## 📁 Repository Structure

```
employee_attrition_prediction_system/
│
├── Dataset/
│   └── WA_Fn-UseC_-HR-Employee-Attrition.csv  # IBM HR Analytics Attrition dataset
│
├── Notebook/
│   └── Employee_Attrition_Analysis.ipynb      # Complete analysis & training notebook
│
├── Model/
│   ├── logistic_regression.joblib             # Serialized Logistic Regression model
│   ├── linear_regression.joblib               # Serialized Linear Regression model
│   ├── random_forest.joblib                   # Serialized Random Forest model
│   ├── xgboost.joblib                         # Serialized XGBoost model
│   ├── preprocessor.joblib                    # ColumnTransformer pipeline
│   ├── feature_names.joblib                   # Extracted post-encoding feature names
│   ├── original_cols.joblib                   # List of original input columns
│   └── metrics_comparison.csv                 # Evaluation metrics on test set
│
├── Streamlit_App/
│   ├── app.py                                 # Streamlit dashboard script
│   ├── utils.py                               # Prediction & risk analysis helpers
│   ├── requirements.txt                       # Project python dependencies
│   └── assets/                                # Evaluated performance plots & heatmaps
│
├── Documentation/
│   └── Project_Report.md                      # Detailed 15-25 page project report
│
├── download_dataset.py                        # Script to download dataset
├── train_models.py                            # Script to preprocess and train models
└── generate_notebook.py                       # Script to compile the Jupyter notebook
```

---

## 🚀 Getting Started

### 1. Prerequisites
Ensure you have **Python 3.8+** installed on your system.

### 2. Environment Setup & Installation
Clone or navigate to the project directory, then install the dependencies listed in `Streamlit_App/requirements.txt`:

```bash
pip install -r Streamlit_App/requirements.txt
```

### 3. Step-by-Step Execution

#### Step A: Download the Dataset
Run the download script to fetch the IBM HR Attrition dataset from the raw public repository:
```bash
python download_dataset.py
```
This downloads and saves the CSV under the `Dataset/` folder.

#### Step B: Train Models & Generate Assets
Run the model building script to run the preprocessing pipeline, train the classifiers, and save serialized assets:
```bash
python train_models.py
```
This saves:
- Models & preprocessors in `Model/`
- Confusion matrices and ROC curves under `Streamlit_App/assets/`

#### Step C: Generate Jupyter Notebook
Run the notebook generation script to build the `.ipynb` file:
```bash
python generate_notebook.py
```
This compiles `Notebook/Employee_Attrition_Analysis.ipynb`.

#### Step D: Launch the Streamlit Dashboard
Launch the web application:
```bash
streamlit run Streamlit_App/app.py
```
Open the local browser link shown in the terminal (usually `http://localhost:8501`) to interact with the application.

---

## 📊 Key Machine Learning Concepts Implemented

1. **Classification**: Predictive modeling of employee attrition status using Logistic Regression, Linear Regression, Random Forest, and XGBoost.
2. **Feature Selection & Engineering**: Dropped zero-variance columns (`EmployeeCount`, `StandardHours`, `Over18`) and key identifier `EmployeeNumber`. Applied IQR capping for numeric outliers and one-hot encoding for categorical variables.
3. **Data Balancing**: Employed **SMOTE** (Synthetic Minority Over-sampling Technique) to address class imbalance (16.1% attrition) on the training split, preventing classifier bias toward staying employees.
4. **Model Evaluation**: Assessed model performance using Accuracy, Precision, Recall, F1-Score, and ROC-AUC. High recall is highlighted as crucial in HR contexts to minimize costly False Negatives.

---

## 🎓 Academic Integrity & Student Code Walkthrough

To comply with the **Academic Integrity Policy** (ensuring students understand the architecture of the system and prevent plain copy-pasting), this guide provides a structural walkthrough of the codebase flow and implementation details:

### 1. Data Ingestion (`download_dataset.py`)
* **Libraries Used**: `urllib.request` (standard library, no external dependency) and `os`.
* **Code Flow**:
  1. `urllib.request.Request` initializes an HTTP request containing a custom `User-Agent` header to prevent HTTP 403 Forbidden responses.
  2. `urlopen` reads the dataset CSV file stream directly from a verified public GitHub repository.
  3. `open(destination, 'wb')` writes the file to the local disk in a binary stream, automatically creating directories if they do not exist.

### 2. Preprocessing & Modeling Pipeline (`train_models.py`)
* **Outlier Treatment**: Implemented via IQR capping. The script calculates $Q1$ (25th percentile) and $Q3$ (75th percentile) for numerical variables. It then calculates the Interquartile Range ($IQR = Q3 - Q1$). Any value exceeding the boundaries $[Q1 - 1.5 \times IQR, Q3 + 1.5 \times IQR]$ is capped using `pandas.DataFrame.clip()`.
* **Preprocessing ColumnTransformer**:
  - Numerical features are transformed using `StandardScaler` to perform Z-score normalization ($z = \frac{x - \mu}{\sigma}$), centering values around a mean of 0 with standard deviation of 1.
  - Categorical features are encoded using `OneHotEncoder(drop='first')`. Dropping the first category for each categorical feature prevents multicollinearity (known as the dummy variable trap).
* **Data Balancing (SMOTE)**: Implemented using `imblearn.over_sampling.SMOTE` on the processed training set only. It calculates the $k$-nearest neighbors for minority class samples in the feature space and creates synthetic observations along the connecting vectors to balance class distribution from 16.1% to a 50/50 split.
* **Linear Regression Classification Baseline**: Since attrition prediction is a binary classification task, Linear Regression is trained as a regressor on the $[0, 1]$ target. In inference, the continuous predictions are clipped to $[0, 1]$ to represent pseudo-probabilities, and then thresholded at $0.5$ (values $\ge 0.5$ predict Attrition (1), and values $< 0.5$ predict Retention (0)).
* **Model Serialization**: In order to save the model states for deployment, the script uses `joblib` to serialize the preprocessor object (`preprocessor.joblib`), original feature lists, and the trained models to the `Model/` directory.

### 3. Web Dashboard Dashboard (`Streamlit_App/app.py` & `utils.py`)
* **Dynamic Retheming Engine**: Uses CSS variables (`:root`) injected using `st.markdown(..., unsafe_allow_html=True)`. When the user selects "Light Mode" or "Dark Mode" in the sidebar, Python swaps the active CSS variable blocks, transforming background colors, card structures, borders, and text colors dynamically.
* **Simulation Process**:
  1. Gathers all front-end inputs (from sliders, selectboxes, and number entries) and compiles them into a single-row Pandas DataFrame.
  2. Orders the input DataFrame columns to match the training dataset using the serialized `original_cols.joblib` list.
  3. Passes the DataFrame through the serialized pipeline using `preprocessor.transform()`.
  4. Runs `model.predict()` and `model.predict_proba()` to determine risk status and risk probability.
  5. Dynamically renders a styled warning card (linear red gradient) for Attrition Risk or success card (green gradient) along with primary risk indicators.

