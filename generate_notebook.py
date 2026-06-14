import json
import os

def create_notebook():
    project_dir = os.path.dirname(os.path.abspath(__file__))
    notebook_dir = os.path.join(project_dir, "Notebook")
    os.makedirs(notebook_dir, exist_ok=True)
    
    # Define notebook structure
    notebook = {
        "cells": [],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 2
    }
    
    # Cell helper functions
    def add_markdown(source_list):
        notebook["cells"].append({
            "cell_type": "markdown",
            "metadata": {},
            "source": [line + "\n" for line in source_list]
        })
        
    def add_code(source_list):
        notebook["cells"].append({
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [line + "\n" for line in source_list]
        })

    # Page 1: Introduction
    add_markdown([
        "# Employee Attrition Prediction System",
        "**Author:** Data Science Team  ",
        "**Objective:** Build a predictive machine learning classification model to determine whether an employee is likely to leave an organization (attrition).",
        "",
        "### Project Methodology (Machine Learning Lifecycle):",
        "1. **Problem Understanding**: Identify business goals and study characteristics of the dataset.",
        "2. **Data Collection**: Fetch the IBM HR Analytics dataset (1470 records, 35 attributes).",
        "3. **Data Preprocessing**: Handle duplicates, analyze missing values, treat outliers, encode features, scale numbers.",
        "4. **Exploratory Data Analysis (EDA)**: Conduct univariate, bivariate, and multivariate analysis with visual representations.",
        "5. **Feature Engineering**: Select important features and address class imbalance using SMOTE.",
        "6. **Model Building**: Train Logistic Regression, Random Forest, and XGBoost models.",
        "7. **Model Evaluation**: Compare models using Accuracy, Precision, Recall, F1-Score, and ROC-AUC metrics.",
        "8. **Deployment**: Package models and build a Streamlit web application."
    ])

    # Cell 2: Imports
    add_markdown(["## Setup & Package Ingestion", "First, we import the core data science, plotting, and machine learning modules."])
    add_code([
        "import os",
        "import numpy as np",
        "import pandas as pd",
        "import matplotlib.pyplot as plt",
        "import seaborn as sns",
        "from sklearn.model_selection import train_test_split",
        "from sklearn.preprocessing import StandardScaler, OneHotEncoder",
        "from sklearn.compose import ColumnTransformer",
        "from sklearn.linear_model import LogisticRegression",
        "from sklearn.ensemble import RandomForestClassifier",
        "from xgboost import XGBClassifier",
        "from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report, roc_curve",
        "from imblearn.over_sampling import SMOTE",
        "import warnings",
        "warnings.filterwarnings('ignore')",
        "sns.set_theme(style='whitegrid', palette='muted')",
        "%matplotlib inline"
    ])

    # Cell 3: Data loading
    add_markdown(["## Data Collection", "Let's load the dataset and view the basic structure of the data."])
    add_code([
        "# Load the downloaded dataset",
        "dataset_path = '../Dataset/WA_Fn-UseC_-HR-Employee-Attrition.csv'",
        "if not os.path.exists(dataset_path):",
        "    dataset_path = 'Dataset/WA_Fn-UseC_-HR-Employee-Attrition.csv' # Fallback for relative paths",
        "",
        "df = pd.read_csv(dataset_path)",
        "print(f'Dataset Dimensions: {df.shape[0]} rows, {df.shape[1]} columns')",
        "df.head()"
    ])

    add_code([
        "# Check basic info and data types",
        "df.info()"
    ])

    # Cell 4: Preprocessing
    add_markdown([
        "## Data Preprocessing",
        "We clean the data by checking for duplicates, missing values, dropping non-informative columns, and capping outliers."
    ])
    add_code([
        "# Check for duplicates and missing values",
        "print('Duplicates:', df.duplicated().sum())",
        "print('Missing Values:\\n', df.isnull().sum().sum())"
    ])
    
    add_code([
        "# Analyze feature statistics",
        "df.describe().T"
    ])
    
    add_code([
        "# Identify constant columns",
        "for col in df.columns:",
        "    if df[col].nunique() == 1:",
        "        print(f'Constant Column: {col} (Value: {df[col].iloc[0]})')",
        "        ",
        "# Drop EmployeeNumber as it is a unique identifier and not predictive",
        "print('Dropping useless columns: EmployeeCount, StandardHours, Over18, EmployeeNumber')",
        "df.drop(columns=['EmployeeCount', 'StandardHours', 'Over18', 'EmployeeNumber'], errors='ignore', inplace=True)"
    ])

    add_code([
        "# Treat Outliers (Capping continuous variables using IQR)",
        "outlier_cols = ['MonthlyIncome', 'TotalWorkingYears', 'YearsAtCompany', ",
        "                'YearsInCurrentRole', 'YearsSinceLastPromotion', 'YearsWithCurrManager']",
        "",
        "def treat_outliers(data, columns):",
        "    cleaned_data = data.copy()",
        "    for col in columns:",
        "        q1 = cleaned_data[col].quantile(0.25)",
        "        q3 = cleaned_data[col].quantile(0.75)",
        "        iqr = q3 - q1",
        "        lower_limit = q1 - 1.5 * iqr",
        "        upper_limit = q3 + 1.5 * iqr",
        "        cleaned_data[col] = cleaned_data[col].clip(lower=lower_limit, upper=upper_limit)",
        "    return cleaned_data",
        "",
        "print('Original statistics for MonthlyIncome outliers:')",
        "print(df['MonthlyIncome'].describe())",
        "df = treat_outliers(df, outlier_cols)",
        "print('\\nCapped statistics for MonthlyIncome outliers:')",
        "print(df['MonthlyIncome'].describe())"
    ])

    # Cell 5: EDA
    add_markdown([
        "## Exploratory Data Analysis (EDA)",
        "We visualize features individually (univariate), in relationships (bivariate), and analyze correlations."
    ])
    
    add_markdown(["### Univariate Analysis: Target Distribution"])
    add_code([
        "# Target variable class distribution",
        "plt.figure(figsize=(6, 4))",
        "sns.countplot(x='Attrition', data=df)",
        "plt.title('Employee Attrition Distribution (Imbalanced Class)')",
        "plt.xlabel('Attrition status')",
        "plt.ylabel('Count')",
        "plt.show()",
        "print('Attrition percentages:')",
        "print(df['Attrition'].value_counts(normalize=True) * 100)"
    ])

    add_markdown(["### Univariate Analysis: Age Distribution"])
    add_code([
        "plt.figure(figsize=(8, 4))",
        "sns.histplot(df['Age'], kde=True, bins=30)",
        "plt.title('Distribution of Employee Age')",
        "plt.xlabel('Age')",
        "plt.show()"
    ])

    add_markdown(["### Bivariate Analysis: Income by Attrition"])
    add_code([
        "plt.figure(figsize=(8, 5))",
        "sns.boxplot(x='Attrition', y='MonthlyIncome', data=df)",
        "plt.title('Monthly Income vs. Attrition Status')",
        "plt.xlabel('Attrition')",
        "plt.ylabel('Monthly Income ($)')",
        "plt.show()"
    ])

    add_markdown(["### Bivariate Analysis: Overtime vs Attrition"])
    add_code([
        "plt.figure(figsize=(6, 4))",
        "sns.countplot(x='OverTime', hue='Attrition', data=df)",
        "plt.title('OverTime impact on Attrition')",
        "plt.xlabel('Works OverTime')",
        "plt.ylabel('Count')",
        "plt.legend(title='Attrition')",
        "plt.show()"
    ])

    add_markdown(["### Correlation Analysis"])
    add_code([
        "# Correlation heatmap for numerical features",
        "numerical_df = df.select_dtypes(include=[np.number])",
        "plt.figure(figsize=(12, 10))",
        "sns.heatmap(numerical_df.corr(), annot=False, cmap='coolwarm', fmt='.2f', linewidths=0.5)",
        "plt.title('Correlation Heatmap of Employee Metrics')",
        "plt.show()"
    ])

    # Cell 6: Feature Engineering and Data Balancing
    add_markdown([
        "## Feature Engineering and Data Balancing",
        "We divide data into features and target, set up hot-encoding for categorical variables, scaling for numerical variables, and apply SMOTE to balance the dataset."
    ])
    add_code([
        "# Target & Features split",
        "y = df['Attrition'].apply(lambda x: 1 if x == 'Yes' else 0)",
        "X = df.drop(columns=['Attrition'])",
        "",
        "num_cols = X.select_dtypes(include=[np.number]).columns.tolist()",
        "cat_cols = X.select_dtypes(include=['object']).columns.tolist()",
        "",
        "print('Numerical Features:', len(num_cols))",
        "print('Categorical Features:', len(cat_cols))"
    ])

    add_code([
        "# Train/Test split (80/20, stratify to maintain target class distribution)",
        "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)",
        "print(f'Train shape: {X_train.shape}, Test shape: {X_test.shape}')"
    ])

    add_code([
        "# Setup Preprocessor pipeline",
        "preprocessor = ColumnTransformer(",
        "    transformers=[",
        "        ('num', StandardScaler(), num_cols),",
        "        ('cat', OneHotEncoder(handle_unknown='ignore', drop='first'), cat_cols)",
        "    ]",
        ")",
        "",
        "X_train_processed = preprocessor.fit_transform(X_train)",
        "X_test_processed = preprocessor.transform(X_test)",
        "",
        "cat_encoder = preprocessor.named_transformers_['cat']",
        "cat_feature_names = cat_encoder.get_feature_names_out(cat_cols).tolist()",
        "feature_names = num_cols + cat_feature_names",
        "print(f'Total Features post-encoding: {len(feature_names)}')"
    ])

    add_code([
        "# Data Balancing: Apply SMOTE on processed training data only",
        "print('Pre-SMOTE Train Target counts:')",
        "print(y_train.value_counts())",
        "",
        "smote = SMOTE(random_state=42)",
        "X_train_res, y_train_res = smote.fit_resample(X_train_processed, y_train)",
        "",
        "print('\\nPost-SMOTE Train Target counts:')",
        "print(pd.Series(y_train_res).value_counts())"
    ])

    # Cell 7: Model Building
    add_markdown([
        "## Model Building",
        "We build and train three classification algorithms:",
        "1. **Logistic Regression**",
        "2. **Random Forest Classifier**",
        "3. **XGBoost Classifier**"
    ])
    add_code([
        "# Define models",
        "models = {",
        "    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),",
        "    'Random Forest': RandomForestClassifier(n_estimators=150, max_depth=12, random_state=42, n_jobs=-1),",
        "    'XGBoost': XGBClassifier(n_estimators=150, max_depth=5, learning_rate=0.08, use_label_encoder=False, eval_metric='logloss', random_state=42)",
        "}",
        "",
        "for name, model in models.items():",
        "    print(f'Training {name}...')",
        "    model.fit(X_train_res, y_train_res)"
    ])

    # Cell 8: Model Evaluation
    add_markdown([
        "## Model Evaluation",
        "We evaluate models on the test dataset using metrics like Accuracy, Precision, Recall, F1-Score, and ROC-AUC. We also examine Confusion Matrices and ROC curves."
    ])
    add_code([
        "metrics = {}",
        "fig, axes = plt.subplots(1, 3, figsize=(18, 5))",
        "roc_fig, roc_ax = plt.subplots(figsize=(8, 6))",
        "",
        "for idx, (name, model) in enumerate(models.items()):",
        "    y_pred = model.predict(X_test_processed)",
        "    y_prob = model.predict_proba(X_test_processed)[:, 1]",
        "    ",
        "    acc = accuracy_score(y_test, y_pred)",
        "    prec = precision_score(y_test, y_pred)",
        "    rec = recall_score(y_test, y_pred)",
        "    f1 = f1_score(y_test, y_pred)",
        "    roc_auc = roc_auc_score(y_test, y_prob)",
        "    ",
        "    metrics[name] = {",
        "        'Accuracy': acc,",
        "        'Precision': prec,",
        "        'Recall': rec,",
        "        'F1-Score': f1,",
        "        'ROC-AUC': roc_auc",
        "    }",
        "    ",
        "    # Plot Confusion Matrix in subplot",
        "    cm = confusion_matrix(y_test, y_pred)",
        "    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=axes[idx],",
        "                xticklabels=['Stayed', 'Left'], yticklabels=['Stayed', 'Left'])",
        "    axes[idx].set_title(f'{name} Confusion Matrix')",
        "    axes[idx].set_ylabel('Actual')",
        "    axes[idx].set_xlabel('Predicted')",
        "    ",
        "    # Plot ROC",
        "    fpr, tpr, _ = roc_curve(y_test, y_prob)",
        "    roc_ax.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.3f})')",
        "",
        "axes[0].set_ylabel('Actual')",
        "fig.tight_layout()",
        "plt.show()",
        "",
        "# Finalize ROC plot",
        "roc_ax.plot([0, 1], [0, 1], 'k--', label='Random')",
        "roc_ax.set_xlabel('False Positive Rate')",
        "roc_ax.set_ylabel('True Positive Rate')",
        "roc_ax.set_title('ROC Curve Comparison')",
        "roc_ax.legend(loc='lower right')",
        "plt.show()"
    ])

    add_code([
        "# Performance Metrics DataFrame",
        "df_metrics = pd.DataFrame(metrics).T",
        "df_metrics"
    ])

    add_markdown(["### Top Predictors of Attrition"])
    add_code([
        "# Plot Top 10 Features from Random Forest",
        "rf_model = models['Random Forest']",
        "rf_importances = rf_model.feature_importances_",
        "indices = np.argsort(rf_importances)[::-1][:10]",
        "",
        "plt.figure(figsize=(10, 5))",
        "sns.barplot(x=rf_importances[indices], y=np.array(feature_names)[indices], palette='viridis')",
        "plt.title('Top 10 Feature Importances (Random Forest)')",
        "plt.xlabel('Importance')",
        "plt.show()"
    ])
    
    # Save notebook file
    notebook_path = os.path.join(notebook_dir, "Employee_Attrition_Analysis.ipynb")
    with open(notebook_path, 'w', encoding='utf-8') as f:
        json.dump(notebook, f, indent=2)
    print(f"Jupyter Notebook successfully created at: {notebook_path}")

if __name__ == "__main__":
    create_notebook()
