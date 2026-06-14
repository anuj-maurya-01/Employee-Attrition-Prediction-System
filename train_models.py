import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, confusion_matrix, roc_curve
)
from imblearn.over_sampling import SMOTE

def treat_outliers(df, columns):
    """Cap numerical outliers at 1.5 * IQR."""
    df_clean = df.copy()
    for col in columns:
        q1 = df_clean[col].quantile(0.25)
        q3 = df_clean[col].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        df_clean[col] = df_clean[col].clip(lower=lower_bound, upper=upper_bound)
    return df_clean

def run_pipeline():
    # Setup directories
    project_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.join(project_dir, "Dataset", "WA_Fn-UseC_-HR-Employee-Attrition.csv")
    model_dir = os.path.join(project_dir, "Model")
    assets_dir = os.path.join(project_dir, "Streamlit_App", "assets")
    
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(assets_dir, exist_ok=True)
    
    print(f"Loading dataset from: {dataset_path}")
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found at {dataset_path}. Run download_dataset.py first.")
        
    df = pd.read_csv(dataset_path)
    
    # Phase 3: Data Preprocessing
    print("Pre-processing data...")
    
    # 1. Check for duplicates and drop them
    duplicate_count = df.duplicated().sum()
    print(f"Duplicates found: {duplicate_count}")
    if duplicate_count > 0:
        df.drop_duplicates(inplace=True)
        
    # 2. Check for missing values
    missing_values = df.isnull().sum().sum()
    print(f"Missing values found: {missing_values}")
    # fillna (if any existed, but IBM dataset has none)
    if missing_values > 0:
        df.fillna(df.median(numeric_only=True), inplace=True)
        
    # 3. Drop constant or useless columns
    useless_cols = ['EmployeeCount', 'StandardHours', 'Over18', 'EmployeeNumber']
    cols_to_drop = [c for c in useless_cols if c in df.columns]
    print(f"Dropping useless columns: {cols_to_drop}")
    df.drop(columns=cols_to_drop, inplace=True)
    
    # 4. Outlier Treatment
    continuous_cols = [
        'Age', 'DailyRate', 'DistanceFromHome', 'HourlyRate', 'MonthlyIncome', 
        'MonthlyRate', 'NumCompaniesWorked', 'PercentSalaryHike', 'TotalWorkingYears', 
        'TrainingTimesLastYear', 'YearsAtCompany', 'YearsInCurrentRole', 
        'YearsSinceLastPromotion', 'YearsWithCurrManager'
    ]
    # We apply IQR capping to columns with high potential for outliers
    outlier_cols = ['MonthlyIncome', 'TotalWorkingYears', 'YearsAtCompany', 
                    'YearsInCurrentRole', 'YearsSinceLastPromotion', 'YearsWithCurrManager']
    print(f"Treating outliers (IQR Capping) for: {outlier_cols}")
    df = treat_outliers(df, outlier_cols)
    
    # 5. Extract target and features
    y = df['Attrition'].apply(lambda x: 1 if x == 'Yes' else 0)
    X = df.drop(columns=['Attrition'])
    
    # Identify numerical and categorical columns
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X.select_dtypes(include=['object']).columns.tolist()
    
    print(f"Numerical features ({len(num_cols)}): {num_cols}")
    print(f"Categorical features ({len(cat_cols)}): {cat_cols}")
    
    # 6. Build the ColumnTransformer Preprocessor
    # We use drop='first' in OneHotEncoder to avoid multicollinearity, especially for Logistic Regression
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), num_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore', drop='first'), cat_cols)
        ]
    )
    
    # 7. Split Train / Test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    print(f"Train set size: {X_train.shape[0]} samples")
    print(f"Test set size: {X_test.shape[0]} samples")
    print(f"Train class distribution:\n{y_train.value_counts(normalize=True)}")
    
    # Fit the preprocessor and transform train and test sets
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    
    # Get feature names after preprocessing
    cat_encoder = preprocessor.named_transformers_['cat']
    # Safe extraction of feature names
    try:
        cat_feature_names = cat_encoder.get_feature_names_out(cat_cols).tolist()
    except AttributeError:
        # Fallback for older scikit-learn versions
        cat_feature_names = cat_encoder.get_feature_names(cat_cols).tolist()
    
    feature_names = num_cols + cat_feature_names
    
    # Save feature names and preprocessing configuration for later use
    joblib.dump(preprocessor, os.path.join(model_dir, 'preprocessor.joblib'))
    joblib.dump(feature_names, os.path.join(model_dir, 'feature_names.joblib'))
    joblib.dump(num_cols, os.path.join(model_dir, 'num_cols.joblib'))
    joblib.dump(cat_cols, os.path.join(model_dir, 'cat_cols.joblib'))
    joblib.dump(X.columns.tolist(), os.path.join(model_dir, 'original_cols.joblib'))
    
    # 8. Address Class Imbalance with SMOTE on Training Data
    print("Applying SMOTE to balance training classes...")
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train_processed, y_train)
    print(f"Balanced Train set size: {X_train_res.shape[0]} samples")
    print(f"Balanced Train class distribution:\n{pd.Series(y_train_res).value_counts(normalize=True)}")
    
    # Save splits for verification if needed
    joblib.dump((X_test_processed, y_test), os.path.join(model_dir, 'test_data.joblib'))
    
    # Phase 6: Model Building
    print("\nTraining models...")
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=150, max_depth=12, random_state=42, n_jobs=-1),
        'XGBoost': XGBClassifier(n_estimators=150, max_depth=5, learning_rate=0.08, use_label_encoder=False, eval_metric='logloss', random_state=42)
    }
    
    metrics = {}
    
    # Visualizations layout for evaluation
    plt.figure(figsize=(15, 5))
    roc_fig, roc_ax = plt.subplots(figsize=(8, 6))
    
    for idx, (name, model) in enumerate(models.items()):
        print(f"Training {name}...")
        model.fit(X_train_res, y_train_res)
        
        # Save model
        model_filename = name.lower().replace(" ", "_") + ".joblib"
        joblib.dump(model, os.path.join(model_dir, model_filename))
        
        # Predictions
        y_pred = model.predict(X_test_processed)
        y_prob = model.predict_proba(X_test_processed)[:, 1]
        
        # Metrics
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_prob)
        
        metrics[name] = {
            'Accuracy': acc,
            'Precision': prec,
            'Recall': rec,
            'F1-Score': f1,
            'ROC-AUC': roc_auc
        }
        
        print(f"{name} Results:")
        print(f"  Accuracy : {acc:.4f} | Precision: {prec:.4f} | Recall: {rec:.4f} | F1: {f1:.4f} | ROC-AUC: {roc_auc:.4f}")
        
        # Plot Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                    xticklabels=['Stayed', 'Left'], yticklabels=['Stayed', 'Left'])
        plt.title(f'{name} Confusion Matrix')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.tight_layout()
        cm_path = os.path.join(assets_dir, f"{name.lower().replace(' ', '_')}_cm.png")
        plt.savefig(cm_path, dpi=100)
        plt.close()
        
        # Plot ROC Curve
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_ax.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc:.3f})")
    
    # Save ROC curves plot
    roc_ax.plot([0, 1], [0, 1], 'k--', label='Random Guess')
    roc_ax.set_xlabel('False Positive Rate')
    roc_ax.set_ylabel('True Positive Rate')
    roc_ax.set_title('ROC Curve Comparison')
    roc_ax.legend(loc='lower right')
    roc_fig.tight_layout()
    roc_fig_path = os.path.join(assets_dir, "roc_comparison.png")
    roc_fig.savefig(roc_fig_path, dpi=150)
    plt.close(roc_fig)
    
    # Phase 7: Model Comparison Summary
    df_metrics = pd.DataFrame(metrics).T
    print("\nModel Comparison Table:")
    print(df_metrics)
    
    # Save metrics table as CSV
    df_metrics.to_csv(os.path.join(model_dir, 'metrics_comparison.csv'))
    
    # Save a comparison bar plot of metrics
    fig, ax = plt.subplots(figsize=(10, 6))
    df_metrics.plot(kind='bar', ax=ax)
    ax.set_title('Model Performance Comparison')
    ax.set_ylabel('Score')
    ax.set_ylim(0, 1.05)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    plt.xticks(rotation=0)
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(os.path.join(assets_dir, 'metrics_comparison.png'), dpi=150)
    plt.close()
    
    # Generate Feature Importance Plot for Random Forest & XGBoost
    # 1. Random Forest Feature Importances
    rf_model = models['Random Forest']
    rf_importances = rf_model.feature_importances_
    rf_indices = np.argsort(rf_importances)[::-1][:10]
    
    plt.figure(figsize=(10, 5))
    sns.barplot(x=rf_importances[rf_indices], y=np.array(feature_names)[rf_indices], palette='viridis')
    plt.title('Top 10 Feature Importances (Random Forest)')
    plt.xlabel('Importance Value')
    plt.tight_layout()
    plt.savefig(os.path.join(assets_dir, 'rf_feature_importances.png'), dpi=150)
    plt.close()
    
    # 2. XGBoost Feature Importances
    xgb_model = models['XGBoost']
    xgb_importances = xgb_model.feature_importances_
    xgb_indices = np.argsort(xgb_importances)[::-1][:10]
    
    plt.figure(figsize=(10, 5))
    sns.barplot(x=xgb_importances[xgb_indices], y=np.array(feature_names)[xgb_indices], palette='mako')
    plt.title('Top 10 Feature Importances (XGBoost)')
    plt.xlabel('Importance Value')
    plt.tight_layout()
    plt.savefig(os.path.join(assets_dir, 'xgb_feature_importances.png'), dpi=150)
    plt.close()
    
    # 3. Logistic Regression Coefficients (Top positive & negative)
    lr_model = models['Logistic Regression']
    lr_coefs = lr_model.coef_[0]
    lr_indices = np.argsort(np.abs(lr_coefs))[::-1][:10]
    
    plt.figure(figsize=(10, 5))
    sns.barplot(x=lr_coefs[lr_indices], y=np.array(feature_names)[lr_indices], palette='coolwarm')
    plt.title('Top 10 Coefficients (Logistic Regression)')
    plt.xlabel('Coefficient Value')
    plt.tight_layout()
    plt.savefig(os.path.join(assets_dir, 'lr_coefficients.png'), dpi=150)
    plt.close()
    
    print("\nAll models trained and evaluated. Serialized models and plots saved successfully!")

if __name__ == "__main__":
    run_pipeline()
