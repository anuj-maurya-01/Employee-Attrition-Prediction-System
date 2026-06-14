import os
import pandas as pd
import joblib

def load_ml_assets():
    """Load pre-trained models, preprocessor, and feature metadata."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.join(os.path.dirname(current_dir), "Model")
    
    assets = {}
    try:
        assets['preprocessor'] = joblib.load(os.path.join(model_dir, 'preprocessor.joblib'))
        assets['original_cols'] = joblib.load(os.path.join(model_dir, 'original_cols.joblib'))
        assets['feature_names'] = joblib.load(os.path.join(model_dir, 'feature_names.joblib'))
        
        assets['models'] = {
            'Logistic Regression': joblib.load(os.path.join(model_dir, 'logistic_regression.joblib')),
            'Linear Regression': joblib.load(os.path.join(model_dir, 'linear_regression.joblib')),
            'Random Forest': joblib.load(os.path.join(model_dir, 'random_forest.joblib')),
            'XGBoost': joblib.load(os.path.join(model_dir, 'xgboost.joblib'))
        }
        assets['status'] = "success"
    except Exception as e:
        assets['status'] = "error"
        assets['error_msg'] = str(e)
        
    return assets

def predict_attrition(model, preprocessor, original_cols, input_dict):
    """Predict attrition status (Yes/No) and probability from input features."""
    # Convert input dict to a 1-row DataFrame
    df_input = pd.DataFrame([input_dict])
    
    # Ensure columns are in the exact order as training dataset
    df_input = df_input[original_cols]
    
    # Apply scaling and encoding pipeline
    processed_input = preprocessor.transform(df_input)
    
    # Predict binary outcome and probability
    if hasattr(model, 'predict_proba'):
        pred = int(model.predict(processed_input)[0])
        prob = float(model.predict_proba(processed_input)[0, 1])
    else:
        # Linear Regression case
        import numpy as np
        prob = float(model.predict(processed_input)[0])
        prob = float(np.clip(prob, 0, 1))
        pred = 1 if prob >= 0.5 else 0
    
    return pred, prob

def extract_risk_factors(model, preprocessor, original_cols, input_dict, feature_names):
    """Analyze input values against model weights to identify top attrition risk factors."""
    # Convert input to DataFrame and transform
    df_input = pd.DataFrame([input_dict])
    df_input = df_input[original_cols]
    processed_input = preprocessor.transform(df_input)[0]
    
    factors = []
    
    # If Logistic Regression, we look at coef_ * feature_value to see what contributed most
    # to the positive log-odds of attrition.
    if hasattr(model, 'coef_'):
        import numpy as np
        coefs = model.coef_[0] if len(model.coef_.shape) > 1 else model.coef_
        # Multiply coefficient by scaled value to find contribution
        contributions = coefs * processed_input
        # Sort indices by contribution (highest contribution first)
        top_contrib_indices = contributions.argsort()[::-1][:5]
        
        for idx in top_contrib_indices:
            feat_name = feature_names[idx]
            contrib_val = contributions[idx]
            if contrib_val > 0.05: # Only show factors that push prediction towards Attrition
                factors.append({
                    'feature': feat_name,
                    'type': 'risk',
                    'impact': contrib_val
                })
    else:
        # For tree-based models like Random Forest/XGBoost, we can look at feature importances
        # and see which feature has high importance AND has high/low values.
        # But to keep it simple and accurate, we can highlight attributes that typically cause attrition
        # based on standard business domain rules (e.g. low satisfaction, high overtime, high distance).
        importances = model.feature_importances_
        top_imp_indices = importances.argsort()[::-1][:15]
        
        for idx in top_imp_indices:
            feat_name = feature_names[idx]
            # Match back to original feature and check if user has a risky value
            # e.g., OverTime_Yes, low job satisfaction, low monthly income, high distance
            
            # Simple rule mapping for tree models
            if feat_name == 'OverTime_Yes' and input_dict.get('OverTime') == 'Yes':
                factors.append({'feature': 'Works Overtime', 'type': 'risk', 'impact': importances[idx]})
            elif feat_name == 'MonthlyIncome' and input_dict.get('MonthlyIncome', 5000) < 3000:
                factors.append({'feature': 'Low Monthly Income', 'type': 'risk', 'impact': importances[idx]})
            elif feat_name == 'DistanceFromHome' and input_dict.get('DistanceFromHome', 0) > 15:
                factors.append({'feature': 'Long Commute Distance', 'type': 'risk', 'impact': importances[idx]})
            elif feat_name == 'JobSatisfaction' and input_dict.get('JobSatisfaction', 4) <= 2:
                factors.append({'feature': 'Low Job Satisfaction', 'type': 'risk', 'impact': importances[idx]})
            elif feat_name == 'EnvironmentSatisfaction' and input_dict.get('EnvironmentSatisfaction', 4) <= 2:
                factors.append({'feature': 'Low Environment Satisfaction', 'type': 'risk', 'impact': importances[idx]})
            elif feat_name == 'YearsAtCompany' and input_dict.get('YearsAtCompany', 0) <= 2:
                factors.append({'feature': 'Short Tenure (New Hire)', 'type': 'risk', 'impact': importances[idx]})
            elif feat_name == 'StockOptionLevel' and input_dict.get('StockOptionLevel', 0) == 0:
                factors.append({'feature': 'No Stock Options', 'type': 'risk', 'impact': importances[idx]})
                
    return factors[:3] # return top 3 risk factors
