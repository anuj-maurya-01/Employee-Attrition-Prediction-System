# PROJECT REPORT: EMPLOYEE ATTRITION PREDICTION SYSTEM

---

## DECLARATION OF ORIGINALITY & CODE COMPREHENSION

In alignment with academic integrity guidelines, this project has been built as a structured educational reference. The user/student submitting this work declares that they:
1. Have reviewed and understand every component of the data preprocessing, model building, and evaluation pipeline.
2. Are capable of explaining and tracing every line of code implemented across all modules:
   * **`download_dataset.py`**: Handles URL requests and local stream writing.
   * **`train_models.py`**: Executes IQR capping, standardized numeric scaling, categorical first-column drop hot encoding, SMOTE synthesis, and model training.
   * **`generate_notebook.py`**: Programs the raw JSON cell configuration of the Jupyter Notebook.
   * **`Streamlit_App/app.py` & `utils.py`**: Injects custom CSS themes, aggregates simulation forms, and scales predictions.
3. Understand the ethical boundaries of AI usage as a learning catalyst and have validated all mathematical pipeline stages independently.

---

## CHAPTER 1: INTRODUCTION

### 1.1 Background of Employee Attrition
Employee attrition, defined as the gradual reduction of an organization's workforce as employees leave, retire, or resign without immediate replacement, is a critical challenge for modern enterprise management. Historically, Human Resource (HR) departments operated under a reactive paradigm. Employees would submit their resignation, exit interviews were conducted, and HR would then seek replacements. However, this reactive strategy fails to address the underlying organizational friction that causes high-performing personnel to exit. 

In the modern knowledge-driven economy, human capital is the primary differentiator for corporate success. When an employee leaves, they carry with them specialized technical knowledge, operational experience, and established client relationships. Thus, understanding the mechanics of why employees leave and preemptively identifying retention risk has shifted from a administrative task to a strategic necessity.

### 1.2 Problem Statement
The cost of employee attrition is exceptionally high. Industry estimates suggest that replacing an employee costs between 50% and 200% of their annual salary, depending on their level of specialization. These costs accumulate through multiple channels:
*   **Direct Costs**: Recruitment advertising, background checks, agency fees, and signing bonuses.
*   **Onboarding & Training**: The resource-intensive process of bringing a new hire up to speed, including mentoring hours from senior staff.
*   **Opportunity Costs & Lost Productivity**: A vacant role leads to project delays, while a new hire typically operates at reduced efficiency for their first six months.
*   **Cultural & Morale Impact**: High turnover rates create a sense of instability within teams, which can trigger a cascading effect, prompting other employees to seek external opportunities.

Traditional HR methods rely on subjective assessments, periodic surveys, and rear-view exit interviews. These methods fail to identify risk early enough for intervention. The challenge is to construct an analytical, data-driven system that uses historical employee metrics to predict the probability of future attrition and identify specific risk factors, enabling targeted retention strategies.

### 1.3 Project Objectives
This project aims to develop a predictive machine learning system to mitigate employee attrition. The specific objectives are:
1.  **Develop a Pipeline**: Establish an automated data pipeline to clean, preprocess, scale, and encode employee records.
2.  **Model Selection & Training**: Train and fine-tune three classification models: Logistic Regression, Random Forest Classifier, and XGBoost Classifier.
3.  **Address Class Imbalance**: Apply Synthetic Minority Over-sampling Technique (SMOTE) to prevent class bias towards staying employees.
4.  **Evaluate Performance**: Benchmark models using Accuracy, Precision, Recall, F1-Score, and ROC-AUC.
5.  **Interpretability & Insight**: Identify the top features driving attrition.
6.  **Interactive Deployment**: Build a Streamlit application allowing HR professionals to input employee profiles and receive real-time risk assessments.

---

## CHAPTER 2: LITERATURE REVIEW

### 2.1 Traditional HR Analytics vs. Predictive Modeling
Historically, organizational research on turnover relied heavily on survey-based methodologies, such as Mobley’s intermediate linkages model (1977), which mapped the psychological steps between job dissatisfaction and actual quitting. While these models provided valuable conceptual frameworks, they were static and qualitative. HR departments lacked the tools to apply these theories to individual employee records.

With the rise of Enterprise Resource Planning (Erp) systems, organizations began collecting vast amounts of transactional employee data. Early analytics involved simple descriptive statistics, such as annual turnover rates broken down by department. While informative, descriptive metrics only report what has already occurred. Predictive modeling shifts the focus to forecasting, using historical data to predict future behaviors.

### 2.2 Machine Learning in Talent Analytics
In recent years, researchers have applied supervised machine learning algorithms to predict employee turnover. 
*   **Logistic Regression**: Often serves as a baseline due to its high interpretability. In a study by Punnoose and Ajit (2016), Logistic Regression was compared with other classifiers for employee turnover. It proved highly effective for understanding which factors (e.g., overtime, low salary) directly scale the log-odds of attrition.
*   **Ensemble Methods (Random Forest)**: Random Forest classifiers build multiple decision trees to reduce variance and improve generalization. Studies show that Random Forest excels at capturing non-linear relationships and interactions between features, such as the combined effect of age and salary hike on retention.
*   **Gradient Boosting (XGBoost)**: Extreme Gradient Boosting (XGBoost) has emerged as a state-of-the-art classifier for tabular data. By sequentially training trees to correct the errors of their predecessors, XGBoost frequently achieves superior predictive accuracy. Recent research highlights its ability to handle complex tabular structures, though it requires careful hyperparameter tuning to prevent overfitting.

### 2.3 The Challenge of Class Imbalance in HR Datasets
A common challenge in attrition prediction is class imbalance. In most healthy organizations, the annual turnover rate ranges from 8% to 18%. Consequently, any representative dataset will have a heavy majority of active employees (class 0) and a minority of departed employees (class 1). 

Standard classifiers trained on imbalanced data tend to optimize for overall accuracy, leading to models that default to predicting "no attrition" for almost all cases. This results in high accuracy but very low recall for the minority class, which is counterproductive since missing a high-risk employee is the most costly error. Researchers address this by employing oversampling techniques like SMOTE (Synthetic Minority Over-sampling Technique) to balance the training data, ensuring the model learns the characteristics of the minority class.

---

## CHAPTER 3: METHODOLOGY

### 3.1 Dataset Description
This study utilizes the standard **IBM HR Analytics Employee Attrition & Performance** dataset. It contains 1,470 employee records with 35 attributes. The target variable is `Attrition` (Yes/No).

The dataset features cover multiple dimensions of the employee profile:
1.  **Demographics**: Age, Gender, MaritalStatus, Education, EducationField.
2.  **Work Character & History**: BusinessTravel, Department, JobRole, JobLevel, JobInvolvement, NumCompaniesWorked, TotalWorkingYears, YearsAtCompany, YearsInCurrentRole, YearsSinceLastPromotion, YearsWithCurrManager, OverTime.
3.  **Satisfaction & Environment**: EnvironmentSatisfaction, JobSatisfaction, RelationshipSatisfaction, WorkLifeBalance, DistanceFromHome.
4.  **Compensation & Training**: MonthlyIncome, DailyRate, HourlyRate, MonthlyRate, PercentSalaryHike, StockOptionLevel, TrainingTimesLastYear, PerformanceRating.

### 3.2 Data Preprocessing Pipeline
To prepare the raw data for modeling, we implement a multi-stage preprocessing pipeline:
*   **Duplicate Elimination**: Validate and drop duplicate records to prevent data contamination.
*   **Useless Features Removal**: Drop columns that carry zero variance or are non-informative:
    *   `EmployeeCount`: Constant value of 1.
    *   `StandardHours`: Constant value of 80.
    *   `Over18`: Constant value of 'Y'.
    *   `EmployeeNumber`: Unique sequential ID.
*   **Outlier Treatment**: Continuous features such as `MonthlyIncome` and `YearsAtCompany` contain extreme values (e.g., highly paid executives or rare long-tenured employees). To prevent these outliers from distorting linear models like Logistic Regression, we apply **IQR Capping**. Values exceeding $Q3 + 1.5 \times IQR$ are capped at the upper boundary.
*   **Feature Transformation**:
    *   **Numerical Features**: Standardized using Z-score scaling ($x' = \frac{x - \mu}{\sigma}$) to ensure distance-based metrics and gradient descent optimize efficiently.
    *   **Categorical Features**: Encoded using One-Hot encoding. To prevent multicollinearity (the "dummy variable trap"), we drop the first category for each feature (`drop='first'`).

### 3.3 Algorithms Evaluated
We evaluate three distinct classification algorithms:
1.  **Logistic Regression**:
    A linear classifier that models the probability of attrition using the logistic sigmoid function:
    $$P(Y=1|X) = \frac{1}{1 + e^{-(\beta_0 + \beta_1 X_1 + \dots + \beta_n X_n)}}$$
    It provides high interpretability, as the coefficients directly relate to the odds ratio of the target event.
    
2.  **Random Forest Classifier**:
    An ensemble bagging algorithm that trains multiple independent decision trees on bootstrapped training samples. The final class is determined by majority voting. It handles non-linear splits, is robust to outliers, and inherently ranks features by Gini importance.
    
3.  **XGBoost Classifier**:
    A gradient boosting framework that builds sequential decision trees. In each step, a new tree is trained to predict the residual errors of the existing ensemble using a gradient descent optimization of a regularized objective function. It includes $L_1$ and $L_2$ regularization to prevent overfitting.

### 3.4 Data Balancing via SMOTE
To address the class imbalance (16.1% Attrition), we apply **SMOTE** to the training dataset. SMOTE works by selecting minority class instances, identifying their $k$-nearest neighbors in the feature space, and generating synthetic examples along the line segments joining these instances:
$$x_{new} = x_i + \lambda(x_{zi} - x_i)$$
where $\lambda \in [0,1]$. 

This technique creates a continuous decision boundary for the minority class, preventing the classifiers from overfitting to individual minority samples. Crucially, SMOTE is applied **only to the training split** after fitting the preprocessing pipeline, keeping the test set as a realistic representation of the original class distribution.

---

## CHAPTER 4: IMPLEMENTATION

### 4.1 Exploratory Data Analysis Highlights
During the EDA phase, several key relationships were identified:
*   **Age and Tenure**: Younger employees (ages 18-30) exhibit higher attrition rates. As tenure increases, attrition likelihood drops, showing that early-career integration is a critical retention point.
*   **Overtime**: Of the employees who work overtime, nearly 30% leave the organization, compared to only 10% of those who do not. This indicates a strong link between workload/burnout and attrition.
*   **Income**: Monthly income is negatively correlated with attrition. Employees earning below $3,000/month represent the largest share of attrition, while high-income brackets (above $10,000/month) show low turnover.

### 4.2 Pipeline Implementation Details
The ML pipeline is constructed using scikit-learn's `ColumnTransformer`. Below is the logical layout of the training execution:

```python
# Segment numerical and categorical features
num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = X.select_dtypes(include=['object']).columns.tolist()

# Define transformers
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), num_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore', drop='first'), cat_cols)
    ]
)

# Fit pipeline on training features and transform
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

# Resample using SMOTE
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train_processed, y_train)
```

The trained models are serialized using `joblib` and stored in the `Model/` directory. Evaluation charts (Confusion Matrices, ROC curves) are saved under `Streamlit_App/assets/` to ensure they are available for the Streamlit dashboard.

---

## CHAPTER 5: RESULTS AND DISCUSSION

### 5.1 Comparative Performance Metrics
After training the models, they were tested on the unseen test set (294 samples). The results are summarized below:

| Metric | Logistic Regression | Random Forest | XGBoost |
| :--- | :---: | :---: | :---: |
| **Accuracy** | 0.7789 | 0.8673 | 0.8707 |
| **Precision** | 0.3951 | 0.7000 | 0.6562 |
| **Recall (Sensitivity)** | 0.6809 | 0.2979 | 0.4468 |
| **F1-Score** | 0.5000 | 0.4179 | 0.5316 |
| **ROC-AUC** | 0.8166 | 0.8118 | 0.8354 |

### 5.2 Comparative Analysis & Discussion
*   **Accuracy vs. Recall Trade-Off**: 
    While **XGBoost** and **Random Forest** achieved high overall accuracy (~87%), their recall on the attrition class was lower (44.68% and 29.79% respectively). 
    In contrast, **Logistic Regression** achieved a higher recall of **68.09%**, though its precision was lower (39.51%).
    
*   **The Cost of Errors in HR**:
    In employee attrition prediction, the cost of a False Negative (failing to identify an employee who is about to leave) is much higher than the cost of a False Positive (identifying a staying employee as a risk). A False Negative leads to the loss of an employee, resulting in replacement costs. A False Positive simply results in HR offering additional support or a check-in to an employee who was already planning to stay.
    
    Therefore, a model with higher **Recall** (such as Logistic Regression) is often preferred for HR retention programs, as it captures a larger share of the at-risk population.

### 5.3 Feature Importance Interpretation
Across all models, the features that consistently showed the highest impact on attrition were:
1.  **OverTime_Yes**: Working overtime is the strongest positive predictor of attrition, suggesting that burnout is a primary driver of turnover.
2.  **MonthlyIncome**: Lower income levels are strongly associated with higher attrition risk.
3.  **MaritalStatus_Single**: Single employees exhibit higher turnover compared to married or divorced peers.
4.  **YearsAtCompany**: Low tenure is associated with higher attrition risk, highlighting the importance of the onboarding experience.
5.  **JobRole_SalesRepresentative**: This role shows the highest risk of attrition among all job categories.

### 5.4 Recommendations for HR Management
Based on these findings, we recommend the following retention strategies:
*   **Workload Monitoring**: Implement caps on consecutive overtime hours and introduce compensation or time-off incentives for projects requiring extended hours.
*   **Compensation Benchmarking**: Conduct market reviews for roles with high attrition rates (such as Sales Representatives and Laboratory Technicians) to ensure base salaries are competitive, particularly in the lower-income brackets.
*   **Targeted Onboarding Support**: Establish mentorship programs and check-in schedules during the first 12 to 24 months of an employee's tenure to improve retention during the critical early-career phase.
*   **Predictive Retention Dashboards**: Integrate the predictive tool into regular HR workflows to identify at-risk employees and implement retention measures before resignation letters are submitted.
