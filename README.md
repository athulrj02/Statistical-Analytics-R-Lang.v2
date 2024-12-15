## **Statistical Analysis Using R**

### **Overview**
This project showcases the application of statistical programming in R to solve three analytical problems involving generalized linear models, Bayesian analytics, and time series forecasting. It explores practical implementations of statistical methods on diverse datasets to uncover insights, evaluate model performance, and make data-driven predictions.

---

### **Key Components**

1. **Generalized Linear Model Analysis**:
   - **Dataset**: [Titanic survival data](https://www.kaggle.com/c/titanic/data).
   - **Objective**: Predict survival probabilities and identify significant predictors.
   - **Methods**:
     - Logistic regression modeling.
     - Evaluation metrics: Accuracy, precision, recall, and F1-score.
   - **Key Results**:
     - Passenger class, gender, age, and family size significantly influenced survival.
     - Model achieved 80.46% accuracy.

2. **Bayesian Analytics**:
   - **Dataset**: Simulated Poisson data.
   - **Objective**: Estimate the Poisson rate parameter (\( \lambda \)) using Bayesian methods.
   - **Methods**:
     - Likelihood computation.
     - Conjugate Gamma prior and posterior estimation.
     - Bayesian Risk Estimator (posterior mean).
   - **Key Results**:
     - Posterior mean (\( \lambda \)) = 4.27.
     - Bayesian framework effectively combined prior beliefs with observed data.

3. **Time Series Analysis**:
   - **Dataset**: TSLA stock price data from Yahoo Finance.
   - **Objective**: Model and forecast stock price trends.
   - **Methods**:
     - Stationarity testing (ADF test, differencing).
     - ARIMA modeling using `auto.arima()`.
     - 10-step ahead forecasting.
   - **Key Results**:
     - ARIMA(0,1,0) identified as the best model.
     - Reliable future stock price forecasts generated.

---

### **Project Structure**

- **Data**:
  - [Titanic survival data](https://www.kaggle.com/c/titanic/data).
  - TSLA stock price data from Yahoo Finance.
  - Simulated Poisson dataset.
  
- **Code**:
  - R scripts for each question demonstrating the full workflow.
  
- **Outputs**:
  - Detailed statistical outputs, including:
    - Confusion matrices.
    - ACF/PACF plots.
    - Forecast visualizations.

- **Report**:
  - A comprehensive report documenting methods, results, and findings for each task.

---

### **How to Run**

1. Clone the repository:
   ```bash
   https://github.com/athulrj02/Statistical-Analytics-R-Lang.v2.git
   ```

2. Open the R scripts in RStudio and run them step by step:
   - Ensure required libraries (`tseries`, `forecast`, `quantmod`) are installed.

3. View the compiled report for detailed insights:
   - Navigate to `StatiAnalysis.docx` for the complete documentation.

---

### **Requirements**

- **Software**:
  - R and RStudio.
  
- **R Libraries**:
  - `tseries`
  - `forecast`
  - `quantmod`

---

### **Results**

This project demonstrates:
- The importance of statistical modeling in understanding data patterns.
- The versatility of R for performing advanced statistical analyses.
- Practical insights from real-world datasets like Titanic and TSLA stock prices.

---

### **Acknowledgments**
- [Kaggle](https://www.kaggle.com/c/titanic/data) for the Titanic survival dataset.
- Yahoo Finance for the TSLA stock price dataset.
- Open-source R libraries for statistical analysis.

---
