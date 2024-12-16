setwd("D:\\DBS\\Semester 1\\Statistics\\CA2")
list.files()

############################
### Question 1 Solutions ###
############################
##GENERALIZED LINEAR MODEL##

# (a) Train the model using 80% of the dataset
# Loaded the Titanic dataset
TData <- read.csv("titanic3.csv")

# Inspect the structure of the dataset
str(TData)  # Understanding the variables
head(TData)  # Quick preview of the data

# Handle missing values in the 'age' column
# Replace missing 'age' values with the mean age
TData$age[is.na(TData$age)] <- mean(TData$age, na.rm = TRUE)

# Encode categorical variables as factors for the GLM
TData$sex <- factor(TData$sex)
TData$pclass <- factor(TData$pclass)
TData$embarked <- factor(TData$embarked)

# Split the dataset into training (80%) and testing (20%) sets
set.seed(123)  # Setting seed for reproducibility
train_index <- sample(1:nrow(TData), 0.8 * nrow(TData))
TData_train <- TData[train_index, ]
TData_test <- TData[-train_index, ]

# Train the logistic regression model (GLM)
glm_model <- glm(survived ~ pclass + sex + age + sibsp + parch + fare, 
                 data = TData_train, family = "binomial")

# model summary
model_summary <- summary(glm_model)
print(model_summary)

# (b) Identify significant variables at alpha = 0.05
# Extracting significant variables based on p-values
significant_vars <- model_summary$coefficients
significant_vars <- significant_vars[significant_vars[, 4] < 0.05, ]
print("Significant Variables (p < 0.05):")
print(significant_vars)

# (c) Predict the output for the test dataset
# Generating predictions for the test data
TData_test$predictions <- predict(glm_model, newdata = TData_test, type = "response")

# Defining the functional form of the model
cat("Functional form of the model:\n")
cat("logit(P(survived)) = ")
cat(paste(round(coef(glm_model), 4), collapse = " + "), "\n")

# (d) Provide confusion matrix and accuracy
# Classifying predictions into binary outcomes using a threshold of 0.5
threshold <- 0.5
TData_test$pred_class <- ifelse(TData_test$predictions > threshold, 1, 0)

# Creating a confusion matrix
conf_matrix <- table(Actual = TData_test$survived, Predicted = TData_test$pred_class)
print("Confusion Matrix:")
print(conf_matrix)

# Calculate the accuracy of the model
accuracy <- sum(diag(conf_matrix)) / sum(conf_matrix)
cat("Accuracy of the model:", round(accuracy, 4), "\n")

# Calculate precision, recall, and F1-score for additional insights
precision <- conf_matrix[2, 2] / sum(conf_matrix[, 2])  # TP / (TP + FP)
recall <- conf_matrix[2, 2] / sum(conf_matrix[2, ])    # TP / (TP + FN)
f1_score <- 2 * (precision * recall) / (precision + recall)

cat("Precision:", round(precision, 4), "\n")
cat("Recall:", round(recall, 4), "\n")
cat("F1-Score:", round(f1_score, 4), "\n")

# --- Summary of Results ---
# Survival is strongly influenced by passenger class, gender, age, and family size. 
# The model performed well with 80.46% accuracy, highlighting these factors as key predictors.



############################
### Question 2 Solutions ###
############################
#####BAYESIAN ANALYTICS#####


# Question 2: Bayesian Analytics

#(a): Compute the Likelihood Function

# Simulate a dataset of 10 observations from a Poisson distribution
# I am assuming λ = 4 as the true rate for simulation
set.seed(123)  # Setting seed for reproducibility
x <- rpois(10, lambda = 4)  # Generate 10 Poisson-distributed observations
print("Simulated dataset (x):")
print(x)

# Defining the likelihood function
# The likelihood for Poisson is L(λ) = Π (λ^xi * e^(-λ) / xi!)
likelihood_function <- function(lambda, x) {
  prod((lambda^x) * exp(-lambda) / factorial(x))
}

# Test the likelihood function for a specific λ
test_lambda <- 4
likelihood <- likelihood_function(test_lambda, x)
cat("Likelihood for λ =", test_lambda, "is", likelihood, "\n")

#(b): Adopt the Conjugate Prior

# Defining the Gamma prior (conjugate prior for Poisson)
# Gamma distribution has parameters α (shape) and β (rate)
alpha <- 2  # Example value for α
beta <- 1   # Example value for β

# Gamma prior density function
gamma_prior <- function(lambda, alpha, beta) {
  dgamma(lambda, shape = alpha, rate = beta)
}

# Test the prior for a specific λ
test_lambda <- 4
prior <- gamma_prior(test_lambda, alpha, beta)
cat("Gamma prior for λ =", test_lambda, "is", prior, "\n")

#(c): Compute the Posterior Distribution

# Update the parameters of the Gamma posterior
# Posterior for Poisson likelihood with Gamma prior is also Gamma
# Updated α = α + Σx, Updated β = β + n (number of observations)
posterior_alpha <- alpha + sum(x)
posterior_beta <- beta + length(x)

cat("Posterior parameters: α =", posterior_alpha, "β =", posterior_beta, "\n")

# Defining the posterior distribution
posterior_distribution <- function(lambda, alpha, beta) {
  dgamma(lambda, shape = alpha, rate = beta)
}

# Test the posterior for a specific λ
posterior <- posterior_distribution(test_lambda, posterior_alpha, posterior_beta)
cat("Posterior for λ =", test_lambda, "is", posterior, "\n")

#(d): Compute the Minimum Bayesian Risk Estimato

# Compute the Bayesian Risk Estimator
# For MSE loss, the Bayesian risk estimator is the posterior mean
bayesian_estimator <- posterior_alpha / posterior_beta
cat("Bayesian Risk Estimator (posterior mean) for λ is", bayesian_estimator, "\n")

# --- Summary of Results ---
# 1. Simulated dataset: [3, 6, 3, 6, 7, 1, 4, 7, 4, 4]
# 2. Likelihood for λ = 4: 8.025e-10; Prior value: 0.07326; Posterior value: 0.61757
# 3. Bayesian Risk Estimator (posterior mean): 4.2727, showing the updated belief for λ.


############################
### Question 3 Solutions ###
############################
####TIME SERIES ANALYSIS####

#libraries required
library(quantmod)
library(tseries)
library(forecast)
#loading the stock data of TESLA from Yahoo finance
getSymbols("TSLA", src = "yahoo", from = "2020-01-01", to = "2023-12-31")
tesla_data <- TSLA
head(tesla_data)


# Data Preparation
# I'll focus on the 'TSLA.Close' column as it represents the closing prices
tsla_close <- Cl(TSLA)  # Extract closing prices
head(tsla_close)

# (a): Check for Stationarity in Mean and Variance
# Plot the time series to visually check for trends and seasonality
plot(tsla_close, main = "TSLA Closing Prices (2020-2023)", ylab = "Closing Price", xlab = "Date")

# Perform the Augmented Dickey-Fuller (ADF) test to check stationarity
adf_test <- adf.test(tsla_close, alternative = "stationary")
print(adf_test)

# If not stationary, difference the series to make it stationary
# First difference to remove trends
tsla_diff <- diff(tsla_close)
plot(tsla_diff, main = "Differenced TSLA Closing Prices", ylab = "Differenced Price", xlab = "Date")

tsla_diff <- na.omit(tsla_diff)
# Perform ADF test on the differenced series
adf_test_diff <- adf.test(tsla_diff, alternative = "stationary")
print(adf_test_diff)

# (b): Identify AR and MA Orders Using ACF and PACF ---
# Plot the ACF and PACF of the differenced series
acf(tsla_diff, main = "ACF of Differenced Series")
pacf(tsla_diff, main = "PACF of Differenced Series")

# Analyze the plots:
# ACF helps determine the order of MA (q)
# PACF helps determine the order of AR (p)

# --- Sub-question (c): Fit ARIMA Model Using auto.arima() ---
# Use auto.arima() to fit the best ARIMA model
arima_model <- auto.arima(tsla_close)
summary(arima_model)

# --- Sub-question (d): Forecast 10 Steps Ahead and Plot ---
# Forecast the next 10 steps (days) using the fitted ARIMA model
forecast_values <- forecast(arima_model, h = 10)

# Plot the original series with forecasts
plot(forecast_values, main = "TSLA Forecast (10 Steps Ahead)", ylab = "Closing Price", xlab = "Date")
print(forecast_values)

# --- Summary of Results ---
# 1. The original TSLA closing price series was non-stationary (ADF p-value = 0.4637).
# 2. Differencing made the series stationary (ADF p-value = 0.01).
# 3. ACF and PACF plots suggested possible ARIMA(1,1,1), but auto.arima() selected ARIMA(0,1,0).
# 4. 10-step ahead forecasts were generated and visualized with confidence intervals.
# 5. The ARIMA(0,1,0) model provided stable predictions for future TSLA closing prices.
  
