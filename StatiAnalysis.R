
setwd("D:\\DBS\\Semester 1\\Statistics\\CA2")
list.files()

############################
### Question 1 Solutions ###
############################
# Load the Titanic dataset
EvData1 <- read.csv("titanic3.csv")

# Take a quick look at the data
str(EvData1)
head(EvData1)

# Handle missing values
EvData1$age[is.na(EvData1$age)] <- mean(EvData1$age, na.rm = TRUE)

# Encode categorical variables as factors
EvData1$sex <- factor(EvData1$sex)
EvData1$pclass <- factor(EvData1$pclass)
EvData1$embarked <- factor(EvData1$embarked)

# Split the data into training and test sets
set.seed(123)  # For reproducibility
train_index <- sample(1:nrow(EvData1), 0.8 * nrow(EvData1))
EvData1_train <- EvData1[train_index, ]
EvData1_test <- EvData1[-train_index, ]

# Train the logistic regression model
glm_model <- glm(survived ~ pclass + sex + age + sibsp + parch + fare, 
                 data = EvData1_train, family = "binomial")

# Summarize the model to check significant variables
model_summary <- summary(glm_model)
print(model_summary)

# Extract significant variables based on p-values (alpha = 0.05)
significant_vars <- summary(glm_model)$coefficients
significant_vars <- significant_vars[significant_vars[, 4] < 0.05, ]
print("Significant Variables (p < 0.05):")
print(significant_vars)

# Predict the output of the test set
EvData1_test$predictions <- predict(glm_model, newdata = EvData1_test, type = "response")

# Define the functional form of the model
cat("Functional form of the model:\n")
cat("logit(P(survived)) = ")
cat(paste(round(coef(glm_model), 4), collapse = " + "), "\n")

# Create a confusion matrix
threshold <- 0.5
EvData1_test$pred_class <- ifelse(EvData1_test$predictions > threshold, 1, 0)
conf_matrix <- table(Actual = EvData1_test$survived, Predicted = EvData1_test$pred_class)
print("Confusion Matrix:")
print(conf_matrix)

# Calculate the probability of correct predictions (accuracy)
accuracy <- sum(diag(conf_matrix)) / sum(conf_matrix)
cat("Accuracy of the model:", round(accuracy, 4), "\n")

# Additional evaluation metrics: Precision, Recall, F1-Score
precision <- conf_matrix[2, 2] / sum(conf_matrix[, 2])  # True Positive / Predicted Positive
recall <- conf_matrix[2, 2] / sum(conf_matrix[2, ])    # True Positive / Actual Positive
f1_score <- 2 * (precision * recall) / (precision + recall)

cat("Precision:", round(precision, 4), "\n")
cat("Recall:", round(recall, 4), "\n")
cat("F1-Score:", round(f1_score, 4), "\n")

############################
### Question 2 Solutions ###
############################


