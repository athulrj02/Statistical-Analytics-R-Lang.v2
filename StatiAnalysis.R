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
set.seed(123)
train_index <- sample(1:nrow(EvData1), 0.8 * nrow(EvData1))
EvData1_train <- EvData1[train_index, ]
EvData1_test <- EvData1[-train_index, ]

# Train the logistic regression model
glm_model <- glm(survived ~ pclass + sex + age + sibsp + parch + fare, data = EvData1_train, family = "binomial")

# Summarize the model
summary(glm_model)

# Make predictions on the test set
EvData1_test$predictions <- predict(glm_model, newdata = EvData1_test, type = "response")

# Compute the confusion matrix
conf_matrix <- table(EvData1_test$survived, EvData1_test$predictions > 0.5)
conf_matrix

# Calculate the probability of correct predictions
accuracy <- sum(diag(conf_matrix)) / sum(conf_matrix)
accuracy

# Functional form of the model
print(glm_model)
