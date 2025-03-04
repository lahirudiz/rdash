# Load necessary libraries
library(dplyr)
library(ggplot2)
library(corrplot)
library(class)
library(e1071)      # For SVM
library(gbm)        # For Gradient Boosting
library(caret)      # For confusion matrix
library(pROC)       # For ROC curve
library(gridExtra)  # For arranging multiple plots

# Set working directory
setwd(dirname(rstudioapi::getSourceEditorContext()$path))

# Load datasets
student_mat <- read.csv("student-mat.csv")
student_por <- read.csv("student-por.csv")

# Combine datasets into one unified dataset
student_data <- bind_rows(student_mat, student_por)

# Preview the data structure
head(student_data)
summary(student_data)

# Check for missing values in each column
colSums(is.na(student_data))

# Create an average grade feature
student_data <- student_data %>%
  mutate(avg_grade = (G1 + G2 + G3) / 3)

# Preview the data structure after average grade
head(student_data)
summary(student_data)

#------------------- PART 03 / 04 ---------------------

# T-test for romantic relationship impact on grades
t_test_result <- t.test(G3 ~ romantic, data = student_data)
print(t_test_result)

# Linear regression model
model <- lm(G3 ~ studytime + absences + Medu + Fedu, data = student_data)
summary(model)

# Convert final grade to binary outcome (pass/fail)
student_data <- student_data %>%
  mutate(pass_fail = ifelse(G3 >= 10, "pass", "fail"))

# Chi-square test for association between family support and pass/fail outcome
chi_test <- chisq.test(student_data$famsup, student_data$pass_fail)
print(chi_test)

#------------------- Data Preparation for Models ---------------------

# Ensure that the target variable is binary
student_data <- student_data %>%
  mutate(pass_fail_binary = ifelse(pass_fail == "pass", 1, 0))

# Set up the training and testing data
set.seed(123)
train_indices <- sample(1:nrow(student_data), 0.8 * nrow(student_data))
train_data <- student_data[train_indices, ]
test_data <- student_data[-train_indices, ]

# Define predictor variables
predictors <- c("studytime", "absences", "Medu", "Fedu")

#------------------- K-Nearest Neighbors (KNN) ---------------------

# Prepare data for KNN
train_data_knn <- train_data[, predictors]
test_data_knn <- test_data[, predictors]

# Normalize the predictor variables
train_data_knn <- scale(train_data_knn)
test_data_knn <- scale(test_data_knn)

# Apply KNN
k <- 6
knn_pred <- knn(train = train_data_knn, test = test_data_knn, 
                cl = train_data$pass_fail_binary, k = k)

# Confusion matrix to evaluate KNN model performance
confusion_matrix_knn <- confusionMatrix(factor(knn_pred), factor(test_data$pass_fail_binary), positive = "1")
print(confusion_matrix_knn)

# Accuracy calculation for KNN
knn_accuracy <- sum(knn_pred == test_data$pass_fail_binary) / length(knn_pred)
print(paste("KNN Accuracy: ", round(knn_accuracy * 100, 2), "%"))

#------------------- Support Vector Machine (SVM) ---------------------

# Train the SVM model
svm_model <- svm(pass_fail_binary ~ studytime + absences + Medu + Fedu,
                 data = train_data,
                 type = "C-classification",
                 kernel = "linear")

# Make predictions on the test data
svm_pred <- predict(svm_model, test_data)

# Confusion matrix to evaluate SVM performance
confusion_matrix_svm <- confusionMatrix(factor(svm_pred), factor(test_data$pass_fail_binary), positive = "1")
print(confusion_matrix_svm)

# SVM accuracy calculation
svm_accuracy <- sum(svm_pred == test_data$pass_fail_binary) / length(svm_pred)
print(paste("SVM Accuracy: ", round(svm_accuracy * 100, 2), "%"))

#------------------- Gradient Boosting (GBM) ---------------------

# Train the Gradient Boosting model
set.seed(123)
gbm_model <- gbm(pass_fail_binary ~ studytime + absences + Medu + Fedu,
                 data = train_data,
                 distribution = "bernoulli",
                 n.trees = 500,          # Number of trees
                 interaction.depth = 3,  # Depth of trees
                 shrinkage = 0.01,       # Learning rate
                 cv.folds = 5)           # Cross-validation folds

# Summarize the GBM model
summary(gbm_model)

# Predict on the test data
gbm_pred <- predict(gbm_model, test_data, n.trees = gbm_model$n.trees, type = "response")

# Convert predictions to binary (0/1) using a threshold of 0.5
gbm_pred_binary <- ifelse(gbm_pred > 0.5, 1, 0)

# Confusion matrix to evaluate GBM performance
confusion_matrix_gbm <- confusionMatrix(factor(gbm_pred_binary), factor(test_data$pass_fail_binary), positive = "1")
print(confusion_matrix_gbm)

# GBM accuracy calculation
gbm_accuracy <- sum(gbm_pred_binary == test_data$pass_fail_binary) / length(gbm_pred_binary)
print(paste("Gradient Boosting Accuracy: ", round(gbm_accuracy * 100, 2), "%"))

#------------------- VISUALIZATIONS ---------------------

# ROC Curves
knn_roc <- roc(test_data$pass_fail_binary, as.numeric(knn_pred))
svm_roc <- roc(test_data$pass_fail_binary, as.numeric(svm_pred))
gbm_roc <- roc(test_data$pass_fail_binary, gbm_pred)

plot(knn_roc, col = "blue", main = "ROC Curves for Models", lwd = 2)
lines(svm_roc, col = "red", lwd = 2)
lines(gbm_roc, col = "green", lwd = 2)
legend("bottomright", legend = c("KNN", "SVM", "GBM"),
       col = c("blue", "red", "green"), lwd = 2)

# Accuracy Comparison Bar Plot
accuracies <- c(knn_accuracy, svm_accuracy, gbm_accuracy)
model_names <- c("KNN", "SVM", "GBM")
barplot(accuracies * 100, names.arg = model_names, col = c("skyblue", "orange", "green"),
        main = "Accuracy Comparison of Models", ylab = "Accuracy (%)", ylim = c(0, 100))

# Predicted Probabilities vs. Observed Outcomes for GBM
ggplot(data = test_data, aes(x = gbm_pred, y = as.numeric(pass_fail_binary))) +
  geom_point(color = "blue", alpha = 0.6) +
  geom_smooth(method = "lm", se = FALSE, color = "red") +
  labs(title = "Predicted Probabilities vs. Observed Outcomes",
       x = "Predicted Probability (GBM)", y = "Observed Outcome") +
  theme_minimal()

