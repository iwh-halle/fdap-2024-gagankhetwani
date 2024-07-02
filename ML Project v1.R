#Data analysis and Machine learning on Crop dataset

# Mehmet Ertugrul (221217027) and Gagan Anil Khetwani (223223662)

#05-07-2024

# Load necessary libraries
library(ggplot2)  # For plotting
library(dplyr)    # For data manipulation
library(caret)    # For machine learning model training
library(e1071)    # For SVM model training
library(readr)    # For reading CSV files
library(randomForest)  # For Random Forest
library(rpart)  # For Decision Tree
library(GGally)  # For Pair Plot
library(ggExtra)  # For Joint Plot with Marginal Histograms
library(reshape2)  # For Correlation Matrix and Heatmap
library(viridis)  # For Correlation Matrix and Heatmap
library(tidyr)  # For Data Reshaping
library(nnet)  # For Neural Networks

if (!requireNamespace("kernlab", quietly = TRUE)) {
  install.packages("kernlab")
}
library(kernlab)
# Read Dataset
crop <- read_csv("C:/My Files/Courses/Masters in Economics, Data Science and Policy/Semester 2/Statistical Machine Learning/Project/Crop_recommendation.csv")

# Display the first 5 rows of the dataset to understand its structure
head(crop, 5)

# Data Analysis

# Count values in 'label' column to see the distribution of different crops
table(crop$label)

# Summary statistics to get an overview of the data
summary(crop)

# Shape of the dataset to understand its size
dim(crop)

# Check for null values in each column
sapply(crop, function(x) sum(is.na(x)))

# Eliminating all duplicated rows to ensure data quality
crop <- crop %>% distinct()

# Check for duplicated values to ensure there are no repeats
stopifnot(sum(duplicated(crop)) == 0)

# Check for unique values in each column to understand the diversity of the data
sapply(crop, function(x) length(unique(x)))

# List all unique values in the 'label' column and count them
unique_labels <- unique(crop$label)
print(unique_labels)
print(length(unique_labels))

# Plotting

# Plot 1: Distribution of temperature
ggplot(crop, aes(x = temperature)) +
  geom_histogram(aes(y = ..density..), bins = 15, fill = "red", alpha = 0.5) +
  geom_density(color = "red") +
  labs(title = "Distribution of Temperature", x = "Temperature", y = "Density") +
  theme_minimal()

#The plot effectively illustrates the central tendency and spread of temperature values in  dataset, 
#providing insights into the typical temperature conditions under which the crops in  study are grown.
#The distribution appears to be approximately normal, with a peak around the 25-30°C range.
#There is a noticeable skew towards higher temperatures, with a long tail extending towards temperatures above 35°C.


# Plot 2: Distribution of pH
ggplot(crop, aes(x = ph)) +
  geom_histogram(aes(y = ..density..), bins = 15, fill = "green", alpha = 0.5) +
  geom_density(color = "green") +
  labs(title = "Distribution of pH", x = "pH", y = "Density") +
  theme_minimal()

#The distribution appears to be approximately normal, centered around a pH value of 7.
#This indicates a slightly skewed distribution, with a majority of values clustered around the neutral pH range (6-8).



# Create the heatmap
heatmap_plot <- ggplot(data = melted_corr, aes(x=Var1, y=Var2, fill=value)) +
  geom_tile(color = "white") +
  scale_fill_viridis(name="Correlation", limits=c(-1, 1)) +
  theme_minimal() +
  labs(title = "Correlation between different features",
       x = "Features",
       y = "Features") +
  theme(axis.text.x = element_text(angle = 45, vjust = 1, hjust = 1),
        plot.title = element_text(size = 15, color = "black", hjust = 0.5)) +
  geom_text(aes(label = value), color = "white", size = 4)

# Save the heatmap plot
ggsave("correlation_heatmap.png", plot = heatmap_plot, width = 15, height = 9, dpi = 300)

# Print the heatmap plot
print(heatmap_plot)


#The heatmap shows correlations between features. Strong positive correlation: P and K (0.74).
#Strong negative correlation: N and P (-0.23). Most other correlations are weak. Useful for feature
#selection and understanding data relationships.


# Summary Statistics

# Create the summary table by grouping by 'label' and calculating the mean
crop_summary <- crop %>%
  group_by(label) %>%
  summarise(across(everything(), mean, na.rm = TRUE))

# Display the first few rows of the summary table
head(crop_summary)

# N P K values comparison between crops

# Reshape the data for plotting
crop_summary_long <- crop_summary %>%
  select(label, N, P, K) %>%
  pivot_longer(cols = c(N, P, K), names_to = "Nutrient", values_to = "Value")

# Create the grouped bar chart for NPK values comparison
p <- ggplot(crop_summary_long, aes(x = label, y = Value, fill = Nutrient)) +
  geom_bar(stat = "identity", position = position_dodge()) +
  scale_fill_manual(values = c("mediumvioletred", "springgreen", "dodgerblue")) +
  theme_minimal() +
  labs(title = "N-P-K values comparison between crops",
       x = "Crops",
       y = "Average Values") +
  theme(plot.title = element_text(hjust = 0.5, size = 15),
        axis.text.x = element_text(angle = -45, hjust = 0.1, size = 8),
        plot.background = element_rect(fill = "white"),
        panel.background = element_rect(fill = "white"))

# Save the plot
ggsave("npk_comparison.png", plot = p, width = 15, height = 9, dpi = 300)

# Print the plot
print(p)

#The bar chart compares average values of Nitrogen (N), Phosphorus (P), and Potassium (K) for different crops. It shows:
#Higher values of K and N compared to P.
#Significant variability in nutrient needs among crops.
#Crops like apple, coconut, and grapes have high K and N values.
#Useful for tailoring fertilization strategies to crop-specific nutrient needs.


# Feature Selection and Data Splitting

# Select features and target
features <- crop %>% select(N, P, K, temperature, humidity, ph, rainfall)
target <- crop$label

# Combine features and target into one dataframe
data <- features %>% mutate(label = target)

# Normalize the features
preProcValues <- preProcess(data, method = c("center", "scale"))
data <- predict(preProcValues, data)

# Split the data into training and testing sets
set.seed(2)
trainIndex <- createDataPartition(data$label, p = .8, list = FALSE, times = 1)
dataTrain <- data[ trainIndex,]
dataTest  <- data[-trainIndex,]

# Separate the features and target for training and testing sets
x_train <- dataTrain %>% select(-label)
y_train <- dataTrain$label
x_test <- dataTest %>% select(-label)
y_test <- dataTest$label

# Ensure factors have the same levels
y_train <- factor(y_train)
y_test <- factor(y_test, levels = levels(y_train))

# Display the first few rows of the training and testing sets
head(x_train)
head(y_train)
head(x_test)
head(y_test)

# Decision Tree Model

# Train a decision tree model
dt_model <- rpart(label ~ ., data = dataTrain, method = "class", 
                  parms = list(split = "information"), 
                  control = rpart.control(maxdepth = 5))

# Predict on the test set
dt_predictions <- predict(dt_model, dataTest, type = "class")
dt_predictions <- factor(dt_predictions, levels = levels(y_test))

# Calculate and print accuracy
dt_accuracy <- mean(dt_predictions == y_test)
print(paste("Decision Tree's Accuracy is: ", dt_accuracy * 100))

# Print classification report
conf_matrix <- confusionMatrix(dt_predictions, y_test)
print("Classification Report")
print(conf_matrix)


# Cross-validation score
cv_scores <- train(label ~ ., data = data, method = "rpart", 
                   trControl = trainControl(method = "cv", number = 5), 
                   tuneGrid = expand.grid(cp = 0.01))
print(paste("Cross-validation score: ", cv_scores$results$Accuracy))

# Training and testing accuracy
dt_train_accuracy <- mean(predict(dt_model, dataTrain, type = "class") == y_train)
print(paste("Training accuracy = ", dt_train_accuracy))
dt_test_accuracy <- mean(dt_predictions == y_test)
print(paste("Testing accuracy = ", dt_test_accuracy))

# Plot confusion matrix
cm_dt <- conf_matrix$table
cm_dt_plot <- ggplot(data = as.data.frame(cm_dt), aes(x = Reference, y = Prediction)) +
  geom_tile(aes(fill = Freq), color = "white") +
  scale_fill_gradient(low = "white", high = "green") +
  theme_minimal() +
  geom_text(aes(label = Freq), vjust = 1) +
  labs(title = "Confusion Matrix: Predicted vs Actual", x = "Actual", y = "Predicted")
print(cm_dt_plot)


#The confusion matrix shows:

#High accuracy with most values on the diagonal.
#Correct classifications for crops like watermelon, apple, and banana (frequency 20).
#Misclassifications include jute as kidneybeans (5) and jute as lentil (4).
#The color gradient highlights correct and incorrect predictions.
#Useful for identifying crops prone to misclassification and refining the model.

# Random Forest Model

# Ensure 'label' is treated as a factor for classification
dataTrain$label <- as.factor(dataTrain$label)
dataTest$label <- as.factor(dataTest$label)

# Train a Random Forest model
rf_model <- randomForest(label ~ ., data = dataTrain, ntree = 20)

# Predict on the test set
rf_predictions <- predict(rf_model, dataTest)

# Ensure predictions are factors with the same levels as y_test
rf_predictions <- factor(rf_predictions, levels = levels(y_test))

# Calculate and print accuracy
rf_accuracy <- mean(rf_predictions == y_test)
print(paste("Random Forest Accuracy is: ", rf_accuracy))

# Print classification report
rf_conf_matrix <- confusionMatrix(rf_predictions, y_test)
print("Classification Report")
print(rf_conf_matrix)

# Cross-validation score
cv_control <- trainControl(method = "cv", number = 5)
cv_model <- train(label ~ ., data = data, method = "rf", trControl = cv_control, tuneGrid = expand.grid(mtry = 2))
cv_scores <- cv_model$results$Accuracy
print(paste("Cross-validation score: ", mean(cv_scores)))

# Training and testing accuracy
rf_train_accuracy <- mean(predict(rf_model, dataTrain) == dataTrain$label)
print(paste("Training accuracy = ", rf_train_accuracy))
rf_test_accuracy <- mean(rf_predictions == dataTest$label)
print(paste("Testing accuracy = ", rf_test_accuracy))

# Plot confusion matrix
cm_rf <- rf_conf_matrix$table
cm_rf_plot <- ggplot(data = as.data.frame(cm_rf), aes(x = Reference, y = Prediction)) +
  geom_tile(aes(fill = Freq), color = "white") +
  scale_fill_gradient(low = "white", high = "green") +
  theme_minimal() +
  geom_text(aes(label = Freq), vjust = 1) +
  labs(title = "Confusion Matrix: Predicted vs Actual", x = "Actual", y = "Predicted")
print(cm_rf_plot)


#The confusion matrix for the Random Forest model shows:

#High Accuracy: Most predictions are correct, seen by the diagonal dominance (e.g., watermelon, apple, banana have 20 correct predictions each).
#Few Misclassifications: Minor misclassifications like maize as kidneybeans (1) and jute as kidneybeans (1).
#Visualization: Green gradient highlights frequencies, making correct predictions and misclassifications clear.

# SVM Model

# Train an SVM model
svm_model <- svm(label ~ ., data = dataTrain)

# Predict on the test set
svm_predictions <- predict(svm_model, dataTest)

# Ensure predictions are factors with the same levels as y_test
svm_predictions <- factor(svm_predictions, levels = levels(y_test))

# Calculate and print accuracy
svm_accuracy <- mean(svm_predictions == y_test)
print(paste("SVM Accuracy is: ", svm_accuracy))

# Print classification report
svm_conf_matrix <- confusionMatrix(svm_predictions, y_test)
print("Classification Report")
print(svm_conf_matrix)

# Plot confusion matrix
cm_svm <- svm_conf_matrix$table
cm_svm_plot <- ggplot(data = as.data.frame(cm_svm), aes(x = Reference, y = Prediction)) +
  geom_tile(aes(fill = Freq), color = "white") +
  scale_fill_gradient(low = "white", high = "green") +
  theme_minimal() +
  geom_text(aes(label = Freq), vjust = 1) +
  labs(title = "Confusion Matrix: Predicted vs Actual (SVM)", x = "Actual", y = "Predicted")

# Print the plot
print(cm_svm_plot)

#The confusion matrix for the SVM model shows:

#High Accuracy: Most predictions are correct, as indicated by the dominant diagonal (e.g., watermelon, apple, banana each have 20 correct predictions).
#Misclassifications: There are a few misclassifications, such as jute being predicted as kidneybeans (4 instances) and mungbean as rice (4 instances).


# k-NN Model

# Train a k-NN model
knn_model <- train(label ~ ., data = dataTrain, method = "knn", tuneLength = 5)

# Predict on the test set
knn_predictions <- predict(knn_model, dataTest)

# Ensure predictions are factors with the same levels as y_test
knn_predictions <- factor(knn_predictions, levels = levels(y_test))

# Calculate and print accuracy
knn_accuracy <- mean(knn_predictions == y_test)
print(paste("k-NN Accuracy is: ", knn_accuracy))

# Print classification report
knn_conf_matrix <- confusionMatrix(knn_predictions, y_test)
print("Classification Report")
print(knn_conf_matrix)


# Plot confusion matrix
cm_knn <- knn_conf_matrix$table
cm_knn_plot <- ggplot(data = as.data.frame(cm_knn), aes(x = Reference, y = Prediction)) +
  geom_tile(aes(fill = Freq), color = "white") +
  scale_fill_gradient(low = "white", high = "green") +
  theme_minimal() +
  geom_text(aes(label = Freq), vjust = 1) +
  labs(title = "Confusion Matrix: Predicted vs Actual (k-NN)", x = "Actual", y = "Predicted")

# Print the plot
print(cm_knn_plot)

#The confusion matrix for the k-NN model shows:

#High Accuracy: Most predictions are correct, seen by the dominant diagonal (e.g., watermelon, apple, banana each have 20 correct predictions).
#Misclassifications: Misclassifications include:
#Rice as mungbean (4 instances).
#Mango as maize (2 instances).
#Lentil as kidneybeans (5 instances).
#Cotton as kidneybeans (2 instances).



# Compare Model Accuracies

# Collect accuracies and model names
acc <- c(dt_accuracy, rf_accuracy, svm_accuracy, knn_accuracy)
model <- c("Decision Tree", "Random Forest", "SVM", "k-NN")

# Create a data frame from the accuracy and model vectors
accuracy_data <- data.frame(Model = model, Accuracy = acc)

# Create the bar plot for accuracy comparison
accuracy_plot <- ggplot(accuracy_data, aes(x = Accuracy, y = Model, fill = Model)) +
  geom_bar(stat = "identity") +
  scale_fill_viridis_d() +
  theme_minimal() +
  labs(title = "Accuracy Comparison", x = "Accuracy", y = "ML Algorithms") +
  theme(
    plot.title = element_text(hjust = 0.5, size = 20, face = "bold"),
    axis.title.x = element_text(size = 14),
    axis.title.y = element_text(size = 14),
    axis.text.x = element_text(size = 12),
    axis.text.y = element_text(size = 12),
    legend.position = "none"
  )

# Save the plot
ggsave("accuracy_comparison_plot.png", plot = accuracy_plot, width = 14, height = 7, dpi = 300)

# Print the plot
print(accuracy_plot)

#The bar chart shows the accuracy comparison of different machine learning algorithms:

#SVM (Support Vector Machine) has the highest accuracy, followed closely by:
#Random Forest, which also shows strong performance.
#k-NN (k-Nearest Neighbors) has a slightly lower accuracy compared to SVM and Random Forest.
#Decision Tree has the lowest accuracy among the four algorithms.
#This visualization highlights that SVM and Random Forest are the top-performing models for this dataset, while Decision Tree and k-NN are less accurate

# Train vs Test Accuracy Comparison

# Collect accuracies and model names for both training and testing sets
label <- c("Decision Tree", "Random Forest", "SVM", "k-NN")
Test <- c(dt_test_accuracy, rf_test_accuracy, svm_accuracy, knn_accuracy)
Train <- c(dt_train_accuracy, rf_train_accuracy, mean(predict(svm_model, dataTrain) == dataTrain$label), mean(predict(knn_model, dataTrain) == dataTrain$label))

# Create a data frame with the accuracies and labels
accuracy_data <- data.frame(
  Algorithm = rep(label, each = 2),
  Accuracy = c(Test, Train),
  Type = rep(c("Test", "Train"), times = length(label))
)

# Create the bar plot for train vs test accuracy comparison
accuracy_plot <- ggplot(accuracy_data, aes(x = Algorithm, y = Accuracy, fill = Type)) +
  geom_bar(stat = "identity", position = position_dodge(width = 0.4)) +
  scale_fill_manual(values = c("midnightblue", "mediumaquamarine")) +
  theme_minimal() +
  labs(title = "Testing vs Training Accuracy", x = "ML Algorithms", y = "Accuracy") +
  theme(
    plot.title = element_text(hjust = 0.5, size = 20, face = "bold"),
    axis.title.x = element_text(size = 14),
    axis.title.y = element_text(size = 14),
    axis.text.x = element_text(size = 12),
    axis.text.y = element_text(size = 12)
  )

# Save the plot
ggsave("train_vs_test_accuracy_plot.png", plot = accuracy_plot, width = 20, height = 7, dpi = 300)

# Print the plot
print(accuracy_plot)

#The plot visually shows that:

#Decision Tree has a small gap between training and testing accuracy, suggesting slight overfitting.
#k-NN has similar accuracies for both training and testing, indicating good generalization.
#Random Forest and SVM exhibit high accuracies with minimal differences between training and testing, reflecting excellent generalization.


# Train a Neural Network model
nn_model <- nnet(label ~ ., data = dataTrain, size = 5, decay = 1e-4, maxit = 200)

# Predict on the test set
nn_predictions <- predict(nn_model, dataTest, type = "class")

# Ensure predictions are factors with the same levels as y_test
nn_predictions <- factor(nn_predictions, levels = levels(y_test))

# Calculate and print accuracy
nn_accuracy <- mean(nn_predictions == y_test)
print(paste("Neural Network Accuracy is: ", nn_accuracy))

# Print classification report
nn_conf_matrix <- confusionMatrix(nn_predictions, y_test)
print("Classification Report")
print(nn_conf_matrix)



# Convert confusion matrix to data frame for plotting
nn_cm_df <- as.data.frame(nn_conf_matrix$table)

# Plot the confusion matrix
nn_cm_plot <- ggplot(data = nn_cm_df, aes(x = Reference, y = Prediction)) +
  geom_tile(aes(fill = Freq), color = "white") +
  scale_fill_viridis(name = "Frequency", option = "D") +
  theme_minimal() +
  geom_text(aes(label = Freq), color = "white", size = 4) +
  labs(title = "Confusion Matrix: Neural Network", x = "Actual", y = "Predicted") +
  theme(plot.title = element_text(hjust = 0.5, size = 15))

# Print the confusion matrix plot
print(nn_cm_plot)

#Confusion Matrix Analysis for Neural Network Model
#High Accuracy: The model shows high accuracy with most predictions correctly classified (e.g., apple, banana, blackgram, chickpea).
#Misclassifications:
#Rice as maize (2 instances)
#Pomegranate as kidneybeans (2 instances)
#Pigeonpeas as rice (1 instance)
#Lentil as maize (4 instances)
#Jute as kidneybeans (3 instances)
#Cotton as kidneybeans (1 instance)
#Coffee as grapes (1 instance)



# Collect accuracies and model names
acc <- c(dt_accuracy, rf_accuracy, svm_accuracy, knn_accuracy, nn_accuracy)
model <- c("Decision Tree", "Random Forest", "SVM", "k-NN", "Neural Network")

# Create a data frame from the accuracy and model vectors
accuracy_data <- data.frame(Model = model, Accuracy = acc)

# Create the bar plot for accuracy comparison
accuracy_plot <- ggplot(accuracy_data, aes(x = Model, y = Accuracy, fill = Model)) +
  geom_bar(stat = "identity") +
  scale_fill_viridis_d() +
  theme_minimal() +
  labs(title = "Accuracy Comparison", x = "ML Algorithms", y = "Accuracy") +
  theme(
    plot.title = element_text(hjust = 0.5, size = 20, face = "bold"),
    axis.title.x = element_text(size = 14),
    axis.title.y = element_text(size = 14),
    axis.text.x = element_text(size = 12),
    axis.text.y = element_text(size = 12),
    legend.position = "none"
  )

# Save the plot
ggsave("accuracy_comparison_plot.png", plot = accuracy_plot, width = 14, height = 7, dpi = 300)

# Print the plot
print(accuracy_plot)

#The bar chart shows the accuracy comparison of various machine learning models including Decision Tree, k-NN, Neural Network, Random Forest, and SVM.

#Neural Network:
#Accuracy: High, similar to Random Forest and SVM.
#Performance: Indicates the model effectively learns and generalizes the patterns in the data.
#Comparison:
#Outperforms Decision Tree and k-NN.
#Comparable to top performers like Random Forest and SVM.
#Neural Network: Exhibits high accuracy, indicating strong learning and generalization capabilities.

# Collect accuracies and model names for both training and testing sets
label <- c("Decision Tree", "Random Forest", "SVM", "k-NN", "Neural Network")
Test <- c(dt_test_accuracy, rf_test_accuracy, svm_accuracy, knn_accuracy, nn_accuracy)
Train <- c(dt_train_accuracy, rf_train_accuracy, mean(predict(svm_model, dataTrain) == dataTrain$label), mean(predict(knn_model, dataTrain) == dataTrain$label), mean(predict(nn_model, dataTrain, type = "class") == dataTrain$label))

# Create a data frame with the accuracies and labels
accuracy_data <- data.frame(
  Algorithm = rep(label, each = 2),
  Accuracy = c(Test, Train),
  Type = rep(c("Test", "Train"), times = length(label))
)

# Create the bar plot for train vs test accuracy comparison
accuracy_plot <- ggplot(accuracy_data, aes(x = Algorithm, y = Accuracy, fill = Type)) +
  geom_bar(stat = "identity", position = position_dodge(width = 0.4)) +
  scale_fill_manual(values = c("midnightblue", "mediumaquamarine")) +
  theme_minimal() +
  labs(title = "Testing vs Training Accuracy", x = "ML Algorithms", y = "Accuracy") +
  theme(
    plot.title = element_text(hjust = 0.5, size = 20, face = "bold"),
    axis.title.x = element_text(size = 14),
    axis.title.y = element_text(size = 14),
    axis.text.x = element_text(size = 12),
    axis.text.y = element_text(size = 12)
  )

# Save the plot
ggsave("train_vs_test_accuracy_plot.png", plot = accuracy_plot, width = 20, height = 7, dpi = 300)

# Print the plot
print(accuracy_plot)





# Define the grid of hyperparameters to search for Random Forest
rf_grid <- expand.grid(mtry = c(2, 3, 4))

# Train the Random Forest model with cross-validation
rf_tune <- train(label ~ ., data = dataTrain, method = "rf", tuneGrid = rf_grid,
                 trControl = trainControl(method = "cv", number = 5))

# Print the best hyperparameters and the corresponding accuracy
print(rf_tune$bestTune)
print(rf_tune$results)

# Plot the accuracy against different values of mtry
rf_plot <- ggplot(rf_tune) +
  labs(title = "Random Forest Hyperparameter Tuning",
       x = "Number of Variables Randomly Sampled as Candidates (mtry)",
       y = "Accuracy",
       subtitle = "Tuning the 'mtry' Hyperparameter") +
  theme_minimal()
print(rf_plot)

#The plot shows the accuracy of the Random Forest model against different values of the mtry hyperparameter,
# which is the number of variables randomly sampled as candidates at each split.

#Highest Accuracy: Achieved at mtry = 2 with an accuracy of approximately 0.9955.
#Decreasing Accuracy: As mtry increases from 2 to 4, the accuracy decreases slightly.

#The optimal value for mtry is 2, as it provides the highest accuracy.

#The Random Forest model performs best when fewer variables are considered at each split, suggesting that considering
# too many variables may lead to overfitting or unnecessary complexity.



# Define the grid of hyperparameters to search for SVM
svm_grid <- expand.grid(C = 2^(-1:1), sigma = 2^(-1:1))

# Train the SVM model with cross-validation
svm_tune <- train(label ~ ., data = dataTrain, method = "svmRadial", tuneGrid = svm_grid,
                  trControl = trainControl(method = "cv", number = 5))

# Print the best hyperparameters and the corresponding accuracy
print(svm_tune$bestTune)
print(svm_tune$results)

# Plot the accuracy against different values of C and sigma
svm_plot <- ggplot(svm_tune) +
  labs(title = "SVM Hyperparameter Tuning",
       x = "Cost Parameter (C)",
       y = "Accuracy",
       subtitle = "Tuning the 'C' and 'sigma' Hyperparameters") +
  theme_minimal()
print(svm_plot)

# Define the grid of hyperparameters to search for k-NN
knn_grid <- expand.grid(k = seq(3, 21, by = 2))

# Train the k-NN model with cross-validation
knn_tune <- train(label ~ ., data = dataTrain, method = "knn", tuneGrid = knn_grid,
                  trControl = trainControl(method = "cv", number = 5))

# Print the best hyperparameters and the corresponding accuracy
print(knn_tune$bestTune)
print(knn_tune$results)

# Plot the accuracy against different values of k
knn_plot <- ggplot(knn_tune) +
  labs(title = "k-NN Hyperparameter Tuning",
       x = "Number of Neighbors (k)",
       y = "Accuracy",
       subtitle = "Tuning the 'k' Hyperparameter") +
  theme_minimal()
print(knn_plot)



#Highest Accuracy: Achieved at k = 5 with an accuracy of approximately 0.98.
#Decreasing Accuracy: As k increases from 5 to 20, the accuracy steadily decreases.
#Optimal Value:

#The optimal value for k is 5, as it provides the highest accuracy.

#The k-NN model performs best when a moderate number of neighbors are considered. A smaller k value tends to be more sensitive
#to the local structure of the data, leading to higher accuracy.
#As k increases, the model becomes less sensitive to local variations and more generalized, resulting in a decrease in accuracy.

# Define the grid of hyperparameters to search for Neural Network
nn_grid <- expand.grid(size = c(3, 5, 7), decay = c(1e-4, 1e-3, 1e-2))

# Train the Neural Network model with cross-validation
nn_tune <- train(label ~ ., data = dataTrain, method = "nnet", tuneGrid = nn_grid,
                 trControl = trainControl(method = "cv", number = 5), linout = TRUE, trace = FALSE, MaxNWts = 1000)

# Print the best hyperparameters and the corresponding accuracy
print(nn_tune$bestTune)
print(nn_tune$results)

# Plot the accuracy against different values of size and decay
nn_plot <- ggplot(nn_tune) +
  labs(title = "Neural Network Hyperparameter Tuning",
       x = "Number of Units in the Hidden Layer (size)",
       y = "Accuracy",
       subtitle = "Tuning the 'size' and 'decay' Hyperparameters") +
  theme_minimal()
print(nn_plot)



#The plot shows how the accuracy of a Neural Network changes with the number of units in the hidden layer (size) and the weight decay (decay). 
#Each line represents a different decay value (1e-02, 1e-03, 1e-04).


#Accuracy Increases:

#Accuracy improves as the number of hidden units increases, especially from 3 to 5 units.
#Decay Impact:

#Different decay values (1e-02, 1e-03, 1e-04) have similar accuracy trends.
#The best accuracy is reached with 5 to 7 hidden units.
#Optimal Settings:

#Optimal accuracy occurs with a decay value of 1e-02 and 5 to 7 hidden units.
#Adding more than 7 units doesn’t improve accuracy much.

#Accuracy improves with more hidden units up to a point, but beyond 7 units, there's little benefit. Tuning both hidden layer size and decay is important for the best performance.

# Create a summary table of the best hyperparameters and their corresponding accuracies



hyperparameter_summary <- data.frame(
  Model = c("Random Forest", "SVM", "k-NN", "Neural Network"),
  Best_Parameters = c(
    paste("mtry =", rf_tune$bestTune$mtry),
    paste("C =", svm_tune$bestTune$C, ", sigma =", svm_tune$bestTune$sigma),
    paste("k =", knn_tune$bestTune$k),
    paste("size =", nn_tune$bestTune$size, ", decay =", nn_tune$bestTune$decay)
  ),
  Accuracy = c(
    max(rf_tune$results$Accuracy),
    max(svm_tune$results$Accuracy),
    max(knn_tune$results$Accuracy),
    max(nn_tune$results$Accuracy)
  )
)
print(hyperparameter_summary)

#The output shows the best hyperparameters and their corresponding accuracies for different machine learning models:

#Random Forest:
#Best Parameters: mtry = 2
#Accuracy: 99.55%

#SVM (Support Vector Machine):
#Best Parameters: C = 1, sigma = 0.5
#Accuracy: 98.81%

#k-NN (k-Nearest Neighbors):
#Best Parameters: k = 5
#Accuracy: 97.90%

#Neural Network:
#Best Parameters: size = 7, decay = 0.001
#Accuracy: 97.44%

#Interpretation
#Random Forest achieved the highest accuracy with the parameter mtry = 2.
#SVM performed well with C = 1 and sigma = 0.5.
#k-NN had optimal performance with k = 5.
#Neural Network showed good results with size = 7 and decay = 0.001.

#Random Forest outperformed the other models in this dataset. However, SVM, k-NN, and Neural Network also achieved high accuracies,
#making them viable options depending on the context and specific requirements of the application.


# PCA Implementation
pca <- prcomp(x_train, center = TRUE, scale. = TRUE)
summary(pca)

# Choose the number of principal components to retain
# For example, retain enough components to explain 95% of the variance
explained_variance <- cumsum(pca$sdev^2) / sum(pca$sdev^2)
n_components <- which(explained_variance >= 0.95)[1]
print(paste("Number of components to retain:", n_components))

# Transform the training and testing data using PCA
train_pca <- as.data.frame(predict(pca, x_train)[, 1:n_components])
test_pca <- as.data.frame(predict(pca, x_test)[, 1:n_components])

# Add the target label back to the PCA-transformed data
train_pca$label <- y_train
test_pca$label <- y_test

# Model Training and Evaluation

# Decision Tree Model with PCA
dt_model_pca <- rpart(label ~ ., data = train_pca, method = "class", 
                      parms = list(split = "information"), 
                      control = rpart.control(maxdepth = 5))
dt_predictions_pca <- predict(dt_model_pca, test_pca, type = "class")
dt_predictions_pca <- factor(dt_predictions_pca, levels = levels(y_test))
dt_accuracy_pca <- mean(dt_predictions_pca == y_test)
print(paste("Decision Tree's Accuracy with PCA is: ", dt_accuracy_pca * 100))

# Random Forest Model with PCA
rf_model_pca <- randomForest(label ~ ., data = train_pca, ntree = 20)
rf_predictions_pca <- predict(rf_model_pca, test_pca)
rf_predictions_pca <- factor(rf_predictions_pca, levels = levels(y_test))
rf_accuracy_pca <- mean(rf_predictions_pca == y_test)
print(paste("Random Forest Accuracy with PCA is: ", rf_accuracy_pca))

# SVM Model with PCA
svm_model_pca <- svm(label ~ ., data = train_pca)
svm_predictions_pca <- predict(svm_model_pca, test_pca)
svm_predictions_pca <- factor(svm_predictions_pca, levels = levels(y_test))
svm_accuracy_pca <- mean(svm_predictions_pca == y_test)
print(paste("SVM Accuracy with PCA is: ", svm_accuracy_pca))

# k-NN Model with PCA
knn_model_pca <- train(label ~ ., data = train_pca, method = "knn", tuneLength = 5)
knn_predictions_pca <- predict(knn_model_pca, test_pca)
knn_predictions_pca <- factor(knn_predictions_pca, levels = levels(y_test))
knn_accuracy_pca <- mean(knn_predictions_pca == y_test)
print(paste("k-NN Accuracy with PCA is: ", knn_accuracy_pca))

# Neural Network Model with PCA
nn_model_pca <- nnet(label ~ ., data = train_pca, size = 5, decay = 1e-4, maxit = 200)
nn_predictions_pca <- predict(nn_model_pca, test_pca, type = "class")
nn_predictions_pca <- factor(nn_predictions_pca, levels = levels(y_test))
nn_accuracy_pca <- mean(nn_predictions_pca == y_test)
print(paste("Neural Network Accuracy with PCA is: ", nn_accuracy_pca))

# Compare Model Accuracies with PCA

# Collect accuracies and model names
acc_pca <- c(dt_accuracy_pca, rf_accuracy_pca, svm_accuracy_pca, knn_accuracy_pca, nn_accuracy_pca)
model_pca <- c("Decision Tree", "Random Forest", "SVM", "k-NN", "Neural Network")

# Create a data frame from the accuracy and model vectors
accuracy_data_pca <- data.frame(Model = model_pca, Accuracy = acc_pca)

# Create the bar plot for accuracy comparison with PCA
accuracy_plot_pca <- ggplot(accuracy_data_pca, aes(x = Accuracy, y = Model, fill = Model)) +
  geom_bar(stat = "identity") +
  scale_fill_viridis_d() +
  theme_minimal() +
  labs(title = "Accuracy Comparison with PCA", x = "Accuracy", y = "ML Algorithms") +
  theme(
    plot.title = element_text(hjust = 0.5, size = 20, face = "bold"),
    axis.title.x = element_text(size = 14),
    axis.title.y = element_text(size = 14),
    axis.text.x = element_text(size = 12),
    axis.text.y = element_text(size = 12),
    legend.position = "none"
  )

# Save the plot
ggsave("accuracy_comparison_pca_plot.png", plot = accuracy_plot_pca, width = 14, height = 7, dpi = 300)

# Print the plot
print(accuracy_plot_pca)

#SVM (Support Vector Machine): Achieves the highest accuracy among the tested models.
#Random Forest: Performs slightly worse than SVM but still maintains high accuracy.
#Neural Network: Also shows strong performance, close to Random Forest.
#k-NN (k-Nearest Neighbors): Has good accuracy but is slightly lower than Neural Network.
#Decision Tree: Shows the lowest accuracy among the compared models.


#PCA helps in reducing the dimensionality of the data, which can simplify the model and speed up the computation while maintaining a high level of accuracy.
#Different models respond differently to PCA. While SVM shows the best performance, Decision Tree lags behind, suggesting that the nature of the model
#and its ability to capture variance from principal components varies.
#Choosing the right model and preprocessing techniques, like PCA, can significantly impact the performance of machine learning applications.
#This analysis and plot help identify which machine learning models are most effective after dimensionality reduction, aiding in making 
#informed decisions for further model development and deployment.


# Compare Performance Before and After PCA
# Collect accuracies for both before and after PCA
accuracy_comparison <- data.frame(
  Model = rep(c("Decision Tree", "Random Forest", "SVM", "k-NN", "Neural Network"), each = 2),
  Accuracy = c(dt_accuracy, rf_accuracy, svm_accuracy, knn_accuracy, nn_accuracy,
               dt_accuracy_pca, rf_accuracy_pca, svm_accuracy_pca, knn_accuracy_pca, nn_accuracy_pca),
  Type = rep(c("Original", "PCA"), times = 5)
)

# Create the bar plot for performance comparison before and after PCA
accuracy_comparison_plot <- ggplot(accuracy_comparison, aes(x = Model, y = Accuracy, fill = Type)) +
  geom_bar(stat = "identity", position = position_dodge(width = 0.4)) +
  scale_fill_manual(values = c("midnightblue", "mediumaquamarine")) +
  theme_minimal() +
  labs(title = "Model Performance Comparison: Original vs PCA", x = "ML Algorithms", y = "Accuracy") +
  theme(
    plot.title = element_text(hjust = 0.5, size = 20, face = "bold"),
    axis.title.x = element_text(size = 14),
    axis.title.y = element_text(size = 14),
    axis.text.x = element_text(size = 12),
    axis.text.y = element_text(size = 12)
  )

# Save the plot
ggsave("model_performance_comparison_plot.png", plot = accuracy_comparison_plot, width = 20, height = 7, dpi = 300)

# Print the plot
print(accuracy_comparison_plot)