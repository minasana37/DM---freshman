library(dplyr) 
data<-read.csv("group_37.csv")
sum(is.na(data))

target <- data[, 1]
features <- data[, -1]
#方差
feature_variances <- apply(features, 2, sd)
boxplot(feature_variances, col = "lightblue", border = "blue")
#q1 <- quantile(feature_variances, 0.25)  
#q3 <- quantile(feature_variances, 0.75) 
#iqr <- q3 - q1
#selected_features <- features[, feature_variances > q3] 
variance_threshold <- 200
selected_features_var <- features[, feature_variances > variance_threshold]
removed_features_var <- features[, feature_variances < variance_threshold]
cat("🔹 经过方差筛选，剩余特征数:", ncol(selected_features_var), "\n")
cat("🔹 经过方差筛选，移除特征数:", ncol(removed_features_var), "\n")

filtered_data <- cbind(target, selected_features_var) 
library(randomForest) 
library(caret)   
target <- filtered_data[, 1]  
features <- filtered_data[, -1]  

set.seed(123)  
trainIndex <- createDataPartition(target, p = 0.7, list = FALSE) 
trainData <- filtered_data[trainIndex, ]  
testData <- filtered_data[-trainIndex, ]  
train_target <- target[trainIndex]  
test_target <- target[-trainIndex] 
train_features <- features[trainIndex, ] 
test_features <- features[-trainIndex, ]
pca <- prcomp(train_features, center = TRUE, scale. = TRUE)
summary(pca)
train_pca_features <- pca$x[, 1:10]
test_pca_features <- predict(pca, newdata = test_features)[, 1:10]
log_model <- glm(train_target ~ ., data = data.frame(train_pca_features), family = binomial)

log_predictions <- predict(log_model, data.frame(test_pca_features), type = "response")
log_predictions <- ifelse(log_predictions > 0.5, 1, 0)  
confusionMatrix(as.factor(log_predictions), as.factor(test_target))


