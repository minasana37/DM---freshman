#packages
library(dplyr) 
library(neuralnet)
library(NeuralNetTools)
library(pROC)
library(rpart)
library(rpart.plot)
library(ROCR)
library(ggplot2)
library(randomForest)
library(MASS)
library(biotools)
library(magick)

data_train <- read.csv("group_37.csv")
data_test <- read.csv("arcene_test (for groups 32-38).csv")
sum(is.na(data_train))
sum(is.na(data_test))
ncol(data_train) == 
  sum(colnames(data_train) %in% colnames(data_test))
class <- data_train[, 1]
features <- data_train[, -1]
data_train[, 1] <- ifelse(data_train[, 1] == -1, 0, 1)

feature_sd <- apply(features, 2, sd)
par(mfrow = c(1, 2))
boxplot(feature_sd, 
        xlab = "Features", 
        ylab = "Standard Deviation", 
        col = "lightblue")
hist(feature_sd, 
     xlab = "Standard Deviation", 
     ylab = "Frequency", 
     main = " ",
     col = "lightblue", 
     border = "black", 
     breaks = 100)

q1 <- quantile(feature_sd, 0.25)  
q3 <- quantile(feature_sd, 0.75) 
sd_threshold <- q3-q1
selected_features <- features[, feature_sd > sd_threshold]
removed_features <- features[, feature_sd <= sd_threshold]
cat("remaing features", ncol(selected_features), "\n")
cat("removed features:", ncol(removed_features), "\n")

#use PCA
pca<-prcomp(selected_features)
sd.pca <- pca$sdev
#based on kaiser
tot.var<-sum(sd.pca^2)
ave.var<-tot.var/ncol(selected_features)
ave.var 
sd.pca^2 > ave.var
all_variance <- sd.pca^2
cum_variance <- cumsum(all_variance) / sum(all_variance)
selected_components <- which(sd.pca^2 > ave.var)
selected_variance <- sum(all_variance[selected_components]) / sum(all_variance)
cat("Cumulative contribution:", selected_variance, "\n")
pca_features1 <- pca$x[, 1:47]

#based on Cumulative variance contribution 95%
cumulative_variance <- cumsum(sd.pca^2) / sum(sd.pca^2)
selected_components_cumulative <- which(cumulative_variance >= 0.95)[1]
selected_components_cumulative
pca_features2 <- pca$x[, 1:selected_components_cumulative]

traindata <- data.frame(pca_features1, Class = data_train[[1]])
traindata2 <- data.frame(pca_features2, Class = data_train[[1]])
traindata3 <- data.frame(selected_features,Class = data_train[1])
data_train$Class<- as.factor(data_train$Class)
#str(traindata)
#test data process
data_test[, 1] <- ifelse(data_test[, 1] == -1, 0, 1)
selected_features_names <- colnames(selected_features)
data_test_new <- data_test[, selected_features_names]
testdata <- data.frame(
  predict(pca, newdata = data_test_new)[,1:47],
  Class = data_test[[1]])
testdata2 <- data.frame(
  predict(pca, newdata = data_test_new)[,1:18],
  Class = data_test[[1]])
testdata3 <- data.frame(data_test_new,Class = data_test[1])
#str(testdata)

boxm <- boxM(traindata2[,-ncol(traindata2)], traindata2$Class)
print(boxm)

p_values <- sapply(traindata2[, -ncol(traindata2)], function(column) shapiro.test(column)$p.value)
alpha <- 0.05
bonferroni_threshold <- alpha / length(p_values)
rejected_normality <- p_values < bonferroni_threshold
sum(rejected_normality)

###LDA
class_lda <- lda(Class~., data = traindata2)
class_pred_lda <- predict(class_lda, testdata2)

acc_lda <- mean(class_pred_lda$class == testdata2$Class)
cat("acc:", acc_lda, "\n")
table(class_pred_lda$class, testdata2$Class)

### QDA
class_qda <- qda(Class~., data = traindata2)

class_pred_qda <- predict(class_qda, testdata2)

acc_qda <- mean(class_pred_qda$class == testdata2$Class)
cat("acc:", acc_qda, "\n")
table(class_pred_qda$class, testdata2$Class)

par(mfrow = c(1, 2))
roc_lda <- roc(testdata2$Class, class_pred_lda$posterior[,2])
plot(roc_lda, main = "ROC for LDA", print.auc = T, auc.polygon = T, legacy.axes = T)

roc_qda <- roc(testdata2$Class, class_pred_qda$posterior[,2])
plot(roc_qda, main = "ROC for QDA", print.auc = T, auc.polygon = T, legacy.axes = T)

set.seed(2023)
par(mfrow = c(1, 1))
model.classification <- rpart(Class~.,
                              data=traindata,
                              method="class",
                              cp=-1,
                              minsplit=2,
                              minbucket=1)
plotcp(model.classification)

printcp(model.classification)

par(mfrow = c(1, 2))
model.classification1 <- prune(model.classification,cp=0.032)
rpart.plot(model.classification1,type = 2,extra = 4)
barplot(model.classification1$variable.importance,
        col = "lightpink",
        main  = "Variable-Importance",
        xlab = "Variables",  
        ylab = "Importance") 

y_pred_pro <- predict(model.classification1,
                      newdata = testdata,
                      type = "prob")[,2]
y_pred_pro <- unname(y_pred_pro)
y_pred_pro <- as.vector(y_pred_pro)
y_pred <- ifelse(y_pred_pro > 0.5, 1, 0)
y_true <- as.numeric(as.character(testdata$Class))
#Accuracy
accuracy <- sum(y_pred == y_true) / length(y_true)
print(paste("Accuracy:", accuracy))
#conf_matrix
conf_matrix <- table(Predicted = y_pred, Actual = y_true)
print(conf_matrix)
#AUC
score <- ROCR::prediction(y_pred_pro, y_true)
auc <- performance(score, "auc")
perf <- performance(score,"tpr","fpr")
print(paste("AUC:", auc@y.values[[1]]))

#bagging
set.seed(2023)
traindata3$Class <- as.factor(traindata3$Class)
bagging_Model <- randomForest(Class~.,
                              data=traindata3,
                              mtry=ncol(traindata3)-1,
                              ntree=1000)

par(mfrow = c(1, 2))
plot(bagging_Model$err.rate[,1],
     type = "l",
     xlab = "Number of trees",
     ylab="OOB Error",
     main="OBB Error vs. Number of Trees")
set.seed(2023)
bagging_Model <- randomForest(Class~.,data=traindata3,
                              mtry=ncol(traindata3)-1,
                              ntree=200)
varImpPlot(bagging_Model, main="Variable Importance")

y_pred_pro <- predict(bagging_Model,
                      newdata = testdata3,
                      type = "prob")[,2]
y_pred_pro <- unname(y_pred_pro)
y_pred_pro <- as.vector(y_pred_pro)
y_pred <- ifelse(y_pred_pro > 0.5, 1, 0)
y_true <- as.numeric(as.character(testdata$Class))
#Accuracy
accuracy <- sum(y_pred == y_true) / length(y_true)
print(paste("Accuracy:", accuracy))
#conf_matrix
conf_matrix <- table(Predicted = y_pred, Actual = y_true)
#AUC
score <- ROCR::prediction(y_pred_pro, y_true)
auc <- performance(score, "auc")
perf <- performance(score,"tpr","fpr")
print(paste("AUC:", auc@y.values[[1]]))

set.seed(2023)
random_Model <- randomForest(Class~.,data=traindata3,ntree=500)

par(mfrow = c(1, 2))
plot(random_Model$err.rate[,1],
     type = "l",
     xlab = "Number of trees",
     ylab="OOB Error",
     main="OBB Error vs. Number of Trees")
varImpPlot(random_Model, main="predicting class")

y_pred_pro <- predict(random_Model,
                      newdata = testdata3,
                      type = "prob")[,2]
y_pred_pro <- unname(y_pred_pro)
y_pred_pro <- as.vector(y_pred_pro)
y_pred <- ifelse(y_pred_pro > 0.5, 1, 0)
y_true <- as.numeric(as.character(testdata$Class))
#Accuracy
accuracy <- sum(y_pred == y_true) / length(y_true)
print(paste("Accuracy:", accuracy))
#conf_matrix
conf_matrix <- table(Predicted = y_pred, Actual = y_true)
#AUC
score <- ROCR::prediction(y_pred_pro, y_true)
auc <- performance(score, "auc")
perf <- performance(score,"tpr","fpr")
print(paste("AUC:", auc@y.values[[1]]))

par(mfrow = c(1, 1))
y_pred_pro1 <- predict(model.classification1,
                       newdata = testdata,
                       type = "prob")[,2]
y_pred_pro2 <- predict(bagging_Model,
                       newdata = testdata3,
                       type = "prob")[,2]
y_pred_pro3 <- predict(random_Model, 
                       newdata = testdata3, 
                       type = "prob")[, 2]
score1 <- ROCR::prediction(y_pred_pro1, y_true)
perf1 <- performance(score1,"tpr","fpr")
perfd1 <- data.frame(x=perf1@x.values[1][[1]],
                     y=perf1@y.values[1][[1]])
score2 <- ROCR::prediction(y_pred_pro2, y_true)
perf2 <- performance(score2,"tpr","fpr")
perfd2 <- data.frame(x=perf2@x.values[1][[1]],
                     y=perf2@y.values[1][[1]])
score3 <- ROCR::prediction(y_pred_pro3, y_true)
perf3 <- performance(score3,"tpr","fpr")
perfd3 <- data.frame(x=perf3@x.values[1][[1]],
                     y=perf3@y.values[1][[1]])
perfd1$model <- "classification tree"
perfd2$model <- "bagging tree"
perfd3$model <- "random forest"
roc_data <- rbind(perfd1, perfd2,perfd3)
ggplot(roc_data, aes(x = x, y = y, color = model)) +
  geom_line(size = 1.2) + 
  xlab("False Positive Rate") + 
  ylab("True Positive Rate") +
  ggtitle("ROC Curve Comparison") +
  scale_color_manual(values = c("skyblue", "pink", "darkorange")) +  
  theme_minimal() +
  theme(legend.position = "right")

set.seed(82)
nn_model1 <- neuralnet(Class ~ .,
                       data=traindata, 
                       hidden = c(20), 
                       act.fct = "logistic",
                       linear.output=FALSE)

y_pred_pro <- predict(nn_model1, 
                      newdata = testdata, 
                      type = "prob")
y_pred_pro <- unname(y_pred_pro)
y_pred_pro <- as.vector(y_pred_pro)
y_pred <- ifelse(y_pred_pro > 0.5, 1, 0)
y_true <- as.numeric(as.character(testdata$Class))

#Accuracy
accuracy <- sum(y_pred == y_true) / length(y_true)
print(paste("Accuracy:", accuracy))
#conf_matrix
conf_matrix <- table(Predicted = y_pred, Actual = y_true)
#AUC
score <- ROCR::prediction(y_pred_pro, y_true)
auc <- performance(score, "auc")
perf <- performance(score,"tpr","fpr")
print(paste("AUC:", auc@y.values[[1]]))

set.seed(83)
nn_model2 <- neuralnet(Class ~ .,
                       data=testdata3, 
                       hidden = c(533, 256, 60, 20), 
                       act.fct = "logistic",
                       linear.output=FALSE)
#plotnet(nn_model)

y_pred_pro <- predict(nn_model2, 
                      newdata = testdata3, 
                      type = "prob")
y_pred_pro <- as.vector(y_pred_pro)
y_pred <- ifelse(y_pred_pro > 0.5, 1, 0)
y_true <- as.numeric(as.character(testdata$Class))

#Accuracy
accuracy <- sum(y_pred == y_true) / length(y_true)
print(paste("Accuracy:", accuracy))
#conf_matrix
conf_matrix <- table(Predicted = y_pred, Actual = y_true)
#AUC
score <- ROCR::prediction(y_pred_pro, y_true)
auc <- performance(score, "auc")
perf <- performance(score,"tpr","fpr")
print(paste("AUC:", auc@y.values[[1]]))

par(mfrow = c(1, 1))
y_pred_pro1 <- predict(nn_model1, 
                       newdata = testdata, 
                       type = "prob")
y_pred_pro2 <- predict(nn_model2, 
                       newdata = testdata3, 
                       type = "prob")

score1 <- ROCR::prediction(y_pred_pro1, y_true)
perf1 <- performance(score1,"tpr","fpr")
perfd1 <- data.frame(x=perf1@x.values[1][[1]],
                     y=perf1@y.values[1][[1]])

score2 <- ROCR::prediction(y_pred_pro2, y_true)
perf2 <- performance(score2,"tpr","fpr")
perfd2 <- data.frame(x=perf2@x.values[1][[1]],
                     y=perf2@y.values[1][[1]])

perfd1$model <- "nn_model1"
perfd2$model <- "nn_model2"
roc_data <- rbind(perfd1, perfd2)


ggplot(roc_data, aes(x = x, y = y, color = model)) +
  geom_line() + 
  xlab("False Positive Rate") + 
  ylab("True Positive Rate") +
  ggtitle("ROC Curve Comparison") +
  scale_color_manual(values = c("blue", "red")) +  
  theme_minimal() +
  theme(legend.position = "bottom")

