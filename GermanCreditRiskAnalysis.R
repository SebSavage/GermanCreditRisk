#Assignment 2 Data Analytics

# set working directory to desktop and clean environment
setwd("~/Desktop")
rm(list = ls())
options(digits=4)


#install any packages
install.packages("readr")
library(readr)
install.packages("rpart")
library(rpart)
install.packages("ggplot2")
library(ggplot2)
install.packages("tree")
library(tree)
install.packages("e1071")
library(e1071)
install.packages(("ROCR"))
library(ROCR)
install.packages("randomForest")
library(randomForest)
install.packages("adabag")
library(adabag)
library(plyr)

GCD <- GCD2018

#summary of data
count(GCD, "Class")
count(GCD, "Job")
count(GCD, "History")
summary(GCD)

#clean data
GCD = na.omit(GCD)
GCD$Class = as.factor(GCD$Class)

#Part 2
set.seed(25960253) #random seed
train.row = sample(1:nrow(GCD), 0.7*nrow(GCD))
GCD.train = GCD[train.row,]
GCD.test = GCD[-train.row,]

#Part 3/4/5/6/7
# Create a decision tree
library(tree)
GCD.tree = tree(as.factor(Class)~.,data=GCD.train)
plot(GCD.tree)
text(GCD.tree,pretty=0) 


# do predictions as Classes and draw a table
GCD.predtree = predict(GCD.tree, GCD.test, type = "class")
t1=table(Predicted_Class = GCD.predtree, Actual_Class = GCD.test$Class)
cat("\n#Decision Tree Confusion \n")
print(t1)


# do predictions as probabilities and draw ROC
library(ROCR)
GCD.pred.tree = predict(GCD.tree, GCD.test, type = "vector")
# computing a simple ROC curve (x-axis: fpr, y-axis: tpr)
# labels are actual values, predictors are probability of Class
GCDpred <- prediction(GCD.pred.tree[,2], GCD.test$Class)
GCDperf <- performance(GCDpred,"tpr","fpr")
plot(GCDperf)
abline(0,1)

#AUC
cauc=performance(GCDpred, "auc")
print(as.numeric(cauc@y.values))

# Calculate naive bayes
library(e1071) 
GCD.bayes = naiveBayes(Class~. , data = GCD.train)
GCD.predbayes = predict(GCD.bayes, GCD.test)
t2=table(Predicted_Class = GCD.predbayes, Actual_Class = GCD.test$Class)
cat("\n#NaiveBayes Confusion\n")
print(t2)


GCDpred.bayes = predict(GCD.bayes, GCD.test, type = 'raw')
GCDBpred <- prediction(GCDpred.bayes[,2], GCD.test$Class)
GCDBperf <- performance(GCDBpred,"tpr","fpr")
plot(GCDBperf, add=TRUE, col = "pink")

#AUC
cauc=performance(GCDBpred, "auc")
print(as.numeric(cauc@y.values))

#bagging
library(adabag)
GCD.train$Class=as.factor(GCD.train$Class)
GCD.bag = bagging(Class ~., data=GCD.train, mfinal=5)
GCDpred.bag <- predict.bagging(GCD.bag, GCD.test)
# GCDpred.bag
GCDBagpred <- prediction(GCDpred.bag$prob[,2], GCD.test$Class)
GCDBagperf <- performance(GCDBagpred,"tpr","fpr")
plot(GCDBagperf, add=TRUE, col = "blue")
cat("\n#Bagging Confusion\n")
print(GCDpred.bag$confusion)

#AUC
cauc=performance(GCDBagpred, "auc")
print(as.numeric(cauc@y.values))

#boosting
library(rpart)
#Boosting
GCD.Boost <- boosting(Class ~. , data = GCD.train, mfinal=10)
GCDpred.boost <- predict.boosting(GCD.Boost, newdata=GCD.test)
# GCDpred.boost
GCDBoostpred <- prediction(GCDpred.boost$prob[,2], GCD.test$Class)
GCDBoostperf <- performance(GCDBoostpred,"tpr","fpr")
plot(GCDBoostperf, add=TRUE, col = "red")
cat("\n#Boosting Confusion\n")
print(GCDpred.boost$confusion)

#AUC
cauc=performance(GCDBoostpred, "auc")
print(as.numeric(cauc@y.values))


#randomForest
library(randomForest)
GCD.rf = randomForest(Class~.,data=GCD.train, na.action=na.exclude)
GCDpredrf = predict(GCD.rf, GCD.test)
t3=table(Predicted_Class = GCDpredrf, Actual_Class = GCD.test$Class)
cat("\n#Random Forest Confusion\n")
print(t3)

GCDpred.rf = predict(GCD.rf, GCD.test, type="prob")
# GCDpred.rf
GCDpred = prediction(GCDpred.rf[,2], GCD.test$Class)
GCDperf = performance(GCDpred,"tpr","fpr")
plot(GCDperf, add=TRUE, col = "darkgreen")

#AUC
cauc=performance(GCDpred, "auc")
print(as.numeric(cauc@y.values))


#Attribute importance
cat("\n#Decision Tree Attribute Importance\n")
print(summary(GCD.tree))
cat("\n#Bagging Attribute Importance\n")
print(GCD.bag$importance)
cat("\n#Boosting Attribute Importance\n")
print(GCD.Boost$importance)
cat("\n#Random Forest Attribute Importance\n")
print(GCD.rf$importance)





# Calculate a decision tree
library(tree)
GCD.tree = tree(as.factor(Class)~.,data=GCD.train)
plot(GCD.tree)
text(GCD.tree,pretty=0) 


# do predictions as Classes and draw a table
GCD.predtree = predict(GCD.tree, GCD.test, type = "class")
t1=table(Predicted_Class = GCD.predtree, Actual_Class = GCD.test$Class)
cat("\n#Decision Tree Confusion \n")
print(t1)

#Cross validation test at different tree sizes:
testGCDtree=cv.tree(GCD.tree,	FUN	=	prune.misclass)
testGCDtree

#pruning the tree
prune.GCDtree =	prune.misclass(GCD.tree,	best	=	3)
summary(prune.GCDtree)
plot(prune.GCDtree)
text(prune.GCDtree, pretty=0)
prune.predictree =	predict(prune.GCDtree,	GCD.test,	type	=	"class")
table(prune.predictree,	GCD.test$Class)

# do predictions as probabilities and draw ROC
library(ROCR)
GCD.prune.tree = predict(prune.GCDtree, GCD.test, type = "vector")
# computing a simple ROC curve (x-axis: fpr, y-axis: tpr)
# labels are actual values, predictors are probability of Class
GCDpred <- prediction(GCD.prune.tree[,2], GCD.test$Class)
GCDperf <- performance(GCDpred,"tpr","fpr")
plot(GCDperf)
abline(0,1)

#AUC
cauc=performance(GCDpred, "auc")
print(as.numeric(cauc@y.values))


#ANN
rm(list = ls())
install.packages("neuralnet")
install.packages("car")
library(neuralnet)
library(car)
GCD <- GCD2018
View(GCD)
GCD = na.omit(GCD)
# convert Class to a numerical form
GCD$Class = as.numeric(GCD$Class)
# make training and test sets
set.seed(25960253)
ind <- sample(2, nrow(GCD), replace = TRUE, prob=c(0.8, 0.2))
GCD.train = GCD[ind == 1,]
GCD.test = GCD[!ind == 1,]

GCD.nn = neuralnet(Class~ Duration+Credit+Age, GCD.train, hidden=3)
plot(GCD.nn)

GCD.pred = compute(GCD.nn, GCD.test[c(2, 5, 13 )])

# now round these down to integers
GCD.pred = as.data.frame(round(GCD.pred$net.result,0))

# plot confusion matrix
cat("\n#Artificial Neural Network\n")
table(observed = GCD.test$Class, predicted = GCD.pred$V1)

