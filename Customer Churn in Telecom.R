#required libraries
library(funModeling) 
library(tidyverse) 
library(Hmisc)
library(dplyr) # for data manipulation
library(plotrix)
library(car)
library(plyr)
library(corrplot)
library(smbinning)
library(Information)
library(InformationValue)
library(pROC) # for AUC calculations
library(DMwR) # for smote implementation
library(ggplot2)
library(dominanceanalysis)
library(caret) # for model-building
library(purrr) # for functional programming (map)
install.packages("e1071")
library(reshape2)
library(tidyverse)
suppressMessages(library(caret))
require(tree)
library(MLeval)
?tree

#reading data into data frame
df_telecom = read.csv("Telco-Customer-Churn.csv")

#### Data Overview ####

#No. of observations and attributes and the head data for each attribute
glimpse(df_telecom)  # No. of observations(rows) : 7043 #No. of attributes(columns) : 21

#Datatypes of different columns of the dataframe
sapply(df_telecom,class)

#check the column with NA values
sapply(df_telecom, function(x) sum(is.na(x))) #totalCharges has 11 na values

#### Data Manipulation ####

#dropping NA values because the tenure is also 0 for those customers
df_telecom = df_telecom[complete.cases(df_telecom), ]

#Converting Senior Citizen Column values to Yes and No
df_telecom$SeniorCitizen <- ifelse(df_telecom$SeniorCitizen==1,"Yes","No")

#Converting Tenure to categorical column
df_telecom %>%
  mutate(tenure_year = case_when(tenure <= 12 ~ "0-1 year",
                                 tenure > 12 & tenure <= 24 ~ "1-2 years",
                                 tenure > 24 & tenure <= 36 ~ "2-3 years",
                                 tenure > 36 & tenure <= 48 ~ "3-4 years",
                                 tenure > 48 & tenure <= 60 ~ "4-5 years",
                                 tenure > 60 & tenure <= 72 ~ "5-6 years")) -> df_telecom

#converting column value "No Internet Service" to "No" for columns OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, StreamingTV, StreamingMovies
cols_r <- c(10:15)
for (i in 1:ncol(df_telecom[,cols_r]))
{
  df_telecom[,cols_r][,i] <- as.factor(mapvalues(df_telecom[,cols_r][,i],
                                                 from=c("No internet service"),to=c("No")))
}

#converting column value "No phone Service" to "No" for column MultipleLines
df_telecom$MultipleLines <- as.character(df_telecom$MultipleLines)
df_telecom$MultipleLines[df_telecom$MultipleLines == "No phone service"] <- "No"
df_telecom$MultipleLines <- as.factor(df_telecom$MultipleLines)

#### Exploratory Data Analysis ####

#Percentage of Churned Customers in the data
value_c <- df_telecom %>% group_by(df_telecom$Churn) %>% summarise(val = n())
labels_c <- unique(df_telecom$Churn)
cols = c("#1E8449","#F39C12")
pct <- round(value_c$val/sum(value_c$val)*100)
lbls <- paste(labels_c, pct) 
lbls <- paste(pct,"%",sep="") 
pie(value_c$val,labels = lbls, main='Customer Churn Statistics',col = cols)
legend("topright", c("Yes","No"), cex = 0.8, fill = cols)

#Monthly Charges, Total Charges and Tenure in Customer Churn
#taking data of churn customers
m.charge <- df_telecom$MonthlyCharges
t.charge <- df_telecom$TotalCharges
t.tenure <- df_telecom$tenure
scatter3d(x = m.charge, y = t.charge, z = t.tenure , 
          groups = df_telecom$Churn,ellipsoid = TRUE,surface = FALSE,grid = FALSE,
          surface.col = c("#999999", "#E69F00"),axis.scales = FALSE,
          main="Monthly Charges, Total Charges and Tenure in Customer Churn")

###Correlation matrix and WOE for categorical variables

#numerical data
df_telecom.cor <- cor(df_telecom[sapply(df_telecom, is.numeric)], method = c("spearman"))
corrplot(df_telecom.cor, type = "upper", order = "hclust",
         tl.col = "black", tl.srt = 90,main = "Correlation Matrix of Numeric Data")
#from the above correlation matrix we see that Total Charge and Monthly Charge is highly correlated

#Model Building

id_col = df_telecom$customerID
target_col = df_telecom$Churn

#Basic Model ---1
set.seed(100)
#splitting data into test and train data
train_ind <- sample(seq_len(nrow(df_telecom)), size = floor(0.7 * nrow(df_telecom)))
train <- df_telecom[train_ind,-c(1,20)]  #training dataset
test <- df_telecom[-train_ind,-c(1,20)]  #test dataset

#Building Logit Model and Predicting
#fitting the model
logitMod <- glm(Churn ~ ., family=binomial(link="logit"),data=train)
summary(logitMod)

#prediction
predicted <- predict(logitMod, newdata = test, type = 'response')

round.predict = ifelse(predicted<0.6, 'No', 'Yes') ##0.6 gives best results. Type 1 error is also very low

##Confusion matrix
cm = table(test$Churn, round.predict)
cm

#Accuracy
acc = (sum(diag(cm))/sum(cm))
acc
#misclassification error
mmce <- 1 - (sum(diag(cm))/sum(cm))
mmce

#ROC (Receiver Operating Characteristics)
test_roc = roc(test$Churn ~ predicted, plot = TRUE, print.auc = TRUE)
as.numeric(test_roc$auc) #area under the curve

#Exploring predictors importance in the logistic regression
anova(logitMod,test = "Chisq") #predictors having (p < 0.05) are statistically significant predictors 

#determining the relative importance of predictors 
#In logistic regressions, several analogues of R2 were proposed as measures of model fit, but only four were considered according to three criteria
#McFadden (r2.m), Cox and Snell (r2.cs), Nagelkerke (r2.n), and Estrella (r2.e)
da.glm.fit()("names") 

#performing dominance analysis
da_telecom <-dominanceAnalysis(logitMod)
getFits(da_telecom,"r2.m")
dominanceMatrix(da_telecom, type="complete",fit.functions = "r2.m", ordered=TRUE) #summarized results for complete dominance

#also using caret used for feature importance
gbmImp <- varImp(logitMod, scale = FALSE)

##################################### CREATING MODELS USING CARET WITHOUT SMOTE
df_telecom_b = data.frame(df_telecom[,-c(1,20)])
df_telecom_b$SeniorCitizen <- as.factor(df_telecom_b$SeniorCitizen)
df_telecom_b$tenure_year <- as.factor(df_telecom_b$tenure_year)
df_telecom_b$Churn <- as.factor(df_telecom_b$Churn)
       
#K-fold cross-validation ---1 (Ensemble Bagging Algorithm)
# Define training control
set.seed(123) 
train.control <- trainControl(method = "cv", number = 10,
                              savePredictions = TRUE,classProbs = TRUE)
model_treebag <- train(Churn ~., data = df_telecom_b, method = "treebag",
                       trControl = train.control,metric = "Accuracy")
print(model_treebag)  

#With ROC as the metric
train.control <- trainControl(method = "cv", number = 10,summaryFunction = twoClassSummary,
                              savePredictions = TRUE,classProbs = TRUE)

model_treebag <- train(Churn ~., data = df_telecom_b, method = "treebag",
                       trControl = train.control,metric = "ROC")
print(model_treebag)  


#K-fold cross-validation ---2 (Logistic Regression Algorithm)
# Define training control
set.seed(123) 
train.control <- trainControl(method = "cv", number = 10,
                              savePredictions = TRUE,classProbs = TRUE)
model_glm <- train(Churn ~., data = df_telecom_b, method = "glm", family = "binomial",
                   trControl = train.control,metric = "Accuracy")
print(model_glm)  

#With ROC as the metric
train.control <- trainControl(method = "cv", number = 10,summaryFunction = twoClassSummary,
                              savePredictions = TRUE,classProbs = TRUE)

model_glm <- train(Churn ~., data = df_telecom_b, method = "glm", family = "binomial",
                   trControl = train.control,metric = "ROC")
print(model_glm)  


#K-fold cross-validation ---3 (Random Forest Algorithm)
set.seed(123) 
train.control <- trainControl(method = "cv", number = 10,
                              savePredictions = TRUE,classProbs = TRUE)
model_rf <- train(Churn ~., data = df_telecom_b, method = "rf",
                  trControl = train.control, metric = "Accuracy")
print(model_rf) 

#With ROC as the metric
train.control <- trainControl(method = "cv", number = 10,summaryFunction = twoClassSummary,
                              savePredictions = TRUE,classProbs = TRUE)

model_rf <- train(Churn ~., data = df_telecom_b, method = "rf",
                  trControl = train.control,metric = "ROC")
print(model_rf)  

#Confusion Matrix
confusionMatrix(model_rf)


#K-fold cross-validation ---4 (C5.0 Decision Trees)
set.seed(123) 
train.control <- trainControl(method = "cv", number = 10,
                              savePredictions = TRUE,classProbs = TRUE)
model_DT <- train(Churn ~., data = df_telecom_b, method = "C5.0",
                  trControl = train.control, metric = "Accuracy")
#With ROC as the metric
train.control <- trainControl(method = "cv", number = 10,summaryFunction = twoClassSummary,
                              savePredictions = TRUE,classProbs = TRUE)

model_DT <- train(Churn ~., data = df_telecom_b, method = "C5.0",
                  trControl = train.control,metric = "ROC")
print(model_DT)  

#Printing the ROC curves for the various models
res <- evalm(list(model_treebag, model_glm, model_rf, model_DT),gnames= c('treebag', 'logistic','random forest','C5.0 Decision Trees'))


###################################### END OF CREATING MODELS IN CARET WITHOUT SMOTE
 
###################################### CREATING MODELS IN CARET USING SMOTE      
#as the data is highly skewed and there is a data imbalance we would sample the data to improve performance on imbalanced data
print(table(df_telecom$Churn)) #No - 5163  Yes - 1869 
print(prop.table(table(df_telecom$Churn))) #No - 0.734215     Yes - 0.265785 

#Applying SMOTE
df_telecom_c = data.frame(df_telecom[,-c(1,20)])
df_telecom_c$SeniorCitizen <- as.factor(df_telecom_c$SeniorCitizen)
df_telecom_c$tenure_year <- as.factor(df_telecom_c$tenure_year)
df_telecom_c$Churn <- as.factor(df_telecom_c$Churn)
df_telecom_c <- SMOTE(Churn ~ ., df_telecom_c, perc.over = 100, perc.under=200)
prop.table(table(df_telecom_c$Churn))

#K-fold cross-validation ---1 (Ensemble Bagging Algorithm)
# Define training control
set.seed(123)
#Accuracy as metric       
train.control <- trainControl(method = "cv", number = 10,
                              savePredictions = TRUE,classProbs = TRUE)
model_treebag <- train(Churn ~., data = df_telecom_c, method = "treebag",
               trControl = train.control,metric = "Accuracy")
print(model_treebag)        

#ROC as metric      
train.control <- trainControl(method = "cv", number = 10,
                              repeats = 3,savePredictions = TRUE,summaryFunction = twoClassSummary,classProbs = TRUE)
model_treebag <- train(Churn ~., data = df_telecom_c, method = "treebag",
               trControl = train.control,metric = "ROC")
print(model_treebag) 


#Confusion Matrix
confusionMatrix(model_treebag)


#K-fold cross-validation ---2 (Logistic Regression Algorithm)
# Define training control
set.seed(123) 
#With Accuracy as the metric       
train.control <- trainControl(method = "cv", number = 10,
                              savePredictions = TRUE,classProbs = TRUE)
model_glm <- train(Churn ~., data = df_telecom_c, method = "glm", family = "binomial",
               trControl = train.control,metric = "Accuracy")
print(model_glm) 

#With ROC as the metric
train.control <- trainControl(method = "cv", number = 10,summaryFunction = twoClassSummary,
                              savePredictions = TRUE,classProbs = TRUE)

model_glm <- train(Churn ~., data = df_telecom_c, method = "glm", family = "binomial",
               trControl = train.control,metric = "ROC")
print(model_glm)  
       

#K-fold cross-validation ---3 (Random Forest Algorithm)
set.seed(123)
#With Accuracy as the metric      
train.control <- trainControl(method = "cv", number = 10,
                              savePredictions = TRUE,classProbs = TRUE)
model_rf <- train(Churn ~., data = df_telecom_c, method = "rf",
                  trControl = train.control, metric = "Accuracy")
print(model_rf)

#With ROC as the metric
train.control <- trainControl(method = "cv", number = 10,summaryFunction = twoClassSummary,
                              savePredictions = TRUE,classProbs = TRUE)       
model_rf <- train(Churn ~., data = df_telecom_c, method = "rf",
               trControl = train.control,metric = "ROC")
print(model_rf) 

#Confusion Matrix
confusionMatrix(model_rf)

#K-fold cross-validation ---4 (C5.0 Decision Trees)
set.seed(123) 
#With Accuraccy as the metric       
train.control <- trainControl(method = "cv", number = 10,
                              savePredictions = TRUE,classProbs = TRUE)
model_DT <- train(Churn ~., data = df_telecom_c, method = "C5.0",
                  trControl = train.control, metric = "Accuracy")
print(model_DT)

#With ROC as the metric
train.control <- trainControl(method = "cv", number = 10,summaryFunction = twoClassSummary,
                              savePredictions = TRUE,classProbs = TRUE)

model_DT <- train(Churn ~., data = df_telecom_c, method = "C5.0",
                  trControl = train.control,metric = "ROC")
print(model_DT)  

###################################### END OF CREATING MODELS IN CARET USING SMOTE             
       
#Printing the ROC Curves
library(MLeval)
res <- evalm(list(model_treebag, model_glm, model_rf, model_DT),gnames= c('treebag', 'logistic','random forest','C5.0 Decision Trees'))
res$

#Trying another model       
#Decision Tree Algorithm ---5
index = createDataPartition(y=df_telecom_c$Churn, p=0.7, list=FALSE)
train.set = df_telecom_c[index,]
test.set = df_telecom_c[-index,]
#fitting the model
churn.tree = tree(Churn ~ ., data=train.set)
summary(churn.tree)
#plotting the tree
plot(churn.tree)
text(churn.tree, pretty = 0)
#prediction
churn.pred = predict(churn.tree, test.set, type="class")
#evaluating the error using a misclassification table
with(test.set, table(churn.pred,Churn))

#cross-validation to prune the tree optimally
cv.churn = cv.tree(churn.tree, FUN = prune.misclass)
cv.churn
plot(cv.churn)

prune.churn = prune.misclass(churn.tree, best = 4)
plot(prune.churn)
text(prune.churn, pretty=0)

churn.pred = predict(prune.churn, test.set, type="class")
with(test.set, table(churn.pred, Churn))

