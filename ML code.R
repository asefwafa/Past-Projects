##########Classification Tree models #############
# The variables being used are Contract, MonthlyCharges, InternetService, and tenure.
# The misclassification error rate is .2082

rm(list=ls())
churndata<-read.table("Lab3Data.csv",header=T,sep=",")
churndata<-churndata[,-1]
table(churndata$Churn)
names(churndata)

#Create the tree

library(tree)
treemodel<- tree(Churn~., data=churndata)
summary(treemodel)
plot(treemodel)
text(treemodel,pretty=0)

########## Bagging models #############
# The lowest error rate we can get from bagging is 0.204 when using ntree=1000 with all of the predictors

rm(list=ls())
library(randomForest)
set.seed(1)
customer = read.csv("Lab3Data.csv", header=TRUE)
customer=na.omit(customer)
customer=customer[,-1]
test = sample(1:nrow(customer),1000)
customer.train=customer[-test,]
summary(customer)
#Perform bagging
bag.customer=randomForest(Churn~.,data=customer.train,mtry=19,ntree=1000)
predict.bag=predict(bag.customer,newdata=customer[test,])
x=table(predict.bag,customer[test,"Churn"])
as.numeric(x["No",]["No"]+x["Yes",]["Yes"])/1000  # Model Accuracy of Bagging
1-as.numeric(x["No",]["No"]+x["Yes",]["Yes"])/1000   # Test Error Rate of Bagging

########## Random Forest models #############
# Error rate of 0.1963546 with mtry 4
rm(list=ls())
library(randomForest)

set.seed(11)
all_customers = read.csv("Lab3Data.csv", header=TRUE)
missing_customers = all_customers[rowSums(is.na(all_customers)) > 0,]
customers = na.exclude(all_customers)
train = sample(1:nrow(customers), nrow(customers)*0.8)
valid = customers[-train,"Churn"]

rf_customers = randomForest(Churn~.-customerID, data=customers, subset=train, mtry=4, importance=TRUE)
rf_predict = predict(rf_customers, newdata=customers[-train,])
rf_error = mean(rf_predict!=valid)
rf_error
importance(rf_customers)
varImpPlot(rf_customers)

########## Boosting models #############
# Error rate of 19.3 using shrinkage of 0.001 and number of trees at 5000
library(MASS)
library(gbm)
library(randomForest)

set.seed(1)
customer = read.csv("Lab3Data.csv", header=TRUE)
customer=na.omit(customer)
customer$Churn <- ifelse(customer$Churn=="Yes", 1, 0)
test = sample(1:nrow(customer),1000)
customer.train=customer[-test,]
customer.test = customer[test,"Churn"]
summary(customer)

boost.customer = gbm(Churn ~.-customerID,                   #formula
                     data = customer.train,    
                     #training dataset
                     distribution = 'multinomial',
                     n.trees = 5000,           
                     interaction.depth = 4,     
                     shrinkage = 0.001          
)

#number of optimal trees =2267, so we lowered number of trees in yhat.boost
#gbm.perf(boost.customer)

#### calculate test error rate
yhat.boost <- predict(boost.customer, newdata = customer[test,], n.trees = 4000,type="response") 
p.pred<- apply(yhat.boost, 1, which.max)  
yhat.pred <- ifelse(p.pred=="2", 0, 1)
x=table(yhat.pred,customer.test)
as.numeric(x["0",]["0"]+x["1",]["1"])/1000  # Model Accuracy of Boosting
1-as.numeric(x["0",]["0"]+x["1",]["1"])/1000   # Test Error Rate of Boosting


########## Conclusion #############
# I got the lowest error rate of 19.3 by using a gradient descent boosting model with an interaction depth of 5 and number of trees= 5000 and shrinkage=0.001