
set.seed(1)
#sets the starting point of random numbers 
require(hash)
# requires the function hash and it will give a warning if the function is not found.
#df <- read.csv("<lungcancer>.csv")
mydata <- read.csv(url("https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data"),
                   header=FALSE)
mydata<-transform(mydata, V2 = as.numeric(V2))
mydata <- mydata[!apply(is.na(mydata) | mydata == "", 1, all),]
#------------splitting the data-----------------------
require(caTools)
set.seed(101) 
#sample = sample.split(mydata$V1, SplitRatio =0.75)
#dtrain = subset(sample, sample == TRUE)
#dtest  = subset(sample, sample == FALSE)

target<-unclass(mydata[,c(2)])
mydata<-mydata[-c(2)]
mydata<-transform(mydata, V1 = as.numeric(V1))

#mydata = read.csv("lungcancer.csv")
#mydata <- mydata[ -c(57) ]
#target <- mydata[,ncol(mydata)]
library(forecast)
library(MXM)
library(class)
library(gmodels)
a <- mmmb(target , mydata , max_k = 3 , threshold = 0.01, test= "testIndFisher", ncores = 1)
ab <- SES(target, mydata, test="testIndFisher")

#---------------------Split data------------------------------
prc_train<-mydata[1:300,]
prc_test<-mydata[301:569,]
prc_train_labels<-target[1:300]
prc_test_labels<-target[301:569]
prc_test_pred <- knn(train = prc_train, test =prc_test,cl = prc_train_labels, k=1)
CrossTable(x=prc_test_pred, y= prc_test_pred, prop.chisq  = FALSE)