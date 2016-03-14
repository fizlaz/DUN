# ----------------------------------------------------------------------
# Word frequency
# ----------------------------------------------------------------------
#' Word frequency
#'
#' Gives word frequencies etc
#'
#' @param a String of a single word.
#' @param train Full training set.
#' @param test Full test set.
#' @return A list with following elements: freqdiff, diff, freqtrain, freqtest
#' @export
word <- function(a,train,test){
  titles <- substr(train[,2],32,nchar(train[,2])-1)
  titlestest <- substr(test[,2],32,nchar(test[,2])-1)
  f <- table(train$popularity)/length(train$popularity)
  t <- table(train$popularity[grep(a,titles)])/length(grep(a,titles))-f
  frdif <- (t[3]-t[2])*100
  l <- c(round(length(grep(a,titles)),0), round(length(grep(a,titles))/30000,4))
  l1 <- c(round(length(grep(a,titlestest)),0), round(length(grep(a,titlestest))/9644,4))
  return(list(freqdiff=round(t,5),diff=frdif,freqtrain=l,freqtest=l1))
}

# ----------------------------------------------------------------------
# DUN current RF model
# ----------------------------------------------------------------------
#' DUN random forest model
#'
#' Creates team DUN submission file
#'
#' @param train Full training set.
#' @param test Full test set.
#' @return A data frame with id and popularity columns
#' @export
full.rf <- function(train,test){

  rf <- randomForest::randomForest(train[,-c(1,2,62)], as.factor(train$popularity), ntree=1000,
                     importance=TRUE)

  submission <- data.frame(id = test$id)
  submission$popularity <- predict(rf, test[,-c(1,2)])
  write.csv(submission, file = "1_DUN_rf1k_r_gsenews.csv", row.names=FALSE)
  return(submission)
}

# ----------------------------------------------------------------------
# DUN current XGB model
# ----------------------------------------------------------------------
#' DUN xgboost model
#'
#' Creates team DUN submission file
#'
#' @param train Full training set.
#' @param test Full test set.
#' @return A data frame with id and popularity columns
#' @export
full.xgb <- function(train,test){

  param <- list("objective"="multi:softmax",
                "eval_metric"="merror",
                "num_class"=5,
                "booster"="gbtree",
                "eta"=0.03,
                "max_depth"=6,
                "subsample"=0.8,
                "colsample_bytree"=1)

  #
  y<-as.integer(train[,62])-1

  bst <- xgboost::xgboost(params = param, data = as.matrix(train[,-c(1,2,62)]),
                          label = y, nrounds = 250)

  submission <- data.frame(id = test$id)
  submission$popularity <- xgboost::predict(bst,as.matrix(test[,-c(1,2)])) +1
  write.csv(submission, file = "1_DUN_xgb_r_gsenews.csv", row.names=FALSE)
  return(submission)
}

# ----------------------------------------------------------------------
# train KNN meta predictions with folds for train
# ----------------------------------------------------------------------
#' KNN meta predictor generator
#'
#' Creates KNN meta predictions for K=(1,2,4,8,16,32,64,128,256)
#'
#' @param train Train set without ID and URL
#' @param test Test set without ID and URL. If left empty then it the
#' functions trains on train with 5 fold cross validation
#' @param csv Boolean, default FALSE. If TRUE, saves csv output.
#' @return A data frame with KNN predictors for each row
#' @export
fmeta.knn <- function(train,test=NULL,csv=FALSE){

  base::library(class) # knn

  if(length(test)==0){
    print("No test set, cross validating train set.")

    base::library(caret) # for creating folds
    set.seed(1234)
    flds <- createFolds(train$popularity,k=5) # 5 subsets for cross validation
    knns <- matrix(NA,nrow(train),9) # one column for each knn pred. label

    for(i in 1:9){
      p <- 2^(i-1) # lnn as powers of 2 -> (1,2,4,8,16,32,64,128,256)
      print(p) # for tracking progress

      for(j in 1:5){
        ktest <- train[flds[[j]],-60]
        ktrain <- train[-flds[[j]],]
        print("fold")
        print(j) # for tracking progress
        knns[ flds[[j]] ,i] <- knn(train = ktrain[,-60], test = ktest,
          ktrain[,60], k = p)
      } # end of looping cross validations
    } # end of looping each knn value
    knns <- as.data.frame(knns)
    colnames(knns) <- unlist(lapply("NN", paste0, c(1,2,4,8,16,32,64,128,256)))
    if(csv==TRUE){
      write.csv(knns,"knnsflds1.csv",row.names=FALSE)
    }
    return(knns)
  } # end of train set cross validation and meta predictions

  else{ # predicting meta for test set, training on full train set
    print("test set loaded, learning on train and predicting on test")

    NN1 <- knn(train = train[,-60], test = test, train[,60], k = 1)
    NN2 <- knn(train = train[,-60], test = test, train[,60], k = 2)
    NN4 <- knn(train = train[,-60], test = test, train[,60], k = 4)
    NN8 <- knn(train = train[,-60], test = test, train[,60], k = 8)
    NN16 <- knn(train = train[,-60], test = test, train[,60], k = 16)
    NN32 <- knn(train = train[,-60], test = test, train[,60], k = 32)
    NN64 <- knn(train = train[,-60], test = test, train[,60], k = 64)
    NN128 <- knn(train = train[,-60], test = test, train[,60], k = 128)
    NN256 <- knn(train = train[,-60], test = test, train[,60], k = 256)

    knnstest <-as.data.frame(cbind(NN1,NN2,NN4,NN8,NN16,NN32,NN64,NN128,NN256))

    if(csv==TRUE){
      write.csv(knnstest,"knnstest.csv",row.names = FALSE)
    }
    return(knnstest)
  } # end of test set meta predictions
} # fmeta.knn end

# ----------------------------------------------------------------------
# train KNN meta predictions, probability output
# ----------------------------------------------------------------------
#' KNN meta predictor generator, probability output
#'
#' Creates probability of highest KNN meta predictions vote
#' for K=(1,2,4,8,16,32,64,128,256)
#'
#' @param train Train set without ID and URL
#' @param test Test set without ID and URL. If left empty then it the
#' functions trains on train with 5 fold cross validation
#' @param csv Boolean, default FALSE. If TRUE, saves csv output.
#' @return A data frame with KNN probability of highest vote
#' @export
fmeta.knnprob <- function(train,test=NULL,csv=FALSE){#fix test part

  base::library(class) # knn

  if(length(test)==0){
    print("No test set, cross validating train set.")

    base::library(caret) # for creating folds
    set.seed(1234)
    flds <- createFolds(train$popularity,k=5) # 5 subsets for cross validation
    knns <- matrix(NA,nrow(train),9) # one column for each knn pred. label

    for(i in 1:9){
      p <- 2^(i-1) # lnn as powers of 2 -> (1,2,4,8,16,32,64,128,256)
      print("knn")
      print(p) # for tracking progress

      for(j in 1:5){
        ktest <- train[flds[[j]],-60]
        ktrain <- train[-flds[[j]],]
        print("fold")
        print(j) # for tracking progress
        knnp <- knn(train = ktrain[,-60], test = ktest,
          ktrain[,60], k = p, prob = TRUE)
        knns[ flds[[j]] ,i] <- attr(knnp,"prob")
      } # end of looping cross validations
    } # end of looping each knn value
    knns <- as.data.frame(knns)
    colnames(knns) <- unlist(lapply("pNN", paste0, c(1,2,4,8,16,32,64,128,256)))
    if(csv==TRUE){
      write.csv(knns,"knnsfldsprob1.csv",row.names=FALSE)
    }
    return(knns)
  } # end of train set cross validation and meta predictions

  else{ # predicting meta for test set, training on full train set
    print("test set loaded, learning on train and predicting on test")

    print("NN1")
    NN1 <- knn(train = train[,-60], test = test, train[,60], k = 1,prob=TRUE)
    pNN1 <- attr(NN1,"prob")
    print("NN2")
    NN2 <- knn(train = train[,-60], test = test, train[,60], k = 2,prob=TRUE)
    pNN2 <- attr(NN2,"prob")
    print("NN4")
    NN4 <- knn(train = train[,-60], test = test, train[,60], k = 4,prob=TRUE)
    pNN4 <- attr(NN4,"prob")
    print("NN8")
    NN8 <- knn(train = train[,-60], test = test, train[,60], k = 8,prob=TRUE)
    pNN8 <- attr(NN8,"prob")
    print("NN16")
    NN16 <- knn(train = train[,-60], test = test, train[,60], k = 16,prob=TRUE)
    pNN16 <- attr(NN16,"prob")
    print("NN32")
    NN32 <- knn(train = train[,-60], test = test, train[,60], k = 32,prob=TRUE)
    pNN32 <- attr(NN32,"prob")
    print("NN64")
    NN64 <- knn(train = train[,-60], test = test, train[,60], k = 64,prob=TRUE)
    pNN64 <- attr(NN64,"prob")
    print("NN128")
    NN128 <-knn(train = train[,-60], test = test, train[,60], k = 128,prob=TRUE)
    pNN128 <- attr(NN128,"prob")
    print("NN256")
    NN256 <-knn(train = train[,-60], test = test, train[,60], k = 256,prob=TRUE)
    pNN256 <- attr(NN256,"prob")

    knnstest <-as.data.frame(
                  cbind(pNN1,pNN2,pNN4,pNN8,pNN16,pNN32,pNN64,pNN128,pNN256))

    if(csv==TRUE){
      write.csv(knnstest,"knnstestprob.csv",row.names = FALSE)
    }
    return(knnstest)
  } # end of test set meta predictions
} # fmeta.knn end

# ----------------------------------------------------------------------
# train random forest meta predictions with folds for train
# ----------------------------------------------------------------------
#' Random Forest meta predictor
#'
#' Creates probabilities for each class
#'
#' @param train Train set without ID and URL
#' @param test Test set without ID and URL. If left empty then it the
#' functions trains on train with 5 fold cross validation
#' @param csv Boolean, default FALSE. If TRUE, saves csv output.
#' @return A data frame with 5 probabilites for each row
#' @export
fmeta.rf <- function(train,test=NULL,csv=FALSE){

  base::library(randomForest)

  if(length(test)==0){
    print("No test set, cross validating train set.")
    base::library(caret)
    set.seed(1234)
    flds <- createFolds(train$popularity,k=5)
    rfpred <- matrix(NA,nrow(train),6)

    for(j in 1:5){
      ktest <- train[flds[[j]],-60]
      ktrain <- train[-flds[[j]],]

      print("fold")
      print(j) # tracing progress

      set.seed(1234)
      rf <- randomForest(ktrain[,-60], as.factor(ktrain$popularity), ntree=800,
                         importance=TRUE,mtry = 14,do.trace=TRUE)

      rfpred[ flds[[j]],6] <- predict(rf, ktest)
      rfpred[ flds[[j]],1:5] <- predict(rf,ktest,type = "prob")


      } # end of looping cross validations
      rfframe <- as.data.frame(rfpred)
      colnames(rfframe) <- c("rfp1","rfp2","rfp3","rfp4","rfp5","rflabel")

      if(csv==TRUE){
        write.csv(rfframe,"rfmetafldsmat1.csv",row.names = FALSE)
      }
      return(rfframe)
  } # end of train set cross validation and meta predictions
  else{ # test set loaded
    print("test set loaded, learning on train and predicting on test")
    set.seed(1234)
    rf <- randomForest(train[,-60], as.factor(train$popularity), ntree=1000,
                       importance=TRUE,mtry = 14,do.trace=TRUE)

    rfmetapred <- predict(rf, test)
    rfmetaprob <- predict(rf, test, type = "prob")
    rfframe <- data.frame(rfmetaprob,rfmetapred)
    colnames(rfframe) <- c("rfp1","rfp2","rfp3","rfp4","rfp5","rflabel")

    if(csv==TRUE){
      write.csv(rfframe, file = "rfmetatestmat.csv", row.names=FALSE)
    }

    return(rfframe)
  } # end of test set meta
} # end of fmeta.rf

# ----------------------------------------------------------------------
# train xgboost meta predictions with folds for train
# ----------------------------------------------------------------------
#' xgboost meta predictor
#'
#' Creates probabilities for each class
#'
#' @param train Train set without ID and URL
#' @param test Test set without ID and URL. If left empty then it the
#' functions trains on train with 5 fold cross validation
#' @param folds Number of cross validation folds
#' @param csv Boolean, default FALSE. If TRUE, saves csv output.
#' @return A data frame with 5 probabilites for each row
#' @export
fmeta.xgb <- function(train,test=NULL,folds=5,csv=FALSE){

  base::library(xgboost)

  param <- list("objective"="multi:softprob",
                "eval_metric"="mlogloss",
                "num_class"=5,
                "booster"="gbtree",
                "eta"=0.03, # 0.3 default
                "max_depth"=7,
                "subsample"=0.8,
                "colsample_bytree"=0.8)
 #new param
 #  param <- list("objective"="multi:softprob",
 #                "eval_metric"="merror",
 #                "num_class"=5,
 #                "booster"="gbtree",
 #                "eta"=0.03, # 0.3 default
 #                "max_depth"=6,
 #                "subsample"=0.8,
 #                "colsample_bytree"=0.8)

  y<-as.integer(train[,60])-1 #xgb label needs to start at 0

  if(length(test)==0){
    print("No test set, cross validating train set.")

    base::library(caret)
    set.seed(1234)
    flds <- createFolds(train$popularity,k=folds)

    xgbpred <- matrix(NA,nrow(train),5)

    for(j in 1:5){
      ktest <- train[flds[[j]],-60]
      ktrain <- train[-flds[[j]],]
      t <- y[-flds[[j]]]

      print("fold")
      print(j) #tracking progress

      bst <- xgboost(params = param, data = as.matrix(ktrain[,-60]),label = t,
                     nrounds = 250)

      xgbpred[ flds[[j]], ] <- predict(bst, as.matrix(ktest))

    } # end of looping cross validations
    print("fold looping complete")
    xgbpred <- as.data.frame(xgbpred)
    colnames(xgbpred) <- c("xgbp1","xgbp2","xgbp3","xgbp4","xgbp5")

    if(csv==TRUE){
      write.csv(xgbpred,"xgbmetafldsmat1.csv",row.names = FALSE)
    }

    return(xgbpred)
  } # end of train set cross validation and meta predictions
  else{ #test set loaded
    print("test set loaded, learning on train and predicting on test")

    bst <- xgboost(params = param, data = as.matrix(train[,-60]),label = y,
                   nrounds = 250)

    xgbpred <- predict(bst, as.matrix(test))
    xgbpred <- matrix(xgbpred,ncol=5,byrow=TRUE)
    xgbpred <- as.data.frame(xgbpred)
    colnames(xgbpred) <- c("xgbp1","xgbp2","xgbp3","xgbp4","xgbp5")

    if(csv==TRUE){
      write.csv(xgbpred,"xgbmetatestmat.csv",row.names = FALSE)
    }

    return(xgbpred)

  }# end of test set meta
}#end of fmeta.xgb

# ----------------------------------------------------------------------
# cross validation random forest
# ----------------------------------------------------------------------
#' cross validation random forest
#'
#' five fold cross validation using random forest
#'
#' @param train Any train set without ID and URL
#' @param tree Number of trees, default 400
#' @return A list containing a vector of each fold accuracy and a mean.
#' @export
cv.rf <- function(train,tree=400){

  base::library(randomForest)

  c <- which(colnames(train)=="popularity")

    base::library(caret)
    set.seed(1234)
    flds <- createFolds(train$popularity,k=5)
    rfpred <- rep(NA,5)

    for(j in 1:5){
      ktest <- train[flds[[j]],]
      ktrain <- train[-flds[[j]],]

      print("fold")
      print(j) # tracing progress
      print(rfpred)
      set.seed(1234)
      rf <- randomForest(ktrain[,-c], as.factor(ktrain$popularity), ntree=tree,
                         importance=TRUE, do.trace=TRUE)

      predictions <- predict(rf,ktest[,-c])
      rfsum <- confusionMatrix(predictions, ktest$popularity)
      rfpred[j] <- rfsum$overall[1]
      print(rfsum)


      } # end of looping cross validations
      avg <- mean(rfpred)

      return(list(vec=rfpred,avg=avg))

} # end of cv.rf

# ----------------------------------------------------------------------
# cross validation xgb
# ----------------------------------------------------------------------
#' cross validation xgboost
#'
#' five fold cross validation using xgboost
#'
#' @param train Any train set without ID and URL
#' @param rounds Number of rounds, default 250
#' @return A list containing a vector of each fold accuracy and a mean.
#' @export
cv.xgb <- function(train,rounds=250){

  base::library(xgboost)

  param <- list("objective"="multi:softmax",
                "eval_metric"="merror",
                "num_class"=5,
                "booster"="gbtree",
                "eta"=0.03, # 0.3 default
                #"eta"=0.01,
                "max_depth"=6,
                #"max_depth"=4,
                "subsample"=0.8,
                "colsample_bytree"=1)
                #"colsample_bytree"=0.8)

    y<-as.integer(train$popularity)-1 #xgb label needs to start at 0

    c <- which(colnames(train)=="popularity")

    base::library(caret)
    set.seed(1234)
    flds <- createFolds(train$popularity,k=5)
    xgbpred <- rep(NA,5)

    for(j in 1:5){
      ktest <- train[flds[[j]],]
      ktrain <- train[-flds[[j]],]
      t <- y[-flds[[j]]]

      print("fold")
      print(j) # tracing progress
      print(xgbpred)
      bst <- xgboost(params = param, data = as.matrix(ktrain[,-c]),label = t,
                     nrounds = rounds)

      predictions <- factor(predict(bst, as.matrix(ktest[,-c]))+1,
                      levels = c(1,2,3,4,5))
      bstsum <- confusionMatrix(predictions, ktest$popularity)

      xgbpred[j] <- bstsum$overall[1]


      } # end of looping cross validations
      avg <- mean(xgbpred)

      return(list(vec=xgbpred,avg=avg))

} # end of cv.xgb

# ----------------------------------------------------------------------
# cross validation multiclass adaboost
# ----------------------------------------------------------------------
#' cross validation maboost
#'
#' five fold cross validation using multiclass adaboost
#'
#' @param train Any train set without ID and URL
#' @param rounds Number of rounds, default 250
#' @return A list containing a vector of each fold accuracy and a mean.
#' @export
cv.mab <- function(train, rounds=250){

  base::library(maboost)

  c <- which(colnames(train)=="popularity")

    base::library(caret)
    set.seed(1234)
    flds <- createFolds(train$popularity,k=5)
    mabpred <- rep(NA,5)

    for(j in 1:5){
      ktest <- train[flds[[j]],]
      ktrain <- train[-flds[[j]],]

      print("fold")
      print(j) # tracing progress
      print(mabpred)

      mab <- maboost(ktrain[,-c],ktrain$popularity, iter=rounds)

      predictions <- predict(mab,ktest[,-c])
      mabsum <- confusionMatrix(predictions, ktest$popularity)
      print(mabsum)

      mabpred[j] <- mabsum$overall[1]

      } # end of looping cross validations
      avg <- mean(mabpred)

      return(list(vec=mabpred,avg=avg))

} # end of cv.mab
