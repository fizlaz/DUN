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

  c <- which(colnames(train)=="popularity")

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
        ktest <- train[flds[[j]],-c]
        ktrain <- train[-flds[[j]],]
        print("fold")
        print(j) # for tracking progress
        knns[ flds[[j]] ,i] <- knn(train = ktrain[,-c], test = ktest,
          ktrain[,c], k = p)
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

    NN1 <- knn(train = train[,-c], test = test, train[,c], k = 1)
    NN2 <- knn(train = train[,-c], test = test, train[,c], k = 2)
    NN4 <- knn(train = train[,-c], test = test, train[,c], k = 4)
    NN8 <- knn(train = train[,-c], test = test, train[,c], k = 8)
    NN16 <- knn(train = train[,-c], test = test, train[,c], k = 16)
    NN32 <- knn(train = train[,-c], test = test, train[,c], k = 32)
    NN64 <- knn(train = train[,-c], test = test, train[,c], k = 64)
    NN128 <- knn(train = train[,-c], test = test, train[,c], k = 128)
    NN256 <- knn(train = train[,-c], test = test, train[,c], k = 256)

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

  c <- which(colnames(train)=="popularity")

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
        ktest <- train[flds[[j]],-c]
        ktrain <- train[-flds[[j]],]
        print("fold")
        print(j) # for tracking progress
        knnp <- knn(train = ktrain[,-c], test = ktest,
          ktrain[,c], k = p, prob = TRUE)
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
    NN1 <- knn(train = train[,-c], test = test, train[,c], k = 1,prob=TRUE)
    pNN1 <- attr(NN1,"prob")
    print("NN2")
    NN2 <- knn(train = train[,-c], test = test, train[,c], k = 2,prob=TRUE)
    pNN2 <- attr(NN2,"prob")
    print("NN4")
    NN4 <- knn(train = train[,-c], test = test, train[,c], k = 4,prob=TRUE)
    pNN4 <- attr(NN4,"prob")
    print("NN8")
    NN8 <- knn(train = train[,-c], test = test, train[,c], k = 8,prob=TRUE)
    pNN8 <- attr(NN8,"prob")
    print("NN16")
    NN16 <- knn(train = train[,-c], test = test, train[,c], k = 16,prob=TRUE)
    pNN16 <- attr(NN16,"prob")
    print("NN32")
    NN32 <- knn(train = train[,-c], test = test, train[,c], k = 32,prob=TRUE)
    pNN32 <- attr(NN32,"prob")
    print("NN64")
    NN64 <- knn(train = train[,-c], test = test, train[,c], k = 64,prob=TRUE)
    pNN64 <- attr(NN64,"prob")
    print("NN128")
    NN128 <-knn(train = train[,-c], test = test, train[,c], k = 128,prob=TRUE)
    pNN128 <- attr(NN128,"prob")
    print("NN256")
    NN256 <-knn(train = train[,-c], test = test, train[,c], k = 256,prob=TRUE)
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

  c <- which(colnames(train)=="popularity")

  if(length(test)==0){
    print("No test set, cross validating train set.")
    base::library(caret)
    set.seed(1234)
    flds <- createFolds(train$popularity,k=5)
    rfpred <- matrix(NA,nrow(train),6)

    for(j in 1:5){
      ktest <- train[flds[[j]],-c]
      ktrain <- train[-flds[[j]],]

      print("fold")
      print(j) # tracing progress

      set.seed(1234)
      rf <- randomForest(ktrain[,-c], as.factor(ktrain$popularity), ntree=800,
                         importance=TRUE, do.trace=TRUE)

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
    rf <- randomForest(train[,-c], as.factor(train$popularity), ntree=1000,
                       importance=TRUE, do.trace=TRUE)

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
                "max_depth"=6,
                "subsample"=0.8,
                "colsample_bytree"=0.8)
 #new param
 #  param <- list("objective"="multi:softprob",
 #                "eval_metric"="merror",
 #                "num_class"=5,
 #                "booster"="gbtree",
 #                "eta"=0.01, # 0.3 default
 #                "max_depth"=6,
 #                "subsample"=0.8,
 #                "colsample_bytree"=0.6)

 c <- which(colnames(train)=="popularity")

 y<-as.integer(train[,c])-1 #xgb label needs to start at 0

  if(length(test)==0){
    print("No test set, cross validating train set.")

    base::library(caret)
    set.seed(1234)
    flds <- createFolds(train$popularity,k=folds)

    xgbpred <- matrix(NA,nrow(train),5)

    for(j in 1:5){
      ktest <- train[flds[[j]],-c]
      ktrain <- train[-flds[[j]],]
      t <- y[-flds[[j]]]

      print("fold")
      print(j) #tracking progress

      bst <- xgboost(params = param, data = as.matrix(ktrain[,-c]),label = t,
                     nrounds = 250)#200

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

    bst <- xgboost(params = param, data = as.matrix(train[,-c]),label = y,
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
# train adaboost meta predictions with folds for train
# ----------------------------------------------------------------------
#' adaoost meta predictor (maboost package)
#'
#' Creates probabilities for each class
#'
#' @param train Train set without ID and URL
#' @param test Test set without ID and URL. If left empty then it the
#' functions trains on train with 5 fold cross validation
#' @param rounds Number of rounds per folds
#' @param csv Boolean, default FALSE. If TRUE, saves csv output.
#' @return A data frame with 5 probabilites for each row
#' @export
fmeta.mab <- function(train,test=NULL,rounds=250,csv=FALSE){

  base::library(maboost)

  c <- which(colnames(train)=="popularity")

  if(length(test)==0){
    print("No test set, cross validating train set.")
    base::library(caret)
    set.seed(1234)
    flds <- createFolds(train$popularity,k=5)
    mabpred <- matrix(NA,nrow(train),10)

    for(j in 1:5){
      ktest <- train[flds[[j]],-c]
      ktrain <- train[-flds[[j]],]

      print("fold")
      print(j) # tracing progress

      mab <- maboost(ktrain[,-c],ktrain$popularity, iter=rounds)

      mabpred[ flds[[j]],1:5] <- predict(mab,ktest,type = "prob")
      mabpred[ flds[[j]],6:10] <- predict(mab,ktest,type = "F")


      } # end of looping cross validations
      mabframe <- as.data.frame(mabpred)
      colnames(mabframe) <- c("mabp1","mabp2","mabp3","mabp4","mabp5",
                              "mabF1","mabF2","mabF3","mabF4","mabF5")

      if(csv==TRUE){
        write.csv(mabframe,"mabmetafldsmat33.csv",row.names = FALSE)
      }
      return(mabframe)
  } # end of train set cross validation and meta predictions
  else{ # test set loaded
    print("test set loaded, learning on train and predicting on test")

    mab <- maboost(train[,-c],train$popularity, iter=rounds)

    mabpred <- matrix(NA,nrow(test),10)
    mabpred[,1:5] <- predict(mab,test,type = "prob")
    mabpred[,6:10] <- predict(mab,test,type = "F")

    mabframe <- data.frame(mabpred)
    colnames(mabframe) <- c("mabp1","mabp2","mabp3","mabp4","mabp5",
                            "mabF1","mabF2","mabF3","mabF4","mabF5")

    if(csv==TRUE){
      write.csv(mabframe, file = "mabmetatestmat33.csv", row.names=FALSE)
    }

    return(mabframe)
  } # end of test set meta
} # end of fmeta.mab


# ----------------------------------------------------------------------
# train multinomial logistic meta predictions with folds for train
# ----------------------------------------------------------------------
#' multinomial logistic meta predictor
#'
#' Creates probabilities for each class
#'
#' @param train Train set without ID and URL
#' @param test Test set without ID and URL. If left empty then it the
#' functions trains on train with 5 fold cross validation
#' @param csv Boolean, default FALSE. If TRUE, saves csv output.
#' @return A data frame with 5 probabilites for each row
#' @export
fmeta.glm <- function(train,test=NULL,csv=FALSE){

  base::library(glmnet)

  if(length(test)==0){
    print("No test set, cross validating train set.")
    base::library(caret)
    set.seed(1234)
    flds <- createFolds(train$popularity,k=5)
    rfpred <- matrix(NA,nrow(train),5)

    for(j in 1:5){
      ktest <- train[flds[[j]],-60]
      ktrain <- train[-flds[[j]],]

      print("fold")
      print(j) # tracing progress

      cv.fit <- cv.glmnet(as.matrix(ktrain[,-60]),
                            ktrain[,60], alpha=1,
                            family="multinomial")
      #
      predictions <- predict(cv.fit, newx=as.matrix(ktest[,-60]),
                            type="response")
      predictions <- predictions[,,]
      rfpred[ flds[[j]],] <- predictions



      } # end of looping cross validations
      rfframe <- as.data.frame(rfpred)
      colnames(rfframe) <- c("glmp1","glmp2","glmp3","glmp4","glmp5")

      if(csv==TRUE){
        write.csv(rfframe,"glmetafldsmat33.csv",row.names = FALSE)
      }
      return(rfframe)
  } # end of train set cross validation and meta predictions
  else{ # test set loaded
    print("test set loaded, learning on train and predicting on test")

    cv.fit <- cv.glmnet(as.matrix(train[,-60]),
                          train[,60], alpha=1,
                          family="multinomial")
    #
    predictions <- predict(cv.fit, newx=as.matrix(test),
                          type="response")
    predictions <- predictions[,,]
    rfframe <- data.frame(predictions)
    colnames(rfframe) <- c("glmp1","glmp2","glmp3","glmp4","glmp5")


    if(csv==TRUE){
      write.csv(rfframe, file = "glmetatestmat33.csv", row.names=FALSE)
    }

    return(rfframe)
  } # end of test set meta
} # end of fmeta.glm


# ----------------------------------------------------------------------
# train h2o neural network meta predictions with folds for train
# ----------------------------------------------------------------------
#' h2o deep learning meta predictor
#'
#' Creates probabilities for each class
#'
#' @param train Train set without ID and URL
#' @param test Test set without ID and URL. If left empty then it the
#' functions trains on train with 5 fold cross validation
#' @param csv Boolean, default FALSE. If TRUE, saves csv output.
#' @return A data frame with 5 probabilites for each row
#' @export
fmeta.h2odl <- function(train,test=NULL,csv=FALSE){#make sure h2o is down

  base::library(h2o)
  localH2O = h2o.init(nthreads=-1)

  if(length(test)==0){
    print("No test set, cross validating train set.")
    base::library(caret)
    set.seed(1234)
    flds <- createFolds(train$popularity,k=5)
    rfpred <- matrix(NA,nrow(train),5)

    for(j in 1:5){

      print("fold")
      print(j) # tracing progress

      ktest <- train[flds[[j]],]
      ktrain <- train[-flds[[j]],]

      data_train_h <- as.h2o(ktrain,destination_frame = "h2o_data_train")
      data_test_h <- as.h2o(ktest,destination_frame = "h2o_data_test")

      y <- "popularity"
      x <- setdiff(names(data_train_h), y)

      data_train_h[,y] <- as.factor(data_train_h[,y])
      data_test_h[,y] <- as.factor(data_test_h[,y])

      model <- h2o.deeplearning(x = x,
                                y = y,
                                training_frame = data_train_h,
                                validation_frame = data_test_h,
                                distribution = "multinomial",
                                activation = "RectifierWithDropout",
                                hidden = c(200,200,200),
                                input_dropout_ratio = 0.2,
                                l1 = 1e-7,
                                epochs = 20)
      #
      pred <- h2o.predict(model, newdata = data_test_h)
      predmat <- as.matrix(pred[,2:6])

      rfpred[ flds[[j]],] <- predmat

      } # end of looping cross validations
      rfframe <- as.data.frame(rfpred)
      colnames(rfframe) <- c("H2ONNp1","H2ONNp2","H2ONNp3","H2ONNp4","H2ONNp5")

      if(csv==TRUE){
        #write.csv(rfframe,"h2oNNmetafldsmat33.csv",row.names = FALSE)
        write.csv(rfframe,"h2oNNmeta2ndlayer.csv",row.names = FALSE)
      }
      return(rfframe)
  } # end of train set cross validation and meta predictions
  else{ # test set loaded
    print("test set loaded, learning on train and predicting on test")

    data_train_h <- as.h2o(train,destination_frame = "h2o_data_train")
    data_test_h <- as.h2o(test,destination_frame = "h2o_data_test")

    y <- "popularity"
    x <- setdiff(names(data_train_h), y)

    data_train_h[,y] <- as.factor(data_train_h[,y])

    model <- h2o.deeplearning(x = x,
                              y = y,
                              training_frame = data_train_h,

                              distribution = "multinomial",
                              activation = "RectifierWithDropout",
                              hidden = c(200,200,200),
                              input_dropout_ratio = 0.2,
                              l1 = 1e-7,
                              epochs = 20)
    #
    pred <- h2o.predict(model, newdata = data_test_h)
    predmat <- as.data.frame(pred[,2:6])

    rfframe <- data.frame(predmat)

    colnames(rfframe) <- c("H2ONNp1","H2ONNp2","H2ONNp3","H2ONNp4","H2ONNp5")

    if(csv==TRUE){
      write.csv(rfframe, file = "h2oNNmetatestmat33.csv", row.names=FALSE)
    }

    return(rfframe)
  } # end of test set meta
  h2o.shutdown()
} # end of fmeta.h2odl


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
# cross validation random forest multicore
# ----------------------------------------------------------------------
#' cross validation random forest multicore
#'
#' five fold cross validation using random forest
#'
#' @param train Any train set without ID and URL
#' @param tree Number of trees, default 1000
#' @param core Number of cores, default 4
#' @return A list containing a vector of each fold accuracy and a mean.
#' @export
cv.rfmc <- function(train,tree=1000,core=4){

  base::library(randomForest)
  base::library(doMC)
  registerDoMC(core=core)

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
      ##
      set.seed(1234)
      rfmc <- foreach(ntree=rep(tree/core, core), .combine=combine,
                      .multicombine=TRUE,.packages='randomForest') %dopar% {
          randomForest(ktrain[,-c], as.factor(ktrain$popularity), ntree=ntree,
             importance=TRUE, do.trace=TRUE)
      }

      predictions <- predict(rfmc,ktest[,-c])
      rfsum <- confusionMatrix(predictions, ktest$popularity)
      rfpred[j] <- rfsum$overall[1]
      print(rfsum)


      } # end of looping cross validations
      avg <- mean(rfpred)

      return(list(vec=rfpred,avg=avg))

} # end of cv.rfmc


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


# ----------------------------------------------------------------------
# cross validation neural network nnet
# ----------------------------------------------------------------------
#' cross validation neural network nnet package
#'
#' five fold cross validation using nnet NN
#'
#' @param train Any train set without ID and URL
#' @param size Number of units in hidden layer, default 20
#' @param rounds Number of rounds, default 500
#' @return A list containing a vector of each fold accuracy and a mean.
#' @export
cv.nn <- function(train, size=20, rounds=500){

  base::library(nnet)

  c <- which(colnames(train)=="popularity")

    base::library(caret)
    set.seed(1234)
    flds <- createFolds(train$popularity,k=5)
    nnpred <- rep(NA,5)

    for(j in 1:5){
      ktest <- train[flds[[j]],]
      ktrain <- train[-flds[[j]],]

      print("fold")
      print(j) # tracing progress
      print(nnpred)

      nn <- nnet(popularity~.,data=ktrain, size=size, maxit=rounds)

      predictions <- factor(predict(nn, newdata=ktest,type="class"),
                            levels=c(1:5))
      nnsum <- confusionMatrix(predictions, ktest$popularity)

      print(nnsum)

      nnpred[j] <- nnsum$overall[1]

    } # end of looping cross validations
    avg <- mean(nnpred)

    return(list(vec=nnpred,avg=avg))

} # end of cv.nn


# ----------------------------------------------------------------------
# cross validation neural network mxnet
# ----------------------------------------------------------------------
#' cross validation neural network mxnet package
#'
#' five fold cross validation using mxnet NN
#'
#' @param train Any train set without ID and URL
#' @param rounds Number of rounds, default 20
#' @return A list containing a vector of each fold accuracy and a mean.
#' @export
cv.mxnet <- function(train, rounds=20){

  base::library(mlbench)
  base::library(mxnet)

  c <- which(colnames(train)=="popularity")

    base::library(caret)
    set.seed(1234)
    flds <- createFolds(train$popularity,k=5)
    nnpred <- rep(NA,5)

    for(j in 1:5){
      ktest <- data.matrix(train[flds[[j]],])
      ktrain <- data.matrix(train[-flds[[j]],])

      print("fold")
      print(j) # tracing progress
      print(nnpred)

      mx.set.seed(0)
      model <- mx.mlp(ktrain[,-c], ktrain[,c]-1, hidden_node=50, out_node=5,
                      out_activation="softmax", num.round=rounds,
                      array.batch.size=50,
                      learning.rate=0.07, momentum=0.9,
                      eval.metric=mx.metric.accuracy)

      preds <- predict(model, ktest[,-c])
      predictions <- max.col(t(preds))

      nnsum <- confusionMatrix(predictions, ktest[,c])

      print(nnsum)

      nnpred[j] <- nnsum$overall[1]

    } # end of looping cross validations
    avg <- mean(nnpred)

    return(list(vec=nnpred,avg=avg))

} # end of cv.mxnet


# ----------------------------------------------------------------------
# cross validation neural network h2o
# ----------------------------------------------------------------------
#' cross validation neural network h2o package
#'
#' five fold cross validation using h2o NN
#'
#' @param train Any train set without ID and URL
#' @param epochs Number of passes over the training dataset, default 10
#' @return A list containing a vector of each fold accuracy and a mean.
#' @export
cv.h2odl <- function(train, epochs=10){

  base::library(h2o)
  localH2O = h2o.init(nthreads=-1)

    base::library(caret)
    set.seed(1234)
    flds <- createFolds(train$popularity,k=5)
    nnpred <- rep(NA,5)

    for(j in 1:5){
      ktest <- data.matrix(train[flds[[j]],])
      ktrain <- data.matrix(train[-flds[[j]],])

      data_train_h <- as.h2o(ktrain,destination_frame = "h2o_data_train")
      data_test_h <- as.h2o(ktest,destination_frame = "h2o_data_test")

      y <- "popularity"
      x <- setdiff(names(data_train_h), y)

      data_train_h[,y] <- as.factor(data_train_h[,y])
      data_test_h[,y] <- as.factor(data_test_h[,y])

      print("fold")
      print(j) # tracing progress
      print(nnpred)

      model <- h2o.deeplearning(x = x,
                                y = y,
                                training_frame = data_train_h,
                                validation_frame = data_test_h,
                                distribution = "multinomial",
                                activation = "RectifierWithDropout",
                                hidden = c(200,200,200),
                                input_dropout_ratio = 0.2,
                                l1 = 1e-7,
                                epochs = epochs)
      #
      pred <- h2o.predict(model, newdata = data_test_h)
      predictions <- factor(as.numeric(as.vector(pred[,1])),levels=c(1,2,3,4,5))

      nnsum <- confusionMatrix(predictions,ktest[,y])

      print(nnsum)

      nnpred[j] <- nnsum$overall[1]

    } # end of looping cross validations
    avg <- mean(nnpred)

    return(list(vec=nnpred,avg=avg))

    h2o.shutdown()
} # end of cv.h2odl


# ----------------------------------------------------------------------
# arithmetic average of two models
# ----------------------------------------------------------------------
#' arithmetic average of two models
#'
#' This version is made for 5-label classification. Inputs are prediction
#' probability matricesof two classifiers.
#'
#' @param mat1 Matrix of first classifier class probabiliets.
#' @param mat2 Matrix of second classifier class probabiliets.
#' @param label True class labels. Default train set popularity column.
#' @param iter Number of weight combinations, default 101 (0.01 increment)
#' @return An ordered matrix with weight combinations and matching accuracies
#' @export
avg.arit <- function(mat1,mat2,label=train$popularity,iter=101){
  mat1 <- as.matrix(mat1)
  mat2 <- as.matrix(mat2)
  mat <- cbind(mat1,mat2)
  result <- matrix(NA,nrow=iter,ncol=3)
  f <- function(vec){
    pred <- which.max(w*vec[1:5]+(1-w)*vec[6:10])
    return(pred)
  }
  for(i in seq(0,1,length.out=iter)){
    print(i)
    w<-i
    pred <- apply(mat,1,f)
    score <- sum(pred==label)/length(label)
    result[1+i/(1/(iter-1)),] <- c(i,(1-i),score)
  }
  return(result[order(result[,3],decreasing=TRUE),])
}


# ----------------------------------------------------------------------
# geometric average of two models
# ----------------------------------------------------------------------
#' geometric average of two models
#'
#' This version is made for 5-label classification. Inputs are prediction
#' probability matricesof two classifiers.
#'
#' @param mat1 Matrix of first classifier class probabiliets.
#' @param mat2 Matrix of second classifier class probabiliets.
#' @param label True class labels. Default train set popularity column.
#' @param iter Number of weight combinations, default 101 (0.01 increment)
#' @return An ordered matrix with weight combinations and matching accuracies
#' @export
avg.geom <- function(mat1,mat2,label=train$popularity,iter=101){
  mat1 <- as.matrix(mat1)
  mat2 <- as.matrix(mat2)
  mat <- cbind(mat1,mat2)
  result <- matrix(NA,nrow=iter,ncol=3)
  f <- function(vec){
    pred <- which.max(vec[1:5]^(w) * vec[6:10]^(1-w))
    return(pred)
  }
  for(i in seq(0,1,length.out=iter)){
    print(i)
    w<-i
    pred <- apply(mat,1,f)
    score <- sum(pred==label)/length(label)
    result[1+i/(1/(iter-1)),] <- c(i,(1-i),score)
  }
  return(result[order(result[,3],decreasing=TRUE),])
}
