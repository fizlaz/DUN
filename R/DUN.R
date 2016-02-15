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

#' News popularity training data.
#'
#' A dataset containing the attributes of 30,000 articles.
#'
#' @format A data frame with 30000 rows and 62 variables
#' @source \url{https://inclass.kaggle.com/c/predicting-online-news-popularity/data}
"train"

#' News popularity test data.
#'
#' A dataset containing the attributes of 9,644 articles.
#'
#' @format A data frame with 9644 rows and 61 variables
#' @source \url{https://inclass.kaggle.com/c/predicting-online-news-popularity/data}
"test"

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
