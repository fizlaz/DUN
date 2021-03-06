% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/DUN.R
\name{fmeta.knnprob}
\alias{fmeta.knnprob}
\title{KNN meta predictor generator, probability output}
\usage{
fmeta.knnprob(train, test = NULL, csv = FALSE)
}
\arguments{
\item{train}{Train set without ID and URL}

\item{test}{Test set without ID and URL. If left empty then it the
functions trains on train with 5 fold cross validation}

\item{csv}{Boolean, default FALSE. If TRUE, saves csv output.}
}
\value{
A data frame with KNN probability of highest vote
}
\description{
Creates probability of highest KNN meta predictions vote
for K=(1,2,4,8,16,32,64,128,256)
}

