% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/DUN.R
\name{cv.mxnet}
\alias{cv.mxnet}
\title{cross validation neural network mxnet package}
\usage{
cv.mxnet(train, rounds = 20)
}
\arguments{
\item{train}{Any train set without ID and URL}

\item{rounds}{Number of rounds, default 20}
}
\value{
A list containing a vector of each fold accuracy and a mean.
}
\description{
five fold cross validation using mxnet NN
}

