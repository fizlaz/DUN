% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/DUN.R
\name{fmeta.glm}
\alias{fmeta.glm}
\title{multinomial logistic meta predictor}
\usage{
fmeta.glm(train, test = NULL, csv = FALSE)
}
\arguments{
\item{train}{Train set without ID and URL}

\item{test}{Test set without ID and URL. If left empty then it the
functions trains on train with 5 fold cross validation}

\item{csv}{Boolean, default FALSE. If TRUE, saves csv output.}
}
\value{
A data frame with 5 probabilites for each row
}
\description{
Creates probabilities for each class
}

