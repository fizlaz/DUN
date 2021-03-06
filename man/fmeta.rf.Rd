% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/DUN.R
\name{fmeta.rf}
\alias{fmeta.rf}
\title{Random Forest meta predictor}
\usage{
fmeta.rf(train, test = NULL, csv = FALSE, trees = 1000, verbose = FALSE)
}
\arguments{
\item{train}{Train set without ID and URL}

\item{test}{Test set without ID and URL. If left empty then it the
functions trains on train with 5 fold cross validation}

\item{csv}{Boolean, default FALSE. If TRUE, saves csv output.}

\item{trees}{Number of trees per fold, default 1000.}

\item{verbose}{Boolean, default FALSE. If TRUE, prints output.}
}
\value{
A data frame with 5 probabilites for each row
}
\description{
Creates probabilities for each class
}

