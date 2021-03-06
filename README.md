# preprocessing
Custom functions for convenient preparation of the dataset for modeling.

## outliers
Functions for detection and processing outliers. The input is the whole dataset, and only numeric columns are selected
and processed automatically. Extreme values are replaced with threshold values. Based on the 1.5-IQR rule for symmetrical 
(normal-like) distributions but modified for skewed (non-negative) distributions where outliers are only one-sided. Works
also for very dense distributions (with high predominance of some one or several values).

## categories
A function for transforming categorical variables (especially helpful for those with a huge number of levels, e.g. car model, city, zip ) into numeric
ones. Each level of the variable will correspond to a certain continuous value based on the distribution of the target variable within this level.
It works for both regression and binary classification problems.

## binaries
A function that restores integer dummy values of binary variables after scaling (e.g., after applying sklearn's StandardScaler).
Otherwise, if they are very impalanced, they may have extremely high values (this is not desired for optimization algorithms). 
Accepts the whole numpy array as input and processes only binary variables. The function can not only restore original 0 and 1
values, but, depending on the distribution, return 0 and -1 or 1 and -1. This allows to keep the mean closer to 0 and the
standard deviation to 1 (as with truly numeric variables).

## timeseries_to_lags
A simple and efficient function for transforming a column (which is a timeseries) of a numpy array into a separate array
where each row is a sequence of lags for a particular observation. This type of transformation is required for preparation of
timeseries for fitting an LSTM or CNN.