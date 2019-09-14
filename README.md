# preprocessing
Custom functions for convenient preparation of the dataset for modeling.

## outliers
Functions for detection and processing outliers. The input is the whole dataset, and only numeric columns are selected
and processed automatically. Extreme values are replaced with threshold values. Based on the 1.5-IQR rule for symmetrical 
(normal-like) distributions but modified for skewed (non-negative) distributions where outliers are only one-sided. Works
also for very dense distributions (with high predominance of some one or several values).