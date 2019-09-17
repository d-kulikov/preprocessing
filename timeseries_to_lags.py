def timeseries_to_lags( nparray, column_number, lags ) :
    
    """ A simple and efficient function for transforming a column (which is a timeseries) of a numpy array into a separate array
    where each row is a sequence of lags for a particular observation. This type of transformation is required for preparation of
    timeseries for fitting an LSTM or CNN.
    
    Example:
    # Producing arrays with 10 lags
    VAR0_LAGGED = timeseries_to_lags( X, 0, 10 )
    VAR5_LAGGED = timeseries_to_lags( X, 5, 10 )
    
    Note:
    It needs to be careful with the beginning of the dataset because the numpy's function "roll" fills missing lags in the biginning
    with the values from the last rows.
    """
    
    import numpy as np
    
    R = nparray.shape[ 0 ]
    
    Vector = np.reshape( nparray[ :, column_number ], ( R, 1 ) )
    
    OUT = np.roll( Vector, lags-1 )   

    for i in range( lags-2, -1, -1 ) :

        OUT = np.append( OUT, np.roll( Vector, i ), axis=1 )
        
    return OUT