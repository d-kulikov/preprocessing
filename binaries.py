def binaries( reference_nparray, nparray_to_transform ) :
    
    """ ***************************************************************************************************************************
    Restores integer dummy values of binary variables after scaling (e.g., after applying sklearn's StandardScaler).
    Otherwise, if they are very impalanced, they may have extremely high values (this is not desired for optimization algorithms). 
    Accepts the whole numpy array as input and processes only binary variables. The function can not only restore original 0 and 1
    values, but, depending on the distribution, return 0 and -1 or 1 and -1. This allows to keep the mean closer to 0 and the
    standard deviation to 1 (as with truly numeric variables).
    
    Example:
    X_TRAIN_RESTORED = binaries( X_TRAIN, X_TRAIN )
    X_TEST_RESTORED = binaries( X_TRAIN, X_TEST  )
    """
    
    import numpy as np
    
    from scipy.stats import mode
    
    Nrows = reference_nparray.shape[ 0 ]
    
    # Creating a new array based on the array we want to modify
    OUT = nparray_to_transform
    
    for i in range( 0, reference_nparray.shape[ 1 ] ) :
        
        # Checks if this is a binary variable
        if len( np.unique( reference_nparray[ :, i ] ) ) == 2 :
            
            Min = np.min( reference_nparray[ :, i ] )
            
            Max = np.max( reference_nparray[ :, i ] )
            
            Mode = mode( reference_nparray[ :, i ] )[ 0 ]
            
            # Checks if the distribution is imbalanced
            if sum( reference_nparray[ :, i ] == Mode ) / Nrows >= 2/3 :
                
                # The most frequent value becomes 0 
                if Mode == Min :
                
                    OUT[ :, i ] = np.where( OUT[ :, i ] == Mode, 0, 1 )
                    
                else:
                    
                    OUT[ :, i ] = np.where( OUT[ :, i ] == Mode, 0, -1 )
                
            else :
                
                # If the values are more or less balanced, they become -1 and 1
                OUT[ OUT[ :, i ] == Min, i ] = -1
                 
                OUT[ OUT[ :, i ] == Max, i ] = 1
                
    return OUT