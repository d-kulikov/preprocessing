import numpy as np
from scipy.stats import mode

class Binaries( object ) :
        """ Restores integer dummy values of binary variables after scaling (e.g., after applying sklearn's StandardScaler).
    Otherwise, if they are very impalanced, they may have extremely high values (this is not desired for optimization algorithms). 
    Accepts the whole numpy array as input and processes only binary variables. The function can not only restore original 0 and 1
    values, but, depending on the distribution, return 0 and -1 or 1 and -1. This allows to keep the mean closer to 0 and the
    standard deviation to 1 (as with truly numeric variables).
    
    Example:
    X_TRAIN_RESTORED = binaries( X_TRAIN, X_TRAIN )
    X_TEST_RESTORED = binaries( X_TRAIN, X_TEST  )
    """
    
    def _init_( self ) :
        self.values = None
        self.ncols = 0
        
    def fit( self, X ) :
        nrows = X.shape[ 0 ]        
        self.ncols = X.shape[ 1 ]        
        # Creating an array of the parameters
        self.values = np.full( ( 2, self.ncols ), fill_value=np.nan, dtype=np.float )        
        for i in range( 0, self.ncols ) :
            # Checks if this is a binary variable
            if len( np.unique( X[ :, i ] ) ) == 2 :    
                minim = np.min( X[ :, i ] )    
                maxim = np.max( X[ :, i ] )    
                mode_value = mode( X[ :, i ] )[ 0 ]    
                # Checks if the distribution is imbalanced
                if sum( X[ :, i ] == mode_value ) / nrows >= 2/3 :        
                    # The most frequent value becomes 0 
                    if mode_value == minim :
                        self.values[ 1, i ] = 0
                        self.values[ 0, i ] = 1
                    else:
                        self.values[ 1, i ] = -1
                        self.values[ 0, i ] = 0
                else :
                    # If the values are more or less balanced, they become -1 and 1
                    self.values[ 1, i ] = -1; self.values[ 0, i ] = 1
                    
    def predict( self, X ) :
        for i in range( 0, self.ncols ) :    
            # Checks if this is a binary variable
            if np.isnan( self.values[ 0, i ] ) == False :
                X[ :, i ] = np.where( X[ :, i ] > np.mean( X[ :, i ] ), self.values[ 0, i ], self.values[ 1, i ] )
        return X