import numpy as np
from scipy.stats import mode

class Binaries( object ) :
    """ Restores integer dummy values of binary variables after scaling (e.g., after applying sklearn's StandardScaler).
    Otherwise, if they are very impalanced, they may have extremely high values (this is not desired for optimization algorithms). 
    Accepts the whole numpy array as input and processes only binary variables. The function can not only restore original 0 and 1
    values, but, depending on the distribution, return 0 and -1 or 1 and -1. This allows to keep the mean closer to 0 and the
    standard deviation to 1 (as with truly numeric variables).
    
    Example:
    a = np.array([ [ 5, 4, 8, 0 ],
                   [ 5, 4, 8, 1 ],
                   [ 5, 4, 7, 2 ],
                   [ 5, 4, 7, 3 ],
                   [ 2, 9, 7, 4 ] ])
    binaries = Binaries()
    binaries.fit( a )
    print( binaries.values_ )
    transformed = binaries.transform( a ) """
    
    def _init_( self ) :
        self.values_ = None
        self.ncols_ = 0
        
    def fit( self, X ) :
        """ Defines distributions of the binary variables
        X: input numpy array """
        nrows = X.shape[ 0 ]        
        self.ncols_ = X.shape[ 1 ]        
        # Creating an array of the parameters
        self.values_ = np.full( ( 2, self.ncols_ ), fill_value=np.nan, dtype=np.float )        
        for i in range( 0, self.ncols_ ) :
            # Checks if this is a binary variable
            if len( np.unique( X[ :, i ] ) ) == 2 :    
                minim = np.min( X[ :, i ] )    
                maxim = np.max( X[ :, i ] )    
                mode_value = mode( X[ :, i ] )[ 0 ]    
                # Checks if the distribution is imbalanced
                if sum( X[ :, i ] == mode_value ) / nrows >= 2/3 :        
                    # The most frequent value becomes 0 
                    if mode_value == minim :
                        self.values_[ 1, i ] = 0
                        self.values_[ 0, i ] = 1
                    else:
                        self.values_[ 1, i ] = -1
                        self.values_[ 0, i ] = 0
                else :
                    # If the values are more or less balanced, they become -1 and 1
                    self.values_[ 1, i ] = -1
                    self.values_[ 0, i ] = 1
                    
    def transform( self, X ) :
        """ Applies the transformations and returns a new array
        X: numpy array to transform """
        out = np.copy( X )
        for i in range( 0, self.ncols_ ) :    
            # Checks if this is a binary variable
            if np.isnan( self.values_[ 0, i ] ) == False :
                out[ :, i ] = np.where( out[ :, i ] > np.mean( out[ :, i ] ), self.values_[ 0, i ], self.values_[ 1, i ] )
        return out