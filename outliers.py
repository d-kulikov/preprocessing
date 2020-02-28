import numpy as np
import pandas as pd

class Outliers( object ) :  
    """ Functions for detection and processing outliers. The input is the whole dataset, and only numeric columns are selected
    and processed automatically. Extreme values are replaced with threshold values. Based on the 1.5-IQR rule for symmetrical 
    (normal-like) distributions but modified for skewed (non-negative) distributions where outliers are only one-sided. Works
    also for very dense distributions (with high predominance of some one or several values).
    
    Example:
    d = pd.DataFrame( { 'a' : [ -1000, -4, -3, -2, -1, 0, 1, 2, 3, 4, 1000  ],
                        'b' : [ 0, 1, 1, 1, 1, 2, 2, 2, 3, 4, 1000 ],
                        'c' : [ -1000, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1000 ],
                        'd' : [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 1000 ],
                        'e' : [ 'I', 'II', 'III', 'IV', 'V', 'VI', 'VII', 'VIII', 'IX', 'X', 'XI' ] } )
    outliers = Outliers()
    outliers.fit( d )    
    print( outliers.bounds_ )
    transformed = outliers.transform( d ) """
    
    def _init_( self ) :
        self.bounds_ = None

    def fit( self, X ) : 
        """ Defines upper and lower threshold values for censoring outliers
        X: input dataframe """
        self.bounds_ = pd.DataFrame( index=[ 0, 1 ], columns=list( X ) )
        for name in list( X ) :
            # Checks if the column is numeric            
            if X[ name ].nunique() >= 3 and X[ name ].dtype in [ 'float64', 'float32', 'float16', 'int64', 'int32', 'int16', 'int8' ] :
                q1 = X[ name ].quantile( 0.25 )
                q2 = X[ name ].quantile( 0.50 )
                q3 = X[ name ].quantile( 0.75 )
                # Checks if the distribution is not very dense
                if ( q2 - q1 ) != 0 and ( q3 - q2 ) != 0 :
                    # Checks if the distribution is non-symmetrical
                    skewed = ( q3 - q2 ) / ( q2 - q1 ) > 1.25 or ( q2 - q1 ) / ( q3 - q2 ) > 1.25
                else:
                    skewed = True  
                # Threshold values for a skewed distribution
                if skewed and X[ name ].min() >= 0 : 
                    lower = 0
                    if q3 > 0 and q2 - q1 >= q3 - q2 :
                        upper = q2 * 6
                    elif q3 > 0 and q2 - q1 < q3 - q2 : 
                        upper = q3 * 6   
                    else :
                        upper = 1
                # Threshold values for a symmetrical distribution
                else :
                    iqr = q3 - q1
                    if iqr > 0 :
                        lower = q1 - 1.5 * iqr
                        upper = q3 + 1.5 * iqr
                    else :
                        lower = q1 - 1
                        upper = q3 + 1
                self.bounds_.loc[ 1, name ] = lower
                self.bounds_.loc[ 0, name ] = upper
      
    def transform( self, X ) :  
        """ Applies censoring and returns a new dataframe
        X: dataframe to censor """
        out = X.copy()
        for name in list( self.bounds_ ) :
            # Checks if the column is numeric
            if self.bounds_[ name ].notnull().all() :
                # Logical indexes for normal values
                condlow = np.logical_or( out[ name ] > self.bounds_.loc[ 1, name ], out[ name ].isnull() )
                condup = np.logical_or( out[ name ] < self.bounds_.loc[ 0, name ], out[ name ].isnull() )
                # Replaces extreme values (where the logical conditions are false)
                out[ name ].where( condlow, self.bounds_.loc[ 1, name ], inplace=True )
                out[ name ].where( condup, self.bounds_.loc[ 0, name ], inplace=True )
        return out