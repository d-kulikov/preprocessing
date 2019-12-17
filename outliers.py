class Outliers :
    
    """ Functions for detection and processing outliers. The input is the whole dataset, and only numeric columns are selected
    and processed automatically. Extreme values are replaced with threshold values. Based on the 1.5-IQR rule for symmetrical 
    (normal-like) distributions but modified for skewed (non-negative) distributions where outliers are only one-sided. Works
    also for very dense distributions (with high predominance of some one or several values).
    
    Example:
    MYBOUNDS = Outliers.fit( TRAIN_DATAFRAME )
    Outliers.transform( TEST_DATAFRAME, MYBOUNDS )
    """

    def fit( dataframe ) : 
        
        """ Produces a dataframe with upper and lower threshold values for censoring outliers """
        
        import numpy as np
    
        import pandas as pd
        
        BOUNDS = pd.DataFrame( index=[ 0, 1 ], columns=list( dataframe ) )
    
        for Varname in list( dataframe ) :
            
            # Checks if the column is numeric            
            if dataframe[ Varname ].nunique() >= 3 and dataframe[ Varname ].dtype in [ 'float64', 'float32', 'int64', 'int32' ] :
                
                Q1 = dataframe[ Varname ].quantile( 0.25 )
                
                Q2 = dataframe[ Varname ].quantile( 0.50 )
      
                Q3 = dataframe[ Varname ].quantile( 0.75 )
                
                # Checks if the distribution is very dense
                if ( Q2 - Q1 ) != 0 and ( Q3 - Q2 ) != 0 :
                    
                    # Checks if the distribution is non-symmetrical
                    Skewed = ( Q3 - Q2 ) / ( Q2 - Q1 ) >= 1.2 or ( Q2 - Q1 ) / ( Q3 - Q2 ) >= 1.2
                    
                else:
                    
                    Skewed = True
                
                # Threshold values for a skewed distribution
                if Skewed and dataframe[ Varname ].min() >= 0 : 
                    
                    Lower = 0
                    
                    if Q3 > 0 and Q2 - Q1 >= Q3 - Q2 :
                    
                        Upper = Q2 * 6
                        
                    elif Q3 > 0 and Q2 - Q1 < Q3 - Q2 :
                        
                        Upper = Q3 * 6
                        
                    else :
                        
                        Upper = 1
                
                # Threshold values for a symmetrical distribution
                else :
            
                    Iqr = Q3 - Q1
                    
                    if Iqr > 0 :
                
                        Lower = Q1 - 1.5 * Iqr
                    
                        Upper = Q3 + 1.5 * Iqr
                        
                    else :
                        
                        Lower = Q1 - 1
                    
                        Upper = Q3 + 1
                
                BOUNDS.loc[ 1, Varname ] = Lower
                
                BOUNDS.loc[ 0, Varname ] = Upper
                
        return BOUNDS
    
    
    
    
    def transform( dataframe, bounds ) :
        
        """ Applying the threshold values to a dataframe """
        
        import numpy as np
    
        import pandas as pd
        
        for Varname in list( bounds ) :
            
            # Checks if the column is numeric
            if bounds[ Varname ].notnull().all() :
                
                # Logical indexes for normal values
                Condlow = np.logical_or( dataframe[ Varname ] > bounds.loc[ 1, Varname ], dataframe[ Varname ].isnull() )
                
                Condup = np.logical_or( dataframe[ Varname ] < bounds.loc[ 0, Varname ], dataframe[ Varname ].isnull() )
                
                # Replacing extreme values (where the logical conditions are false)
                dataframe[ Varname ].where( Condlow, bounds.loc[ 1, Varname ], inplace=True )
                
                dataframe[ Varname ].where( Condup, bounds.loc[ 0, Varname ], inplace=True )