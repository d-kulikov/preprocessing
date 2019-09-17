def many_levels( dataframe, categorical_names, target, conservatism=30 ) :
    
    """ A function for transforming categorical variables with a huge number of levels (e.g. car model, city, zip ) into numeric
    ones. Based on odds ratio within each level. Applied to binary classification problems. Missing values are treated as a
    separate level. Adds columns (with suffix "_num") to an existing dataframe which are numeric representation of those
    categorical variables.
    
    Parameters
    dataframe: an input dataframe
    categorical_names: a list of variable names requiring to be transformed
    target: the name of a target binary variable in the dataframe (should be coded as 1 and 0)
    conservatism: a conservative term added both to the numenator and denominator of the odds ratio in order to take into account
        the size of the level (its frequency). For rare levels (consisting of only several observations), which are not reliable, 
        the value will not be very indicative and will be close to the mean level across all the data.
        
    Example
    many_levels( MYDATAFRAME, [ 'variable1', 'variable2' ], 'mytarget' )
    
    Note
    Should be used very carefully with a train and test datasets. A test dataset should have parameters determined on a train 
    dataset only. Otherwise it is a pure overfit (will be implemented later).
    """
    
    import numpy as np
    
    import pandas as pd
    
    from gc import collect
    
    Positives = dataframe[ target ].sum()
    
    # Calculates the ratio of positives and negatives in general
    Balance = Positives / ( dataframe.shape[ 0 ] - Positives ) 

    for Varname in categorical_names :
    
        # Assigns a separate level to missing values
        dataframe.loc[ dataframe[ Varname ].isnull(), Varname ] = 'none'
        
        # Creating a summary dataframe
        LOOKUP = dataframe.groupby( Varname )[ target ].agg([ 'count', 'sum' ])
        
        LOOKUP[ 'negatives' ] = LOOKUP[ 'count' ] - LOOKUP[ 'sum' ]
        
        # Calculating the numeric values corresponding to each presented level of a variable
        LOOKUP[ 'ratio' ] = ( LOOKUP[ 'sum' ] + conservatism * Balance ) / ( LOOKUP[ 'negatives' ] + conservatism )
        
        # Assigning corresponding values in a new numeric column of the same dataframe
        dataframe[ Varname + '_num' ] = np.array( LOOKUP.loc[ dataframe[ Varname ], 'ratio' ] )
        
        del LOOKUP
        
        collect()