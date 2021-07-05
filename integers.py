import numpy as np
import pandas as pd

class Integers( object ) :
    
    """ A function for transforming categorical variables (especially helpful for those with a huge number of categories, e.g. car model, city, zip ) into numeric
    ones. Each level of the variable will correspond to a certain continuous value based on the distribution of the target variable within this level.
    It works for both regression and binary classification problems.
    
    features: a list of feature names to transform
    label: the name of a target binary variable in the dataframe (continuous or binary)
    sample_frac: the share of the observations to sample from the input dataset (using not the whole dataset should reduce overfit)
    dummy_number: a number of dummy observations equal to the average added to each level group in order to mitigate the effect of rare categories (categories with a few
                  data points will just become close to the general average )
    
    Example:    
    d = pd.DataFrame( { 'a' : [ 'i', 'i', 'i', 'k', 'k', 'k', 'l', 'l', 'l', 'm', 'm', 'm' ],
                        'b' : [ 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 1, 1 ] } )
    categories = Categories( features=[ 'a' ], label='b', sample_frac=0.7, dummy_number=1 )
    categories.fit( d )
    print( categories.vars_[ 'a' ] )
    transformed = categories.transform( d ) """
    
    def __init__( self, features, timestamp ) :
        self.features_ = features
        self.timestamp_ = timestamp
        self.vars_ = {}
        
    def fit( self, X ) :
        """ Defines numeric values for replacing categorical values. This should be applied to the train sample only. Otherwise, 
        this is a pure overfit.
        X: input dataframe """
        X[ self.timestamp_ ] = pd.to_numeric( X[ self.timestamp_ ] )
        for name in self.features_ :
            self.vars_[ name ] = X.groupby( name ).agg( number = ( name, 'count' ), recency = ( self.timestamp_, 'mean' ) ).sort_values( by=[ 'number', 'recency' ], ascending=[ True, False ] )
            self.vars_[ name ][ name+'_integer' ] = range( self.vars_[ name ].shape[ 0 ] )
            
    def transform( self, X ) :  
        """ Applies replacement of categorical variables with numeric ones and returns a new dataframe. A small portion of missing values may appear if
        sample_frac is set < 1.
        X: dataframe to transform """
        out = X.copy()
        for name in self.features_ :
            out = out.merge( self.vars_[ name ][ name+'_integer' ], on=name, how='left' )
            out[ name+'_integer' ] = np.int32( np.where( out[ name+'_integer' ].isnull(), 0, out[ name+'_integer' ] ) )
        return out