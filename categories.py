import numpy as np
import pandas as pd

class Categories( object ) :
    
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
    
    def __init__( self, features, label, sample_frac=0.8, dummy_number=30 ) :
        self.features_ = features
        self.label_ = label
        self.sample_frac_ = sample_frac
        self.dummy_number_ = dummy_number
        self.vars_ = {}
        self.avg_ = np.nan
        
    def fit( self, X ) :
        """ Defines numeric values for replacing categorical values. This should be applied to the train sample only. Otherwise, 
        this is a pure overfit.
        X: input dataframe """
        self.avg_ = np.mean( X[ self.label_ ].values )
        for name in self.features_ :
            self.vars_[ name ] = X.loc[ X[ name ].notnull() ].sample( frac=self.sample_frac_ ).groupby( name )[ self.label_ ].agg( [ 'count', 'sum' ] )
            self.vars_[ name ][ 'count' ] = self.vars_[ name ][ 'count' ] + self.dummy_number_
            self.vars_[ name ][ 'sum' ] = self.vars_[ name ][ 'sum' ] + self.dummy_number_ * self.avg_
            self.vars_[ name ][ 'value' ] = self.vars_[ name ][ 'sum' ] / self.vars_[ name ][ 'count' ]
            
    def transform( self, X ) :  
        """ Applies replacement of categorical variables with numeric ones and returns a new dataframe. A small portion of missing values may appear if
        sample_frac is set < 1.
        X: dataframe to transform """
        out = X.copy()
        for name in self.features_ :
            notna = out[ name ].notnull().values
            out.loc[ notna, name ] = self.vars_[ name ][ 'value' ][ out.loc[ notna, name ] ].values
            out[ name ] = pd.to_numeric( out[ name ] )
        return out