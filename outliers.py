def outliers( DF ) : 
    
    """ Censoring outliers in a numeric dataframe """
    
    import numpy as np
    
    import pandas as pd
    
    OUT = pd.DataFrame( index=[ 0, 1 ], columns=list( DF ) )

    for Varname in list( DF ) :
        
        if DF[ Varname ].nunique() >= 3 and DF[ Varname ].dtype in [ 'float64', 'float32', 'int64', 'int32' ] :
            
            Q1 = DF[ Varname ].quantile( 0.25 )
            
            Q2 = DF[ Varname ].quantile( 0.50 )
  
            Q3 = DF[ Varname ].quantile( 0.75 )
            
            if ( Q2 - Q1 ) != 0 and ( Q3 - Q2 ) != 0 :
                
                Skewed = ( Q3 - Q2 ) / ( Q2 - Q1 ) >= 1.2 or ( Q2 - Q1 ) / ( Q3 - Q2 ) >= 1.2
                
            else:
                
                Skewed = True
            
            if Skewed and DF[ Varname ].min() >= 0 : 
                
                Lower = 0
                
                if Q3 > 0 :
                
                    Upper = Q3 * 6
                    
                else :
                    
                    Upper = 1
                
            else :
        
                Iqr = Q3 - Q1
                
                if Iqr > 0 :
            
                    Lower = Q1 - 1.5 * Iqr
                
                    Upper = Q3 + 1.5 * Iqr
                    
                else :
                    
                    Lower = Q1 - 1
                
                    Upper = Q3 + 1
            
            OUT.loc[ 1, Varname ] = Lower
            
            OUT.loc[ 0, Varname ] = Upper
            
            Condlow = np.logical_or( DF[ Varname ] > Lower, DF[ Varname ].isnull() )
            
            Condup = np.logical_or( DF[ Varname ] < Upper, DF[ Varname ].isnull() )
            
            DF[ Varname ].where( Condlow, Lower, inplace=True )
            
            DF[ Varname ].where( Condup, Upper, inplace=True )
            
    return OUT




def bounds( DF ) : 
    
    """ Producing threshold values for censoring outliers """
    
    import numpy as np
    
    import pandas as pd
    
    OUT = pd.DataFrame( index=[ 0, 1 ], columns=list( DF ) )

    for Varname in list( DF ) :
        
        if DF[ Varname ].nunique() >= 3 and DF[ Varname ].dtype in [ 'float64', 'float32', 'int64', 'int32' ] :
            
            Q1 = DF[ Varname ].quantile( 0.25 )
            
            Q2 = DF[ Varname ].quantile( 0.50 )
  
            Q3 = DF[ Varname ].quantile( 0.75 )
            
            if ( Q2 - Q1 ) != 0 and ( Q3 - Q2 ) != 0 :
                
                Skewed = ( Q3 - Q2 ) / ( Q2 - Q1 ) >= 1.2 or ( Q2 - Q1 ) / ( Q3 - Q2 ) >= 1.2
                
            else:
                
                Skewed = True
            
            if Skewed and DF[ Varname ].min() >= 0 : 
                
                Lower = 0
                
                if Q3 > 0 :
                
                    Upper = Q3 * 6
                    
                else :
                    
                    Upper = 1
                
            else :
        
                Iqr = Q3 - Q1
                
                if Iqr > 0 :
            
                    Lower = Q1 - 1.5 * Iqr
                
                    Upper = Q3 + 1.5 * Iqr
                    
                else :
                    
                    Lower = Q1 - 1
                
                    Upper = Q3 + 1
            
            OUT.loc[ 1, Varname ] = Lower
            
            OUT.loc[ 0, Varname ] = Upper
            
    return OUT




def censor( DF, IN ) :
    
    """ Censoring outliers in a test dataframe """
    
    import numpy as np
    
    import pandas as pd
    
    for Varname in list( IN ) :
        
        if DF[ Varname ].nunique() >= 3 and DF[ Varname ].dtype in [ 'float64', 'float32', 'int64', 'int32' ] :
            
            Condlow = np.logical_or( DF[ Varname ] > IN.loc[ 1, Varname ], DF[ Varname ].isnull() )
            
            Condup = np.logical_or( DF[ Varname ] < IN.loc[ 0, Varname ], DF[ Varname ].isnull() )
            
            DF[ Varname ].where( Condlow, IN.loc[ 1, Varname ], inplace=True )
            
            DF[ Varname ].where( Condup, IN.loc[ 0, Varname ], inplace=True )