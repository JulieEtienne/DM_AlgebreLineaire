import numpy as np
import pandas as p


#import data
data = p.read_csv('dCt_values.tab',sep='\t')
df = p.DataFrame(data)
#print(df)
nb_col = len(df.columns)
#print(nb_col)

def gaussian_kernel(x, y, eps) :
    norm_sq = (x - y) * (x - y)
    value = np.exp(-norm_sq / eps)
    return value
