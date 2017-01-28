import numpy as np
import pandas as p


#import data
data = p.read_csv('dCt_values.tab',sep='\t')
df = p.DataFrame(data)
print(df)
nb_col = len(df.columns)
#print(nb_col)

#Computes the gaussian kernel
def gaussian_kernel(x, y, eps) :
    norm_sq = (x - y)**2
    value = np.exp(-norm_sq / eps)
    return value

#Computes the similarity matrix L of size n
def simil_matrix(n) :
    L = np.zeroes((n,n))
    for i in range(n) :
        L[i,i+1:] = gaussian_kernel(df)
        L[i+1:,i] = grid[i,i+1:]
