import numpy as np
import pandas as p

# data : 'dCt_values.tab'

class Blood_cells :


    def __init__(self, epsilon_, data_file) :
        #Parameter epsilon
        self.epsilon = epsilon_
        #Initializes the DataFrame
        self.df = self.import_data(data_file)
        a = self.size_()
        #Nb rows of the DF = nb of cells
        self.n = a[0]
        #Nb of cols of the DF = nb genes
        self.p = a[1]
        #Similarity matrix
        self.L = self.simil_matrix(self.n, self.epsilon)
        print(self.L)


    #Imports data
    def import_data(self, file_) :
        data = p.read_csv(file_,sep='\t')
        df = p.DataFrame(data)
        #print(df)
        return df


    #Computes future size of self.L
    def size_(self) :
        return [len(self.df.index), len(self.df.columns)]


    #Computes the gaussian kernel
    def gaussian_kernel(self, x, y, eps) :
        norm_sq = np.sum((x - y)**2)
        k = np.exp(-norm_sq / eps)
        return k


    #Computes the similarity matrix L of size n (n being number of cells)
    def simil_matrix(self, n, eps) :
        L = np.zeros((n,n))
        for i in range(n-1) :
            print("i = ", i)
            x_i = self.df.iloc[i, 1:self.p] #line i = cell i
            x_j = self.df.iloc[i+1, 1:self.p] #line j
            L[i, i+1:] = self.gaussian_kernel(x_i, x_j, eps)
            L[i+1:, i] = L[i, i + 1:]
        return L
