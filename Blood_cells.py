import numpy as np
import pandas as p

# data : 'dCt_values.tab'

class Blood_cells :


    def __init__(self, epsilon_, data_file) :
        #Parameter epsilon
        self.epsilon = epsilon_
        #Initializes the DataFrame
        self.df = self.import_data(data_file)
        print("Data : \n", self.df)
        a = self.size_()
        #Nb rows of the DF = nb of cells
        self.n = a[0]
        print("\nSize rows : ", self.n)
        #Nb of cols of the DF = nb genes
        self.p = a[1]
        print("Size col : ", self.p)
        #Similarity matrix
        self.L = self.simil_matrix(self.n, self.epsilon)
        print("\n Matrice de similarité L : \n", self.L)
        #Diagonal matrix D
        self.D = self.diag_matrix(self.n)
        print("\n Matrice D : \n", self.D)
        #Defines M_s matrix
        self.M_s = self.Ms_matrix(self.n)
        print("\n Matrice M_s : \n", self.M_s)
        #Decomposed M_s : V M_s V*
        res = self.decompose_Ms()
        self.M_s_decomp = res[0]
        print("\n Result M_s decomposed : \n", self.M_s_decomp)
        self.V = res[1]
        print("\n Matrice V : \n", self.V)
        self.lambda_ = res[2]
        print("\n Matrice lambda : \n", self.lambda_)


    #Imports data
    def import_data(self, file_) :
        data = p.read_csv(file_,sep='\t')
        df = p.DataFrame(data)

        return df.iloc[0:4,:]


    #Computes size n x p
    def size_(self) :

        return [len(self.df.index), len(self.df.columns)]


    #Cast dtype float64 into float
    def cast_float(self, i, j) :
        #Gets line n°i and j
        x_tab = self.df.iloc[i,1:self.p]
        y_tab = self.df.iloc[j,1:self.p]

        x = np.zeros((1,46))
        y = np.zeros((1,46))

        for m in x_tab :
            for n in range(46) :
                #If data = -14, means it hasn't been measured => 0
                if m == -14 :
                    x[0][n] = 0
                else :
                    x[0][n] = float(m)

        for m in y_tab :
            for n in range(46) :
                if m == -14 :
                    y[0][n] = 0
                else :
                    y[0][n] = float(m)

        return [x,y]



    #Computes the gaussian kernel
    def gaussian_kernel(self, x, y, eps) :
        cast = self.cast_float(x, y)
        x_ = cast[0]
        y_ = cast[1]
        #Euclidean norm, squared
        eucl_norm_sq = (np.linalg.norm(x_ - y_))**2
        k = np.exp(-eucl_norm_sq / eps)

        return k


    #Computes the similarity matrix L of size n (n being number of cells)
    def simil_matrix(self, n, eps) :
        L = np.zeros((n,n))
        for i in range(n) :
            for j in range(n) :
                if i != j and L[i, j] == 0 :
                    L[i, j] = self.gaussian_kernel(i, j, eps)
            L[j, i] = L[i, j]

        return L


    #Computes diagonal matrix
    def diag_matrix(self, n) :
        D = np.zeros((n,n))
        for i in range(n) :
            D[i, i] = np.sum(self.L[i, :], axis=0)
        return D

    #Computes the M_s matrix
    def Ms_matrix(self, n) :
        #Computes D^(-1/2)
        D_pow = self.D
        for i in range(n) :
            D_pow[i, i] = np.power(D_pow[i, i], -1/2)
        print("\n D^(-1/2) : \n", D_pow)

        #Performs the product of matrices
        tmp = np.dot(D_pow, self.L)
        M_s = np.dot(tmp, D_pow)

        return M_s


    #Decomposes M_s in M_s = V lambda V*
    def decompose_Ms(self) :
        #Returns a tuple of arrays
        result = np.linalg.eig(self.M_s)
        eigenvalues = result[0]
        eigenvectors = result[1]

        #Sorts the eigenvectors and their associated eigenvalue by decreasing order
        idx = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[idx]
        #Matrice V : composées des eigenvectors
        V = eigenvectors[:,idx]
        print("\n Ordonnés : \n Valeurs : \n", eigenvalues, "\n Vecteurs : \n", eigenvectors)

        #Matrice lambda : eigenvalues sur la diagonale
        l = len(eigenvalues)
        lambda_ = np.zeros((l,l))
        for i in range(l) :
            lambda_[i, i] = eigenvalues[i]
        print("Matrice lambda : \n", lambda_)

        #Compute
        M_s_decomp = np.dot(np.dot(V, lambda_),np.linalg.inv(V))

        return (M_s_decomp, V, lambda_)

    #Computes the eigenvectors of M
    def eigenvectors_M(self) :
        eig_M = self.V
        for i in range(n) :
            for j in range(n) :
                eig_M[i, j] = self.V[i,j] / np.pow(self.D[j,j], 1/2)
