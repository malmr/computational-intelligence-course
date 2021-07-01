#Filename: HW5_skeleton.py
#Author: Christian Knoll
#Edited: May, 2018

import numpy as np
import matplotlib.cm as cm
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import scipy.stats as stats

from scipy.stats import multivariate_normal
import pdb

import sklearn
from sklearn import datasets


import matplotlib.pyplot as plt
from matplotlib.mlab import bivariate_normal
from mpl_toolkits.mplot3d import Axes3D


#--------------------------------------------------------------------------------
# Assignment 5
def main():
    # -------SETTINGS---------
    # ------------------------
    run_2d = True
    run_em = True
    run_km = True
    run_4d = True
    run_4d_em = True
    run_4d_km = True
    forceDiagCov_4d = False  # default False. For ex. 2.2 2)
    run_pca = False


    # 0) Get the input
    ## (a) load the modified iris data
    data, labels = load_iris_data()
    labels_before = labels

    ## (b) construct the datasets 
    x_2dim = data[:, [0,2]]
    x_4dim = data

    if run_pca:
        x_2dim, var_exp = PCA(data,nr_dimensions=2,whitening=True)
        cov_pca = np.cov(x_2dim, rowvar=False)
        print("cov pca: ",cov_pca)

    plot_iris_data(x_2dim,labels)
    plt.show()


    #TODO: run each algorithms 100 times, and pick the result with max likelihood
    runs = 2    # set to 100 to then select maximum likelihood
    LL = 0
    cum_dist = 0
    LL_runs = np.zeros(runs)
    CD_runs = np.zeros(runs)

    for l in range(runs):
        print('--- run ', l,' ---')
        # ------------------------
        # 1) Consider a 2-dim slice of the data and evaluate the EM- and the KMeans- Algorithm
        if run_2d:
            if run_pca:
                scenario = 3
            else:
                scenario = 1
            dim = 2
            nr_components = 3

            tol = 10^(-5)  # tolerance
            max_iter = 100  # maximum iterations for GN

            if run_em:
                # EM ALGORITHM
                #

                (alpha_0, mean_0, cov_0) = init_EM(dimension = dim, nr_components= nr_components, scenario=scenario)

                (alpha, mean, cov, LL, labels) = EM(x_2dim,nr_components, alpha_0, mean_0, cov_0, max_iter, tol)

                # plot assigned classes
                plot_iris_data(x_2dim ,labels)
                plt.title('Classified data with EM (2D)')
                plt.show()

                # iteration plot
                plt.plot(LL)
                plt.title('Log-likelihood over iterations (2D)')
                plt.show()

                # plot gauss
                plot_iris_data(x_2dim, labels)
                for k in range(nr_components):
                    if run_pca:
                        plot_gauss_contour(mean[:, k], cov[k, :, :], -4, 4, -4, 4, 100)
                    else:
                        plot_gauss_contour(mean[:, k], cov[k, :, :], 4, 8, 0, 8, 100)
                plt.title('Classified data with respective Gauss components (2D)')
                plt.show()

                printmisclassified(labels,labels_before)

            if run_km:
                # K-MEANS ALGORITHM
                #
                initial_centers = init_k_means(dimension = dim, nr_clusters=nr_components, scenario=scenario)
                (centers, cum_dist, labels) = k_means(x_2dim, nr_components, initial_centers, max_iter, tol)

                plot_iris_data(x_2dim ,labels)
                plt.scatter(centers[0,:],centers[1,:],marker="x")
                plt.title('Classified data with k-means (2D)')
                plt.show()

                plt.plot(cum_dist)
                plt.title('Cumulative distance over iterations (2D)')
                plt.show()

                printmisclassified(labels,labels_before)



        #------------------------
        # 2) Consider 4-dimensional data and evaluate the EM- and the KMeans- Algorithm
        if run_4d:
            scenario = 2
            dim = 4
            nr_components = 3

            tol = 10^(-5)  # tolerance
            max_iter = 100  # maximum iterations for GN

            if run_4d_em:
                # EM ALGORITHM
                #
                (alpha_0, mean_0, cov_0) = init_EM(dimension = dim, nr_components= nr_components, scenario=scenario)
                (alpha, mean, cov, LL, labels) = EM(x_4dim,nr_components, alpha_0, mean_0, cov_0, max_iter, tol, forceDiagCov_4d)

                # plot assigned classes
                plot_iris_data(x_2dim ,labels)
                plt.title('Classified data with EM (4D)')
                plt.show()

                # iteration plot
                plt.plot(LL)
                plt.title('Log-likelihood over iterations (4D)')
                plt.show()

                # plot gauss
                plot_iris_data(x_2dim ,labels)
                for k in range(nr_components):
                    plot_gauss_contour(mean[:, k], cov[k, :, :], 4, 8, 0, 8, 100)
                plt.title('Classified data with respective Gauss components (4D)')
                plt.show()

                printmisclassified(labels,labels_before)

            if run_4d_km:
                # K-MEANS ALGORITHM
                #
                initial_centers = init_k_means(dimension = dim, nr_clusters=nr_components, scenario=scenario)
                (centers, cum_dist, labels) = k_means(x_4dim, nr_components, initial_centers, max_iter, tol)

                plot_iris_data(x_2dim ,labels)
                plt.scatter(centers[0,:],centers[1,:],marker="x")
                plt.title('Classified data with k-means (4D)')
                plt.show()

                plt.plot(cum_dist)
                plt.title('Cumulative distance over iterations (4D)')
                plt.show()

                printmisclassified(labels,labels_before)


        # evaluate the max likelihood (for EM) or min cumulative distance (for k-means) of the runs
        if (run_4d_em and run_4d) or (run_em and run_2d):
            LL_runs[l] = LL[-1]
        elif (run_4d_km and run_4d) or (run_km and run_2d):
            CD_runs[l] = cum_dist[-1]

    if (run_4d_em and run_4d) or (run_em and run_2d):
        [max_LL, optimal_run] = [np.max(LL_runs), np.argmax(LL_runs)]
        print('Maximal likelihood of all runs ', max_LL, ' at run ',optimal_run)
    elif (run_4d_km and run_4d) or (run_km and run_2d):
        [min_CD, optimal_run] = [np.min(CD_runs), np.argmin(CD_runs)]
        print('Minimal cum distance of all runs ', min_CD, ' at run ',optimal_run)


    #4) SAMPLES FROM GAUSSIAN MIXTURE MODEL
    nr_components = 5
    dimension = 2
    N = 10000
    # equally initialized alpha
    alpha = np.ones(nr_components)/nr_components

    # unequal initialized alpha
    #alpha = np.multiply(np.ones(nr_components),[1/24,5/6,1/24,1/24,1/24])

    mu = np.random.rand(dimension, nr_components) * 15
    # spread means
    cov = np.array([np.eye(dimension)] * nr_components)
    y = sample_GMM(alpha,mu,cov,N)

    plt.plot(y[:,0],y[:,1],'x')
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.title('Drawn samples from GMM distribution')
    plt.show()

    pdb.set_trace()

#--------------------------------------------------------------------------------
#--------------------------------------------------------------------------------
def init_EM(dimension=2,nr_components=3, scenario=None):
    """ initializes the EM algorithm
    Input: 
        dimension... dimension D of the dataset, scalar
        nr_components...scalar
        scenario... optional parameter that allows to further specify the settings, scalar
    Returns:
        alpha_0... initial weight of each component, 1 x nr_components
        mean_0 ... initial mean values, D x nr_components
        cov_0 ...  initial covariance for each component, D x D x nr_components"""
    # TODO choose suitable initial values for each scenario
    alpha_0 = np.array([1/nr_components] * nr_components)
    mean_0 = np.random.rand(dimension, nr_components)
    if scenario == 1:
        # 0 2
        mean_0 = mean_0 /2
        mean_0[0, :] += 6
        mean_0[1, :] += 4
    elif scenario == 2:
        # 0 1 2 3
        mean_0 = mean_0 / 2
        mean_0[0, :] += 6
        mean_0[1, :] += 2.75
        mean_0[2, :] += 4
        mean_0[3, :] += 1.5
    elif scenario == 3:
        mean_0 = mean_0 / 2

    print('Initial mean:')
    print(mean_0)

    # identity matrix
    cov_0 = np.array([np.eye(dimension)] * nr_components)

    # positive definite matrix:
    #cov_0 = np.array([sklearn.datasets.make_spd_matrix(dimension),sklearn.datasets.make_spd_matrix(dimension),sklearn.datasets.make_spd_matrix(dimension)])

    return (alpha_0,mean_0,cov_0)

#--------------------------------------------------------------------------------
def EM(X,K,alpha_0,mean_0,cov_0, max_iter, tol, forceDiagCov_4d = False):
    """ perform the EM-algorithm in order to optimize the parameters of a GMM
    with K components
    Input:
        X... samples, nr_samples x dimension (D)
        K... nr of components, scalar
        alpha_0... initial weight of each component, 1 x K
        mean_0 ... initial mean values, D x K
        cov_0 ...  initial covariance for each component, D x D x K        
    Returns:
        alpha... final weight of each component, 1 x K
        mean...  final mean values, D x K
        cov...   final covariance for ech component, D x D x K
        log_likelihood... log-likelihood over all iterations, nr_iterations x 1
        labels... class labels after performing soft classification, nr_samples x 1"""
    # compute the dimension 
    D = X.shape[1]
    nr_samples = X.shape[0]
    assert D == mean_0.shape[0]
    #TODO: iteratively compute the posterior and update the parameters

    alpha = alpha_0
    alpha = np.reshape(alpha,(K,1))
    alpha_new = np.zeros((K,1))

    mean = mean_0
    mean_new = np.zeros((D,K))

    cov = cov_0
    cov_new = np.zeros(np.shape(cov_0))

    r = np.zeros((nr_samples,K))
    LL = 0
    LL_diff = 1
    labels = np.zeros(nr_samples)
    normal_vector = np.zeros((K,1))
    N = np.zeros((K,1))
    LL_list = []
    LL_prev = 0

    print('Start EM algorithm...')
    for i in range(max_iter):
        if np.abs(LL_diff) < tol:
            print('Tolerance of ', tol, ' reached.')
            break
        # expectation
        for n in range(nr_samples):
            for k_ in range(K):
                normal_vector[k_] = multivariate_normal.pdf(X[n, :], mean[:, k_], cov[k_, :, :])
            for k in range(K):
                r[n][k] = alpha[k] * multivariate_normal.pdf(X[n, :], mean[:, k], cov[k, :, :]) / np.dot(alpha.T,
                                                                                               normal_vector)

        # maximization
        for n in range(nr_samples):
            for k in range(K):
                N[k] = np.sum(r[:,k])
                mean_new[:,k] = 1/N[k] * np.dot(r[:,k],X)
                cov_new[k,:,:] = 1/N[k] * np.dot(np.multiply(np.subtract(X,mean_new[:,k]).T,r[:,k]), np.subtract(X,mean_new[:,k]))
                if forceDiagCov_4d:
                    for q in range(cov_new.shape[0]):
                        for p in range(cov_new.shape[1]):
                            if q != p:
                                cov_new[k,p,q] = 0

                alpha_new[k] = N[k]/nr_samples

        mean = np.matrix.copy(mean_new)
        cov = np.matrix.copy(cov_new)
        alpha =np.matrix.copy(alpha_new)


        LL = 0
        # likelihood calculation
        for k in range(K):
            LL += alpha[k]*likelihood_multivariate_normal(X, mean[:,k],cov[k,:,:])
        LL = np.sum(np.log(LL))

        LL_diff = LL-LL_prev
        LL_prev = LL
        #print(i, LL_diff)
        LL_list.append(LL)
    for n in range(nr_samples):
        labels[n] = np.argmax(r[n,:])
    print('EM finished.')

    LL = np.array(LL_list)

    # reassign labels
    assigned_label = reassign_class_labels(labels)

    labels_new = np.zeros(np.shape(labels))
    for i in range(nr_samples):
        labels_new[i] = assigned_label[labels.astype(int)[i]]
    return (alpha, mean, cov, LL, labels_new)


#--------------------------------------------------------------------------------
def init_k_means(dimension=2, nr_clusters=3, scenario=None):
    """ initializes the k_means algorithm
    Input: 
        dimension... dimension D of the dataset, scalar
        nr_clusters...scalar
        scenario... optional parameter that allows to further specify the settings, scalar
    Returns:
        initial_centers... initial cluster centers,  D x nr_clusters"""
    # TODO chosse suitable inital values for each scenario
    centers_0 = np.random.rand(dimension, nr_clusters)

    if scenario == 1:
        # 0 2
        centers_0 = centers_0 / 2
        centers_0[0, :] += 6
        centers_0[1, :] += 4
    elif scenario == 2:
        # 0 1 2 3
        centers_0 = centers_0 / 30
        centers_0[0, :] += 6
        centers_0[1, :] += 2.75
        centers_0[2, :] += 4
        centers_0[3, :] += 1.5
    elif scenario == 3:
        centers_0 = centers_0 / 2


    print('Initial center:')
    print(centers_0)
    return(centers_0)
#--------------------------------------------------------------------------------
def k_means(X,K, centers_0, max_iter, tol):
    """ perform the KMeans-algorithm in order to cluster the data into K clusters
    Input:
        X... samples, nr_samples x dimension (D)
        K... nr of clusters, scalar
        centers_0... initial cluster centers,  D x nr_clusters
    Returns:
        centers... final centers, D x nr_clusters
        cumulative_distance... cumulative distance over all iterations, nr_iterations x 1
        labels... class labels after performing hard classification, nr_samples x 1"""
    D = X.shape[1]
    nr_samples = X.shape[0]
    assert D == centers_0.shape[0]
    #TODO: iteratively update the cluster centers

    centers = centers_0
    c = np.zeros((nr_samples, K))
    labels = np.zeros(nr_samples)
    N=0
    cum_dist = np.zeros((max_iter,1))

    print('Start k-means algorithm...')
    for i in range(max_iter):
        #Assignment of points
        for n in range(nr_samples):
            for k in range(K):
                c[n][k] = np.linalg.norm(X[n,:].T-centers[:,k])
            labels[n] = np.argmin(c[n,:])
            cum_dist[i] += np.min(c[n,:])

        #Update centers
        centers = np.zeros((D,K))
        for k in range(K):
            for n in range(nr_samples):
                if labels[n] == k:
                    N += 1
                    centers[:,k] += X[n,:]
            centers[:,k] = centers[:,k]/N
            N = 0
    print('K-means finished.')

    # reassign labels
    assigned_label = reassign_class_labels(labels)
    labels_new = np.zeros(np.shape(labels))
    for i in range(nr_samples):
        labels_new[i] = assigned_label[labels.astype(int)[i]]

    return(centers,cum_dist,labels_new.astype(int))

#--------------------------------------------------------------------------------
def PCA(data,nr_dimensions=None, whitening=True):
    """ perform PCA and reduce the dimension of the data (D) to nr_dimensions
    Input:
        data... samples, nr_samples x D
        nr_dimensions... dimension after the transformation, scalar
        whitening... False -> standard PCA, True -> PCA with whitening
        
    Returns:
        transformed data... nr_samples x nr_dimensions
        variance_explained... amount of variance explained by the the first nr_dimensions principal components, scalar"""
    if nr_dimensions is not None:
        dim = nr_dimensions
    else:
        dim = 2

    # mean center the data
    data -= data.mean(axis=0)
    cov = np.cov(data, rowvar=False)
    eig_vals, eig_vecs = np.linalg.eigh(cov)
    # sort eigenvalue in decreasing order
    idx = np.argsort(eig_vals)[::-1]
    eig_vecs = eig_vecs[:, idx]
    eig_vals = eig_vals[idx]
    # select the first dim eigenvectors
    eig_vecs = eig_vecs[:, :dim]

    variance_explained = np.sum(eig_vals[:dim])/np.sum(eig_vals)

    if whitening:
        return np.dot(np.dot(np.diag(1/np.sqrt(eig_vals[:dim])),eig_vecs.T),data.T).T, variance_explained
    else:
        return np.dot(eig_vecs.T, data.T).T, variance_explained
    #TODO: Estimate the principal components and transform the data
    # using the first nr_dimensions principal_components
    
    
    #TODO: Have a look at the associated eigenvalues and compute the amount of varianced explained
#--------------------------------------------------------------------------------
#--------------------------------------------------------------------------------
# Helper Functions
#--------------------------------------------------------------------------------
def load_iris_data():
    """ loads and modifies the iris data-set
    Input: 
    Returns:
        X... samples, 150x4
        Y... labels, 150x1"""
    iris = datasets.load_iris()
    X = iris.data
    X[50:100,2] =  iris.data[50:100,2]-0.25
    Y = iris.target    
    return X,Y   
#--------------------------------------------------------------------------------
def plot_iris_data(data,labels):
    """ plots a 2-dim slice according to the specified labels
    Input:
        data...  samples, 150x2
        labels...labels, 150x1"""

    plt.scatter(data[labels==0,0], data[labels==0,1], label='Iris-Setosa')
    plt.scatter(data[labels==1,0], data[labels==1,1], label='Iris-Versicolor')
    plt.scatter(data[labels==2,0], data[labels==2,1], label='Iris-Virgnica')

    plt.legend()
    #plt.show()
#--------------------------------------------------------------------------------
def likelihood_multivariate_normal(X, mean, cov, log=False):
   """Returns the likelihood of X for multivariate (d-dimensional) Gaussian 
   specified with mu and cov.
   
   X  ... vector to be evaluated -- np.array([[x_00, x_01,...x_0d], ..., [x_n0, x_n1, ...x_nd]])
   mean ... mean -- [mu_1, mu_2,...,mu_d]
   cov ... covariance matrix -- np.array with (d x d)
   log ... False for likelihood, true for log-likelihood
   """

   dist = multivariate_normal(mean, cov)
   if log is False:
       P = dist.pdf(X)
   elif log is True:
       P = dist.logpdf(X)

   return P 

#--------------------------------------------------------------------------------
def plot_gauss_contour(mu,cov,xmin,xmax,ymin,ymax,nr_points,title="Title"):   
    """ creates a contour plot for a bivariate gaussian distribution with specified parameters
    
    Input:
      mu... mean vector, 2x1
      cov...covariance matrix, 2x2
      xmin,xmax... minimum and maximum value for width of plot-area, scalar
      ymin,ymax....minimum and maximum value for height of plot-area, scalar
      nr_points...specifies the resolution along both axis
      title... title of the plot (optional), string"""
    
	#npts = 100
    delta_x = float(xmax-xmin) / float(nr_points)
    delta_y = float(ymax-ymin) / float(nr_points)
    x = np.arange(xmin, xmax, delta_x)
    y = np.arange(ymin, ymax, delta_y)
    
    X, Y = np.meshgrid(x, y)
    Z = mlab.bivariate_normal(X,Y,np.sqrt(cov[0][0]),np.sqrt(cov[1][1]),mu[0], mu[1], cov[0][1])
    plt.plot([mu[0]],[mu[1]],'r+') # plot the mean as a single point
    CS = plt.contour(X, Y, Z)
    plt.clabel(CS, inline=1, fontsize=10)
    plt.title(title)
    #plt.show()
    return
#--------------------------------------------------------------------------------    
def sample_discrete_pmf(X, PM, N):
    """Draw N samples for the discrete probability mass function PM that is defined over 
    the support X.
       
    X ... Support of RV -- np.array([...])
    PM ... P(X) -- np.array([...])
    N ... number of samples -- scalar
    """
    assert np.isclose(np.sum(PM), 1.0)
    assert all(0.0 <= p <= 1.0 for p in PM)
    
    y = np.zeros(N)
    cumulativePM = np.cumsum(PM) # build CDF based on PMF
    offsetRand = np.random.uniform(0, 1) * (1 / N) # offset to circumvent numerical issues with cumulativePM
    comb = np.arange(offsetRand, 1 + offsetRand, 1 / N) # new axis with N values in the range ]0,1[
    
    j = 0
    for i in range(0, N):
        while comb[i] >= cumulativePM[j]: # map the linear distributed values comb according to the CDF
            j += 1	
        y[i] = X[j]
        
    return np.random.permutation(y) # permutation of all samples


#--------------------------------------------------------------------------------
def sample_GMM(alpha, mu, cov, N):
    """Draw N samples from the two-dimensional Gaussian Mixture distribution defined by alpha, mu and cov.
    """
    nr_components = len(alpha)
    component_idx = np.random.choice(np.arange(nr_components), size=N, replace=True, p=alpha)
    y = np.zeros((N,2))

    # draw samples from components with their respective mu and cov
    for (i,k) in enumerate(component_idx):
        y[i,:] = np.random.multivariate_normal(mu[:,k],cov[k,:,:])

    return y


#--------------------------------------------------------------------------------
def reassign_class_labels(labels):
    """ reassigns the class labels in order to make the result comparable. 
    new_labels contains the labels that can be compared to the provided data,
    i.e., new_labels[i] = j means that i corresponds to j.
    Input:
        labels... estimated labels, 150x1
    Returns:
        new_labels... 3x1"""
    class_assignments = np.array([[np.sum(labels[0:50]==0)   ,  np.sum(labels[0:50]==1)   , np.sum(labels[0:50]==2)   ],
                                  [np.sum(labels[50:100]==0) ,  np.sum(labels[50:100]==1) , np.sum(labels[50:100]==2) ],
                                  [np.sum(labels[100:150]==0),  np.sum(labels[100:150]==1), np.sum(labels[100:150]==2)]])
    new_labels = np.array([np.argmax(class_assignments[:,0]),
                           np.argmax(class_assignments[:,1]),
                           np.argmax(class_assignments[:,2])])
    return new_labels


def printmisclassified(labels, labels_before):
    nr_wrong_classified = 0
    nr_wrong_classified0 = 0
    nr_wrong_classified1 = 0
    nr_wrong_classified2 = 0

    # misclassified
    for i in range(len(labels)):
        if labels[i] != labels_before[i]:
            nr_wrong_classified += 1
            if labels[i] == 0:
                nr_wrong_classified0 += 1
            elif labels[i] == 1:
                nr_wrong_classified1 += 1
            elif labels[i] == 2:
                nr_wrong_classified2 += 1

    print('wrong classified (general)', nr_wrong_classified)
    print('wrong classified 0', nr_wrong_classified0)
    print('wrong classified 1', nr_wrong_classified1)
    print('wrong classified 2', nr_wrong_classified2)

#--------------------------------------------------------------------------------
def sanity_checks():
    # likelihood_multivariate_normal
    mu =  [0.0, 0.0]
    cov = [[1, 0.2],[0.2, 0.5]]
    x = np.array([[0.9, 1.2], [0.8, 0.8], [0.1, 1.0]])
    P = likelihood_multivariate_normal(x, mu, cov)

    #plot_gauss_contour(mu, cov, -2, 2, -2, 2,100, 'Gaussian')

    # sample_discrete_pmf
    PM = np.array([0.2, 0.5, 0.2, 0.1])
    N = 1000
    X = np.array([1, 2, 3, 4])
    Y = sample_discrete_pmf(X, PM, N)
    
    print('Nr_1:', np.sum(Y == 1),
          'Nr_2:', np.sum(Y == 2),
          'Nr_3:', np.sum(Y == 3),
          'Nr_4:', np.sum(Y == 4))
    
    # re-assign labels
    class_labels_unordererd = np.array([2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
       2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
       2, 2, 2, 2, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1,
       0, 0, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0,
       0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 0, 0,
       0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 0])
    new_labels = reassign_class_labels(class_labels_unordererd)
    reshuffled_labels =np.zeros_like(class_labels_unordererd)
    reshuffled_labels[class_labels_unordererd==0] = new_labels[0]
    reshuffled_labels[class_labels_unordererd==1] = new_labels[1]
    reshuffled_labels[class_labels_unordererd==2] = new_labels[2]


    
#--------------------------------------------------------------------------------
#--------------------------------------------------------------------------------
if __name__ == '__main__':
    
    sanity_checks()
    main()
