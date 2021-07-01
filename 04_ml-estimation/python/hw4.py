#Filename: HW4_skeleton.py
#Author: Christian Knoll, Florian Kaum
#Edited: May, 2018

import numpy as np
import matplotlib.cm as cm
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import scipy.stats as stats

from scipy.stats import multivariate_normal

#--------------------------------------------------------------------------------
# Assignment 4
#
# choose the scenario
#scenario = 1    # all anchors are Gaussian
#scenario = 2     # 1 anchor is exponential, 3 are Gaussian
scenario = 3    # all anchors are exponential


def main():
    # specify position of anchors
    p_anchor = np.array([[5,5],[-5,5],[-5,-5],[5,-5]])
    nr_anchors = np.size(p_anchor,0)

    # position of the agent for the reference mearsurement
    p_ref = np.array([[0,0]])
    # true position of the agent (has to be estimated)
    p_true = np.array([[2,-4]])

    #plot_anchors_and_agent(nr_anchors, p_anchor, p_true, p_ref)

    # load measured data and reference measurements for the chosen scenario
    data,reference_measurement = load_data(scenario)

    # get the number of measurements
    assert(np.size(data,0) == np.size(reference_measurement,0))
    nr_samples = np.size(data,0)

    #1) ML estimation of model parameters
    #TODO
    params = parameter_estimation(reference_measurement,nr_anchors,p_anchor,p_ref)

    #2) Position estimation using least squares
    #TODO
    position_estimation_least_squares(data,nr_anchors,p_anchor, p_true, True)

    if(scenario == 3):
        # TODO: don't forget to plot joint-likelihood function for the first measurement

        #3) Postion estimation using numerical maximum likelihood
        #TODO
        position_estimation_numerical_ml(data,nr_anchors,p_anchor, params, p_true)

        #4) Position estimation with prior knowledge (we roughly know where to expect the agent)
        #TODO
        # specify the prior distribution
        prior_mean = p_true
        prior_cov = np.eye(2)
        position_estimation_bayes(data,nr_anchors,p_anchor,prior_mean,prior_cov, params, p_true)

    pass

#--------------------------------------------------------------------------------
#--------------------------------------------------------------------------------
def parameter_estimation(reference_measurement,nr_anchors,p_anchor,p_ref):
    """ estimate the model parameters for all 4 anchors based on the reference measurements, i.e., for anchor i consider reference_measurement[:,i]
    Input:
        reference_measurement... nr_measurements x nr_anchors
        nr_anchors... scalar
        p_anchor... position of anchors, nr_anchors x 2
        p_ref... reference point, 1x2 """
    plt.hist(reference_measurement[:,1],bins='auto')
    plt.xlabel('distance')
    plt.ylabel('frequency')
    plt.show()
    params = np.zeros([1, nr_anchors])
    means = np.zeros((nr_anchors, 1))
    variances = np.zeros((nr_anchors,1))

    for i in range(nr_anchors):
        means[i] = np.mean(reference_measurement[:,i])

    for i in range(nr_anchors):
        variances[i] = np.var(reference_measurement[:,i])

    params = variances

    if scenario == 2:
        params[0] = 1 / means[0]

    elif scenario == 3:
        params = 1/means

    return params
#--------------------------------------------------------------------------------
def position_estimation_least_squares(data,nr_anchors,p_anchor, p_true, use_exponential):
    """estimate the position by using the least squares approximation.
    Input:
        data...distance measurements to unkown agent, nr_measurements x nr_anchors
        nr_anchors... scalar
        p_anchor... position of anchors, nr_anchors x 2
        p_true... true position (needed to calculate error) 2x2
        use_exponential... determines if the exponential anchor in scenario 2 is used, bool"""
    nr_samples = np.size(data,0)

    tol = 0.00000000001   # tolerance
    max_iter = 100**10  # maximum iterations for GN

    p_start = [np.random.uniform(-5,5),np.random.uniform(-5,5)]
    print(p_start, 'p_start')
    p_est = np.zeros((2000,2))
    for i in range(nr_samples):
        p_est[i] = np.reshape(least_squares_GN(p_anchor, p_start, data[i,:], max_iter, tol),(2))

    error = p_true-p_est
    mean = np.mean(error)
    var = np.var(error)

    #print('p_est: ',p_est, len(p_est))
    #print('error: ',error)

    print('Error mean: ',mean, 'Error var: ',var)


    #########
    # plot scater plot of estimated errors
    #########
    plot_estimated_positions(nr_samples, p_est, p_true)

    #########
    # plot gauss contour
    #########
    mean_est = np.mean(p_est, axis=0)
    cov_est = np.cov(p_est.T)
    #print('cov matrix', cov_est)
    plot_gauss_contour(mean_est, cov_est, -6, 6, -6, 6, title="Gauss Contour Plot")

    #########
    # plot CDF
    #########
    N = 1000
    realization = np.random.normal(mean, var, N)
    Fx,x = ecdf(realization)
    plt.plot(x,Fx)
    plt.title('Cumulative distribution function of the position estimation error')
    plt.xlabel("estimation error")
    plt.ylabel("probability")
    plt.show()

    pass


#--------------------------------------------------------------------------------
def position_estimation_numerical_ml(data,nr_anchors,p_anchor, lambdas, p_true):
    """ estimate the position by using a numerical maximum likelihood estimator
    Input:
        data...distance measurements to unkown agent, nr_measurements x nr_anchors
        nr_anchors... scalar
        p_anchor... position of anchors, nr_anchors x 2
        lambdas... estimated parameters (scenario 3), nr_anchors x 1
        p_true... true position (needed to calculate error), 2x2 """
    #TODO
    pass
#--------------------------------------------------------------------------------
def position_estimation_bayes(data,nr_anchors,p_anchor,prior_mean,prior_cov,lambdas, p_true):
    """ estimate the position by accounting for prior knowledge that is specified by a bivariate Gaussian
    Input:
         data...distance measurements to unkown agent, nr_measurements x nr_anchors
         nr_anchors... scalar
         p_anchor... position of anchors, nr_anchors x 2
         prior_mean... mean of the prior-distribution, 2x1
         prior_cov... covariance of the prior-dist, 2x2
         lambdas... estimated parameters (scenario 3), nr_anchors x 1
         p_true... true position (needed to calculate error), 2x2 """
    # TODO
    pass
#--------------------------------------------------------------------------------
def least_squares_GN(p_anchor,p_start, r, max_iter, tol):
    """ apply Gauss Newton to find the least squares solution
    Input:
        p_anchor... position of anchors, nr_anchors x 2
        p_start... initial position, 2x1
        r... distance_estimate, nr_anchors x 1
        max_iter... maximum number of iterations, scalar
        tol... tolerance value to terminate, scalar"""

    p_iter_prev = np.reshape(p_start,(2,1))
    p_iter = np.zeros((2,1))

    for i in range(max_iter):
        J = jacobian(p_anchor, p_iter_prev)
        p_iter = p_iter_prev - np.linalg.pinv(J).dot(np.reshape((r - np.sqrt((p_anchor[:,0] - p_iter_prev[0]) ** 2 + (p_anchor[:,1] - p_iter_prev[1]) ** 2)),(4,1)))

        if (np.linalg.norm((p_iter_prev - p_iter) < tol)):
            return p_iter
        p_iter_prev = p_iter
    return p_iter

#--------------------------------------------------------------------------------
#--------------------------------------------------------------------------------
# Helper Functions
#--------------------------------------------------------------------------------
def jacobian(p_anchor, p_iter_prev):
    """ Jacobian matrix calculation """
    J = np.zeros((4,2))

    #print(p_iter_prev, 'piterprev', ' shape', np.shape(p_iter_prev))
    for i in range(np.shape(p_anchor)[0]):
        J[i,0] = (p_anchor[i,0] - p_iter_prev[0]) / np.sqrt((p_anchor[i,0] - p_iter_prev[0]) ** 2 + (p_anchor[i,1] - p_iter_prev[1]) ** 2)
        J[i,1] = (p_anchor[i,1] - p_iter_prev[1]) / np.sqrt((p_anchor[i,0] - p_iter_prev[0]) ** 2 + (p_anchor[i,1] - p_iter_prev[1]) ** 2)
    return J

def plot_gauss_contour(mu,cov,xmin,xmax,ymin,ymax,title="Title"):

    """ creates a contour plot for a bivariate gaussian distribution with specified parameters

    Input:
      mu... mean vector, 2x1
      cov...covariance matrix, 2x2
      xmin,xmax... minimum and maximum value for width of plot-area, scalar
      ymin,ymax....minimum and maximum value for height of plot-area, scalar
      title... title of the plot (optional), string"""

	#npts = 100
    delta = 0.025
    x = np.arange(xmin, xmax, delta)
    y = np.arange(ymin, ymax, delta)
    X, Y = np.meshgrid(x, y)
    Z = mlab.bivariate_normal(X,Y,np.sqrt(cov[0][0]),np.sqrt(cov[1][1]),mu[0], mu[1], cov[0][1])
    plt.plot([mu[0]],[mu[1]],'r+') # plot the mean as a single point
    CS = plt.contour(X, Y, Z)
    plt.clabel(CS, inline=1, fontsize=10)
    plt.title(title)
    plt.show()
    return

#--------------------------------------------------------------------------------
def ecdf(realizations):
    """ computes the empirical cumulative distribution function for a given set of realizations.
    The output can be plotted by plt.plot(x,Fx)

    Input:
      realizations... vector with realizations, Nx1
    Output:
      x... x-axis, Nx1
      Fx...cumulative distribution for x, Nx1"""
    x = np.sort(realizations)
    Fx = np.linspace(0,1,len(realizations))
    return Fx,x

#--------------------------------------------------------------------------------
def load_data(scenario):
    """ loads the provided data for the specified scenario
    Input:
        scenario... scalar
    Output:
        data... contains the actual measurements, nr_measurements x nr_anchors
        reference.... contains the reference measurements, nr_measurements x nr_anchors"""
    data_file = 'measurements_' + str(scenario) + '.data'
    ref_file =  'reference_' + str(scenario) + '.data'

    data = np.loadtxt(data_file,skiprows = 0)
    reference = np.loadtxt(ref_file,skiprows = 0)

    return (data,reference)
#--------------------------------------------------------------------------------
def plot_anchors_and_agent(nr_anchors, p_anchor, p_true, p_ref=None):
    """ plots all anchors and agents
    Input:
        nr_anchors...scalar
        p_anchor...positions of anchors, nr_anchors x 2
        p_true... true position of the agent, 2x1
        p_ref(optional)... position for reference_measurements, 2x1"""
    # plot anchors and true position
    plt.axis([-6, 6, -6, 6])
    for i in range(0, nr_anchors):
        plt.plot(p_anchor[i, 0], p_anchor[i, 1], 'bo')
        plt.text(p_anchor[i, 0] + 0.2, p_anchor[i, 1] + 0.2, r'$p_{a,' + str(i) + '}$')
    plt.plot(p_true[0, 0], p_true[0, 1], 'r*')
    plt.text(p_true[0, 0] + 0.2, p_true[0, 1] + 0.2, r'$p_{true}$')
    if p_ref is not None:
        plt.plot(p_ref[0, 0], p_ref[0, 1], 'r*')
        plt.text(p_ref[0, 0] + 0.2, p_ref[0, 1] + 0.2, '$p_{ref}$')
    plt.xlabel("x/m")
    plt.ylabel("y/m")
    plt.show()
    pass

def plot_estimated_positions(nr_samples, p_est, p_true, p_ref=None):
    """ plots all estimated positions"""

    plt.axis([-6, 6, -6, 6])
    for i in range(0, nr_samples):
        plt.scatter(p_est[i, 0], p_est[i, 1], s=2)
    plt.plot(p_true[0, 0], p_true[0, 1], 'r*')
    plt.text(p_true[0, 0] + 0.2, p_true[0, 1] + 0.2, r'$p_{true}$')
    if p_ref is not None:
        plt.plot(p_ref[0, 0], p_ref[0, 1], 'r*')
        plt.text(p_ref[0, 0] + 0.2, p_ref[0, 1] + 0.2, '$p_{ref}$')
    plt.xlabel("x/m")
    plt.ylabel("y/m")
    plt.show()
    pass
#--------------------------------------------------------------------------------
#--------------------------------------------------------------------------------
if __name__ == '__main__':
    main()
