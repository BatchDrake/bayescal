import numpy as np
import h5py
import matplotlib.pyplot as plt
from scipy.stats import invwishart, multivariate_normal

def interlaced2comp(intr):
    flattened = intr.flatten()
    J = len(flattened) // 2
    
    z = np.ndarray(shape = J, dtype = 'complex128')
    for j in range(J):
        z[j] = intr[2 * j + 0] + intr[2 * j + 1] * 1j
    
    new_shape = list(intr.shape)
    new_shape[-1] //= 2
    
    return z.reshape(new_shape)

def comp2interlaced(z):
    flattened = z.flatten()
    J = len(flattened)
    intr = np.ndarray(shape = 2 * J)
    for j in range(J):
        intr[2 * j + 0] = np.real(flattened[j])
        intr[2 * j + 1] = np.imag(flattened[j])
    
    new_shape = list(z.shape)
    new_shape[-1] *= 2
    
    return intr.reshape(new_shape)

class GenericCalibrator:
    def __init__(self, f, J : int, sigma2: float):
        """
        __init__ constructs a GenericCalibrator

        :param f: base evaluation function. This function takes three
        arguments, in order: the index j (starting from 0, up to J - 1) that
        identifies the specific basis function, the coordinate x of the
        evaluation point, the coordinate y of the evaluation point. The return
        type may be either a complex number or a 2-tuple.
        :param J: number of basis functions to use in this model.
        :param sigma2: noise error variance.
        """

        self.f = f
        self.J = J
        self.sigma2 = sigma2
        
        assumed_N = 2

        self.sess_alpha_niw_mu     = np.zeros([2 * J, 1])
        self.sess_alpha_niw_Psi    = np.identity(2 * J)
        self.sess_alpha_niw_nu     = 2 * self.J + assumed_N
        self.sess_alpha_niw_lambda = assumed_N

        self.calibrating = False

    def set_initial_prior(self, mu, cov, observations = 2):
        """
        set_initial_prior establishes a first, non-informative prior for
        subsequent calibrations. This effectively sets the mean mean
        and the mean covariance matrix of the Normal-Inverse-Wishart
        distribution from which prior distributions for calibration sessions
        are drawn, setting the number of observations to the minimum
        meaningful value possible.

        :param mu: mean mean of the model coefficients (Jx1 complex array or 2Jx1 real array)
        :param cov: mean covariance matrix of the model coefficients.
        :param observations: number of actual observations that back the useage of this prior.
        """

        if mu.dtype == 'complex' or mu.dtype == 'complex128':
            raise ValueError('Complex coefficients are not directly allowed.')
        
        if mu.shape != self.sess_alpha_niw_mu.shape:
            raise ValueError(
                'Invalid mean mean vector shape ' + 
                fr'(provided is {mu.shape}, but {self.sess_alpha_niw_mu.shape} ' +
                'was espected.)')

        if cov.shape != self.sess_alpha_niw_Psi.shape:
            raise ValueError(
                'Invalid mean covariance matrix shape ' + 
                fr'(provided is {cov.shape}, but {self.sess_alpha_niw_Psi.shape} ' +
                'was espected.)')

        self.sess_alpha_niw_nu     = 2 * self.J + observations
        self.sess_alpha_niw_lambda = observations
        self.sess_alpha_niw_mu     = mu.copy()
        self.sess_alpha_niw_Psi    = cov.copy() * (self.sess_alpha_niw_nu - 2 * self.J - 1)

    def sample_repeatability(self):
        # Step 1: sample from inverse Wishart
        Sigma0 = invwishart.rvs(self.sess_alpha_niw_nu, self.sess_alpha_niw_Psi)
        alpha0 = multivariate_normal.rvs(
            self.sess_alpha_niw_mu.flatten(),
            Sigma0 / self.sess_alpha_niw_lambda)

        return alpha0.reshape(self.sess_alpha_niw_mu.shape), Sigma0

    def sample_coefficients(self):
        alpha = multivariate_normal.rvs(
            self.cal_alpha_mu.flatten(),
            self.cal_alpha_Sigma)
        return alpha.reshape(self.cal_alpha_mu.shape)

    def get_observations(self):
        return self.sess_alpha_niw_lambda - 3

    def save_priors(self, path):
        """
        save_prior saves the current state of the calibrator (e.g. mean 
        means, mean covariances and so on) to a file.

        :param path: path of the file where these parameters are to be saved
        """

        hf = h5py.File(path, 'w')

        hf.create_dataset('mu',     data = self.sess_alpha_niw_mu)
        hf.create_dataset('Psi',    data = self.sess_alpha_niw_Psi)
        hf.create_dataset('nu',     data = np.array(self.sess_alpha_niw_nu))
        hf.create_dataset('lambda', data = np.array(self.sess_alpha_niw_lambda))

        hf.close()

    def load_priors(self, path):
        """
        load_prior restores the state of a previously-saved calibration. See 
        save_priors.
        """
        
        hf = h5py.File(path, 'r')

        keys = hf.keys()

        if u'mu' not in keys:
            raise TypeError('Invalid prior file (no mean)')
        
        if u'Psi' not in keys:
            raise TypeError('Invalid prior file (no covariance)')
        
        if u'nu' not in keys or u'lambda' not in keys:
            raise TypeError('Invalid prior file (no counters)')
        
        self.sess_alpha_niw_mu     = np.array(hf.get('mu'))
        self.sess_alpha_niw_Psi    = np.array(hf.get('Psi'))
        self.sess_alpha_niw_nu     = np.array(hf.get('nu')).flatten()[0]
        self.sess_alpha_niw_lambda = np.array(hf.get('lambda')).flatten()[0]

        print(self.sess_alpha_niw_nu, self.sess_alpha_niw_lambda)

        hf.close()

    def start_calibration(self):
        """
        start_calibration draws a representative mean and covariance matrix
        from the session prior sets it as the current coefficient distribution.
        """

        Sigma0 = self.sess_alpha_niw_Psi / (self.sess_alpha_niw_nu - 2 * self.J - 1)
        alpha0 = self.sess_alpha_niw_mu.copy()

        alpha, Sigma0 = self.sample_repeatability()

        self.cal_alpha_mu     = alpha
        self.cal_alpha_Lambda = np.linalg.inv(Sigma0)
        self.cal_alpha_num    = 0 # Number of provided evidence vectors
        self.calibrating = True

        self.alpha = alpha0
        self.Sigma = Sigma0
        
    def Z(self, j, x, y):
        ret = self.f(j, x, y)
        if type(ret) == complex:
            ret = (np.real(ret), np.imag(ret))
        elif type(ret) != tuple or len(ret) != 2:
            raise ValueError(
                fr'Evaluation function must return either a 2-tuple or a complex value (it was {type(ret)}).')
        
        return ret[0], ret[1]

    def collocation_matrix(self, points):
        C = points.shape[0] # Number of points

        # int stands for "interlaced": we interlace horizontal and vertical
        # components along the coefficient axis in the collocation matrix.

        Z_int = np.zeros((C, 2 * self.J))

        for j in range(self.J):
            for i in range(C):
                e_x, e_y = self.Z(j, points[i, 0], points[i, 1])
                Z_int[i, 2 * j + 0] = e_x
                Z_int[i, 2 * j + 1] = e_y
        
        Z_points = C
        Z_colno  = 2 * self.J
        Z_rowno  = 2 * Z_points

        # intint stands for "doubly interlaced", i.e. we interlace horizontal
        # and vertical componentes along both axes of the collocation matrix.

        # This is a bit trickier, as we need to translate the complex arithmetics
        # into matrix algebra with real numbers.
        
        Z_intint = np.zeros((Z_rowno, Z_colno))
        
        Z_intint[0:0 + 2 * C:2, 0::2] = +Z_int[:, 0::2]
        Z_intint[0:0 + 2 * C:2, 1::2] = -Z_int[:, 1::2]
        Z_intint[1:1 + 2 * C:2, 0::2] = +Z_int[:, 1::2]
        Z_intint[1:1 + 2 * C:2, 1::2] = +Z_int[:, 0::2]

        return Z_intint

    def estimate_error(self, points, sigma2 = None, per_axis = False):
        """
        estimate_error produces informed estimations of the error of the model
        a given set of points.

        :param points: a Nx2 array of points in which to estimate the error.
        :param sigma2: optional, override the noise variance used for this
        calculation.
        :type sigma2: float, array of N elements
        :type per_axis: bool, the error is provided in a per-axis fashion
        
        :returns: a Nx1 variance vector of the euclidean norm of the error, or
        a 2Nx1 variance vector of the error in each component.
        """
        if not self.calibrating:
            raise ValueError('Please call start_calibration() first.')
        
        if sigma2 is None:
            sigma2 = self.sigma2
        elif type(sigma2) == np.ndarray:
            if sigma2.shape != (points.shape[0]):
                raise ValueError('Invalid shape for the error variance vector')
        elif type(sigma2) != float:
            raise ValueError('Unrecognized type for the error variance')
        
        
        Z_intint = self.collocation_matrix(points)

        #nu        = self.sess_alpha_niw_nu
        Sigma     = self.Sigma
        errcov    = np.matmul(np.matmul(Z_intint, Sigma), Z_intint.transpose())
        
        # errcov is the covariance matrix of the predicted values, therefore
        # its diagonal is just the intrinsic variance of each predicted value
        diag  = np.diag(errcov)

        if per_axis:
            error = diag + sigma2
        else:
            error = diag[0::2] + diag[1::2] + 2 * sigma2

        return error

    def estimate_mse(self, points, sigma2 = None, per_axis = False):
        """
        estimate_mse produces informed estimations of the mean squared error
        (MSE). This calculation comes from an analysis of the MSE as a
        random variable of the measurement process and the uncertainty of
        the parameters. In this model, the MSE behaves as a generalized
        Chi-squared distribution whose weights are the eigenvalues of
        the error covariance matrix (ZΣZ^T) plus the measurement error variance
        multiplied by the identity.

        :param points: a Nx2 array of points in which to estimate the error.
        :param sigma2: optional, override the noise variance used for this
        calculation.
        :type sigma2: float, array of N elements
        :type per_axis: bool, the error is provided in a per-axis fashion. The
        per-axis MSE is 2 times smaller than the regular MSE.
        
        :returns: a 2-tuple with the mean MSE and the variance of the MSE.
        """
        if not self.calibrating:
            raise ValueError('Please call start_calibration() first.')
        
        if sigma2 is None:
            sigma2 = self.sigma2
        elif type(sigma2) == np.ndarray:
            if sigma2.shape != (points.shape[0]):
                raise ValueError('Invalid shape for the error variance vector')
        elif type(sigma2) != float:
            raise ValueError('Unrecognized type for the error variance')
        
        
        Z_intint = self.collocation_matrix(points)

        Sigma     = self.Sigma
        errcov    = np.matmul(np.matmul(Z_intint, Sigma), Z_intint.transpose())
        errcov   += sigma2 * np.identity(Z_intint.shape[0])

        lambd,_   = np.linalg.eig(errcov)

        if per_axis:
            K = errcov.shape[0]
        else:
            K = errcov.shape[0] / 2
        
        mean_MSE  = np.sum(np.array(lambd)) / K
        sig2_MSE  = 2 * np.sum(lambd ** 2) / K ** 2

        return mean_MSE, sig2_MSE

    def estimate_rms(self, points, sigma2 = None, per_axis = False):
        """
        estimate_rms produces informed estimations of the root-mean-square erorr
        (RMS). This calculation performs a linearization of the square root
        of the MSE around its mean, and propagates the variance. It also assumes
        that the MSE behaves almost like a Gaussian variable, which is not
        necessarily true (it actually is a generalized Chi-squared). May not be
        a good approximation for high incertitude in the parameters.

        :param points: a Nx2 array of points in which to estimate the error.
        :param sigma2: optional, override the noise variance used for this
        calculation.
        :type sigma2: float, array of N elements
        :type per_axis: bool, the error is provided in a per-axis fashion. The
        per-axis RMS is sqrt(2) times smaller than the regular RMS.
        
        :returns: a 2-tuple with the mean RMS and the variance of the RMS.
        """

        mean_MSE, sig2_MSE = self.estimate_mse(points, sigma2, per_axis)

        mean_RMS = np.sqrt(mean_MSE)
        sig2_RMS = sig2_MSE / (4 * mean_MSE)

        return mean_RMS, sig2_RMS
        
    def feed(self, points, epsilon, sigma2 = None, scale_factor = 1e-6):
        """
        feed provides new observations of the 2D error of the quantity to
        calibrate.

        :param points: a Nx2 array of calibration point locations.
        :param epsilon: a a Nx2 array of error measurements, in 2D.
        :param sigma2: a custom noise variance for these measurements.
        """

        if not self.calibrating:
            raise ValueError('Please call start_calibration() first.')

        K     = 1 / scale_factor
        K2    = K ** 2
        Kinv  =  scale_factor
        Kinv2 = Kinv ** 2

        if sigma2 is None:
            sigma2 = self.sigma2
        elif type(sigma2) == np.ndarray:
            if sigma2.shape != (points.shape[0]):
                raise ValueError('Invalid shape for the error variance vector')
        elif type(sigma2) != float:
            raise ValueError('Unrecognized type for the error variance')
        
        C = points.shape[0] # Number of points

        Z_intint = self.collocation_matrix(points)
        
        # Scale parameters prior to perform any calculation
        eps_int  = K * epsilon.ravel().reshape(2 * C, 1) # Interlace

        Lambda_0 = self.cal_alpha_Lambda.copy() / K2
        alpha_0  = self.cal_alpha_mu.copy() * K
        ron2_0   = sigma2 * K2

        # Calculate posterior parameters
        Lambda_1 = np.matmul(Z_intint.transpose(), Z_intint) / ron2_0 + Lambda_0
        Sigma_1  = np.linalg.inv(Lambda_1)
        alpha_1  = np.matmul(
            Sigma_1, 
            np.matmul(Z_intint.transpose(), eps_int) / ron2_0 + np.matmul(Lambda_0, alpha_0))

        # Rescale
        self.cal_alpha_mu     = alpha_1 * Kinv
        self.cal_alpha_Lambda = Lambda_1 / Kinv2
        self.cal_alpha_Sigma  = Sigma_1  * Kinv2
        self.cal_alpha_num   += epsilon.shape[0]
        
        # Draw the best representative of the model coefficients
        self.alpha = self.cal_alpha_mu.copy()
        self.Sigma = self.cal_alpha_Sigma.copy()

    def accept_calibration(self):
        """
        accept_calibration takes a representative vector of model coefficients
        according to all the observations provided so far, and uses it to
        train the prior distribution used for future calibrations.
        """
        
        if not self.calibrating:
            raise ValueError('Please call start_calibration() first.')

        for i in range(1):
            #
            # Update rule for NIW posteriors.
            #
            mu0       = self.sess_alpha_niw_mu.copy()
            lambda0   = self.sess_alpha_niw_lambda
            nu0       = self.sess_alpha_niw_nu
            psi0      = self.sess_alpha_niw_Psi.copy()

            # Get the current coefficients
            # ybar      = self.alpha.copy()
            ybar      = self.sample_coefficients()

            # Bayesian update
            mun       = (lambda0 * mu0 + ybar) / (lambda0 + 1)
            lambdan   = lambda0 + 1
            nun       = nu0 + 1
            delta_psi = np.matmul(ybar - mu0, (ybar - mu0).transpose())
            psin      = psi0 + lambda0 / lambdan * delta_psi
            
            # Replace prior by the posterior
            self.sess_alpha_niw_mu     = mun
            self.sess_alpha_niw_lambda = lambdan
            self.sess_alpha_niw_nu     = nun
            self.sess_alpha_niw_Psi    = psin

        self.calibrating = False

    def discard_calibration(self):
        """
        discard_calibration ignores the measurements provided in the current
        calibration and reverts it back to a sane state.
        """
        
        self.calibrating = False

    def get_coefficients(self, type = 'complex'):
        """
        get_coefficients returns a representative vector of model coefficients
        according to the current state of the cofficient distribution.
        """
        
        if self.calibrating:
            if type == 'complex':
                return interlaced2comp(self.alpha.flatten())
            return self.alpha.copy()
        
        if type == 'complex':
            return interlaced2comp(self.sess_alpha_niw_mu.flatten())

        return self.sess_alpha_niw_mu.copy()