#
# Copyright (c) 2023 Gonzalo J. Carracedo <BatchDrake@gmail.com>
#
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its contributors
#    may be used to endorse or promote products derived from this software
#    without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#

import numpy as np
import matplotlib.pyplot as plt
import time
import sys

from abstract_calibrator import GenericCalibrator, interlaced2comp, comp2interlaced
from complex_zernike import ComplexZernike
from ocs_generator import OCSGenerator

J     = 10       # Number of (complex) coefficients of the model
R     = 0.2      # Radius of the evaluation circle [m]
TESTS = 10000    # Number of calibrations to perform in every run
GOAL  = 6e-6     # A reasonable calibration goal [m]
NM    = 1        # Number of repeated measurements
TRAIN = 25       # Number of tests in the training phase
FORCE_UNINFORMATIVE = False # Force uninformative prior (block usage of prior information)
NOBAYES = False  # Exit before performing Bayesian tests

#
# This is a naif instrument model: it assumes certain FPRS distortion, that
# (for some reason) the NGSS center gets off axis from day to day and that
# the IRW angle bias changes from day to day for another reason. 
#
# Additionally, we provide a read error of the spot center that is purely random,
# and it is defined by its standard deviation in each separate axis.
#

SIGMA = .2e-6    # Standard deviation of the read error IN EACH AXIS [m]
IRW_ERR = 30     # Standard deviation of the derotator bias [arcsec]
POS_ERR_X = 1e-3 # Standard deviation of the NGSS center repeatability, X axis [m]
POS_ERR_Y = 2e-3 # Standard deviation of the NGSS center repeatability, Y axis [m]

# Basis functions
Z = [] 
for j in range(J):
    m, n = ComplexZernike.j_to_mn(j)
    Z.append(ComplexZernike.Z(m, n))

def tovector(x, y):
    return np.array([x, y]).transpose()

def tocomplex(x, y):
    return np.array([x + 1j * y])

#
# Let's start by simulating something simple: a fixed, high-order distortion
# plus a varying first order distortion. We provide a function that generates
# a new distortion (as an evaluation function) and a measurement simulator, that
# adds the measurement noise.
#
# The result of both functios is a complex number. The real part is the
# horizontal coordinate, and the imaginary part the vertical.
#

fprs_a = np.zeros(J, 'complex128')
fprs_a[0] = + 1.32736994e-05 + 1.70706859e-05j
fprs_a[1] = + 8.47032141e-06 - 4.15957764e-05j
fprs_a[2] = - 9.23376138e-05 + 5.37598995e-05j
fprs_a[3] = - 1.32721850e-06 + 5.68264772e-07j
fprs_a[4] = + 1.31890698e-05 + 1.69757650e-05j
fprs_a[5] = + 1.30213240e-05 - 1.67406899e-05j
fprs_a[6] = - 1.14561288e-07 - 3.66124349e-08j
fprs_a[7] = + 4.14842859e-08 - 1.58805615e-07j
fprs_a[8] = - 3.17849617e-06 - 1.75738707e-08j
fprs_a[9] = + 1.54276951e-07 + 5.94889095e-07j

def generate_distortion():
    # This is the fixed part
    a = fprs_a.copy()

    optical_error = ComplexZernike(a)

    # This is certain rotational error, in degrees (e.g., 30 arcseconds)
    angle_error = IRW_ERR / (180 * 3600) * np.pi * np.random.randn()

    # This is a displacement error, in meters
    pos_error = POS_ERR_X * np.random.randn() + POS_ERR_Y * np.random.randn() * 1j

    # The resulting lambda function does the following:
    #  1. Transforms the x, y vectors into a complex vector
    #  2. Adds the optical distortion (as a complex value)
    #  3. Adds certain positioning offset
    #  4. Rotates everything by a random angle
    
    return lambda x, y:                           \
          ((tocomplex(x, y)                       \
        + optical_error(tovector(x, y) / R))      \
        + pos_error)                              \
        * np.exp(1j * angle_error)  \

def measure(distortion, x, y):
    n        = x.shape[0]
    x        = x.reshape(n)
    y        = y.reshape(n)
    ron      = SIGMA * (np.random.randn(n) + 1j * np.random.randn(n))
    eps_real = distortion(x, y) - tocomplex(x, y)
    return (eps_real + ron).reshape(n)

def test_model_classic(coefs, distortion, points):
    model  = ComplexZernike(coefs)
    estimated = model(points / R)
    actual = measure(distortion, points[:, 0], points[:, 1]).flatten()
    diff   = comp2interlaced(estimated - actual) # Error vector
    err    = np.abs(estimated - actual)          # Error distances
    rms    = np.sqrt(.5 * np.mean(err ** 2))     # RMS of all distances

    return rms, err, diff

def test_model(calibrator, distortion, points):
    coefs  = calibrator.get_coefficients()
    rms, err, diff = test_model_classic(coefs, distortion, points)
    sig2   = calibrator.estimate_error(points, per_axis = True)

    return rms, np.mean(err), np.max(err), diff, sig2

#################### TIME TO CALIBRATE ##################

# Our basis function is just the Zernike j evaluated in (x, y), normlized by
# the radius R

def basis_function(j, x, y):
    value = Z[j](tovector([x], [y]) / R)
    return np.real(value), np.imag(value)

calibrator = GenericCalibrator(basis_function, J, SIGMA ** 2)

try:
    calibrator.load_priors('priors.hf5')
    first_run = False
except:
    print('Failed to load prior file. Starting with a very uninformative prior')
    # Let us define some terrible prior: 0.2 mm variance over the whole field
    calibrator.set_initial_prior(np.zeros((2 * J, 1)), np.identity(2 * J) * .1e-3 ** 2)
    first_run = True

needed = []

ocs = OCSGenerator(J, 0, 0)
ocs_points_basic = ocs.generate_points(R)

ocs = OCSGenerator(J, 2, 2)
ocs_points_redundant = ocs.generate_points(R)

ocs_points = np.vstack((ocs_points_basic, ocs_points_redundant))

np.save('ocs_points_basic', ocs_points_basic)
np.save('ocs_points_redundant', ocs_points_redundant)
uncertainty = []
rms = []
est = []
var = []

k0 = calibrator.get_observations()

coefs = []

np.random.seed(seed=int(1e6 * time.time()) % 1000000000) 

Lcal = []

if first_run and not FORCE_UNINFORMATIVE:
    print('NOTE: This is a first run. I am leveraging these tests to generate a more informative prior.')
    alphas = []
    clasrms = []

    for k in range(TRAIN):
        distortion = generate_distortion()

        Z_r = calibrator.collocation_matrix(ocs_points_basic)

        err     = measure(distortion, ocs_points_basic[:, 0], ocs_points_basic[:, 1])
        epsilon = tovector(np.real(err), np.imag(err)).flatten()
        
        result = np.linalg.lstsq(Z_r, epsilon, rcond = None)
        alpha  = result[0]
        alphas.append(alpha)

        rms_e, err, dist = test_model_classic(interlaced2comp(alpha), distortion, ocs_points)
        print(fr'  Classical calibration #{k + 1}, RMS = {rms_e * 1e6:.2f} µm')
        clasrms.append(rms_e)

    alphas     = np.array(alphas)
    alpha_mean = np.mean(alphas, axis = 0)
    alpha_cov  = np.cov(alphas.transpose())

    print(fr'  Done. RMS file saved to classic_rms.npy')
    np.save('classic_rms', clasrms)

    if NOBAYES:
        sys.exit(0)
    
    calibrator.set_initial_prior(alpha_mean.reshape(2 * J, 1), alpha_cov, observations = TRAIN)

for k in range(k0, k0 + TESTS):
    # New night, new distortion.
    distortion = generate_distortion()

    if first_run and FORCE_UNINFORMATIVE:
        calibrator.set_initial_prior(
            np.zeros((2 * J, 1)),
            np.identity(2 * J) * .2e-3 ** 2)

    
    print(fr'Good evening! Beautiful night to perform observation number #{k + 1}')
    calibrator.start_calibration()

    goal_reached = False
    count = 0
    err_std = 0

    points = np.zeros((NM, 2))

    for i in range(ocs_points.shape[0]):
        rho   = np.sqrt(np.random.uniform(0, R ** 2))
        theta = np.random.uniform(0, 2 * np.pi)

        x = np.array([rho * np.cos(theta)])
        y = np.array([rho * np.sin(theta)])

        point = ocs_points[i:i+1, :]
        for j in range(NM):
            points[j, :] = point[0, :]
        
        # Feed the calibrator with this new piece of information. We could provide
        # a different sigma2 here, if we knew the read-out noise for this particular
        # set of measurements.

        pred_mean_rms, pred_var_rms = calibrator.estimate_rms(point, per_axis = True)
        # unc_var2   = calibrator.estimate_error(point)
        # full_var2  = np.mean(unc_var2)
        # err_std    = np.sqrt(full_var2)
        # err_std_um = 1e6 * err_std
        # if i == 0:
        #   uncertainty.append(err_std)

        pred_mean_rms_um  = 1e6 * pred_mean_rms
        pred_std_rms_um   = 1e6 * np.sqrt(pred_var_rms)
        
        if i == 0:
          uncertainty.append(pred_mean_rms)

        print(fr'  With {i:3d} calibration points: Est. RMS {pred_mean_rms_um:7.2f} ± {pred_std_rms_um:7.2f} µm (RON: {SIGMA * 1e6:.2f} µm)')
        if pred_mean_rms <= GOAL:
            goal_reached = True
            break

        x = points[:, 0:1]
        y = points[:, 1:2]

        err = measure(distortion, x, y)

        # Generate error vector in the form of Nx2. Since we are providing only
        # one point, this vector must be 1x2

        epsilon = tovector(np.real(err), np.imag(err))
        calibrator.feed(points, epsilon)

        count += 1
        
    rms_e, mean_e, max_e, dist, sig2 = test_model(calibrator, distortion, ocs_points)
    rms.append(rms_e)
    est.append(pred_mean_rms)
    var.append(pred_var_rms)

    Lcal.append([sig2, dist])

    if goal_reached:
        coefs.append(calibrator.get_coefficients('float').ravel())
        print(fr'{k + 1:4}: GOAL REACHED AFTER {count} POINTS. Estimated RMS: {pred_mean_rms_um:7.2f} ± {pred_std_rms_um:7.2f} µm, observed: {rms_e * 1e6:.2f} µm')
        if FORCE_UNINFORMATIVE:
            calibrator.discard_calibration()
        else:
            calibrator.accept_calibration()
    else:
        print(fr'{k + 1:4}: GOAL NOT REACHED. Instrument intervention?')
        calibrator.discard_calibration()

    print()

    needed.append(count)
print("Saving session priors...")

np.savez('distances.bin', np.array(Lcal))

calibrator.save_priors('priors.hf5')

nn = range(k0, k0 + TESTS)
plt.figure()
plt.scatter(nn, needed, 10)
plt.title('Number of measurements needed to achieve goal')
plt.xlabel('Measurement number')
plt.ylabel('Measurements')
plt.grid()
plt.tight_layout()

Sigma = calibrator.sess_alpha_niw_Psi / (calibrator.sess_alpha_niw_nu - 2 * calibrator.J - 1)

plt.figure()
plt.imshow(np.log10(1e12 * np.abs(Sigma)), cmap = 'inferno')
plt.title(r'$log_{10}|\Sigma|$ [µm²]')
plt.colorbar()
plt.tight_layout()

plt.figure()
plt.semilogy(nn, np.array(uncertainty) * 1e6)
plt.title('Standard deviation of prior model uncertainty')
plt.xlabel('Measurement number')
plt.ylabel(r'$\sigma_\varepsilon$ [µm]')
plt.tight_layout()

plt.figure()
plt.semilogy(nn, np.array(est) * 1e6, label = r'$\sigma_\varepsilon$ (model estimate)')
plt.semilogy(nn, np.array(rms) * 1e6, label = r'RMS ($\varepsilon$)')
plt.title('Standard deviation of posterior model uncertainty')
plt.xlabel('Measurement number')
plt.ylabel(r'$\sigma_\varepsilon$ [µm]')
plt.legend()
plt.tight_layout()

np.savez('caldata.bin', nn, needed, uncertainty, est, var, rms, Sigma)

plt.show()
