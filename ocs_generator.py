#
# Copyright (c) 2021 Gonzalo J. Carracedo <BatchDrake@gmail.com>
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

HARMONI_OCS_STRATEGY_NUMBER = 11

class OCSGenerator():
    def __init__(self, N, radial_redundancy = 0, azimuthal_redundancy = 0):
        self.N = N
        self.alpha = 0
        self.rr = radial_redundancy
        self.rt = azimuthal_redundancy
        self.extend = False
        
        if self.N < 1:
            raise ValueError("Too few calibration points")
        
        # This comes from inverting n = j * (j + 1) / 2
        self.max_deg = int(np.ceil(-1.5 + .5 * np.sqrt(1 + 8 * self.N)))

    @staticmethod
    def optimize_distance(p):
        as_set = set(range(0, len(p)))
        calpoints = []

        p_last = np.array([0, 0])

        while len(as_set) > 0:
            n = len(as_set)
            max_dist = 0
            i = 0
            avail  = list(as_set)
            best_i = avail[i]

            for i in range(n):
                dx = p_last[0] - p[avail[i]][0]
                dy = p_last[1] - p[avail[i]][1]
                dist  = dx ** 2 + dy ** 2

                if dist > max_dist:
                    max_dist = dist
                    best_i   = avail[i]

            as_set.remove(best_i)
            p_last = p[best_i]
            calpoints.append(p_last)

        return calpoints

    def generate_points(self, scale = 1., rho_index = None, az_index = None):
        n = self.max_deg
        k = int(np.floor(n / 2) + 1)
        total = self.N * (self.rr + 1) * (self.rt + 1)
        P = []
        center_added = False
        
        # If either index is none, it means that we want the whole redundant pattern
        # Therefore, the initial index must be necessarily 0

        rho_start = rho_index if rho_index is not None else 0
        az_start  = az_index  if az_index  is not None else 0

        if rho_start < 0 or rho_start >= self.rr + 1:
            raise ValueError(
                f'Invalid radial subpattern index (provided {rho_start}, must be between 0 and {self.rr}')
        
        if az_start < 0 or az_start >= self.rt + 1:
            raise ValueError(
                f'Invalid azimuthal subpattern index (provided {az_start}, must be between 0 and {self.rt}')
        
        # On the other hand, if either index IS none, it means that we want the
        # specific subpattern in that direction. The redundancy is interpreted as
        # the number of extra points to add to the existing OCS pattern

        rho_step = self.rr + 1 if rho_index is not None else 1
        az_step  = self.rt + 1 if az_index  is not None else 1

        offsetting = 2 * k != n + 1

        mn = n * (self.rr + 1)
        for mj in range(rho_start, k * (self.rr + 1), rho_step):
            r_patno = mj % (self.rr + 1)
            j = mj / (self.rr + 1)
            jf = mj // (self.rr + 1)
            off = j - jf
            j = jf - off


            if self.extend:
                ang = (2 * mj + 1) / (2 * (mn + 1))

            else:
                if offsetting:
                    ang = (j + jf + 1) / (2 * (n + 1))
                else:
                    ang = (2 * j + 1) / (2 * (n + 1))
            
            zeta = np.cos(ang * np.pi)
            r_j  = 1.1565 * zeta - 0.76535 * zeta ** 2 + 0.60517 * zeta ** 3
            n_j  = 2 * n - 4 * jf + 1
                   
            r_j *= scale
            
            # We also attempt to remove the alignment of the redundant patterns
            # in the radial direction, to prevent excessive alignment. The idea is that,
            # if the radial redundancy is RR, the pattern with rho_index = RR + 1 -while does
            # not exist- should have performed a complete an azimuthal step between two
            # conscutive redundant patterns. This is achieved by adding this extra phase
            # term.
            az_step_angle = 2 * np.pi / n_j
            phase = ((jf + 1) / k) * r_patno / (self.rr + 1) * az_step_angle

            for mi in range(az_start, n_j * (self.rt + 1), az_step):
                # Other issue we need to solve is the fact that the central point is a fixed
                # point with respect to rotation. In order to prevent the addition to the same
                # central point more than once, we detect this situation and skip posterior additions

                if np.abs(r_j) < 1e-5:
                    if center_added:
                        break
                    center_added = True
                
                i = mi / (self.rt + 1)
                theta = self.alpha + az_step_angle * i + phase
                P.append([r_j * np.cos(theta), r_j * np.sin(theta)])
            
        calpoints = np.array(self.optimize_distance(P))
        
        if total < calpoints.shape[0]:
            indices = list(np.random.sample(range(calpoints.shape[0]), total))
            indices.sort()
            calpoints = calpoints[indices, :]
            
        return np.array(calpoints)
