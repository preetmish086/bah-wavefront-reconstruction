"""
=========================================================
Wavefront Reconstruction Engine
=========================================================

Input:
    Block-wise Shift Parameters
        [(dx1,dy1), (dx2,dy2), ...]

↓

1. Local Slope Calculation
2. Slope Normalization
3. Southwell/Fried Wavefront Reconstruction
4. Zernike Polynomial Fitting

↓

Output
    • Wavefront W(x,y)
    • Zernike Coefficients
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import lsqr


class WavefrontReconstructionEngine:

    ##########################################################
    # Constructor
    ##########################################################

    def __init__(self, grid_size=8):

        self.grid_size = grid_size

    ##########################################################
    # STEP 1
    # Local slope calculation
    ##########################################################

    def calculate_local_slopes(self, shift_data):

        shift_data = np.asarray(shift_data, dtype=float)

        sx = shift_data[:, 0]
        sy = shift_data[:, 1]

        return sx, sy

    ##########################################################
    # STEP 2
    # Normalize slopes
    ##########################################################

    def normalize_slopes(self, sx, sy):

        sx = sx / np.max(np.abs(sx))
        sy = sy / np.max(np.abs(sy))

        return sx, sy

    ##########################################################
    # STEP 3
    # Southwell/Fried reconstruction
    ##########################################################

    def reconstruct_wavefront(self, sx, sy):

        N = self.grid_size

        unknowns = N * N

        equations = 2 * N * (N - 1)

        A = lil_matrix((equations, unknowns))

        b = np.zeros(equations)

        eq = 0

        # Horizontal equations

        for i in range(N):
            for j in range(N - 1):

                p1 = i * N + j
                p2 = i * N + j + 1

                A[eq, p2] = 1
                A[eq, p1] = -1

                b[eq] = sx[p1]

                eq += 1

        # Vertical equations

        for i in range(N - 1):
            for j in range(N):

                p1 = i * N + j
                p2 = (i + 1) * N + j

                A[eq, p2] = 1
                A[eq, p1] = -1

                b[eq] = sy[p1]

                eq += 1

        wavefront = lsqr(A, b)[0]

        return wavefront.reshape((N, N))

    ##########################################################
    # STEP 4
    # Zernike Basis
    ##########################################################

    def zernike_basis(self, x, y):

        r = np.sqrt(x ** 2 + y ** 2)

        theta = np.arctan2(y, x)

        basis = []

        # a1
        basis.append(np.ones_like(r))

        # a2
        basis.append(2 * r * np.cos(theta))

        # a3
        basis.append(2 * r * np.sin(theta))

        # a4
        basis.append(np.sqrt(3) * (2 * r ** 2 - 1))

        # a5
        basis.append(np.sqrt(6) * r ** 2 * np.sin(2 * theta))

        # a6
        basis.append(np.sqrt(6) * r ** 2 * np.cos(2 * theta))

        return np.vstack(basis).T

    ##########################################################
    # STEP 5
    # Zernike fitting
    ##########################################################

    def fit_zernike(self, wavefront):

        N = self.grid_size

        x = np.linspace(-1, 1, N)
        y = np.linspace(-1, 1, N)

        X, Y = np.meshgrid(x, y)

        mask = X ** 2 + Y ** 2 <= 1

        Z = wavefront[mask]

        basis = self.zernike_basis(X[mask], Y[mask])

        coeff, _, _, _ = np.linalg.lstsq(
            basis,
            Z,
            rcond=None
        )

        names = [
            "a1",
            "a2",
            "a3",
            "a4",
            "a5",
            "a6"
        ]

        return dict(zip(names, np.round(coeff, 5)))

    ##########################################################
    # COMPLETE PIPELINE
    ##########################################################

    def process(self, shift_data):

        print("STEP 1 : Calculating Local Slopes")

        sx, sy = self.calculate_local_slopes(shift_data)

        print("STEP 2 : Normalizing Slopes")

        sx, sy = self.normalize_slopes(sx, sy)

        print("STEP 3 : Southwell/Fried Reconstruction")

        wavefront = self.reconstruct_wavefront(sx, sy)

        print("STEP 4 : Zernike Fitting")

        coeff = self.fit_zernike(wavefront)

        return wavefront, coeff


##############################################################
# Demo
##############################################################

if __name__ == "__main__":

    np.random.seed(42)

    # Example block shift parameters
    shift_data = []

    for _ in range(64):

        dx = np.random.normal(0, 0.3)
        dy = np.random.normal(0, 0.3)

        shift_data.append((dx, dy))

    engine = WavefrontReconstructionEngine(grid_size=8)

    wavefront, coeff = engine.process(shift_data)

    print("\n========== Zernike Coefficients ==========\n")

    for k, v in coeff.items():

        print(f"{k} : {v}")

    plt.figure(figsize=(6, 5))

    plt.imshow(wavefront, cmap="jet")

    plt.title("Reconstructed Wavefront W(x,y)")

    plt.colorbar(label="Wavefront")

    plt.tight_layout()

    plt.show()