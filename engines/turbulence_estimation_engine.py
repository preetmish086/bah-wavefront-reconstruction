"""
===========================================================
Turbulence Estimation Engine
===========================================================

Input:
    Stream of Zernike coefficient dictionaries

Output:
    r0      (Fried Parameter)
    tau0    (Atmospheric Coherence Time)
    Strength Classification
"""

import numpy as np


class TurbulenceEstimationEngine:

    def __init__(self):

        self.history = []

    ############################################################
    # Add new Zernike frame
    ############################################################

    def add_frame(self, zernike_coefficients):

        self.history.append(zernike_coefficients)

    ############################################################
    # Temporal Statistical Analysis
    ############################################################

    def temporal_statistics(self):

        if len(self.history) < 2:
            raise ValueError("Need multiple frames.")

        keys = list(self.history[0].keys())

        stats = {}

        for key in keys:

            values = np.array([frame[key] for frame in self.history])

            stats[key] = {
                "mean": np.mean(values),
                "variance": np.var(values),
                "std": np.std(values)
            }

        return stats

    ############################################################
    # Spatial Analysis
    ############################################################

    def spatial_strength(self):

        latest = self.history[-1]

        coeff = np.array(list(latest.values()))

        rms = np.sqrt(np.mean(coeff ** 2))

        return rms

    ############################################################
    # Estimate Fried Parameter
    ############################################################

    def estimate_r0(self):

        stats = self.temporal_statistics()

        total_variance = sum(
            stats[k]["variance"] for k in stats
        )

        # Demo inverse relationship
        r0 = 0.15 / (1 + 15 * total_variance)

        return r0

    ############################################################
    # Estimate Tau0
    ############################################################

    def estimate_tau0(self):

        coeff_matrix = []

        keys = list(self.history[0].keys())

        for frame in self.history:

            coeff_matrix.append(
                [frame[k] for k in keys]
            )

        coeff_matrix = np.array(coeff_matrix)

        velocity = np.mean(
            np.abs(np.diff(coeff_matrix, axis=0))
        )

        tau0 = 0.020 / (1 + 25 * velocity)

        return tau0

    ############################################################
    # Characterize Turbulence
    ############################################################

    def characterize(self, r0):

        if r0 > 0.12:
            return "Weak"

        elif r0 > 0.07:
            return "Moderate"

        else:
            return "Strong"

    ############################################################
    # Complete Pipeline
    ############################################################

    def process(self):

        r0 = self.estimate_r0()

        tau0 = self.estimate_tau0()

        strength = self.characterize(r0)

        return {

            "r0": round(r0 * 100, 2),      # cm

            "tau0": round(tau0 * 1000, 2), # ms

            "strength": strength
        }


# ###############################################################
# # Demo
# ###############################################################

# if __name__ == "__main__":

#     engine = TurbulenceEstimationEngine()

#     ###########################################################
#     # Simulated stream of Zernike coefficients
#     ###########################################################

#     zernike_stream = [

#         {"a1":0.00,"a2":0.12,"a3":0.08,"a4":0.05,"a5":0.01,"a6":-0.02},

#         {"a1":0.00,"a2":0.13,"a3":0.09,"a4":0.04,"a5":0.02,"a6":-0.01},

#         {"a1":0.01,"a2":0.15,"a3":0.11,"a4":0.05,"a5":0.02,"a6":-0.03},

#         {"a1":0.00,"a2":0.16,"a3":0.10,"a4":0.06,"a5":0.03,"a6":-0.02},

#         {"a1":0.01,"a2":0.14,"a3":0.12,"a4":0.05,"a5":0.02,"a6":-0.01},

#         {"a1":0.00,"a2":0.17,"a3":0.11,"a4":0.07,"a5":0.03,"a6":-0.03},

#         {"a1":0.01,"a2":0.18,"a3":0.12,"a4":0.06,"a5":0.04,"a6":-0.02},

#         {"a1":0.00,"a2":0.19,"a3":0.13,"a4":0.07,"a5":0.04,"a6":-0.03}

#     ]

#     for frame in zernike_stream:

#         engine.add_frame(frame)

#     report = engine.process()

#     print("\n========== Turbulence Report ==========\n")

#     print("Fried Parameter (r0) :", report["r0"], "cm")
#     print("Coherence Time (tau0):", report["tau0"], "ms")
#     print("Strength             :", report["strength"])