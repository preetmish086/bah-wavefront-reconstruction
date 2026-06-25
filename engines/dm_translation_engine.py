"""
===========================================================
DM Translation Engine
===========================================================

Input:
    1. Wavefront W(x,y)
    2. Zernike Coefficients

Output:
    Actuator Commands

Pipeline

Wavefront
    │
    ▼
Conjugate Wavefront

    │
    ▼
Influence Function Model

    │
    ▼
Actuator Fitting

    │
    ▼
Coupling Compensation

    │
    ▼
DM Commands
"""

import numpy as np


class DMTranslationEngine:

    def __init__(self, actuator_grid=8):

        self.N = actuator_grid

    ##########################################################
    # STEP 1
    # Build conjugate wavefront
    ##########################################################

    def build_conjugate_wavefront(self, wavefront):

        # DM must generate opposite phase

        return -wavefront

    ##########################################################
    # STEP 2
    # Influence Function Model
    ##########################################################

    def influence_function(self, conjugate_wavefront):

        kernel = np.array([
            [0.05, 0.10, 0.05],
            [0.10, 0.40, 0.10],
            [0.05, 0.10, 0.05]
        ])

        padded = np.pad(conjugate_wavefront, 1)

        output = np.zeros_like(conjugate_wavefront)

        for i in range(self.N):
            for j in range(self.N):

                region = padded[i:i+3, j:j+3]

                output[i, j] = np.sum(region * kernel)

        return output

    ##########################################################
    # STEP 3
    # Actuator fitting
    ##########################################################

    def actuator_fit(self, influence_map):

        commands = {}

        actuator = 1

        for i in range(self.N):
            for j in range(self.N):

                commands[f"actuator_{actuator}"] = float(
                    np.round(influence_map[i, j], 4)
                )

                actuator += 1

        return commands

    ##########################################################
    # STEP 4
    # Coupling Compensation
    ##########################################################

    def coupling_compensation(self, commands):

        keys = list(commands.keys())

        values = np.array(list(commands.values()))

        compensated = values.copy()

        for i in range(1, len(values)-1):

            compensated[i] = (
                0.8 * values[i]
                + 0.1 * values[i-1]
                + 0.1 * values[i+1]
            )

        final = {}

        for k, v in zip(keys, compensated):

            final[k] = float(np.round(v, 4))

        return final

    ##########################################################
    # COMPLETE PIPELINE
    ##########################################################

    def process(self, wavefront, zernike):

        print("STEP 1 : Building Conjugate Wavefront")

        conjugate = self.build_conjugate_wavefront(
            wavefront
        )

        print("STEP 2 : Applying Influence Function")

        influence = self.influence_function(
            conjugate
        )

        print("STEP 3 : Actuator Fitting")

        commands = self.actuator_fit(
            influence
        )

        print("STEP 4 : Coupling Compensation")

        commands = self.coupling_compensation(
            commands
        )

        return commands