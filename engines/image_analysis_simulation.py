"""
===========================================================
Image Analysis Simulation
===========================================================

Temporary replacement for the Image Analysis Engine.

Output:
    Block-wise shift parameters

Each block returns:
    (Δx, Δy)

Later this file can be replaced with the actual image
analysis engine without changing the rest of the pipeline.
"""

import numpy as np


class ImageAnalysisSimulation:

    def __init__(self,
                 grid_size=8,
                 noise_level=0.02,
                 random_seed=None):

        self.grid_size = grid_size
        self.noise_level = noise_level

        if random_seed is not None:
            np.random.seed(random_seed)

    ##########################################################
    # Generate one frame
    ##########################################################

    def generate_shift_frame(self):

        shift_data = []

        N = self.grid_size

        for y in range(N):
            for x in range(N):

                # Smooth wavefront-like motion
                dx = (
                    0.20 * np.sin(x / 2.5)
                    + 0.08 * np.cos(y / 3.0)
                )

                dy = (
                    0.20 * np.cos(y / 2.5)
                    + 0.08 * np.sin(x / 3.0)
                )

                # Random turbulence
                dx += np.random.normal(0, self.noise_level)
                dy += np.random.normal(0, self.noise_level)

                shift_data.append((round(dx, 4),
                                   round(dy, 4)))

        return shift_data

    ##########################################################
    # Generate multiple frames
    ##########################################################

    def generate_stream(self, num_frames=10):

        frames = []

        for _ in range(num_frames):
            frames.append(
                self.generate_shift_frame()
            )

        return frames


##############################################################
# Demo
##############################################################

# if __name__ == "__main__":

#     simulator = ImageAnalysisSimulation(
#         grid_size=8,
#         noise_level=0.02,
#         random_seed=42
#     )

#     frame = simulator.generate_shift_frame()

#     print("Generated", len(frame), "block shifts\n")

#     for i, shift in enumerate(frame[:10], start=1):
#         print(f"Block {i:2d} -> {shift}")