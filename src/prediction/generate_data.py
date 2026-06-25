import numpy as np
import pandas as pd
from pathlib import Path


def generate_zernike_timeseries(
    num_frames=1000,
    noise_level=0.05,
    save_path="data/synthetic/zernike_timeseries.csv"
):
    t = np.arange(num_frames)

    # Base coefficient

    z1 = np.zeros(num_frames)

    for i in range(1, num_frames):
        z1[i] = (
            0.97 * z1[i - 1]
            + np.random.normal(0, noise_level)
        )

    # Correlated coefficients

    z2 = (
        0.7 * z1
        + np.random.normal(
            0,
            noise_level,
            num_frames
        )
    )

    z3 = (
        0.5 * z2
        + np.random.normal(
            0,
            noise_level,
            num_frames
        )
    )

    z4 = np.zeros(num_frames)

    for i in range(1, num_frames):
        z4[i] = (
            0.94 * z4[i - 1]
            + np.random.normal(
                0,
                noise_level
            )
        )

    z5 = (
        0.6 * z4
        + np.random.normal(
            0,
            noise_level,
            num_frames
        )
    )

    z6 = (
        0.4 * z1
        + 0.4 * z4
        + np.random.normal(
            0,
            noise_level,
            num_frames
        )
    )

    for _ in range(10):

        idx = np.random.randint(
            50,
            num_frames - 50
        )

        magnitude = np.random.uniform(
            0.5,
            1.5
        )

        burst = np.array([
            0.2,
            0.5,
            0.8,
            1.0,
            0.8,
            0.5,
            0.2
        ]) * magnitude

        z1[idx:idx+7] += burst
        z4[idx:idx+7] += burst

    df = pd.DataFrame({
        "frame": t,
        "z1": z1,
        "z2": z2,
        "z3": z3,
        "z4": z4,
        "z5": z5,
        "z6": z6
    })

    Path(save_path).parent.mkdir(
        parents=True,
        exist_ok=True
    )

    df.to_csv(save_path, index=False)

    print(f"Saved dataset to {save_path}")
    print(df.head())


if __name__ == "__main__":
    generate_zernike_timeseries()