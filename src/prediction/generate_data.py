import numpy as np
import pandas as pd
from pathlib import Path


def generate_zernike_timeseries(
    num_frames=1000,
    noise_level=0.05,
    save_path="data/synthetic/zernike_timeseries.csv"
):
    t = np.arange(num_frames)

    z1 = np.sin(0.05 * t)

    z2 = np.cos(0.04 * t)

    z3 = np.sin(0.03 * t + 1)

    z4 = np.sin(0.02 * t) + np.random.normal(
        0, noise_level, num_frames
    )

    z5 = np.cos(0.015 * t) + np.random.normal(
        0, noise_level, num_frames
    )

    df = pd.DataFrame({
        "frame": t,
        "z1": z1,
        "z2": z2,
        "z3": z3,
        "z4": z4,
        "z5": z5
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