import numpy as np
import cv2


def get_reference_centroids(ref_frame, sub_aperture_size, n_lenslets_x, n_lenslets_y):
    # Detects spot positions from a clean reference (plane wavefront) frame.
    height, width = ref_frame.shape
    # Estimate rough grid spacing
    spacing_x = width // n_lenslets_x
    spacing_y = height // n_lenslets_y

    reference_centroids = []

    normalized = ref_frame.astype(np.float32) / np.max(ref_frame)

    for row in range(n_lenslets_y):
        for col in range(n_lenslets_x):
            # Estimated centre of this sub-aperture
            cx_est = int((col + 0.5) * spacing_x)
            cy_est = int((row + 0.5) * spacing_y)

            # Carve out ROI around estimate
            x_start = cx_est - sub_aperture_size // 2
            x_end = cx_est + sub_aperture_size // 2
            y_start = cy_est - sub_aperture_size // 2
            y_end = cy_est + sub_aperture_size // 2

            roi = normalized[y_start:y_end, x_start:x_end]

            # Threshold and centroid (same logic as main function)
            peak = np.max(roi)
            if peak == 0:
                continue  # dead lenslet, skip

            mask = roi > (peak * 0.11)
            cleaned = roi * mask
            total = np.sum(cleaned)

            if total > 0:
                ny, nx = cleaned.shape
                yy, xx = np.mgrid[0:ny, 0:nx]
                cx = np.sum(cleaned * xx) / total + x_start
                cy = np.sum(cleaned * yy) / total + y_start
                reference_centroids.append((cx, cy))

    return reference_centroids


def get_analytical_centroids(
    sensor_width, sensor_height, n_lenslets_x, n_lenslets_y, offset_x=0, offset_y=0
):
    # Builds reference centroids purely from MLA geometry, to be used if we have a calibrated sensor with known pitch.

    spacing_x = sensor_width / n_lenslets_x
    spacing_y = sensor_height / n_lenslets_y

    centroids = []
    for row in range(n_lenslets_y):
        for col in range(n_lenslets_x):
            cx = offset_x + (col + 0.5) * spacing_x
            cy = offset_y + (row + 0.5) * spacing_y
            centroids.append((cx, cy))

    return centroids


def estimate_sub_aperture_size(ref_frame, n_lenslets_x, n_lenslets_y):
    # Estimates sub-aperture size directly from image dimensions.
    height, width = ref_frame.shape
    size_x = width // n_lenslets_x
    size_y = height // n_lenslets_y
    # Use the smaller axis to keep the ROI square and safe from overlap
    return min(size_x, size_y)


# to call - sub_aperture_size = estimate_sub_aperture_size(ref_frame, 10, 10)


def estimate_threshold(ref_frame, sample_region=(0, 50, 0, 50)):
    # Estimates noise floor from a known-dark corner of the frame.
    y0, y1, x0, x1 = sample_region
    dark_patch = ref_frame[y0:y1, x0:x1].astype(np.float32)
    noise_mean = np.mean(dark_patch)
    noise_std = np.std(dark_patch)
    # Threshold at mean + 3 sigma above noise
    threshold_absolute = noise_mean + 3 * noise_std
    peak = np.max(ref_frame)
    return threshold_absolute / peak  # express as fraction of peak


threshold_percent = estimate_threshold(ref_frame)
print(f"Recommended threshold: {threshold_percent:.3f}")


def process_sh_frame(
    frame, reference_centroids, sub_aperture_size, threshold_percent=0.11
):
    # Processes a single SH-WFS frame to find spot deviations.
    # 1. Normalisation & Restrict pixel values to [1] for robustness
    normalized_frame = frame.astype(np.float32) / (np.max(frame) * 1.1)

    measured_centroids = []

    # 2. Iterating through Sub-apertures (Regions of Interest) in our adaptive model, these indices might be non-uniform
    for ref_x, ref_y in reference_centroids:
        # Define the local grid boundary (ROI) around the expected spot position
        x_start, x_end = (
            int(ref_x - sub_aperture_size // 2),
            int(ref_x + sub_aperture_size // 2),
        )
        y_start, y_end = (
            int(ref_y - sub_aperture_size // 2),
            int(ref_y + sub_aperture_size // 2),
        )

        roi = normalized_frame[y_start:y_end, x_start:x_end]

        # 3. Thresholding
        roi_peak = np.max(roi)
        mask = roi > (roi_peak * threshold_percent)
        cleaned_roi = roi * mask

        # 4. Centroid Calculation (First Moment)
        total_intensity = np.sum(cleaned_roi)
        if total_intensity > 0:
            # Weighted average of pixel coordinates by intensity
            ny, nx = cleaned_roi.shape
            yy, xx = np.mgrid[0:ny, 0:nx]

            cx = np.sum(cleaned_roi * xx) / total_intensity
            cy = np.sum(cleaned_roi * yy) / total_intensity

            # Map back to global frame coordinates
            measured_centroids.append((cx + x_start, cy + y_start))
        else:
            measured_centroids.append(
                (ref_x, ref_y)
            )  # Default to reference if spot missing

    # 5. Calculate Deviations (Slopes)
    deviations = np.array(measured_centroids) - np.array(reference_centroids)
    return deviations


# --- Load frames ---
ref_frame = cv2.imread("reference.bmp", cv2.IMREAD_GRAYSCALE)
test_frame = cv2.imread("turbulent.bmp", cv2.IMREAD_GRAYSCALE)

# --- Derive inputs ---
sub_aperture_size = estimate_sub_aperture_size(
    ref_frame, n_lenslets_x=10, n_lenslets_y=10
)
reference_centroids = get_reference_centroids(ref_frame, sub_aperture_size, 10, 10)
threshold_percent = estimate_threshold(ref_frame)

# --- Run the pipeline ---
deviations = process_sh_frame(
    test_frame, reference_centroids, sub_aperture_size, threshold_percent
)

print(f"Spot Shift Matrix shape: {deviations.shape}")  # (N, 2)
print(f"Max deviation: {np.max(np.abs(deviations)):.2f} px")
