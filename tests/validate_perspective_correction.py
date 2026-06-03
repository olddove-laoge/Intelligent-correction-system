from pathlib import Path

import cv2
import numpy as np

from preprocessing.paper_perspective_correction import (
    A4_ASPECT_RATIO,
    auto_correct_paper_perspective,
    correct_paper_perspective,
)


ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT / "artifacts"
ARTIFACTS.mkdir(exist_ok=True)

input_path = ARTIFACTS / "synthetic_input.png"
output_path = ARTIFACTS / "synthetic_corrected.png"
auto_output_path = ARTIFACTS / "synthetic_corrected_auto.png"
expected_path = ARTIFACTS / "synthetic_expected.png"
auto_debug_mask_path = ARTIFACTS / "synthetic_auto_mask.png"

canvas = np.full((1000, 1400, 3), 210, dtype=np.uint8)
expected = np.full((900, 636, 3), 245, dtype=np.uint8)

cv2.rectangle(expected, (0, 0), (635, 899), (255, 255, 255), -1)
for y in range(80, 820, 90):
    cv2.line(expected, (70, y), (560, y), (30, 30, 30), 3)
cv2.putText(expected, "EXAM PAPER", (150, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (20, 20, 20), 3, cv2.LINE_AA)
cv2.putText(expected, "Q1", (75, 160), cv2.FONT_HERSHEY_SIMPLEX, 1, (20, 20, 20), 2, cv2.LINE_AA)
cv2.putText(expected, "Q2", (75, 340), cv2.FONT_HERSHEY_SIMPLEX, 1, (20, 20, 20), 2, cv2.LINE_AA)
cv2.putText(expected, "Q3", (75, 520), cv2.FONT_HERSHEY_SIMPLEX, 1, (20, 20, 20), 2, cv2.LINE_AA)
cv2.putText(expected, "Q4", (75, 700), cv2.FONT_HERSHEY_SIMPLEX, 1, (20, 20, 20), 2, cv2.LINE_AA)

source_points = np.array(
    [[0, 0], [expected.shape[1] - 1, 0], [expected.shape[1] - 1, expected.shape[0] - 1], [0, expected.shape[0] - 1]],
    dtype=np.float32,
)

destination_points = np.array(
    [[280, 120], [1060, 170], [1130, 900], [220, 860]],
    dtype=np.float32,
)

forward_matrix = cv2.getPerspectiveTransform(source_points, destination_points)
warped = cv2.warpPerspective(expected, forward_matrix, (canvas.shape[1], canvas.shape[0]), borderMode=cv2.BORDER_TRANSPARENT)
mask = warped.sum(axis=2) > 0
canvas[mask] = warped[mask]

cv2.imwrite(str(input_path), canvas)
cv2.imwrite(str(expected_path), expected)

corrected, matrix, ordered_points = correct_paper_perspective(
    canvas,
    destination_points,
    aspect_ratio=A4_ASPECT_RATIO,
    output_height=900,
)
cv2.imwrite(str(output_path), corrected)
manual_diff = float(np.mean(np.abs(corrected.astype(np.float32) - expected.astype(np.float32))))

auto_corrected, auto_matrix, auto_points, auto_mask, mask_name = auto_correct_paper_perspective(
    canvas,
    aspect_ratio=A4_ASPECT_RATIO,
    output_height=900,
)
cv2.imwrite(str(auto_output_path), auto_corrected)
cv2.imwrite(str(auto_debug_mask_path), auto_mask)
auto_diff = float(np.mean(np.abs(auto_corrected.astype(np.float32) - expected.astype(np.float32))))

print(f"manual_points={ordered_points.tolist()}")
print("manual_matrix=")
print(matrix)
print(f"manual_mean_abs_diff={manual_diff:.4f}")
print(f"auto_points={auto_points.tolist()}")
print(f"auto_mask_name={mask_name}")
print("auto_matrix=")
print(auto_matrix)
print(f"auto_mean_abs_diff={auto_diff:.4f}")
print(f"input={input_path}")
print(f"expected={expected_path}")
print(f"manual_output={output_path}")
print(f"auto_output={auto_output_path}")
print(f"auto_mask={auto_debug_mask_path}")

if manual_diff > 15.0:
    raise SystemExit(f"Manual validation failed: mean_abs_diff={manual_diff:.4f}")

if auto_diff > 20.0:
    raise SystemExit(f"Auto validation failed: mean_abs_diff={auto_diff:.4f}")
