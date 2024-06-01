from typing import Tuple, Union
import math
import cv2
import numpy as np

# STEP 1: Import the necessary modules.
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

base_options = python.BaseOptions(model_asset_path="ultils/blaze_face_short_range.tflite")
options = vision.FaceDetectorOptions(base_options=base_options)
detector = vision.FaceDetector.create_from_options(options)


def _normalized_to_pixel_coordinates(
    normalized_x: float, normalized_y: float, image_width: int, image_height: int
) -> Union[None, Tuple[int, int]]:
    """Converts normalized value pair to pixel coordinates."""

    # Checks if the float value is between 0 and 1.
    def is_valid_normalized_value(value: float) -> bool:
        return (value > 0 or math.isclose(0, value)) and (
            value < 1 or math.isclose(1, value)
        )

    if not (
        is_valid_normalized_value(normalized_x)
        and is_valid_normalized_value(normalized_y)
    ):
        # TODO: Draw coordinates even if it's outside of the image bounds.
        return None
    x_px = min(math.floor(normalized_x * image_width), image_width - 1)
    y_px = min(math.floor(normalized_y * image_height), image_height - 1)
    return x_px, y_px


def align_eyes(image, eye_left, eye_right):
    # Calculate the angle between the line connecting the eyes and the horizontal axis
    dx = eye_right[0] - eye_left[0]
    dy = eye_right[1] - eye_left[1]
    angle = np.degrees(np.arctan2(dy, dx))

    # Get the center of the image
    center = (image.shape[1] // 2, image.shape[0] // 2)

    # Calculate the rotation matrix
    M = cv2.getRotationMatrix2D(center, angle, 1)

    # Rotate the image
    aligned_image = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))

    return aligned_image


def align_img(image=None, image_path: str = None, size=224):
    if image is None:
        image = mp.Image.create_from_file(image_path)
    else:
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)

    detection_result = detector.detect(image)
    if not detection_result.detections:
        return None
    image_copy = np.copy(image.numpy_view())
    # Visualize and align image
    height, width, _ = image_copy.shape
    for detection in detection_result.detections:
        for i, keypoint in enumerate(detection.keypoints):
            keypoint_px = _normalized_to_pixel_coordinates(
                keypoint.x, keypoint.y, width, height
            )
            if i == 0:
                eye_left = keypoint_px
            elif i == 1:
                eye_right = keypoint_px

        # Align the eyes horizontally
        if eye_left and eye_right:
            image_copy = align_eyes(image_copy, eye_left, eye_right)

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_copy)
    detection_result = detector.detect(mp_image)
    if not detection_result.detections:
        return None

    # Draw bounding boxes
    image = mp_image.numpy_view()
    for detection in detection_result.detections:
        bbox = detection.bounding_box
        start_point = bbox.origin_x, bbox.origin_y
        end_point = bbox.origin_x + bbox.width, bbox.origin_y + bbox.height
        face = image[start_point[1] : end_point[1], start_point[0] : end_point[0]]
        face = cv2.resize(face, (160, 160))
        # face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
    return face
