import onnxruntime
import numpy as np
from ultils.extract_face import align_img
import cv2
import time

# Load the ONNX model
session = onnxruntime.InferenceSession("models/face_verification_batch.onnx")


def preprocess_image(image_numpy):
    image = cv2.cvtColor(image_numpy, cv2.COLOR_BGR2RGB)
    image = align_img(image)
    if image is None:
        return None
    image = image.astype(np.float32) / 255.0
    image = (image - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
    return image


def compute_embedding(image_tensor):
    # print(np.array(image_tensor).shape)
    if image_tensor.ndim == 3:
        image_tensor = np.expand_dims(image_tensor, axis=0)
    image_tensor = image_tensor.transpose((0, 3, 1, 2))
    image_tensor = image_tensor.astype(np.float32)
    inputs = {session.get_inputs()[0].name: image_tensor}
    outputs = session.run(None, inputs)
    embeddings = outputs[0]
    return embeddings


def process_list(input_list, thress_hold_frame=30):
    result = []
    count = 0
    input_list = np.array(input_list)
    for (
        num
    ) in input_list.flatten():  # Flatten the array to iterate over individual elements
        if num == 1:
            count += 1
        else:
            if count > 0:
                result.append(count)
                count = 0
            result.append(num)
    if count > 0:
        result.append(count)

    raw_frames_index = []
    for i in result:
        if i == 0:
            raw_frames_index.append(0)
        elif i >= thress_hold_frame:
            for j in range(i):
                raw_frames_index.append(1)
        else:
            for j in range(i):
                raw_frames_index.append(0)
    return raw_frames_index


def export_different_face_video(
    video_path,
    save=True,
    threshold=0.8,
    threshold_time=2,
    output_name="optimize",
    optimize=400,
):
    """
    Export a video with frames containing different faces.

    Args:
        video_path (str): The path to the input video file.
        save (bool, optional): Whether to save the output video. Defaults to False.
        threshold (float, optional): The distance threshold for face verification. Defaults to 0.6.
        threshold_time (int, optional): The number of consecutive frames with distances greater than the threshold to consider as a different face. Defaults to 1.
        output_name (str, optional): The name of the output video file. Defaults to "optimize".
        optimize (int, optional): The number of frames to process at once. Defaults to 200.

    Returns:
        None
    """
    # Read the video
    cap = cv2.VideoCapture(video_path)
    anchor = None
    frames = []
    raw_frames = []
    count = 0
    start = time.time()

    if save:
        fourcc = cv2.VideoWriter_fourcc(*"MP4V")
        out = cv2.VideoWriter(f"{output_name}.avi", fourcc, 30.0, (1920, 1080))

    while cap.isOpened():
        count += 1

        ret, frame = cap.read()
        if ret == True:
            if count >= 5 and anchor is None:
                anchor = preprocess_image(frame)
                if anchor is not None:
                    print("Get anchor at frame: ", count)
                else:
                    continue
            elif count > 5 and anchor is not None:
                raw_frames.append(frame)
                frame = preprocess_image(frame)
                if frame is None:
                    frame = np.zeros((160, 160, 3))
                frames.append(frame)

            if (
                count % optimize == 0 or not ret
            ):  # Process every 1000 frames or at the end of the video
                video_tensor = np.stack(frames)
                video_embed = compute_embedding(video_tensor)
                anchor_embed = compute_embedding(anchor)

                distances = np.sqrt(np.sum((video_embed - anchor_embed) ** 2, axis=1))
                binary_matrix = np.where(distances > threshold, 1, 0)
                processed_list = process_list(
                    binary_matrix, thress_hold_frame=threshold_time * 20
                )
                if save:
                    print(
                        f"Save video with distances greater than {threshold} to {output_name}"
                    )
                    # print(processed_list)
                    for i, num in enumerate(processed_list):
                        if num == 1:
                            out.write(raw_frames[i])

                # Reset frames and raw_frames for the next chunk
                frames = []
                raw_frames = []
        else:
            break

    end = time.time()
    cap.release()
    if save:
        out.release()

    print("Time taken to read the video: ", end - start)


if __name__ == "__main__":
    export_different_face_video(
        "./video.mp4",
        save=True,
        threshold=0.8,
        optimize=400,
        threshold_time=2,
        output_name="onnx_batch",
    )
