
import cv2
import numpy as np

def split_video_left_right(input_video_path, output_left_path, output_right_path, width_threshold=None):

    
    cap = cv2.VideoCapture(input_video_path)
    
    if not cap.isOpened():
        print(f"Error: Could not open video {input_video_path}")
        return
    
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if width_threshold is None:
        split_x = frame_width // 2 
    else:
        split_x = min(width_threshold, frame_width)
    
    new_width = split_x
    new_height = frame_height
    
    print(f"Original video: {frame_width}x{frame_height}, {fps} FPS, {total_frames} frames")
    print(f"Split point: {split_x}px")
    print(f"Output dimensions: {new_width}x{new_height}")
    print("converting...")
    
    fourcc = cv2.VideoWriter_fourcc(*'avc1')  

    split_x = (frame_width + 1) // 2
    half_width = split_x 

    out_left = cv2.VideoWriter(output_left_path, fourcc, fps, (half_width, frame_height))
    out_right = cv2.VideoWriter(output_right_path, fourcc, fps, (half_width, frame_height))

    if not out_left.isOpened() or not out_right.isOpened():
        print("One or both writers failed to open! Try a different codec.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        left = frame[:, :split_x]
        right = frame[:, split_x:]

        if left.shape[1] < half_width:
            left = np.hstack((left, np.zeros((frame_height, half_width - left.shape[1], 3), dtype=np.uint8)))
        if right.shape[1] < half_width:
            right = np.hstack((right, np.zeros((frame_height, half_width - right.shape[1], 3), dtype=np.uint8)))

        out_left.write(left)
        out_right.write(right)

    out_left.release()
    out_right.release()
    cap.release()
    
    #print(f"Processing complete! Created {frame_count} frames.")
    print(f"Left half saved to: {output_left_path}")
    print(f"Right half saved to: {output_right_path}")
