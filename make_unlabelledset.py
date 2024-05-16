import cv2
import os
import numpy as np
import gc

matrix = np.array([
    [1.51130629e+00, 1.92260258e+00, -4.89180994e+02],
    [1.33971426e-16, 5.14071420e+00, -9.87483229e+02],
    [1.10457464e-19, 2.00326720e-03, 1.00000000e+00]
], dtype=np.float32)

coords = [
    (210, 90, 210, 690, 810, 690, 810, 90),
    (1110, 90, 1110, 690, 1710, 690, 1710, 90),
    (510, 390, 510, 990, 1110, 990, 1110, 390),
    (810, 390, 810, 990, 1410, 990, 1410, 390)
]

def save_every_10th_frame_from_videos(video_folder, output_folder):
    # Ensure output directory exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    video_files = [os.path.join(video_folder, f) for f in os.listdir(video_folder) if f.endswith(('.mp4', '.avi'))]
    print(video_files[:2])
    video_files=video_files[4:]
    if not video_files:
        print("No video files found in the specified folder.")
        return

    for video_file in video_files:
        cap = cv2.VideoCapture(video_file)
        if not cap.isOpened():
            print(f"Error opening video file {video_file}")
            continue
        
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_name = os.path.splitext(os.path.basename(video_file))[0]
        
        for frame_idx in range(3250, frame_count, 10):  # Save every 10th frame        at 3340 vid 3
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()

            if not ret:
                break
            
            frame = cv2.warpPerspective(frame, matrix, (1920, 1080))
            i=0
            for coord in coords:
                x1, y1, x2, y2, x3, y3, x4, y4 = coord
                src_pts = np.array([[x1, y1], [x2, y2], [x3, y3], [x4, y4]], dtype=np.float32)

                # Calculate bounding box for cropping
                ymin, xmin = int(np.min(src_pts[:, 1])), int(np.min(src_pts[:, 0]))
                ymax, xmax = int(np.max(src_pts[:, 1])), int(np.max(src_pts[:, 0]))
                cropped = frame[ymin:ymax, xmin:xmax]
                output_path = os.path.join(output_folder, f'{video_name}_frame_{frame_idx}_{i}.jpg')
                try:
                    cv2.imwrite(output_path, cropped)
                    
                    #print(f'Saved frame {frame_idx} of {video_file}')
                except cv2.error as e:
                    print(f"Failed to write frame {frame_idx} of {video_file} to {output_path}: {e}")
                finally:
                    i+=1
            if frame_idx % 200==0:
                cap.release()
                cap = cv2.VideoCapture(video_file)
                os.sync()
                gc.collect()
                if not cap.isOpened():
                    print(f"Error opening video file {video_file}")
                    continue
        cap.release()
        print(f'Finished processing {video_file}')

# Usage
video_folder = './data/corn/videos'  # Corrected path
output_folder = './data/corn/corn/unlabelled'  # Corrected path
save_every_10th_frame_from_videos(video_folder, output_folder)
