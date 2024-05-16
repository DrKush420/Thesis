import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import cv2
import glob



def get_unlabelled_data(path):
    data=sorted(glob.glob(os.path.join(path, '*.png')))
    print(len(data))
    return data



class CustomDataset(Dataset):
    def __init__(self, image_paths, coordinates):
        self.image_paths = image_paths
        self.coordinates = coordinates
        self.transform = transforms.ToTensor()
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)
        return image, self.coordinates

def crop_and_warp_batch(batch_images, batch_coords, perspective_matrix):
    batch_size = batch_images.size(0)
    perspective_matrix = torch.tensor(perspective_matrix, dtype=torch.float32, device=batch_images.device)

    # Convert coordinates to tensors
    coords_tensor = torch.tensor(batch_coords, dtype=torch.float32, device=batch_images.device)
    
    cropped_images = []
    warped_images = []

    for i in range(batch_size):
        image = batch_images[i]
        coords = coords_tensor

        # Crop each image based on the given coordinates
        for coord in coords:
            x1, y1, x2, y2, x3, y3, x4, y4 = coord
            src_pts = np.array([[x1, y1], [x2, y2], [x3, y3], [x4, y4]], dtype=np.float32)

            # Calculate bounding box for cropping
            ymin, xmin = int(np.min(src_pts[:, 1])), int(np.min(src_pts[:, 0]))
            ymax, xmax = int(np.max(src_pts[:, 1])), int(np.max(src_pts[:, 0]))
            cropped = image[:, ymin:ymax, xmin:xmax]
            cropped_images.append(cropped)

            # Warp perspective
            h, w = cropped.shape[1], cropped.shape[2]
            src_pts -= [xmin, ymin]  # Adjust points to cropped image coordinates

            dst_pts = np.dot(perspective_matrix.cpu().numpy(), np.vstack([src_pts.T, np.ones((1, src_pts.shape[0]))]))
            dst_pts = (dst_pts / dst_pts[2, :])[:2, :].T

            M = torch.tensor(cv2.getPerspectiveTransform(src_pts, dst_pts), dtype=torch.float32, device=batch_images.device)
            grid = torch.nn.functional.affine_grid(M[:2, :].unsqueeze(0), cropped.unsqueeze(0).size(), align_corners=False)
            warped = torch.nn.functional.grid_sample(cropped.unsqueeze(0), grid, align_corners=False)
            warped_images.append(warped.squeeze(0))

    return cropped_images, warped_images

def save_images(images, base_path, batch_index, prefix):
    if not os.path.exists(base_path):
        os.makedirs(base_path)
    for i, image in enumerate(images):
        image_np = image.permute(1, 2, 0).cpu().numpy().astype(np.uint8)
        output_path = os.path.join(base_path, f'{prefix}_batch_{batch_index}_image_{i}.png')
        Image.fromarray(image_np).save(output_path)

# Example usage
image_paths = get_unlabelled_data("./data/corn/corn/unlabelled")

# Coordinates for cropping (x1, y1, x2, y2, x3, y3, x4, y4)
coordinates = [
    (210, 90, 210, 690, 810, 690, 810, 90),
    (1110, 90, 1110, 690, 1710, 690, 1710, 90),
    (510, 390, 510, 990, 1110, 990, 1110, 390),
    (810, 390, 810, 990, 1410, 990, 1410, 390)
]

# Provided perspective transformation matrix (3x3)
perspective_matrix = np.array([
    [1.51130629e+00, 1.92260258e+00, -4.89180994e+02],
    [1.33971426e-16, 5.14071420e+00, -9.87483229e+02],
    [1.10457464e-19, 2.00326720e-03, 1.00000000e+00]
], dtype=np.float32)

batch_size = 16
dataset = CustomDataset(image_paths, coordinates)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

for batch_index, (batch_images, batch_coords) in enumerate(dataloader):
    batch_images = batch_images.to(device)
    cropped_images, warped_images = crop_and_warp_batch(batch_images, batch_coords, perspective_matrix)



    # Save warped images
    save_images(warped_images, './data/corn/corn/unlabelled_processed', batch_index, 'warped')
