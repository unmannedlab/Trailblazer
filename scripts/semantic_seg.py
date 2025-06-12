import os
import cv2
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from models.segformer import Segformer
from configs.segformer_test import config as cfg
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt
from rospkg import RosPack
import sys

class SegmentationInference:
    def __init__(self, model_path, device=None, patch_size=512, overlap=128):
        """
        Initialize the segmentation model and parameters.
        """
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.load_model(model_path)
        self.patch_size = patch_size
        self.overlap = overlap
        self.palette = np.random.randint(0, 256, (256, 3), dtype=np.uint8)

    def load_model(self, model_path):
        """
        Load the SegFormer model.
        """
        model = Segformer(
            pretrained=model_path
        ).to(self.device)
        model.eval()
        return model

    def infer(self, image_patch):
        """
        Perform inference on a single image patch.
        """
        shape = image_patch.shape[:2]
        image_tensor = transforms.ToTensor()(image_patch).unsqueeze(0).to(self.device)
        pred = self.model(image_tensor)
        pred = F.interpolate(pred, size=shape, mode="bilinear", align_corners=False)
        pred = torch.argmax(pred, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
        return pred

    def preprocess_with_overlap(self, image_array):
        """
        Preprocess the image with overlapping patches, ensuring full coverage including edges.
        """
        patches = []
        coords = []
        h, w = image_array.shape[:2]
        step = self.patch_size - self.overlap

        for i in range(0, h - self.overlap, step):
            for j in range(0, w - self.overlap, step):
                patch = image_array[i:i + self.patch_size, j:j + self.patch_size]
                patches.append(patch)
                coords.append((i, j))

        # Handle the bottom edge
        for j in range(0, w - self.overlap, step):
            patch = image_array[h - self.patch_size:h, j:j + self.patch_size]
            patches.append(patch)
            coords.append((h - self.patch_size, j))

        # Handle the right edge
        for i in range(0, h - self.overlap, step):
            patch = image_array[i:i + self.patch_size, w - self.patch_size:w]
            patches.append(patch)
            coords.append((i, w - self.patch_size))

        # Handle the bottom-right corner
        patch = image_array[h - self.patch_size:h, w - self.patch_size:w]
        patches.append(patch)
        coords.append((h - self.patch_size, w - self.patch_size))

        return patches, coords

    def merge_patches(self, patches, coords, image_shape):
        """
        Merge overlapping patches into a single segmentation mask.
        """
        h, w = image_shape
        merged_mask = np.zeros((h, w), dtype=np.float32)
        weight_map = np.zeros((h, w), dtype=np.float32)
        step = self.patch_size - self.overlap

        for patch, (i, j) in zip(patches, coords):
            merged_mask[i:i + self.patch_size, j:j + self.patch_size] += patch
            weight_map[i:i + self.patch_size, j:j + self.patch_size] += 1

        merged_mask = np.divide(merged_mask, weight_map, out=np.zeros_like(merged_mask), where=weight_map != 0)
        return merged_mask.astype(np.uint8)

    def process_image(self, image):
        """
        Main function to process an image and return the segmentation mask.
        """
        shape = image.size
        print(shape)
        # Resize image to the closest 512 multiple
        new_size = tuple(map(lambda x: ((x - 1) // 512 + 1) * 512, shape))
        image = image.resize(new_size, Image.BILINEAR)
        image_array = np.array(image)

        patches, coords = self.preprocess_with_overlap(image_array)

        print("Predicting masks...")
        pred_patches = []
        for patch in tqdm(patches):
            pred = self.infer(patch)
            pred_patches.append(pred)

        # Merge patches into the complete mask
        complete_mask = self.merge_patches(pred_patches, coords, image_array.shape[:2])

        return complete_mask

# Example usage in another script:
if __name__ == "__main__":
    # Initialize the SegmentationInference class
    rospack = RosPack()
    package_path = rospack.get_path('camel')
    model_path = os.path.join(package_path,'scripts/weights/segformer/best.pth')
    seg_infer = SegmentationInference(model_path)

    # Input image path (replace with actual path)
    image_path = '/media/usl/NV/Camp_roberts/images/16038995.800000000,6043502.000000000,1790982.900000000,1795509.800000000 [].tif'

    # Process image and get the segmentation mask
    complete_mask = seg_infer.process_image(image_path)

    # Display the complete mask
    plt.figure(figsize=(20, 20))
    plt.imshow(seg_infer.palette[complete_mask])
    plt.title('Complete Segmentation Mask')
    plt.axis('off')
    plt.show()
