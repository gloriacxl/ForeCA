import os
import numpy as np
from torch.utils.data import Dataset
import torch
import cv2
import glob
import imgaug.augmenters as iaa
from perlin import rand_perlin_2d_np
from skimage.feature import hog
from skimage import exposure
from noise import Simplex_CLASS
# from torchvision import transforms
import random

'''
class ToTensor(object):
    def __call__(self, image):
        try:
            image = torch.from_numpy(image.transpose(2, 0,1))
        except:
            print('Invalid_transpose, please make sure images have shape (H, W, C) before transposing')
        if not isinstance(image, torch.FloatTensor):
            image = image.float()
        return image


class Normalize(object):
    """
    Only normalize images
    """
    def __init__(self, mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]):
        self.mean = np.array(mean)
        self.std = np.array(std)
    def __call__(self, image):
        # print('image.shape=',image.shape) # (3, 256, 256)
        image = (image.transpose([1,2,0]) - self.mean) / self.std # image需要是256 256 3
        return image


def get_data_transforms(size, isize):
    data_transforms = transforms.Compose([Normalize(),\
                    ToTensor()])
    gt_transforms = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor()])
    return data_transforms, gt_transforms
'''
    

class MVTecDRAEMTestDataset(Dataset):

    def __init__(self, root_dir, resize_shape=None):
        self.root_dir = root_dir
        self.images = sorted(glob.glob(root_dir+"/*/*.JPG"))
        self.resize_shape=resize_shape

    def __len__(self):
        return len(self.images)

    def transform_image(self, image_path, mask_path):
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if mask_path is not None:
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        else:
            mask = np.zeros((image.shape[0],image.shape[1]))
        if self.resize_shape != None:
            image = cv2.resize(image, dsize=(self.resize_shape[1], self.resize_shape[0]))
            mask = cv2.resize(mask, dsize=(self.resize_shape[1], self.resize_shape[0]))

        image = image / 255.0
        mask = mask / 255.0

        image = np.array(image).reshape((image.shape[0], image.shape[1], 3)).astype(np.float32)
        mask = np.array(mask).reshape((mask.shape[0], mask.shape[1], 1)).astype(np.float32)

        image = np.transpose(image, (2, 0, 1))
        mask = np.transpose(mask, (2, 0, 1))
        return image, mask


    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = self.images[idx]
        dir_path, file_name = os.path.split(img_path)
        base_dir = os.path.basename(dir_path)
        if base_dir == 'good':
            image, mask = self.transform_image(img_path, None)
            has_anomaly = np.array([0], dtype=np.float32)
        else:
            mask_path = os.path.join(dir_path, '../../ground_truth/')
            mask_path = os.path.join(mask_path, base_dir)
            mask_file_name = file_name.split(".")[0]+".png"
            mask_path = os.path.join(mask_path, mask_file_name)
            image, mask = self.transform_image(img_path, mask_path)
            has_anomaly = np.array([1], dtype=np.float32)           

        sample = {'image': image, 'has_anomaly': has_anomaly, 'mask': mask, 'idx': idx}
        #print("has_anomaly=",has_anomaly)
        return sample



class MVTecDRAEMTrainDataset(Dataset):

    def __init__(self, root_dir, anomaly_source_path, resize_shape=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.resize_shape=resize_shape

        self.image_paths = sorted(glob.glob(root_dir+"/*.png"))
        self.obj_name = self.image_paths[0].split('/')[-3] 
        foreground_paths = os.path.join('/home/dataset/Visa_pytorch_foreground/1cls/',self.obj_name , 'foreground/' )  # use your VisA w/ its foreground path!
        self.image_foreground_paths = sorted(glob.glob(foreground_paths+"/*.png"))

        self.augmenters = [iaa.GammaContrast((0.5,2.0),per_channel=True),
                      iaa.MultiplyAndAddToBrightness(mul=(0.8,1.2),add=(-30,30)),
                      iaa.pillike.EnhanceSharpness(),
                      iaa.AddToHueAndSaturation((-50,50),per_channel=True),
                      iaa.Solarize(0.5, threshold=(32,128)),
                      iaa.Posterize(),
                      iaa.Invert(),
                      iaa.pillike.Autocontrast(),
                      iaa.pillike.Equalize(),
                      #iaa.Affine(rotate=(-45, 45))
                      ]

        self.simplexNoise = Simplex_CLASS()


    def __len__(self):
        return len(self.image_paths)


    def randAugmenter(self):
        aug_ind = np.random.choice(np.arange(len(self.augmenters)), 3, replace=False)
        aug = iaa.Sequential([self.augmenters[aug_ind[0]],
                              self.augmenters[aug_ind[1]],
                              self.augmenters[aug_ind[2]]]
                             )
        return aug


    def shuffle_fore_region(self, anomaly_source_img_new):
        non_zero_rows, non_zero_cols = np.nonzero(anomaly_source_img_new[0]) 
        min_row, max_row = np.min(non_zero_rows), np.max(non_zero_rows)
        min_col, max_col = np.min(non_zero_cols), np.max(non_zero_cols)

        center_square = anomaly_source_img_new[:, min_row:max_row + 1, min_col:max_col + 1]

        k = 8
        block_height = center_square.shape[1] // k
        block_width = center_square.shape[2] // k

        blocks = []
        for i in range(k):
            for j in range(k):
                block = center_square[:, i * block_height:(i + 1) * block_height, j * block_width:(j + 1) * block_width]
                blocks.append(block)

        random.shuffle(blocks)

        shuffled_center_square = np.concatenate(blocks, axis=1)
        shuffled_center_square = np.concatenate(np.split(shuffled_center_square, k, axis=1), axis=2)

        output_img = np.copy(anomaly_source_img_new)
        output_img[:, min_row:min_row+block_height*k, min_col:min_col+block_width*k] = shuffled_center_square

        return output_img

    def find_fore_region(self, anomaly_source_img, foreground_i):
        contours, _ = cv2.findContours(foreground_i, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) # 256 256

        rectangles = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            rectangles.append((x, y, x + w, y + h))

        if rectangles:
            external_rectangle = (
                min(rect[0] for rect in rectangles),
                min(rect[1] for rect in rectangles),
                max(rect[2] for rect in rectangles),
                max(rect[3] for rect in rectangles)
            )
            # print("External Rectangle:", external_rectangle)
        else:
            print("No objects found.")

        output_image = np.zeros_like(foreground_i)
        cv2.rectangle(output_image, (external_rectangle[0], external_rectangle[1]), 
                    (external_rectangle[2], external_rectangle[3]), (255,), -1)
        output_image = output_image / 255.0
        output_image = output_image[None, ...]
        anomaly_source_img_new = anomaly_source_img * (output_image)
        anomaly_source_img_new2 = self.shuffle_fore_region(anomaly_source_img_new)
        
        return anomaly_source_img_new2    
    
    def remove_outer_points(self, object_map, foreground_mask):
        object_map = (object_map).astype('uint8')
        _, labeled_map, stats, centroids = cv2.connectedComponentsWithStats(object_map)
        valid_indices = []
        for index, centroid in enumerate(centroids[1:], start=1):
            x, y = centroid.astype(int)
            if foreground_mask[y, x] != 0:
                valid_indices.append(index)
        filtered_object_map = np.zeros_like(object_map)
        for index in valid_indices:
            filtered_object_map[labeled_map == index] = 1
        return filtered_object_map 

    def check_overlap(self, array1, array2, min_overlap=10):
        flattened_array1 = array1.flatten()
        flattened_array2 = array2.flatten()

        common_elements = np.intersect1d(flattened_array1, flattened_array2)
        return len(common_elements) >= min_overlap

    def augment_image(self, image, anomaly_source_img, foreground_mask):
        perlin_scale = 6
        min_perlin_scale = 0
        image = torch.tensor(image).permute(1,2,0).numpy()
        anomaly_source_img = torch.tensor(anomaly_source_img).permute(1,2,0).numpy()
        perlin_scalex = 2 ** (torch.randint(min_perlin_scale, perlin_scale, (1,)).numpy()[0])
        perlin_scaley = 2 ** (torch.randint(min_perlin_scale, perlin_scale, (1,)).numpy()[0])

        threshold = 0.5
        has_overlap = False

        while not has_overlap:
            perlin_noise = rand_perlin_2d_np((self.resize_shape[0], self.resize_shape[1]), (perlin_scalex, perlin_scaley))
            perlin_thr = np.where(perlin_noise > threshold, np.ones_like(perlin_noise), np.zeros_like(perlin_noise))
            perlin_thr = self.remove_outer_points(perlin_thr, foreground_mask)
            has_overlap = self.check_overlap(perlin_thr, foreground_mask,min_overlap=0)
                
        perlin_thr = np.expand_dims(perlin_thr, axis=2)
        img_thr = anomaly_source_img.astype(np.float32) * perlin_thr
        beta = 0.2
        
        augmented_image = image * (1 - perlin_thr) + (1 - beta) * img_thr + beta * image * (
            perlin_thr)
        
        augmented_image = augmented_image.astype(np.float32)
        msk = (perlin_thr).astype(np.float32)
        augmented_image = msk * augmented_image + (1-msk)*image
        has_anomaly = 1.0
        if np.sum(msk) == 0:
            has_anomaly=0.0
        return augmented_image, msk, np.array([has_anomaly],dtype=np.float32)

    def transform_image(self, image, anomaly_source_img, foreground_path):

        foreground_image = cv2.imread(foreground_path,cv2.IMREAD_GRAYSCALE)
        foreground_mask = foreground_image / 255.0
        
        img_noise_new = self.find_fore_region(anomaly_source_img, foreground_image)
        foreground_image = torch.tensor(foreground_image).unsqueeze(0).cpu().numpy().transpose([1, 2, 0])

        augmented_image, anomaly_mask, has_anomaly = self.augment_image(image, img_noise_new, foreground_mask) # new
        augmented_image = np.transpose(augmented_image, (2, 0, 1)) 
        anomaly_mask = np.transpose(anomaly_mask, (2, 0, 1))

        foreground_image = np.array(foreground_image).reshape((foreground_image.shape[0], foreground_image.shape[1], foreground_image.shape[2])).astype(np.float32) / 255.0
        foreground_image = np.transpose(foreground_image, (2, 0, 1))
        return image, augmented_image, anomaly_mask, has_anomaly, foreground_image, img_noise_new
            

    def __getitem__(self, idx):
        cv2.setNumThreads(0)
        idx = torch.randint(0, len(self.image_paths), (1,)).item()

        h_noise = 256
        w_noise = 256
        start_h_noise = 0
        start_w_noise = 0
        
        noise_size = (h_noise, w_noise)
        simplex_noise = self.simplexNoise.rand_3d_octaves((3, *noise_size), 8, 0.6)
        init_zero = np.zeros((256,256,3)) # ori
        init_zero[start_h_noise: start_h_noise + h_noise, start_w_noise: start_w_noise+w_noise, :] = 0.2 * simplex_noise.transpose(1,2,0)
        init_zero = init_zero.transpose([2,0,1])
        image = cv2.imread(self.image_paths[idx])
        image = cv2.resize(image, dsize=(self.resize_shape[1], self.resize_shape[0]))
        image = np.array(image).reshape((image.shape[0], image.shape[1], image.shape[2])).astype(np.float32) / 255.0
        image = np.transpose(image, (2, 0, 1))
        
        img_noise = image + init_zero
        img_noise = np.where(img_noise > 1, 1, img_noise)

        image, augmented_image, anomaly_mask, has_anomaly, foreground_image, img_noise_new = self.transform_image(image,
                                                                           img_noise,self.image_foreground_paths[idx])
        anomaly_mask_2 = torch.tensor(anomaly_mask).squeeze(0).cpu().numpy()
        anomaly_mask_2 = anomaly_mask_2.flatten()
        foreground_image_2 = torch.tensor(foreground_image).squeeze(0).cpu().numpy()
        foreground_image_2 = foreground_image_2.flatten()
        for i in range(65536):
            if anomaly_mask_2[i] != 0:
                foreground_image_2[i] = 1
        foreground_image_2_ = foreground_image_2.reshape(256,256)
        foreground_image_2_ = torch.as_tensor(foreground_image_2_).unsqueeze(0).cpu().numpy()

        sample = {'image': image, "anomaly_mask": anomaly_mask,
                  'augmented_image': augmented_image, 'has_anomaly': has_anomaly, 'idx': idx, 'foreground_image': foreground_image, 'foreground_image_new':foreground_image_2_, 'img_noise':img_noise_new}

        return sample
