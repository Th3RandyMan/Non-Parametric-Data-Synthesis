import glob
from typing import Tuple
from matplotlib import pyplot as plt
import numpy as np
from PIL import Image
import random

class Synthesize:
    def __init__(self, example_image:np.ndarray, synth_size:Tuple[int,int], kernel_size:int, start_patch_size:int, sigma:float=3.0, threshold:float=0.8, attenuation_factor:float=80, save_path:str=None):
        self.example_image = example_image
        self.synth_size = synth_size
        self.kernel_size = kernel_size
        self.start_patch_size = start_patch_size
        self.sigma = sigma
        self.threshold = threshold
        self.attenuation_factor = attenuation_factor
        self.save_path = save_path

        self.height, self.width = self.example_image.shape[:2]
        self.half_kernel = self.kernel_size // 2
        self.synth_image, self.potential_map = self._initialize_synthesized_image()
        self.example_patches = self._get_patches(example_image)
        self.gaussian_kernel = np.repeat(self._create_gaussian_kernel(kernel_size, sigma)[:,:,np.newaxis], example_image.shape[-1], axis=2)

        self._synthesize()

    def _initialize_synthesized_image(self) -> Tuple[np.ndarray, np.ndarray]:   # CHANGE THIS TO BE LIST OF PIXELS IN ORDER OF PROCESSING
        """
        Initialize the synthesized image by randomly selecting a patch from the example image.
        Add this patch to the center of the synthesized image, and update the potential map.
        The potential map will be valued based on the distance from the center patch.
        """
        diff = self.start_patch_size // 2

        # Create a blank image to store the synthesized image
        synthesized_img = np.zeros(self.synth_size + (self.example_image.shape[-1],))
        potential_map = np.zeros(self.synth_size, dtype=int)

        # Randomly select a 3x3 patch to add to the center of the synthesized image
        center_i = (synthesized_img.shape[0] // 2)
        center_j = (synthesized_img.shape[1] // 2)
        row = random.randint(0, self.example_image.shape[0] - self.start_patch_size)
        col = random.randint(0, self.example_image.shape[1] - self.start_patch_size)
        synthesized_img[center_i-diff:center_i+diff+1, center_j-diff:center_j+diff+1] = self.example_image[row:row+self.start_patch_size, col:col+self.start_patch_size,:]/255.0

        # Update the potential map as distance from the center patch    # CHANGE THIS TO BE LIST OF PIXELS IN ORDER OF PROCESSING
        for i in range(synthesized_img.shape[0]):
            for j in range(synthesized_img.shape[1]):
                c_i, c_j = center_i, center_j
                if i <= center_i - diff:
                    c_i = center_i - diff
                elif i >= center_i + diff:
                    c_i = center_i + diff
                else:
                    c_i = i

                if j <= center_j - diff:
                    c_j = center_j - diff
                elif j >= center_j + diff:
                    c_j = center_j + diff
                else:
                    c_j = j

                potential_map[i,j] = (i - c_i)**2 + (j - c_j)**2

        potential_map = np.max(potential_map) - potential_map + 1
        potential_map[center_i-diff:center_i+diff+1, center_j-diff:center_j+diff+1] = 0

        return synthesized_img, potential_map
    
    def _get_patches(self, img, stride:int=1):
        patches = []
        height, width = img.shape[:2]
        for i in range(height - self.kernel_size, stride):
            for j in range(width - self.kernel_size, stride):
                patch = img[i:i+self.kernel_size, j:j+self.kernel_size]
                patches.append(patch)
        return np.array(patches)
    
    def _create_gaussian_kernel(self, size, sigma=None):
        if sigma is None:
            sigma = 0.3 * ((size - 1) * 0.5 - 1) + 0.8
        kernel = np.fromfunction(lambda x, y: (1/(2*np.pi*sigma**2)) * np.exp(-((x-(self.kernel_size-1)/2)**2 + (y-(self.kernel_size-1)/2)**2)/(2*sigma**2)), (self.kernel_size, self.kernel_size))
        return kernel / np.sum(kernel)

    def _synthesize(self):
        n_no_pixels = np.where(self.potential_map != 0)[0].shape[0]   # Number of pixels that have not been processed

        perc = 0.1
        # Iterate over the unfilled pixels in the synthesized image
        for n_pixel in range(n_no_pixels):
            if n_pixel/n_no_pixels > perc:
                print(f"Progress: {perc*100:.1f}%")
                perc += 0.1
                if self.save_path:
                    self.plot_synthesized_image(save_path=self.save_path+f"_{int(perc*100)}.png")
            
            # Find the pixel with the highest potential
            y, x = np.unravel_index(np.argmax(self.potential_map), self.potential_map.shape)

            # Get the target patch and its mask
            target_patch, patch_mask = self._get_target_patch(y, x)

            # Find the best matching patch from the example image
            best_patch = self._find_best_patch(target_patch, patch_mask)

            # Update the synthesized image and potential map
            self.synth_image[y,x] = best_patch[self.half_kernel, self.half_kernel]
            self.potential_map[y,x] = 0
        #return self.synth_image

    def _get_target_patch(self, y, x):
        # Create a zero matrix to store the target patch
        target_patch = np.zeros((self.kernel_size, self.kernel_size, self.synth_image.shape[-1]))
        patch_mask = np.zeros((self.kernel_size, self.kernel_size))

        start_y, start_x = max(y-self.half_kernel, 0), max(x-self.half_kernel, 0)
        end_y, end_x = min(y+self.half_kernel+1, self.synth_image.shape[0]), min(x+self.half_kernel+1, self.synth_image.shape[1])

        # Copy the patch from the synthesized image to the target patch
        target_patch[start_y-y+self.half_kernel:end_y-y+self.half_kernel, start_x-x+self.half_kernel:end_x-x+self.half_kernel] = self.synth_image[start_y:end_y, start_x:end_x]

        # Create mask for known pixels
        patch_mask[start_y-y+self.half_kernel:end_y-y+self.half_kernel, start_x-x+self.half_kernel:end_x-x+self.half_kernel] = self.potential_map[start_y:end_y, start_x:end_x]
        patch_mask = (patch_mask == 0)

        return target_patch, np.repeat(patch_mask[:,:,np.newaxis], self.synth_image.shape[-1], axis=2)
    
    def _find_best_patch(self, target_patch, patch_mask):
                # Calculate the squared difference between the target patch and all example patches
        squared_diffs = np.sum(self.gaussian_kernel * ( patch_mask * (self.example_patches - target_patch))**2, axis=(1, 2, 3)) # sum over the height, width, and channels with gaussian weighting

        # Convert the squared differences to probabilities
        prob = 1 - squared_diffs / np.max(squared_diffs)
        prob *= (prob > self.threshold*np.max(prob))  # thresholding the probabilities
        prob = prob**self.attenuation_factor  # attenuate the probabilities
        prob /= np.sum(prob)  # normalize the probabilities
        
        # Find the index of the best matching patch
        best_matching_patch_index = np.random.choice(np.arange(self.example_patches.shape[0]), p=prob)
        
        return self.example_patches[best_matching_patch_index]
    
    def plot_example_patches(self, n:int=5, m:int=5, figsize=(15, 15), save_path=None):
        fig, ax = plt.subplots(n, m, figsize=figsize)
        for i in range(n):
            for j in range(m):
                ax[i, j].imshow(self.example_patches[i*n+j])
                ax[i, j].axis('off')
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()

    def plot_synthesized_image(self, figsize=(10, 10), save_path=None):
        plt.figure(figsize=figsize)
        plt.imshow(self.synth_image)
        plt.axis('off')
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()

    def plot_potential_map(self, figsize=(10, 10), save_path=None):
        _, potential_map = self._initialize_synthesized_image()

        plt.figure(figsize=figsize)
        plt.imshow(potential_map)
        plt.axis('off')
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()
    



if __name__ == '__main__':
    IMG_NUM = -3
    SYNTH_SIZE = (530, 530) # Height, Width of the synthesized image
    KERNEL_SIZE = 15 # Size of the kernel for the window search
    START_PATCH_SIZE = 3 # Size of the starting patch
    SIGMA = 3.0 # Standard deviation of the Gaussian kernel
    THRESHOLD = 0.8 # Threshold for the probabilities
    ATTENUATION_FACTOR = 80 # Attenuation factor for the probabilities

    # Load the images
    image_paths = glob.glob('Texture Patches/*.png')
    images = [np.array(Image.open(file)) for file in image_paths]

    # Select image to synthesize
    image = images[IMG_NUM]

    # Plot the image
    print(f"Image {image_paths[IMG_NUM]} shape: {image.shape}")

    # Synthesize the image
    Synth = Synthesize(image, SYNTH_SIZE, KERNEL_SIZE, START_PATCH_SIZE, SIGMA, THRESHOLD, ATTENUATION_FACTOR, save_path='synth_7')


