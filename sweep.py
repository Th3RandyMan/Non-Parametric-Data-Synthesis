import glob
import random
from typing import Tuple
import numpy as np
from PIL import Image
import time
import matplotlib.pyplot as plt
import os


# Get patches for the image
def get_patches(image_array:np.ndarray, kernel_size:int, stride:int=1) -> np.ndarray:
    patches = []
    for i in range(0, image_array.shape[0]-kernel_size, stride):
        for j in range(0, image_array.shape[1]-kernel_size, stride):
            patch = image_array[i:i+kernel_size, j:j+kernel_size, :] / 255.0
            patches.append(patch)
    return np.array(patches)

def create_gaussian_kernel(kernel_size:int, sigma=None) -> np.ndarray:
    if sigma is None:
        sigma = 0.3 * ((kernel_size - 1) * 0.5 - 1) + 0.8   # Scaling sigma based on kernel size, may need to adjust the coefficients
    kernel = np.fromfunction(lambda x, y: (1/(2*np.pi*sigma**2)) * np.exp(-((x-(kernel_size-1)/2)**2 + (y-(kernel_size-1)/2)**2)/(2*sigma**2)), (kernel_size, kernel_size))
    return kernel / np.sum(kernel)

def initialize_synthesized_image(example_img:np.ndarray, synth_size:Tuple[int,int], patch_size:int=3) -> Tuple[np.ndarray, np.ndarray]:
    diff = patch_size // 2

    # Create a blank image to store the synthesized image
    synthesized_img = np.zeros(synth_size + (example_img.shape[-1],), dtype=np.float32)
    potential_map = np.zeros(synth_size, dtype=int)
    #visited_map = np.zeros(synth_size, dtype=bool)

    # Randomly select a 3x3 patch to add to the center of the synthesized image
    center_i = (synthesized_img.shape[0] // 2)
    center_j = (synthesized_img.shape[1] // 2)
    row = random.randint(0, example_img.shape[0] - patch_size)
    col = random.randint(0, example_img.shape[1] - patch_size)
    synthesized_img[center_i-diff:center_i+diff+1, center_j-diff:center_j+diff+1] = example_img[row:row+patch_size, col:col+patch_size,:]/255.0

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

def get_target_patch(synthesized_img:np.ndarray, potential_map:np.ndarray, i:int, j:int, kernel_size:int, diff:int) -> Tuple[np.ndarray, np.ndarray]:
    # Create a zero matrix to store the target patch
    target_patch = np.zeros((kernel_size, kernel_size, synthesized_img.shape[-1]))
    patch_mask = np.zeros((kernel_size, kernel_size))

    start_i, start_j = max(i-diff, 0), max(j-diff, 0)
    end_i, end_j = min(i+diff+1, synthesized_img.shape[0]), min(j+diff+1, synthesized_img.shape[1])

    # Copy the patch from the synthesized image to the target patch
    target_patch[start_i-i+diff:end_i-i+diff, start_j-j+diff:end_j-j+diff] = synthesized_img[start_i:end_i, start_j:end_j]

    # Create mask for known pixels
    patch_mask[start_i-i+diff:end_i-i+diff, start_j-j+diff:end_j-j+diff] = potential_map[start_i:end_i, start_j:end_j]
    patch_mask = (patch_mask == 0)

    return target_patch, np.repeat(patch_mask[:,:,np.newaxis], synthesized_img.shape[-1], axis=2)
        

def find_best_matching_patch(example_patches:np.ndarray, target_patch:np.ndarray, patch_mask:np.ndarray, gaussian_kernel:np.ndarray, threshold:float = 0.8, attenuation_factor:float=80) -> np.ndarray:
    # Calculate the squared difference between the target patch and all example patches
    patches = np.copy(example_patches)
    patches -= target_patch # In place operations is faster than a single line operation
    patches *= patch_mask
    #patches **= 2
    #patches *= gaussian_kernel  # MIGHT NEED TO MOVE THIS UP ONE LINE, BEFORE SQUARING
    patches *= gaussian_kernel
    patches **= 2
    squared_diffs = np.sum(patches, axis=(1, 2, 3))

    # Convert the squared differences to probabilities
    prob = 1 - squared_diffs / np.max(squared_diffs)
    prob *= (prob > threshold*np.max(prob))  # thresholding the probabilities
    prob = prob**attenuation_factor  # attenuate the probabilities
    prob /= np.sum(prob)  # normalize the probabilities
    
    # Find the index of the best matching patch
    best_matching_patch_index = np.random.choice(np.arange(example_patches.shape[0]), p=prob)
    
    return example_patches[best_matching_patch_index]

def synthesize_texture(example_img:np.ndarray, synth_size:Tuple[int,int], kernel_size:int, sigma:float=3, patch_size:int=3, threshold:float=0.8, attenuation_factor:float=80) -> np.ndarray:
    # Initialize the synthesized image and potential map
    synthesized_img, potential_map = initialize_synthesized_image(example_img, synth_size, patch_size=patch_size)  # get started with a random patch and potential map
    
    # Get patches for the example image
    example_patches = get_patches(example_img, kernel_size) # get all the patches from the example image
    diff = kernel_size // 2

    # Create a Gaussian kernel 
    gaussian_kernel = np.repeat(create_gaussian_kernel(kernel_size, sigma)[:,:,np.newaxis], example_img.shape[-1], axis=2)  # create a Gaussian kernel

    # Get the number of non-zero pixels in the potential map
    n_no_pixels = np.where(potential_map != 0)[0].shape[0]
    
    # Iterate over the pixels in the synthesized image
    perc = 0.1
    #print(f"Processing pixel 0 of {n_no_pixels}")
    for n_pixel in range(n_no_pixels):
        pe = n_pixel / n_no_pixels
        os.system('cls' if os.name == 'nt' else 'clear')
        print(f"Progress: {pe*100:.4f}%")
        if pe > perc:
            #print(f"Processing pixel {n_pixel} of {n_no_pixels}")
            break
            # plt.imshow(synthesized_img)
            # plt.axis('off')
            # plt.savefig(f"img7_{perc*100:.1f}.png")
            # print(f"Processing pixel {n_pixel} of {n_no_pixels}")
            # perc += 0.1

        # Find the pixel coordinates with the highest potential
        i, j = np.unravel_index(np.argmax(potential_map), potential_map.shape)  # get the pixel with the highest potential
        
        # Get the target patch from the synthesized image
        target_patch, patch_mask = get_target_patch(synthesized_img, potential_map, i, j, kernel_size, diff)  # get the target patch
        # target_patch = synthesized_img[i-diff:i+diff+1, j-diff:j+diff+1, :]
        
        # Find the best matching patch from the example image
        best_matching_patch = find_best_matching_patch(example_patches, target_patch, patch_mask, gaussian_kernel, threshold=threshold, attenuation_factor=attenuation_factor)    # BROKEN, NEEDS TO BE FIXED
                
        # Replace the pixel with the center pixel of the best matching patch
        synthesized_img[i, j] = np.copy(best_matching_patch[diff, diff])
        
        # Update the potential map
        potential_map[i, j] = 0
        
    return synthesized_img

if __name__ == '__main__':
    # Get a list of all image file paths in the "Texture Patches" folder
    image_paths = glob.glob("Texture Patches/*.png")

    SYNTH_SIZE = (530, 530)
    THRESHOLDS = [0.8, 0.6]
    ATTENUATION_FACTORS = [80, 10]
    PATCH_SIZES = [3, 15]
    KERNEL_SIZES = [9, 11, 15]
    SIGMAS = [1, 3, 9]

    image_path = image_paths[-3]
    print(image_path)
    image_array = np.array(Image.open(image_path))

    for threshold in THRESHOLDS:
        for attenuation_factor in ATTENUATION_FACTORS:
            for patch_size in PATCH_SIZES:
                for kernel_size in KERNEL_SIZES:
                    for sigma in SIGMAS:
                        start = time.perf_counter()
                        synthesized_img = synthesize_texture(image_array, SYNTH_SIZE, kernel_size, sigma, patch_size, threshold, attenuation_factor)
                        # Save the synthesized image using matplotlib
                        save_path = f"Synthesized Images\img7_{threshold}_{attenuation_factor}_{patch_size}_{kernel_size}_{sigma}.png"
                        plt.imsave(save_path, synthesized_img)
                        print(f"Synthesized image saved at {save_path}")
                        print(f"\tTime taken: {time.perf_counter() - start} seconds")

