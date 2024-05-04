import numpy as np
import matplotlib.pyplot as plt

def calculate_2dft(input):
    ft = np.fft.ifftshift(input)
    ft = np.fft.fft2(ft)
    return np.fft.fftshift(ft)
def calculate_2dift(input, until=None):
    side_length = input.shape[0]
    centre = int((side_length - 1) / 2)
    if until:  # sort and transform terms with lower frequencies
        if type(until) == float:
            until = int(side_length*centre*until)
        coords_left_half = get_coords_left_half(input)
        coords_left_half = coords_left_half[:until]
        symm_coords = find_symmetric_coordinates(coords_left_half, centre)
        coords = np.concatenate((coords_left_half, symm_coords), axis=0)
        coords = convert_np_positions_to_indices(coords)
        
        target = np.zeros(input.shape, dtype=complex)
        target[coords] = input[coords]
        input = target
        
    ift = np.fft.ifftshift(input)
    ift = np.fft.ifft2(ift)
    ift = np.fft.fftshift(ift)
    return ift.real
def calculate_distance_from_centre(coords, centre):
    # Distance from centre is √(x^2 + y^2)
    return np.sqrt(
        (coords[0] - centre) ** 2 + (coords[1] - centre) ** 2
    )
def find_symmetric_coordinates(coords, centre):
    return 2*centre - coords
    return (centre + (centre - coords[0]),
            centre + (centre - coords[1]))
def display_plots(individual_grating, reconstruction, idx):
    plt.subplot(121)
    plt.imshow(individual_grating)
    plt.axis("off")
    plt.subplot(122)
    plt.imshow(reconstruction)
    plt.axis("off")
    plt.suptitle(f"Terms: {idx}")
    plt.pause(0.01)

def get_coords_left_half(image_array):
    side_length = len(image_array)
    centre = int((side_length - 1) / 2)
    # Get all coordinate pairs in the left half of the array,
    # including the column at the centre of the array (which
    # includes the centre pixel)
    coords_left_half = [
            [x, y] for x in range(side_length) for y in range(centre+1)
        ]
    coords_left_half = np.array(coords_left_half)
    # Sort points based on distance from centre
    coords_left_half = sorted(
        coords_left_half,
        key=lambda x: calculate_distance_from_centre(x, centre)
    )
    return np.array(coords_left_half, dtype=np.int32)

def convert_np_positions_to_indices(indices:np.ndarray):
    indices = list(map(tuple, indices))
    indices = tuple(np.transpose(indices))
    return indices


def inverseFDA(source, target, beta=0.1, distance='approx'):
    # beta is the percentage of replacement of high frequency terms from the target image
    if len(source.shape) == 2:
        grayscale=True
        source, target = source.reshape(source.shape+(-1,)), target.reshape(target.shape+(-1,))
    else:
        grayscale=False
    result = np.zeros_like(source)
    for c in range(source.shape[-1]):
        source_fd, target_fd = calculate_2dft(source[...,c]), calculate_2dft(target[...,c])
        center_w, center_h = (source_fd.shape[1]-1)//2, (source_fd.shape[0]-1)//2
        beta_w = int((1-beta**0.5)*source_fd.shape[1]//2)
        beta_h = int((1-beta**0.5)*source_fd.shape[0]//2)
        beta_w, beta_h = min(beta_w, center_w-1), min(beta_h, center_h-1)
        #print(center_w, center_h, beta_w, beta_h)
        if distance == 'approx':
            coords_to_swap = np.array([[i, j] for j in range(center_w-beta_w, center_w+beta_w) for i in range(center_h-beta_h, center_h+beta_h)]).T
            coords_to_swap = tuple(coords_to_swap)
        # replace with low frequency 
        if coords_to_swap:
            target_fd[coords_to_swap] = source_fd[coords_to_swap]
        result[...,c] = calculate_2dift(target_fd)
    if grayscale:
        result = result.reshape(result.shape[:-1])
    return result
