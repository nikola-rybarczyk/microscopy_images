import numpy as np
import matplotlib.pyplot as plt
import sys
import json
from numpy.typing import NDArray
from typing import Dict, Any


def load_config(config_file: str = 'config.json') -> Dict[str, Any]:
    try:
        with open(config_file, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error reading config file: {e}")
        sys.exit(1)


def process_image_with_shift(image_matrix: NDArray, shift: int, config: Dict[str, Any]) -> NDArray:
    rows, cols = image_matrix.shape

    # Copy original matrix
    processed_matrix = image_matrix.copy()

    # Only shift odd rows (1, 3, 5, ...)
    for i in range(rows):
        if i % 2 == 1:  # Odd rows
            processed_matrix[i, shift:] = image_matrix[i, :-shift]  # Shift right
    # Then cut the area if specified
    area = config['input'].get('area')
    if area:
        min_idx = int(area['min'] * (image_matrix.shape[0] / config['display']['extent'][1]))
        max_idx = int(area['max'] * (image_matrix.shape[0] / config['display']['extent'][1]))
        processed_matrix = processed_matrix[min_idx:max_idx, min_idx:max_idx]

    return processed_matrix


def read_and_process_images(filename: str, config: Dict[str, Any]) -> None:
    try:
        # Read the full matrix
        image_matrix = np.loadtxt(filename)

        # Process and save 1px left shift
        processed_matrix = process_image_with_shift(image_matrix, 1, config)
        save_image(processed_matrix, config)

    except Exception as e:
        print(f"Error processing images: {e}")


def save_image(image_matrix: NDArray, config: Dict[str, Any]) -> None:
    # Create figure
    plt.figure(figsize=tuple(config['output']['figure_size']))

    # Flip and rotate
    image_matrix = np.flipud(image_matrix)
    image_matrix = np.fliplr(image_matrix)
    image_matrix = np.rot90(image_matrix, k=-1)

    # Calculate extent based on area if specified
    area = config['input'].get('area')
    if area:
        extent = [0, area['max'] - area['min'], 0, area['max'] - area['min']]
    else:
        extent = config['display']['extent'].copy()

    # Adjust extent based on shift
    extent[1] = extent[1] * (image_matrix.shape[1] / (image_matrix.shape[1] + 1))

    # Create the image
    plt.imshow(image_matrix,
               cmap=config['display']['cmap'],
               extent=extent)
    plt.colorbar()

    # Add labels
    plt.xlabel(config['display']['xlabel'])
    plt.ylabel(config['display']['ylabel'])

    # Save and close
    plt.savefig(config['output']['filename'])
    plt.close()
    print(f"Saved image as {config['output']['filename']}")


if __name__ == "__main__":
    config = load_config()
    read_and_process_images(config['input']['filename'], config)
