import torch
import torch.nn.functional as F
import numpy as np

DOORS = [[100.375, 84.4375, 118.625],
         [76, 131.75, 76],
         [122.5625, 81.625, 81.625],
         [131.75, 131.75, 76.0],
         [131.75, 76.0, 76.0],
         [97.5625, 97.5625, 97.5625],
         [76.0, 76.0, 131.75]]

COLORS = {
    "red": np.array([255, 0, 0]),
    "green": np.array([0, 255, 0]),
    "blue": np.array([0, 0, 255]),
    "purple": np.array([112, 39, 195]),
    "yellow": np.array([255, 255, 0]),
    "grey": np.array([100, 100, 100]),
}


def classify_block(block):
    avg_color = block.mean(axis=(0, 1)).tolist()
    if avg_color[0] == avg_color[1] == avg_color[2] == 8.078125:
        return 0  # unseen
    if avg_color[0] == avg_color[1] == avg_color[2] == 146:
        return 1  # wall
    if avg_color[0] == avg_color[1] == avg_color[2] == 81.625:
        return 2  # floor
    if avg_color in DOORS:
        return 3  # door
    if block[4][4].tolist() == COLORS['green'].tolist():
        return 4  # goal
    return 5 #open door 


def process_grid(image,block_size=8): #for reward stuff
    height, width, channels = image.shape
    num_blocks_y = height // block_size
    num_blocks_x = width // block_size
    classification_grid = np.empty((num_blocks_y, num_blocks_x), dtype=int)

    for i in range(num_blocks_y):
        for j in range(num_blocks_x):
            y_start = i * block_size
            y_end = y_start + block_size
            x_start = j * block_size
            x_end = x_start + block_size
            block = image[y_start:y_end, x_start:x_end, :]
            classification_grid[i, j] = classify_block(block)

    classification_grid[6, 3] = 2  # Hardcoded player position for now.
    return classification_grid


def process_image(image, block_size=8):
    height, width, channels = image.shape
    num_blocks_y = height // block_size
    num_blocks_x = width // block_size
    classification_grid = np.empty((num_blocks_y, num_blocks_x), dtype=int)

    for i in range(num_blocks_y):
        for j in range(num_blocks_x):
            y_start = i * block_size
            y_end = y_start + block_size
            x_start = j * block_size
            x_end = x_start + block_size
            block = image[y_start:y_end, x_start:x_end, :]
            classification_grid[i, j] = classify_block(block)

    classification_grid[6, 3] = 2  # Hardcoded player position for now.
    one_hot_grid = F.one_hot(torch.tensor(classification_grid), num_classes=6)
    one_hot_grid = one_hot_grid[:, :, 1:]
    one_hot_grid = one_hot_grid.permute(2, 0, 1).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return one_hot_grid
