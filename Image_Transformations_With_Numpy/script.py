import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# Improved helper function to display images with additional options
def show_image(image, name_identifier, cmap="gray", ax=None):
    if ax is None:
        plt.figure(figsize=(5, 5))
        ax = plt.gca()
    
    img_plot = ax.imshow(image, cmap=cmap)
    ax.set_title(name_identifier)
    ax.axis('off')  # Remove axes for cleaner visualization
    
    return img_plot

# Function to display multiple images in a grid
def show_multiple_images(images, titles, cmaps=None, figsize=(12, 8)):
    n = len(images)
    if cmaps is None:
        cmaps = ["gray"] * n
    
    # Determine number of rows and columns for the grid
    cols = min(3, n)
    rows = (n + cols - 1) // cols
    
    # Create figure and grid
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(rows, cols)
    
    # Add each image to the grid
    for i in range(n):
        ax = fig.add_subplot(gs[i])
        show_image(images[i], titles[i], cmaps[i], ax)
    
    plt.tight_layout()
    plt.show()

# 1. Create heart image
heart_img = np.array([
    [255, 0, 0, 255, 0, 0, 255],
    [0, 127.5, 127.5, 0, 127.5, 127.5, 0],
    [0, 127.5, 127.5, 127.5, 127.5, 127.5, 0],
    [0, 127.5, 127.5, 127.5, 127.5, 127.5, 0],
    [255, 0, 127.5, 127.5, 127.5, 0, 255],
    [255, 255, 0, 127.5, 0, 255, 255],
    [255, 255, 255, 0, 255, 255, 255]
])

# 2. Invert the colors of the heart image
inverted_heart_img = 255 - heart_img

# 3. Rotate the heart image
rotated_heart_img = heart_img.T

# 4. Generate a random image
np.random.seed(42)  # For reproducibility
random_img = np.random.randint(0, 255, (7, 7))

# 5. Solve the linear system safely
# Let's create a non-singular matrix for the example
random_img_nonsingular = np.random.randint(1, 255, (7, 7))
# Ensure the matrix is not singular by adding an identity matrix
random_img_nonsingular = random_img_nonsingular + np.eye(7) * 10

try:
    x = np.linalg.solve(random_img_nonsingular, heart_img)
    solved_heart_img = random_img_nonsingular @ x
    solving_successful = True
except np.linalg.LinAlgError as e:
    print(f"Error solving linear system: {e}")
    solving_successful = False

# 6. Create a cross image
cross_img = np.zeros((5, 5))
cross_img[2, :] = 255  # Middle row
cross_img[:, 2] = 255  # Middle column

# 7. Permutation of the cross rows
permutated_cross_img = np.zeros((5, 5))
permutated_cross_img[1, :] = cross_img[1, :]
permutated_cross_img[3:, :] = cross_img[3:, :]
permutated_cross_img[0, :] = cross_img[2, :]
permutated_cross_img[2, :] = cross_img[0, :]

# 8. Complex image with gradients and RGB channels
height, width = 300, 500
x = np.linspace(0, 1, width)
y = np.linspace(0, 1, height)
X, Y = np.meshgrid(x, y)

# Color channels
cx, cy = width // 2, height // 2
radius = np.sqrt((np.arange(height)[:, None] - cy)**2 + (np.arange(width) - cx)**2)

# Red channel (horizontal gradient)
R = X
# Green channel (vertical gradient)
G = Y
# Blue channel (circular pattern)
B = np.sin(radius / 10) * 0.5 + 0.5

# Add controlled noise
np.random.seed(0)  # For reproducibility
noise = np.random.rand(height, width) * 0.1
R = np.clip(R + noise, 0, 1)
G = np.clip(G + noise, 0, 1)
B = np.clip(B + noise, 0, 1)

# Stack the three channels
complex_img = np.stack([R, G, B], axis=2)

# Visualize the results in logical groups
print("==== Visualizing Results ====")

# Group 1: Heart images and their transformations
show_multiple_images(
    [heart_img, inverted_heart_img, rotated_heart_img],
    ["Heart Image", "Inverted Heart", "Rotated Heart"],
    ["pink", "cool", "pink"]
)

# Group 2: Linear system (if successful)
if solving_successful:
    show_multiple_images(
        [random_img_nonsingular, heart_img, solved_heart_img],
        ["Non-Singular Matrix", "Target Image", "Solved Image"],
        ["viridis", "pink", "pink"]
    )

# Group 3: Cross patterns with different color maps
show_multiple_images(
    [cross_img, permutated_cross_img, cross_img, cross_img],
    ["Original Cross", "Permutated Cross", "Cross (Summer)", "Cross (Cool)"],
    ["gray", "gray", "summer", "cool"]
)

# Group 4: Complex RGB image
plt.figure(figsize=(10, 6))
plt.imshow(complex_img)
plt.title("Complex Image with RGB Gradient")
plt.axis('off')
plt.tight_layout()
plt.show()

# Display the separate channels of the complex image
show_multiple_images(
    [R, G, B],
    ["Red Channel", "Green Channel", "Blue Channel"],
    ["Reds", "Greens", "Blues"]
)

print("==== Image processing completed ====")