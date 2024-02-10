import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm

# Number of colors to extract
# num_colors = 2

# Get the turbo colormap
# turbo_map = cm.get_cmap('turbo', num_colors)

# Get the leftmost and rightmost colors
# leftmost_color = turbo_map(0)
# rightmost_color = turbo_map(num_colors - 1)

# print("Leftmost color in turbo colormap:", leftmost_color)
# print("Rightmost color in turbo colormap:", rightmost_color)

# import matplotlib.pyplot as plt

# Grayscale value
# gray_value = 0.5  # This will give a medium gray color

# Plotting example
# plt.plot([0, 1], [0, 1], color=rightmost_color, label=f'Gray value: {gray_value}')
# plt.xlabel('X')
# plt.ylabel('Y')
# plt.title('Gray Color Example')
# plt.legend()
# plt.show()

# import matplotlib.pyplot as plt
# import numpy as np
# import matplotlib.cm as cm

# # Generate some random data
# x = np.random.rand(100)
# y = np.random.rand(100)
# colors = np.random.rand(100)  # Random color values for each point

# # Number of colors to extract
# num_colors = 2

# # Get the turbo colormap
# turbo_map = cm.get_cmap('turbo', num_colors)

# # Get the leftmost and rightmost colors
# leftmost_color = turbo_map(0)
# rightmost_color = turbo_map(num_colors - 1)

# # Plot the scatter plot with leftmost and rightmost colors
# #plt.scatter(x, y, c=colors, cmap='turbo')  # You can use cmap='turbo' directly if you don't need custom colors
# #plt.colorbar(label='Random values')

# plt.scatter(x, y, c=[leftmost_color, rightmost_color], label=['Leftmost', 'Rightmost'])
# plt.colorbar(label='Random values')
# plt.legend()
#plt.show()
import matplotlib.pyplot as plt
import numpy as np

# # Define the arrow properties
# x_start, y_start = 0.2, 0.2  # Starting point of the arrow
# x_end, y_end = 0.8, 0.8  # Ending point of the arrow
# head_width = 0.05  # Width of the arrow head
# head_length = 0.05  # Length of the arrow head

# # Create a figure and axis
# fig, ax = plt.subplots()

# # Plot the arrow with a color gradient
# gradient = np.linspace(0, 1, 100)  # Gradient from gray to red
# for i in range(len(gradient)):
#     color = (gradient[i], gradient[i], gradient[i])  # Gray to red
#     ax.arrow(x_start, y_start, x_end - x_start, y_end - y_start,
#              head_width=head_width, head_length=head_length, fc=color, ec=color)

# # Set the limits of the plot
# ax.set_xlim(0, 1)
# ax.set_ylim(0, 1)

# # Set labels and title
# ax.set_xlabel('X')
# ax.set_ylabel('Y')
# ax.set_title('Arrow with Color Transition from Gray to Red')

# # Show the plot
# plt.show()

import matplotlib.pyplot as plt
import numpy as np

# # Define arrow properties
# x_start, y_start = 0.2, 0.2  # Starting point of the arrow
# x_end, y_end = 0.8, 0.8      # Ending point of the arrow

# # Create a figure and axis
# fig, ax = plt.subplots()

# # Plot the arrow with a color gradient
# gradient = np.linspace(0, 1, 100)  # Gradient from gray to red
# for i in range(len(gradient)):
#     color = (gradient[i], 0, 0)  # Gray to red
#     ax.arrow(x_start, y_start, x_end - x_start, y_end - y_start,
#              head_width=0.05, head_length=0.05, fc=color, ec=color, alpha=0.8)

# # Set the limits of the plot
# ax.set_xlim(0, 1)
# ax.set_ylim(0, 1)

# # Set labels and title
# ax.set_xlabel('X')
# ax.set_ylabel('Y')
# ax.set_title('Arrow with Color Transition from Gray to Red')

# # Show the plot
# plt.show()

import numpy as np
import matplotlib.pyplot as plt

# Generate some example data
data = np.random.rand(10, 10)

# Create a figure
fig, ax = plt.subplots()

# Plot the data
heatmap = ax.imshow(data, cmap='viridis')

# Create a colorbar
cbar = plt.colorbar(heatmap)

# Set the color limits of the colorbar
cbar.set_clim(vmin=0, vmax=1)  # Set the range from 0 to 1

# Adjust the extent of the colorbar
cbar.ax.set_aspect(10)  # Change the aspect ratio of the colorbar

plt.show()

