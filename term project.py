import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.dates as mdates

data = pd.read_excel('/Users/baozi/CIS 635/term project/test2.xlsx')

# Define the grid dimensions
x_min, x_max = 7603950, 7717500
y_min, y_max = 651190, 733990

num_cells_x = int((x_max - x_min) / 600)
num_cells_y = int((y_max - y_min) / 600)

days_per_layer = 150
total_days = (data['occ_date'].max() - data['occ_date'].min()).days
z_max = total_days // days_per_layer

def show_graph ():
    # Plot 2D heatmap for the current layer
    for z in range(z_max):
        fig, ax = plt.subplots(figsize=(10, 6))
        
        im = ax.imshow(count_per_cell[:, :, z], cmap='viridis', extent=[x_min, x_max, y_min, y_max], origin='lower', aspect='auto')

        ax.set_title(f'Layer {z + 1}')
        ax.set_xlabel('X Coordinate')
        ax.set_ylabel('Y Coordinate')
        ax.set_title(f"Time Interval: {layer_start_dates[z].strftime('%Y-%m-%d')} to {layer_end_dates[z].strftime('%Y-%m-%d')}")
        
        # Add colorbar
        fig.colorbar(im, ax=ax, orientation='vertical', pad=0.1)

        plt.show()

def calcualte_PAI(n, N, a, A):
    return (n/N)/(a/A)
def calculate_PEI(PAI,n,N,a,A):
    return PAI/((n/N)/(a/A))

def print_cells():
    for z in range(z_max):
        print(f"Top 100 cells in Layer {z + 1}:")
        for i, (x_idx, y_idx, count) in enumerate(top_cells_per_layer[z], 1):
            print(f"  {i}. Cell ({x_idx + 1}, {y_idx + 1}) - Count: {count}")
        print("\n")

# Convert 'occ_date' column to ordinal format
data['occ_date'] = pd.to_datetime(data['occ_date'], format='%m/%d/%y')
data['occ_date'] = data['occ_date'].apply(mdates.date2num)

# Assign points to the grid
points = data[['x_coordinate', 'y_coordinate', 'occ_date']].values

indices_x = np.digitize(points[:, 0], np.linspace(x_min, x_max, num_cells_x))
indices_y = np.digitize(points[:, 1], np.linspace(y_min, y_max, num_cells_y))

count_per_cell = np.zeros((num_cells_x, num_cells_y, z_max))

for i in range(len(points)):
    x_idx = indices_x[i] - 1
    y_idx = indices_y[i] - 1
    z_idx = int((points[i, 2] - data['occ_date'].min()) / days_per_layer)

    # Ensure z_idx is within bounds
    z_idx = max(0, min(z_idx, z_max - 1))

    count_per_cell[x_idx, y_idx, z_idx] += 1

sum_per_layer = np.sum(count_per_cell, axis=(0, 1))

layer_start_dates = [mdates.num2date(data['occ_date'].min() + days_per_layer * i) for i in range(z_max)]
layer_end_dates = [mdates.num2date(data['occ_date'].min() + days_per_layer * (i + 1)) for i in range(z_max)]
top_cells_per_layer = []

for z in range(z_max):
    # Flatten the 2D array for the current layer
    flat_count_layer = count_per_cell[:, :, z].flatten()
    
    # Get the indices of the top 100 cells
    top_indices = np.argpartition(flat_count_layer, -100)[-100:]
    
    # Convert flattened indices to 2D indices
    top_indices_2d = np.unravel_index(top_indices, count_per_cell[:, :, z].shape)
    
    # Create a list of tuples containing (x_index, y_index, count) for the top cells
    top_cells = [(top_indices_2d[0][i], top_indices_2d[1][i], flat_count_layer[top_indices[i]]) for i in range(len(top_indices))]
    
    # Add the list of top cells for the current layer to the result
    top_cells_per_layer.append(top_cells)


# Print the sum of points in each layer
for z in range(z_max):
    print(f"Layer {z + 1} - Time Interval: {layer_start_dates[z].strftime('%Y-%m-%d')} to {layer_end_dates[z].strftime('%Y-%m-%d')}")

    if z == 0:
        # For the first layer, print the sum of points only
        total_top_count_actual = sum(cell[2] for cell in top_cells_per_layer[z])
        print(f"  Total count for the actual top 100 cells: {total_top_count_actual}")
        print(f"  Sum of Points: {int(sum_per_layer[z])} points\n")
    else:
        # Calculate the sum of counts using indices from the last layer
        total_top_count_last_layer = sum(count_per_cell[cell[0], cell[1], z] for cell in top_cells_per_layer[z - 1])
        
        # Calculate the sum of counts for the actual top 100 cells of the next layer
        total_top_count_actual = sum(cell[2] for cell in top_cells_per_layer[z])

        print(f"  Total count using indices from the last layer: {total_top_count_last_layer}")
        print(f"  Total count for the actual top 100 cells: {total_top_count_actual}")
        print(f"  Sum of Points: {int(sum_per_layer[z])} points")
        pai = calcualte_PAI(total_top_count_last_layer,int(sum_per_layer[z]),100,num_cells_x*num_cells_y)
        pei = calculate_PEI(pai,total_top_count_actual,int(sum_per_layer[z]),100,num_cells_x*num_cells_y)
        print(f"  PEI: {pei}\n")

# print_cells()
# show_graph()

