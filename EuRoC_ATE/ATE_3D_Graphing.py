import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Read data from the text file
file_path = 'C:\\Users\\joshu\\Desktop\\EuRoC_ATE\\ATE_data_offset\\MH_02.txt'  # Replace with the path to your text file
data = []

with open(file_path, 'r') as file:
    for line in file:
        values = [float(x) for x in line.split()]
        data.append(values)

# Separate data into columns
time, x_pos, y_pos, z_pos = zip(*data)

# Create a 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot the data
ax.scatter(x_pos, y_pos, z_pos, c=time, cmap='viridis', marker='o')

# Set labels
ax.set_xlabel('X Position')
ax.set_ylabel('Y Position')
ax.set_zlabel('Z Position')
ax.set_title('3D Scatter Plot of ATE Position over Time')

# Show the plot
plt.show()



    
    