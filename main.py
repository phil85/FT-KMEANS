import matplotlib.pyplot as plt
from fault_tolerant_kmeans import fault_tolerant_kmeans
from sklearn.datasets import make_blobs

# Generate locations of illustrative example
locations, _ = make_blobs(n_samples=50, n_features=2, centers=2, random_state=24, center_box=(0, 15))

# Define parameters
n_facilities = 3
n_assignments_per_client = 2

# Apply FT-KMEANS algorithm
locations_of_facilities, assignments, total_assignment_costs = fault_tolerant_kmeans(
    client_locations=locations,
    potential_locations=locations,
    n_facilities=n_facilities,
    n_assignments_per_client=n_assignments_per_client,
    random_state=0)

# Plot solution
colors = ['blue', 'lightblue']
plt.figure(figsize=(8, 8), dpi=200)
plt.gca().set_aspect('equal')
plt.xticks([])
plt.yticks([])
plt.scatter(locations[:, 0], locations[:, 1], color='gray', marker='o', facecolor='white', zorder=1)
plt.scatter(locations_of_facilities[:, 0], locations_of_facilities[:, 1], color='red', marker='x', s=100, zorder=2)
for i in range(locations.shape[0]):
    for j in range(n_assignments_per_client):
        plt.plot([locations[i, 0], locations_of_facilities[assignments[i, j], 0]],
                 [locations[i, 1], locations_of_facilities[assignments[i, j], 1]],
                 color=colors[j], alpha=0.5, zorder=-j)
plt.title('Solution with total assignment costs of {:.2f}'.format(total_assignment_costs))
plt.show()
