from sklearn.neighbors import KDTree
import numpy as np
import time


def update_centers(client_locations, centers, n_facilities, assignments):
    for i in range(n_facilities):
        centers[i] = client_locations[np.where(assignments == i)[0], :].mean(axis=0)
    return centers


def assign_objects(client_locations, centers, n_assignments):
    kd_tree = KDTree(centers)
    _, labels = kd_tree.query(client_locations, k=n_assignments)
    return labels


def get_total_distance(client_locations, centers, labels):
    kd_tree = KDTree(centers)
    dist, _ = kd_tree.query(client_locations, k=labels.shape[1])
    return dist.sum()


def get_locations_assignments_and_total_assignment_costs(client_locations, potential_locations, centers, n_facilities,
                                                         n_assignments_per_client):
    location_ids = []
    kd_tree = KDTree(potential_locations)
    _, closest_potential_location = kd_tree.query(centers, k=n_facilities)
    for i in range(n_facilities):
        j = 0
        while j < n_facilities:
            if closest_potential_location[i, j] not in location_ids:
                location_ids.append(closest_potential_location[i, j])
                break
            else:
                j += 1
    location_ids = np.array(location_ids)
    new_centers = potential_locations[location_ids, :]
    kd_tree = KDTree(new_centers)
    distances, assignments = kd_tree.query(client_locations, k=n_assignments_per_client)
    return new_centers, assignments, distances.sum()


def fault_tolerant_kmeans(client_locations, potential_locations=None, n_facilities=3, n_assignments_per_client=2,
                          random_state=None, time_limit=1e7, max_iter=100):

    # Start stopwatch
    tic = time.perf_counter()

    # Choose initial cluster centers randomly
    np.random.seed(random_state)
    center_ids = np.random.choice(np.arange(client_locations.shape[0]), size=n_facilities, replace=False)
    centers = client_locations[center_ids, :]

    # Assign clients
    assignments = assign_objects(client_locations, centers, n_assignments_per_client)

    # Initialize best assignments
    best_assignments = assignments

    # Update centers
    centers = update_centers(client_locations, centers, n_facilities, assignments)
    best_centers = centers.copy()

    # Compute total assignment costs
    best_total_assignment_costs = get_total_distance(client_locations, centers, assignments)

    n_iter = 0
    elapsed_time = time.perf_counter() - tic
    while (n_iter < max_iter) and (elapsed_time < time_limit):

        # Assign clients
        assignments = assign_objects(client_locations, centers, n_assignments_per_client)

        # Update centers
        centers = update_centers(client_locations, centers, n_facilities, assignments)

        # Compute total assignment costs
        total_assignment_costs = get_total_distance(client_locations, centers, assignments)

        # Check stopping criterion
        if total_assignment_costs >= best_total_assignment_costs:
            break
        else:
            # Update the best assignments and the best total assignment costs
            best_assignments = assignments
            best_total_assignment_costs = total_assignment_costs
            best_centers = centers.copy()

        # Increase iteration counter
        n_iter += 1

        # Update elapsed time
        elapsed_time = time.perf_counter() - tic

    # Select final locations among potential locations
    if potential_locations is not None:
        best_centers, best_assignments, best_total_assignment_costs = \
            get_locations_assignments_and_total_assignment_costs(client_locations,
                                                                 potential_locations,
                                                                 best_centers,
                                                                 n_facilities,
                                                                 n_assignments_per_client)

    return best_centers, best_assignments, best_total_assignment_costs
