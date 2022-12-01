# FT-KMEANS

A fast algorithm for the fault-tolerant k-median problem

## Dependencies

FT-KMEANS depends on:
* matplotlib=3.5.2
* scikit-learn=1.1.1
* numpy=1.22.4

## Installation

1) Clone this repository (git clone https://github.com/phil85/FT-KMEANS.git)

## Usage

The main.py file contains code that applies the FT-KMEANS algorithm to an illustrative example.

```python
locations_of_facilities, assignments, total_assignment_costs = fault_tolerant_kmeans(
    client_locations=locations,
    potential_locations=locations,
    n_facilities=n_facilities,
    n_assignments_per_client=n_assignments_per_client,
    random_state=0)
```

## Reference

Please cite the following paper if you use this algorithm.

**Baumann, P.** (2022): FT-KMEANS: A fast algorithm for fault-tolerant facility location. submitted 2022

Bibtex:
```
@inproceedings{baumann2022ft-kmeans,
	author={Philipp Baumann},
	booktitle={Proceedings of the 2022 IEEE International Conference on Industrial Engineering and Engineering Management},
	title={FT-KMEANS: A fast algorithm for fault-tolerant facility location},
	year={2022},
	pages={to appear},
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details


