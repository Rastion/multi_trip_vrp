from qubots.base_problem import BaseProblem
import math, random
import os

PENALTY = 1e9

def read_elem(filename):

    # Resolve relative path with respect to this module’s directory.
    if not os.path.isabs(filename):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        filename = os.path.join(base_dir, filename)
            
    with open(filename) as f:
        return [str(elem) for elem in f.read().split()]

def compute_distance_matrix(depots_x, depots_y, customers_x, customers_y, nb_depot_copies):
    nb_customers = len(customers_x)
    nb_depots = len(depots_x)
    nb_total_locations = nb_customers + nb_depots * nb_depot_copies
    dist_matrix = [[0 for _ in range(nb_total_locations)] for _ in range(nb_total_locations)]
    # Compute customer-to-customer distances.
    for i in range(nb_customers):
        dist_matrix[i][i] = 0
        for j in range(i, nb_customers):
            d = compute_dist(customers_x[i], customers_x[j], customers_y[i], customers_y[j])
            dist_matrix[i][j] = d
            dist_matrix[j][i] = d
        # Compute distances from customer i to each depot copy.
        for d in range(nb_depots):
            d_val = compute_dist(customers_x[i], depots_x[d], customers_y[i], depots_y[d])
            for c in range(nb_depot_copies):
                j = nb_customers + d * nb_depot_copies + c
                dist_matrix[i][j] = d_val
                dist_matrix[j][i] = d_val
    # For depot copies, we set inter-depot distances very high.
    for i in range(nb_customers, nb_total_locations):
        for j in range(nb_customers, nb_total_locations):
            dist_matrix[i][j] = 100000
    return dist_matrix

def compute_dist(xi, xj, yi, yj):
    exact = math.sqrt((xi - xj)**2 + (yi - yj)**2)
    return int(math.floor(exact + 0.5))

def read_input_multi_trip_vrp_dat(filename):
    file_it = iter(read_elem(filename))
    nb_customers = int(next(file_it))
    nb_depots = int(next(file_it))
    # Read depot coordinates.
    depots_x = [int(next(file_it)) for _ in range(nb_depots)]
    depots_y = [int(next(file_it)) for _ in range(nb_depots)]
    # Read customer coordinates.
    customers_x = [int(next(file_it)) for _ in range(nb_customers)]
    customers_y = [int(next(file_it)) for _ in range(nb_customers)]
    # Read truck capacity (divide by 2 as specified).
    truck_capacity = int(next(file_it)) // 2
    # Skip depot capacity infos.
    for _ in range(nb_depots):
        next(file_it)
    # Read customer demands.
    demands_data = [int(next(file_it)) for _ in range(nb_customers)]
    # Set constant: number of depot copies.
    nb_depot_copies = 20
    nb_total_locations = nb_customers + nb_depots * nb_depot_copies
    # Maximum travel distance allowed.
    max_dist = 400
    # Number of trucks is fixed (e.g., 3).
    nb_trucks = 3
    dist_matrix_data = compute_distance_matrix(depots_x, depots_y, customers_x, customers_y, nb_depot_copies)
    return nb_customers, nb_trucks, truck_capacity, dist_matrix_data, nb_depots, nb_depot_copies, nb_total_locations, demands_data, max_dist

def read_input_multi_trip_vrp(filename):
    if filename.endswith(".dat"):
        return read_input_multi_trip_vrp_dat(filename)
    else:
        raise Exception("Unknown file format")

class MultiTripVRPProblem(BaseProblem):
    """
    Multi Trip Capacitated Vehicle Routing Problem for Qubots.
    
    A fleet of trucks (with uniform capacity and limited travel distance) must serve a set of customers.
    Trucks start and end at a common depot (represented by a fixed depot location with index nb_customers in the distance matrix).
    To allow depots to be visited multiple times, each depot is replicated a fixed number of times (nb_depot_copies).
    The decision variable is a list of visit orders (one per truck, plus an extra fictive truck to collect unused depot copies).
    The partition constraint ensures that every customer (indices 0..nb_customers-1) is visited exactly once.
    For each real truck, the route distance is computed as the sum of:
      - The distance from the depot (index nb_customers) to the first customer,
      - The distances between consecutive customer visits,
      - The distance from the last customer back to the depot.
    In addition, the truck’s total delivered quantity (sum of customer demands) must not exceed its capacity,
    and its route distance must not exceed max_dist.
    
    The objective is to minimize the total distance traveled by all trucks (each route cost equals a fixed opening route cost plus its travel distance).
    """
    def __init__(self, instance_file: str, **kwargs):
        (self.nb_customers, self.nb_trucks, self.truck_capacity, self.dist_matrix,
         self.nb_depots, self.nb_depot_copies, self.nb_total_locations,
         self.demands, self.max_dist) = read_input_multi_trip_vrp(instance_file)
        # For this problem, a fixed route opening cost is assumed (could be provided in instance; here we use 0 for simplicity)
        self.opening_route_cost = 0

    def evaluate_solution(self, solution) -> int:
        """
        Expects:
          solution: a dictionary with key "visit_orders" mapping to a list of length (nb_trucks+1).
                    The first nb_trucks lists are the routes for the real trucks.
                    Each route is a list of integers (indices in 0..nb_total_locations-1).
                    The extra (fictive) route should contain no customer (i.e. indices < nb_customers).
        Returns:
          The total distance traveled (sum over used trucks of (opening_route_cost + route distance)).
          If any truck exceeds its capacity or its route distance exceeds max_dist,
          or if the union of routes (for trucks 0..nb_trucks-1) does not equal {0,...,nb_customers-1},
          then returns a high penalty.
        """
        # Check that solution is a dict with key "visit_orders" of length nb_trucks+1.
        if not isinstance(solution, dict) or "visit_orders" not in solution:
            return PENALTY
        visit_orders = solution["visit_orders"]
        if not isinstance(visit_orders, list) or len(visit_orders) != self.nb_trucks + 1:
            return PENALTY
        
        # Collect all customer indices from real trucks.
        assigned = []
        for k in range(self.nb_trucks):
            route = visit_orders[k]
            if not isinstance(route, list):
                return PENALTY
            assigned.extend(route)
        if sorted(assigned) != list(range(self.nb_customers)):
            return PENALTY

        total_distance = 0
        # For each real truck, compute route cost.
        for k in range(self.nb_trucks):
            route = visit_orders[k]
            if len(route) == 0:
                continue  # unused truck contributes 0 cost.
            # Compute total demand for this route.
            route_demand = sum(self.demands[i] for i in route if i < self.nb_customers)
            if route_demand > self.truck_capacity:
                return PENALTY
            # Compute route travel distance.
            # We assume that the depot is represented by the fixed index nb_customers.
            depot_index = self.nb_customers
            # Check that route contains only customer indices (if a depot copy appears, ignore it)
            filtered = [i for i in route if i < self.nb_customers]
            if len(filtered) == 0:
                continue
            r_dist = self.dist_matrix[depot_index][filtered[0]]
            for i in range(1, len(filtered)):
                r_dist += self.dist_matrix[filtered[i-1]][filtered[i]]
            r_dist += self.dist_matrix[filtered[-1]][depot_index]
            if r_dist > self.max_dist:
                return PENALTY
            total_distance += self.opening_route_cost + r_dist

        return total_distance

    def random_solution(self):
        """
        Generates a random candidate solution.
        
        Randomly assigns each customer (indices 0..nb_customers-1) to one of the real trucks (0..nb_trucks-1),
        then shuffles the order within each truck. The extra (fictive) truck is assigned an empty list.
        """
        routes = [[] for _ in range(self.nb_trucks)]
        for cust in range(self.nb_customers):
            r = random.randrange(self.nb_trucks)
            routes[r].append(cust)
        for r in range(self.nb_trucks):
            random.shuffle(routes[r])
        # Fictive truck (index nb_trucks) gets an empty list.
        routes.append([])
        return {"visit_orders": routes}
