import gurobipy as gp
from gurobipy import GRB, quicksum
import numpy as np

# Set random seed
np.random.seed(11)


def small_instance_parameters(number_orders = 5, number_depots = 2, number_vehicles = 3):
    # Randomly generate orders' demands in [1, 5]
    orders_demands = np.random.randint(1, 5, number_orders)

    # Defining indices for depots, pickups and deliveries nodes
    depot_inices = np.array(range(number_depots))
    pickups_indices = np.array(range(number_depots, number_depots + number_orders))
    deliveries_indices = np.array(range(number_depots + number_orders, 2 * number_orders + number_depots))
    # Calculating total number of nodes
    number_nodes = number_depots + 2 * number_orders
    # Defining the load to be picked up at each pickup node
    pickups_loads = orders_demands
    # Defining the load to be delivered at each delivery node
    deliveries_loads = orders_demands

    # Randomly set initial depot for each vehicle
    initial_depots = np.random.randint(0, number_depots, number_vehicles)
    # Randomly generate vehicle's speed in [1, 5]
    speed = np.random.uniform(1, 5, number_vehicles)
    # Randomly generate vehicle's cost per distance in [1, 5]
    cost_per_distance = np.random.randint(1, 5, number_vehicles)
    # Randomly generate vehicle's vehicles_capacity in [15, 25]
    vehicles_capacity = np.random.randint(10, 15, number_vehicles)
    # Randomly generate vehicle's fixed cost in [50, 100]
    vehicles_fixed_cost = np.random.randint(50, 100, number_vehicles)

    # Randomly generate coordinates in [-20, 20]
    coordinates = np.random.randint(-20, 20, (number_nodes, 2))
    # Defining the distance matrix
    times_matrix = np.zeros((number_nodes, number_nodes, number_vehicles))
    # Defining the cost matrix
    cost_matrix = np.zeros((number_nodes, number_nodes, number_vehicles))

    for vehicle_index in range(number_vehicles):
        for first_node in range(number_nodes):
            for scnd_node in range(number_nodes):
                # Calculate Euclidean distance between nodes
                distance = np.linalg.norm(coordinates[first_node, :] - coordinates[scnd_node, :])
                times_matrix[first_node, scnd_node, vehicle_index] = np.round(distance / speed[vehicle_index])
                cost_matrix[first_node, scnd_node, vehicle_index] = np.round(
                    distance * cost_per_distance[vehicle_index])

    # Create parameters' dictionary
    parameters = {
        'number_orders': number_orders,
        'number_vehicles': number_vehicles,
        'depot_inices': depot_inices,
        'pickups_indices': pickups_indices,
        'deliveries_indices': deliveries_indices,
        'number_nodes': number_nodes,
        'pickups_loads': pickups_loads,
        'deliveries_loads': deliveries_loads,
        'initial_depots': initial_depots,
        'vehicles_capacity': vehicles_capacity,
        'vehicles_fixed_cost': vehicles_fixed_cost,
        'times_matrix': times_matrix,
        'cost_matrix': cost_matrix
    }

    return parameters


def build_model(parameters):
    # Extracting parameters from parameters' dictionary
    number_orders = parameters['number_orders']
    number_vehicles = parameters['number_vehicles']
    pickups_indices = parameters['pickups_indices']
    deliveries_indices = parameters['deliveries_indices']
    number_nodes = parameters['number_nodes']
    pickups_loads = parameters['pickups_loads']
    deliveries_loads = parameters['deliveries_loads']
    initial_depots = parameters['initial_depots']
    vehicles_capacity = parameters['vehicles_capacity']
    vehicles_fixed_cost = parameters['vehicles_fixed_cost']
    times_matrix = parameters['times_matrix']
    cost_matrix = parameters['cost_matrix']

    # Defining big M
    big_m = 1000
    # Defining the set of pickup and delivery nodes (N)
    pickup_and_deliveries = np.concatenate((pickups_indices, deliveries_indices))

    # Create the optimization model
    model = gp.Model('VRPPDP')

    # Create the decision variables
    x = model.addVars(range(number_nodes), range(number_nodes), range(number_vehicles), vtype = GRB.BINARY, name = 'x')
    y = model.addVars(range(number_vehicles), vtype = GRB.BINARY, name = 'y')
    arrival_time = model.addVars(range(number_nodes), range(number_vehicles), vtype = GRB.CONTINUOUS, lb = 0,
                                 name = 'T')
    load_amount = model.addVars(range(number_nodes), range(number_vehicles), vtype = GRB.CONTINUOUS, lb = 0,
                                name = 'L')

    # Set the objective function
    model.setObjective(quicksum(vehicles_fixed_cost[vehicle_index] * y[vehicle_index]
                                for vehicle_index in range(number_vehicles)) +
                       quicksum(cost_matrix[i, j, vehicle_index] * x[i, j, vehicle_index]
                                for i in range(number_nodes)
                                for j in range(number_nodes)
                                for vehicle_index in range(number_vehicles)),
                       GRB.MINIMIZE)

    # Set the constraints

    # Constraint (2): All vehicles start their routes at the origin depot
    model.addConstrs((quicksum(x[initial_depots[k], p, k]
                               for p in pickups_indices) == y[k]
                      for k in range(number_vehicles)),
                     name = 'constraint_2')

    # Constraint (3): Forbid vehicles to start from a different depot node than the designed depot
    model.addConstrs((quicksum(x[d, j, k]
                               for d in initial_depots
                               if d != initial_depots[k]
                               for j in range(number_nodes)) == 0
                      for k in range(number_vehicles)), name = 'constraint_3')

    # Constraint (4): A vehicle must be labeled as used to make a route
    model.addConstrs((big_m * y[k] >= quicksum(x[i, j, k]
                                               for i in range(number_nodes)
                                               for j in range(number_nodes))
                      for k in range(number_vehicles)),
                     name = 'constraint_4')

    # Constraint (5): No vehicle can traverse more than one edge at a time
    model.addConstrs((quicksum(x[i, j, k] for j in range(number_nodes)) <= 1
                      for i in range(number_nodes)
                      for k in range(number_vehicles)),
                     name = 'constraint_5')

    # Constraint (6): Vehicles' flow balance
    model.addConstrs((quicksum(x[j, i, k] for j in range(number_nodes) if j != i) -
                      quicksum(x[i, j, k] for j in range(number_nodes) if j != i) == 0
                      for i in range(number_nodes)
                      for k in range(number_vehicles)),
                     name = 'constraint_6')

    # Constraint (7): All pickups and customers must be visited exactly once
    model.addConstrs((quicksum(quicksum(x[i, j, vehicle_index] for i in range(number_nodes) if i != j)
                               for vehicle_index in range(number_vehicles)) == 1
                      for j in pickup_and_deliveries),
                     name = 'constraint_7')

    # Constraint (8): The same vehicle that picks up an order must deliver it
    model.addConstrs((quicksum(x[deliveries_indices[o], j, k]
                               for j in range(number_nodes) if j != deliveries_indices[o]) ==
                      quicksum(x[i, pickups_indices[o], k]
                               for i in range(number_nodes) if i != pickups_indices[o])
                      for o in range(number_orders)
                      for k in range(number_vehicles)), name = 'constraint_8')

    # Constraint (9): Arrival time in the routes
    model.addConstrs((arrival_time[i, k] + times_matrix[i, j, k] <= arrival_time[j, k] + big_m * (1 - x[i, j, k])
                      for i in range(number_nodes)
                      for j in pickup_and_deliveries
                      if i != j
                      for k in range(number_vehicles)),
                     name = 'constraint_9')

    # Constraint (10): Pickup before delivery
    model.addConstrs((
        arrival_time[pickups_indices[o], k] + times_matrix[pickups_indices[o], deliveries_indices[o], k] <=
        arrival_time[deliveries_indices[o], k] +
        big_m * (1 - quicksum(x[i, deliveries_indices[o], k] for i in pickup_and_deliveries))
        for o in range(number_orders)
        for k in range(number_vehicles)),
        name = 'constraint_10')

    # Constraint (11) and (12): Load quantity through the route
    model.addConstrs(
        (load_amount[p, k] >= load_amount[i, k] + pickups_loads[o] - big_m * (1 - x[i, p, k])
         for p in pickups_indices
         for i in range(number_nodes)
         if i != p
         for o in range(number_orders)
         for k in range(number_vehicles)),
        name = 'constraint_11')

    model.addConstrs((load_amount[c, k] >= load_amount[i, k] - deliveries_loads[o] - big_m * (1 - x[i, c, k])
                      for c in deliveries_indices
                      for i in range(number_nodes)
                      if i != c
                      for o in range(number_orders)
                      for k in range(number_vehicles)),
                     name = 'constraint_12')

    # Constraint (13): No vehicle exceeds its load vehicles_capacity
    model.addConstrs((load_amount[i, k] <= vehicles_capacity[k]
                      for i in range(number_nodes)
                      for k in range(number_vehicles)),
                     name = 'constraint_13')

    # Write the model
    model.write('VRPPDP.lp')

    # Solve the model
    model.optimize()

    # If the model is infeasible
    if model.status == GRB.INFEASIBLE:
        # Compute an Irreducible Inconsistent Subsystem (IIS)
        model.computeIIS()
        # Write the IIS in a file
        model.write('VRPPDP.ilp')
    elif model.status == GRB.OPTIMAL:
        # Print the route for each vehicle
        for k in range(number_vehicles):
            print('Route for vehicle %g:' % k)
            # If the vehicle is used
            if y[k].x > 0.5:
                # Start from the initial depot
                starting_node = initial_depots[k]
                destination_node = np.where(np.array([x[starting_node, j, k].x for j in range(number_nodes)]) > 0.5)
                # While the destination node is not the initial depot
                while destination_node[0][0] != initial_depots[k]:
                    destination_node = destination_node[0][0]
                    # Print starting node -> destination node
                    print('%g -> %g' % (starting_node, int(destination_node)))
                    # Update the starting node
                    starting_node = destination_node
                    # Update the destination node
                    destination_node = np.where(np.array([x[starting_node, j, k].x for j in range(number_nodes)]) > 0.5)
                # Print the final route
                print('%g -> %g' % (starting_node, initial_depots[k]))


def main():
    # Generate parameters for a small instance
    parameters = small_instance_parameters()
    # Build and optimize the model
    build_model(parameters)


main()
