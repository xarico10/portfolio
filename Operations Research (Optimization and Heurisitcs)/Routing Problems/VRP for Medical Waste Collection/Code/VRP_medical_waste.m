clear, clc
rng(10)
% Coordinates for 5 demand nodes
demand_nodes = [10, 10;
    20, 15;
    30, 25;
    15, 30;
    5, 20];

% Coordinates for 2 parking nodes
parking_nodes = [5, 5;
    35, 30];

% Coordinates for 1 processing node
processing_nodes = [25, 20; 10, 15];

% Append nodes coordinates in a single matrix
nodes = [demand_nodes; parking_nodes; processing_nodes];
number_nodes = size(nodes, 1);
% Extracting each type of node indices
demand_indices = 1:size(demand_nodes, 1);
parking_indices = demand_indices(end) + 1:demand_indices(end) + ...
    size(parking_nodes, 1);
processing_indices = parking_indices(end) + 1:parking_indices(end) + ...
    size(processing_nodes, 1);
% Speeds for 2 vehicles (units per time step)
vehicles_speed = [3, 4];
number_vehicles = numel(vehicles_speed);
% Starting parking nodes
starting_nodes = [1, 2];

% Distance matrix
distance_matrix = pdist2(nodes, nodes);
% Reasonable final time (t_f)
average_distance = mean(distance_matrix, 'all');
average_speed = mean(vehicles_speed);
time_per_trip = average_distance / average_speed; % Average time per trip
% Assuming each vehicle might need to visit each demand node once
number_of_trips = number_nodes;
final_time = ceil(time_per_trip * number_of_trips);

% Traveling time matrix
traveling_times = zeros(number_nodes, number_nodes, number_vehicles);

for vehicle = 1:number_vehicles
    traveling_times(:, :, vehicle) = round(distance_matrix ./ ...
        vehicles_speed(vehicle));
end

% Base rate of waste generation for each demand node (example values)
alpha = rand(numel(demand_indices), 1);

% Exposure risk factor parameters for each demand node (example values)
beta = [0.1; 0.2; 0.15; 0.25; 0.3];

% Maximum vehicle capacities
vehicles_capacities = randi([20, 30], [number_vehicles, 1]);

% Traveling costs
traveling_costs = randi([1, 3], [number_vehicles, 1]);

%% Optimization Problem
clear, clc

load instance_parameters.mat

x = optimvar('x', number_nodes, number_nodes, number_vehicles, final_time,...
    'Type', 'integer', 'LowerBound', 0, 'UpperBound', 1);

W = optimvar('W', numel(demand_indices), final_time, 'Type', 'continuous',...
    'LowerBound', 0);

l = optimvar('l', number_vehicles, final_time, 'Type','continuous',...
    'LowerBound', 0, 'UpperBound', repmat(...
    vehicles_capacities, [1, final_time]));

WX = optimvar('WX', numel(demand_indices), number_vehicles, final_time,...
    'Type', 'continuous', 'LowerBound', 0);

f1 = sum(traveling_costs .* reshape(...
    sum(sum(repmat(distance_matrix, [1, 1, number_vehicles]) .* ...
    sum(x, 4), 2), 1),[number_vehicles, 1]), 1);

f2 = sum(beta .* sum(W, 2), 1);

objective_function = f1 + f2;

disp('Assigning Constraints')

cons1 = optimconstr(1, number_vehicles);
for vehicle = 1:number_vehicles
    starting_parking_index = parking_indices(starting_nodes(vehicle));
    cons1(vehicle) = sum(...
        x(starting_parking_index, demand_indices, vehicle, 1), 2) == 1;
end
disp('Constraint #1 Assigned')

cons2 = sum(sum(x,2),1) <= 1;
disp('Constraint #2 Assigned')

cons3 = optimconstr(number_nodes, number_nodes, number_vehicles, final_time);
sum_right = optimexpr(number_nodes, number_nodes, number_vehicles, final_time);
for vehicle = 1:number_vehicles
    for i_node = 1:number_nodes
        for j_node = 1:number_nodes
            k_nodes = setdiff(1:number_nodes, j_node);
            if i_node ~= j_node
                traveling_time = traveling_times(i_node, j_node, vehicle);
                t_indices = 1:final_time - traveling_time + 1;
                for t_index = 1:numel(t_indices)
                    t = t_indices(t_index);
                    sum_right(i_node, j_node, vehicle, t) = sum(sum(sum(...
                        x(j_node, k_nodes, vehicle, t + 1:t + traveling_time - 1)....
                        , 4), 2), 1);
                end
                cons3(i_node, j_node, vehicle, t_indices) = ...
                    x(i_node, j_node, vehicle, t_indices) + ...
                    sum_right(i_node, j_node, vehicle, t_indices) <= 1;
                
            end
        end
    end
end
disp('Constraint #3 Assigned')

cons4 = optimconstr(number_nodes, number_nodes, number_vehicles, final_time);
for vehicle = 1:number_vehicles
    for i_node = 1:number_nodes
        for j_node = 1:number_nodes
            if j_node ~= i_node && ~ismember(j_node, parking_indices)
                traveling_time = traveling_times(i_node, j_node, vehicle);
                t_indices = 1:final_time - traveling_time;
                cons4(i_node, j_node, vehicle, t_indices) = ...
                    x(i_node, j_node, vehicle, t_indices)...
                    <= sum(x(j_node, :, vehicle, traveling_time + 1:end), 2);
            end
        end
    end
end
disp('Constraint #4 Assigned')

sum_right = optimexpr(number_nodes, number_vehicles, final_time);
sum_left = optimexpr(number_nodes, number_vehicles, final_time);

for vehicle = 1:number_vehicles
    for j_node = 1:number_nodes
        if ~ismember(j_node, parking_indices)
            k_nodes = setdiff(1:number_nodes, j_node);
            h_indices = k_nodes;
            min_traveling_time = min(traveling_times(h_indices, j_node, vehicle));
            sum_left(j_node, vehicle, :) = ...
                sum(x(j_node, k_nodes, vehicle, :), 2);
            for t = 1:final_time
                h_nodes = find(traveling_times(:, j_node, vehicle) <= t - 1);
                h_nodes = setdiff(h_nodes, j_node);
                for h_index = 1:numel(h_nodes)
                    h_node = h_nodes(h_index);
                    traveling_time = traveling_times(h_node, j_node, vehicle);
                    sum_right(j_node, vehicle, t) = sum_right(j_node, vehicle, t)...
                        + x(h_node, j_node, vehicle, t - traveling_time);
                end
            end

        end
    end
end

cons5 = sum_left <= sum_right;
disp('Constraint #5 Assigned')

cons6 = optimconstr(number_nodes, number_vehicles);
for i_node = 1:number_nodes
    cons6(i_node, :) = sum(x(i_node, i_node, :, :), 4) == 0;
end
disp('Constraint #6 Assigned')

cons7 = sum(sum(x(:, demand_indices, :, :), 4), 1) >= 1;
disp('Constraint #7 Assigned')

cons8 = sum(sum(sum(x(processing_indices, parking_indices, :, :),4), 2), 1)...
    == 1;
disp('Constraint #8 Assigned')

cons9 = optimconstr(numel(parking_indices), number_vehicles, final_time);

M = 500;

for parking_index = 1:numel(parking_indices)
    parking_node = parking_indices(parking_index);
    for t_prime = 1:final_time - 1
        cons9(parking_index, :, t_prime) = M * (1 - ...
            sum(x(:, parking_node, :, t_prime), 1)) >= ...
            sum(sum(sum(x(:, :, :, t_prime + 1:end), 4), 2), 1);
    end
end
disp('Constraint #9 Assigned')

cons10 = W(:, 1) == 0;
disp('Constraint #10 Assigned')

cons11 = optimconstr(numel(demand_indices), final_time);
for j_index = 1:numel(demand_indices)
    j_node = demand_indices(j_index);
    cons11(j_index, 1:end - 1) = W(j_index, 2:end) >= W(j_index, 1:end - 1) +...
        alpha(j_index) - M * reshape(...
        sum(sum(x(j_node, :, :, 2:end), 3), 2),...
        1, final_time - 1);
end
disp('Constraint #11 Assigned')

cons12 = l(:, 1) == 0;
disp('Constraint #12 Assigned')

cons13 = l(:, 1:end - 1) - M * reshape(...
    sum(sum(x(processing_indices, :, :, 2:end), 2), 1),...
    [number_vehicles, final_time - 1]) +...
    reshape(sum(WX(:, :, 2:end), 1), [number_vehicles, final_time - 1])...
    <= l(:, 2:end);

disp('Constraint #13 Assigned')

disp('Constraint 14 is contained in the boundaries of the variable')

cons15 = optimconstr(numel(demand_indices), number_vehicles, final_time);

for vehicle = 1:number_vehicles
    cons15(:, vehicle, 2:end) = WX(:, vehicle, 2:end) >= reshape(...
        W(:, 1:end - 1) + repmat(alpha, [1, final_time - 1]) - M * ...
        (1 - reshape(...
        sum(x(demand_indices, :, vehicle, 2:end), 2),...
        [numel(demand_indices), final_time - 1])),...
        [numel(demand_indices), 1, final_time - 1]);
end

disp('Constraint #15 Assigned')

prob = optimproblem('ObjectiveSense', 'minimize');

prob.Objective = objective_function;
prob.Constraints.cons1 = cons1;
prob.Constraints.cons2 = cons2;
prob.Constraints.cons3 = cons3;
prob.Constraints.cons4 = cons4;
prob.Constraints.cons5 = cons5;
prob.Constraints.cons6 = cons6;
prob.Constraints.cons7 = cons7;
prob.Constraints.cons8 = cons8;
prob.Constraints.cons9 = cons9;
prob.Constraints.cons10 = cons10;
prob.Constraints.cons11 = cons11;
prob.Constraints.cons12 = cons12;
prob.Constraints.cons13 = cons13;
prob.Constraints.cons15 = cons15;

options = optimoptions('intlinprog');
[x, fval, exitflag, output] = solve(prob, 'Options', options);

%% Reading solution
clear, clc, close all
load instance_parameters.mat
load optimization_results.mat

% Number of routes per vehicle
number_routes = sum(sum(sum(x_values(:, :, :, :), 4), 2), 1);

routes = cell(number_vehicles, 1);

for vehicle = 1:number_vehicles
    vehicle_route = zeros(number_routes(vehicle), 3);
    routeIndex = 0;

    for t = 1:final_time
        for i = 1:number_nodes
            for j = 1:number_nodes
                if i ~= j && x_values(i, j, vehicle, t) == 1
                    routeIndex = routeIndex + 1;
                    vehicle_route(routeIndex, :) = [t, i, j];
                end
            end
        end
    end

    routes{vehicle} = vehicle_route;
end


for vehicle = 1:number_vehicles
    fprintf('Route for Vehicle %d:\n', vehicle);
    disp(routes{vehicle});
end

figure
for demand_index = 1:numel(demand_indices)
    plot(1:final_time, W_values(demand_index, :)); hold on
end
title('Uncollected Waste at Demand Nodes Over Time');
xlabel('Time');
ylabel('Uncollected Waste');
legend(arrayfun(@(x) sprintf('Node %d', x), demand_indices, 'UniformOutput',...
    false));

