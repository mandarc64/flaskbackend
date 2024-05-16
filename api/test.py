import json
global json_data
import numpy as np


def minmodel_deter(D1, D2, D3, D4, lt_vl_mean, lt_vl_std, ut_vl_mean, ut_vl_std, 
                   lt_l_mean, lt_l_std, ut_l_mean, ut_l_std, speed_lt_vl_mean, speed_lt_vl_std, 
                   speed_ut_vl_mean, speed_ut_vl_std, speed_lt_l_mean, speed_lt_l_std, 
                   speed_ut_l_mean, speed_ut_l_std,dropdown1,dropdown2,dropdown3,dropdown4,num_vl,num_l,checkbox1,checkbox2,checkbox3,checkbox4):
    print(num_vl,num_l)
    from gurobipy import Model, GRB, quicksum
    import numpy as np
    import math
    def generate_machine_names(num_vl, num_l):
        vl_names = [f'VL{i+1}' for i in range(num_vl)]
        l_names = [f'L{i+1+num_vl}' for i in range(num_l)]
        return vl_names + l_names
    
    machines = generate_machine_names(num_vl, num_l)
    print(machines)

    def generate_distances(machines, distances_for_destinations, no_fly_conditions):
        additional_distances = {
            'D1': 252,
            'D2': 246,
            'D3': 57,
            'D4': 112
        }
        distances_dict = {}
        for machine in machines:
            for destination, base_distance in distances_for_destinations.items():
                if no_fly_conditions.get(destination, False):
                    adjusted_distance = base_distance + additional_distances[destination]
                else:
                    adjusted_distance = base_distance
                distances_dict[(machine, destination)] = adjusted_distance
        return distances_dict

    # Define base distances for each destination
    distances_for_destinations = {
        'D1': 5821.28,
        'D2': 3919.69,
        'D3': 3572.20,
        'D4': 716.47
    }

    # Define no-fly conditions for each destination
    no_fly_conditions = {
        'D1': checkbox1,
        'D2': checkbox2,
        'D3': checkbox3,
        'D4': checkbox4
    }
    distance_costs = generate_distances(machines, distances_for_destinations, no_fly_conditions)
    print(distance_costs)

    def generate_capacities(machines, vl_capacity, l_capacity):
        capacities = {}
        for machine in machines:
            if 'VL' in machine:
                capacities[machine] = vl_capacity
            elif 'L' in machine:
                capacities[machine] = l_capacity
        return capacities

    # Capacities for each type of machine
    vl_capacity = 100  # Capacity for VL machines
    l_capacity = 75    # Capacity for L machines

    # Generate the capacities dictionary
    capacity = generate_capacities(machines, vl_capacity, l_capacity)
    print(capacity)

    def generate_cost_factors(machines, vl_cost_factor, l_cost_factor):
        cost_factors = {}
        for machine in machines:
            if 'VL' in machine:
                cost_factors[machine] = vl_cost_factor
            elif 'L' in machine:
                cost_factors[machine] = l_cost_factor
        return cost_factors

    # Cost factors for each type of machine
    vl_cost_factor = 1    # Cost factor for VL machines
    l_cost_factor = 0.75  # Cost factor for L machines

    # Generate the cost factors dictionary
    cost_factors = generate_cost_factors(machines, vl_cost_factor, l_cost_factor)
    print(cost_factors)


    jobs = ['D1', 'D2', 'D3', 'D4']  # Demand locations

    job_quantities = {'D1': D1, 'D2': D2, 'D3': D3, 'D4': D4}  # Job quantities



    # Generate random load and unload times
    lt_vl = np.random.normal(lt_vl_mean, lt_vl_std) / 60
    ut_vl = np.random.normal(ut_vl_mean, ut_vl_std) / 60
    lt_l = np.random.normal(lt_l_mean, lt_l_std) / 60
    ut_l = np.random.normal(ut_l_mean, ut_l_std) / 60

    # Generate random speeds for VL and L aircraft
    speed_lt_vl = np.random.normal(speed_lt_vl_mean, speed_lt_vl_std)
    speed_ut_vl = np.random.normal(speed_ut_vl_mean, speed_ut_vl_std)
    speed_lt_l = np.random.normal(speed_lt_l_mean, speed_lt_l_std)
    speed_ut_l = np.random.normal(speed_ut_l_mean, speed_ut_l_std)


    # Dictionary to store processing times with stochastic variations
    processing_times = {}

    # Loop through each aircraft and destination pair to calculate processing times
    for i in machines:
        ac_type = 'VL' if i.startswith('VL') else 'L'
        for j in jobs:
            if ac_type == 'VL':
                load_time = lt_vl
                unload_time = ut_vl
                speed_lt = speed_lt_vl
                speed_ut = speed_ut_vl
            else:
                load_time = lt_l
                unload_time = ut_l
                speed_lt = speed_lt_l
                speed_ut = speed_ut_l
            
            # Calculate travel time including stochastic variations
            travel_time = load_time + unload_time + distance_costs[i, j] / speed_lt + distance_costs[i, j] / speed_ut
            
            # Round up to the nearest minute
            travel_time_rounded = math.ceil(travel_time)
            
            # Assign the calculated processing time to the dictionary
            processing_times[(i, j)] = travel_time_rounded
    print(processing_times)   



    shifts = [1, 2, 3, 4, 5]  # Assuming a maximum of 5 shifts to start

    # Initialize the model
    m = Model("Job_Scheduling_with_Flexible_Objectives")

    # Variables
    x = m.addVars(machines, jobs, shifts, vtype=GRB.INTEGER, lb=0, name="x")
    t = m.addVars(machines, shifts, vtype=GRB.CONTINUOUS, lb=0, name="start_time")
    d = m.addVars(machines, shifts, vtype=GRB.CONTINUOUS, lb=0, name="duration")
    job_active = m.addVars(machines, jobs, shifts, vtype=GRB.BINARY, name="job_active")

    # Total cost variable
    total_cost = m.addVar(vtype=GRB.CONTINUOUS, lb=0, name="total_cost")

    # Total completion time variable
    total_completion_time = m.addVar(vtype=GRB.CONTINUOUS, lb=0, name="total_completion_time")

    # Objective functions
    # Choose objective
    #m.setObjective(total_cost, GRB.MINIMIZE)  # Minimize total cost
    m.setObjective(total_completion_time, GRB.MINIMIZE)  # Minimize total completion time

    # Constraints
    # Job quantity constraints
    # Job quantity constraints: Ensure that the total quantity of each job assigned to all machines and shifts equals the required job quantity

    for j in jobs:
        m.addConstr(quicksum(x[i, j, s] for i in machines for s in shifts) == job_quantities[j], name=f"job_qty_{j}")

    # Capacity constraints
    # Capacity constraints: Ensure that the total quantity of jobs assigned to each machine in each shift does not exceed the machine's capacity

    for i in machines:
        for s in shifts:
            m.addConstr(quicksum(x[i, j, s] for j in jobs) <= capacity[i], name=f"cap_{i}_{s}")

    # Linking job processing to binary activation
    # Capacity constraints: Ensure that the total quantity of jobs assigned to each machine in each shift does not exceed the machine's capacity

    for i in machines:
        for j in jobs:
            for s in shifts:
                m.addConstr(job_active[i, j, s] >= x[i, j, s] / capacity[i])
                m.addConstr(job_active[i, j, s] * capacity[i] >= x[i, j, s])

    # Duration and start times of shifts
    # Duration and start times of shifts: Calculate the duration of each machine in each shift based on the processing times of active jobs, and ensure that the completion time of each machine in each shift is less than or equal to the total completion time

    for i in machines:
        for s in shifts:
            m.addConstr(d[i, s] == quicksum(processing_times[(i, j)] * job_active[i, j, s] for j in jobs), name=f"dur_{i}_{s}")
            m.addConstr(t[i, s] + d[i, s] <= total_completion_time, name=f"completion_time_{i}_{s}")

    # Sequential shift start time constraints
    # Sequential shift start time constraints: Ensure that the start time of each machine in each shift is greater than or equal to the completion time of the previous shift

    for i in machines:
        for s in range(2, len(shifts) + 1):
            m.addConstr(t[i, s] >= t[i, s-1] + d[i, s-1], name=f"start_time_{i}_{s}")

    # Total cost constraint
    # Total cost constraint: Calculate the total cost of the schedule based on the distance costs, job activation, and cost factors
    m.addConstr(total_cost == quicksum(distance_costs[(i, j)] * job_active[i, j, s] * cost_factors[i] for i in machines for j in jobs for s in shifts), name="total_cost_constraint")

    # Solve the model
    m.optimize()
    # Output results
    if m.status == GRB.OPTIMAL:

        print(f"Total cost: {total_cost.x:.2f}")

        print(f"Total completion time: {total_completion_time.x:.2f}")

        total_trips = 0

        for i in machines:

            active_shifts = [s for s in shifts if sum(job_active[i, j, s].x for j in jobs) > 0]

            total_shifts = len(active_shifts)

            total_trips += total_shifts

            print(f"Machine {i} used {total_shifts} shifts.")

            for s in active_shifts:

                duration = float(d[i, s].x)

                cost = float(sum(distance_costs[(i, j)] * job_active[i, j, s].x * cost_factors[i] for j in jobs))

                completion_time_shift = float(t[i, s].x + d[i, s].x)

                print(f"Machine {i}, Shift {s}, Start Time: {t[i, s].x:.2f}, Duration: {duration:.2f}, Cost: {cost:.2f}, Completion Time: {completion_time_shift:.2f}")

                for j in jobs:

                    if x[i, j, s].x > 0:

                        print(f"  Job {j}: {x[i, j, s].x} units")

        print(f"Total trips: {total_trips}")

    

        # Dictionary to store machines and the number of shifts they worked for each job

        job_machine_assignments = {job: {} for job in jobs}

    

        # Populate the dictionary with machines and their shift counts

        for i in machines:

            for j in jobs:

                shift_count = sum(x[i, j, s].x > 0 for s in shifts)  # Total shifts machine i works on job j

                if shift_count > 0:

                    if i in job_machine_assignments[j]:

                        job_machine_assignments[j][i] += shift_count

                    else:

                        job_machine_assignments[j][i] = shift_count

    

        # Printing the summary for each job with machines and their shift counts

        print("Machine assignments per job with shift counts:")

        for job, machines in job_machine_assignments.items():

            formatted_machines = [(machine, shifts) for machine, shifts in machines.items()]

            print(f"Job {job}: {formatted_machines}")

        return {
            'job_machine_assignments': job_machine_assignments,
            'total_cost': total_cost.x,
            'total_completion_time': total_completion_time.x,
            'total_trips': total_trips
        }

def exp_min_time(D1, D2, D3, D4, lt_vl_mean, lt_vl_std, ut_vl_mean, ut_vl_std, 
                 lt_l_mean, lt_l_std, ut_l_mean, ut_l_std, speed_lt_vl_mean, speed_lt_vl_std, 
                 speed_ut_vl_mean, speed_ut_vl_std, speed_lt_l_mean, speed_lt_l_std, 
                 speed_ut_l_mean, speed_ut_l_std, dropdown1,dropdown2,dropdown3,dropdown4,num_vl,num_l,checkbox1,checkbox2,checkbox3,checkbox4,num_runs=500):
    
    import matplotlib.pyplot as plt
    from gurobipy import Model, GRB, quicksum
    import numpy as np
    import math
    import gurobipy as gp
    from gurobipy import GRB
    import pandas as pd
    import math
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.stats import pareto
    from scipy.optimize import fsolve
    import scipy.stats

    print(num_vl,num_l)

    num_runs = 500  # Number of times to run the optimization

    total_costs = []
    total_completion_times = []
    total_trips_list = []

    processing_times_tracker = {}
    scenario_counter = 0 
    #weighted average of the 500 sample average the opt. 
    #sample approximation
    # find the deciisons rhat will give the best optimal decision when we have multiple tamge
    for _ in range(num_runs):
        def generate_machine_names(num_vl, num_l):
            vl_names = [f'VL{i+1}' for i in range(num_vl)]
            l_names = [f'L{i+1+num_vl}' for i in range(num_l)]
            return vl_names + l_names

        # Generate machine names
        # num_vl = 3  # Number of VL machines
        # num_l = 3   # Number of L machines
        machines = generate_machine_names(num_vl, num_l)
        print(machines)

        def generate_distances(machines, distances_for_destinations, no_fly_conditions):
            additional_distances = {
                'D1': 252,
                'D2': 246,
                'D3': 57,
                'D4': 112
            }
            distances_dict = {}
            for machine in machines:
                for destination, base_distance in distances_for_destinations.items():
                    if no_fly_conditions.get(destination, False):
                        adjusted_distance = base_distance + additional_distances[destination]
                    else:
                        adjusted_distance = base_distance
                    distances_dict[(machine, destination)] = adjusted_distance
            return distances_dict

        # Define base distances for each destination
        distances_for_destinations = {
            'D1': 5821.28,
            'D2': 3919.69,
            'D3': 3572.20,
            'D4': 716.47
        }

        # Define no-fly conditions for each destination
        no_fly_conditions = {
            'D1': checkbox1,
            'D2': checkbox2,
            'D3': checkbox3,
            'D4': checkbox4
        }
        distance_costs = generate_distances(machines, distances_for_destinations, no_fly_conditions)
        print(distance_costs)

        def generate_capacities(machines, vl_capacity, l_capacity):
            capacities = {}
            for machine in machines:
                if 'VL' in machine:
                    capacities[machine] = vl_capacity
                elif 'L' in machine:
                    capacities[machine] = l_capacity
            return capacities

        # Capacities for each type of machine
        vl_capacity = 100  # Capacity for VL machines
        l_capacity = 75    # Capacity for L machines

        # Generate the capacities dictionary
        capacity = generate_capacities(machines, vl_capacity, l_capacity)
        print(capacity)

        def generate_cost_factors(machines, vl_cost_factor, l_cost_factor):
            cost_factors = {}
            for machine in machines:
                if 'VL' in machine:
                    cost_factors[machine] = vl_cost_factor
                elif 'L' in machine:
                    cost_factors[machine] = l_cost_factor
            return cost_factors

        # Cost factors for each type of machine
        vl_cost_factor = 1    # Cost factor for VL machines
        l_cost_factor = 0.75  # Cost factor for L machines

        # Generate the cost factors dictionary
        cost_factors = generate_cost_factors(machines, vl_cost_factor, l_cost_factor)
        print(cost_factors)


        jobs = ['D1', 'D2', 'D3', 'D4']  # Demand locations

        job_quantities = {'D1': D1, 'D2': D2, 'D3': D3, 'D4': D4}  # Job quantities
        
        # STOCHASTIC PARAMETERS
        
       # Generate random load and unload times
        lt_vl = np.random.normal(lt_vl_mean, lt_vl_std) / 60
        ut_vl = np.random.normal(ut_vl_mean, ut_vl_std) / 60
        lt_l = np.random.normal(lt_l_mean, lt_l_std) / 60
        ut_l = np.random.normal(ut_l_mean, ut_l_std) / 60

        # Generate random speeds for VL and L aircraft
        speed_lt_vl = np.random.normal(speed_lt_vl_mean, speed_lt_vl_std)
        speed_ut_vl = np.random.normal(speed_ut_vl_mean, speed_ut_vl_std)
        speed_lt_l = np.random.normal(speed_lt_l_mean, speed_lt_l_std)
        speed_ut_l = np.random.normal(speed_ut_l_mean, speed_ut_l_std)


        # Dictionary to store processing times with stochastic variations
        processing_times = {}

        # Loop through each aircraft and destination pair to calculate processing times
        for i in machines:
            ac_type = 'VL' if i.startswith('VL') else 'L'
            for j in jobs:
                if ac_type == 'VL':
                    load_time = lt_vl
                    unload_time = ut_vl
                    speed_lt = speed_lt_vl
                    speed_ut = speed_ut_vl
                else:
                    load_time = lt_l
                    unload_time = ut_l
                    speed_lt = speed_lt_l
                    speed_ut = speed_ut_l
                
                # Calculate travel time including stochastic variations
                travel_time = load_time + unload_time + distance_costs[i, j] / speed_lt + distance_costs[i, j] / speed_ut
                
                # Round up to the nearest minute
                travel_time_rounded = math.ceil(travel_time)
                
                # Assign the calculated processing time to the dictionary
                processing_times[(i, j)] = travel_time_rounded

        #print(processing_times)
        processing_times_tracker[scenario_counter] = processing_times
        scenario_counter += 1
        shifts = [1, 2, 3, 4, 5]  # Assuming a maximum of 5 shifts to start

        # Initialize the model
        m = Model("Job_Scheduling_with_Flexible_Objectives")

        # Variables
        x = m.addVars(machines, jobs, shifts, vtype=GRB.INTEGER, lb=0, name="x")
        t = m.addVars(machines, shifts, vtype=GRB.CONTINUOUS, lb=0, name="start_time")
        d = m.addVars(machines, shifts, vtype=GRB.CONTINUOUS, lb=0, name="duration")
        job_active = m.addVars(machines, jobs, shifts, vtype=GRB.BINARY, name="job_active")

        # Total cost variable
        total_cost = m.addVar(vtype=GRB.CONTINUOUS, lb=0, name="total_cost")
        # all scenarios have the equal weight, what isthe opt valye 
        # Total completion time variable
        total_completion_time = m.addVar(vtype=GRB.CONTINUOUS, lb=0, name="total_completion_time")

        # Objective functions
        # Choose objective
        #m.setObjective(total_cost, GRB.MINIMIZE)  # Minimize total cost
        m.setObjective(total_completion_time, GRB.MINIMIZE)  # Minimize total completion time

        # Constraints
        # Job quantity constraints
        for j in jobs:
            m.addConstr(quicksum(x[i, j, s] for i in machines for s in shifts) == job_quantities[j], name=f"job_qty_{j}")

        # Capacity constraints
        for i in machines:
            for s in shifts:
                m.addConstr(quicksum(x[i, j, s] for j in jobs) <= capacity[i], name=f"cap_{i}_{s}")

        # Linking job processing to binary activation
        for i in machines:
            for j in jobs:
                for s in shifts:
                    m.addConstr(job_active[i, j, s] >= x[i, j, s] / capacity[i])
                    m.addConstr(job_active[i, j, s] * capacity[i] >= x[i, j, s])

        # Duration and start times of shifts
        for i in machines:
            for s in shifts:
                m.addConstr(d[i, s] == quicksum(processing_times[(i, j)] * job_active[i, j, s] for j in jobs), name=f"dur_{i}_{s}")
                m.addConstr(t[i, s] + d[i, s] <= total_completion_time, name=f"completion_time_{i}_{s}")

        # Sequential shift start time constraints
        for i in machines:
            for s in range(2, len(shifts) + 1):
                m.addConstr(t[i, s] >= t[i, s-1] + d[i, s-1], name=f"start_time_{i}_{s}")

        # Total cost constraint
        m.addConstr(total_cost == quicksum(distance_costs[(i, j)] * job_active[i, j, s] * cost_factors[i] for i in machines for j in jobs for s in shifts), name="total_cost_constraint")

        # Solve the model
        m.optimize()

        # Store the results
        if m.status == GRB.OPTIMAL:
            total_costs.append(total_cost.x)
            total_completion_times.append(total_completion_time.x)
            
            total_trips = 0
            for i in machines:
                active_shifts = [s for s in shifts if sum(job_active[i, j, s].x for j in jobs) > 0]
                total_shifts = len(active_shifts)
                total_trips += total_shifts
            total_trips_list.append(total_trips)

    df = pd.DataFrame({
        'total_costs': total_costs,
        'total_completion_times': total_completion_times,
        'total_trips': total_trips_list
    })

    # Sort the DataFrame by total_costs
    df_sorted = df.sort_values(by='total_completion_times')

    # Calculate the median of total costs
    median_total_time = df_sorted['total_completion_times'].median()


    # Find the index of the median cost value
    median_index = df_sorted[df_sorted['total_completion_times'] == median_total_time].index[0]  # In case of multiple median matches, take the first
    print(f"median Index is: {median_index}")
    # Extract corresponding completion time and trips
    corresponding_cost = df_sorted.loc[median_index, 'total_costs']
    corresponding_trips = df_sorted.loc[median_index, 'total_trips']

    # Output the results
    print(f"Median Total Cost: {corresponding_cost}")
    print(f"Corresponding Completion Time: {median_total_time}")
    print(f"Corresponding Total Trips: {corresponding_trips}")

    # Plot the histograms
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.hist(total_costs, bins=20, edgecolor='black')
    plt.xlabel('Total Cost')
    plt.ylabel('Frequency')
    plt.title('Histogram of Total Cost')

    plt.subplot(1, 3, 2)
    plt.hist(total_completion_times, bins=20, edgecolor='black')
    plt.xlabel('Total Completion Time')
    plt.ylabel('Frequency')
    plt.title('Histogram of Total Completion Time')

    plt.subplot(1, 3, 3)
    plt.hist(total_trips_list, bins=20, edgecolor='black')
    plt.xlabel('Total Trips')
    plt.ylabel('Frequency')
    plt.title('Histogram of Total Trips')

    plt.tight_layout()
    plt.show()

    print("Total Completiion Times: ",total_completion_times)
    print("Total Cost ",total_costs)
    print("Total Trips" ,total_trips_list)


    ##### BELOW IS THE CODE WHICH WE RETURN AS FINAL RESULT FOR THE UI when the option is selected as expected min cost #######


    # print the best optimal solnts
    processing_times = {}
    processing_times  = processing_times_tracker[median_index]
    processing_times
    #machines = ['VL1', 'VL2', 'VL3', 'VL4', 'VL5', 'VL6']  # Aircraft
    def generate_machine_names(num_vl, num_l):
        vl_names = [f'VL{i+1}' for i in range(num_vl)]
        l_names = [f'L{i+1+num_vl}' for i in range(num_l)]
        return vl_names + l_names

    # Generate machine names
    # num_vl = 3  # Number of VL machines
    # num_l = 3   # Number of L machines
    machines = generate_machine_names(num_vl, num_l)
    print(machines)

    def generate_distances(machines, distances_for_destinations, no_fly_conditions):
        additional_distances = {
            'D1': 252,
            'D2': 246,
            'D3': 57,
            'D4': 112
        }
        distances_dict = {}
        for machine in machines:
            for destination, base_distance in distances_for_destinations.items():
                if no_fly_conditions.get(destination, False):
                    adjusted_distance = base_distance + additional_distances[destination]
                else:
                    adjusted_distance = base_distance
                distances_dict[(machine, destination)] = adjusted_distance
        return distances_dict

    # Define base distances for each destination
    distances_for_destinations = {
        'D1': 5821.28,
        'D2': 3919.69,
        'D3': 3572.20,
        'D4': 716.47
    }

    # Define no-fly conditions for each destination
    no_fly_conditions = {
            'D1': checkbox1,
            'D2': checkbox2,
            'D3': checkbox3,
            'D4': checkbox4
    }
    distance_costs = generate_distances(machines, distances_for_destinations, no_fly_conditions)
    print(distance_costs)

    def generate_capacities(machines, vl_capacity, l_capacity):
        capacities = {}
        for machine in machines:
            if 'VL' in machine:
                capacities[machine] = vl_capacity
            elif 'L' in machine:
                capacities[machine] = l_capacity
        return capacities

    # Capacities for each type of machine
    vl_capacity = 100  # Capacity for VL machines
    l_capacity = 75    # Capacity for L machines

    # Generate the capacities dictionary
    capacity = generate_capacities(machines, vl_capacity, l_capacity)
    print(capacity)

    def generate_cost_factors(machines, vl_cost_factor, l_cost_factor):
        cost_factors = {}
        for machine in machines:
            if 'VL' in machine:
                cost_factors[machine] = vl_cost_factor
            elif 'L' in machine:
                cost_factors[machine] = l_cost_factor
        return cost_factors

    # Cost factors for each type of machine
    vl_cost_factor = 1    # Cost factor for VL machines
    l_cost_factor = 0.75  # Cost factor for L machines

    # Generate the cost factors dictionary
    cost_factors = generate_cost_factors(machines, vl_cost_factor, l_cost_factor)
    print(cost_factors)


    jobs = ['D1', 'D2', 'D3', 'D4']  # Demand locations

    job_quantities = {'D1': D1, 'D2': D2, 'D3': D3, 'D4': D4}  # Job quantities
    # Initialize the model
    m = Model("Job_Scheduling_with_Flexible_Objectives")

    # Variables
    x = m.addVars(machines, jobs, shifts, vtype=GRB.INTEGER, lb=0, name="x")
    t = m.addVars(machines, shifts, vtype=GRB.CONTINUOUS, lb=0, name="start_time")
    d = m.addVars(machines, shifts, vtype=GRB.CONTINUOUS, lb=0, name="duration")
    job_active = m.addVars(machines, jobs, shifts, vtype=GRB.BINARY, name="job_active")

    # Total cost variable
    total_cost = m.addVar(vtype=GRB.CONTINUOUS, lb=0, name="total_cost")

    # Total completion time variable
    total_completion_time = m.addVar(vtype=GRB.CONTINUOUS, lb=0, name="total_completion_time")

    # Objective functions
    # Choose objective
    #m.setObjective(total_cost, GRB.MINIMIZE)  # Minimize total cost
    m.setObjective(total_completion_time, GRB.MINIMIZE)  # Minimize total completion time

    # Constraints
    # Job quantity constraints
    # Job quantity constraints: Ensure that the total quantity of each job assigned to all machines and shifts equals the required job quantity

    for j in jobs:
        m.addConstr(quicksum(x[i, j, s] for i in machines for s in shifts) == job_quantities[j], name=f"job_qty_{j}")

    # Capacity constraints
    # Capacity constraints: Ensure that the total quantity of jobs assigned to each machine in each shift does not exceed the machine's capacity

    for i in machines:
        for s in shifts:
            m.addConstr(quicksum(x[i, j, s] for j in jobs) <= capacity[i], name=f"cap_{i}_{s}")

    # Linking job processing to binary activation
    # Capacity constraints: Ensure that the total quantity of jobs assigned to each machine in each shift does not exceed the machine's capacity

    for i in machines:
        for j in jobs:
            for s in shifts:
                m.addConstr(job_active[i, j, s] >= x[i, j, s] / capacity[i])
                m.addConstr(job_active[i, j, s] * capacity[i] >= x[i, j, s])

    # Duration and start times of shifts
    # Duration and start times of shifts: Calculate the duration of each machine in each shift based on the processing times of active jobs, and ensure that the completion time of each machine in each shift is less than or equal to the total completion time

    for i in machines:
        for s in shifts:
            m.addConstr(d[i, s] == quicksum(processing_times[(i, j)] * job_active[i, j, s] for j in jobs), name=f"dur_{i}_{s}")
            m.addConstr(t[i, s] + d[i, s] <= total_completion_time, name=f"completion_time_{i}_{s}")

    # Sequential shift start time constraints
    # Sequential shift start time constraints: Ensure that the start time of each machine in each shift is greater than or equal to the completion time of the previous shift

    for i in machines:
        for s in range(2, len(shifts) + 1):
            m.addConstr(t[i, s] >= t[i, s-1] + d[i, s-1], name=f"start_time_{i}_{s}")

    # Total cost constraint
    # Total cost constraint: Calculate the total cost of the schedule based on the distance costs, job activation, and cost factors
    m.addConstr(total_cost == quicksum(distance_costs[(i, j)] * job_active[i, j, s] * cost_factors[i] for i in machines for j in jobs for s in shifts), name="total_cost_constraint")

    # Solve the model
    m.optimize()
    # Output results
    # if m.status == GRB.OPTIMAL:
    #     print(f"Total cost: {total_cost.x:.2f}")
    #     print(f"Total completion time: {total_completion_time.x:.2f}")
    #     total_trips = 0
    #     for i in machines:
    #         active_shifts = [s for s in shifts if sum(job_active[i, j, s].x for j in jobs) > 0]
    #         total_shifts = len(active_shifts)
    #         total_trips += total_shifts
    #         print(f"Machine {i} used {total_shifts} shifts.")
    #         for s in active_shifts:
    #             duration = float(d[i, s].x)
    #             cost = float(sum(distance_costs[(i, j)] * job_active[i, j, s].x * cost_factors[i] for j in jobs))
    #             completion_time_shift = float(t[i, s].x + d[i, s].x)
    #             print(f"Machine {i}, Shift {s}, Start Time: {t[i, s].x:.2f}, Duration: {duration:.2f}, Cost: {cost:.2f}, Completion Time: {completion_time_shift:.2f}")
    #             for j in jobs:
    #                 if x[i, j, s].x > 0:
    #                     print(f"  Job {j}: {x[i, j, s].x} units")
    #     print(f"Total trips: {total_trips}")
        
    #     # Dictionary to store sets of machines assigned to each job
    #     job_machine_assignments = {job: set() for job in jobs}
        
    #     # Example of populating the dictionary based on hypothetical model outputs
    #     # For each machine, job, and shift, check if the machine worked on the job during any shift
    #     for i in machines:
    #         for j in jobs:
    #             for s in shifts:
    #                 # Hypothetically accessing a decision variable from Gurobi model, need real model data here
    #                 if x[i, j, s].x > 0:  # Assuming x[i, j, s].x indicates machine i worked on job j in shift s
    #                     job_machine_assignments[j].add(i)
        
    #     # Convert sets to lists to finalize the output
    #     for job, machines in job_machine_assignments.items():
    #         job_machine_assignments[job] = list(machines)
        
    #     # Printing the summary for each job
    #     print("Machine assignments per job:")
    #     for job, machines in job_machine_assignments.items():
    #         print(f"Job {job}: {machines}")
    if m.status == GRB.OPTIMAL:

        print(f"Total cost: {total_cost.x:.2f}")

        print(f"Total completion time: {total_completion_time.x:.2f}")

        total_trips = 0

        for i in machines:

            active_shifts = [s for s in shifts if sum(job_active[i, j, s].x for j in jobs) > 0]

            total_shifts = len(active_shifts)

            total_trips += total_shifts

            print(f"Machine {i} used {total_shifts} shifts.")

            for s in active_shifts:

                duration = float(d[i, s].x)

                cost = float(sum(distance_costs[(i, j)] * job_active[i, j, s].x * cost_factors[i] for j in jobs))

                completion_time_shift = float(t[i, s].x + d[i, s].x)

                print(f"Machine {i}, Shift {s}, Start Time: {t[i, s].x:.2f}, Duration: {duration:.2f}, Cost: {cost:.2f}, Completion Time: {completion_time_shift:.2f}")

                for j in jobs:

                    if x[i, j, s].x > 0:

                        print(f"  Job {j}: {x[i, j, s].x} units")

        print(f"Total trips: {total_trips}")

    

        # Dictionary to store machines and the number of shifts they worked for each job

        job_machine_assignments = {job: {} for job in jobs}

    

        # Populate the dictionary with machines and their shift counts

        for i in machines:

            for j in jobs:

                shift_count = sum(x[i, j, s].x > 0 for s in shifts)  # Total shifts machine i works on job j

                if shift_count > 0:

                    if i in job_machine_assignments[j]:

                        job_machine_assignments[j][i] += shift_count

                    else:

                        job_machine_assignments[j][i] = shift_count

    

        # Printing the summary for each job with machines and their shift counts

        print("Machine assignments per job with shift counts:")

        for job, machines in job_machine_assignments.items():

            formatted_machines = [(machine, shifts) for machine, shifts in machines.items()]

            print(f"Job {job}: {formatted_machines}")

        return {
            'job_machine_assignments': job_machine_assignments,
            'total_cost': total_cost.x,
            'total_completion_time': total_completion_time.x,
            'total_trips': total_trips,
            'total_trips_list': total_trips_list,
            'total_comp_list': total_completion_times
        }    
    

def exp_min_cost(D1, D2, D3, D4, lt_vl_mean, lt_vl_std, ut_vl_mean, ut_vl_std, 
                 lt_l_mean, lt_l_std, ut_l_mean, ut_l_std, speed_lt_vl_mean, speed_lt_vl_std, 
                 speed_ut_vl_mean, speed_ut_vl_std, speed_lt_l_mean, speed_lt_l_std, 
                 speed_ut_l_mean, speed_ut_l_std,dropdown1,dropdown2,dropdown3,dropdown4,num_vl,num_l,checkbox1,checkbox2,checkbox3,checkbox4,num_runs=500):
    import matplotlib.pyplot as plt
    from gurobipy import Model, GRB, quicksum
    import numpy as np
    import math
    import gurobipy as gp
    from gurobipy import GRB
    import pandas as pd
    import math
    import numpy as np
    import matplotlib.pyplot as plt

    num_runs = 500  # Number of times to run the optimization

    total_costs = []
    total_completion_times = []
    total_trips_list = []

    processing_times_tracker = {}
    scenario_counter = 0 
    #weighted average of the 500 sample average the opt. 
    #sample approximation
    # find the deciisons rhat will give the best optimal decision when we have multiple tamge
    for _ in range(num_runs):
        def generate_machine_names(num_vl, num_l):
            vl_names = [f'VL{i+1}' for i in range(num_vl)]
            l_names = [f'L{i+1+num_vl}' for i in range(num_l)]
            return vl_names + l_names

        # Generate machine names
        # num_vl = 3  # Number of VL machines
        # num_l = 3   # Number of L machines
        machines = generate_machine_names(num_vl, num_l)
        print(machines)

        def generate_distances(machines, distances_for_destinations, no_fly_conditions):
            additional_distances = {
                'D1': 252,
                'D2': 246,
                'D3': 57,
                'D4': 112
            }
            distances_dict = {}
            for machine in machines:
                for destination, base_distance in distances_for_destinations.items():
                    if no_fly_conditions.get(destination, False):
                        adjusted_distance = base_distance + additional_distances[destination]
                    else:
                        adjusted_distance = base_distance
                    distances_dict[(machine, destination)] = adjusted_distance
            return distances_dict

        # Define base distances for each destination
        distances_for_destinations = {
            'D1': 5821.28,
            'D2': 3919.69,
            'D3': 3572.20,
            'D4': 716.47
        }

        # Define no-fly conditions for each destination
        no_fly_conditions = {
            'D1': checkbox1,
            'D2': checkbox2,
            'D3': checkbox3,
            'D4': checkbox4
        }
        distance_costs = generate_distances(machines, distances_for_destinations, no_fly_conditions)
        print(distance_costs)

        def generate_capacities(machines, vl_capacity, l_capacity):
            capacities = {}
            for machine in machines:
                if 'VL' in machine:
                    capacities[machine] = vl_capacity
                elif 'L' in machine:
                    capacities[machine] = l_capacity
            return capacities

        # Capacities for each type of machine
        vl_capacity = 100  # Capacity for VL machines
        l_capacity = 75    # Capacity for L machines

        # Generate the capacities dictionary
        capacity = generate_capacities(machines, vl_capacity, l_capacity)
        print(capacity)

        def generate_cost_factors(machines, vl_cost_factor, l_cost_factor):
            cost_factors = {}
            for machine in machines:
                if 'VL' in machine:
                    cost_factors[machine] = vl_cost_factor
                elif 'L' in machine:
                    cost_factors[machine] = l_cost_factor
            return cost_factors

        # Cost factors for each type of machine
        vl_cost_factor = 1    # Cost factor for VL machines
        l_cost_factor = 0.75  # Cost factor for L machines

        # Generate the cost factors dictionary
        cost_factors = generate_cost_factors(machines, vl_cost_factor, l_cost_factor)
        print(cost_factors)


        jobs = ['D1', 'D2', 'D3', 'D4']  # Demand locations

        job_quantities = {'D1': D1, 'D2': D2, 'D3': D3, 'D4': D4}  # Job quantities
        
        
    
        # STOCHASTIC PARAMETERS
        
        # # Generate random load and unload times
        # lt_vl = np.random.normal(45, 20) / 60  # Load time for VL aircraft
        # ut_vl = np.random.normal(45, 20) / 60  # Unload time for VL aircraft
        # lt_l = np.random.normal(45, 20) / 60  # Load time for L aircraft
        # ut_l = np.random.normal(45, 20) / 60  # Unload time for L aircraft

        # # Generate random speeds for VL and L aircraft
        # speed_lt_vl = np.random.normal(350, 100)  # Load time speed for VL aircraft
        # speed_ut_vl = np.random.normal(380, 100)  # Unload time speed for VL aircraft
        # speed_lt_l = np.random.normal(400, 100)  # Load time speed for L aircraft
        # speed_ut_l = np.random.normal(400, 100)  # Unload time speed for L aircraft

        # Generate random load and unload times
        lt_vl = np.random.normal(lt_vl_mean, lt_vl_std) / 60
        ut_vl = np.random.normal(ut_vl_mean, ut_vl_std) / 60
        lt_l = np.random.normal(lt_l_mean, lt_l_std) / 60
        ut_l = np.random.normal(ut_l_mean, ut_l_std) / 60

        # Generate random speeds for VL and L aircraft
        speed_lt_vl = np.random.normal(speed_lt_vl_mean, speed_lt_vl_std)
        speed_ut_vl = np.random.normal(speed_ut_vl_mean, speed_ut_vl_std)
        speed_lt_l = np.random.normal(speed_lt_l_mean, speed_lt_l_std)
        speed_ut_l = np.random.normal(speed_ut_l_mean, speed_ut_l_std)
        

        # # Function to find mu and sigma for the log-normal distribution
        # def find_lognorm_params(target_mean, target_std):
        #     def equations(params):
        #         mu, sigma = params
        #         mean_eq = np.exp(mu + sigma**2 / 2) - target_mean
        #         std_eq = np.sqrt((np.exp(sigma**2) - 1) * np.exp(2 * mu + sigma**2)) - target_std
        #         return (mean_eq, std_eq)
            
        #     initial_guess = [np.log(target_mean), 0.1]  # More conservative guess for sigma
        #     mu, sigma = fsolve(equations, initial_guess)
        #     return mu, sigma
        
        # # Load and unload times for VL and L aircraft
        # mu_lt_vl, sigma_lt_vl = find_lognorm_params(45, 20)
        # lt_vl = np.random.lognormal(mu_lt_vl, sigma_lt_vl) / 60

        # mu_ut_vl, sigma_ut_vl = find_lognorm_params(45, 20)
        # ut_vl = np.random.lognormal(mu_ut_vl, sigma_ut_vl) / 60
        
        # mu_lt_l, sigma_lt_l = find_lognorm_params(40, 20)
        # lt_l = np.random.lognormal(mu_lt_l, sigma_lt_l) / 60
        
        # mu_ut_l, sigma_ut_l = find_lognorm_params(40, 20)
        # ut_l = np.random.lognormal(mu_ut_l, sigma_ut_l) / 60
        
        # # Speeds for VL and L aircraft
        # mu_speed_lt_vl, sigma_speed_lt_vl = find_lognorm_params(350, 150)
        # speed_lt_vl = np.random.lognormal(mu_speed_lt_vl, sigma_speed_lt_vl)
        
        # mu_speed_ut_vl, sigma_speed_ut_vl = find_lognorm_params(380, 150)
        # speed_ut_vl = np.random.lognormal(mu_speed_ut_vl, sigma_speed_ut_vl)
        
        # mu_speed_lt_l, sigma_speed_lt_l = find_lognorm_params(400, 150)
        # speed_lt_l = np.random.lognormal(mu_speed_lt_l, sigma_speed_lt_l)
        
        # mu_speed_ut_l, sigma_speed_ut_l = find_lognorm_params(400, 150)
        # speed_ut_l = np.random.lognormal(mu_speed_ut_l, sigma_speed_ut_l)
        
        # # Print results to verify
        # print("Load time VL (hours):", lt_vl)
        # print("Unload time VL (hours):", ut_vl)
        # print("Load time L (hours):", lt_l)
        # print("Unload time L (hours):", ut_l)
        # print("Speed during load time VL (km/h):", speed_lt_vl)
        # print("Speed during unload time VL (km/h):", speed_ut_vl)
        # print("Speed during load time L (km/h):", speed_l)
        
        

        # Dictionary to store processing times with stochastic variations
        processing_times = {}

        # Loop through each aircraft and destination pair to calculate processing times
        for i in machines:
            ac_type = 'VL' if i.startswith('VL') else 'L'
            for j in jobs:
                if ac_type == 'VL':
                    load_time = lt_vl
                    unload_time = ut_vl
                    speed_lt = speed_lt_vl
                    speed_ut = speed_ut_vl
                else:
                    load_time = lt_l
                    unload_time = ut_l
                    speed_lt = speed_lt_l
                    speed_ut = speed_ut_l
                
                # Calculate travel time including stochastic variations
                travel_time = load_time + unload_time + distance_costs[i, j] / speed_lt + distance_costs[i, j] / speed_ut
                
                # Round up to the nearest minute
                travel_time_rounded = math.ceil(travel_time)
                
                # Assign the calculated processing time to the dictionary
                processing_times[(i, j)] = travel_time_rounded

        #print(processing_times)
        processing_times_tracker[scenario_counter] = processing_times
        scenario_counter += 1
        # cost_factors = {
        #     'VL1': 1, 'VL2': 1, 'VL3': 1, 'VL4': 1, 'VL5': 1, 'VL6': 1,
        #     'L1': 0.75, 'L2': 0.75, 'L3': 0.75, 'L4': 0.75, 'L5': 0.75, 'L6': 0.75,
        # } # Cost factors for each machine
        shifts = [1, 2, 3, 4, 5]  # Assuming a maximum of 5 shifts to start

        # Initialize the model
        m = Model("Job_Scheduling_with_Flexible_Objectives")

        # Variables
        x = m.addVars(machines, jobs, shifts, vtype=GRB.INTEGER, lb=0, name="x")
        t = m.addVars(machines, shifts, vtype=GRB.CONTINUOUS, lb=0, name="start_time")
        d = m.addVars(machines, shifts, vtype=GRB.CONTINUOUS, lb=0, name="duration")
        job_active = m.addVars(machines, jobs, shifts, vtype=GRB.BINARY, name="job_active")

        # Total cost variable
        total_cost = m.addVar(vtype=GRB.CONTINUOUS, lb=0, name="total_cost")
    # all scenarios have the equal weight, what isthe opt valye 
        # Total completion time variable
        total_completion_time = m.addVar(vtype=GRB.CONTINUOUS, lb=0, name="total_completion_time")

        # Objective functions
        # Choose objective
        m.setObjective(total_cost, GRB.MINIMIZE)  # Minimize total cost
        #m.setObjective(total_completion_time, GRB.MINIMIZE)  # Minimize total completion time

        # Constraints
        # Job quantity constraints
        for j in jobs:
            m.addConstr(quicksum(x[i, j, s] for i in machines for s in shifts) == job_quantities[j], name=f"job_qty_{j}")

        # Capacity constraints
        for i in machines:
            for s in shifts:
                m.addConstr(quicksum(x[i, j, s] for j in jobs) <= capacity[i], name=f"cap_{i}_{s}")

        # Linking job processing to binary activation
        for i in machines:
            for j in jobs:
                for s in shifts:
                    m.addConstr(job_active[i, j, s] >= x[i, j, s] / capacity[i])
                    m.addConstr(job_active[i, j, s] * capacity[i] >= x[i, j, s])

        # Duration and start times of shifts
        for i in machines:
            for s in shifts:
                m.addConstr(d[i, s] == quicksum(processing_times[(i, j)] * job_active[i, j, s] for j in jobs), name=f"dur_{i}_{s}")
                m.addConstr(t[i, s] + d[i, s] <= total_completion_time, name=f"completion_time_{i}_{s}")

        # Sequential shift start time constraints
        for i in machines:
            for s in range(2, len(shifts) + 1):
                m.addConstr(t[i, s] >= t[i, s-1] + d[i, s-1], name=f"start_time_{i}_{s}")

        # Total cost constraint
        m.addConstr(total_cost == quicksum(distance_costs[(i, j)] * job_active[i, j, s] * cost_factors[i] for i in machines for j in jobs for s in shifts), name="total_cost_constraint")

        # Solve the model
        m.optimize()

        # Store the results
        if m.status == GRB.OPTIMAL:
            total_costs.append(total_cost.x)
            total_completion_times.append(total_completion_time.x)
            
            total_trips = 0
            for i in machines:
                active_shifts = [s for s in shifts if sum(job_active[i, j, s].x for j in jobs) > 0]
                total_shifts = len(active_shifts)
                total_trips += total_shifts
            total_trips_list.append(total_trips)

    df = pd.DataFrame({
        'total_costs': total_costs,
        'total_completion_times': total_completion_times,
        'total_trips': total_trips_list
    })

    # Sort the DataFrame by total_costs
    df_sorted = df.sort_values(by='total_costs')

    # Calculate the median of total costs
    median_total_cost = df_sorted['total_costs'].median()

    # Find the index of the median cost value
    median_index = df_sorted[df_sorted['total_costs'] == median_total_cost].index[0]  # In case of multiple median matches, take the first
    print(f"median Index is: {median_index}")
    # Extract corresponding completion time and trips
    corresponding_time = df_sorted.loc[median_index, 'total_completion_times']
    corresponding_trips = df_sorted.loc[median_index, 'total_trips']

    # Output the results
    print(f"Median Total Cost: {median_total_cost}")
    print(f"Corresponding Completion Time: {corresponding_time}")
    print(f"Corresponding Total Trips: {corresponding_trips}")

        
    # Plot the histograms
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.hist(total_costs, bins=20, edgecolor='black')
    plt.xlabel('Total Cost')
    plt.ylabel('Frequency')
    plt.title('Histogram of Total Cost')

    plt.subplot(1, 3, 2)
    plt.hist(total_completion_times, bins=20, edgecolor='black')
    plt.xlabel('Total Completion Time')
    plt.ylabel('Frequency')
    plt.title('Histogram of Total Completion Time')

    plt.subplot(1, 3, 3)
    plt.hist(total_trips_list, bins=20, edgecolor='black')
    plt.xlabel('Total Trips')
    plt.ylabel('Frequency')
    plt.title('Histogram of Total Trips')

    plt.tight_layout()
    plt.show()


    ##### BELOW IS THE CODE WHICH WE RETURN AS FINAL RESULT FOR THE UI when the option is selected as expected min cost #######

    # print the best optimal solnts
    processing_times = {}
    processing_times  = processing_times_tracker[median_index]
    #machines = ['VL1', 'VL2', 'VL3', 'VL4', 'VL5', 'VL6']  # Aircraft
    def generate_machine_names(num_vl, num_l):
        vl_names = [f'VL{i+1}' for i in range(num_vl)]
        l_names = [f'L{i+1+num_vl}' for i in range(num_l)]
        return vl_names + l_names

    # Generate machine names
    # num_vl = 3  # Number of VL machines
    # num_l = 3   # Number of L machines
    machines = generate_machine_names(num_vl, num_l)
    print(machines)

    def generate_distances(machines, distances_for_destinations, no_fly_conditions):
        additional_distances = {
            'D1': 252,
            'D2': 246,
            'D3': 57,
            'D4': 112
        }
        distances_dict = {}
        for machine in machines:
            for destination, base_distance in distances_for_destinations.items():
                if no_fly_conditions.get(destination, False):
                    adjusted_distance = base_distance + additional_distances[destination]
                else:
                    adjusted_distance = base_distance
                distances_dict[(machine, destination)] = adjusted_distance
        return distances_dict

    # Define base distances for each destination
    distances_for_destinations = {
        'D1': 5821.28,
        'D2': 3919.69,
        'D3': 3572.20,
        'D4': 716.47
    }

    # Define no-fly conditions for each destination
    no_fly_conditions = {
            'D1': checkbox1,
            'D2': checkbox2,
            'D3': checkbox3,
            'D4': checkbox4
    }
    distance_costs = generate_distances(machines, distances_for_destinations, no_fly_conditions)
    print(distance_costs)

    def generate_capacities(machines, vl_capacity, l_capacity):
        capacities = {}
        for machine in machines:
            if 'VL' in machine:
                capacities[machine] = vl_capacity
            elif 'L' in machine:
                capacities[machine] = l_capacity
        return capacities

    # Capacities for each type of machine
    vl_capacity = 100  # Capacity for VL machines
    l_capacity = 75    # Capacity for L machines

    # Generate the capacities dictionary
    capacity = generate_capacities(machines, vl_capacity, l_capacity)
    print(capacity)

    def generate_cost_factors(machines, vl_cost_factor, l_cost_factor):
        cost_factors = {}
        for machine in machines:
            if 'VL' in machine:
                cost_factors[machine] = vl_cost_factor
            elif 'L' in machine:
                cost_factors[machine] = l_cost_factor
        return cost_factors

    # Cost factors for each type of machine
    vl_cost_factor = 1    # Cost factor for VL machines
    l_cost_factor = 0.75  # Cost factor for L machines

    # Generate the cost factors dictionary
    cost_factors = generate_cost_factors(machines, vl_cost_factor, l_cost_factor)
    print(cost_factors)


    jobs = ['D1', 'D2', 'D3', 'D4']  # Demand locations

    job_quantities = {'D1': D1, 'D2': D2, 'D3': D3, 'D4': D4}  # Job quantities
    # Initialize the model
    m = Model("Job_Scheduling_with_Flexible_Objectives")

    # Variables
    x = m.addVars(machines, jobs, shifts, vtype=GRB.INTEGER, lb=0, name="x")
    t = m.addVars(machines, shifts, vtype=GRB.CONTINUOUS, lb=0, name="start_time")
    d = m.addVars(machines, shifts, vtype=GRB.CONTINUOUS, lb=0, name="duration")
    job_active = m.addVars(machines, jobs, shifts, vtype=GRB.BINARY, name="job_active")

    # Total cost variable
    total_cost = m.addVar(vtype=GRB.CONTINUOUS, lb=0, name="total_cost")

    # Total completion time variable
    total_completion_time = m.addVar(vtype=GRB.CONTINUOUS, lb=0, name="total_completion_time")

    # Objective functions
    # Choose objective
    m.setObjective(total_cost, GRB.MINIMIZE)  # Minimize total cost
    #m.setObjective(total_completion_time, GRB.MINIMIZE)  # Minimize total completion time

    # Constraints
    # Job quantity constraints
    # Job quantity constraints: Ensure that the total quantity of each job assigned to all machines and shifts equals the required job quantity

    for j in jobs:
        m.addConstr(quicksum(x[i, j, s] for i in machines for s in shifts) == job_quantities[j], name=f"job_qty_{j}")

    # Capacity constraints
    # Capacity constraints: Ensure that the total quantity of jobs assigned to each machine in each shift does not exceed the machine's capacity

    for i in machines:
        for s in shifts:
            m.addConstr(quicksum(x[i, j, s] for j in jobs) <= capacity[i], name=f"cap_{i}_{s}")

    # Linking job processing to binary activation
    # Capacity constraints: Ensure that the total quantity of jobs assigned to each machine in each shift does not exceed the machine's capacity

    for i in machines:
        for j in jobs:
            for s in shifts:
                m.addConstr(job_active[i, j, s] >= x[i, j, s] / capacity[i])
                m.addConstr(job_active[i, j, s] * capacity[i] >= x[i, j, s])

    # Duration and start times of shifts
    # Duration and start times of shifts: Calculate the duration of each machine in each shift based on the processing times of active jobs, and ensure that the completion time of each machine in each shift is less than or equal to the total completion time

    for i in machines:
        for s in shifts:
            m.addConstr(d[i, s] == quicksum(processing_times[(i, j)] * job_active[i, j, s] for j in jobs), name=f"dur_{i}_{s}")
            m.addConstr(t[i, s] + d[i, s] <= total_completion_time, name=f"completion_time_{i}_{s}")

    # Sequential shift start time constraints
    # Sequential shift start time constraints: Ensure that the start time of each machine in each shift is greater than or equal to the completion time of the previous shift

    for i in machines:
        for s in range(2, len(shifts) + 1):
            m.addConstr(t[i, s] >= t[i, s-1] + d[i, s-1], name=f"start_time_{i}_{s}")

    # Total cost constraint
    # Total cost constraint: Calculate the total cost of the schedule based on the distance costs, job activation, and cost factors
    m.addConstr(total_cost == quicksum(distance_costs[(i, j)] * job_active[i, j, s] * cost_factors[i] for i in machines for j in jobs for s in shifts), name="total_cost_constraint")

    # Solve the model
    m.optimize()
    # Output results
    # if m.status == GRB.OPTIMAL:
    #     print(f"Total cost: {total_cost.x:.2f}")
    #     print(f"Total completion time: {total_completion_time.x:.2f}")
    #     total_trips = 0
    #     for i in machines:
    #         active_shifts = [s for s in shifts if sum(job_active[i, j, s].x for j in jobs) > 0]
    #         total_shifts = len(active_shifts)
    #         total_trips += total_shifts
    #         print(f"Machine {i} used {total_shifts} shifts.")
    #         for s in active_shifts:
    #             duration = float(d[i, s].x)
    #             cost = float(sum(distance_costs[(i, j)] * job_active[i, j, s].x * cost_factors[i] for j in jobs))
    #             completion_time_shift = float(t[i, s].x + d[i, s].x)
    #             print(f"Machine {i}, Shift {s}, Start Time: {t[i, s].x:.2f}, Duration: {duration:.2f}, Cost: {cost:.2f}, Completion Time: {completion_time_shift:.2f}")
    #             for j in jobs:
    #                 if x[i, j, s].x > 0:
    #                     print(f"  Job {j}: {x[i, j, s].x} units")
    #     print(f"Total trips: {total_trips}")
        
    #     # Dictionary to store sets of machines assigned to each job
    #     job_machine_assignments = {job: set() for job in jobs}
        
    #     # Example of populating the dictionary based on hypothetical model outputs
    #     # For each machine, job, and shift, check if the machine worked on the job during any shift
    #     for i in machines:
    #         for j in jobs:
    #             for s in shifts:
    #                 # Hypothetically accessing a decision variable from Gurobi model, need real model data here
    #                 if x[i, j, s].x > 0:  # Assuming x[i, j, s].x indicates machine i worked on job j in shift s
    #                     job_machine_assignments[j].add(i)
        
    #     # Convert sets to lists to finalize the output
    #     for job, machines in job_machine_assignments.items():
    #         job_machine_assignments[job] = list(machines)
        
    #     # Printing the summary for each job
    #     print("Machine assignments per job:")
    #     for job, machines in job_machine_assignments.items():
    #         print(f"Job {job}: {machines}")
    if m.status == GRB.OPTIMAL:

        print(f"Total cost: {total_cost.x:.2f}")

        print(f"Total completion time: {total_completion_time.x:.2f}")

        total_trips = 0

        for i in machines:

            active_shifts = [s for s in shifts if sum(job_active[i, j, s].x for j in jobs) > 0]

            total_shifts = len(active_shifts)

            total_trips += total_shifts

            print(f"Machine {i} used {total_shifts} shifts.")

            for s in active_shifts:

                duration = float(d[i, s].x)

                cost = float(sum(distance_costs[(i, j)] * job_active[i, j, s].x * cost_factors[i] for j in jobs))

                completion_time_shift = float(t[i, s].x + d[i, s].x)

                print(f"Machine {i}, Shift {s}, Start Time: {t[i, s].x:.2f}, Duration: {duration:.2f}, Cost: {cost:.2f}, Completion Time: {completion_time_shift:.2f}")

                for j in jobs:

                    if x[i, j, s].x > 0:

                        print(f"  Job {j}: {x[i, j, s].x} units")

        print(f"Total trips: {total_trips}")

    

        # Dictionary to store machines and the number of shifts they worked for each job

        job_machine_assignments = {job: {} for job in jobs}

    

        # Populate the dictionary with machines and their shift counts

        for i in machines:

            for j in jobs:

                shift_count = sum(x[i, j, s].x > 0 for s in shifts)  # Total shifts machine i works on job j

                if shift_count > 0:

                    if i in job_machine_assignments[j]:

                        job_machine_assignments[j][i] += shift_count

                    else:

                        job_machine_assignments[j][i] = shift_count

    

        # Printing the summary for each job with machines and their shift counts

        print("Machine assignments per job with shift counts:")

        for job, machines in job_machine_assignments.items():

            formatted_machines = [(machine, shifts) for machine, shifts in machines.items()]

            print(f"Job {job}: {formatted_machines}")

        return {
            'job_machine_assignments': job_machine_assignments,
            'total_cost': total_cost.x,
            'total_completion_time': total_completion_time.x,
            'total_trips': total_trips,
            'total_trips_list': total_trips_list,
            'total_comp_list': total_completion_times
        }