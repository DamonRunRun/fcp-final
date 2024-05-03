import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import math
import argparse
import copy
from collections import deque
import random


class Node:

    def __init__(self, value, number, connections=None):
        self.index = number
        self.connections = connections
        self.value = value


class Network:

    def __init__(self, nodes=None):
        if nodes is None:
            self.nodes = []
        else:
            self.nodes = nodes

    def _total_connections(self):
        # Calculate the total number of connections in the network
        return sum(sum(node.connections) for node in self.nodes)

    def get_mean_degree(self):
        count = self._total_connections()
        if len(self.nodes) == 0:
            return 0
        # Calculate the average number of connections per node
        return count / len(self.nodes)

    # Your code  for task 3 goes here

    def get_mean_path_length(self):
        n = len(self.nodes)
        if n == 0:
            return 0
        connection_metric = self._create_connection_metric()
        distance_matrix = self._calculate_distance_matrix(connection_metric)
        # Calculate and return the average path length across all pairs of nodes
        return self._average_path_length(distance_matrix)

    # Your code  for task 3 goes here

    def _create_connection_metric(self):
        # Create a matrix that represents node connections
        n = len(self.nodes)
        connection_metric = [[] for _ in range(n)]
        for node in self.nodes:
            connection_metric[node.index] = copy.copy(node.connections)
        return np.array(connection_metric)

    def _calculate_distance_matrix(self, connection_metric):
        # Calculate the distance matrix using the connection matrix
        n = len(connection_metric)
        distance_matrix = np.zeros((n, n), dtype=int)
        # Calculate distances using Breadth-First Search (BFS) for each node
        for i in range(n):
            distance_matrix[i] = self._bfs_distance(connection_metric, i)
        return distance_matrix

    def _bfs_distance(self, graph, start_node):
        N = len(graph)
        # Initialize distances array with -1 (indicating unvisited nodes)
        distances = [-1] * N
        distances[start_node] = 0
        # Use a queue to manage the BFS process
        queue = deque([start_node])
        # Process the queue until empty
        while queue:
            current = queue.popleft()
            current_distance = distances[current]
            # Check all possible neighbors
            for neighbor in range(N):
                # If there is a connection and neighbor hasn't been visited
                if graph[current][neighbor] == 1 and distances[neighbor] == -1:
                    distances[neighbor] = current_distance + 1
                    queue.append(neighbor)
        return distances

    def _average_path_length(self, distance_matrix):
        n = len(distance_matrix)
        count_metric = []
        # Calculate the average path length for each node
        for i in range(n):
            positive_numbers = [num for num in distance_matrix[i] if num > 0]
            count = len(positive_numbers)
            total_sum = sum(positive_numbers)
            if count == 0:
                count_metric.append(0)
            else:
                count_metric.append(total_sum / count)
        # Calculate the overall average path length
        return sum(count_metric) / n if count_metric else 0

    def get_mean_clustering(self):
        n = len(self.nodes)
        if n == 0:
            return 0
        connection_metric = self._create_connection_metric()
        # Calculate and return the average clustering coefficient
        return self._calculate_clustering_coefficient(connection_metric)

    def _calculate_clustering_coefficient(self, connection_metric):
        n = len(connection_metric)
        count_metric = []
        # Calculate the clustering coefficient for each node
        for i in range(n):
            if sum(connection_metric[i]) == 0:
                count_metric.append(0)
            else:
                lines = sum(connection_metric[i]) * (sum(connection_metric[i]) - 1) / 2
                positive_indices = np.where(connection_metric[i, :] > 0)[0]
                new_matrix = connection_metric[np.ix_(positive_indices, positive_indices)]
                if lines == 0:
                    count_metric.append(0)
                else:
                    count_metric.append(np.sum(new_matrix) / 2 / lines)

        return sum(count_metric) / n if count_metric else 0

    # Your code for task 3 goes here

    def make_random_network(self, N, connection_probability):
        '''
        This function makes a *random* network of size N.
        Each node is connected to each other node with probability p
        '''

        self.nodes = []
        for node_number in range(N):
            value = np.random.random()
            connections = [0 for _ in range(N)]
            self.nodes.append(Node(value, node_number, connections))

        for (index, node) in enumerate(self.nodes):
            for neighbour_index in range(index + 1, N):
                if np.random.random() < connection_probability:
                    node.connections[neighbour_index] = 1
                    self.nodes[neighbour_index].connections[index] = 1

    def make_ring_network(self, N, neighbour_range=1):
        self.nodes = []
        num_nodes = N
        # Loop to create each node in the network
        for node_number in range(num_nodes):
            connections = [0 for val in range(num_nodes)]
            # Set connections for each node within the specified neighbour range
            for n in range(1, neighbour_range + 1):
                connections[(node_number - n) % num_nodes] = 1
                connections[(node_number + n) % num_nodes] = 1
            new_node = Node(np.random.random(), node_number, connections=connections)
            self.nodes.append(new_node)

    # Your code  for task 4 goes here

    def make_small_world_network(self, N, re_wire_prob=0.2):
        self.nodes = []
        num_nodes = N
        # Loop to create each node in the network
        for node_number in range(num_nodes):
            connections = [0 for val in range(num_nodes)]
            # Generate random probabilities and destination indices for potential re-wiring
            prob = np.random.random(4)
            des = np.random.randint(0, N, (4))
            bias = [-1, -2, 1, 2]
            # Loop to establish connections based on re-wiring probability
            for i in range(4):
                if prob[i] >= re_wire_prob:
                    connections[(node_number + bias[i]) % num_nodes] = 1
                elif des[i] != i:
                    connections[des[i]] = 1
            # Create a new node instance with a random opinion value, its index, and its connections
            new_node = Node(np.random.random(), node_number, connections=connections)
            self.nodes.append(new_node)

    # Your code for task 4 goes here

    def plot(self):

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_axis_off()

        num_nodes = len(self.nodes)
        network_radius = num_nodes * 10
        ax.set_xlim([-1.1*network_radius, 1.1*network_radius])
        ax.set_ylim([-1.1*network_radius, 1.1*network_radius])

        for (i, node) in enumerate(self.nodes):
            node_angle = i * 2 * np.pi / num_nodes
            node_x = network_radius * np.cos(node_angle)
            node_y = network_radius * np.sin(node_angle)

            circle = plt.Circle((node_x, node_y), 0.3*num_nodes, color=cm.hot(node.value))
            ax.add_patch(circle)

            for neighbour_index in range(i+1, num_nodes):
                if node.connections[neighbour_index]:
                    neighbour_angle = neighbour_index * 2 * np.pi / num_nodes
                    neighbour_x = network_radius * np.cos(neighbour_angle)
                    neighbour_y = network_radius * np.sin(neighbour_angle)

                    ax.plot((node_x, neighbour_x), (node_y, neighbour_y), color='black')

def test_networks():

    #Ring network
    nodes = []
    num_nodes = 10
    for node_number in range(num_nodes):
        connections = [0 for val in range(num_nodes)]
        connections[(node_number-1)%num_nodes] = 1
        connections[(node_number+1)%num_nodes] = 1
        new_node = Node(0, node_number, connections=connections)
        nodes.append(new_node)
    network = Network(nodes)

    print("Testing ring network")
    assert(network.get_mean_degree()==2), network.get_mean_degree()
    assert(network.get_clustering()==0), network.get_clustering()
    assert(network.get_path_length()==2.777777777777778), network.get_path_length()

    nodes = []
    num_nodes = 10
    for node_number in range(num_nodes):
        connections = [0 for val in range(num_nodes)]
        connections[(node_number+1)%num_nodes] = 1
        new_node = Node(0, node_number, connections=connections)
        nodes.append(new_node)
    network = Network(nodes)

    print("Testing one-sided network")
    assert(network.get_mean_degree()==1), network.get_mean_degree()
    assert(network.get_clustering()==0),  network.get_clustering()
    assert(network.get_path_length()==5), network.get_path_length()

    nodes = []
    num_nodes = 10
    for node_number in range(num_nodes):
        connections = [1 for val in range(num_nodes)]
        connections[node_number] = 0
        new_node = Node(0, node_number, connections=connections)
        nodes.append(new_node)
    network = Network(nodes)

    print("Testing fully connected network")
    assert(network.get_mean_degree()==num_nodes-1), network.get_mean_degree()
    assert(network.get_clustering()==1),  network.get_clustering()
    assert(network.get_path_length()==1), network.get_path_length()

    print("All tests passed")

'''
==============================================================================================================
This section contains code for the Ising Model - task 1 in the assignment
==============================================================================================================
'''

def calculate_agreement(population, row, col, external=0.0):
    '''
    This function should return the *change* in agreement that would result if the cell at (row, col) was to flip it's value
    Inputs: population (numpy array)
            row (int)
            col (int)
            external (float)
    Returns:
            change_in_agreement (float)
    '''

    # Your code for task 1 goes here
    # Determine the number of rows and columns in the population array
    n_rows, n_cols = population.shape
    # Initialize a counter to accumulate the agreement value
    count = 0
    # Check the cell above, below, left and right, if it exists, and calculate the product with the current cell
    if row >= 1:
        count += population[row, col] * population[row - 1, col]
    if col >= 1:
        count += population[row, col] * population[row, col - 1]
    if row < n_rows - 1:
        count += population[row, col] * population[row + 1, col]
    if col < n_cols - 1:
        count += population[row, col] * population[row, col + 1]
    count += population[row, col] * external
    return count



def ising_step(population, external=0.0, alpha=1):
    '''
    This function will perform a single update of the Ising model
    Inputs: population (numpy array)
            external (float) - optional - the magnitude of any external "pull" on opinion
    '''

    n_rows, n_cols = population.shape
    row = np.random.randint(0, n_rows)
    col = np.random.randint(0, n_cols)

    agreement = calculate_agreement(population, row, col, external)
    probability = math.exp(-agreement / alpha)
    # Determine if the cell should flip its state
    if (agreement < 0) or (np.random.rand() < probability):
        population[row, col] *= -1
    return population
    # Your code for task 1 goes here


def plot_ising(im, population):
    '''
    This function will display a plot of the Ising model
    '''

    new_im = np.array([[255 if val == -1 else 1 for val in rows] for rows in population], dtype=np.int8)
    im.set_data(new_im)
    plt.pause(0.1)

def test_ising():
    '''
    This function will test the calculate_agreement function in the Ising model
    '''

    print("Testing ising model calculations")
    population = -np.ones((3, 3))
    assert(calculate_agreement(population,1,1)==4), "Test 1"

    population[1, 1] = 1.
    assert(calculate_agreement(population,1,1)==-4), "Test 2"

    population[0, 1] = 1.
    assert(calculate_agreement(population,1,1)==-2), "Test 3"

    population[1, 0] = 1.
    assert(calculate_agreement(population,1,1)==0), "Test 4"

    population[2, 1] = 1.
    assert(calculate_agreement(population,1,1)==2), "Test 5"

    population[1, 2] = 1.
    assert(calculate_agreement(population,1,1)==4), "Test 6"

    "Testing external pull"
    population = -np.ones((3, 3))
    assert(calculate_agreement(population,1,1,1)==3), "Test 7"
    assert(calculate_agreement(population,1,1,-1)==5), "Test 8"
    assert(calculate_agreement(population,1,1,-10)==14), "Test 9"
    assert(calculate_agreement(population,1,1,10)==-6), "Test 10"

    print("Tests passed")


def ising_main(population, alpha=None, external=0.0):
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_axis_off()
    im = ax.imshow(population, interpolation='none', cmap='RdPu_r')

    # Iterating an update 100 times
    for frame in range(100):
        # Iterating single steps 1000 times to form an update
        for step in range(1000):
            ising_step(population, external)
        print('Step:', frame, end='\r')
        plot_ising(im, population)

def calculate_agreement_net(node, nodes, external):
    count = 0
    for i, connected in enumerate(node.connections):
        if connected:
            count += node.value * nodes[i].value
    count = node.value * external
    return count


def ising_step_net(network, alpha=1, external=0.0):
    node = network.nodes[np.random.randint(len(network.nodes))]
    agreement = calculate_agreement_net(node, network.nodes, external)
    probability = math.exp(-agreement / alpha)
    if (agreement < 0) or (np.random.rand() < probability):
        node.value *= -1


def ising_main_net(network, alpha, external):
    ax = plt.figure().add_subplot()
    epoch = 100
    each_epoch_iters = 1000
    for frame in range(epoch):
        for step in range(each_epoch_iters):
            ising_step_net(network, alpha, external)
        ax.clear()
        ax.set_axis_off()
        network.plot_net(ax)
        plt.pause(0.1)
    plt.show()
'''
==============================================================================================================
This section contains code for the Defuant Model - task 2 in the assignment
==============================================================================================================
'''
def calculate_defuant_agreement(population, i, j, beta=0.2):
    temp = population[i] # Store the current value of the ith individual
    # Update the value of the ith and jth individual
    population[i] += beta * (population[j] - population[i])
    population[j] += beta * (temp - population[j])
    return population

def defuant_step(population, beta=0.2, threshold=0.2):
    # Randomly select two different indices from the population array
    i, j = np.random.choice(len(population), 2, replace=False)
    # Check if the difference between the opinions of the two individuals is less than the threshold
    if abs(population[i] - population[j]) < threshold:
        population = calculate_defuant_agreement(population, i, j, beta)
    return population

def defuant_main(population, beta=0.2, threshold=0.2):
    # Create a figure with two subplots for displaying the histogram and scatter plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    plt.suptitle(f"Coupling: {beta}, Threshold: {threshold}", fontsize=16)
    x_values = []
    y_values = []
    for frame in range(100):
        for step in range(8):
            population = defuant_step(population, beta, threshold)
        print('Step:', frame, end='\r')
        # Plot the current state of the population using the plot_defuant function
        plot_defuant(ax1, ax2, frame, population, x_values, y_values, 100)
    #Your code for task 2 goes here

def plot_defuant(ax1, ax2, frame, population, x_values, y_values, epoch):
    # Generate visualization images
    bin_size = 0.05
    bins = np.arange(0, 1 + bin_size, bin_size)
    hist, bin_edges = np.histogram(population, bins=bins)
    ax1.cla()  # Clear any existing content in the axis
    ax1.bar(bin_edges[:-1], hist, width=bin_size)
    ax1.set_xticks(np.arange(0, 1.1, 0.2))
    ax1.set_xlabel('Opinion')

    ax2.cla()
    x_values.extend([frame] * len(population))
    y_values.extend(population)
    ax2.scatter(x_values, y_values, alpha=0.6, color='red')
    ax2.set_xlim(0, epoch)
    ax2.set_ylim(0, 1)
    ax2.set_ylabel('Opinion')

    plt.pause(0.1)

def test_defuant():
    '''
        This function will test the defuant function in the defuant model
    '''

    print("Testing defuant calculations")
    population = [1, -1, -1]
    calculate_defuant_agreement(population, 0, 2, 1)
    assert (population[0] == -1), "Test 1"
    assert (population[2] == 1), "Test 1"

    calculate_defuant_agreement(population, 0, 2, 0)
    assert (population[0] == -1), "Test 2"
    assert (population[2] == 1), "Test 2"

    print("Testing defuant step")
    population = [1, 0, -1]
    defuant_step(population, beta=1, threshold=0)
    assert (population[0] == 1), "Test 3"
    assert (population[1] == 0), "Test 3"
    assert (population[2] == -1), "Test 3"

    population = [1, -1]
    defuant_step(population, beta=1, threshold=2.1)
    assert (population[0] == -1), "Test 4"
    assert (population[1] == 1), "Test 4"
    print("Tests passed")
    # Your code for task 2 goes here


'''
==============================================================================================================
This section contains code for the main function- you should write some code for handling flags here
==============================================================================================================
'''

def main():
    parser = argparse.ArgumentParser(description='The Test')
    parser.add_argument('-ising_model', action='store_true')
    parser.add_argument('-external', type=float, default=0)
    parser.add_argument('-alpha', type=float, default=1.0)
    parser.add_argument('-test_ising', action='store_true')

    parser.add_argument('-defuant', action='store_true')
    parser.add_argument('-beta', type=float, default=0.2)
    parser.add_argument('-threshold', type=float, default=0.2)
    parser.add_argument('-test_defuant', action='store_true')

    parser.add_argument('-network', type=int, default=0)
    parser.add_argument('-test_network', action='store_true')

    parser.add_argument('-ring_network', type=int, default=0)
    parser.add_argument('-small_world', type=int, default=0)
    parser.add_argument('-re_wire', type=float, default=0.2)

    parser.add_argument('-use_network', type=int, default=0)
    args = parser.parse_args()

    if args.ising_model:
        if args.use_network:
            network = Network()
            network.make_small_world_network(args.use_network, 0.2)
            ising_main_net(network, args.alpha, args.external)
        else:
            population = np.random.choice([-1, 1], size=(100, 100))
            ising_main(population, args.alpha, args.external)
            # plt.show()

    if args.test_ising:
        test_ising()

    if args.defuant:
        if args.use_network:
            network = Network()
            network.make_small_world_network(args.use_network, 0.2)
            defuant_main_net(network, args.alpha, args.threshold)
        else:
            population = np.random.rand(100)
            defuant_main(population, args.beta, args.threshold)
            plt.show()

    if args.test_defuant:
        population = np.random.rand(100)
        test_defuant(population, args.beta, args.threshold)

    if args.network:
        network = Network()
        network.make_random_network(args.network, 0.2)
        network.plot()
        plt.show()
        print(f"Mean degree: {network.get_mean_degree()}")
        print(f"Average path length: {network.get_mean_path_length()}")
        print(f"Clustering co-efficient: {network.get_mean_clustering()}")

    if args.test_network:
        test_networks()

    if args.ring_network:
        network = Network()
        network.make_ring_network(args.ring_network)
        network.plot()
        plt.show()

    if args.small_world:
        network = Network()
        network.make_small_world_network(args.small_world, args.re_wire)
        network.plot()
        plt.show()
    #You should write some code for handling flags here

if __name__=="__main__":
    main()
