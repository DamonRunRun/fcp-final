import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

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

	def get_mean_degree(self):
		#Your code  for task 3 goes here

	def get_mean_clustering(self):
		#Your code for task 3 goes here

	def get_mean_path_length(self):
		#Your code for task 3 goes here

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
			for neighbour_index in range(index+1, N):
				if np.random.random() < connection_probability:
					node.connections[neighbour_index] = 1
					self.nodes[neighbour_index].connections[index] = 1

	def make_ring_network(self, N, neighbour_range=1):
		#Your code  for task 4 goes here

	def make_small_world_network(self, N, re_wire_prob=0.2):
		#Your code for task 4 goes here

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
	#You should write some code for handling flags here

if __name__=="__main__":
	main()
