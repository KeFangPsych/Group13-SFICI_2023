import numpy as np
import igraph as ig
import random
import matplotlib.pyplot as plt
from tqdm import tqdm
import matplotlib

# Initial Parameters
w_constant = 0.5
w_function = True
beta = 10
number_of_nodes = 1000
number_of_beliefs = 20
belief_levels = 10
time_steps = 600
social_connectivity = 0.01
belief_connectivity = 0.5
lower_bound = 1e-2
number_of_connections = int(number_of_nodes * 1/(1-social_connectivity))
constant_graph = True

belief_no = 1
# Set the random seed value
seed = 11
np.random.seed(seed)
random.seed(seed)


# Calculate the social dissonance
def calculate_social_dissonance(person_index, belief_index):
    # Find neighbors in social network
    neighbors = social_network.neighbors(person_index)
    # return column-means
    return np.mean(opinion_matrices[neighbors, belief_index], axis=0)


# Calculate internal dissonance
def calculate_internal_dissonance(person_index, belief_index):
    # Find neighbors in belief network
    neighbors = belief_network.neighbors(belief_index)
    # return column-means
    return np.mean(opinion_matrices[person_index, neighbors], axis=0)  # return column-means


def calculate_distance(person_index):
    soc_dis = np.zeros(shape=(belief_levels,))
    internal_dis = np.zeros(shape=(belief_levels,))
    for i in range(belief_levels):
        temp = np.zeros((belief_levels,))
        temp[i] = 1
        soc_dis[i] = np.dot(temp - social_dissonances[person_index],
                              temp - social_dissonances[person_index])
        internal_dis[i] = np.dot(temp - internal_dissonances[person_index],
                                   temp - internal_dissonances[person_index])
    num_edges = len(social_network.neighbors(person_index))
    # record w value - to check what happens to it over time and between people
    # w = 1 - 1/num_edges
    neighbors = social_network.neighbors(person_index)
    beliefs = belief_network.neighbors(focal_belief)
    if len(neighbors) <= 1 or len(beliefs) <= 1 or not w_function:
        w = w_constant
    else:
        social_belief_array = opinion_matrices[neighbors, focal_belief].T
        social_var = (np.cov(social_belief_array) * np.eye(social_belief_array.shape[0])).sum()
        internal_belief_array = opinion_matrices[person_index, beliefs].T
        internal_var = (np.cov(internal_belief_array) * np.eye(internal_belief_array.shape[0])).sum()
        w = (internal_var+lower_bound)/(internal_var+social_var+lower_bound/w_constant)
    for i in range(number_of_beliefs):
        if i == focal_belief:
            ws[t, person_index, i] = w
        elif t == 0:
            ws[t, person_index, i] = w_constant
        else:
            ws[t, person_index, i] = ws[t-1, person_index, i]
    total_dissonance = soc_dis * w + internal_dis * (1 - w)
    return total_dissonance


def update_opinion(person_index, belief_index, total_dissonance):
    softmax = np.exp(-beta * total_dissonance) / sum(np.exp(-beta * total_dissonance))
    opinion_matrices[person_index, belief_index] = softmax
    return opinion_matrices


if __name__ == "__main__":

    # Construct Networks
    if constant_graph:
        belief_network = ig.Graph.Erdos_Renyi(number_of_beliefs, belief_connectivity)
        social_network = ig.Graph.Erdos_Renyi(number_of_nodes, social_connectivity)
    else:
        # belief_network = ig.Graph.Static_Power_Law(number_of_nodes, m=10, exponent_out=2.5)
        belief_network = ig.Graph.Erdos_Renyi(number_of_beliefs, belief_connectivity)
        social_network = ig.Graph.Static_Power_Law(number_of_nodes, m=number_of_connections, exponent_out=2.5)
    # Assign initial opinions to each person randomly
    opinion_matrices = np.zeros(shape=(number_of_nodes, number_of_beliefs, belief_levels))

    # I am not sure what this step does
    for i in range(number_of_nodes):
        what_is_this = np.random.randint(1000, size=(number_of_beliefs, belief_levels))
        row_sums = what_is_this.sum(axis=1)
        opinion_matrices[i] = what_is_this / row_sums[:, np.newaxis]

    temporal_opinions = np.zeros(shape=(time_steps, number_of_nodes, number_of_beliefs, belief_levels))
    ws = np.zeros(shape=(time_steps, number_of_nodes, number_of_beliefs))
    var = np.zeros(shape=(time_steps,number_of_beliefs))

    # Iterate Agent Model over t time steps
    for t in tqdm(range(time_steps)):

        social_dissonances = np.zeros(shape=(number_of_nodes, belief_levels))
        internal_dissonances = np.zeros(shape=(number_of_nodes, belief_levels))

        # randomly choose focal belief for this time step
        focal_belief = np.random.randint(number_of_beliefs)

        for i in range(number_of_nodes):
            social_dissonances[i] = calculate_social_dissonance(i, focal_belief)
            internal_dissonances[i] = calculate_internal_dissonance(i, focal_belief)

        for i in range(number_of_nodes):
            total_dis = calculate_distance(i)
            opinion_matrices = update_opinion(i, focal_belief, total_dis)

        temporal_opinions[t] = opinion_matrices
        for i in range(number_of_beliefs):
            var[t, i] = np.var(np.argmax(temporal_opinions[t, :, :, :], axis=2))

    # person_no = 6
    # print(np.argmax(temporal_opinions[-1, :, belief_no, :], axis=1))
    # print(np.argmax(temporal_opinions[:, person_no, belief_no, :], axis=1))

    for i in range(1):
        for j in range(number_of_beliefs):
            plt.plot(ws[:, i, j])
        plt.show()

    fig, ax = plt.subplots()

    for i in range(number_of_nodes):
        ax.plot(np.argmax(temporal_opinions[:, i, belief_no, :], axis=1))
    ax.set_xlabel("Timesteps")
    ax.set_ylabel("Belief levels")
    ax.set_title(f"Belief values of social network over time")
    fig.suptitle(f'w_constant={w_constant}, beta={beta}, w_function={w_function}, constant_graph={constant_graph}')
    plt.show()

    fig, ax2 = plt.subplots()
    plt.hist(np.argmax(temporal_opinions[-1, :, belief_no, :], axis=1))
    plt.show()

    for i in range(number_of_beliefs):
        plt.plot(var[:, i])
    plt.show()

    for i in range(20):
        print(np.argmax(temporal_opinions[-1, i, :, :], axis=1))
    plt.imshow(np.argmax(temporal_opinions[-1, :20, :, :], axis=2), interpolation='nearest')
    plt.show()

    # Visualizing Graph
    cmap = matplotlib.cm.get_cmap('Spectral')
    belief = 0
    vertex_colors = []
    for i in range(number_of_nodes):
        level = np.argmax(temporal_opinions[-1, i, belief],axis=0)
        vertex_colors.append(cmap(level / belief_levels))

    social_network.vs["color"] = vertex_colors

    fig, ax = plt.subplots()
    ig.plot(social_network, target=ax, bbox=(600, 600), vertex_size=30, edge_width=0.01)
    plt.show()

    print(ws)


