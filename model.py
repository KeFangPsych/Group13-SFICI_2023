import numpy as np
import igraph as ig
import random

# Initial Parameters
W = 1
BETA = 10
number_of_nodes = 20
number_of_beliefs = 10
belief_levels = 6
time_steps = 100
number_of_connections = 10
seed = 7
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


def calculate_distance(person_index, weight):
    soc_dis = np.zeros(shape=(belief_levels,))
    internal_dis = np.zeros(shape=(belief_levels,))
    for i in range(belief_levels):
        temp = np.zeros((belief_levels,))
        temp[i] = 1
        soc_dis[i] = np.dot(temp - social_dissonances[person_index],
                              temp - social_dissonances[person_index])
        internal_dis[i] = np.dot(temp - internal_dissonances[person_index],
                                   temp - internal_dissonances[person_index])

    total_dissonance = soc_dis * weight + internal_dis * (1 - weight)
    return total_dissonance


def update_opinion(person_index, belief_index, total_dissonance):
    softmax = np.exp(-beta * total_dissonance) / sum(np.exp(-beta * total_dissonance))
    opinion_matrices[person_index, belief_index] = softmax
    return opinion_matrices


if __name__ == "__main__":

    # Construct Networks
    belief_network = ig.Graph.Erdos_Renyi(number_of_beliefs, number_of_connections/number_of_beliefs)
    social_network = ig.Graph.Erdos_Renyi(number_of_nodes, number_of_connections/number_of_nodes)

    # Assign initial opinions to each person randomly
    opinion_matrices = np.zeros(shape=(number_of_nodes, number_of_beliefs, belief_levels))

    # I am not sure what this step does
    for i in range(number_of_nodes):
        what_is_this = np.random.randint(1000, size=(number_of_beliefs, belief_levels))
        row_sums = what_is_this.sum(axis=1)
        opinion_matrices[i] = what_is_this / row_sums[:, np.newaxis]

    # W and Beta can be changed with time
    w = W
    beta = BETA
    temporal_opinions = np.zeros(shape=(time_steps, number_of_nodes, number_of_beliefs, belief_levels))

    # Iterate Agent Model over t time steps
    for t in range(time_steps):
        social_dissonances = np.zeros(shape=(number_of_nodes, belief_levels))
        internal_dissonances = np.zeros(shape=(number_of_nodes, belief_levels))

        # randomly choose focal belief for this time step
        focal_belief = np.random.randint(number_of_beliefs)

        for i in range(number_of_nodes):
            social_dissonances[i] = calculate_social_dissonance(i, focal_belief)
            internal_dissonances[i] = calculate_internal_dissonance(i, focal_belief)

        for i in range(number_of_nodes):
            total_dis = calculate_distance(i, focal_belief)
            opinion_matrices = update_opinion(i, focal_belief, total_dis)

        temporal_opinions[t] = opinion_matrices

    person_number = 19
    print(temporal_opinions[0, person_number] - temporal_opinions[1, person_number])
