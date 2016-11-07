


"""Testing pbnt. Run this before anything else to get pbnt to work!"""
import sys
if('pbnt/combined' not in sys.path):
    sys.path.append('pbnt/combined')
from exampleinference import inferenceExample

inferenceExample()
# Should output:
# ('The marginal probability of sprinkler=false:', 0.80102921)
#('The marginal probability of wetgrass=false | cloudy=False, rain=True:', 0.055)

'''
WRITE YOUR CODE BELOW. DO NOT CHANGE ANY FUNCTION HEADERS FROM THE NOTEBOOK.
'''

# Part 1: Bayesian network tutorial
# -------
# 40 points total
#
# To start, design a basic probabilistic model for the following system:
#
# There's a nuclear power plant in which an alarm is supposed to ring when the core temperature, indicated by a gauge, exceeds a fixed threshold. For simplicity, we assume that the temperature is represented as either high or normal. However, the alarm is sometimes faulty, and the gauge is more likely to fail when the temperature is high. Use the following Boolean variables in your implementation:
#
# - A = alarm sounds
# - F<sub>A</sub> = alarm is faulty
# - G = gauge reading (high = True, normal = False)
# - F<sub>G</sub> = gauge is faulty
# - T = actual temperature (high = True, normal = False)
#
# You will test your implementation at the end of the section.

# 1a: Casting the net
# --
# 10 points
#
# Design a Bayesian network for this system, using pbnt to represent the nodes and conditional probability arcs connecting nodes. Don't worry about the probabilities for now. Fill out the function below to create the net.
#
# The following command will create a BayesNode with 2 values, an id of 0 and the name "alarm":
#
#     A = BayesNode(0,2,name='alarm')
#
# NOTE: Do not use any special characters(like $,_,-) for the name parameter, spaces are ok.
#
# You will use BayesNode.add\_parent() and BayesNode.add\_child() to connect nodes. For example, to connect the alarm and temperature nodes that you've already made (i.e. assuming that temperature affects the alarm probability):
#
#     A.add_parent(T_node)
#     T_node.add_child(A)
#
# You can run probability\_tests.network\_setup\_test() to make sure your network is set up correctly.

#        Hint : Checkout ExampleModels.py under pbnt/combined

# In[ ]:

from Node import BayesNode
from Graph import BayesNet



#
def make_power_plant_net():
    """Create a Bayes Net representation of the above power plant problem.
    Name the nodes as "alarm","faulty alarm", "gauge","faulty gauge", "temperature".
    """
    nodes = []
    # TODO: finish this function

    #create nodes ---> refer node.py
    A = BayesNode(0, 2, name="alarm")
    F_A = BayesNode(1, 2, name="faulty alarm")
    G_node = BayesNode(2, 2, name="gauge")
    F_G_node = BayesNode(3, 2, name="faulty gauge")
    T_node = BayesNode(4, 2, name="temperature")

    #alarm
    A.add_parent(G_node)
    A.add_parent(F_A)

    #faulty alarm
    F_A.add_child(A)

    #gauge
    G_node.add_parent(T_node)
    G_node.add_parent(F_G_node)
    G_node.add_child(A)

    #faulty gauge
    F_G_node.add_parent(T_node)
    F_G_node.add_child(G_node)

    #temperature
    T_node.add_child(G_node)
    T_node.add_child(F_G_node)

    nodes = [A, F_A, G_node, F_G_node, T_node]

    return BayesNet(nodes)


# In[ ]:

from probability_tests import network_setup_test
power_plant = make_power_plant_net()
network_setup_test(power_plant)


# 1b: Polytrees
# --
# 5 points
#
# Is the network for the power plant system a polytree? Why or why not? Choose from the following answers.

# In[ ]:

def is_polytree():
    """Multiple choice question about polytrees."""

    # TODO: make a choice!
    choice = 'c'
    answers = {
        'a' : 'Yes, because it can be decomposed into multiple sub-trees.',
        'b' : 'Yes, because its underlying undirected graph is a tree.',
        'c' : 'No, because its underlying undirected graph is not a tree.',
        'd' : 'No, because it cannot be decomposed into multiple sub-trees.'
    }
    return answers[choice]


# 1c: Setting the probabilities
# ---
# 10 points
#
# Assume that the following statements about the system are true:
#
# 1. The temperature gauge reads the correct temperature with 95% probability when it is not faulty and 20% probability when it is faulty. For simplicity, say that the gauge's "true" value corresponds with its "hot" reading and "false" with its "normal" reading, so the gauge would have a 95% chance of returning "true" when the temperature is hot and it is not faulty.
# 2. The alarm is faulty 15% of the time.
# 3. The temperature is hot (call this "true") 20% of the time.
# 4. When the temperature is hot, the gauge is faulty 80% of the time. Otherwise, the gauge is faulty 5% of the time.
# 5. The alarm responds correctly to the gauge 55% of the time when the alarm is faulty, and it responds correctly to the gauge 90% of the time when the alarm is not faulty. For instance, when it is faulty, the alarm sounds 55% of the time that the gauge is "hot" and remains silent 55% of the time that the gauge is "normal."
#
# Knowing these facts, set the conditional probabilities for the necessary variables on the network you just built.
#
# Using pbnt's Distribution class: if you wanted to set the distribution for P(A) to 70% true, 30% false, you would invoke the following commands.
#
#     A_distribution = DiscreteDistribution(A)
#     index = A_distribution.generate_index([],[])
#     A_distribution[index] = [0.3,0.7]
#     A.set_dist(A_distribution)
#
# If you wanted to set the distribution for P(A|G) to be
#
# |$G$|$P(A=true| G)$|
# |------|-----|
# |T| 0.75|
# |F| 0.85|
#
# you would invoke:
#
#     from numpy import zeros, float32
#     dist = zeros([G_node.size(), A.size()], dtype=float32)
#     dist[0,:] = [0.15, 0.85]
#     dist[1,:] = [0.25, 0.75]
#     A_distribution = ConditionalDiscreteDistribution(nodes=[G_node,A], table=dist)
#     A.set_dist(A_distribution)
#
# Modeling a three-variable relationship is a bit trickier. If you wanted to set the following distribution for $P(A|G,T)$ to be
#
# |$G$|$T$|$P(A=true| G, T)$|
# |--|--|:----:|
# |T|T|0.15|
# |T|F|0.6|
# |F|T|0.2|
# |F|F|0.1|
#
# you would invoke:
#
#     from numpy import zeros, float32
#     dist = zeros([G_node.size(), T_node.size(), A.size()], dtype=float32)
#     dist[1,1,:] = [0.85, 0.15]
#     dist[1,0,:] = [0.4, 0.6]
#     dist[0,1,:] = [0.8, 0.2]
#     dist[0,0,:] = [0.9, 0.1]
#     A_distribution = ConditionalDiscreteDistribution(nodes=[G_node, T_node, A], table=dist)
#     A.set_dist(A_distribution)
#
# The key is to remember that 0 represents the index of the false probability, and 1 represents true.
#
# You can check your probability distributions with probability\_tests.probability\_setup\_test().

#        Hint : Checkout example_inference.py under pbnt/combined

# In[ ]:

from numpy import zeros, float32
import Distribution
from Distribution import DiscreteDistribution, ConditionalDiscreteDistribution
def set_probability(bayes_net):
    """Set probability distribution for each node in the power plant system."""

    A = bayes_net.get_node_by_name("alarm")
    F_A = bayes_net.get_node_by_name("faulty alarm")
    G_node = bayes_net.get_node_by_name("gauge")
    F_G_node = bayes_net.get_node_by_name("faulty gauge")
    T_node = bayes_net.get_node_by_name("temperature")
    nodes = [A, F_A, G_node, F_G_node, T_node]
    # TODO: set the probability distribution for each node

    # Gauge reads the correct temperature with 95% probability when it is not faulty and 20% probability when it is faulty
    dist = zeros([T_node.size(), F_G_node.size(), G_node.size()], dtype=float32)
    dist[1,1,:] = [0.8, 0.2]
    dist[1,0,:] = [0.05, 0.95]
    dist[0,1,:] = [0.2, 0.8]
    dist[0,0,:] = [0.95, 0.05]
    G_distribution = ConditionalDiscreteDistribution(nodes=[T_node, F_G_node, G_node], table=dist)
    G_node.set_dist(G_distribution)

    # Alarm is faulty 15% of the time
    F_A_distribution = DiscreteDistribution(F_A)
    index = F_A_distribution.generate_index([],[])
    F_A_distribution[index] = [0.85,0.15]
    F_A.set_dist(F_A_distribution)

    # Temperature is hot (call this "true") 20% of the time
    T_distribution = DiscreteDistribution(T_node)
    index = T_distribution.generate_index([],[])
    T_distribution[index] = [0.8,0.2]
    T_node.set_dist(T_distribution)

    # When temp is hot, the gauge is faulty 80% of the time. Otherwise, the gauge is faulty 5% of the time
    dist = zeros([T_node.size(), F_G_node.size()], dtype=float32)
    dist[0,:] = [0.95, 0.05]
    dist[1,:] = [0.2, 0.8]
    F_G_distribution = ConditionalDiscreteDistribution(nodes=[T_node,F_G_node], table=dist)
    F_G_node.set_dist(F_G_distribution)

    # Alarm responds correctly to the gauge 55% of the time when the alarm is faulty,
    # and it responds correctly to the gauge 90% of the time when the alarm is not faulty.
    dist = zeros([G_node.size(), F_A.size(), A.size()], dtype=float32)
    dist[1,1,:] = [0.45, 0.55]
    dist[1,0,:] = [0.1, 0.9]
    dist[0,1,:] = [0.55, 0.45]
    dist[0,0,:] = [0.9, 0.1]
    A_distribution = ConditionalDiscreteDistribution(nodes=[G_node, F_A, A], table=dist)
    A.set_dist(A_distribution)

    return bayes_net


# In[ ]:

set_probability(power_plant)
from probability_tests import probability_setup_test
probability_setup_test(power_plant)


# 1d: Probability calculations : Perform inference
# ---
# 15 points
#
# To finish up, you're going to perform inference on the network to calculate the following probabilities:
#
# - the marginal probability that the alarm sounds
# - the marginal probability that the gauge shows "hot"
# - the probability that the temperature is actually hot, given that the alarm sounds and the alarm and gauge are both working
#
# You'll fill out the "get_prob" functions to calculate the probabilities.
#
# Here's an example of how to do inference for the marginal probability of the "faulty alarm" node being True (assuming "bayes_net" is your network):
#
#     F_A = bayes_net.get_node_by_name('faulty alarm')
#     engine = JunctionTreeEngine(bayes_net)
#     Q = engine.marginal(F_A)[0]
#     index = Q.generate_index([True],range(Q.nDims))
#     prob = Q[index]
#
# To compute the conditional probability, set the evidence variables before computing the marginal as seen below (here we're computing $P(A = false | F_A = true, T = False)$):
#
#     engine.evidence[F_A] = True
#     engine.evidence[T_node] = False
#     Q = engine.marginal(A)[0]
#     index = Q.generate_index([False],range(Q.nDims))
#     prob = Q[index]
#
# If you need to sanity-check to make sure you're doing inference correctly, you can run inference on one of the probabilities that we gave you in 1c. For instance, running inference on $P(T=true)$ should return 0.19999994 (i.e. almost 20%). You can also calculate the answers by hand to double-check.

# In[ ]:

def get_alarm_prob(bayes_net, alarm_rings):
    """Calculate the marginal
    probability of the alarm
    ringing (T/F) in the
    power plant system."""
    # TODO: finish this function
    A = bayes_net.get_node_by_name('alarm')
    engine = JunctionTreeEngine(bayes_net)
    Q = engine.marginal(A)[0]
    index = Q.generate_index([alarm_rings],range(Q.nDims))
    alarm_prob = Q[index]

    return alarm_prob


# In[ ]:

def get_gauge_prob(bayes_net, gauge_hot):
    """Calculate the marginal
    probability of the gauge
    showing hot (T/F) in the
    power plant system."""
    # TOOD: finish this function
    G_node = bayes_net.get_node_by_name('gauge')
    engine = JunctionTreeEngine(bayes_net)
    Q = engine.marginal(G_node)[0]
    index = Q.generate_index([gauge_hot],range(Q.nDims))
    gauge_prob = Q[index]

    return gauge_prob


# In[ ]:

from Inference import JunctionTreeEngine
def get_temperature_prob(bayes_net,temp_hot):
    """Calculate theprobability of the
    temperature being hot (T/F) in the
    power plant system, given that the
    alarm sounds and neither the gauge
    nor alarm is faulty."""
    # TODO: finish this function
    A = bayes_net.get_node_by_name('alarm')
    F_A = bayes_net.get_node_by_name('faulty alarm')
    F_G_node = bayes_net.get_node_by_name('faulty gauge')
    T_node = bayes_net.get_node_by_name('temperature')
    engine = JunctionTreeEngine(bayes_net)
    engine.evidence[A] = True
    engine.evidence[F_A] = False
    engine.evidence[F_G_node] = False
    Q = engine.marginal(T_node)[0]
    index = Q.generate_index([temp_hot],range(Q.nDims))
    temp_prob = Q[index]

    return temp_prob


# Part 2: Sampling
# -----
# 60 points total
#
# For the main exercise, consider the following scenario:
#
# There are five frisbee teams (T1, T2, T3,...,T5). A match is played between teams Ti and Ti+1 to give a total of 5 matches, i.e. T1vsT2, T2vsT3,...,T4vsT5,T5vsT1.
# Each team can either win, lose, or draw in a match. Each team has a fixed but unknown skill level, represented as an integer from 0 to 3. Each match's outcome is probabilistically proportional to the difference in skill level between the teams.
#
# We want to ESTIMATE the outcome of the last match (T5vsT1), given prior knowledge of other 4 matches.
# But wait! First, work on a similar, smaller network! With just 3 teams (Part 2a, 2b).
#
# Rather than using inference, we will do so by sampling the network using two [Markov Chain Monte Carlo](http://www.statistics.com/papers/LESSON1_Notes_MCMC.pdf) models: Gibbs sampling (2c) and Metropolis - Hastings sampling (3a).
#

# 2a: Build a small network with for 3 teams.
# -----
# 10 points
#
# For the first sub-part, consider a smaller network with 3 teams : the Airheads, the Buffoons, and the Clods (A, B and C for short). 3 total matches are played.
# Build a Bayes Net to represent the three teams and their influences on the match outcomes. Assume the following variable conventions:
#
# | variable name | description|
# |---------|:------:|
# |A| A's skill level|
# |B | B's skill level|
# |C | C's skill level|
# |AvB | the outcome of A vs. B <br> (0 = A wins, 1 = B wins, 2 = tie)|
# |BvC | the outcome of B vs. C <br> (0 = B wins, 1 = C wins, 2 = tie)|
# |CvA | the outcome of C vs. A <br> (0 = C wins, 1 = A wins, 2 = tie)|
#
# Assume that each team has the following prior distribution of skill levels:
#
# |skill level|P(skill level)|
# |----|:----:|
# |0|0.15|
# |1|0.45|
# |2|0.30|
# |3|0.10|
#
# In addition, assume that the differences in skill levels correspond to the following probabilities of winning:
#
# | skill difference <br> (T2 - T1) | T1 wins | T2 wins| Tie |
# |------------|----------|---|:--------:|
# |0|0.10|0.10|0.80|
# |1|0.20|0.60|0.20|
# |2|0.15|0.75|0.10|
# |3|0.05|0.90|0.05|

# In[ ]:

def get_game_network():
    """Create a Bayes Net representation of the game problem.
    Name the nodes as "A","B","C","AvB","BvC" and "CvA".  """
    nodes = []
    # TODO: fill this out

    # Create Nodes
    A = BayesNode(0, 4, name="A")
    B_node = BayesNode(1, 4, name="B")
    C_node = BayesNode(2, 4, name="C")
    AvB = BayesNode(3, 3, name="AvB")
    BvC_node = BayesNode(4, 3, name="BvC")
    CvA = BayesNode(5, 3, name="CvA")

    # A skill
    A.add_child(AvB)
    A.add_child(CvA)

    # B skill
    B_node.add_child(AvB)
    B_node.add_child(BvC_node)

    # C skill
    C_node.add_child(BvC_node)
    C_node.add_child(CvA)

    # AvB outcome
    AvB.add_parent(A)
    AvB.add_parent(B_node)

    # BvC outcome
    BvC_node.add_parent(B_node)
    BvC_node.add_parent(C_node)

    # CvA outcome
    CvA.add_parent(C_node)
    CvA.add_parent(A)


    # Set Probability Distributions
    # A, B, C team skill levels
    A_distribution = DiscreteDistribution(A)
    index = A_distribution.generate_index([],[])
    A_distribution[index]=[0.15, 0.45, 0.3, 0.1]
    A.set_dist(A_distribution)

    B_distribution = DiscreteDistribution(B_node)
    index = B_distribution.generate_index([],[])
    B_distribution[index]=[0.15, 0.45, 0.3, 0.1]
    B_node.set_dist(B_distribution)

    C_distribution = DiscreteDistribution(C_node)
    index = C_distribution.generate_index([],[])
    C_distribution[index]=[0.15, 0.45, 0.3, 0.1]
    C_node.set_dist(C_distribution)

    # AvB, BvC, CvA game outcomes
    dist = zeros([A.size(), B_node.size(), AvB.size()], dtype=float32)
    dist[0,0,:] = [0.1, 0.1, 0.8]
    dist[0,1,:] = [0.2, 0.6, 0.2]
    dist[0,2,:] = [0.15, 0.75, 0.1]
    dist[0,3,:] = [0.05, 0.9, 0.05]
    dist[1,0,:] = [0.6, 0.2, 0.2]
    dist[1,1,:] = [0.1, 0.1, 0.8]
    dist[1,2,:] = [0.2, 0.6, 0.2]
    dist[1,3,:] = [0.15, 0.75, 0.1]
    dist[2,0,:] = [0.75, 0.15, 0.1]
    dist[2,1,:] = [0.6, 0.2, 0.2]
    dist[2,2,:] = [0.1, 0.1, 0.8]
    dist[2,3,:] = [0.2, 0.6, 0.2]
    dist[3,0,:] = [0.9, 0.05, 0.05]
    dist[3,1,:] = [0.75, 0.15, 0.1]
    dist[3,2,:] = [0.6, 0.2, 0.2]
    dist[3,3,:] = [0.1, 0.1, 0.8]

    AvB_distribution = ConditionalDiscreteDistribution(nodes=[A, B_node, AvB], table=dist)
    AvB.set_dist(AvB_distribution)
    BvC_distribution = ConditionalDiscreteDistribution(nodes=[B_node, C_node, BvC_node], table=dist)
    BvC_node.set_dist(BvC_distribution)
    CvA_distribution = ConditionalDiscreteDistribution(nodes=[C_node, A, CvA], table=dist)
    CvA.set_dist(CvA_distribution)

    nodes = [A, B_node, C_node, AvB, BvC_node, CvA]

    return BayesNet(nodes)


# In[ ]:

game_net = get_game_network()
from probability_tests import games_network_test
games_network_test(game_net)


# 2b: Calculate posterior distribution for the 3rd match.
# ----
# 5 points
#
# Suppose that you know the following outcome of two of the three games: A beats B and A draws with C. Start by calculating the posterior distribution for the outcome of the BvC match in calculate_posterior(). Use EnumerationEngine ONLY.

# In[ ]:

def calculate_posterior(games_net):
    """Calculate the posterior distribution of the BvC match given that A won against B and tied C.
    Return a list of probabilities corresponding to win, loss and tie likelihood."""
    posterior = [0,0,0]
    # TODO: finish this function
    from Inference import EnumerationEngine
    AvB = games_net.get_node_by_name('AvB')
    BvC_node = games_net.get_node_by_name('BvC')
    CvA = games_net.get_node_by_name('CvA')
    engine = EnumerationEngine(games_net)
    engine.evidence[AvB] = 0
    engine.evidence[CvA] = 2
    Q = engine.marginal(BvC_node)[0]
    index = Q.generate_index([], range(Q.nDims))
    posterior = [Q[index][0], Q[index][1], Q[index][2]]
    return posterior


# In[ ]:

posterior = calculate_posterior(game_net)
from probability_tests import posterior_test
posterior_test(posterior)


# 2c: Gibbs sampling
# ---
# 20 points
#
# Now suppose you have 5 teams. You don't necessarily need to create a new network. You can just use the probability distributions tables from the previous part. Although be careful while indexing them. Check Hints 1 and 2 below, for more details.
#
# Implement the Gibbs sampling algorithm, which is a special case of Metropolis-Hastings. You'll do this in Gibbs_sampling(), which takes a Bayesian network and initial state value as a parameter and returns a sample state drawn from the network's distribution. The method should just consist of a single iteration of the algorithm. If an initial value is not given, default to a state chosen uniformly at random from the possible states.
#
# Note: DO NOT USE the given inference engines to run the sampling method, since the whole point of sampling is to calculate marginals without running inference.
#
#
#      "YOU WILL SCORE 0 POINTS ON THIS ASSIGNMENT IF YOU USE THE GIVEN INFERENCE ENGINES FOR THIS PART!!"
#
#
# You may find [this](http://gandalf.psych.umn.edu/users/schrater/schrater_lab/courses/AI2/gibbs.pdf) helpful in understanding the basics of Gibbs sampling over Bayesian networks. (Make sure to identify what makes it different from Metropolis-Hastings.)

# Hint 1: in both Metropolis-Hastings and Gibbs sampling, you'll need access to each node's probability distribution and nodes. You can access these by calling :
#
# A.dist.table, AvB.dist.table :Returns the same numpy array that you provided when constructing the probability distribution.
#
# Hint 2: To use the AvB.dist.table (needed for joint probability calculations), you could do something like:
#
# match_table = AvB.dist.table
# p = match_table[initial_value[x-n],initial_value[(x+1-n)%n],initial_value[x]], where n = 5 and x = 5,6,..,9
#
# Hint 3: you'll also want to use the random package (e.g. random.randint()) for the probabilistic choices that sampling makes.
#
# Hint 4: in order to count the sample states later on, you'll want to make sure the sample that you return is hashable. One way to do this is by returning the sample as a tuple.
#

# In[ ]:

def Gibbs_sampling(games_net, initial_value, number_of_teams=5):
    """Complete a single iteration of the Gibbs sampling algorithm
    given a Bayesian network and an initial state value.
    initial_value is a list of length 10 where:
    index 0-4: represent skills of teams T1, .. ,T5 (values lie in [0,3] inclusive)
    index 5-9: represent results of matches T1vT2,...,T5vT1 (values lie in [0,2] inclusive)

    Returns the new state sampled from the probability distribution as a tuple of length 10. """
    A= games_net.get_node_by_name("A")
    AvB= games_net.get_node_by_name("AvB")
    match_table = AvB.dist.table
    team_table = A.dist.table
    sample = tuple(initial_value)
    # TODO: finish this function

    import numpy as np

    # Select a random variable to change
    n = number_of_teams
    x = np.random.randint(2*n, size=1)[0]

    # If an initial value is not given, default to a state chosen uniformly at random from the possible states
    if initial_value is None:
        skills_state = np.random.randint(4, size=n)
        results_state = np.random.randint(3, size=n)
        initial_value = skills_state.tolist() + results_state.tolist()
        # print "Randomized initial state: ", initial_value

    # Update skill variable based on conditional joint probabilities
    if (x < n):
        skill_prob = np.zeros(4)
        # skill_prob_num = team_table[initial_value[x]] * match_table[initial_value[x], initial_value[(x+1)%n], initial_value[x+n]] * match_table[initial_value[(x-1)%n], initial_value[x], initial_value[(x+(2*n)-1)%(2*n)]]
        normalize = 0
        for i in range(4):
            skill_prob[i] = team_table[i] * match_table[i, initial_value[(x+1)%n], initial_value[x+n]] * match_table[initial_value[(x-1)%n], i, initial_value[(2*n-1) if x==0 else (x+n-1)]]
            normalize += skill_prob[i]
        skill_prob = skill_prob / normalize
        initial_value[x] = np.random.choice(4, p=skill_prob)

    # Update game result variable based on parent skills and match probabilities
    else:
        result_prob = match_table[initial_value[x-n], initial_value[(x+1-n)%n], :]
        initial_value[x] = np.random.choice(3, p=result_prob)

    sample = tuple(initial_value)
    return sample


# In[ ]:

# arbitrary initial state for the game system :
# 5 for teams T1,T2,...,T5
# 5 for matches T1vT2,T2vT3,....,T4vT5,T5vT1
number_of_teams=5
n = number_of_teams
initial_state = [0]*(2*n)
# initial_state = None
sample = Gibbs_sampling(game_net, initial_state, number_of_teams=5)


# 2d: Iterations to converge
# ----
# 20 points
#
# Suppose that you know the outcomes of 4 of the 5 matches.
#
# Estimate the likelihood of different outcomes for the 5 match (T5vT1) by running Gibbs sampling until it converges to a stationary distribution. We'll say that the sampler has converged when, for 10 successive iterations, the difference in expected outcome for the 5th match differs from the previous estimated outcome by less than 0.1.
#
# Note: Just measure how many iterations it takes for Gibbs to converge to a stable distribution over the posterior, regardless of how close to the actual posterior your approximations are. This is meant to show you that even though sampling methods are fast, their accuracy isn't perfect.

# In[ ]:

def converge_count_Gibbs(bayes_net, initial_state, match_results, number_of_teams=5):
    """Calculate number of iterations for Gibbs sampling to converge to any stationary distribution.
    And return the likelihoods for the last match. """
    count=0
    prob_win = 0.0
    prob_loss = 0.0
    prob_tie = 0.0
    posterior = [prob_win,prob_loss,prob_tie]
    # TODO: finish this function

    import numpy as np
    match_count = np.zeros(3)
    old_probs = np.array(posterior)
    probs = np.array(posterior)
    n = number_of_teams
    initial_state[n:n+len(match_results)] = match_results   # Assumes a valid initial_state is given
    burnin = 1000
    threshold = 0.0001
    converge = 10

    # Burn-in the initial_state with evidence set and fixed to match_results
    A= bayes_net.get_node_by_name("A")
    AvB= bayes_net.get_node_by_name("AvB")
    match_table = AvB.dist.table
    team_table = A.dist.table
    sample = tuple(initial_state)

    while (count < burnin):
        count += 1

        # Select a random variable to change, among the non-evidence variables
        x = np.random.randint(n+1, size=1)[0]
        if (x == 5): x = 9

        # Update skill variable based on conditional joint probabilities
        if (x < n):
            skill_prob = np.zeros(4)
            normalize = 0
            for i in range(4):
                skill_prob[i] = team_table[i] * match_table[i, initial_state[(x+1)%n], initial_state[x+n]] * match_table[initial_state[(x-1)%n], i, initial_state[(2*n-1) if x==0 else (x+n-1)]]
                normalize += skill_prob[i]
            skill_prob = skill_prob / normalize
            initial_state[x] = np.random.choice(4, p=skill_prob)

        # Update game result variable based on parent skills and match probabilities
        else:
            result_prob = match_table[initial_state[x-n], initial_state[(x+1-n)%n], :]
            initial_state[x] = np.random.choice(3, p=result_prob)

    # Discard burn-in samples and find convergence to a threshold value
    # for 10 successive iterations, the difference in expected outcome differs from the previous by less than 0.1
    count = 0
    convergence_count = 0
    while (convergence_count < converge):
        count += 1

        # Select a random variable to change, among the non-evidence variables
        x = np.random.randint(n+1, size=1)[0]
        if (x == 5): x = 9

        # Update skill variable based on conditional joint probabilities
        if (x < n):
            skill_prob = np.zeros(4)
            normalize = 0
            for i in range(4):
                skill_prob[i] = team_table[i] * match_table[i, initial_state[(x+1)%n], initial_state[x+n]] * match_table[initial_state[(x-1)%n], i, initial_state[(2*n-1) if x==0 else (x+n-1)]]
                normalize += skill_prob[i]
            skill_prob = skill_prob / normalize
            initial_state[x] = np.random.choice(4, p=skill_prob)

        # Update game result variable based on parent skills and match probabilities
        else:
            result_prob = match_table[initial_state[x-n], initial_state[(x+1-n)%n], :]
            initial_state[x] = np.random.choice(3, p=result_prob)

        sample = tuple(initial_state)

        # Keep track of EvA match results
        if (sample[9] == 0):
            match_count[0] += 1
        elif (sample[9] == 1):
            match_count[1] += 1
        else:
            match_count[2] += 1

        # Check for convergence in consecutive sample probabilities
        probs = match_count/count
        max_diff = np.amax(abs(probs-old_probs))
        old_probs = probs
        # print "  max_diff: ", max_diff
        if (max_diff < threshold and count > 100):
            convergence_count += 1

    posterior = probs.tolist()
    return count+burnin, posterior


# In[ ]:

from random import randint,uniform

# Now for an initial_state:
match_results = [0,0,1,1]
converge_count_Gibbs(game_net, initial_state, match_results, number_of_teams=5)


# 2e: Theoretical follow-up
# ---
# 5 points
#
# For n teams, using inference by enumeration, how does the complexity of predicting the last match vary with $n$?
#
# Fill in complexity_question() to answer, using big-O notation. For example, write 'O(n^2)' for second-degree polynomial runtime.

# In[ ]:

def complexity_question():
    # TODO: write an expression for complexity
    complexity = 'O(n^2)' 	#d^n
    return complexity



# 3a: Metropolis-Hastings sampling
# ---
# 20 points
#
# Now you will implement the Metropolis-Hastings algorithm, which is another method for estimating a probability distribution. You'll do this in MH_sampling(), which takes a Bayesian network and initial state as a parameter and returns a sample state drawn from the network's distribution. The method should just perform a single iteration of the algorithm. If an initial value is not given, default to a state chosen uniformly at random from the possible states.
#
# The general idea is to build an approximation of a latent probability distribution by repeatedly generating a "candidate" value for each random variable in the system, and then probabilistically accepting or rejecting the candidate value based on an underlying acceptance function. These [slides](https://www.cs.cmu.edu/~scohen/psnlp-lecture6.pdf) provide a nice intro, and this [cheat sheet](http://www.bcs.rochester.edu/people/robbie/jacobslab/cheat_sheet/MetropolisHastingsSampling.pdf) provides an explanation of the details.
#
# Note: DO NOT USE the given inference engines to run the sampling method, since the whole point of sampling is to calculate marginals without running inference.
#
#
#      "YOU WILL SCORE 0 POINTS IF YOU USE THE GIVEN INFERENCE ENGINES FOR THIS PART!!"
#

# In[ ]:

def MH_sampling(games_net, initial_value, n=5):
    """Complete a single iteration of the MH sampling algorithm given a Bayesian network and an initial state value.
    initial_value is a list of length 10 where:
    index 0-4: represent skills of teams T1, .. ,T5 (values lie in [0,3] inclusive)
    index 5-9: represent results of matches T1vT2,...,T5vT1 (values lie in [0,2] inclusive)

    Returns the new state sampled from the probability distribution as a tuple of length 10.
    """
    A= games_net.get_node_by_name("A")
    AvB= games_net.get_node_by_name("AvB")
    match_table = AvB.dist.table
    team_table = A.dist.table
    sample = tuple(initial_value)
    # TODO: finish this function

    import numpy as np

    # If an initial value is not given, default to a state chosen uniformly at random from the possible states
    if initial_value is None:
        skills_state = np.random.randint(4, size=n)
        results_state = np.random.randint(3, size=n)
        initial_value = skills_state.tolist() + results_state.tolist()
    print "initial_value: ", initial_value

    # # Update skill variable based on conditional joint probabilities
    # if (x < n):
    #     skill_prob = np.zeros(4)
    #     normalize = 0
    #     for i in range(4):
    #         skill_prob[i] = team_table[i] * match_table[i, initial_value[(x+1)%n], initial_value[x+n]] * match_table[initial_value[(x-1)%n], i, initial_value[(2*n-1) if x==0 else (x+n-1)]]
    #         normalize += skill_prob[i]
    #     skill_prob = skill_prob / normalize
    #     initial_value[x] = np.random.choice(4, p=skill_prob)

    # # Update game result variable based on parent skills and match probabilities
    # else:
    #     result_prob = match_table[initial_value[x-n], initial_value[(x+1-n)%n], :]
    #     initial_value[x] = np.random.choice(3, p=result_prob)

    # sample = tuple(initial_value)
    # return sample


    # Propose candidate
    skills_state = np.random.randint(4, size=n)
    results_state = np.random.randint(3, size=n)
    temp_state = [i for i in skills_state]+[j for j in results_state]

    # Select a random variable to change
    x = np.random.randint(2*n, size=1)[0]
    new_state = [i for i in initial_value]
    new_state[x] = temp_state[x]

    # Compute acceptance probability
    # current_weight = A.dist.table[initial_value[0]]*A.dist.table[initial_value[1]]*A.dist.table[initial_value[2]] \
    #     *AvB.dist.table[initial_value[0]][initial_value[1]][initial_value[3]]\
    #     *AvB.dist.table[initial_value[1]][initial_value[2]][initial_value[4]]\
    #     *AvB.dist.table[initial_value[2]][initial_value[0]][initial_value[5]]

    # new_weight =  A.dist.table[new_state[0]]*A.dist.table[new_state[1]]*A.dist.table[new_state[2]] \
    #     *AvB.dist.table[new_state[0]][new_state[1]][new_state[3]]\
    #     *AvB.dist.table[new_state[1]][new_state[2]][new_state[4]]\
    #     *AvB.dist.table[new_state[2]][new_state[0]][new_state[5]]

    current_weight = 0.5
    new_weight = 0.51

    # Accept or reject the proposal
    alpha = min(1, new_weight/current_weight)
    import random
    if (random.random() < alpha):
        sample = tuple((i for i in new_state))
    else:
    	sample = tuple((i for i in initial_value))

    print "sample: ", sample
    return sample


# In[ ]:

# arbitrary initial state for the game system
sample = MH_sampling(game_net, initial_state, n)


# In[ ]:

def converge_count_MH(bayes_net, initial_state, match_results, number_of_teams=5):
    """Calculate number of iterations for MH sampling to converge to any stationary distribution.
    And return the likelihoods for the last match. """
    count=0
    prob_win = 0.0
    prob_loss = 0.0
    prob_tie = 0.0
    posterior = [prob_win,prob_loss,prob_tie]
    # TODO: finish this function


    return count,posterior


# In[ ]:

converge_count_MH(game_net, initial_state, match_results, n)


# 3b: Compare the two sampling performances
# ---
# 10 points
#
# Which algorithm converges more quickly? By approximately what factor? For instance, if Metropolis-Hastings takes twice as many iterations to converge as Gibbs sampling, you'd say that it converged faster by a factor of 2. Fill in sampling_question() to answer both parts.

# In[ ]:

def compare_sampling(bayes_net, initial_state, match_results, n):
    """Compare Gibbs and Metropolis-Hastings sampling by calculating how long it takes for each method to converge
    to the provided posterior."""
    Gibbs_count = 0
    MH_count = 0
    # TODO: finish this function
    Gibbs_count = converge_count_Gibbs(bayes_net, initial_state, match_results, number_of_teams=n)[0]
    MH_count = converge_count_MH(bayes_net, initial_state, match_results, n)[0]

    return Gibbs_count, MH_count


# In[ ]:

# initial_state = initial_value
compare_sampling(game_net, initial_state, match_results, n)


# In[ ]:

def sampling_question():
    """Question about sampling performance."""
    # TODO: assign value to choice and factor
    choice = 0
    options = ['Gibbs','Metropolis-Hastings']
    factor = 2
    return options[choice], factor


# You're done! Write all the code out to a Python file "probability_solution.py" and submit it on T-Square before March 1, 11:59 PM UTC-12. This assignment will be graded on the accuracy of the functions you completed.



























"""Testing pbnt.
Run this before anything else
to get pbnt to work!"""
import sys
# from importlib import reload
if('pbnt/combined' not in sys.path):
    sys.path.append('pbnt/combined')
from exampleinference import inferenceExample

# Should output:
# ('The marginal probability of sprinkler=false:', 0.80102921)
#('The marginal probability of wetgrass=false | cloudy=False, rain=True:', 0.055)

inferenceExample()


from Node import BayesNode
from Graph import BayesNet

def make_power_plant_net():
    """Create a Bayes Net representation of
    the above power plant problem."""
    nodes = []
    A_node = BayesNode(0,2,name='alarm')
    FA_node = BayesNode(1,2,name='faulty alarm')
    G_node = BayesNode(2,2,name='gauge')
    FG_node = BayesNode(3,2,name='faulty gauge')
    T_node = BayesNode(4,2,name='temperature')

    T_node.add_child(FG_node)
    FG_node.add_parent(T_node)

    T_node.add_child(G_node)
    G_node.add_parent(T_node)

    FG_node.add_child(G_node)
    G_node.add_parent(FG_node)

    G_node.add_child(A_node)
    A_node.add_parent(G_node)

    FA_node.add_child(A_node)
    A_node.add_parent(FA_node)

    nodes.append(A_node)
    nodes.append(FA_node)
    nodes.append(G_node)
    nodes.append(FG_node)
    nodes.append(T_node)

    return BayesNet(nodes)

from probability_tests import network_setup_test
power_plant = make_power_plant_net()
network_setup_test(power_plant)



from numpy import zeros, float32
import Distribution
from Distribution import DiscreteDistribution, ConditionalDiscreteDistribution
def set_probability(bayes_net):
    """Set probability distribution for each
    node in the power plant system."""

    A_node = bayes_net.get_node_by_name("alarm")
    F_A_node = bayes_net.get_node_by_name("faulty alarm")
    G_node = bayes_net.get_node_by_name("gauge")
    F_G_node = bayes_net.get_node_by_name("faulty gauge")
    T_node = bayes_net.get_node_by_name("temperature")
    nodes = [A_node, F_A_node, G_node, F_G_node, T_node]
    # TODO: set the probability distribution for each node

    T_distribution = DiscreteDistribution(T_node)
    index = T_distribution.generate_index([],[])
    T_distribution[index] = [0.8,0.2]
    T_node.set_dist(T_distribution)

    F_A_distribution = DiscreteDistribution(F_A_node)
    index = F_A_distribution.generate_index([],[])
    F_A_distribution[index] = [0.85,0.15]
    F_A_node.set_dist(F_A_distribution)

    dist = zeros([T_node.size(), F_G_node.size()], dtype=float32)
    dist[0,:] = [0.95, 0.05]
    dist[1,:] = [0.2, 0.8]
    F_G_distribution = ConditionalDiscreteDistribution(nodes=[T_node,F_G_node], table=dist)
    F_G_node.set_dist(F_G_distribution)

    dist = zeros([G_node.size(), F_A_node.size(), A_node.size()], dtype=float32)
    dist[0,0,:] = [0.9, 0.1]
    dist[0,1,:] = [0.55, 0.45]
    dist[1,0,:] = [0.1, 0.9]
    dist[1,1,:] = [0.45, 0.55]
    A_distribution = ConditionalDiscreteDistribution(nodes=[G_node, F_A_node, A_node], table=dist)
    A_node.set_dist(A_distribution)

    dist = zeros([T_node.size(), F_G_node.size(), G_node.size()], dtype=float32)
    dist[0,0,:] = [0.95, 0.05]
    dist[0,1,:] = [0.2, 0.8]
    dist[1,0,:] = [0.05, 0.95]
    dist[1,1,:] = [0.8, 0.2]
    G_distribution = ConditionalDiscreteDistribution(nodes=[T_node, F_G_node, G_node], table=dist)
    G_node.set_dist(G_distribution)


    return bayes_net


set_probability(power_plant)
from probability_tests import probability_setup_test
probability_setup_test(power_plant)

from Inference import JunctionTreeEngine

def get_alarm_prob(bayes_net, alarm_rings):
    """Calculate the marginal
    probability of the alarm
    ringing (T/F) in the
    power plant system."""
    # TODO: finish this function
    A_node = bayes_net.get_node_by_name('alarm')
    engine = JunctionTreeEngine(bayes_net)
    Q = engine.marginal(A_node)[0]
    index = Q.generate_index([True],range(Q.nDims))
    alarm_prob = Q[index]
    return alarm_prob


def get_gauge_prob(bayes_net, gauge_hot):
    """Calculate the marginal
    probability of the gauge
    showing hot (T/F) in the
    power plant system."""
    # TOOD: finish this function
    G_node = bayes_net.get_node_by_name('gauge')
    engine = JunctionTreeEngine(bayes_net)
    Q = engine.marginal(G_node)[0]
    index = Q.generate_index([True],range(Q.nDims))
    gauge_prob = Q[index]
    return gauge_prob


def get_temperature_prob(bayes_net,temp_hot):
    """Calculate the probability of the
    temperature being hot (T/F) in the
    power plant system, given that the
    alarm sounds and neither the gauge
    nor alarm is faulty."""
    # TODO: finish this function
    A_node = bayes_net.get_node_by_name('alarm')
    F_A_node = bayes_net.get_node_by_name('faulty alarm')
    F_G_node = bayes_net.get_node_by_name('faulty gauge')
    T_node = bayes_net.get_node_by_name('temperature')
    engine = JunctionTreeEngine(bayes_net)
    engine.evidence[A_node] = True
    engine.evidence[F_A_node] = False
    engine.evidence[F_G_node] = False
    Q = engine.marginal(T_node)[0]
    index = Q.generate_index([True],range(Q.nDims))
    prob = Q[index]
    return prob

def get_game_network():
    """Create a Bayes Net representation
    of the game problem."""
    nodes = []
    # TODO: fill this out
    A_node = BayesNode(0,4,name='A')
    B_node = BayesNode(1,4,name='B')
    C_node = BayesNode(2,4,name='C')
    AvB_node = BayesNode(3,3,name='AvB')
    BvC_node = BayesNode(4,3,name='BvC')
    CvA_node = BayesNode(5,3,name='CvA')

    A_node.add_child(AvB_node)
    AvB_node.add_parent(A_node)

    B_node.add_child(AvB_node)
    AvB_node.add_parent(B_node)

    B_node.add_child(BvC_node)
    BvC_node.add_parent(B_node)

    C_node.add_child(BvC_node)
    BvC_node.add_parent(C_node)

    C_node.add_child(CvA_node)
    CvA_node.add_parent(C_node)

    A_node.add_child(CvA_node)
    CvA_node.add_parent(A_node)

    nodes.append(A_node)
    nodes.append(B_node)
    nodes.append(C_node)
    nodes.append(AvB_node)
    nodes.append(BvC_node)
    nodes.append(CvA_node)

    A_distribution = DiscreteDistribution(A_node)
    index = A_distribution.generate_index([],[])
    A_distribution[index] = [0.15,0.45,0.3,0.1]
    A_node.set_dist(A_distribution)

    B_distribution = DiscreteDistribution(B_node)
    index = B_distribution.generate_index([],[])
    B_distribution[index] = [0.15,0.45,0.3,0.1]
    B_node.set_dist(B_distribution)

    C_distribution = DiscreteDistribution(C_node)
    index = C_distribution.generate_index([],[])
    C_distribution[index] = [0.15,0.45,0.3,0.1]
    C_node.set_dist(C_distribution)

    dist = zeros([A_node.size(), B_node.size(), AvB_node.size()], dtype=float32)
    dist[0,0,:] = [0.1, 0.1, 0.8]
    dist[0,1,:] = [0.2, 0.6, 0.2]
    dist[0,2,:] = [0.15, 0.75, 0.1]
    dist[0,3,:] = [0.05, 0.9, 0.05]

    dist[1,0,:] = [0.6, 0.2, 0.2]
    dist[1,1,:] = [0.1, 0.1, 0.8]
    dist[1,2,:] = [0.2, 0.6, 0.2]
    dist[1,3,:] = [0.15, 0.75, 0.1]

    dist[2,0,:] = [0.75, 0.15, 0.1]
    dist[2,1,:] = [0.6, 0.2, 0.2]
    dist[2,2,:] = [0.1, 0.1, 0.8]
    dist[2,3,:] = [0.2, 0.6, 0.2]

    dist[3,0,:] = [0.9, 0.05, 0.05]
    dist[3,1,:] = [0.75, 0.15, 0.1]
    dist[3,2,:] = [0.6, 0.2, 0.2]
    dist[3,3,:] = [0.1, 0.1, 0.8]

    AvB_distribution = ConditionalDiscreteDistribution(nodes=[A_node, B_node, AvB_node], table=dist)
    AvB_node.set_dist(AvB_distribution)

    BvC_distribution = ConditionalDiscreteDistribution(nodes=[B_node, C_node, BvC_node], table=dist)
    BvC_node.set_dist(BvC_distribution)

    CvA_distribution = ConditionalDiscreteDistribution(nodes=[C_node, A_node, CvA_node], table=dist)
    CvA_node.set_dist(CvA_distribution)

    print "Printing table"
    print type(AvB_node.dist.table)
    for i in range(3):
        print AvB_node.dist.table[0][0][i]

    return BayesNet(nodes)

game_net = get_game_network()


import random

def findstate(randomnum, probdist, lenprobdist):
    if lenprobdist == 4:
        if randomnum >= 0 and randomnum < probdist[0]:
            return 0
        elif randomnum >= probdist[0] and randomnum < (probdist[0] + probdist[1]):
            return 1
        elif randomnum >= (probdist[0] + probdist[1]) and randomnum < (probdist[0] + probdist[1] + probdist[2]):
            return 2
        else:
            return 3

    elif lenprobdist == 3:
        if randomnum >= 0 and randomnum < probdist[0]:
            return 0
        elif randomnum >= probdist[0] and randomnum < (probdist[0] + probdist[1]):
            return 1
        else:
            return 2




def Gibbs_sampling(bayes_net, initial_value):
    """Complete a single iteration of the
    Gibbs sampling algorithm given a
    Bayesian network and an initial state
    value. Returns the state sampled from
    the probability distribution."""
    # TODO: finish this function
    pa = random.random()
    pb = random.random()
    pc = random.random()
    pavb = random.random()
    pbvc = random.random()
    pcva = random.random()

    A_node = bayes_net.get_node_by_name('A')
    B_node = bayes_net.get_node_by_name('B')
    C_node = bayes_net.get_node_by_name('C')
    AvB_node = bayes_net.get_node_by_name('AvB')
    BvC_node = bayes_net.get_node_by_name('BvC')
    CvA_node = bayes_net.get_node_by_name('CvA')

    a = findstate(pa, A_node.dist.table, 4)
    b = findstate(pb, B_node.dist.table, 4)
    c = findstate(pc, C_node.dist.table, 4)

    avb = findstate(pavb, AvB_node.dist.table[a][b], 3)
    bvc = findstate(pbvc, AvB_node.dist.table[b][c], 3)
    cva = findstate(pcva, AvB_node.dist.table[c][a], 3)

    sample = [a, b, c, avb, bvc, cva]

    return tuple(sample)

# arbitrary initial state for the game system
initial_value = [0,0,0,0,0,0]
sample = Gibbs_sampling(game_net, initial_value)

print sample



def MH_sampling(bayes_net, initial_value):
    """Complete a single iteration of the
    Metropolis-Hastings algorithm given a
    Bayesian network and an initial state
    value. Returns the state sampled from
    the probability distribution."""
    # TODO: finish this function
    A_node = bayes_net.get_node_by_name('A')
    B_node = bayes_net.get_node_by_name('B')
    C_node = bayes_net.get_node_by_name('C')
    AvB_node = bayes_net.get_node_by_name('AvB')
    BvC_node = bayes_net.get_node_by_name('BvC')
    CvA_node = bayes_net.get_node_by_name('CvA')

    for i in range(6):
        new_value = initial_value
        if i <= 2:
            new_value[i] = random.randint(0,3)
        else:
            new_value[i] = random.randint(0,2)

        numerator = A_node.dist.table[new_value[0]] * B_node.dist.table[new_value[1]] * C_node.dist.table[new_value[2]] * AvB_node.dist.table[new_value[0]][new_value[1]][new_value[3]] * BvC_node.dist.table[new_value[1]][new_value[2]][new_value[4]] * CvA_node.dist.table[new_value[2]][new_value[0]][new_value[5]]
        denominator = A_node.dist.table[initial_value[0]] * B_node.dist.table[initial_value[1]] * C_node.dist.table[initial_value[2]] * AvB_node.dist.table[initial_value[0]][initial_value[1]][initial_value[3]] * BvC_node.dist.table[initial_value[1]][initial_value[2]][initial_value[4]] * CvA_node.dist.table[initial_value[2]][initial_value[0]][initial_value[5]]

        alpha = min(1,(numerator/denominator))

        if random.random() < alpha:
            initial_value = new_value


    return tuple(new_value)


initial_value = [0,0,0,0,0,0]
sample = MH_sampling(game_net, initial_value)

print 'MH'
print sample


from Inference import EnumerationEngine
def calculate_posterior(games_net):
    """Calculate the posterior distribution
    of the BvC match given that A won against
    B and tied C. Return a list of probabilities
    corresponding to win, loss and tie likelihood."""
    posterior = [0,0,0]
    # TODO: finish this function
    A_node = games_net.get_node_by_name('A')
    B_node = games_net.get_node_by_name('B')
    C_node = games_net.get_node_by_name('C')
    AvB_node = games_net.get_node_by_name('AvB')
    BvC_node = games_net.get_node_by_name('BvC')
    CvA_node = games_net.get_node_by_name('CvA')
    engine = EnumerationEngine(games_net)
    engine.evidence[AvB_node] = 0
    engine.evidence[CvA_node] = 2
    Q = engine.marginal(BvC_node)[0]
    posterior = Q.table
    return posterior


iter_counts = [1e1,1e3,1e5,1e6]
def compare_sampling(bayes_net, posterior):
    """Compare Gibbs and Metropolis-Hastings
    sampling by calculating how long it takes
    for each method to converge to the
    provided posterior."""
    # TODO: finish this function
    initial_value = [0,0,0,0,0,0]
    #for iter_count in iter_counts:
        #for i in iter_count:
    bvc0 = 0
    bvc1 = 0
    bvc2 = 0
    oldpbvc0 = 1000
    oldpbvc1 = 1000
    oldpbvc2 = 1000
    count = 0
    converged = 0
    while(True):
        count += 1
        sample = MH_sampling(game_net, initial_value)
        print 'Printing sample'
        print sample
        if sample[3] == 0 and sample[5] == 2:
            if sample[4] == 0:
                bvc0 += 1
            elif sample[4] == 1:
                bvc1 += 1
            elif sample[4] == 2:
                bvc2 += 1

        pbvc0 = float(bvc0)/count
        pbvc1 = float(bvc1)/count
        pbvc2 = float(bvc2)/count


        if abs(pbvc0 - oldpbvc0) < 0.001 and abs(pbvc1 - oldpbvc1) < 0.001 and abs(pbvc2 - oldpbvc2) < 0.001:
            converged += 1
        else:
            converged = 0

        print 'Converged', converged
        print pbvc0,pbvc1,pbvc2
        if converged == 10 and count > 50:
            break

        oldpbvc0 = pbvc0
        oldpbvc1 = pbvc1
        oldpbvc2 = pbvc2


    return count
    #print count


    #return Gibbs_convergence, MH_convergence

# test your sampling methods here
posterior = calculate_posterior(game_net)
print compare_sampling(game_net, posterior)

    #return Gibbs_convergence, MH_convergence