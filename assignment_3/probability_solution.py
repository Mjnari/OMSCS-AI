"""Testing pbnt. Run this before anything else to get pbnt to work!"""
import sys
from random import randrange

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

from numpy import zeros, float32, random
import Distribution, Node, Graph
from Node import BayesNode
from Graph import BayesNet
from Distribution import DiscreteDistribution, ConditionalDiscreteDistribution
from Inference import JunctionTreeEngine
from Inference import EnumerationEngine


def make_exam_net():
    """Create a Bayes Net representation of the above power plant problem.
    Name the nodes as "alarm","faulty alarm", "gauge","faulty gauge", "temperature".
    """
    nodes = []
    # TODO: finish this function

    #create nodes ---> refer node.py
    MA = BayesNode(0, 2, name="main alarm")
    A1 = BayesNode(1, 2, name="alarm 2")
    A2 = BayesNode(2, 2, name="alarm 1")
    A3 = BayesNode(3, 2, name="alarm 3")
    G = BayesNode(4, 2, name="ghost")
    B = BayesNode(5, 2, name="burglar")

    MA.add_parent(G)
    MA.add_parent(B)


    dist = zeros([G.size(), B.size(), MA.size()], dtype=float32)
    dist[0,0,:] = [0.99, 0.01]
    dist[0,1,:] = [0.36, 0.64]
    dist[1,0,:] = [0.75, 0.25]
    dist[1,1,:] = [0.02, 0.98]

    MA_distribution = ConditionalDiscreteDistribution(nodes=[G, B, MA], table=dist)
    MA.set_dist(MA_distribution)


    G.add_child(MA)
    G.add_child(A2)
    G.add_child(A1)

    G_distribution = DiscreteDistribution(G)
    index = G_distribution.generate_index([],[])
    G_distribution[index] = [0.6,0.4]
    G.set_dist(G_distribution)

    B.add_child(MA)
    B.add_child(A2)
    B.add_child(A3)

    B_distribution = DiscreteDistribution(B)
    index = B_distribution.generate_index([],[])
    B_distribution[index] = [0.68,0.32]
    B.set_dist(B_distribution)

    A1.add_parent(G)

    A1_distribution = DiscreteDistribution(A1)
    dist = zeros([G.size(), A1.size()], dtype=float32)   #Note the order of G_node, A_node
    dist[0,:] = [0.91, 0.09]
    dist[1,:] = [0.18, 0.82]
    A1_distribution = ConditionalDiscreteDistribution(nodes=[G,A1], table=dist)
    A1.set_dist(A1_distribution)


    A2.add_parent(G)
    A2.add_parent(B)

    dist = zeros([G.size(), B.size(), A2.size()], dtype=float32)
    dist[0,0,:] = [0.95, 0.05]
    dist[0,1,:] = [0.37, 0.63]
    dist[1,0,:] = [0.69, 0.31]
    dist[1,1,:] = [0.13, 0.87]

    A2_distribution = ConditionalDiscreteDistribution(nodes=[G, B, A2], table=dist)
    A2.set_dist(A2_distribution)

    A3.add_parent(B)

    A3_distribution = DiscreteDistribution(A3)
    dist = zeros([B.size(), A3.size()], dtype=float32)   #Note the order of G_node, A_node
    dist[0,:] = [0.28, 0.72]
    dist[1,:] = [0.81, 0.19]
    A3_distribution = ConditionalDiscreteDistribution(nodes=[B,A3], table=dist)
    A3.set_dist(A3_distribution)

    nodes = [MA,A1,A2,A3,G,B]

    return BayesNet(nodes)

def get_burglar_prob(bayes_net):
    """Calculate theprobability of the
    temperature being hot (T/F) in the
    power plant system, given that the
    alarm sounds and neither the gauge
    nor alarm is faulty."""
    # TODO: finish this function
    A2 = bayes_net.get_node_by_name('alarm 2')
    B = bayes_net.get_node_by_name('burglar')

    engine = JunctionTreeEngine(bayes_net)
    engine.evidence[A2] = True
    Q = engine.marginal(B)[0]
    index = Q.generate_index([True],range(Q.nDims))
    temp_prob = Q[index]


    engine2 = EnumerationEngine(bayes_net)
    engine2.evidence[A2] = True # A won against B

    Q = engine.marginal(B)[0]
    posterior = Q.table
    print "Or:" , posterior.tolist()

    return temp_prob

def get_ghost_prob(bayes_net):
    """Calculate theprobability of the
    temperature being hot (T/F) in the
    power plant system, given that the
    alarm sounds and neither the gauge
    nor alarm is faulty."""
    # TODO: finish this function
    A1 = bayes_net.get_node_by_name('alarm 1')
    A2 = bayes_net.get_node_by_name('alarm 2')
    G = bayes_net.get_node_by_name('ghost')

    engine = JunctionTreeEngine(bayes_net)
    engine.evidence[A2] = True
    engine.evidence[A1] = True
    Q = engine.marginal(G)[0]
    index = Q.generate_index([True],range(Q.nDims))
    temp_prob = Q[index]

    engine2 = EnumerationEngine(bayes_net)
    engine2.evidence[A2] = True # A won against B
    engine2.evidence[A1] = True # A won against B

    Q = engine.marginal(G)[0]
    posterior = Q.table
    print "Or:" , posterior.tolist()
    return temp_prob

def get_alarm3_prob(bayes_net):
    """Calculate theprobability of the
    temperature being hot (T/F) in the
    power plant system, given that the
    alarm sounds and neither the gauge
    nor alarm is faulty."""
    # TODO: finish this function
    MA = bayes_net.get_node_by_name('main alarm')
    B = bayes_net.get_node_by_name('burglar')
    A2 = bayes_net.get_node_by_name('alarm 2')
    A3 = bayes_net.get_node_by_name('alarm 3')
    engine = JunctionTreeEngine(bayes_net)
    engine.evidence[A2] = False
    engine.evidence[MA] = False
    engine.evidence[B] = True
    Q = engine.marginal(A3)[0]
    index = Q.generate_index([True],range(Q.nDims))
    temp_prob = Q[index]

    engine2 = EnumerationEngine(bayes_net)
    engine2.evidence[A2] = False # A won against B
    engine2.evidence[MA] = False # A won against B
    engine2.evidence[B] = True # A won against B

    Q = engine.marginal(A3)[0]
    posterior = Q.table
    print "Or:" , posterior.tolist()

    return temp_prob

def get_ghost_prob2(bayes_net):
    """Calculate theprobability of the
    temperature being hot (T/F) in the
    power plant system, given that the
    alarm sounds and neither the gauge
    nor alarm is faulty."""
    # TODO: finish this function
    A3 = bayes_net.get_node_by_name('alarm 3')
    A2 = bayes_net.get_node_by_name('alarm 2')
    G = bayes_net.get_node_by_name('ghost')

    engine = JunctionTreeEngine(bayes_net)
    engine.evidence[A2] = False
    engine.evidence[A3] = True
    Q = engine.marginal(G)[0]
    index = Q.generate_index([False],range(Q.nDims))
    temp_prob = Q[index]

    return temp_prob


# 1a
def make_power_plant_net():
    #testing basic bayes net class implementation
    numberOfNodes = 5
    #name the nodes
    alarm = 0
    faulty_alarm = 1
    gauge = 2
    faulty_gauge = 3
    temperature = 4

    #create nodes ---> refer Node.py
    alarm_node = BayesNode(0, 2, name="alarm")
    faulty_alarm_node = BayesNode(1, 2, name="faulty alarm")
    gauge_node = BayesNode(2, 2, name="gauge")
    faulty_gauge_node = BayesNode(3, 2, name="faulty gauge")
    temperature_node = BayesNode(4, 2, name="temperature")

    temperature_node.add_child(faulty_gauge_node)
    temperature_node.add_child(gauge_node)

    faulty_alarm_node.add_child(alarm_node)

    faulty_gauge_node.add_parent(temperature_node)
    faulty_gauge_node.add_child(gauge_node)

    gauge_node.add_parent(faulty_gauge_node)
    gauge_node.add_parent(temperature_node)
    gauge_node.add_child(alarm_node)

    alarm_node.add_parent(faulty_alarm_node)
    alarm_node.add_parent(gauge_node)

    nodes = [alarm_node, faulty_alarm_node, gauge_node,faulty_gauge_node,temperature_node]

    return BayesNet(nodes)

# 1b
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

# 1c
def set_probability(bayes_net):
    """Set probability distribution for each node in the power plant system."""

    A_node = bayes_net.get_node_by_name("alarm")
    F_A_node = bayes_net.get_node_by_name("faulty alarm")
    G_node = bayes_net.get_node_by_name("gauge")
    F_G_node = bayes_net.get_node_by_name("faulty gauge")
    T_node = bayes_net.get_node_by_name("temperature")

     # temperature distribution
    temperature_distribution = DiscreteDistribution(T_node)
    index = temperature_distribution.generate_index([],[])
    temperature_distribution[index] = [0.80,0.20]
    T_node.set_dist(temperature_distribution)

    # faulty alarm distribution
    faulty_alarm_distribution = DiscreteDistribution(F_A_node)
    index = faulty_alarm_distribution.generate_index([],[])
    faulty_alarm_distribution[index] = [0.85,0.15]
    F_A_node.set_dist(faulty_alarm_distribution)

    # faulty gauge distribution
    dist = zeros([T_node.size(), F_G_node.size()], dtype=float32)   #Note the order of temp, Fg
    dist[0,:] = [0.95, 0.05]
    dist[1,:] = [0.20, 0.80]
    faulty_gauge_distribution = ConditionalDiscreteDistribution(nodes=[T_node,F_G_node], table=dist)
    F_G_node.set_dist(faulty_gauge_distribution)

    # guage distribution
    dist = zeros([T_node.size(), F_G_node.size(), G_node.size()], dtype=float32)
    dist[0,0,:] = [0.95, 0.05]
    dist[0,1,:] = [0.2, 0.8]
    dist[1,0,:] = [0.05, 0.95]
    dist[1,1,:] = [0.80, 0.20]

    gauge_node_distribution = ConditionalDiscreteDistribution(nodes=[T_node, F_G_node, G_node], table=dist)
    G_node.set_dist(gauge_node_distribution)

    # alarm distribution
    dist = zeros([G_node.size(), F_A_node.size(), A_node.size()], dtype=float32)
    dist[0,0,:] = [0.90, 0.10]
    dist[0,1,:] = [0.55, 0.45]
    dist[1,0,:] = [0.10, 0.90]
    dist[1,1,:] = [0.45, 0.55]

    alarm_node_distribution = ConditionalDiscreteDistribution(nodes=[G_node, F_A_node, A_node], table=dist)
    A_node.set_dist(alarm_node_distribution)

    nodes = [A_node, F_A_node, G_node, F_G_node, T_node]

    return bayes_net

# 1d
def get_alarm_prob(bayes_net, alarm_rings):
    """Calculate the marginal probability of the alarm ringing (T/F) in the power plant system."""
    # TODO: finish this function
    A_node = bayes_net.get_node_by_name('alarm')
    engine = JunctionTreeEngine(bayes_net)
    Q = engine.marginal(A_node)[0]
    index = Q.generate_index([alarm_rings],range(Q.nDims))
    prob = Q[index]
    return prob

def get_gauge_prob(bayes_net, gauge_hot):
    """Calculate the marginal probability of the gauge showing hot (T/F) in the power plant system."""
    # TODO: finish this function
    G_node = bayes_net.get_node_by_name('gauge')
    engine = JunctionTreeEngine(bayes_net)
    Q = engine.marginal(G_node)[0]
    index = Q.generate_index([gauge_hot],range(Q.nDims))
    prob = Q[index]
    return prob

def get_temperature_prob(bayes_net,temp_hot):
    """Calculate the probability of the temperature being hot (T/F) in the power plant system, given that the
      alarm sounds and neither the gauge nor alarm is faulty."""
    # TODO: finish this function
    T_node = bayes_net.get_node_by_name('temperature')
    F_A_node = bayes_net.get_node_by_name('faulty alarm')
    F_G_node = bayes_net.get_node_by_name('faulty gauge')
    A_node = bayes_net.get_node_by_name('alarm')
    engine = JunctionTreeEngine(bayes_net)
    engine.evidence[A_node] = True # alarm sounds
    engine.evidence[F_A_node] = False # alarm is NOT faulty
    engine.evidence[F_G_node] = False # gauge is NOT faulty
    Q = engine.marginal(T_node)[0]
    index = Q.generate_index([temp_hot],range(Q.nDims))
    prob = Q[index]
    return prob

def get_game_network():
    """Create a Bayes Net representation of the game problem.
    Name the nodes as "A","B","C","AvB","BvC" and "CvA".  """

    teama_node = BayesNode(0,4,name='A')
    teamb_node = BayesNode(1,4,name='B')
    teamc_node = BayesNode(2,4,name='C')
    matchAvB_node = BayesNode(3,3,name='AvB')
    matchBvC_node = BayesNode(4,3,name='BvC')
    matchCvA_node = BayesNode(5,3,name='CvA')


    teama_node.add_child(matchAvB_node)
    matchAvB_node.add_parent(teama_node)

    teama_node.add_child(matchCvA_node)
    matchCvA_node.add_parent(teama_node)

    teamb_node.add_child(matchAvB_node)
    matchAvB_node.add_parent(teamb_node)

    teamb_node.add_child(matchBvC_node)
    matchBvC_node.add_parent(teamb_node)

    teamc_node.add_child(matchBvC_node)
    matchBvC_node.add_parent(teamc_node)

    teamc_node.add_child(matchCvA_node)
    matchCvA_node.add_parent(teamc_node)

    skill_dist = [0.15,0.45,0.30,0.10]

    teama_distribution = DiscreteDistribution(teama_node)
    index = teama_distribution.generate_index([],[])
    teama_distribution[index] = skill_dist
    teama_node.set_dist(teama_distribution)

    teamb_distribution = DiscreteDistribution(teamb_node)
    index = teamb_distribution.generate_index([],[])
    teamb_distribution[index] = skill_dist
    teamb_node.set_dist(teamb_distribution)

    teamc_distribution = DiscreteDistribution(teamc_node)
    index = teamc_distribution.generate_index([],[])
    teamc_distribution[index] = skill_dist
    teamc_node.set_dist(teamc_distribution)

    match_dist = zeros([teama_node.size(), teamb_node.size(), matchAvB_node.size()], dtype=float32)
    match_dist[0,0,:] = [0.10, 0.10, 0.80]
    match_dist[0,1,:] = [0.20, 0.60, 0.20]
    match_dist[0,2,:] = [0.15, 0.75, 0.10]
    match_dist[0,3,:] = [0.05, 0.90, 0.05]

    match_dist[1,0,:] = [0.60, 0.20, 0.20]
    match_dist[1,1,:] = [0.10, 0.10, 0.80]
    match_dist[1,2,:] = [0.20, 0.60, 0.20]
    match_dist[1,3,:] = [0.15, 0.75, 0.10]

    match_dist[2,0,:] = [0.75, 0.15, 0.10]
    match_dist[2,1,:] = [0.60, 0.20, 0.20]
    match_dist[2,2,:] = [0.10, 0.10, 0.80]
    match_dist[2,3,:] = [0.20, 0.60, 0.20]

    match_dist[3,0,:] = [0.90, 0.05, 0.05]
    match_dist[3,1,:] = [0.75, 0.15, 0.10]
    match_dist[3,2,:] = [0.60, 0.20, 0.20]
    match_dist[3,3,:] = [0.10, 0.10, 0.80]

    matchAvB_distribution = ConditionalDiscreteDistribution(nodes=[teama_node, teamb_node, matchAvB_node], table=match_dist)
    matchBvC_distribution = ConditionalDiscreteDistribution(nodes=[teamb_node, teamc_node, matchBvC_node], table=match_dist)
    matchCvA_distribution = ConditionalDiscreteDistribution(nodes=[teamc_node, teama_node, matchCvA_node], table=match_dist)

    matchAvB_node.set_dist(matchAvB_distribution)
    matchBvC_node.set_dist(matchBvC_distribution)
    matchCvA_node.set_dist(matchCvA_distribution)

    nodes = [teama_node,teamb_node,teamc_node,matchAvB_node,matchBvC_node,matchCvA_node]

    return BayesNet(nodes)

def calculate_posterior(games_net):
    """Calculate the posterior distribution of the BvC match given that A won against B and tied C.
    Return a list of probabilities corresponding to win, loss and tie likelihood."""
    posterior = [0,0,0]

    matchAvB_node = games_net.get_node_by_name('AvB')
    matchBvC_node = games_net.get_node_by_name('BvC')
    matchCvA_node = games_net.get_node_by_name('CvA')

    engine = EnumerationEngine(games_net)
    engine.evidence[matchAvB_node] = 0 # A won against B
    engine.evidence[matchCvA_node] = 2 # A tied team C
    Q = engine.marginal(matchBvC_node)[0]
    posterior = Q.table
    return posterior.tolist()

def Gibbs_sampler(games_net, initial_value, number_of_teams=5, evidence=None):
    """Complete a single iteration of the Gibbs sampling algorithm
    given a Bayesian network and an initial state value.

    initial_value is a list of length 10 where:
    index 0-4: represent skills of teams T1, .. ,T5 (values lie in [0,3] inclusive)
    index 5-9: represent results of matches T1vT2,...,T5vT1 (values lie in [0,2] inclusive)

    Returns the new state sampled from the probability distribution as a tuple of length 10.
    Return the sample as a tuple.

    You will need the evidence variable for part 2d, for now there is None.
    You can implement this any way you want (i.e as a list/tuple of evidence indices)
    """
    A= games_net.get_node_by_name("A")
    AvB= games_net.get_node_by_name("AvB")
    match_table = AvB.dist.table
    team_table = A.dist.table
    sample = tuple(initial_value)
    # TODO: finish this function
    choice = 0
    # if no initial state, generate one
    if initial_value == None or initial_value == []:
        if evidence:
            initial_value = [0] * (number_of_teams)
            initial_value.extend(evidence)
            initial_value.append(0)
        else:
            initial_value = [0] * (2 * number_of_teams)
    else:
        if evidence:
            initial_value[5:9] = evidence

        # print initial_value

    # pick a random variable to sample
    if evidence:
        # ignore evidence variables
        index = random.choice([0,1,2,3,4,9])
    else:
        index = randrange(0,len(initial_value))

    # print index
    # print match_table
    # print team_table
    # print initial_value
    if index >= number_of_teams: # update match variable
        prob = match_table[initial_value[index-number_of_teams],initial_value[(index+1-number_of_teams)%number_of_teams], :]
        choice = random.choice(3, p=prob) # random choice betweeen 0,1,2 (game states) based on probabilities
    else: # update skill variable
        num_skills = len(team_table)
        prob = [0] * num_skills
        sum = 0
        for i in range(num_skills):
            # print i
            # print (index+1)%number_of_teams
            # print index+number_of_teams
            # print (index+number_of_teams-1)
            # print team_table[i]
            # print '*'*20
            prob[i] = team_table[i] *\
                      match_table[i, initial_value[(index+1)%number_of_teams], initial_value[index+number_of_teams]] * \
                      match_table[initial_value[(index-1)%number_of_teams], i, initial_value[index+number_of_teams-1 if index>0 else -1 ]]
            sum += prob[i]
        prob = prob / sum
        # print prob

        choice = random.choice(4, p=prob) # random choice betweeen 0,1,2,3 (skill levels) based on probabilities

    # print index
    initial_value[index] = choice
    return tuple(initial_value)

def converge_count_Gibbs(game_net, initial_state, match_results, n):
    """Calculate number of iterations for Gibbs sampling to converge to a stationary distribution.
    And return the likelihoods for the last match. """
    count=0
    posterior = [0.0] * 3
    game_tally = [0] * 3
    # TODO: finish this function
    burn_in = 1500
    # ignored samples
    for i in range(burn_in):
        sample = Gibbs_sampler(game_net,initial_state, n, evidence=match_results)

    converge = 10  # number of converges until stationary distribution is reached
    rounds = 0
    while count < converge:
        rounds+=1

        sample = Gibbs_sampler(game_net,initial_state, n, evidence=match_results)
        result = sample[-1]  # T5vT1 match
        game_tally[result]+=1
        # calculate the current probabilities
        curr_posterior = [float(x) / rounds for x in game_tally]
        # print curr_posterior
        # print curr_posterior , " rounds: ", rounds
        # print [abs(x - y) for x, y in zip(curr_posterior, posterior)]

        # calculate the difference in the current and previous posterior probabilities
        diff = max([abs(x - y) for x, y in zip(curr_posterior, posterior)])

        # update posterior
        posterior = curr_posterior

        # converged if the sum of difference in posteriors is less than delta (in this case .0001)

        if (rounds > 100 and diff < .0001):
            count+=1

    return rounds+burn_in, posterior

def complexity_question():
    # TODO: write an expression for complexity
    # For n teams, using inference by enumeration, how does the complexity of predicting the last match vary with n?
    complexity = '3^n' # where 3 is the arity and n is the number of teams
    return complexity


def MH_sampler(games_net, initial_value, n=5, evidence=None):
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

    return sample