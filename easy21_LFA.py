import random
import numpy as np
import matplotlib as ml
import matplotlib.pyplot as plt
from matplotlib import animation 
from matplotlib import colors

####### Q1. implement enviornment ####### 

black_cards = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
red_cards = [-1, -2, -3, -4, -5, -6, -7, -8, -9, -10]
cheat_cards = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]
cards = 2*black_cards + red_cards

hit = 0
stick = 1

dealer_sum = 0
player_sum = 0

def init():
    global dealer_sum
    global player_sum
    dealer_sum = random.choice(black_cards)
    #player_sum = random.choice(black_cards)
    player_sum = random.choice(cheat_cards) #for more observation, drop to random state and start episode

def reset_list_w_action():
    E = []
    for i in range(10):
        tmp = []
        for j in range(21):
            tmp.append([0.0,0.0])
        E.append(tmp)
    return E

def reset_list_wo_action():
    E = []
    for i in range(10):
        tmp = []
        for j in range(21):
            tmp.append(0)
        E.append(tmp)
    return E

def random_policy():    
    Policy = []
    for i in range(10):
        list_tmp = []
        for j in range(21):
            list_tmp.append(0.5) #initial policy: randomly stochastic
        Policy.append(list_tmp)
    return Policy

def update_N(N, dealer_sum, player_sum, action):
    N[dealer_sum-1][player_sum-1][action] += 1

def update_action_val(Q, N, dealer_sum, player_sum, action, target):
    if target != 0:
        Q[dealer_sum-1][player_sum-1][action] += target/N[dealer_sum-1][player_sum-1][action]

def update_policy_e_greedy(Policy, Q, dealer_sum, player_sum, e):
    Policy[dealer_sum - 1][player_sum - 1] = e/2 + np.argmax(Q[dealer_sum - 1][player_sum - 1])*(1-e)

def follow_policy(Policy, dealer_sum, player_sum):
    stochastic_action = Policy[dealer_sum-1][player_sum-1]
    action = random.choices([hit, stick], [1 - stochastic_action, stochastic_action])[0]
    return action

def step(stochastic_action):
    global dealer_sum
    global player_sum

    action = random.choices([hit, stick], [1 - stochastic_action, stochastic_action])[0]

    if action == hit:
        player_sum += random.choice(cards)
        if player_sum < 1 or player_sum > 21:
            reward = -1
            term = True
        else:
            reward = 0
            term = False
    if action == stick:
        term = True
        while dealer_sum <= 17 and dealer_sum >= 1:
            dealer_sum += random.choice(cards)
        if dealer_sum > 21 or dealer_sum < 1 or player_sum > dealer_sum:
            reward = 1
        if player_sum == dealer_sum:
            reward = 0
        if player_sum < dealer_sum:
            reward = -1

    return reward, term

####### Q4. Linear Function Approximation ####### 

#lambdas = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
lambdas = [1]
iters = [1000, 5000, 10000, 50000, 100000]

def feature_vec(dealer, player, action):
    vec = [0]*36
    i_s = [(1 <= dealer and dealer <= 4), (4 <= dealer and dealer <= 7), (7 <= dealer and dealer <= 10)]
    j_s = [(1 <= player and player <= 6), (4 <= player and player <= 9), (7 <= player and player <= 12), (10 <= player and player <= 15), (13 <= player and player <= 18), (16 <= player and player <= 21)]

    for n, i in enumerate(i_s):
        for m, j in enumerate(j_s):
            if i and j:
                vec[n + 3*m + 18*action] = 1

    return vec

def approx_action_val(theta, dealer, player, action):
    vec = feature_vec(dealer, player, action)
    val = 0
    for i in range(36):
        val += vec[i]*theta[i]
    return val

def get_action_from_Qw(theta, dealer, player):
    x = 0
    action = 0
    if approx_action_val(theta, dealer, player, hit) > approx_action_val(theta, dealer, player, stick):
        action = random.choices([hit, stick], [1 - 0.05/2, 0.05/2])[0]
    elif approx_action_val(theta, dealer, player, hit) < approx_action_val(theta, dealer, player, stick):
        action = random.choices([hit, stick], [0.05/2, 1 - 0.05/2])[0]
    else:
        action = random. choices([hit, stick], [0.5, 0.5])[0]

    return action

def LFA_sarsa_lambda(lambd, iter):
    theta = [0]*36  #theta is Q and Policy

    for i in range(iter):
        init()
        E = [0]*36
        d = dealer_sum
        rwd = 0

        p = player_sum
        p_next = 0
        action = get_action_from_Qw(theta, d, p)
        action_next = 0

        while(True):
            rwd, term = step(action)
    
            if term:
                td = rwd - approx_action_val(theta, d, p, action)
            else:
                p_next = player_sum
                action_next = get_action_from_Qw(theta, d, p_next)
                td = rwd + approx_action_val(theta, d, p_next, action_next) - approx_action_val(theta, d, p, action)
            
            for i in range(36):
                E[i] += feature_vec(d,p,action)[i]

            for k in range(36):
                theta[k] += 0.01*td*E[k]
                E[k] *= lambd

            if term:
                break

            p = p_next
            action = action_next
    
    return theta

# Set up grid and test data
x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]
y = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
X, Y = np.meshgrid(x, y)  # `plot_surface` expects `x` and `y` data to be 2D


for iter in iters:
    for lambd in lambdas:
        theta = LFA_sarsa_lambda(lambd, iter)

        Q = reset_list_w_action()
        V = reset_list_wo_action()
        Pol = reset_list_wo_action()

        for i in range(10):
            for j in range(21):
                for act in range(2):
                    Q[i][j][act] = approx_action_val(theta, i+1, j+1, act)
                V[i][j] = max(Q[i][j])
        for i in range(10):
            for j in range(21):
                Pol[i][j] = np.argmax(Q[i][j])

        data_val = np.array(V)
        data_pol = np.array(Pol)

        hf = plt.figure()
        ha = hf.add_subplot(111, projection='3d')

        ha.plot_surface(X, Y, data_val)

        plt.title("Action Value with LA : iter = " + str(iter) + ", lambda: " + str(lambd))
        plt.xlabel("Player sum")
        plt.ylabel("Dealer showing")

        plt.figure()
        plt.imshow(data_pol, interpolation = 'none', cmap = 'binary')
        
        plt.xticks(np.arange(len(x)), x)
        plt.yticks(np.arange(len(y)), y)

        plt.title("Policy with LA : iter = " + str(iter) + ", lambda: " + str(lambd))
        plt.colorbar()
        plt.xlabel("Player sum")
        plt.ylabel("Dealer showing")
        plt.text(18, 16, "white: hit, black: stick")

        plt.show()