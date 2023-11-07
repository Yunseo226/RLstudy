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


####### Q3. SARSA(lambda) ####### 

#lambdas = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
lambdas = [1]

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

def reset_policy():    
    Policy = []
    for i in range(10):
        list_tmp = []
        for j in range(21):
            list_tmp.append(0.5) #initial policy: randomly stochastic
        Policy.append(list_tmp)
    return Policy

def N_update(dealer_sum, player_sum, action):
    N[dealer_sum-1][player_sum-1][action] += 1

def action_val_update(dealer_sum, player_sum, action, target):
    if target != 0:
        Q[dealer_sum-1][player_sum-1][action] += target/N[dealer_sum-1][player_sum-1][action]

def e_greedy_update_indiv(dealer_sum, player_sum, e):
    Policy[dealer_sum - 1][player_sum - 1] = e/2 + np.argmax(Q[dealer_sum - 1][player_sum - 1])*(1-e)

def e_greedy_update(e):
    for i in range(10):
        for j in range(21):
            Policy[i][j] = e/2 + np.argmax(Q[i][j])*(1-e)

def follow_policy(dealer_sum, player_sum):
    stochastic_action = Policy[dealer_sum-1][player_sum-1]
    action = random.choices([hit, stick], [1 - stochastic_action, stochastic_action])[0]
    return action

for lambd in lambdas:
    Policy = reset_policy()
    N = reset_list_w_action()
    Q = reset_list_w_action()
    V = reset_list_wo_action()

    for i in range(10000):
        init()
        E = reset_list_w_action()
        d = dealer_sum
        rwd = 0
        p_sums = []

        p = player_sum
        p_next = 0
        action = follow_policy(d, p)
        action_next = 0

        while(True):
            p_sums.append(p)
            rwd, term = step(action)
    
            if term:
                td = rwd - Q[d-1][p-1][action]
            else:
                p_next = player_sum
                action_next = follow_policy(d, p_next)
                td = rwd + Q[d-1][p_next-1][action_next] - Q[d-1][p-1][action]
            
            N_update(d, p, action)
            E[d-1][p-1][action] += 1

            for i in range(10):
                for j in range(21):
                    for act in range(2):
                        action_val_update(i+1, j+1, act, td*E[i][j][act]) 
                        E[i][j][act] *= lambd

            if term:
                break

            p = p_next
            action = action_next

        p_sums = list(set(p_sums))

        for p in p_sums:
            e_greedy_update_indiv(d, p, 100/(100 + N[d-1][p-1][hit] + N[d-1][p-1][stick]))

    for i in range(10):
        for j in range(21):
            V[i][j] = max(Q[i][j])

    data_val = np.array(V)
    data_pol = np.array(Policy)

    # Set up grid and test data
    x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]
    y = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    hf = plt.figure()
    ha = hf.add_subplot(111, projection='3d')

    X, Y = np.meshgrid(x, y)  # `plot_surface` expects `x` and `y` data to be 2D
    ha.plot_surface(X, Y, data_val)

    plt.title("Value function : lambda = " + str(lambd))
    plt.xlabel("Player sum")
    plt.ylabel("Dealer showing")

    plt.figure()
    plt.imshow(data_pol, interpolation = 'none', cmap = 'binary')
    
    plt.xticks(np.arange(len(x)), x)
    plt.yticks(np.arange(len(y)), y)

    plt.title("Optimal Policy : lambda = " + str(lambd))
    plt.colorbar()
    plt.xlabel("Player sum")
    plt.ylabel("Dealer showing")
    plt.text(18, 16, "white: hit, black: stick")

plt.show()