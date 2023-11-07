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


####### Q2. Monte-carlo ####### 

N = []
Q = []
V = []

for i in range(10):
    list_tmp1 = []
    list_tmp2 = []
    list_tmp3 = []
    for j in range(21):
        list_tmp1.append([0.0,0.0])
        list_tmp2.append([0.0,0.0])
        list_tmp3.append(0.0)
    N.append(list_tmp1)
    Q.append(list_tmp2)
    V.append(list_tmp3)

Policy = []
for i in range(10):
    list_tmp = []
    for j in range(21):
        #list_tmp.append((j + 1) > 12 or (j + 1) < 4) # initial policy: hit if 4 <= sum <= 12
        #list_tmp.append(random.choice([0,1])) #initial policy: randomly deterministic
        list_tmp.append(0.5) #initial policy: randomly stochastic
    Policy.append(list_tmp)

def N_update(dealer_sum, player_sum, action):
    N[dealer_sum-1][player_sum-1][action] += 1

def action_val_update(dealer_sum, player_sum, action, goal):
    Q[dealer_sum-1][player_sum-1][action] += (goal - Q[dealer_sum-1][player_sum-1][action])/N[dealer_sum-1][player_sum-1][action]

def state_val_update(dealer_sum, player_sum, goal):
    V[dealer_sum-1][player_sum-1] += (goal - V[dealer_sum-1][player_sum-1])/(N[dealer_sum-1][player_sum-1][0] + N[dealer_sum-1][player_sum-1][1])

def e_greedy_update_indiv(dealer_sum, player_sum, e):
    Policy[dealer_sum - 1][player_sum - 1] = e/2 + np.argmax(Q[dealer_sum - 1][player_sum - 1])*(1-e)

def e_greedy_update(e):
    for i in range(10):
        for j in range(21):
            Policy[i][j] = e/2 + np.argmax(Q[i][j])*(1-e)


for i in range(100000):
    init()

    d = dealer_sum
    p_sums = []
    actions = []
    rwd = 0

    while(True):
        stochastic_action = Policy[d-1][player_sum-1]
        action = random.choices([hit, stick], [1 - stochastic_action, stochastic_action])[0]

        if player_sum not in p_sums:  #First-visit MC
            p_sums.append(player_sum)
            actions.append(action)

        rwd, term = step(action)
        if term:
            break


    for p, act in zip(p_sums, actions):
        N_update(d, p, act)
        action_val_update(d, p, act, rwd)
        #state_val_update(d, p, rwd)
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

plt.title("Value function")
plt.xlabel("Player sum")
plt.ylabel("Dealer showing")

plt.figure()
plt.imshow(data_pol, interpolation = 'none', cmap = 'binary')
  
plt.xticks(np.arange(len(x)), x)
plt.yticks(np.arange(len(y)), y)

plt.title("Optimal Policy")
plt.colorbar()
plt.xlabel("Player sum")
plt.ylabel("Dealer showing")
plt.text(18, 16, "white: hit, black: stick")
plt.show()