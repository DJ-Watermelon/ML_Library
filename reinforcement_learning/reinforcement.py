import random
import numpy as np
import sys
import matplotlib.pyplot as plt
from tqdm import tqdm

class Sender:
    """
    A Q-learning agent that sends messages to a Receiver

    """

    def __init__(self, num_sym:int, grid_rows:int, grid_cols:int, alpha_i:float, alpha_f:float, num_ep:int, epsilon:float, discount:float):
        """
        Initializes this agent with a state, set of possible actions, and a means of storing Q-values

        :param num_sym: The number of arbitrary symbols available for sending
        :type num_sym: int
        :param grid_rows: The number of rows in the grid
        :type grid_rows: int
        :param grid_cols: The number of columns in the grid
        :type grid_cols: int
        :param alpha_i: The initial learning rate
        :type alpha: float
        :param alpha_f: The final learning rate
        :type alpha: float
        :param num_ep: The total number of episodes
        :type num_ep: int
        :param epsilon: The epsilon in epsilon-greedy exploration
        :type epsilon: float
        :param discount: The discount factor
        :type discount: float
        """
        self.actions = range(num_sym)
        self.alpha = alpha_i
        self.alpha_i = alpha_i
        self.alpha_f = alpha_f
        self.num_ep = num_ep
        self.epsilon = epsilon
        self.discount = discount
        self.q_vals = np.zeros((grid_rows, grid_cols, num_sym)) # your code here!
        
    def select_action(self, state):
        """
        This function is called every time the agent must act. It produces the action that the agent will take
        based on its current state

        :param state: the state the agent is acting from, in the form (x,y), which are the coordinates of the prize
        :type state: (int, int)
        :return: The symbol to be transmitted (must be an int < N)
        :rtype: int
        """
        # Your code here!
        x, y = state
        if np.random.binomial(1, self.epsilon):
            # pick random action
            return np.random.choice(self.actions)
        else:
            # pick best action
            return np.argmax(self.q_vals[x,y])

    def update_q(self, old_state, action, reward):
        """
        This function is called after an action is resolved so that the agent can update its Q-values

        :param old_state: the state the agent was in when it acted, in the form (x,y), which are the coordinates
                          of the prize
        :type old_state: (int, int)
        :param action: the action that was taken
        :type action: int
        :param reward: the reward that was received
        :type reward: float
        """
        # Your code here!
        x,y = old_state
        self.q_vals[x,y,action] = (1-self.alpha) * self.q_vals[x,y,action] + self.alpha * reward 
        # enters random state
        return


class Receiver:
    """
    A Q-learning agent that receives a message from a Sender and then navigates a grid

    """

    def __init__(self, num_sym:int, grid_rows:int, grid_cols:int, alpha_i:float, alpha_f:float, num_ep:int, epsilon:float, discount:float):
        """
        Initializes this agent with a state, set of possible actions, and a means of storing Q-values

        :param num_sym: The number of arbitrary symbols available for sending
        :type num_sym: int
        :param grid_rows: The number of rows in the grid
        :type grid_rows: int
        :param grid_cols: The number of columns in the grid
        :type grid_cols: int
        :param alpha_i: The initial learning rate
        :type alpha: float
        :param alpha_f: The final learning rate
        :type alpha: float
        :param num_ep: The total number of episodes
        :type num_ep: int
        :param epsilon: The epsilon in epsilon-greedy exploration
        :type epsilon: float
        :param discount: The discount factor
        :type discount: float
        """
        self.actions = [0,1,2,3] 
        self.alpha = alpha_i
        self.alpha_i = alpha_i
        self.alpha_f = alpha_f
        self.num_ep = num_ep
        self.epsilon = epsilon
        self.discount = discount
        self.q_vals = np.zeros((num_sym, grid_rows, grid_cols, 4)) # Your code here!

    def select_action(self, state):
        """
        This function is called every time the agent must act. It produces the action that the agent will take
        based on its current state
        :param state: the state the agent is acting from, in the form (m,x,y), where m is the message received
                      and (x,y) are the board coordinates
        :type state: (int, int, int)
        :return: The direction to move, where 0 is up, 1 is down, 2 is left, and 3 is right
        :rtype: int
        """
        # Your code here!
        m, x, y = state
        if np.random.binomial(1, self.epsilon):
            # pick random action
            return np.random.choice(self.actions)
        else:
            # pick best action
            return np.argmax(self.q_vals[m,x,y])
        return 0

    def update_q(self, old_state, new_state, action, reward):
        """
        This function is called after an action is resolved so that the agent can update its Q-values

        :param old_state: the state the agent was in when it acted in the form (m,x,y), where m is the message received
                          and (x,y) are the board coordinates
        :type old_state: (int, int, int)
        :param new_state: the state the agent entered after it acted
        :type new_state: (int, int, int)
        :param action: the action that was taken
        :type action: int
        :param reward: the reward that was received
        :type reward: float
        """
        # Your code here!
        m0,x0,y0 = old_state
        m,x,y = new_state
        self.q_vals[m0,x0,y0, action] = (1-self.alpha) * self.q_vals[m0,x0,y0, action] \
                                + self.alpha * (reward + self.discount * np.max(self.q_vals[m,x,y]))
        return


def get_grid(grid_name:str):
    """
    This function produces one of the three grids defined in the assignment as a nested list

    :param grid_name: the name of the grid. Should be one of 'fourroom', 'maze', or 'empty'
    :type grid_name: str
    :return: The corresponding grid, where True indicates a wall and False a space
    :rtype: list[list[bool]]
    """
    grid = [[False for i in range(5)] for j in range(5)] # default case is 'empty'
    if grid_name == 'fourroom':
        grid[0][2] = True
        grid[2][0] = True
        grid[2][1] = True
        grid[2][3] = True
        grid[2][4] = True
        grid[4][2] = True
    elif grid_name == 'maze':
        grid[1][1] = True
        grid[1][2] = True
        grid[1][3] = True
        grid[2][3] = True
        grid[3][1] = True
        grid[4][1] = True
        grid[4][2] = True
        grid[4][3] = True
        grid[4][4] = True
    return grid


def legal_move(posn_x:int, posn_y:int, move_id:int, grid:list[list[bool]]):
    """
    Produces the new position after a move starting from (posn_x,posn_y) if it is legal on the given grid (i.e. not
    out of bounds or into a wall)

    :param posn_x: The x position (column) from which the move originates
    :type posn_x: int
    :param posn_y: The y position (row) from which the move originates
    :type posn_y: int
    :param move_id: The direction to move, where 0 is up, 1 is down, 2 is left, and 3 is right
    :type move_id: int
    :param grid: The grid on which to move, where False indicates a space and True a wall
    :type grid: list[list[bool]]
    :return: The new (x,y) position if the move was legal, or the old position if it was not
    :rtype: (int, int)
    """
    moves = [[0,-1],[0,1],[-1,0],[1,0]] # left, right, up, down
    new_x = posn_x + moves[move_id][0]
    new_y = posn_y + moves[move_id][1]
    result = (new_x,new_y)
    if new_x < 0 or new_y < 0 or new_x >= len(grid[0]) or new_y >= len(grid):
        result = (posn_x,posn_y)
    else:
        if grid[new_y][new_x]:
            result = (posn_x,posn_y)
    return result


def run_episodes(sender:Sender, receiver:Receiver, grid:list[list[bool]], num_ep:int, delta:float):
    """
    Runs the reinforcement learning scenario for the specified number of episodes

    :param sender: The Sender agent
    :type sender: Sender
    :param receiver: The Receiver agent
    :type receiver: Receiver
    :param grid: The grid on which to move, where False indicates a space and True a wall
    :type grid: list[list[bool]]
    :param num_ep: The number of episodes
    :type num_ep: int
    :param delta: The chance of termination after every step of the receiver
    :type delta: float [0,1]
    :return: A list of the reward received by each agent at the end of every episode
    :rtype: list[float]
    """
    reward_vals = []
    sender_alpha = np.linspace(sender.alpha_i, sender.alpha_f, num_ep)
    receiver_alpha = np.linspace(receiver.alpha_i, receiver.alpha_f, num_ep)

    # Episode loop
    for ep in tqdm(range(num_ep)):
        
        # Set receiver starting position
        receiver_x = 2
        receiver_y = 2

        # Choose prize position
        prize_x = np.random.randint(len(grid[0]))
        prize_y = np.random.randint(len(grid))
        while grid[prize_y][prize_x] or (prize_x == receiver_x and prize_y == receiver_y):
            prize_x = np.random.randint(len(grid[0]))
            prize_y = np.random.randint(len(grid))

        # Initialize new episode
        # (sender acts), set alpha
        # Your code here!
        sender.alpha = sender_alpha[ep]
        receiver.alpha = receiver_alpha[ep]
        
        sym = sender.select_action((prize_x, prize_y))

        # Receiver loop
        # (receiver acts, check for prize, check for random termination, update receiver Q-value)
        terminate = False
        t = 0
        while not terminate:
            # Your code here!
            t += 1
            move = receiver.select_action((sym, receiver_x, receiver_y))
            new_loc = legal_move(receiver_x,receiver_y, move, grid) 
            reward = 1 if new_loc == (prize_x, prize_y) else 0 
            
            receiver.update_q((sym, receiver_x, receiver_y), (sym, ) + new_loc, move, reward)
            # update location
            receiver_x, receiver_y = new_loc
            if reward:
                break
            terminate = np.random.binomial(1, delta)
        #Finish up episode
        # (update sender Q-value, update alpha values, append reward to output list)
        # Your code here!
        sender.update_q((prize_x,prize_y), sym, reward)
        reward_vals.append(reward * sender.discount** t)
    return reward_vals


if __name__ == "__main__":

    ## Part 2a ##
    # Define parameters here
    num_learn_episodes_lst = np.array([10, 100, 1000, 10000, 50000, 100000])
    num_test_episodes = 1000
    grid_name = 'fourroom'
    grid = get_grid(grid_name)
    num_signals = 4
    discount = 0.95
    delta = 1 - discount
    epsilon_lst = [0.01, 0.1, 0.4]
    alpha_init = 0.9
    alpha_final = 0.01
    reward_mean = np.zeros((len(epsilon_lst), len(num_learn_episodes_lst)))
    reward_std = np.zeros((len(epsilon_lst), len(num_learn_episodes_lst)))
    for i,epsilon in enumerate(epsilon_lst):
        for j, num_learn_episodes in enumerate(num_learn_episodes_lst):
            discount_test_reward = np.zeros(10)
            for k in range(10):
                # Initialize agents
                sender = Sender(num_signals, len(grid), len(grid[0]), alpha_init, alpha_final, num_learn_episodes, epsilon, discount)
                receiver = Receiver(num_signals, len(grid), len(grid[0]), alpha_init, alpha_final, num_learn_episodes, epsilon, discount)

                # Learn
                run_episodes(sender, receiver, grid, num_learn_episodes, delta)

                # Test
                sender.epsilon = 0.0
                sender.alpha = 0.0
                sender.alpha_i = 0.0
                sender.alpha_f = 0.0
                receiver.epsilon = 0.0
                receiver.alpha = 0.0
                receiver.alpha_i = 0.0
                receiver.alpha_f = 0.0
                test_rewards = run_episodes(sender, receiver, grid, num_test_episodes, delta)
                discount_test_reward[k] = np.mean(test_rewards) 
            reward_mean[i,j] = np.mean(discount_test_reward)
            reward_std[i,j] = np.std(discount_test_reward)
    plt.errorbar(np.log10(num_learn_episodes_lst), reward_mean[0], yerr=reward_std[0], capsize=5, label=r"$\epsilon=0.01$")
    plt.errorbar(np.log10(num_learn_episodes_lst), reward_mean[1], yerr=reward_std[1], capsize=5, label=r"$\epsilon=0.1$")
    plt.errorbar(np.log10(num_learn_episodes_lst), reward_mean[2], yerr=reward_std[2], capsize=5, label=r"$\epsilon=0.4$")
    plt.xlabel(r"$\log_{10}(N_{ep})$")
    plt.ylabel("Average Test Discount Reward") 
    plt.title(r"Average Test Discount Reward Over Different $\epsilon$ and $N_{ep}$")
    plt.legend()
    plt.grid(True)
    plt.show()
    
    
    
    ## Part 2b ##
    
    num_learn_episodes = 100000
    num_test_episodes = 1000
    grid_name = 'fourroom'
    grid = get_grid(grid_name)
    num_signals = 4
    discount = 0.95
    delta = 1 - discount
    epsilon = 0.1
    alpha_init = 0.9
    alpha_final = 0.01
    
    # Initialize agents
    sender = Sender(num_signals, len(grid), len(grid[0]), alpha_init, alpha_final, num_learn_episodes, epsilon, discount)
    receiver = Receiver(num_signals, len(grid), len(grid[0]), alpha_init, alpha_final, num_learn_episodes, epsilon, discount)

    # Learn
    run_episodes(sender, receiver, grid, num_learn_episodes, delta)
    for i in range(4):
        print("Symbol", i)
        print(np.argmax(receiver.q_vals[i], axis=2).T)
    
    print("Policy", np.argmax(sender.q_vals, axis=2).T)
    
    # Part 3
    
    num_learn_episodes_lst = np.array([10, 100, 1000, 10000, 50000, 100000])
    num_test_episodes = 1000
    grid_name = 'fourroom'
    grid = get_grid(grid_name)
    num_signals_lst = [2,4,10]
    discount = 0.95
    delta = 1 - discount
    epsilon = 0.1
    alpha_init = 0.9
    alpha_final = 0.01
    reward_mean = np.zeros((len(num_signals_lst), len(num_learn_episodes_lst)))
    reward_std = np.zeros((len(num_signals_lst), len(num_learn_episodes_lst)))
    for i,num_signals in enumerate(num_signals_lst):
        for j, num_learn_episodes in enumerate(num_learn_episodes_lst):
            discount_test_reward = np.zeros(10)
            for k in range(10):
                # Initialize agents
                sender = Sender(num_signals, len(grid), len(grid[0]), alpha_init, alpha_final, num_learn_episodes, epsilon, discount)
                receiver = Receiver(num_signals, len(grid), len(grid[0]), alpha_init, alpha_final, num_learn_episodes, epsilon, discount)

                # Learn
                run_episodes(sender, receiver, grid, num_learn_episodes, delta)

                # Test
                sender.epsilon = 0.0
                sender.alpha = 0.0
                sender.alpha_i = 0.0
                sender.alpha_f = 0.0
                receiver.epsilon = 0.0
                receiver.alpha = 0.0
                receiver.alpha_i = 0.0
                receiver.alpha_f = 0.0
                test_rewards = run_episodes(sender, receiver, grid, num_test_episodes, delta)
                discount_test_reward[k] = np.mean(test_rewards) 
            reward_mean[i,j] = np.mean(discount_test_reward)
            reward_std[i,j] = np.std(discount_test_reward)
    plt.errorbar(np.log10(num_learn_episodes_lst), reward_mean[0], yerr=reward_std[0], capsize=5, label=r"$N=2$")
    plt.errorbar(np.log10(num_learn_episodes_lst), reward_mean[1], yerr=reward_std[1], capsize=5, label=r"$N=4$")
    plt.errorbar(np.log10(num_learn_episodes_lst), reward_mean[2], yerr=reward_std[2], capsize=5, label=r"$N=10$")
    plt.xlabel(r"$\log_{10}(N_{ep})$")
    plt.ylabel("Average Test Discount Reward") 
    plt.title(r"Average Test Discount Reward Over Different N and $N_{ep}$")
    plt.legend()
    plt.grid(True)
    plt.show()
    
    
    # Part 4
    
    num_learn_episodes_lst = np.array([10, 100, 1000, 10000, 50000, 100000])
    num_test_episodes = 1000
    grid_name = 'maze'
    grid = get_grid(grid_name)
    num_signals_lst = [2,3,5]
    discount = 0.95
    delta = 1 - discount
    epsilon = 0.1
    alpha_init = 0.9
    alpha_final = 0.01
    reward_mean = np.zeros((len(num_signals_lst), len(num_learn_episodes_lst)))
    reward_std = np.zeros((len(num_signals_lst), len(num_learn_episodes_lst)))
    for i,num_signals in enumerate(num_signals_lst):
        for j, num_learn_episodes in enumerate(num_learn_episodes_lst):
            discount_test_reward = np.zeros(10)
            for k in range(10):
                # Initialize agents
                sender = Sender(num_signals, len(grid), len(grid[0]), alpha_init, alpha_final, num_learn_episodes, epsilon, discount)
                receiver = Receiver(num_signals, len(grid), len(grid[0]), alpha_init, alpha_final, num_learn_episodes, epsilon, discount)

                # Learn
                run_episodes(sender, receiver, grid, num_learn_episodes, delta)

                # Test
                sender.epsilon = 0.0
                sender.alpha = 0.0
                sender.alpha_i = 0.0
                sender.alpha_f = 0.0
                receiver.epsilon = 0.0
                receiver.alpha = 0.0
                receiver.alpha_i = 0.0
                receiver.alpha_f = 0.0
                test_rewards = run_episodes(sender, receiver, grid, num_test_episodes, delta)
                discount_test_reward[k] = np.mean(test_rewards) 
            reward_mean[i,j] = np.mean(discount_test_reward)
            reward_std[i,j] = np.std(discount_test_reward)
    plt.errorbar(np.log10(num_learn_episodes_lst), reward_mean[0], yerr=reward_std[0], capsize=5, label="N=2")
    plt.errorbar(np.log10(num_learn_episodes_lst), reward_mean[1], yerr=reward_std[1], capsize=5, label="N=3")
    plt.errorbar(np.log10(num_learn_episodes_lst), reward_mean[2], yerr=reward_std[2], capsize=5, label="N=5")
    plt.xlabel(r"$\log_{10}(N_{ep})$")
    plt.ylabel("Average Test Discount Reward") 
    plt.title(r"Average Test Discount Reward Over Different N and $N_{ep}$")
    plt.legend()
    plt.grid(True)
    plt.show()
    
    
    # Part 5
    num_learn_episodes_lst = np.array([10, 100, 1000, 10000, 50000, 100000])
    num_test_episodes = 1000
    grid_name = 'empty'
    grid = get_grid(grid_name)
    num_signals = 1
    discount = 0.95
    delta = 1 - discount
    epsilon = 0.1
    alpha_init = 0.9
    alpha_final = 0.01
    reward_mean = np.zeros(len(num_learn_episodes_lst))
    reward_std = np.zeros(len(num_learn_episodes_lst))

    for j, num_learn_episodes in enumerate(num_learn_episodes_lst):
        discount_test_reward = np.zeros(10)
        for k in range(10):
            # Initialize agents
            sender = Sender(num_signals, len(grid), len(grid[0]), alpha_init, alpha_final, num_learn_episodes, epsilon, discount)
            receiver = Receiver(num_signals, len(grid), len(grid[0]), alpha_init, alpha_final, num_learn_episodes, epsilon, discount)

            # Learn
            run_episodes(sender, receiver, grid, num_learn_episodes, delta)

            # Test
            sender.epsilon = 0.0
            sender.alpha = 0.0
            sender.alpha_i = 0.0
            sender.alpha_f = 0.0
            receiver.epsilon = 0.0
            receiver.alpha = 0.0
            receiver.alpha_i = 0.0
            receiver.alpha_f = 0.0
            test_rewards = run_episodes(sender, receiver, grid, num_test_episodes, delta)
            discount_test_reward[k] = np.mean(test_rewards) 
        reward_mean[j] = np.mean(discount_test_reward)
        reward_std[j] = np.std(discount_test_reward)
    plt.errorbar(np.log10(num_learn_episodes_lst), reward_mean, yerr=reward_std, capsize=5, label="N=1")
    plt.xlabel(r"$\log_{10}(N_{ep})$")
    plt.ylabel("Average Test Discount Reward") 
    plt.title(r"Average Test Discount Reward Over Different $N_{ep}$")
    plt.legend()
    plt.grid(True)
    plt.show()
