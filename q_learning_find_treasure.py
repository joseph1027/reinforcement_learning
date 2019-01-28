import numpy as np
import pandas as pd
import time

np.random.seed(2)

N_STATE = 100
WIDTH = 10
LENGTH = 10 

ACTIONS = ['left','right','up','down']
EPSILON = 0.9
ALPHA = 0.1
GAMMA = 1.0
MAX_EPISODE = 1000
FRESH_TIME = 0.001
TARGET = 99

def build_q_table(n_states,actions):
    table = pd.DataFrame(np.zeros((n_states,len(actions))),columns = actions)
    print(table)
    return table

def choose_action(state,q_table):
    state_actions = q_table.iloc[state,:]
    if ((np.random.uniform()>EPSILON) or (state_actions.all()== 0)):
        action_name = np.random.choice(ACTIONS)
    else:
        action_name = state_actions.idxmax()
    return action_name

def get_env_feedback(state_now,action_now):
    reward = 0
    state_next = state_now

    if action_now == 'right':
        if(state_now % WIDTH == WIDTH-1):
            state_next = state_now
        else:
            state_next = state_now +1
    elif action_now == 'left':
        if(state_now % WIDTH == 0):
            state_next = state_now
        else:
            state_next = state_now -1
    elif action_now == 'up':
        if(int(state_now/WIDTH) == LENGTH-1):
            state_next = state_now
        else:
            state_next = state_now +WIDTH
    elif action_now == 'down':
        if(int(state_now/WIDTH) == 0):
            state_next = state_now
        else:
            state_next = state_now -WIDTH

    #print(state_now,action_now,state_next)
    if(state_next == TARGET):
        reward = 1

    return state_next,reward

def update_env(S,episode,step_counter):
    env_list = ((['-']*(WIDTH-1)+['-\n'])*(LENGTH-1))+(['-']*(WIDTH-1) + ['T\n'])
    if S == TARGET:
        interaction = 'Episode %s: total_steps = %s' % (episode+1, step_counter)
        print('\r{}'.format(interaction), end='')
        time.sleep(2)
        print('\r                                ', end='')
    else:
        if (S%WIDTH == (WIDTH-1)):
            env_list[S] = 'o\n'
        else:
            env_list[S] = 'o'
        interaction = ''.join(env_list)
        print('\r{}'.format(interaction), end='')
        time.sleep(FRESH_TIME)

def rl():
    q_table = build_q_table(N_STATE, ACTIONS)
    for episode in range(MAX_EPISODE):
        step_counter = 0
        S = 0
        is_terminated = False
        update_env(S, episode, step_counter)
        while not is_terminated:

            A = choose_action(S, q_table)
            S_, R = get_env_feedback(S, A)
            q_predict = q_table.loc[S, A]
            if S_ != TARGET:
                q_target = R + GAMMA * q_table.iloc[S_, :].max()
            else:
                q_target = R
                is_terminated = True

            q_table.loc[S, A] += ALPHA * (q_target - q_predict)
            S = S_

            update_env(S, episode, step_counter+1)

            step_counter += 1
    return q_table

if __name__ == '__main__':
    q_table = rl()
    print('\r\nQ-table:\n')
    print(q_table)
