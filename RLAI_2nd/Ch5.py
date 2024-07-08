#!/usr/bin/env python
# coding: utf-8

# In[13]:


from IPython.display import display, HTML

# Set the notebook width to 80%
display(HTML("<style>.container { width: 80% !important; }</style>"))


# In[2]:


get_ipython().system('jupyter notebook list')


# In[3]:


# Needs to paste `http://localhost:3110`, no ending `/`
port = 2770

import IPython
import json
import requests

hostname = get_ipython().getoutput('hostname')

# Get the current Jupyter server's info
result = get_ipython().getoutput('jupyter notebook list')
for res in result:
    if f'http://localhost:{port}/' in res:
        result = res.split(' :: ')[0]
        break

# Print the server URL
print(f'Current Jupyter server {hostname} URL: {result}')

# Get the list of running notebooks
response = requests.get(f'{result}api/sessions')

# # Convert the JSON data to a string and print it
# print(json.dumps(response.json(), indent=4))

nbs = response.json()
nb_names = [nb['name'] for nb in nbs]
print(len(nb_names), nb_names)


# # Import dep

# In[4]:


from itertools import product
import math
import time
from joblib import Parallel, delayed, parallel_backend

import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'notebook')


# In[5]:


import plotly
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import plotly.io as pio

pio.renderers.default = "notebook"


# In[6]:


COLOR_LIST = plotly.colors.DEFAULT_PLOTLY_COLORS
len(COLOR_LIST)


# # Example 5.1: Blackjack
# - Episodic Monte Carlo for policy evaluation.
# - Evaluate a fixed policy from the dealer and player.

# In[7]:


# # ChatGPT version
# def Monte_Carlo_sim_blackjack_0(n_episodes=1000000):
#     """
#     Monte Carlo simulation for Blackjack game.
    
#     Parameters
#     ----------
#     n_episodes : int
#         Number of episodes to simulate.
        
#     Returns
#     -------
#     Q : dict
#         Dictionary of state-action values.
#     N : dict
#         Dictionary of state-action visit counts.
#     """
#     # Initialize dictionaries of state-action values and visit counts
#     Q = {}
#     N = {}
    
#     # Loop over episodes
#     for i in range(n_episodes):
#         # Initialize an empty list to store state-action pairs
#         episode = []
        
#         # Initialize the state
#         state = (np.random.randint(12, 22), np.random.randint(1, 11), False)
        
#         # Loop over steps in the episode
#         while True:
#             # If the state is not in the dictionary, add it
#             if state not in Q:
#                 Q[state] = {}
#                 N[state] = {}
#                 for action in range(2):
#                     Q[state][action] = 0
#                     N[state][action] = 0
            
#             # Choose an action
#             action = np.random.randint(2)
            
#             # Append the state-action pair to the episode
#             episode.append((state, action))
            
#             # Increment the visit count for the state-action pair
#             N[state][action] += 1
            
#             # Take the action
#             player_sum, dealer_card, usable_ace = state
#             if action == 1:
#                 player_sum += np.random.randint(1, 11)
#                 if player_sum > 21:
#                     if usable_ace:
#                         player_sum -= 10
#                         usable_ace = False
#                     else:
#                         break
#             else:
#                 break
            
#             # Update the state
#             state = (player_sum, dealer_card, usable_ace)
        
#         # Loop over state-action pairs in the episode
#         for state, action in episode:
#             # Calculate the return
#             G = 1 if state[0] > 21 else 1.5 if state[0] == 21 else 0
#             # Update the state-action value
#             Q[state][action] += (G - Q[state][action]) / N[state][action]
    
#     return Q, N
    


# In[27]:


N_JOBS = 20

def organize_state_val(state_val_pair):
    
    def _update_state_val(state, val):
        o_val, o_cnt = state_val[state]
        state_val[state] = (o_val+(val-o_val)/(o_cnt+1), o_cnt+1)
        
    # Initial state values and cnts
    state_val = {}
    for i in range(1, 11):
        for j in range(12, 22):
            state_val[i, j, 1] = (0, 0)         
            state_val[i, j, 0] = (0, 0)
    
    for state, val in state_val_pair:
        _update_state_val(state, val)
        
    state_val_0 = np.zeros((10, 10))
    state_val_cnt_0 = np.zeros((10, 10), dtype=int)
    state_val_1 = np.zeros((10, 10))
    state_val_cnt_1 = np.zeros((10, 10), dtype=int)
    for i in range(1, 11):
        for j in range(12, 22):
            state_val_0[i-1, j-12] = state_val[i, j, 0][0]
            state_val_cnt_0[i-1, j-12] = state_val[i, j, 0][1]
            state_val_1[i-1, j-12] = state_val[i, j, 1][0]
            state_val_cnt_1[i-1, j-12] = state_val[i, j, 1][1]
        
    return np.array(state_val_0), np.array(state_val_1), np.array(state_val_cnt_0), np.array(state_val_cnt_1)
    

def Monte_Carlo_sim_blackjack_1(n_ep, dealer_thre=17, player_thre=20, n_jobs=1, verbose=0):
    n_suit = 13
    card_suit = range(1, 1+n_suit)
    
    def _ini_card_val(card):
        return min(card, 10)+10*(card==1)
    
    def _card_val(card):
        return min(card, 10)
    
    def _hits_or_sticks_round(player_cards, stick_thre):
        # Initial dealing
        player_sum = sum(_ini_card_val(card) for card in player_cards)
        usable_ace = int(np.any([card==1 for card in player_cards]))
        if usable_ace:
            if player_sum>21:
                player_sum -= 10
        else:
            while player_sum<12:
                new_card = np.random.choice(card_suit, 1)[0]
                if new_card==1:
                    if player_sum>=11:
                        player_sum += 1
                    else:
                        player_sum += 11
                        usable_ace = 1
                else:
                    player_sum += _card_val(new_card)
                    
        assert player_sum<=21, "Currently sum should <= 21"
        initial_sum = player_sum
        
        # Hits or sticks
        can_use_ace = usable_ace
        while player_sum<stick_thre:
            player_sum += _card_val(np.random.choice(card_suit, 1)[0])
            if player_sum>21 and can_use_ace:
                player_sum -= 10
                can_use_ace = 0
                
        return initial_sum, usable_ace, player_sum
    
    def _one_episode():
        # Initial dealing
        dealer_show = _card_val(np.random.choice(card_suit, 1)[0])
        
        player_cards = list(np.random.choice(card_suit, 2))
        player_sum, usable_ace, final_player_sum = _hits_or_sticks_round(player_cards, player_thre)
        
        state = (dealer_show, player_sum, usable_ace)
        if final_player_sum>21:
            reward = -1
        else:
            dealer_sum = _ini_card_val(dealer_show)
            had_ace = d_usable_ace = dealer_show==1
            while dealer_sum<dealer_thre:
                new_card = np.random.choice(card_suit, 1)[0]
                if had_ace:
                    dealer_sum += _card_val(new_card)
                else:
                    dealer_sum += _ini_card_val(new_card)
                    if new_card==1:
                        had_ace = d_usable_ace = 1
                if dealer_sum>21 and d_usable_ace:
                    dealer_sum -= 10
                    d_usable_ace = 0
                if dealer_sum>final_player_sum:
                    break
            if dealer_sum>21:
                reward = 1
            else:
                reward = np.sign(final_player_sum-dealer_sum)
            
        return state, reward
    
    with parallel_backend('loky', n_jobs=n_jobs):
        res = Parallel(verbose=verbose, pre_dispatch="1.5*n_jobs")(
            delayed(_one_episode)() for _ in range(n_ep)
        )
    
    return organize_state_val(res)


# We don't care about how we reach the state, just start with the states
def Monte_Carlo_sim_blackjack_2(n_ep, dealer_thre=17, player_thre=20, n_jobs=1, verbose=0):
    n_suit = 13
    card_suit = range(1, 1+n_suit)
    
    def _ini_card_val(card):
        return min(card, 10)+10*(card==1)
    
    def _card_val(card):
        return min(card, 10)
    
    def _hits_or_sticks_round(state, stick_thre):
        _, player_sum, usable_ace = state
        # Hits or sticks
        while player_sum<stick_thre:
            player_sum += _card_val(np.random.choice(card_suit, 1)[0])
            if player_sum>21 and usable_ace:
                player_sum -= 10
                usable_ace = 0
                
        return player_sum
    
    def _one_episode():
        state = (
            np.random.choice(range(1, 11), 1)[0],
            np.random.choice(range(12, 22), 1)[0],
            np.random.choice(range(2), 1)[0]
        )
        
        dealer_show = state[0]
        
        final_player_sum = _hits_or_sticks_round(state, player_thre)
        
        if final_player_sum>21:
            reward = -1
        else:
            dealer_sum = _ini_card_val(dealer_show)
            had_ace = d_usable_ace = dealer_show==1
            while dealer_sum<dealer_thre:
                new_card = np.random.choice(card_suit, 1)[0]
                if had_ace:
                    dealer_sum += _card_val(new_card)
                else:
                    dealer_sum += _ini_card_val(new_card)
                    if new_card==1:
                        had_ace = d_usable_ace = 1
                if dealer_sum>21 and d_usable_ace:
                    dealer_sum -= 10
                    d_usable_ace = 0
                if dealer_sum>final_player_sum:
                    break
            if dealer_sum>21:
                reward = 1
            else:
                reward = np.sign(final_player_sum-dealer_sum)
        
        return state, reward
    
    with parallel_backend('loky', n_jobs=n_jobs):
        res = Parallel(verbose=verbose, pre_dispatch="1.5*n_jobs")(
            delayed(_one_episode)() for _ in range(n_ep)
        )
                
    return organize_state_val(res)

def plot_arr_bj_state_val(state_val_res, postfix):
    state_val_0, state_val_1, state_val_cnt_0, state_val_cnt_1 = state_val_res
    fig = make_subplots(rows=2, cols=2, horizontal_spacing=0.2, subplot_titles=('No Ace', 'With Ace', 'Count No Ace', 'Count With Ace'))
    fig.add_trace(
        go.Heatmap(
            z=state_val_0.T,
            colorscale='Jet',
            x=[i for i in range(1, 11)],
            y=[j for j in range(12, 22)],
            hoverongaps=False,
            zmin=-1,
            zmax=1,
            colorbar=dict(title="Val", x=0.42, y=0.8, len=0.4)  # Adjust x to position the color bar between subplots
        ), row=1, col=1
    )
    fig.add_trace(
        go.Heatmap(
            z=state_val_1.T,
            colorscale='Jet',
            x=[i for i in range(1, 11)],
            y=[j for j in range(12, 22)],
            hoverongaps=False,
            zmin=-1,
            zmax=1,
            colorbar=dict(title="Val", x=1.02, y=0.8, len=0.4)  # Adjust x to position the color bar between subplots
        ), row=1, col=2
    )
    
    fig.add_trace(
        go.Heatmap(
            z=state_val_cnt_0.T,
            colorscale='Jet',
            x=[i for i in range(1, 11)],
            y=[j for j in range(12, 22)],
            hoverongaps=False,
            colorbar=dict(title="Cnt", x=0.42, y=0.2, len=0.4)  # Adjust x to position the color bar between subplots
        ), row=2, col=1
    )
    fig.add_trace(
        go.Heatmap(
            z=state_val_cnt_1.T,
            colorscale='Jet',
            x=[i for i in range(1, 11)],
            y=[j for j in range(12, 22)],
            hoverongaps=False,
            colorbar=dict(title="Cnt", x=1.02, y=0.2, len=0.4)  # Adjust x to position the color bar between subplots
        ), row=2, col=2
    )

    fig.update_layout(
        title=f'State-Value Heatmaps {postfix}',
        xaxis=dict(title='Dealer showing', showgrid=True, gridwidth=1, gridcolor='black'),
        yaxis=dict(title='Player sum', showgrid=True, gridwidth=1, gridcolor='black'),
        xaxis2=dict(title='Dealer showing', showgrid=True, gridwidth=1, gridcolor='black'),
        yaxis2=dict(title='Player sum', showgrid=True, gridwidth=1, gridcolor='black'),
        xaxis3=dict(title='Dealer showing', showgrid=True, gridwidth=1, gridcolor='black'),
        yaxis3=dict(title='Player sum', showgrid=True, gridwidth=1, gridcolor='black'),
        xaxis4=dict(title='Dealer showing', showgrid=True, gridwidth=1, gridcolor='black'),
        yaxis4=dict(title='Player sum', showgrid=True, gridwidth=1, gridcolor='black'),
        autosize=False,
        width=1000,
        height=1000,
    )

    fig.show()
    
    
def plot_arr_bj_state_val_3d(state_val_arr, postfix):
    # Assuming state_val_arr is your 2D array of state values
    # Generate a meshgrid for your state dimensions
    # For example, if player's sum ranges from 12 to 21 and dealer's showing card from 1 to 10
    player_sum = np.arange(12, 22)  # Player's sum range
    dealer_showing = np.arange(1, 11)  # Dealer's showing card range
    X, Y = np.meshgrid(dealer_showing, player_sum)

    # Create a 3D surface plot
    fig = go.Figure(data=[
        go.Surface(
        z=state_val_arr, x=X, y=Y,
            colorbar=dict(
                title='Value',  # Title of the colorbar
                titleside='right',
                tickmode='array',
                tickvals=[-1, 1],  # Custom tick marks
                ticktext=['-1', '1'],  # Custom tick text
            )    
        )
    ])

    # Customize the layout
    fig.update_layout(
        title=f'State-Value {postfix}', autosize=False,
        scene=dict(
            xaxis_title='Dealer Showing',
            yaxis_title='Player Sum',
            zaxis=dict(range=(-1, 1), title='State Value')
        ),
        # scene=dict(
        #     xaxis=dict(
        #         title='Dealer Showing',
        #         showgrid=True  # Show grid lines on the x-axis
        #     ),
        #     yaxis=dict(
        #         title='Player Sum',
        #         showgrid=True  # Show grid lines on the y-axis
        #     ),
        #     zaxis=dict(
        #         range=(-1, 1),
        #         title='State Value',
        #         showgrid=True  # Show grid lines on the z-axis
        #     )
        # ),
        width=700, height=700,
        margin=dict(l=25, r=20, b=25, t=40),
        scene_camera=dict(
            up=dict(x=0, y=0, z=1),  # Sets the upward direction
            center=dict(x=0, y=0, z=0),  # Centers the view
            eye=dict(x=2, y=-2, z=2)  # Positions the camera view point
        )
    )

    # Show the plot
    fig.show()


# In[8]:


ls_n_ep = [10_000, 500_000, 1_000_000]


# ## Func 1

# In[22]:


get_ipython().run_cell_magic('time', '', 'ls_state_val_1 = []\nfor n_ep in ls_n_ep:\n    state_val_1 = Monte_Carlo_sim_blackjack_1(n_ep=n_ep)\n    ls_state_val_1.append(state_val_1)\n')


# In[36]:


for n_ep, state_val_1 in zip(ls_n_ep, ls_state_val_1):
    plot_arr_bj_state_val(state_val_1, f'(n_ep={n_ep})')


# In[ ]:


get_ipython().run_cell_magic('time', '', 'ls_state_val_1_paral = []\nfor n_ep in ls_n_ep:\n    state_val_1_paral = Monte_Carlo_sim_blackjack_1(n_ep=n_ep, n_jobs=N_JOBS, verbose=2)\n    ls_state_val_1_paral.append(state_val_1_paral)\n')


# In[37]:


for n_ep, state_val_1_paral in zip(ls_n_ep, ls_state_val_1_paral):
    plot_arr_bj_state_val(state_val_1_paral, f'(n_ep={n_ep})')


# In[ ]:





# ## Func 2

# In[25]:


get_ipython().run_cell_magic('time', '', 'ls_state_val_2 = []\nfor n_ep in ls_n_ep:\n    state_val_2 = Monte_Carlo_sim_blackjack_2(n_ep=n_ep)\n    ls_state_val_2.append(state_val_2)\n')


# In[38]:


for n_ep, state_val_2 in zip(ls_n_ep, ls_state_val_2):
    plot_arr_bj_state_val(state_val_2, f'(n_ep={n_ep})')


# In[27]:


get_ipython().run_cell_magic('time', '', 'ls_state_val_2_paral = []\nfor n_ep in ls_n_ep:\n    state_val_2_paral = Monte_Carlo_sim_blackjack_2(n_ep=n_ep, n_jobs=N_JOBS, verbose=2)\n    ls_state_val_2_paral.append(state_val_2_paral)\n')


# In[39]:


for n_ep, state_val_2_paral in zip(ls_n_ep, ls_state_val_2_paral):
    plot_arr_bj_state_val(state_val_2_paral, f'(n_ep={n_ep})')


# ## Case 1: dealer_thre=20, player_thre=20

# In[24]:


state_val_c1 = Monte_Carlo_sim_blackjack_2(n_ep=1_000_000, dealer_thre=20, player_thre=20, n_jobs=N_JOBS, verbose=2)


# In[28]:


plot_arr_bj_state_val_3d(state_val_c1[0].T, f'(n_ep=1_000_000, (dealer, player)=(20, 20), no ace)')
plot_arr_bj_state_val_3d(state_val_c1[1].T, f'(n_ep=1_000_000, (dealer, player)=(20, 20), with ace)')


# In[14]:


plot_arr_bj_state_val(state_val_c1, f'(n_ep=1_000_000, dealer_thre=20, player_thre=20)')


# ## Case 2: dealer_thre=17, player_thre=17

# In[29]:


state_val_c2 = Monte_Carlo_sim_blackjack_2(n_ep=1_000_000, dealer_thre=17, player_thre=17, n_jobs=N_JOBS, verbose=2)


# In[30]:


plot_arr_bj_state_val_3d(state_val_c2[0].T, f'(n_ep=1_000_000, (dealer, player)=(17, 17), no ace)')
plot_arr_bj_state_val_3d(state_val_c2[1].T, f'(n_ep=1_000_000, (dealer, player)=(17, 17), with ace)')


# In[16]:


plot_arr_bj_state_val(state_val_c2, f'(n_ep=1_000_000, dealer_thre=17, player_thre=17)')


# ## Case 3: dealer_thre=20, player_thre=17

# In[31]:


state_val_c3 = Monte_Carlo_sim_blackjack_2(n_ep=1_000_000, dealer_thre=20, player_thre=17, n_jobs=N_JOBS, verbose=2)


# In[32]:


plot_arr_bj_state_val_3d(state_val_c3[0].T, f'(n_ep=1_000_000, (dealer, player)=(20, 17), no ace)')
plot_arr_bj_state_val_3d(state_val_c3[1].T, f'(n_ep=1_000_000, (dealer, player)=(20, 17), with ace)')


# In[33]:


plot_arr_bj_state_val(state_val_c3, f'(n_ep=1_000_000, dealer_thre=20, player_thre=17)')


# # Example 5.2: Blackjack 

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




