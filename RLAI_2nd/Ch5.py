#!/usr/bin/env python
# coding: utf-8

# In[1]:


from IPython.display import display, HTML

# Set the notebook width to 80%
display(HTML("<style>.container { width: 80% !important; }</style>"))


# In[2]:


get_ipython().system('jupyter notebook list')


# In[3]:


# Needs to paste `http://localhost:3110`, no ending `/`
port = 2912

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


import logging

LOGGING_FORMAT = "%(asctime)s|(%(pathname)s)[%(lineno)d]: %(message)s"

logging.basicConfig(format=LOGGING_FORMAT)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

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


# # Example 5.1: Blackjack fixed policy
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
    


# In[8]:


N_JOBS = 20
ALPHA = 0.8
CUST_JET = [
    [0.0, f'rgba(0, 0, 131, {ALPHA})'],    # Dark blue, more transparent
    [0.11, f'rgba(0, 60, 170, {ALPHA})'],  # Blue
    [0.22, f'rgba(5, 255, 255, {ALPHA})'], # Cyan
    [0.33, f'rgba(255, 255, 0, {ALPHA})'], # Yellow
    [0.44, f'rgba(250, 0, 0, {ALPHA})'],   # Red
    [0.55, f'rgba(128, 0, 0, {ALPHA})'],   # Dark red
    [1.0, f'rgba(128, 0, 0, {ALPHA})']     # Dark red, same as above to end
]
# COLORSCALE = CUST_JET
COLORSCALE = "Jet"

def organize_state_val(state_val_pair):
    
    def _update_state_val(state, val):
        o_val, o_cnt = state_val[state]
        state_val[state] = (o_val+(val-o_val)/(o_cnt+1), o_cnt+1)
        
    # Initial state values and cnts
    state_val = {}
    for i in range(1, 11):
        for j in range(12, 22):
            state_val[i, j, 1] = (0, 0) # usable ace
            state_val[i, j, 0] = (0, 0) # no usable ace
    
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
        return int(min(card, 10)+10*(card==1))
    
    def _card_val(card):
        return int(min(card, 10))
    
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
            had_ace = d_usable_ace = int(dealer_show==1)
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
        return int(min(card, 10)+10*(card==1))
    
    def _card_val(card):
        return int(min(card, 10))
    
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
            had_ace = d_usable_ace = int(dealer_show==1)
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


def add_axex_gridline(fig, x_vals, y_vals, ls_rc):
    for ri, ci in ls_rc:
        # Manually add gridlines at half-tick positions for x-axis
        for x in x_vals:
            fig.add_shape(
                type="line", x0=x, y0=min(y_vals), x1=x, y1=max(y_vals),
                line=dict(color="Grey", width=1, dash="dot"),
                row=ri, col=ci
            )

        # Manually add gridlines at half-tick positions for y-axis
        for y in y_vals:
            fig.add_shape(
                type="line", x0=min(x_vals), y0=y, x1=max(x_vals), y1=y,
                line=dict(color="Grey", width=1, dash="dot"),
                row=ri, col=ci
            )

def plot_state_policy(state_policy, postfix, colorscale=COLORSCALE):
    plot_val_over_state_space("Policy", state_policy, postfix, colorscale=colorscale)

def plot_val_over_state_space(title, val_over_state_space, postfix, zlim=None, colorscale=COLORSCALE):
    fig = make_subplots(rows=1, cols=2, horizontal_spacing=0.2, subplot_titles=('No Ace', 'With Ace'))
    if zlim is not None:
        zmin, zmax = zlim
    else:
        zmin, zmax = min(np.min(vals) for vals in val_over_state_space), max(np.max(vals) for vals in val_over_state_space)
    fig.add_trace(
        go.Heatmap(
            z=val_over_state_space[0].T,
            colorscale=colorscale,
            x=[i for i in range(1, 11)],
            y=[j for j in range(12, 22)],
            hoverongaps=False,
            zmin=zmin,
            zmax=zmax,
            colorbar=dict(title=title, x=0.42)  # Adjust x to position the color bar between subplots
        ), row=1, col=1
    )
    fig.add_trace(
        go.Heatmap(
            z=val_over_state_space[1].T,
            colorscale=colorscale,
            x=[i for i in range(1, 11)],
            y=[j for j in range(12, 22)],
            hoverongaps=False,
            zmin=zmin,
            zmax=zmax,
            colorbar=dict(title=title, x=1.02)  # Adjust x to position the color bar between subplots
        ), row=1, col=2
    )
    
    xaxis_kwargs = dict(
        title='x: Dealer showing', 
        tickvals=np.arange(1, 11),
        ticktext=[str(i) for i in range(1, 11)],
        showgrid=False
        # showgrid=True, gridwidth=1, gridcolor='grey', layer='above traces'
    )
    yaxis_kwargs = dict(
        title='y: Player sum', 
        tickvals=np.arange(12, 22),
        ticktext=[str(i) for i in range(12, 22)],
        showgrid=False
        # showgrid=True, gridwidth=1, gridcolor='grey', layer='above traces'
    )
    
    fig.update_layout(
        title=f'State {title} Heatmaps {postfix}',
        xaxis=xaxis_kwargs,
        yaxis=yaxis_kwargs,
        xaxis2=xaxis_kwargs,
        yaxis2=yaxis_kwargs,
        autosize=False,
        width=1000,
        height=500,
    )
    
    gl_x_vals = np.arange(0.5, 11, 1)
    gl_y_vals = np.arange(11.5, 22, 1)
    add_axex_gridline(fig, gl_x_vals, gl_y_vals, [(1, 1), (1, 2)])

    # add_axex_gridline(fig, np.arange(1, 11), np.arange(12, 22))

    fig.show()

def plot_arr_bj_state_val(state_val_res, postfix, max_cnt=None, colorscale=COLORSCALE):
    state_val_0, state_val_1, state_val_cnt_0, state_val_cnt_1 = state_val_res
    fig = make_subplots(
        rows=2, cols=2, 
        horizontal_spacing=0.2, vertical_spacing=0.2,
        subplot_titles=('No Ace', 'With Ace', 'Count No Ace', 'Count With Ace'))
    fig.add_trace(
        go.Heatmap(
            z=state_val_0.T,
            colorscale=colorscale,
            x=[i for i in range(1, 11)],
            y=[j for j in range(12, 22)],
            hoverongaps=False,
            zmin=-1,
            zmax=1,
            colorbar=dict(title="Val", x=0.42, y=0.82, len=0.4)  # Adjust x to position the color bar between subplots
        ), row=1, col=1
    )
    fig.add_trace(
        go.Heatmap(
            z=state_val_1.T,
            colorscale=colorscale,
            x=[i for i in range(1, 11)],
            y=[j for j in range(12, 22)],
            hoverongaps=False,
            zmin=-1,
            zmax=1,
            colorbar=dict(title="Val", x=1.02, y=0.82, len=0.4)  # Adjust x to position the color bar between subplots
        ), row=1, col=2
    )
    
    fig.add_trace(
        go.Heatmap(
            z=state_val_cnt_0.T,
            colorscale=colorscale,
            x=[i for i in range(1, 11)],
            y=[j for j in range(12, 22)],
            hoverongaps=False,
            zmin=0,
            zmax=max_cnt,
            colorbar=dict(title="Cnt", x=0.42, y=0.2, len=0.4)  # Adjust x to position the color bar between subplots
        ), row=2, col=1
    )
    fig.add_trace(
        go.Heatmap(
            z=state_val_cnt_1.T,
            colorscale=colorscale,
            x=[i for i in range(1, 11)],
            y=[j for j in range(12, 22)],
            hoverongaps=False,
            zmin=0,
            zmax=max_cnt,
            colorbar=dict(title="Cnt", x=1.02, y=0.2, len=0.4)  # Adjust x to position the color bar between subplots
        ), row=2, col=2
    )
    
    xaxis_kwargs = dict(
        title='x: Dealer showing', 
        tickvals=np.arange(1, 11),
        ticktext=[str(i) for i in range(1, 11)],
        showgrid=False
        # showgrid=True, gridwidth=1, gridcolor='grey', layer='above traces'
    )
    yaxis_kwargs = dict(
        title='y: Player sum', 
        tickvals=np.arange(12, 22),
        ticktext=[str(i) for i in range(12, 22)],
        showgrid=False
        # showgrid=True, gridwidth=1, gridcolor='grey', layer='above traces'
    )

    fig.update_layout(
        title=f'State-Value Heatmaps {postfix}',
        xaxis=xaxis_kwargs,
        yaxis=yaxis_kwargs,
        xaxis2=xaxis_kwargs,
        yaxis2=yaxis_kwargs,
        xaxis3=xaxis_kwargs,
        yaxis3=yaxis_kwargs,
        xaxis4=xaxis_kwargs,
        yaxis4=yaxis_kwargs,
        autosize=False,
        width=1000,
        height=1000,
    )
    
    gl_x_vals = np.arange(0.5, 11, 1)
    gl_y_vals = np.arange(11.5, 22, 1)
    add_axex_gridline(fig, gl_x_vals, gl_y_vals, [(1, 1), (1, 2), (2, 1), (2, 2)])

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
            xaxis_title='x: Dealer Showing',
            yaxis_title='y: Player Sum',
            zaxis=dict(range=(-1, 1), title='z: State Value')
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


# In[9]:


ls_n_ep = [10_000, 500_000, 1_000_000]


# ## Func 1

# In[10]:


get_ipython().run_cell_magic('time', '', 'ls_state_val_1 = []\nfor n_ep in ls_n_ep:\n    state_val_1 = Monte_Carlo_sim_blackjack_1(n_ep=n_ep)\n    ls_state_val_1.append(state_val_1)\n')


# In[11]:


for n_ep, state_val_1 in zip(ls_n_ep, ls_state_val_1):
    plot_arr_bj_state_val(state_val_1, f'(n_ep={n_ep})')


# In[12]:


get_ipython().run_cell_magic('time', '', 'ls_state_val_1_paral = []\nfor n_ep in ls_n_ep:\n    state_val_1_paral = Monte_Carlo_sim_blackjack_1(n_ep=n_ep, n_jobs=N_JOBS, verbose=2)\n    ls_state_val_1_paral.append(state_val_1_paral)\n')


# In[13]:


for n_ep, state_val_1_paral in zip(ls_n_ep, ls_state_val_1_paral):
    plot_arr_bj_state_val(state_val_1_paral, f'(n_ep={n_ep})')


# In[ ]:





# ## Func 2

# In[14]:


get_ipython().run_cell_magic('time', '', 'ls_state_val_2 = []\nfor n_ep in ls_n_ep:\n    state_val_2 = Monte_Carlo_sim_blackjack_2(n_ep=n_ep)\n    ls_state_val_2.append(state_val_2)\n')


# In[15]:


for n_ep, state_val_2 in zip(ls_n_ep, ls_state_val_2):
    plot_arr_bj_state_val(state_val_2, f'(n_ep={n_ep})')


# In[16]:


get_ipython().run_cell_magic('time', '', 'ls_state_val_2_paral = []\nfor n_ep in ls_n_ep:\n    state_val_2_paral = Monte_Carlo_sim_blackjack_2(n_ep=n_ep, n_jobs=N_JOBS, verbose=2)\n    ls_state_val_2_paral.append(state_val_2_paral)\n')


# In[17]:


for n_ep, state_val_2_paral in zip(ls_n_ep, ls_state_val_2_paral):
    plot_arr_bj_state_val(state_val_2_paral, f'(n_ep={n_ep})')


# ## Case 0: dealer_thre=17, player_thre=20

# In[18]:


get_ipython().run_cell_magic('time', '', 'state_val_c0 = Monte_Carlo_sim_blackjack_2(n_ep=1_000_000, dealer_thre=17, player_thre=20, n_jobs=N_JOBS, verbose=2)\n')


# In[19]:


plot_arr_bj_state_val_3d(state_val_c0[0].T, f'(n_ep=1_000_000), (dealer, player)=(20, 20), no ace)')
plot_arr_bj_state_val_3d(state_val_c0[1].T, f'(n_ep=1_000_000), (dealer, player)=(20, 20), with ace)')


# In[20]:


plot_arr_bj_state_val(state_val_c0, f'(n_ep=1_000_000), (dealer, player)=(17, 20)')


# ## Case 1: dealer_thre=20, player_thre=20

# In[21]:


get_ipython().run_cell_magic('time', '', 'state_val_c1 = Monte_Carlo_sim_blackjack_2(n_ep=1_000_000, dealer_thre=20, player_thre=20, n_jobs=N_JOBS, verbose=2)\n')


# In[22]:


plot_arr_bj_state_val_3d(state_val_c1[0].T, f'(n_ep=1_000_000, (dealer, player)=(20, 20), no ace)')
plot_arr_bj_state_val_3d(state_val_c1[1].T, f'(n_ep=1_000_000, (dealer, player)=(20, 20), with ace)')


# In[23]:


plot_arr_bj_state_val(state_val_c1, f'(n_ep=1_000_000, dealer_thre=20, player_thre=20)')


# ## Case 2: dealer_thre=17, player_thre=17

# In[24]:


get_ipython().run_cell_magic('time', '', 'state_val_c2 = Monte_Carlo_sim_blackjack_2(n_ep=1_000_000, dealer_thre=17, player_thre=17, n_jobs=N_JOBS, verbose=2)\n')


# In[25]:


plot_arr_bj_state_val_3d(state_val_c2[0].T, f'(n_ep=1_000_000, (dealer, player)=(17, 17), no ace)')
plot_arr_bj_state_val_3d(state_val_c2[1].T, f'(n_ep=1_000_000, (dealer, player)=(17, 17), with ace)')


# In[26]:


plot_arr_bj_state_val(state_val_c2, f'(n_ep=1_000_000, dealer_thre=17, player_thre=17)')


# ## Case 3: dealer_thre=20, player_thre=17

# In[27]:


get_ipython().run_cell_magic('time', '', 'state_val_c3 = Monte_Carlo_sim_blackjack_2(n_ep=1_000_000, dealer_thre=20, player_thre=17, n_jobs=N_JOBS, verbose=2)\n')


# In[28]:


plot_arr_bj_state_val_3d(state_val_c3[0].T, f'(n_ep=1_000_000, (dealer, player)=(20, 17), no ace)')
plot_arr_bj_state_val_3d(state_val_c3[1].T, f'(n_ep=1_000_000, (dealer, player)=(20, 17), with ace)')


# In[29]:


plot_arr_bj_state_val(state_val_c3, f'(n_ep=1_000_000, dealer_thre=20, player_thre=17)')


# # Example 5.2: Blackjack optimal policy
# - Using Monte Carlo ES (Exploring Starts)
# - The optimal policy is a threshold policy. Meaning, if a player decides to stick at certain state, it will not hit when reaching state higher than the threshold. This is can be proved by the following.

# In[10]:


FLOAT_TOL = 1e-8
REL_ABS_DIFF_RD_HL_MUL = 5
EPS_PROB = 0.05

def Monte_Carlo_ES_blackjack_1(
    n_ep, n_paral, dealer_thre=17, ini_player_thre=20, dealer_policy="fixed", sampling_schema="uniform",
    pol_impr=True, tol=1e-6, show_rdis=None, n_jobs=1, verbose=0
):
    assert dealer_thre<=21
    assert ini_player_thre<=21
    assert n_ep % n_paral == 0
    assert n_paral >= n_jobs
    
    n_suit = 13
    ndl, npl = 10, 10
    d_s0, p_s0 = 1, 12
    state_act_val_sum = np.zeros((ndl, npl, 2, 2))
    state_act_val_cnt = np.ones((ndl, npl, 2, 2)) # So cnt is never 0 and can always be at the denominator
    state_act_val = np.zeros((ndl, npl, 2, 2))
    state_policy = np.zeros((ndl, npl, 2), dtype=int) # be careful about dtype here, if we need to use it as index, it needs to be int
    for i in range(ndl): # Dealer showing
        for j in range(npl): # Player sum
            for k in range(0, 2): # Usable ace
                # State
                if j+p_s0<ini_player_thre:
                    # Initial policy
                    # Whenever the player sum is smaller than ini_player_thre, hit 
                    state_policy[i, j, k] = 1
                    
    n_rd = n_ep // n_paral
    
    # def _get_state_act_val(i, j, k, a):
    #     state_act_val = (
    #         state_act_val_sum[i,j,k,a]/state_act_val_cnt[i,j,k,a] 
    #         if state_act_val_cnt[i,j,k,a]>0
    #         else 1 # Can actually choose any value. Choose 1 to promote exploring zero experience state-act
    #     )
    #     return state_act_val
    
    log_step = n_rd//20
    rel_abs_diff_rd_hl = REL_ABS_DIFF_RD_HL_MUL*state_act_val.size/n_paral
    lam = np.exp(-np.log(2)/rel_abs_diff_rd_hl)
    # norm_state_act_val_abs_diff = np.ones((ndl, npl, 2, 2))/state_act_val.size
    norm_state_act_val_act_similar = np.ones((ndl, npl, 2))/state_policy.size
    
    def _ini_card_val(card):
        return int(min(card, 10)+10*(card==1))
    
    def _card_val(card):
        return int(min(card, 10))
    
    def _organize_state_val(state_act_val, state_act_val_cnt, state_policy):
        state_val = np.max(state_act_val, axis=-1)
        state_val_cnt = np.sum(state_act_val_cnt, axis=-1)
        state_val_0 = state_val[:, :, 0]
        state_val_1 = state_val[:, :, 1]
        state_val_cnt_0 = state_val_cnt[:, :, 0]
        state_val_cnt_1 = state_val_cnt[:, :, 1]
        state_policy_0 = state_policy[:, :, 0]
        state_policy_1 = state_policy[:, :, 1]
        state_act_val_00 = state_act_val[:, :, 0, 0]
        state_act_val_01 = state_act_val[:, :, 0, 1]
        state_act_val_10 = state_act_val[:, :, 1, 0]
        state_act_val_11 = state_act_val[:, :, 1, 1]
        state_act_val_cnt_00 = state_act_val_cnt[:, :, 0, 0]
        state_act_val_cnt_01 = state_act_val_cnt[:, :, 0, 1]
        state_act_val_cnt_10 = state_act_val_cnt[:, :, 1, 0]
        state_act_val_cnt_11 = state_act_val_cnt[:, :, 1, 1]
        
        return (
            state_val_0, state_val_1, state_val_cnt_0, state_val_cnt_1, state_policy_0, state_policy_1,
            (state_act_val_00, state_act_val_10, state_act_val_cnt_00, state_act_val_cnt_10),
            (state_act_val_01, state_act_val_11, state_act_val_cnt_01, state_act_val_cnt_11)
        )
    
    def _dealer_turn_fixed(dealer_show, final_player_sum):
        assert final_player_sum<=21
        dealer_sum = _ini_card_val(dealer_show)
        had_ace = d_usable_ace = int(dealer_show == 1)
        while 1:
            new_card = np.random.randint(1, 1+n_suit)
            if new_card == 1:
                if had_ace: # if already had an ace or previously had an ace, but not usable now
                    dealer_sum += 1
                else:
                    had_ace = d_usable_ace = 1
                    dealer_sum += _ini_card_val(new_card)
            else:
                dealer_sum += _card_val(new_card)
            if dealer_sum > 21:
                if d_usable_ace:
                    d_usable_ace = 0
                    dealer_sum -= 10
                else: # dealer busted
                    reward = 1 # player reward
                    break
            elif dealer_sum>=dealer_thre:
                reward = np.sign(final_player_sum-dealer_sum)
                break
        return reward
    
    def _dealer_turn_smart(dealer_show, final_player_sum):
        assert final_player_sum<=21
        dealer_sum = _ini_card_val(dealer_show)
        had_ace = d_usable_ace = int(dealer_show == 1)
        while 1:
            new_card = np.random.randint(1, 1+n_suit)
            if new_card == 1:
                if had_ace: # if already had an ace or previously had an ace, but not usable now
                    dealer_sum += 1
                else:
                    had_ace = d_usable_ace = 1
                    dealer_sum += _ini_card_val(new_card)
            else:
                dealer_sum += _card_val(new_card)
            if dealer_sum > 21:
                if d_usable_ace:
                    d_usable_ace = 0
                    dealer_sum -= 10
                else: # dealer busted
                    reward = 1 # player reward
                    break
            elif dealer_sum > final_player_sum or dealer_sum==21 or dealer_sum>=dealer_thre:
                reward = np.sign(final_player_sum-dealer_sum)
                break
        return reward
    
    def _one_episode():
        ls_state_act = []
        # initial state-act
        if sampling_schema == "uniform":
            cur_state_act = (
                np.random.randint(ndl),
                np.random.randint(npl),
                np.random.randint(2),
                np.random.randint(2), # Also need to randomly choose action, b/c we need some experience to evaluate a state-act pair.
            )
        else: # sampling state-act based on the similarity of action values at each state space
            flat_idx = np.random.choice(state_policy.size, p=norm_state_act_val_act_similar.reshape(-1)*(1-EPS_PROB)+EPS_PROB/state_policy.size)
            rnd_i = flat_idx // state_policy[0].size
            flat_idx -= rnd_i*state_policy[0].size
            rnd_j = flat_idx // state_policy[0, 0].size
            flat_idx -= rnd_j*state_policy[0, 0].size
            rnd_k = flat_idx
            cur_state_act = (rnd_i, rnd_j, rnd_k, np.random.randint(2))
        ls_state_act.append(cur_state_act)
        dealer_show = cur_state_act[0] + d_s0 # Convert from dealer state idx to actual dealer showing card
        
        busted = False
        while cur_state_act[-1]: # As long as current act is to hit
            new_card = np.random.randint(1, 1+n_suit)
            player_sum = cur_state_act[1] + p_s0 + _card_val(new_card) # Convert from player state idx to actual player sum
            if player_sum>21: 
                if cur_state_act[2]:
                    player_sum -= 10
                    cur_state_act = [cur_state_act[0], player_sum-p_s0, 0] # Convert from actual player sum to player state idx
                else: # player busted
                    busted = True
                    break
            else:
                cur_state_act = [cur_state_act[0], player_sum-p_s0, cur_state_act[2]]
            policy_act = int(state_policy[tuple(cur_state_act)]) # Follow the current policy
            cur_state_act.append(policy_act)
            ls_state_act.append(tuple(cur_state_act))
        
        if busted:
            reward = -1
        else:
            final_player_sum = cur_state_act[1] + p_s0
            if dealer_policy == "fixed":
                reward = _dealer_turn_fixed(dealer_show, final_player_sum)
            else:
                reward = _dealer_turn_smart(dealer_show, final_player_sum)
        
        state_act_val_sum = np.zeros((ndl, npl, 2, 2))
        state_act_val_cnt = np.zeros((ndl, npl, 2, 2))
        
        for state_act in ls_state_act:
            assert isinstance(state_act, tuple) and len(state_act)==4
            state_act_val_sum[state_act] += reward
            state_act_val_cnt[state_act] += 1
        
        return state_act_val_sum, state_act_val_cnt
    
    for rdi in range(n_rd):
        if rdi%log_step==0:
            logger.info("="*20+f"round: {rdi}")
            
        state_act_val_prev = state_act_val.copy()
        state_act_val = state_act_val_sum/state_act_val_cnt
        state_act_val_abs_diff = np.abs(state_act_val-state_act_val_prev) 
        abs_state_act_val = np.abs(state_act_val)           
        max_abs_diff, max_abs_val = np.max(state_act_val_abs_diff), np.max(abs_state_act_val)
        if rdi>0:
            if max_abs_diff<tol*max_abs_val:
                logger.info(f"Converged at round with relative tol {tol:.1e} : {rdi}")
                break
            if pol_impr:
                state_policy = np.argmax(state_act_val, axis=-1)
            # Show the normalized relative absolute difference to see how state-val changes across state space
            # n_norm_state_act_val_abs_diff = state_act_val_abs_diff/np.maximum(abs_state_act_val, (abs_state_act_val<FLOAT_TOL).astype(int))
            # n_norm_state_act_val_abs_diff = n_norm_state_act_val_abs_diff/np.sum(n_norm_state_act_val_abs_diff)
            # norm_state_act_val_abs_diff += (1-lam)*(n_norm_state_act_val_abs_diff-norm_state_act_val_abs_diff)
            # Show the normalized relative similarity of action values for ech state.
            abs_state_act_val_sum = np.sum(abs_state_act_val, axis=-1)
            n_norm_state_act_val_act_similar = (
                2/(1+(
                    np.abs(np.diff(state_act_val, axis=-1).squeeze(axis=-1))/
                    np.maximum(abs_state_act_val_sum, (abs_state_act_val_sum<FLOAT_TOL).astype(int))
                ))-1
            )
            n_norm_state_act_val_act_similar /= np.sum(n_norm_state_act_val_act_similar)
            norm_state_act_val_act_similar += (1-lam)*(n_norm_state_act_val_act_similar-norm_state_act_val_act_similar) 
            norm_state_act_val_act_similar /= np.sum(norm_state_act_val_act_similar)
            
        if show_rdis is not None and rdi in show_rdis:
            logger.info("-"*20+f"round: {rdi}")
            tmp_res = _organize_state_val(state_act_val, state_act_val_cnt, state_policy)
            plot_state_policy([tmp_res[4], tmp_res[5]], f'{n_ep}, ({dealer_thre}, {ini_player_thre}), rdi={rdi}, {max_abs_diff:.2e}, {max_abs_diff/max_abs_val:.2e}')
            if rdi>0:
                # plot_val_over_state_space(
                #     "Rel-Abs-Diff", 
                #     [norm_state_act_val_abs_diff[:, :, 0, 0], norm_state_act_val_abs_diff[:, :, 1, 0]],
                #     f'{n_ep}, ({dealer_thre}, {ini_player_thre}), rdi={rdi}, stick',
                #     zlim=(0, 1)
                # )
                # plot_val_over_state_space(
                #     "Rel-Abs-Diff", 
                #     [norm_state_act_val_abs_diff[:, :, 0, 1], norm_state_act_val_abs_diff[:, :, 1, 1]],
                #     f'{n_ep}, ({dealer_thre}, {ini_player_thre}), rdi={rdi}, hit',
                #     zlim=(0, 1)
                # )
                plot_val_over_state_space(
                    "Rel-Act-Simil", 
                    [n_norm_state_act_val_act_similar[:, :, 0], n_norm_state_act_val_act_similar[:, :, 1]],
                    f'{n_ep}, ({dealer_thre}, {ini_player_thre}), rdi={rdi}, stick v.s. hit',
                    # zlim=(0, 1)
                )
            plot_arr_bj_state_val(
                tmp_res[:4], 
                f'{n_ep}, ({dealer_thre}, {ini_player_thre})', 
                max_cnt=int((n_ep//state_policy.size)*(1.5 if sampling_schema=="uniform" else 10))
            )
            
        with parallel_backend('loky', n_jobs=n_jobs):
            res = Parallel(verbose=verbose, pre_dispatch="1.5*n_jobs")(
                delayed(_one_episode)() for _ in range(n_paral)
            )
        ls_new_val_sum, ls_new_val_cnt = zip(*res)
        new_val_sum = sum(ls_new_val_sum)
        new_val_cnt = sum(ls_new_val_cnt)
        state_act_val_sum += new_val_sum
        state_act_val_cnt += new_val_cnt
            
    state_act_val = state_act_val_sum/state_act_val_cnt
    
    return _organize_state_val(state_act_val, state_act_val_cnt, state_policy)


# In[11]:


show_rdis = [0, 1000, 5000, 10_000, 20_000, 30_000, 40_000, 50_000, 60_000, 70_000, 80_000, 90_000, 100_000, 500_000, 1_000_000]


# ## Dealer policy: "fixed"; Sampling: "uniform" 

# In[32]:


get_ipython().run_cell_magic('time', '', '# 46 mins\nn_ep=4_000_000\nstate_act_res = Monte_Carlo_ES_blackjack_1(\n    n_ep=n_ep, n_paral=50, dealer_thre=17, ini_player_thre=20, \n    pol_impr=True, show_rdis=show_rdis, n_jobs=5, verbose=0\n)\n')


# In[33]:


plot_arr_bj_state_val_3d(state_act_res[0].T, f'(n_ep={n_ep}, (dealer, player_ini)=(17, 20), no ace)')
plot_arr_bj_state_val_3d(state_act_res[1].T, f'(n_ep={n_ep}, (dealer, player_ini)=(17, 20), with ace)')


# In[34]:


plot_state_policy([state_act_res[4], state_act_res[5]], f'(n_ep={n_ep}, (dealer, player_ini)=(17, 20))')


# In[35]:


plot_arr_bj_state_val(state_act_res[6], f'(n_ep={n_ep}, (dealer, player_ini)=(17, 20)), stick')


# In[36]:


plot_arr_bj_state_val(state_act_res[7], f'(n_ep={n_ep}, (dealer, player_ini)=(17, 20)), hit')


# ## Dealer policy: "fixed"; Sampling: "similar" 

# In[12]:


get_ipython().run_cell_magic('time', '', '# 46 mins\nn_ep=4_000_000\nstate_act_res = Monte_Carlo_ES_blackjack_1(\n    n_ep=n_ep, n_paral=50, dealer_thre=17, ini_player_thre=20, sampling_schema="similar",\n    pol_impr=True, show_rdis=show_rdis, n_jobs=5, verbose=0\n)\n')


# In[13]:


plot_arr_bj_state_val_3d(state_act_res[0].T, f'(n_ep={n_ep}, (dealer, player_ini)=(17, 20), no ace, similar_spl)')
plot_arr_bj_state_val_3d(state_act_res[1].T, f'(n_ep={n_ep}, (dealer, player_ini)=(17, 20), with ace, similar_spl)')


# In[14]:


plot_state_policy([state_act_res[4], state_act_res[5]], f'(n_ep={n_ep}, (dealer, player_ini)=(17, 20), similar_spl)')


# In[15]:


plot_arr_bj_state_val(state_act_res[6], f'(n_ep={n_ep}, (dealer, player_ini)=(17, 20)), stick, similar_spl')


# In[16]:


plot_arr_bj_state_val(state_act_res[7], f'(n_ep={n_ep}, (dealer, player_ini)=(17, 20)), hit, similar_spl')


# ## Dealer policy: "smart"; Sampling: "uniform"

# In[37]:


get_ipython().run_cell_magic('time', '', "# 49 mins\nn_ep=4_000_000\nstate_act_res = Monte_Carlo_ES_blackjack_1(\n    n_ep=n_ep, n_paral=50, dealer_thre=17, ini_player_thre=20, dealer_policy='smart',\n    pol_impr=True, show_rdis=show_rdis, n_jobs=5, verbose=0\n)\n")


# In[38]:


plot_arr_bj_state_val_3d(state_act_res[0].T, f'(n_ep={n_ep}, (dealer, player_ini)=(17, 20), no ace)')
plot_arr_bj_state_val_3d(state_act_res[1].T, f'(n_ep={n_ep}, (dealer, player_ini)=(17, 20), with ace)')


# In[39]:


plot_state_policy([state_act_res[4], state_act_res[5]], f'(n_ep={n_ep}, (dealer, player_ini)=(17, 20))')


# In[40]:


plot_arr_bj_state_val(state_act_res[6], f'(n_ep={n_ep}, (dealer, player_ini)=(17, 20)), stick')


# In[41]:


plot_arr_bj_state_val(state_act_res[7], f'(n_ep={n_ep}, (dealer, player_ini)=(17, 20)), hit')


# ## Dealer policy: "smart"; Sampling: "similar"

# In[17]:


get_ipython().run_cell_magic('time', '', '# 46 mins\nn_ep=4_000_000\nstate_act_res = Monte_Carlo_ES_blackjack_1(\n    n_ep=n_ep, n_paral=50, dealer_thre=17, ini_player_thre=20, dealer_policy=\'smart\', sampling_schema="similar",\n    pol_impr=True, show_rdis=show_rdis, n_jobs=5, verbose=0\n)\n')


# In[ ]:


plot_arr_bj_state_val_3d(state_act_res[0].T, f'(n_ep={n_ep}, (dealer, player_ini)=(17, 20), no ace, similar_spl)')
plot_arr_bj_state_val_3d(state_act_res[1].T, f'(n_ep={n_ep}, (dealer, player_ini)=(17, 20), with ace, similar_spl)')


# In[ ]:


plot_state_policy([state_act_res[4], state_act_res[5]], f'(n_ep={n_ep}, (dealer, player_ini)=(17, 20), similar_spl)')


# In[ ]:


plot_arr_bj_state_val(state_act_res[6], f'(n_ep={n_ep}, (dealer, player_ini)=(17, 20)), stick, similar_spl')


# In[ ]:


plot_arr_bj_state_val(state_act_res[7], f'(n_ep={n_ep}, (dealer, player_ini)=(17, 20)), hit, similar_spl')


# In[ ]:





# # End

# In[ ]:




