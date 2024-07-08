#!/usr/bin/env python
# coding: utf-8

# <h1>Table of Contents<span class="tocSkip"></span></h1>
# <div class="toc"><ul class="toc-item"><li><span><a href="#Import-dep" data-toc-modified-id="Import-dep-1"><span class="toc-item-num">1&nbsp;&nbsp;</span>Import dep</a></span></li><li><span><a href="#10-arms-bandit" data-toc-modified-id="10-arms-bandit-2"><span class="toc-item-num">2&nbsp;&nbsp;</span>10 arms bandit</a></span><ul class="toc-item"><li><span><a href="#New-bandit" data-toc-modified-id="New-bandit-2.1"><span class="toc-item-num">2.1&nbsp;&nbsp;</span>New bandit</a></span></li><li><span><a href="#Same-bandit" data-toc-modified-id="Same-bandit-2.2"><span class="toc-item-num">2.2&nbsp;&nbsp;</span>Same bandit</a></span></li></ul></li><li><span><a href="#The-mean-value-of-the-maximum-of-k-normal-RV-with-mean-0-and-std-1" data-toc-modified-id="The-mean-value-of-the-maximum-of-k-normal-RV-with-mean-0-and-std-1-3"><span class="toc-item-num">3&nbsp;&nbsp;</span>The mean value of the maximum of k normal RV with mean 0 and std 1</a></span></li><li><span><a href="#Ex-2.3" data-toc-modified-id="Ex-2.3-4"><span class="toc-item-num">4&nbsp;&nbsp;</span>Ex 2.3</a></span></li><li><span><a href="#Initial-value-effect" data-toc-modified-id="Initial-value-effect-5"><span class="toc-item-num">5&nbsp;&nbsp;</span>Initial value effect</a></span></li><li><span><a href="#Ex-2.5" data-toc-modified-id="Ex-2.5-6"><span class="toc-item-num">6&nbsp;&nbsp;</span>Ex 2.5</a></span><ul class="toc-item"><li><span><a href="#Use-sample-avg-as-the-valuation-scheme" data-toc-modified-id="Use-sample-avg-as-the-valuation-scheme-6.1"><span class="toc-item-num">6.1&nbsp;&nbsp;</span>Use sample avg as the valuation scheme</a></span></li><li><span><a href="#Use-the-constant-step-size-as-the-valuation-scheme" data-toc-modified-id="Use-the-constant-step-size-as-the-valuation-scheme-6.2"><span class="toc-item-num">6.2&nbsp;&nbsp;</span>Use the constant step size as the valuation scheme</a></span></li></ul></li><li><span><a href="#Ex-2.7" data-toc-modified-id="Ex-2.7-7"><span class="toc-item-num">7&nbsp;&nbsp;</span>Ex 2.7</a></span></li><li><span><a href="#Ex-2.8" data-toc-modified-id="Ex-2.8-8"><span class="toc-item-num">8&nbsp;&nbsp;</span>Ex 2.8</a></span></li><li><span><a href="#Gradient-Bandit" data-toc-modified-id="Gradient-Bandit-9"><span class="toc-item-num">9&nbsp;&nbsp;</span>Gradient Bandit</a></span></li><li><span><a href="#Ex-2.11" data-toc-modified-id="Ex-2.11-10"><span class="toc-item-num">10&nbsp;&nbsp;</span>Ex 2.11</a></span></li><li><span><a href="#End" data-toc-modified-id="End-11"><span class="toc-item-num">11&nbsp;&nbsp;</span>End</a></span></li></ul></div>

# In[1]:


from IPython.display import display, HTML

# Set the notebook width to 80%
display(HTML("<style>.container { width: 80% !important; }</style>"))


# In[2]:


get_ipython().system('hostname')


# In[ ]:


# Needs to paste `http://localhost:3110`, no ending `/`
port = 3760

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

# In[6]:


import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[7]:


import plotly
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import plotly.io as pio

pio.renderers.default = "notebook"


# In[8]:


COLOR_LIST = plotly.colors.DEFAULT_PLOTLY_COLORS
len(COLOR_LIST)


# In[9]:


print(plotly.__version__, plotly.__path__)


# In[10]:


import notebook
import ipywidgets
print(notebook.__version__, notebook.__path__)
print(ipywidgets.__version__, ipywidgets.__path__)


# In[11]:


from joblib import Parallel, delayed, parallel_backend
from itertools import product
import joblib
print(joblib.__version__, joblib.__path__)


# In[12]:


N_JOBS = 10

def plot_perf_metrics(ls_rewards, ls_opt_act_flgs, ls_param, param_prefix='epsilon', plt_lib='mlp', fig=None, axes=None):
    if isinstance(param_prefix, str):
        param_prefix = [param_prefix for _ in range(len(ls_param))]
    if plt_lib == 'mlp':
        if fig is None or axes is None:
            fig, axes = plt.subplots(2, 1, figsize=(10, 10))
        
        for i, eps in enumerate(ls_param):
            axes[0].plot(ls_rewards[i], label='{} = {}'.format(param_prefix[i], eps))
            axes[1].plot(ls_opt_act_flgs[i], label='{} = {}'.format(param_prefix[i], eps))
        
        if fig is None:
            axes[0].set_title('Average Reward')
            axes[0].set_xlabel('Steps')
            axes[0].set_ylabel('Average Reward')
            axes[0].legend()
            
            axes[1].set_title('% Optimal Action')
            axes[1].set_xlabel('Steps')
            axes[1].set_ylabel('% Optimal Action')
        axes[1].legend()
        
        plt.show()
    else:
        # Create subplots with 2 rows and 1 column
        if fig is None:
            fig = make_subplots(rows=2, cols=1)

        for i, eps in enumerate(ls_param):
            
            # Add traces to the first subplot for rewards
            fig.add_trace(go.Scatter(x=np.arange(ls_rewards[i].shape[0]),
                                    y=ls_rewards[i],
                                    mode='lines',
                                    name='{} = {}'.format(param_prefix[i], eps),
                                    line=dict(color=COLOR_LIST[i])), row=1, col=1)

            # Add traces to the second subplot for optimal action rate
            fig.add_trace(go.Scatter(x=np.arange(ls_opt_act_flgs[i].shape[0]),
                                    y=ls_opt_act_flgs[i],
                                    mode='lines',
                                    # name='param = {}'.format(eps),
                                    line=dict(color=COLOR_LIST[i]),
                                    showlegend=False),
                                    row=2, col=1)
        
        if fig is None:
            # Add axis titles
            fig.update_xaxes(title_text='Steps', row=1, col=1)
            fig.update_yaxes(title_text='Average Reward', row=1, col=1)
            fig.update_xaxes(title_text='Steps', row=2, col=1)
            fig.update_yaxes(title_text='% Optimal Action', row=2, col=1)
    
            # Update layout
            fig.update_layout(height=600, width=800, title_text='Subplots', hovermode='x')

        # Show the figure
        fig.show()
        


# # 10 arms bandit

# In[13]:


def k_bandit_sim_eps(k, n_steps, n_exps, epsilon, new_bandit=True, init_q_star=0, n_jobs=N_JOBS):
    q_star, opt_a = None, None
    if not new_bandit:
        # Initialize the q_star values
        q_star = np.random.normal(0, 1, k)
        opt_a = np.argmax(q_star)
        
    def _one_sim(i, q_star, opt_a):
        if new_bandit:
            # Initialize the q_star values
            q_star = np.random.normal(0, 1, k)
            opt_a = np.argmax(q_star)

        # Initialize the q values
        q = np.ones(k)*init_q_star

        # Initialize the number of times each action was taken
        n = np.zeros(k)
        
        rewards = np.zeros(n_steps)
        opt_act_flgs = np.zeros(n_steps)

        for j in range(n_steps):
            # Choose an action
            if np.random.rand() < epsilon:
                a = np.random.randint(k)
            else:
                a = np.argmax(q)

            # Get the reward
            reward = np.random.normal(q_star[a], 1)

            # Update the q values
            n[a] += 1
            q[a] += (reward - q[a]) / n[a]

            # Store the reward
            rewards[j] = reward
            opt_act_flgs[j] = int(a == opt_a)
        
        return rewards, opt_act_flgs
    
    with parallel_backend('loky', n_jobs=n_jobs):
        results = Parallel()(delayed(_one_sim)(i, q_star, opt_a) for i in range(n_exps))
    
    rewards, opt_act_flgs = zip(*results)
    rewards = np.mean(np.array(rewards), axis=0)
    opt_act_flgs = np.mean(np.array(opt_act_flgs), axis=0)
            
    return rewards, opt_act_flgs



# ## New bandit
# - The plot doesn't depend on the randomization of initial q_star too much.

# In[14]:


get_ipython().run_cell_magic('time', '', 'k_arms, n_steps, n_exps = 10, 1000, 2000\nls_epsilon = [0, 0.01, 0.1, 0.3]\nls_rewards = []\nls_opt_act_flgs = []\nfor eps in ls_epsilon:\n    rewards, opt_act_flgs = k_bandit_sim_eps(k_arms, n_steps, n_exps, eps)\n    ls_rewards.append(rewards)\n    ls_opt_act_flgs.append(opt_act_flgs)\n')


# In[15]:


plot_perf_metrics(ls_rewards, ls_opt_act_flgs, ls_epsilon, plt_lib='plotly')


# ## Same bandit
# - The plot results will heavily depend on the inital randomization of the q_star.

# In[16]:


k_arms, n_steps, n_exps = 10, 1000, 2000
ls_epsilon = [0, 0.01, 0.1, 0.3]
ls_rewards = []
ls_opt_act_flgs = []
for eps in ls_epsilon:
    rewards, opt_act_flgs = k_bandit_sim_eps(k_arms, n_steps, n_exps, eps, new_bandit=False)
    ls_rewards.append(rewards)
    ls_opt_act_flgs.append(opt_act_flgs)


# In[17]:


plot_perf_metrics(ls_rewards, ls_opt_act_flgs, ls_epsilon, plt_lib='plotly')


# # The mean value of the maximum of k normal RV with mean 0 and std 1
# - This is the best possible value of long-run k-arms bandit problem.
# - In the bandit, when the maximum arm samples a maximum value at the first time, it won't guarantee that it will eventually choose the maximum arm, b/c later sampling can result it in a lower value than others.

# In[18]:


max_k_rvs = np.zeros(n_exps)
max_q_star_first_realization_max_flgs = np.zeros(n_exps)
for i in range(n_exps):
    q_stars = np.random.normal(0, 1, k_arms)
    max_k_rvs[i] = max(q_stars)
    q_vals = np.array([np.random.normal(q_stars[j], 1) for j in range(k_arms)])
    max_q_star_first_realization_max_flgs[i] = int(np.argmax(q_vals) == np.argmax(q_stars))
print(np.mean(max_k_rvs), np.mean(max_q_star_first_realization_max_flgs))


# # Ex 2.3
# - For $\epsilon=0.1$, the reward should be 

# In[19]:


np.round(np.mean(max_k_rvs)*0.9, 2), np.round(np.mean(max_k_rvs)*0.99, 2)


# # Initial value effect
# - Large(optimistic) initial value encourage exploration.

# In[20]:


k_arms, n_steps, n_exps = 10, 1000, 2000
ls_epsilon = [0, 0.01, 0.1, 0.3]
ls_rewards = []
ls_opt_act_flgs = []
for eps in ls_epsilon:
    rewards, opt_act_flgs = k_bandit_sim_eps(k_arms, n_steps, n_exps, eps, new_bandit=True, init_q_star=5)
    ls_rewards.append(rewards)
    ls_opt_act_flgs.append(opt_act_flgs)


# In[21]:


plot_perf_metrics(ls_rewards, ls_opt_act_flgs, ls_epsilon, plt_lib='plotly')


# # Ex 2.5
# - The constant-step-size scheme gives a better % in getting the optimal action.

# In[22]:


def k_bandit_sim_eps_nonstationary(k, n_steps, n_exps, epsilon, init_q_star=0, rw_sig=0.1, val_scheme='sample_avg', alpha=0.1):
    def _one_sim(i):
        # Initialize the q_star values
        q_star = np.random.normal(0, 1, k)
        opt_a = np.argmax(q_star)

        # Initialize the q values
        q = np.ones(k)*init_q_star

        # Initialize the number of times each action was taken
        n = np.zeros(k)

        rewards = np.zeros(n_steps)
        opt_act_flgs = np.zeros(n_steps)

        for j in range(n_steps):
            if rw_sig > 0:
                q_star += np.random.normal(0, rw_sig, k)
                opt_a = np.argmax(q_star)
            
            # Choose an action
            if np.random.rand() < epsilon:
                a = np.random.randint(k)
            else:
                a = np.argmax(q)
                
            # Get the reward
            reward = np.random.normal(q_star[a], 1)
            
            # Update the q values
            if val_scheme == 'sample_avg':
                n[a] += 1
                q[a] += (reward - q[a]) / n[a]
            else: # constant step-size
                q[a] += alpha*(reward - q[a])
                
            # Store the reward
            rewards[j] = reward
            opt_act_flgs[j] = int(a == opt_a)
            
        return rewards, opt_act_flgs
    
    with parallel_backend('loky', n_jobs=N_JOBS):
        results = Parallel()(delayed(_one_sim)(i) for i in range(n_exps))
        
    rewards, opt_act_flgs = zip(*results)
    rewards = np.mean(np.array(rewards), axis=0)
    opt_act_flgs = np.mean(np.array(opt_act_flgs), axis=0)
            
    return rewards, opt_act_flgs


# ## Use sample avg as the valuation scheme

# In[23]:


k_arms, n_steps, n_exps = 10, 10000, 2000
ls_epsilon = [0, 0.01, 0.1, 0.3]
ls_rewards = []
ls_opt_act_flgs = []
for eps in ls_epsilon:
    rewards, opt_act_flgs = k_bandit_sim_eps_nonstationary(k_arms, n_steps, n_exps, eps)
    ls_rewards.append(rewards)
    ls_opt_act_flgs.append(opt_act_flgs)


# In[24]:


plot_perf_metrics(ls_rewards, ls_opt_act_flgs, ls_epsilon, plt_lib='plotly')


# ## Use the constant step size as the valuation scheme

# In[25]:


k_arms, n_steps, n_exps = 10, 10000, 2000
ls_epsilon = [0, 0.01, 0.1, 0.3]
ls_rewards = []
ls_opt_act_flgs = []
for eps in ls_epsilon:
    rewards, opt_act_flgs = k_bandit_sim_eps_nonstationary(k_arms, n_steps, n_exps, eps, val_scheme='constant_step_size', alpha=0.1)
    ls_rewards.append(rewards)
    ls_opt_act_flgs.append(opt_act_flgs)


# In[26]:


plot_perf_metrics(ls_rewards, ls_opt_act_flgs, ls_epsilon, plt_lib='plotly')


# # Ex 2.7
# 
# \begin{align*}
# Q_{n+1} &= Q_n + \frac{\alpha }{\bar{o}_n} (R_n - Q_n) \\
#     &= \frac{\alpha}{\bar{o}_n} R_n + \left(1 - \frac{\alpha}{\bar{o}_n}\right) Q_n \\
#     &= \frac{\alpha}{\bar{o}_n} R_n + \left(1 - \frac{\alpha}{\bar{o}_n}\right) \left(\frac{\alpha}{\bar{o}_{n-1}} R_{n-1} + \left(1 - \frac{\alpha}{\bar{o}_{n-1}}\right) Q_{n-1}\right) \\
#     &= \sum_{i=1}^{n} \frac{\alpha R_i}{\bar{o}_i} \prod_{j=i+1}^{n} \left(1 - \frac{\alpha}{\bar{o}_j}\right) + Q_1 \prod_{i=1}^{n} \left(1 - \frac{\alpha}{\bar{o}_i}\right) \\
# \bar{o}_n &= 1-(1-\alpha)^n
# \end{align*}

# # Ex 2.8

# In[27]:


SAMPLE_AVG = 'sample_avg'
CONSTANT_STEP_SIZE = 'constant_step_size'

def k_bandit_sim_ucb(k, n_steps, n_exps, init_q_star=0, val_scheme='sample_avg', alpha=0.1, ucb_c=1):
    def _one_sim(i):
        # Initialize the q_star values
        q_star = np.random.normal(0, 1, k)
        opt_a = np.argmax(q_star)

        # Initialize the q values
        q = np.ones(k)*init_q_star

        # Initialize the number of times each action was taken
        n = np.zeros(k)

        rewards = np.zeros(n_steps)
        opt_act_flgs = np.zeros(n_steps)

        for j in range(n_steps):
            # Choose an action
            if ucb_c>=0:
                ucb = q + ucb_c*np.sqrt(np.log(j+1)/(n+1e-5))
                a = np.argmax(ucb)
            else:
                a = np.argmax(q)
            
            # Get the reward
            reward = np.random.normal(q_star[a], 1)

            # Update the q values
            if val_scheme == 'sample_avg':
                n[a] += 1
                q[a] += (reward - q[a]) / n[a]
            else: # constant step-size
                q[a] += alpha*(reward - q[a])
                
            # Store the reward
            rewards[j] = reward
            opt_act_flgs[j] = int(a == opt_a)
            
        return rewards, opt_act_flgs
                
    with parallel_backend('loky', n_jobs=N_JOBS):
        results = Parallel()(delayed(_one_sim)(i) for i in range(n_exps))
        
    rewards, opt_act_flgs = zip(*results)
    rewards = np.mean(np.array(rewards), axis=0)
    opt_act_flgs = np.mean(np.array(opt_act_flgs), axis=0)
    
    return rewards, opt_act_flgs      
    


# In[28]:


# before parallelization, it took 7 minutes
ls_ucb_c = [0, 1, 2, 5, 10]
epsilon = 0.1
k_arms, n_steps, n_exps = 10, 5000, 2000
ls_rewards = []
ls_opt_act_flgs = []
rewards, opt_act_flgs = k_bandit_sim_eps(k_arms, n_steps, n_exps, epsilon)
ls_rewards.append(rewards)
ls_opt_act_flgs.append(opt_act_flgs)
for ucb_c in ls_ucb_c:
    rewards, opt_act_flgs = k_bandit_sim_ucb(k_arms, n_steps, n_exps, ucb_c=ucb_c)
    ls_rewards.append(rewards)
    ls_opt_act_flgs.append(opt_act_flgs)


# In[29]:


param_prefix = ['epsilon'] + ['ucb_c']*len(ls_ucb_c)
ls_param = [epsilon] + ls_ucb_c
plot_perf_metrics(ls_rewards, ls_opt_act_flgs, ls_param, param_prefix=param_prefix, plt_lib='plotly')


# # Gradient Bandit

# In[30]:


def k_bandit_sim_gradient(k, n_steps, n_exps, alpha=0.1, baseline=True, init_h=0):
    def _one_sim(i):
        # Initialize the q_star values
        q_star = np.random.normal(0, 1, k)
        opt_a = np.argmax(q_star)

        # Initialize the avg reward
        avg_reward = 0
        
        # Initialize the preferences
        h = np.ones(k) * init_h
        sum_h = np.sum(h)

        rewards = np.zeros(n_steps)
        opt_act_flgs = np.zeros(n_steps)

        for j in range(n_steps):
            # Choose an action
            pi = np.exp(h) / np.sum(np.exp(h))
            a = np.random.choice(k, p=pi)
            
            # Get the reward
            reward = np.random.normal(q_star[a], 1)
            
            # Update the avg reward
            avg_reward += (reward - avg_reward)/(j+1)
            
            # Update the preferences
            one_hot = np.zeros(k)
            one_hot[a] = 1
            bv = avg_reward if baseline else 0
            h += alpha*(reward - bv)*(one_hot - pi)
            
            # # Make sure h doesn't change the sum val
            # h -= (np.sum(h) - sum_h)/k
            
            # Store the reward
            rewards[j] = reward
            opt_act_flgs[j] = int(a == opt_a)
            
        return rewards, opt_act_flgs
    
    with parallel_backend('loky', n_jobs=N_JOBS):
        results = Parallel()(delayed(_one_sim)(i) for i in range(n_exps))
        
    rewards, opt_act_flgs = zip(*results)
    rewards = np.mean(np.array(rewards), axis=0)
    opt_act_flgs = np.mean(np.array(opt_act_flgs), axis=0)
    
    return rewards, opt_act_flgs


# In[31]:


k_arms, n_steps, n_exps = 10, 1000, 2000
ls_alpha = [0.1, 0.4]
ls_baseline = [True, False]
ls_rewards = []
ls_opt_act_flgs = []
for alpha, baseline in product(ls_alpha, ls_baseline):
    rewards, opt_act_flgs = k_bandit_sim_gradient(k_arms, n_steps, n_exps, alpha=alpha, baseline=baseline)
    ls_rewards.append(rewards)
    ls_opt_act_flgs.append(opt_act_flgs)


# In[32]:


ls_param = list(product(ls_alpha, ls_baseline))
ls_param = ['{}, {}'.format(alpha, baseline) for alpha, baseline in ls_param]
plot_perf_metrics(ls_rewards, ls_opt_act_flgs, ls_param, param_prefix='alpha, baseline', plt_lib='plotly')


# # Ex 2.11

# In[33]:


k_arms, n_steps, n_exps = 10, 20000, 500

# epsilon-greedy
ls_epsilon = [2.0**i for i in np.arange(-7, -2, .5)]
ls_eps_rewards = []
ls_eps_opt_act_flgs = []
for epsilon in ls_epsilon:
    rewards, opt_act_flgs = k_bandit_sim_eps(k_arms, n_steps, n_exps, epsilon)
    ls_eps_rewards.append(rewards)
    ls_eps_opt_act_flgs.append(opt_act_flgs)
    
# espilon-greedy with optimistic initialization
ls_opt_init = [2.0**i for i in np.arange(-5, 3, .5)]
ls_eps_opt_init_rewards = []
ls_eps_opt_init_opt_act_flgs = []
for init_q_star in ls_opt_init:
    rewards, opt_act_flgs = k_bandit_sim_eps(k_arms, n_steps, n_exps, 0.1, init_q_star=init_q_star)
    ls_eps_opt_init_rewards.append(rewards)
    ls_eps_opt_init_opt_act_flgs.append(opt_act_flgs)
    
# UCB
ls_ucb_c = [2.0**i for i in np.arange(-6, 3, .5)]
ls_ubc_rewards = []
ls_ubc_opt_act_flgs = []
for ucb_c in ls_ucb_c:
    rewards, opt_act_flgs = k_bandit_sim_ucb(k_arms, n_steps, n_exps, ucb_c=ucb_c)
    ls_ubc_rewards.append(rewards)
    ls_ubc_opt_act_flgs.append(opt_act_flgs)
    
# Gradient
ls_alpha = [2.0**i for i in np.arange(-7, 3, .5)]
ls_gradient_rewards = []
ls_gradient_opt_act_flgs = []
for alpha in ls_alpha:
    rewards, opt_act_flgs = k_bandit_sim_gradient(k_arms, n_steps, n_exps, alpha=alpha)
    ls_gradient_rewards.append(rewards)
    ls_gradient_opt_act_flgs.append(opt_act_flgs)


# In[34]:


fig = make_subplots(rows=2, cols=1)
ls_avg_rewards = []
ls_avg_opt_act_flgs = []
for i, (ls_rewards, ls_opt_act_flgs, ls_param, method_name) in enumerate([
    (ls_eps_rewards, ls_eps_opt_act_flgs, ls_epsilon, 'epsilon'), 
    (ls_eps_opt_init_rewards, ls_eps_opt_init_opt_act_flgs, ls_opt_init, 'epsilon_opt_init'),
    (ls_ubc_rewards, ls_ubc_opt_act_flgs, ls_ucb_c, 'ucb'),
    (ls_gradient_rewards, ls_gradient_opt_act_flgs, ls_alpha, 'gradient')]
):
    avg_rewards = np.mean(np.array(ls_rewards)[:, :n_steps//2], axis=1)
    avg_opt_act_flgs = np.mean(np.array(ls_opt_act_flgs)[:, :n_steps//2], axis=1)
    ls_avg_rewards.append(avg_rewards)
    ls_avg_opt_act_flgs.append(avg_opt_act_flgs)
    text = [None for _ in range(len(ls_param))]
    text[len(ls_param)//2] = method_name
    fig.add_trace(
        go.Scatter(
            x=ls_param, y=avg_rewards, mode='lines+text', line=dict(color=COLOR_LIST[i]), name=f"{method_name}",
            text=text, textposition='top center', textfont=dict(color=COLOR_LIST[i])  
        ), 
        row=1, col=1)
    fig.add_trace(
        go.Scatter(
            x=ls_param, y=avg_opt_act_flgs, mode='lines+text', line=dict(color=COLOR_LIST[i]), name=f"{method_name}",
            text=text, textposition='top center', textfont=dict(color=COLOR_LIST[i]),
            showlegend=False    
        ), 
        row=2, col=1)
    
fig.update_xaxes(title_text='Parameter', row=1, col=1)
fig.update_yaxes(title_text='Average Reward', row=1, col=1)
fig.update_xaxes(title_text='Parameter', row=2, col=1)
fig.update_yaxes(title_text='% Optimal Action', row=2, col=1)

fig.update_layout(height=800, width=800, title_text='First half avg rewards', hovermode='x', xaxis_type='log', xaxis2_type='log')
fig.show()


# In[35]:


fig = make_subplots(rows=2, cols=1)
ls_avg_rewards = []
ls_avg_opt_act_flgs = []
for i, (ls_rewards, ls_opt_act_flgs, ls_param, method_name) in enumerate([
    (ls_eps_rewards, ls_eps_opt_act_flgs, ls_epsilon, 'epsilon'), 
    (ls_eps_opt_init_rewards, ls_eps_opt_init_opt_act_flgs, ls_opt_init, 'epsilon_opt_init'),
    (ls_ubc_rewards, ls_ubc_opt_act_flgs, ls_ucb_c, 'ucb'),
    (ls_gradient_rewards, ls_gradient_opt_act_flgs, ls_alpha, 'gradient')]
):
    avg_rewards = np.mean(np.array(ls_rewards)[:, n_steps//2:], axis=1)
    avg_opt_act_flgs = np.mean(np.array(ls_opt_act_flgs)[:, n_steps//2:], axis=1)
    ls_avg_rewards.append(avg_rewards)
    ls_avg_opt_act_flgs.append(avg_opt_act_flgs)
    text = [None for _ in range(len(ls_param))]
    text[len(ls_param)//2] = method_name
    fig.add_trace(
        go.Scatter(
            x=ls_param, y=avg_rewards, mode='lines+text', line=dict(color=COLOR_LIST[i]), name=f"{method_name}",
            text=text, textposition='top center', textfont=dict(color=COLOR_LIST[i])  
        ), 
        row=1, col=1)
    fig.add_trace(
        go.Scatter(
            x=ls_param, y=avg_opt_act_flgs, mode='lines+text', line=dict(color=COLOR_LIST[i]), name=f"{method_name}",
            text=text, textposition='top center', textfont=dict(color=COLOR_LIST[i]),
            showlegend=False    
        ), 
        row=2, col=1)
    
fig.update_xaxes(title_text='Parameter', row=1, col=1)
fig.update_yaxes(title_text='Average Reward', row=1, col=1)
fig.update_xaxes(title_text='Parameter', row=2, col=1)
fig.update_yaxes(title_text='% Optimal Action', row=2, col=1)

fig.update_layout(height=800, width=800, title_text='Second half avg rewards', hovermode='x', xaxis_type='log', xaxis2_type='log')
fig.show()


# In[ ]:





# # End

# In[ ]:




