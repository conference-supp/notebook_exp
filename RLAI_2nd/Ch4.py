# %% [markdown]
# <h1>Table of Contents<span class="tocSkip"></span></h1><ul>
# <li style="margin-left: 0px;"><a href="#1.-import-dep">1. Import dep</a></li>
# <li style="margin-left: 0px;"><a href="#2.-example-4.1">2. Example 4.1</a></li>
# <li style="margin-left: 20px;"><a href="#2.1.-ex-4.1">2.1. Ex 4.1</a></li>
# <li style="margin-left: 20px;"><a href="#2.2.-ex-4.2">2.2. Ex 4.2</a></li>
# <li style="margin-left: 20px;"><a href="#2.3.-ex-4.4">2.3. Ex 4.4</a></li>
# <li style="margin-left: 0px;"><a href="#3.-ex-4.5">3. Ex 4.5</a></li>
# <li style="margin-left: 0px;"><a href="#4.-ex-4.7">4. Ex 4.7</a></li>
# <li style="margin-left: 20px;"><a href="#4.1.-replicate-fig.-4.2">4.1. Replicate Fig. 4.2</a></li>
# <li style="margin-left: 40px;"><a href="#4.1.1.-one-step-of-policy-evaluation">4.1.1. One step of policy evaluation</a></li>
# <li style="margin-left: 40px;"><a href="#4.1.2.-policy-evaluation">4.1.2. Policy evaluation</a></li>
# <li style="margin-left: 40px;"><a href="#4.1.3.-policy-improvement">4.1.3. Policy improvement</a></li>
# <li style="margin-left: 40px;"><a href="#4.1.4.-run-the-policy-iteration">4.1.4. Run the policy iteration</a></li>
# <li style="margin-left: 20px;"><a href="#4.2.-value-iteration-as-a-comparison">4.2. Value iteration as a comparison</a></li>
# <li style="margin-left: 20px;"><a href="#4.3.-add-more-details">4.3. Add more details</a></li>
# <li style="margin-left: 0px;"><a href="#5.-example-4.8">5. Example 4.8</a></li>
# <li style="margin-left: 20px;"><a href="#5.1.-ex-4.9">5.1. Ex 4.9</a></li>
# <li style="margin-left: 20px;"><a href="#5.2.-optimal-sol-in-the-book">5.2. Optimal sol in the book</a></li>
# <li style="margin-left: 0px;"><a href="#6.-ex-4.10">6. Ex 4.10</a></li>
# <li style="margin-left: 0px;"><a href="#7.-end">7. End</a></li>
# </ul>

# %%
from IPython.display import display, HTML

# Set the notebook width to 80%
display(HTML("<style>.container { width: 80% !important; }</style>"))

# %%
!jupyter notebook list

# %%
# Needs to paste `http://localhost:3110`, no ending `/`
port = 2810

import IPython
import json
import requests

hostname = !hostname

# Get the current Jupyter server's info
result = !jupyter notebook list
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

# %% [markdown]
# # 1. Import dep
# <a id="1.-import-dep"></a>

# %%
from itertools import product

import numpy as np
import matplotlib.pyplot as plt
%matplotlib notebook

# %%
import plotly
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import plotly.io as pio

pio.renderers.default = "notebook"

# %%
COLOR_LIST = plotly.colors.DEFAULT_PLOTLY_COLORS
len(COLOR_LIST)

# %%
print(plotly.__version__, plotly.__path__)

# %% [markdown]
# # 2. Example 4.1
# <a id="2.-example-4.1"></a>

# %%
def state_val_update1(vnn, ga):
    nr, nc = vnn.shape
    # Using the vnn values below to use the newly obtained v_{n+1} values.
    for i, j in product(range(nr), range(nc)):
        if (i==0 and j==0) or (i==nr-1 and j==nc-1):
            vnn[i, j] = 0
        else:
            # Corners
            if i==0 and j==nc-1:
                vnn[i, j] = -1+sum([ga*vnn[i, j-1], ga*vnn[i, j], ga*vnn[i, j], ga*vnn[i+1, j]])/4
            elif i==nr-1 and j==0:
                vnn[i, j] = -1+sum([ga*vnn[i, j], ga*vnn[i, j+1], ga*vnn[i-1, j], ga*vnn[i, j]])/4
            # Boundaries
            elif i==0:
                vnn[i, j] = -1+sum([ga*vnn[i, j-1], ga*vnn[i, j+1], ga*vnn[i, j], ga*vnn[i+1, j]])/4
            elif i==nr-1:
                vnn[i, j] = -1+sum([ga*vnn[i, j-1], ga*vnn[i, j+1], ga*vnn[i-1, j], ga*vnn[i, j]])/4
            elif j==0:
                vnn[i, j] = -1+sum([ga*vnn[i, j], ga*vnn[i, j+1], ga*vnn[i-1, j], ga*vnn[i+1, j]])/4
            elif j==nc-1:
                vnn[i, j] = -1+sum([ga*vnn[i, j-1], ga*vnn[i, j], ga*vnn[i-1, j], ga*vnn[i+1, j]])/4
            # Inner
            else:
                vnn[i, j] = -1+sum([ga*vnn[i, j-1], ga*vnn[i, j+1], ga*vnn[i-1, j], ga*vnn[i+1, j]])/4


def eval_schema2(
    state_val_update_func,
    v0: np.ndarray, ga: float, max_iter: int, memo_step: int = 1, plt_trace=False, trace_ij=(0, 0)
):
    """
    When updating the value function at n+1, newly calcualted n+1 values are also used.
    RLAI book call this "in-place" algorithm.
    """
    vn = v0.copy()
    memo_v = {}
    for si in range(max_iter):
        vnn = vn.copy()
        
        state_val_update_func(vnn, ga)
            
        if si%memo_step==0:
            memo_v[si] = vn

        vn = vnn
    
    memo_v[max_iter] = vn
    
    if plt_trace:
        fig = make_subplots(rows=1, cols=1)
        arr_si = np.array(sorted(memo_v.keys()))
        vs = [memo_v[si][trace_ij] for si in arr_si]
        fig.add_trace(go.Scatter(x=arr_si+1, y=vs, mode='lines+markers'), row=1, col=1)
        fig.update_layout(
            xaxis=dict(title='Iteration', type='log'),
            title=f'Value Function at Iteration {trace_ij}', height=600, width=600)
        fig.show()
    
    return memo_v

def pr_val_func(val: np.ndarray, fmt='%8.3f', reshape=False, nr=None, nc=None):
    if reshape:
        print(np.array2string(val.reshape(nr, nc, order='C'), formatter={'float_kind':lambda x: fmt % x}))
    else:
        print(np.array2string(val, formatter={'float_kind':lambda x: fmt % x}))
        

# %% [markdown]
# ## 2.1. Ex 4.1
# <a id="2.1.-ex-4.1"></a>

# %%
v0 = np.zeros((4, 4))
ga = 1.0
max_iter = 1000
memo_step = 10
memo_v = eval_schema2(state_val_update1, v0, ga, max_iter, memo_step, plt_trace=True, trace_ij=(1, 3))
pr_val_func(memo_v[max_iter])

# %%
q_11_down = -1+memo_v[max_iter][3, 3]
q_7_down = -1+memo_v[max_iter][2, 3]
q_11_down, q_7_down

# %% [markdown]
# ## 2.2. Ex 4.2
# <a id="2.2.-ex-4.2"></a>

# %%
def state_val_update2(vnn, ga):
    def _ij(i, j):
        return i*4+j
    nr, nc = 4, 4
    # Using the vnn values below to use the newly obtained v_{n+1} values.
    for i, j in product(range(nr), range(nc)):
        if (i==0 and j==0) or (i==nr-1 and j==nc-1):
            vnn[_ij(i, j)] = 0
        else:
            # Corners
            if i==0 and j==nc-1:
                vnn[_ij(i, j)] = -1+sum([ga*vnn[_ij(i, j-1)], ga*vnn[_ij(i, j)], ga*vnn[_ij(i, j)], ga*vnn[_ij(i+1, j)]])/4
            elif i==nr-1 and j==0:
                vnn[_ij(i, j)] = -1+sum([ga*vnn[_ij(i, j)], ga*vnn[_ij(i, j+1)], ga*vnn[_ij(i-1, j)], ga*vnn[_ij(i, j)]])/4
            # Boundaries
            elif i==0:
                vnn[_ij(i, j)] = -1+sum([ga*vnn[_ij(i, j-1)], ga*vnn[_ij(i, j+1)], ga*vnn[_ij(i, j)], ga*vnn[_ij(i+1, j)]])/4
            elif i==nr-1:
                vnn[_ij(i, j)] = -1+sum([ga*vnn[_ij(i, j-1)], ga*vnn[_ij(i, j+1)], ga*vnn[_ij(i-1, j)], ga*vnn[_ij(i, j)]])/4
            elif j==0:
                vnn[_ij(i, j)] = -1+sum([ga*vnn[_ij(i, j)], ga*vnn[_ij(i, j+1)], ga*vnn[_ij(i-1, j)], ga*vnn[_ij(i+1, j)]])/4
            elif j==nc-1:
                vnn[_ij(i, j)] = -1+sum([ga*vnn[_ij(i, j-1)], ga*vnn[_ij(i, j)], ga*vnn[_ij(i-1, j)], ga*vnn[_ij(i+1, j)]])/4
            # Inner
            else:
                vnn[_ij(i, j)] = -1+sum([ga*vnn[_ij(i, j-1)], ga*vnn[_ij(i, j+1)], ga*vnn[_ij(i-1, j)], ga*vnn[_ij(i+1, j)]])/4
    vnn[-1] = -1+sum([ga*vnn[_ij(3, 0)], ga*vnn[_ij(3, 1)], ga*vnn[_ij(3, 2)], ga*vnn[-1]])/4

# %%
v0 = np.zeros((17,))
ga = 1.0
max_iter = 1000
memo_step = 10
memo_v = eval_schema2(state_val_update2, v0, ga, max_iter, memo_step, plt_trace=True, trace_ij=-1)
pr_val_func(memo_v[max_iter][:16], reshape=True, nr=-1, nc=4)
memo_v[max_iter][-1]

# %% [markdown]
# ## 2.3. Ex 4.4
# <a id="2.3.-ex-4.4"></a>
# - The bug is that the optimal policy is not unique. So even though the policy improvement shows that the `policy-stable=false`, the algorithm may not give a better state-value function in the next policy evaluation. Actually, the algorithm can circulating between different optimal policies. (In other words, ties should be broken in a consistent order.)
#     - One way to fix it is that in policy improvement, we don't use the current way to assign $\pi(s)$. Instead, we check if there is an action giving a better value of existing $p(s, a)$. If so, we assign the action to the $\pi(s)$ and assign `policy-stable=false`, otherwise we don't do anything for the current $\pi(s)$.
#     - Another way of fixing is that, within policy evaluation, we can also check if the current state-value function is close to the state-value function in the last time policy evaluation. If there is a strict improvement in 

# %% [markdown]
# # 3. Ex 4.5
# <a id="3.-ex-4.5"></a>
# 
# \begin{align*}
# q_{\pi}(s, a) &= \mathbb{E}[R_{t+1}+\gamma*v_{\pi}(S_{t+1})|s, a] \\
# &= \mathbb{E}[R_{t+1}+\gamma*\sum_{a}\pi(a|S_{t+1})q_{\pi}(S_{t+1}, a)|s, a] \\
# &= \sum_{s', r}p(s',r|s,a)(r+\gamma*\sum_{a}\pi(a|s')q_{\pi}(s', a))
# \end{align*}
# 
# At each time step $k$, we have $q_k(s,a)$. Then, we can use this $q_k(s,a)$ to evaluate $q_{k+1}(s,a)$ using the equation above. This is policy evaluation for the action-value function.

# %% [markdown]
# # 4. Ex 4.7
# <a id="4.-ex-4.7"></a>

# %% [markdown]
# ## 4.1. Replicate Fig. 4.2
# <a id="4.1.-replicate-fig.-4.2"></a>
# - The solution is the same with Fig. 4.2.

# %%
from itertools import product
import math
import time

from joblib import Parallel, delayed, parallel_backend

N_JOBS = 20

MV_COST = 2
RENT_RWD = 10
ARR1, ARR2 = 3, 4
RET1, RET2 = 3, 2
MAX_CARS = 20
MAX_CAR_MOVE = 5

POISSON_CACHE_ARR1 = [np.exp(-ARR1)*ARR1**arr1/math.factorial(arr1) for arr1 in range(MAX_CARS)]
POISSON_CACHE_CUMU_ARR1 = np.cumsum(POISSON_CACHE_ARR1)
POISSON_CACHE_ARR2 = [np.exp(-ARR2)*ARR2**arr2/math.factorial(arr2) for arr2 in range(MAX_CARS)]
POISSON_CACHE_CUMU_ARR2 = np.cumsum(POISSON_CACHE_ARR2)
POISSON_CACHE_RET1 = [np.exp(-RET1)*RET1**ret1/math.factorial(ret1) for ret1 in range(MAX_CARS)]
POISSON_CACHE_CUMU_RET1 = np.cumsum(POISSON_CACHE_RET1)
POISSON_CACHE_RET2 = [np.exp(-RET2)*RET2**ret2/math.factorial(ret2) for ret2 in range(MAX_CARS)]
POISSON_CACHE_CUMU_RET2 = np.cumsum(POISSON_CACHE_RET2)


# %% [markdown]
# ### 4.1.1. One step of policy evaluation
# <a id="4.1.1.-one-step-of-policy-evaluation"></a>
# - Transition prob and average rewards from $s$ to $s'$
# 
# \begin{align*}
# & \sum_{r}p(s', r|s, \pi(s))[r+\gamma V(s')] \\ 
# = & \sum_{r}p(s', r|s, \pi(s))r + \gamma V(s')\sum_{r}p(s', r|s, \pi(s)) \\
# = & \left[r(s,\pi(s),s') + \gamma V(s')\right]p(s'|s,\pi(s))
# \end{align*}
# 
# - In this problem, the two car rental locations can be decoupled from each other, once the number of cars moved overnight is taken into account. This largely reduces the computational complexity. Naively implementing the transition probability function would make the program too slow.
# 
# - The computation complexity of one transition probability calculation is $\mu_T = O(N)$, where $N$ is the maximum number of car one location is allowed.

# %%
# # The following is a naive implementation of the car rental problem, which is very slow.
# def car_rental_trans(c1, c2, mv, c1n, c2n):
#     """
#     Calculate the transition probability and expected reward for the car rental problem from 
#     state (c1, c2) to state (c1n, c2n) by moving mv cars from location 1 to location 2.
    
#     c1 is the number of cars at location 1.
#     c2 is the number of cars at location 2.
#     mv is the number of cars moved from location 1 to location 2.
#     c1n is the number of cars at location 1 at the end of the next day.
#     c2n is the number of cars at location 2 at the end of the next day.
#     """
#     if ((mv>0 and c1<mv) or (mv<0 and c2<-mv)):
#         return 0.0, 0.0
#     c1_be = min(MAX_CARS, c1-mv)
#     c2_be = min(MAX_CARS, c2+mv)
    
#     # c1n = min(MAX_CARS, c1_be-min(c1_be, arr1)+ret1)
#     r_sa_sp, p_sa_sp = 0, 0
#     for arr1, arr2 in product(range(c1_be+1), range(c2_be+1)):
#         ret1 = c1n-(c1_be-arr1)
#         ret2 = c2n-(c2_be-arr2)
#         if ret1<0 or ret2<0:
#             continue
#         p_arr1 = POISSON_CACHE_ARR1[arr1] if arr1<c1_be else 1-POISSON_CACHE_CUMU_ARR1[c1_be-1]
#         p_arr2 = POISSON_CACHE_ARR2[arr2] if arr2<c2_be else 1-POISSON_CACHE_CUMU_ARR2[c2_be-1]
#         p_ret1 = POISSON_CACHE_RET1[ret1] if c1n<MAX_CARS else 1-POISSON_CACHE_CUMU_RET1[ret1-1]
#         p_ret2 = POISSON_CACHE_RET2[ret2] if c2n<MAX_CARS else 1-POISSON_CACHE_CUMU_RET2[ret2-1]            
#         prob = p_arr1*p_arr2*p_ret1*p_ret2
#         r_sa_sp += RENT_RWD*(arr1+arr2)*prob
#         p_sa_sp += prob
#     r_sa_sp -= MV_COST*abs(mv)*p_sa_sp
#     return r_sa_sp, p_sa_sp

# Decouple the car rental problem into two locations to speed up the calculation.
def car_rental_one_loc_trans(c_be, c_ed, loc):
    """
    Calculate the transition probability and expected reward for the car rental problem from 
    state c_be to state c_ed.
    
    c_be is the number of cars at the beginning of the day.
    c_ed is the number of cars at the end of the day.
    loc is the location of the cars.
    """
    r_sa_sp, p_sa_sp = 0, 0
    # arr>=c_be-c_ed
    arr_arr = np.arange(max(0, c_be-c_ed), c_be+1)
    pois_cache_arr = POISSON_CACHE_ARR1 if loc==1 else POISSON_CACHE_ARR2
    pois_cache_cum_arr = POISSON_CACHE_CUMU_ARR1 if loc==1 else POISSON_CACHE_CUMU_ARR2
    pois_cache_ret = POISSON_CACHE_RET1 if loc==1 else POISSON_CACHE_RET2
    pois_cache_cum_ret = POISSON_CACHE_CUMU_RET1 if loc==1 else POISSON_CACHE_CUMU_RET2
    # There can be a subtle bug here when c_be=0
    p_arr = np.array([    
        pois_cache_arr[arr] if arr<c_be else ((1-pois_cache_cum_arr[c_be-1]) if c_be>0 else 1)
        for arr in arr_arr
    ])
    # There can be a subtle bug here when ret=0
    def _ret_prob(arr):
        ret = c_ed-(c_be-arr)
        return pois_cache_ret[ret] if c_ed<MAX_CARS else ((1-pois_cache_cum_ret[ret-1]) if ret>0 else 1)
    p_ret = np.array([_ret_prob(arr) for arr in arr_arr])
    r_sa_sp = RENT_RWD*np.sum(arr_arr*p_arr*p_ret)
    p_sa_sp = np.sum(p_arr*p_ret)
    
    return r_sa_sp, p_sa_sp

def car_rental_trans1(c1, c2, mv, c1n, c2n):
    """
    Calculate the transition probability and expected reward for the car rental problem from 
    state (c1, c2) to state (c1n, c2n) by moving mv cars from location 1 to location 2.
    
    c1 is the number of cars at location 1.
    c2 is the number of cars at location 2.
    mv is the number of cars moved from location 1 to location 2.
    c1n is the number of cars at location 1 at the end of the next day.
    c2n is the number of cars at location 2 at the end of the next day.
    """
    if ((mv>0 and c1<mv) or (mv<0 and c2<-mv)):
        return 0.0, 0.0
    c1_be = min(MAX_CARS, c1-mv)
    c2_be = min(MAX_CARS, c2+mv)
    
    r_sa_sp1, p_sa_sp1 = car_rental_one_loc_trans(c1_be, c1n, loc=1)
    r_sa_sp2, p_sa_sp2 = car_rental_one_loc_trans(c2_be, c2n, loc=2)
    p_sa_sp = p_sa_sp1*p_sa_sp2
    # Be careful with the probability of the cost
    r_sa_sp = r_sa_sp1*p_sa_sp2+r_sa_sp2*p_sa_sp1-MV_COST*abs(mv)*p_sa_sp
    return r_sa_sp, p_sa_sp


# %% [markdown]
# ### 4.1.2. Policy evaluation
# <a id="4.1.2.-policy-evaluation"></a>
# - Loop over all the state $s$
# 
# \begin{align*}
# V_{k+1}(s) \leftarrow & \sum_{s'}\left[r(s,\pi(s),s') + \gamma V_k(s')\right]p(s'|s,\pi(s)) \\
# = & r(s, \pi(s)) + \gamma \sum_{s'} p(s'|s,\pi(s)) V_k(s')
# \end{align*}
# 
# - Then loop over $k$ until the $max_{s}|V_{k+1}(s)-V_{k}(s)|$ is smaller than a tolerance.
# 
# - Parallizing each state in the policy evaluation instead of within the evaluation of each state makes the policy evaluation loop much faster. This way program finishes in a reasonable time.
# 
# - The computation complexity of one iteration from $k$ to $k+1$ is $|\mathcal{S}|^2*\mu_T$, where $|\mathcal{S}| = O(N^2)$.

# %%
def car_rental_pol_eval(
    car_rental_trans_func: callable,
    arr_pol: np.ndarray, arr_val: np.ndarray, ga: float, pol_iter: int,
    tol: 1e-4, max_iter: int, memo_step: int = 1, plt_trace=False, trace_ij=(0, 0)
):
    nr, nc = arr_pol.shape
    arr_valn = arr_val.copy()
    memo = {0: arr_valn[trace_ij]}
    m_diff, avg_diff, tic = 0, 0, time.time()
    # abs_tol = max(tol*RENT_RWD, 2**(-pol_iter))
    abs_tol = tol*RENT_RWD
    
    def _pol_eval_one_sweep(arr_val):
        def _one_update(i, j):
            arr_r_sa_sp, arr_p_sa_sp = np.zeros_like(arr_val), np.zeros_like(arr_val)
            for ni, nj in product(range(nr), range(nc)):
                r_sa_sp, p_sa_sp = car_rental_trans_func(i, j, arr_pol[i, j], ni, nj)
                arr_r_sa_sp[ni, nj], arr_p_sa_sp[ni, nj] = r_sa_sp, p_sa_sp
            return np.sum(arr_r_sa_sp+ga*arr_p_sa_sp*arr_val)
        with parallel_backend('loky', n_jobs=N_JOBS):
            res = Parallel(verbose=0, pre_dispatch="1.5*n_jobs")(
                delayed(_one_update)(i, j) for i, j in product(range(nr), range(nc))
            )
        arr_valn = np.array(res).reshape(nr, nc, order='C')
        return arr_valn
                
    for si in range(max_iter):
        m_diff, avg_diff, tic = 0, 0, time.time()
        # for i, j in product(range(nr), range(nc)):
        #     # print(i, j)
        #     # valn = 0
        #     # for mv, ni, nj in product(range(-MAX_CAR_MOVE, MAX_CAR_MOVE+1), range(nr), range(nc)):
        #     #     print(i, j, mv, ni, nj)
        #     #     r_sa_sp, p_sa_sp = car_rental_trans(i, j, mv, ni, nj)
        #     #     valn += r_sa_sp + ga*p_sa_sp*arr_valn[ni, nj]
        #     with parallel_backend('loky', n_jobs=N_JOBS):
        #         res = Parallel(verbose=0, pre_dispatch="1.5*n_jobs")(
        #             delayed(car_rental_trans)(i, j, arr_pol[i, j], ni, nj) 
        #             for ni, nj in product(range(nr), range(nc))
        #         )
        #     arr_r_sa_sp, arr_p_sa_sp = zip(*res)
        #     arr_r_sa_sp, arr_p_sa_sp = np.array(arr_r_sa_sp), np.array(arr_p_sa_sp)
        #     valn = np.sum(arr_r_sa_sp+ga*arr_p_sa_sp*arr_valn.reshape(-1, order='C'))
        #     m_diff = max(m_diff, abs(valn-arr_valn[i, j]))
        #     avg_diff += abs(valn-arr_valn[i, j])
        #     arr_valn[i, j] = valn
        arr_valn = _pol_eval_one_sweep(arr_val)
        m_diff = np.max(np.abs(arr_valn-arr_val))
        avg_diff = np.mean(np.abs(arr_valn-arr_val))
        arr_val = arr_valn
        if si%memo_step==0:
            memo[si] = arr_valn[trace_ij]
        if si%10==0:
            print(f'Policy evaluation iteration {si}: {m_diff:.2f}:{avg_diff:.2f} ({time.time()-tic:.2f}s)')
        if m_diff<abs_tol:
            print(f'Converged at iteration {si}')
            break
            
    if plt_trace and len(memo)>1:
        fig = make_subplots(rows=1, cols=1)
        arr_si = np.array(sorted(memo.keys()))
        vs = [memo[si] for si in arr_si]
        fig.add_trace(go.Scatter(x=arr_si+1, y=vs, mode='lines+markers'), row=1, col=1)
        fig.update_layout(
            xaxis=dict(title='Iteration', type='log'),
            title=f'Value Function at Iteration {trace_ij}', height=600, width=600
        )
        fig.show()
    return arr_valn


# %% [markdown]
# ### 4.1.3. Policy improvement
# <a id="4.1.3.-policy-improvement"></a>
# - Loop over all the state $s$
# 
# \begin{align*}
# \pi(s) \leftarrow & \argmax_{a} q_{\pi}(s, a) \\
# = & \argmax_{a} \sum_{s', r}p(s', r|s, a)[r+\gamma V_{\pi}(s')]
# \end{align*}
# 
# - Check if the policy changed or not. If not changed, stop. Otherwise, go to Policy Evaluation.
# 
# - The computation complexity of this step is $|\mathcal{S}|^2*|\mathcal{A}|*\mu_{T}$, where $\mu_{T}$ is the computation complexity of one transition probability calculation.

# %%
def eval_act_val(car_rental_trans_func, arr_val, i, j, mv, ga):
    nr, nc = arr_val.shape
    q_sa = 0
    trans_prob = 0
    for ni, nj in product(range(nr), range(nc)):
        r_sa_sp, p_sa_sp = car_rental_trans_func(i, j, mv, ni, nj)
        q_sa += r_sa_sp + ga*p_sa_sp*arr_val[ni, nj]
        trans_prob += p_sa_sp
    return q_sa

# def car_rental_pol_impr(arr_pol: np.ndarray, arr_val: np.ndarray, ga: float):
#     nr, nc = arr_pol.shape
#     arr_poln = arr_pol.copy()
#     pol_stab = True
#     for i, j in product(range(nr), range(nc)):
#         q_sa_memo = {}
#         a_max, q_max = arr_pol[i, j], -np.inf
#         for mv in range(-MAX_CAR_MOVE, MAX_CAR_MOVE+1):
#             q_sa_memo[mv] = eval_act_val(arr_val, i, j, mv, ga)
#             if q_sa_memo[mv]>q_max:
#                 q_max = q_sa_memo[mv]
#                 a_max = mv
        
#         if q_max>q_sa_memo[arr_pol[i, j]]:
#             arr_poln[i, j] = a_max
#             pol_stab = False
#     return pol_stab, arr_poln

def car_rental_pol_impr(car_rental_trans_func: callable, arr_pol: np.ndarray, arr_val: np.ndarray, ga: float):
    nr, nc = arr_pol.shape
    arr_poln = arr_pol.copy()
    
    def _one_update(i, j):
        q_sa_memo = {}
        a_max, q_max = arr_pol[i, j], -np.inf
        for mv in range(-MAX_CAR_MOVE, MAX_CAR_MOVE+1):
            q_sa_memo[mv] = eval_act_val(car_rental_trans_func, arr_val, i, j, mv, ga)
            if q_sa_memo[mv]>q_max:
                a_max, q_max = mv, q_sa_memo[mv]
        if q_max>arr_val[i, j]:
            return a_max
        else:
            return arr_pol[i, j]
    
    with parallel_backend('loky', n_jobs=N_JOBS):
        res = Parallel(verbose=0, pre_dispatch="1.5*n_jobs")(
            delayed(_one_update)(i, j) for i, j in product(range(nr), range(nc))
        )
    arr_poln = np.array(res).reshape(nr, nc, order='C')
    pol_stab = np.all(arr_poln==arr_pol)
    
    return pol_stab, arr_poln


# %% [markdown]
# ### 4.1.4. Run the policy iteration
# <a id="4.1.4.-run-the-policy-iteration"></a>
# - The total computation complexity is $n_{poli\_impr}*(n_{poli\_eval}+|\mathcal{A}|)*|\mathcal{S}|^2*\mu_T$.

# %%
# def plot_arr_pol(arr_pol):
#     fig = go.Figure(
#         data=go.Heatmap(
#             z=arr_pol,
#             # colorscale='aggrnyl',
#             colorscale='Jet',
#             x=[i for i in range(MAX_CARS+1)],
#             y=[j for j in range(MAX_CARS+1)],
#             hoverongaps=False,
#             zmin=-MAX_CAR_MOVE,
#             zmax=MAX_CAR_MOVE
#         )
#     )
#     fig.update_layout(
#         title='Policy Heatmap',
#         xaxis=dict(title='Number of Cars at Location 2', showgrid=True, gridwidth=1, gridcolor='black'),
#         yaxis=dict(title='Number of Cars at Location 1', showgrid=True, gridwidth=1, gridcolor='black'),
#         autosize=False,
#         width=500,
#         height=500,
#     )
#     fig.show()

# def plot_arr_val(arr_val):
#     fig = go.Figure(
#         data=go.Heatmap(
#             z=arr_val,
#             # colorscale='aggrnyl',
#             colorscale='Jet',
#             x=[i for i in range(MAX_CARS+1)],
#             y=[j for j in range(MAX_CARS+1)],
#             hoverongaps=False,
#             zmin=400,
#             zmax=650
#         )
#     )

#     fig.update_layout(
#         title='Heatmap of arr_val',
#         xaxis=dict(title='Number of Cars at Location 2', showgrid=True, gridwidth=1, gridcolor='black'),
#         yaxis=dict(title='Number of Cars at Location 1', showgrid=True, gridwidth=1, gridcolor='black'),
#         autosize=False,
#         width=500,
#         height=500,
#     )

#     fig.show()

def plot_arr_pol_and_val(arr_pol, arr_val):
    fig = make_subplots(rows=1, cols=2, horizontal_spacing=0.2, subplot_titles=('Policy Heatmap', 'Value Heatmap'))
    fig.add_trace(
        go.Heatmap(
            z=arr_pol,
            colorscale='Jet',
            x=[i for i in range(MAX_CARS+1)],
            y=[j for j in range(MAX_CARS+1)],
            hoverongaps=False,
            zmin=-MAX_CAR_MOVE,
            zmax=MAX_CAR_MOVE,
            colorbar=dict(title="Policy", x=0.42)  # Adjust x to position the color bar between subplots
        ), row=1, col=1
    )
    fig.add_trace(
        go.Heatmap(
            z=arr_val,
            colorscale='Jet',
            x=[i for i in range(MAX_CARS+1)],
            y=[j for j in range(MAX_CARS+1)],
            hoverongaps=False,
            zmin=400,
            zmax=650,
            colorbar=dict(title="Value", x=1.02)  # Adjust x to position the color bar outside the second subplot
        ), row=1, col=2
    )

    fig.update_layout(
        title='Policy and Value Heatmaps',
        xaxis=dict(title='Number of Cars at Location 2', showgrid=True, gridwidth=1, gridcolor='black'),
        yaxis=dict(title='Number of Cars at Location 1', showgrid=True, gridwidth=1, gridcolor='black'),
        xaxis2=dict(title='Number of Cars at Location 2', showgrid=True, gridwidth=1, gridcolor='black'),
        yaxis2=dict(title='Number of Cars at Location 1', showgrid=True, gridwidth=1, gridcolor='black'),
        autosize=False,
        width=1000,
        height=500,
    )

    fig.show()

def car_rental_pol_iter(car_rental_trans_func, ga, max_pol_iter=-1, plt_iter=False, arr_pol_ini=None, arr_val_ini=None, **pol_eval_kwargs):
    """
    pol_eval_kwargs = {'tol': 1e-4, 'max_iter': 1000, 'memo_step': 1, 'plt_trace': False, 'trace_ij': (0, 0)}
    """        
    arr_pol = np.zeros((MAX_CARS+1, MAX_CARS+1), dtype=int) if arr_pol_ini is None else arr_pol_ini.copy()
    arr_val = np.zeros((MAX_CARS+1, MAX_CARS+1), dtype=float) if arr_val_ini is None else arr_val_ini.copy()
    # for i, j in product(range(MAX_CARS+1), range(MAX_CARS+1)):
    #     arr_val[i, j] = min(i, ARR1)+min(j, ARR2)
    # arr_val *= RENT_RWD/(1-ga)
    
    scnt = 0
    while True:
        scnt += 1
        print("@"*10+f'Policy iteration {scnt}'+"@"*10)
        arr_valn = car_rental_pol_eval(car_rental_trans_func, arr_pol, arr_val, ga, scnt, **pol_eval_kwargs)
        pol_stab, arr_poln = car_rental_pol_impr(car_rental_trans_func, arr_pol, arr_valn, ga)
        if plt_iter:
            print("="*10+f'Iteration {scnt}'+"="*10)
            plot_arr_pol_and_val(arr_pol, arr_val)
            # plot_arr_pol(arr_pol)
            # plot_arr_val(arr_valn)
        if pol_stab or (max_pol_iter>0 and scnt>=max_pol_iter):
            print(f'Policy at iteration {scnt} (policy stable: {pol_stab})')
            arr_val, arr_pol = arr_valn, arr_poln
            break
        arr_val_diff = arr_valn-arr_val
        print(f'Policy at iteration {scnt} (policy stable: {pol_stab}) with min and max val improvement: {np.min(arr_val_diff):.2f}, {np.max(arr_val_diff):.2f}')
        arr_val, arr_pol = arr_valn, arr_poln
    
    return arr_pol, arr_val

# %%
# arr_val = np.zeros((MAX_CARS+1, MAX_CARS+1), dtype=float)
# for i, j in product(range(MAX_CARS+1), range(MAX_CARS+1)):
#     arr_val[i, j] = min(i, ARR1)+min(j, ARR2)
# arr_val *= RENT_RWD/(1-0.9)
# plot_arr_val(arr_val)

# %%
%%time
pol_eval_kwargs = {'tol': 1e-3, 'max_iter': 1000, 'memo_step': 10, 'plt_trace': True, 'trace_ij': (10, 10)}
_ = car_rental_pol_iter(car_rental_trans1, 0.9, max_pol_iter=5, plt_iter=True, **pol_eval_kwargs)

# %% [markdown]
# ## 4.2. Value iteration as a comparison
# <a id="4.2.-value-iteration-as-a-comparison"></a>
# - The value iteration is much faster than policy iteration, b/c it does a little more policy improvements but does far fewer policy evaluations.
# - Even though the final state-value function does not equal to the optimal one, the final policy is the optimal.
# - Actually the final policy is not the optimal, probably b/c the value function is not consistent. But the policy is very close the optimal one. If we run policy iteration after the value iteration, it converges to the optimal one within two iteration, each with very few iterations of policy evaluation.

# %%
%%time
pol_eval_kwargs = {'tol': 1e-3, 'max_iter': 1, 'memo_step': 10, 'plt_trace': True, 'trace_ij': (10, 10)}
arr_pol_val_iter, arr_val_val_iter = car_rental_pol_iter(car_rental_trans1, 0.9, max_pol_iter=100, plt_iter=True, **pol_eval_kwargs)

# %%
%%time
pol_eval_kwargs = {'tol': 1e-3, 'max_iter': 1000, 'memo_step': 10, 'plt_trace': True, 'trace_ij': (10, 10)}
arr_pol_val_iter, arr_val_val_iter = car_rental_pol_iter(
    car_rental_trans1, 0.9, max_pol_iter=100, plt_iter=True, 
    arr_pol_ini=arr_pol_val_iter, arr_val_ini=arr_val_val_iter,
    **pol_eval_kwargs
)

# %%


# %% [markdown]
# ## 4.3. Add more details
# <a id="4.3.-add-more-details"></a>

# %%
PARKING_THRE = 10
EXTRA_PARKING_COST = 4

def car_rental_trans2(c1, c2, mv, c1n, c2n):
    """
    Calculate the transition probability and expected reward for the car rental problem from 
    state (c1, c2) to state (c1n, c2n) by moving mv cars from location 1 to location 2.
    
    c1 is the number of cars at location 1.
    c2 is the number of cars at location 2.
    mv is the number of cars moved from location 1 to location 2.
    c1n is the number of cars at location 1 at the end of the next day.
    c2n is the number of cars at location 2 at the end of the next day.
    """
    if ((mv>0 and c1<mv) or (mv<0 and c2<-mv)):
        return 0.0, 0.0
    c1_be = min(MAX_CARS, c1-mv)
    c2_be = min(MAX_CARS, c2+mv)
    
    r_sa_sp1, p_sa_sp1 = car_rental_one_loc_trans(c1_be, c1n, loc=1)
    r_sa_sp2, p_sa_sp2 = car_rental_one_loc_trans(c2_be, c2n, loc=2)
    p_sa_sp = p_sa_sp1*p_sa_sp2
    # Be careful with the probability of the cost
    cost = MV_COST*abs(mv-int(mv>0))+EXTRA_PARKING_COST*(int(c1n>PARKING_THRE)+int(c2n>PARKING_THRE))
    r_sa_sp = r_sa_sp1*p_sa_sp2+r_sa_sp2*p_sa_sp1-cost*p_sa_sp
    return r_sa_sp, p_sa_sp

# %%
%%time
pol_eval_kwargs = {'tol': 1e-3, 'max_iter': 1000, 'memo_step': 10, 'plt_trace': True, 'trace_ij': (10, 10)}
_ = car_rental_pol_iter(car_rental_trans2, 0.9, max_pol_iter=5, plt_iter=True, **pol_eval_kwargs)

# %% [markdown]
# # 5. Example 4.8
# <a id="5.-example-4.8"></a>
# - All solutions are close to each others.

# %%
import time

import matplotlib.cm as cm
MPL_COLORS = cm.tab20(range(20))

# print(MPL_COLORS)

def gen_plotly_rgba(arr_rgba):
    arr_rgb_rd = np.round(arr_rgba[:3]*255).astype(int)
    return f'rgba({arr_rgb_rd[0]}, {arr_rgb_rd[1]}, {arr_rgb_rd[2]}, {arr_rgba[3]})'

# %% [markdown]
# ## 5.1. Ex 4.9
# <a id="5.1.-ex-4.9"></a>

# %%
def gambler_val_iter(arr_val, ph, tol=1e-4, inplace=False, plt=False, c_iters=None):
    n = len(arr_val)
    goal = n - 1
    arr_act = np.zeros_like(arr_val)
    arr_val = arr_val.copy()
    c_iter = 0
    m_diff = -1
    fig = make_subplots(rows=2, cols=1)
    while True:
        c_iter += 1
        if inplace:
            arr_valn = arr_val
        else:
            arr_valn = arr_val.copy()
        if c_iter%100 == 1:
            print("="*10+str(c_iter)+":"+f"{m_diff}")
        m_diff = 0
        for c in range(1, n-1):
            m_act, m_val = arr_act[c], arr_valn[c]
            for b in range(min(c, goal-c)+1):
                val = (1-ph)*arr_val[c-b]+ph*arr_val[c+b]
                # if val>m_val: # Can also be val>=m_val, which gives a different final policy
                if val>m_val or (val==m_val and 0<b<m_act):
                    m_act, m_val = b, val
            m_diff = max(m_diff, abs(arr_val[c]-m_val))
            arr_act[c], arr_valn[c] = m_act, m_val
        arr_val = arr_valn
        
        if c_iter in c_iters:
            fig.add_trace(
                go.Scatter(x=np.arange(1, n), y=arr_val[1:n], mode='lines',
                           line=dict(color=gen_plotly_rgba(MPL_COLORS[c_iter%len(MPL_COLORS)]), shape='hv'),
                           name=f'{c_iter}', showlegend=True), 
                row=1, col=1
            )
            fig.add_trace(
                go.Scatter(x=np.arange(1, n), y=arr_act[1:n], mode='lines',
                           line=dict(color=gen_plotly_rgba(MPL_COLORS[c_iter%len(MPL_COLORS)]), shape='hv'),
                           name=f'{c_iter}', showlegend=False), 
                row=2, col=1
            )
            
        if m_diff<tol:
            print(f"The value iteration ends at iter {c_iter} with m_dff {m_diff:.8f}")
            break
        
    if plt:
        fig.update_layout(
            xaxis=dict(title='Capital'),
            yaxis=dict(title='Value Function'),
            yaxis2=dict(title='Policy'),
            height=800, width=800
        )
        fig.show()
    
    return arr_act, arr_valn

# Chatgpt solution
def gambler_val_iter_1(arr_val, ph, tol=1e-4, c_iters=None):
    nr = len(arr_val)
    arr_valn = arr_val.copy()
    arr_act = np.zeros_like(arr_val)
    m_diff, avg_diff, tic = 0, 0, time.time()
    abs_tol = tol
    fig = make_subplots(rows=2, cols=1)
    for si in range(1000):
        if si%10==0:
            print(f'Value iteration {si}: {m_diff:.2f}:{avg_diff:.2f} ({time.time()-tic:.2f}s)')
        m_diff, avg_diff, tic = 0, 0, time.time()
        for i in range(1, nr-1):
            act, q_max = 0, -np.inf
            for a in range(min(i, nr-1-i)+1):
                q = ph*arr_valn[i+a]+(1-ph)*arr_valn[i-a]
                if q>q_max:
                    act, q_max = a, q
            m_diff = max(m_diff, abs(q_max-arr_valn[i]))
            avg_diff += abs(q_max-arr_valn[i])
            arr_act[i], arr_valn[i] = act, q_max
            
        if si in c_iters:
            fig.add_trace(
                go.Scatter(x=np.arange(1, nr), y=arr_valn[1:nr], mode='lines',
                           line=dict(color=gen_plotly_rgba(MPL_COLORS[si%len(MPL_COLORS)]), shape='hv'),
                           name=f'{si}', showlegend=True), 
                row=1, col=1
            )
            fig.add_trace(
                go.Scatter(x=np.arange(1, nr), y=arr_act[1:nr], mode='lines',
                           line=dict(color=gen_plotly_rgba(MPL_COLORS[si%len(MPL_COLORS)]), shape='hv'),
                           name=f'{si}', showlegend=False), 
                row=2, col=1
            )
            
        if m_diff<abs_tol:
            print(f'Converged at iteration {si}')
            break
        
    fig.update_layout(
        xaxis=dict(title='Capital'),
        yaxis=dict(title='Value Function'),
        yaxis2=dict(title='Policy'),
        height=800, width=800
    )
    fig.show()
    
    return arr_act, arr_valn
    

# %%
goal = 100
arr_val_ini = np.zeros(goal+1)
arr_val_ini[-1] = 1
ph, tol = .4, 1e-7
c_iters = [1, 2, 3, 5, 10, 20, 99, 100, 101]
arr_act_1, arr_val_1 = gambler_val_iter(arr_val_ini, ph, tol, plt=True, c_iters=c_iters)

# %%
arr_act_2, arr_val_2 = gambler_val_iter(arr_val_ini, ph, tol=1e-12, inplace=True, plt=True, c_iters=c_iters)

# %%
arr_act_3, arr_val_3 = gambler_val_iter_1(arr_val_ini, ph, tol=1e-8, c_iters=c_iters)

# %% [markdown]
# ## 5.2. Optimal sol in the book
# <a id="5.2.-optimal-sol-in-the-book"></a>

# %%
ini_arr = np.arange(1, 13)
ini_arr

# %%
def mirror_arr(arr, last_val=None):
    if last_val is None:
        return np.concatenate((arr, arr[::-1]))
    else:
        return np.concatenate((arr, [last_val], arr[::-1]))

def plt_arr(arr, max_x, max_y, si=1):
    fig = go.Figure(
        data=go.Scatter(x=np.arange(si, len(arr)+si), y=arr, mode='lines', line=dict(shape='hv'))
    )
    fig.update_layout(
        xaxis=dict(range=[0, max_x]),
        yaxis=dict(range=[0, max_y]),
    )
    fig.show()

# %%
max_x, max_y = 100, 55
arr1 = mirror_arr(ini_arr, None)
print(f"len(arr1): {len(arr1)}")
arr2 = mirror_arr(arr1, 25)
print(f"len(arr2): {len(arr2)}")
arr3 = mirror_arr(arr2, 50)
print(f"len(arr3): {len(arr3)}")
arr3 = np.concatenate(([0], arr3, [0]))
plt_arr(arr3, max_x, max_y, si=0)

# %%
OPT_ARR_ACT = arr3

def gambler_pol_eval(arr_val, ph, tol=1e-4, inplace=False, plt=False, c_iters=None):
    n = len(arr_val)
    goal = n - 1
    # arr_act = np.zeros_like(arr_val)
    arr_act = OPT_ARR_ACT.copy()
    arr_val = arr_val.copy()
    c_iter = 0
    m_diff = -1
    fig = make_subplots(rows=2, cols=1)
    while True:
        c_iter += 1
        if inplace:
            arr_valn = arr_val
        else:
            arr_valn = arr_val.copy()
        if c_iter%100 == 1:
            print("="*10+str(c_iter)+":"+f"{m_diff}")
        m_diff = 0
        for c in range(1, n-1):
            m_act, m_val = arr_act[c], arr_valn[c]
            for b in range(min(c, goal-c)+1):
                # val = (1-ph)*arr_val[c-b]+ph*arr_val[c+b]
                # if val>m_val or (val==m_val and b!=OPT_ARR_ACT[c]):
                #     m_act, m_val = b, val
                m_val = (1-ph)*arr_val[c-b]+ph*arr_val[c+b]
            m_diff = max(m_diff, abs(arr_val[c]-m_val))
            arr_act[c], arr_valn[c] = m_act, m_val
        arr_val = arr_valn
        
        if c_iter in c_iters:
            fig.add_trace(
                go.Scatter(x=np.arange(1, n), y=arr_val[1:n], mode='lines',
                           line=dict(color=gen_plotly_rgba(MPL_COLORS[c_iter%len(MPL_COLORS)]), shape='hv'),
                           name=f'{c_iter}', showlegend=True), 
                row=1, col=1
            )
            fig.add_trace(
                go.Scatter(x=np.arange(1, n), y=arr_act[1:n], mode='lines',
                           line=dict(color=gen_plotly_rgba(MPL_COLORS[c_iter%len(MPL_COLORS)]), shape='hv'),
                           name=f'{c_iter}', showlegend=False), 
                row=2, col=1
            )
            
        if m_diff<tol:
            print(f"The value iteration ends at iter {c_iter} with m_dff {m_diff:.10f}")
            break
        
    if plt:
        fig.update_layout(
            xaxis=dict(title='Capital'),
            yaxis=dict(title='Value Function'),
            yaxis2=dict(title='Policy'),
            height=800, width=800
        )
        fig.show()
    
    return arr_act, arr_valn

# %%
goal = 100
arr_val_ini = np.zeros(goal+1)
arr_val_ini[-1] = 1
ph, tol = .4, 1e-9
c_iters = [1, 2, 3, 5, 10, 20, 30, 99, 100, 101]
arr_act, arr_val = gambler_pol_eval(arr_val_ini, ph, tol, plt=True, c_iters=c_iters)

# %%
(np.max(np.abs(arr_val_1-arr_val_2)), np.max(np.abs(arr_val_1-arr_val_3)), np.max(np.abs(arr_val_2-arr_val_3)))

# %% [markdown]
# # 6. Ex 4.10
# <a id="6.-ex-4.10"></a>
# 
# \begin{align*}
# q_{k+1}(s, a) = & \mathbb{E}\left[ \left. R_{t+1}+\gamma \max_{a'} q_{k}(S_{t+1}, a') \right| S_t=s, A_t=a \right] \\
# = & \sum_{s', r}p(s',r | s, a)\left[ r+\gamma \max_{a'} q_{k}(s', a') \right]
# \end{align*}

# %%


# %% [markdown]
# # 7. End
# <a id="7.-end"></a>

# %%


# %%



