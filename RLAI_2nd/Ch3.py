#!/usr/bin/env python
# coding: utf-8

# <h1>Table of Contents<span class="tocSkip"></span></h1><ul>
# <li style="margin-left: 0px;"><a href="#import-dep">1. Import dep</a></li>
# <li style="margin-left: 0px;"><a href="#example-3.5">2. Example 3.5</a></li>
# <li style="margin-left: 20px;"><a href="#solve-the-linear-equation-system">2.1. Solve the linear equation system</a></li>
# <li style="margin-left: 20px;"><a href="#iterative-evaluation">2.2. Iterative evaluation</a></li>
# <li style="margin-left: 0px;"><a href="#ex-3.17">3. Ex 3.17</a></li>
# <li style="margin-left: 0px;"><a href="#example-3.8">4. Example 3.8</a></li>
# <li style="margin-left: 0px;"><a href="#end">5. End</a></li>
# </ul>

# In[ ]:


# <h1>Table of Contents<span class="tocSkip"></span></h1>
# <div class="toc"><ul class="toc-item"><li><span><a href="#Import-dep" data-toc-modified-id="Import-dep-1"><span class="toc-item-num">1&nbsp;&nbsp;</span>Import dep</a></span></li><li><span><a href="#Example-3.5" data-toc-modified-id="Example-3.5-2"><span class="toc-item-num">2&nbsp;&nbsp;</span>Example 3.5</a></span><ul class="toc-item"><li><span><a href="#Solve-the-linear-equation-system" data-toc-modified-id="Solve-the-linear-equation-system-2.1"><span class="toc-item-num">2.1&nbsp;&nbsp;</span>Solve the linear equation system</a></span></li><li><span><a href="#Iterative-evaluation" data-toc-modified-id="Iterative-evaluation-2.2"><span class="toc-item-num">2.2&nbsp;&nbsp;</span>Iterative evaluation</a></span></li></ul></li><li><span><a href="#Ex-3.17" data-toc-modified-id="Ex-3.17-3"><span class="toc-item-num">3&nbsp;&nbsp;</span>Ex 3.17</a></span></li><li><span><a href="#Example-3.8" data-toc-modified-id="Example-3.8-4"><span class="toc-item-num">4&nbsp;&nbsp;</span>Example 3.8</a></span></li><li><span><a href="#End" data-toc-modified-id="End-5"><span class="toc-item-num">5&nbsp;&nbsp;</span>End</a></span></li></ul></div>


# In[1]:


from IPython.display import display, HTML

# Set the notebook width to 80%
display(HTML("<style>.container { width: 80% !important; }</style>"))


# In[2]:


get_ipython().system('jupyter notebook list')


# In[3]:


# Needs to paste `http://localhost:3110`, no ending `/`
port = 2550

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


# # 1. Import dep<a id="import-dep"></a>

# In[4]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[5]:


import plotly
import plotly.express as px
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import plotly.io as pio

pio.renderers.default = "notebook"


# # 2. Example 3.5<a id="example-3.5"></a>
# - Solve a linear equation system for a certain simple policy.

# ## 2.1. Solve the linear equation system<a id="solve-the-linear-equation-system"></a>

# In[6]:


from sympy import latex, symbols, Eq, solve, linear_eq_to_matrix

# Define the variables
x = symbols('x[0:3]')

aa = 3

def _idx(i):
    return i

print(x)

# Define the equations
equations = [
    Eq(2*x[_idx(0)] + aa*x[1] - x[2], 1),
    Eq(3*x[0] - 2*x[1] + 2*x[2], -2),
    Eq(x[0] + x[1] + x[2], 3)
]

for equation in equations:
    print(latex(equation))

solution = solve(equations, x)

print(solution)


# In[7]:


ga = 0.9

v = symbols('v[0:25]')

def ij(i, j):
    return i*5 + j

def cor_eq(i, j, r, c):
    # corner equation
    return Eq(v[ij(i, j)]-1/4*((-1+ga*v[ij(i, j)])*2+ga*(v[ij(i+r, j)]+v[ij(i, j+c)])), 0)

def bd_eq(i, j, r, c):
    # boundary equation
    if r!=0:
        return Eq(v[ij(i, j)]-1/4*((-1+ga*v[ij(i, j)])+ga*(v[ij(i+r, j)]+v[ij(i, j-1)]+v[ij(i, j+1)])), 0)
    else: # c!=0
        return Eq(v[ij(i, j)]-1/4*((-1+ga*v[ij(i, j)])+ga*(v[ij(i, j+c)]+v[ij(i-1, j)]+v[ij(i+1, j)])), 0)
    
def in_eq(i,j):
    # inner equation
    return Eq(v[ij(i, j)]-1/4*ga*(v[ij(i-1, j)]+v[ij(i+1, j)]+v[ij(i, j-1)]+v[ij(i, j+1)]), 0)

equations = [
    # row 0
    cor_eq(0, 0, 1, 1),
    Eq(v[ij(0, 1)]-(10+ga*v[ij(4, 1)]), 0), # special state A
    bd_eq(0, 2, 1, 0),
    Eq(v[ij(0, 3)]-(5+ga*v[ij(2, 3)]), 0), # special state B
    cor_eq(0, 4, 1, -1),
    # row 1
    bd_eq(1, 0, 0, 1), in_eq(1, 1), in_eq(1, 2), in_eq(1, 3), bd_eq(1, 4, 0, -1),
    # row 2
    bd_eq(2, 0, 0, 1), in_eq(2, 1), in_eq(2, 2), in_eq(2, 3), bd_eq(2, 4, 0, -1),
    # row 3
    bd_eq(3, 0, 0, 1), in_eq(3, 1), in_eq(3, 2), in_eq(3, 3), bd_eq(3, 4, 0, -1),
    # row 4
    cor_eq(4, 0, -1, 1), bd_eq(4, 1, -1, 0), bd_eq(4, 2, -1, 0), bd_eq(4, 3, -1, 0), cor_eq(4, 4, -1, -1)
]

for eq in equations:
    print(latex(eq))
    
A, b = linear_eq_to_matrix(equations, v)

display(A, b)
    
# # Try to solve the equations symbolically just breaks the jupyter kernel
# sol = solve(equations, v)

# v_array = np.zeros((5, 5))
# for i in range(5):
#     for j in range(5):
#         v_array[i, j] = sol[v[ij(i, j)]]

# print(v_array)

# for i in range(5):
#     for j in range(5):
#         print(f'v[{i},{j}] = {sol[v[ij(i, j)]]}')


# In[8]:


# I checked with the solution in the book, the solution is correct
arr_A = np.array(A, dtype='float')
arr_b = np.array(b, dtype='float')

sol = np.linalg.solve(arr_A, arr_b)

print(np.array2string(sol.reshape(5, 5, order='C'), formatter={'float_kind':lambda x: "%8.3f" % x}))


# ## 2.2. Iterative evaluation<a id="iterative-evaluation"></a>
# - This is a content for Ch 4.1

# In[9]:


from itertools import product

def eval_schema1(v0: np.ndarray, ga: float, max_iter: int, memo_step: int = 1):
    """
    Each time step, update all the value function, and then go to the next step. 
    When updating the value function at n+1, only the value function at step n is used.
    """
    nr, nc = v0.shape
    vn = v0.copy()
    memo_v = {}
    for si in range(max_iter):
        vnn = vn.copy()
        for i, j in product(range(nr), range(nc)):
            if i==0 and j==1:
                vnn[i, j] = 10 + ga*vn[4, 1]
            elif i==0 and j==3:
                vnn[i, j] = 5 + ga*vn[2, 3]
            else:
                # Corners
                if i==0 and j==0:
                    vnn[i, j] = sum([-1+ga*vn[i, j], ga*vn[i, j+1], -1+ga*vn[i, j], ga*vn[i+1, j]])/4
                elif i==0 and j==4:
                    vnn[i, j] = sum([ga*vn[i, j-1], -1+ga*vn[i, j], -1+ga*vn[i, j], ga*vn[i+1, j]])/4
                elif i==4 and j==0:
                    vnn[i, j] = sum([-1+ga*vn[i, j], ga*vn[i, j+1], ga*vn[i-1, j], -1+ga*vn[i, j]])/4
                elif i==4 and j==4:
                    vnn[i, j] = sum([ga*vn[i, j-1], -1+ga*vn[i, j], ga*vn[i-1, j], -1+ga*vn[i, j]])/4
                # Boundaries
                elif i==0:
                    vnn[i, j] = sum([ga*vn[i, j-1], ga*vn[i, j+1], -1+ga*vn[i, j], ga*vn[i+1, j]])/4
                elif i==4:
                    vnn[i, j] = sum([ga*vn[i, j-1], ga*vn[i, j+1], ga*vn[i-1, j], -1+ga*vn[i, j]])/4
                elif j==0:
                    vnn[i, j] = sum([-1+ga*vn[i, j], ga*vn[i, j+1], ga*vn[i-1, j], ga*vn[i+1, j]])/4
                elif j==4:
                    vnn[i, j] = sum([ga*vn[i, j-1], -1+ga*vn[i, j], ga*vn[i-1, j], ga*vn[i+1, j]])/4
                # Inner
                else:
                    vnn[i, j] = sum([ga*vn[i, j-1], ga*vn[i, j+1], ga*vn[i-1, j], ga*vn[i+1, j]])/4

        if si%memo_step==0:
            memo_v[si] = vn
        
        vn = vnn
    
    memo_v[max_iter] = vn
    
    return memo_v


def eval_schema2(v0: np.ndarray, ga: float, max_iter: int, memo_step: int = 1):
    """
    When updating the value function at n+1, newly calcualted n+1 values are also used.
    RLAI book call this "in-place" algorithm.
    """
    nr, nc = v0.shape
    vn = v0.copy()
    memo_v = {}
    for si in range(max_iter):
        vnn = vn.copy()
        # Using the vnn values below to use the newly obtained v_{n+1} values.
        for i, j in product(range(nr), range(nc)):
            if i==0 and j==1:
                vnn[i, j] = 10 + ga*vnn[4, 1]
            elif i==0 and j==3:
                vnn[i, j] = 5 + ga*vnn[2, 3]
            else:
                # Corners
                if i==0 and j==0:
                    vnn[i, j] = sum([-1+ga*vnn[i, j], ga*vnn[i, j+1], -1+ga*vnn[i, j], ga*vnn[i+1, j]])/4
                elif i==0 and j==4:
                    vnn[i, j] = sum([ga*vnn[i, j-1], -1+ga*vnn[i, j], -1+ga*vnn[i, j], ga*vnn[i+1, j]])/4
                elif i==4 and j==0:
                    vnn[i, j] = sum([-1+ga*vnn[i, j], ga*vnn[i, j+1], ga*vnn[i-1, j], -1+ga*vnn[i, j]])/4
                elif i==4 and j==4:
                    vnn[i, j] = sum([ga*vnn[i, j-1], -1+ga*vnn[i, j], ga*vnn[i-1, j], -1+ga*vnn[i, j]])/4
                # Boundaries
                elif i==0:
                    vnn[i, j] = sum([ga*vnn[i, j-1], ga*vnn[i, j+1], -1+ga*vnn[i, j], ga*vnn[i+1, j]])/4
                elif i==4:
                    vnn[i, j] = sum([ga*vnn[i, j-1], ga*vnn[i, j+1], ga*vnn[i-1, j], -1+ga*vnn[i, j]])/4
                elif j==0:
                    vnn[i, j] = sum([-1+ga*vnn[i, j], ga*vnn[i, j+1], ga*vnn[i-1, j], ga*vnn[i+1, j]])/4
                elif j==4:
                    vnn[i, j] = sum([ga*vnn[i, j-1], -1+ga*vnn[i, j], ga*vnn[i-1, j], ga*vnn[i+1, j]])/4
                # Inner
                else:
                    vnn[i, j] = sum([ga*vnn[i, j-1], ga*vnn[i, j+1], ga*vnn[i-1, j], ga*vnn[i+1, j]])/4
            
        if si%memo_step==0:
            memo_v[si] = vn

        vn = vnn
    
    memo_v[max_iter] = vn
    
    return memo_v

def pr_val_func(val: np.ndarray, fmt='%8.3f', reshape=False, nr=None, nc=None):
    if reshape:
        print(np.array2string(val.reshape(nr, nc, order='C'), formatter={'float_kind':lambda x: fmt % x}))
    else:
        print(np.array2string(val, formatter={'float_kind':lambda x: fmt % x}))
        


# In[10]:


v0 = np.zeros((5, 5))
ga, alp = 0.9, 0.5
max_iter = 1000
memo_step = 10
memo1_v = eval_schema1(v0, ga, max_iter, memo_step)
memo2_v = eval_schema2(v0, ga, max_iter, memo_step)


# In[11]:


pr_val_func(sol, reshape=True, nr=5, nc=5)
pr_val_func(memo1_v[max_iter])
pr_val_func(memo2_v[max_iter])
display(
    np.allclose(sol, memo1_v[max_iter].reshape((-1, 1), order='C')),
    np.allclose(sol, memo2_v[max_iter].reshape((-1, 1), order='C')),
)


# In[12]:


arr_si = []
v_0_0_schema1 = []
v_0_0_schema2 = []

memo1_v = eval_schema1(v0, ga, max_iter, memo_step)
memo2_v = eval_schema2(v0, ga, max_iter, memo_step)
if len(arr_si)==0:
    arr_si = np.array(sorted(list(memo1_v.keys())))
v_0_0_schema1.append([memo1_v[si][0, 0] for si in arr_si])
v_0_0_schema2.append([memo2_v[si][0, 0] for si in arr_si])

arr_si += 1


# In[13]:


colors = px.colors.qualitative.Plotly

# Create a trace for each iteration
traces = []
for i, (v1, v2) in enumerate(zip(v_0_0_schema1, v_0_0_schema2)):
    trace1 = go.Scatter(
        x=arr_si,
        y=v1,
        mode='lines',
        line=dict(color=colors[i], dash='dash'),
        name=f'opt_schema1'
    )
    trace2 = go.Scatter(
        x=arr_si,
        y=v2,
        mode='lines',
        line=dict(color=colors[i]),
        name=f'opt_schema2'
    )
    traces.extend([trace1, trace2])

# Create the layout
layout = go.Layout(
    title='Value Function (0, 0) vs Iteration',
    xaxis=dict(title='Iteration', type='log'),
    yaxis=dict(title='Value Function (0, 0)')
)

# Create the figure
fig = go.Figure(data=traces, layout=layout)

# Show the figure
fig.show()


# In[ ]:





# # 3. Ex 3.17<a id="ex-3.17"></a>
# 
# \begin{align*}
# q_{\pi}(s, a) =& \sum_{s', r}p(s', r | s, a)(r+\gamma*v_{\pi}(s')) \\
# = & \sum_{s', r}p(s', r | s, a)(r+\gamma*\sum_{a'}\pi(a'|s')q_{\pi}(s', a')) \\
# \end{align*}

# # 4. Example 3.8<a id="example-3.8"></a>
# - A naive iterative method to solve the Bellman equation for Example 3.5:
#     - Given the solution from the Example 3.5 as the initial values, meaning the value of $v_0(s)$.
#     - At the step $n$ find the best value for the action-value function $\tilde{v}(s)=\max_{a}q_{\pi_n}(s, a)$.
#     - Use the following updating schema with $\alpha$ as a parameter
# 
# \begin{align*}
# v_{n+1}(s) = v_{n}(s)+\alpha(\tilde{v}(s)-v_{n}(s))
# \end{align*}
# - The "in-place" algorithm converges faster.
# 
# 

# In[14]:


from itertools import product

def ij(i, j):
    return i*5 + j

def opt_schema1(v0: np.ndarray, ga: float, alp: float, max_iter: int, memo_step: int = 1):
    """
    Each time step, update all the value function, and then go to the next step. 
    When updating the value function at n+1, only the value function at step n is used.
    """
    nr, nc = v0.shape
    vn = v0.copy()
    memo_v = {}
    memo_q = {}
    for si in range(max_iter):
        qn = np.zeros((vn.shape[0], vn.shape[1], 4))
        vnn = vn.copy()
        for i, j in product(range(nr), range(nc)):
            if i==0 and j==1:
                qn[i, j, :] = 10 + ga*vn[4, 1]
            elif i==0 and j==3:
                qn[i, j, :] = 5 + ga*vn[2, 3]
            else:
                # Corners
                if i==0 and j==0:
                    qn[i, j, :] = [-1+ga*vn[i, j], ga*vn[i, j+1], -1+ga*vn[i, j], ga*vn[i+1, j]]
                elif i==0 and j==4:
                    qn[i, j, :] = [ga*vn[i, j-1], -1+ga*vn[i, j], -1+ga*vn[i, j], ga*vn[i+1, j]]
                elif i==4 and j==0:
                    qn[i, j, :] = [-1+ga*vn[i, j], ga*vn[i, j+1], ga*vn[i-1, j], -1+ga*vn[i, j]]
                elif i==4 and j==4:
                    qn[i, j, :] = [ga*vn[i, j-1], -1+ga*vn[i, j], ga*vn[i-1, j], -1+ga*vn[i, j]]
                # Boundaries
                elif i==0:
                    qn[i, j, :] = [ga*vn[i, j-1], ga*vn[i, j+1], -1+ga*vn[i, j], ga*vn[i+1, j]]
                elif i==4:
                    qn[i, j, :] = [ga*vn[i, j-1], ga*vn[i, j+1], ga*vn[i-1, j], -1+ga*vn[i, j]]
                elif j==0:
                    qn[i, j, :] = [-1+ga*vn[i, j], ga*vn[i, j+1], ga*vn[i-1, j], ga*vn[i+1, j]]
                elif j==4:
                    qn[i, j, :] = [ga*vn[i, j-1], -1+ga*vn[i, j], ga*vn[i-1, j], ga*vn[i+1, j]]
                # Inner
                else:
                    qn[i, j, :] = [ga*vn[i, j-1], ga*vn[i, j+1], ga*vn[i-1, j], ga*vn[i+1, j]]
            
            vnn[i, j] += alp*(max(qn[i, j, :]) - vnn[i, j])
        if si%memo_step==0:
            memo_v[si] = vn
            memo_q[si] = qn
        vn = vnn
    memo_q[max_iter-1] = qn
    memo_v[max_iter] = vn
    
    return memo_v, memo_q

def opt_schema2(v0: np.ndarray, ga: float, alp: float, max_iter: int, memo_step: int = 1):
    """
    When updating the value function at n+1, newly calcualted n+1 values are also used.
    RLAI book call this "in-place" algorithm.
    """
    nr, nc = v0.shape
    vn = v0.copy()
    memo_v = {}
    memo_q = {}
    for si in range(max_iter):
        qn = np.zeros((vn.shape[0], vn.shape[1], 4))
        vnn = vn.copy()
        # Using the vnn values below to use the newly obtained v_{n+1} values.
        for i, j in product(range(nr), range(nc)):
            if i==0 and j==1:
                qn[i, j, :] = 10 + ga*vnn[4, 1]
            elif i==0 and j==3:
                qn[i, j, :] = 5 + ga*vnn[2, 3]
            else:
                # Corners
                if i==0 and j==0:
                    qn[i, j, :] = [-1+ga*vnn[i, j], ga*vnn[i, j+1], -1+ga*vnn[i, j], ga*vnn[i+1, j]]
                elif i==0 and j==4:
                    qn[i, j, :] = [ga*vnn[i, j-1], -1+ga*vnn[i, j], -1+ga*vnn[i, j], ga*vnn[i+1, j]]
                elif i==4 and j==0:
                    qn[i, j, :] = [-1+ga*vnn[i, j], ga*vnn[i, j+1], ga*vnn[i-1, j], -1+ga*vnn[i, j]]
                elif i==4 and j==4:
                    qn[i, j, :] = [ga*vnn[i, j-1], -1+ga*vnn[i, j], ga*vnn[i-1, j], -1+ga*vnn[i, j]]
                # Boundaries
                elif i==0:
                    qn[i, j, :] = [ga*vnn[i, j-1], ga*vnn[i, j+1], -1+ga*vnn[i, j], ga*vnn[i+1, j]]
                elif i==4:
                    qn[i, j, :] = [ga*vnn[i, j-1], ga*vnn[i, j+1], ga*vnn[i-1, j], -1+ga*vnn[i, j]]
                elif j==0:
                    qn[i, j, :] = [-1+ga*vnn[i, j], ga*vnn[i, j+1], ga*vnn[i-1, j], ga*vnn[i+1, j]]
                elif j==4:
                    qn[i, j, :] = [ga*vnn[i, j-1], -1+ga*vnn[i, j], ga*vnn[i-1, j], ga*vnn[i+1, j]]
                # Inner
                else:
                    qn[i, j, :] = [ga*vnn[i, j-1], ga*vnn[i, j+1], ga*vnn[i-1, j], ga*vnn[i+1, j]]
            
            vnn[i, j] += alp*(max(qn[i, j, :]) - vnn[i, j])
        if si%memo_step==0:
            memo_v[si] = vn
            memo_q[si] = qn
        vn = vnn
    memo_q[max_iter-1] = qn
    memo_v[max_iter] = vn
    
    return memo_v, memo_q


# In[15]:


v0 = sol.reshape(5, 5, order='C')
ga, alp = 0.9, 0.5
max_iter = 1000
memo_step = 10
memo1_v, memo1_q = opt_schema1(v0, ga, alp, max_iter, memo_step)
memo2_v, memo2_q = opt_schema2(v0, ga, alp, max_iter, memo_step)


# In[16]:


pr_val_func(memo1_v[max_iter])
pr_val_func(memo2_v[max_iter])


# In[17]:


arr_si = []
v_0_0_schema1 = []
v_0_0_schema2 = []
ls_alp = [0.1, 0.5, 0.9, 1.0, 1.1]
for alp in ls_alp:
    memo1_v, memo1_q = opt_schema1(v0, ga, alp, max_iter, memo_step)
    memo2_v, memo2_q = opt_schema2(v0, ga, alp, max_iter, memo_step)
    if len(arr_si)==0:
        arr_si = np.array(sorted(list(memo1_v.keys())))
    v_0_0_schema1.append([memo1_v[si][0, 0] for si in arr_si])
    v_0_0_schema2.append([memo2_v[si][0, 0] for si in arr_si])
arr_si += 1


# In[18]:


colors = px.colors.qualitative.Plotly

# Create a trace for each iteration
traces = []
for i, (v1, v2) in enumerate(zip(v_0_0_schema1, v_0_0_schema2)):
    trace1 = go.Scatter(
        x=arr_si,
        y=v1,
        mode='lines',
        line=dict(color=colors[i], dash='dash'),
        name=f'opt_schema1: alpha {ls_alp[i]}'
    )
    trace2 = go.Scatter(
        x=arr_si,
        y=v2,
        mode='lines',
        line=dict(color=colors[i]),
        name=f'opt_schema2: alpha {ls_alp[i]}'
    )
    traces.extend([trace1, trace2])

# Create the layout
layout = go.Layout(
    title='Value Function (0, 0) vs Iteration',
    xaxis=dict(title='Iteration', type='log'),
    yaxis=dict(title='Value Function (0, 0)')
)

# Create the figure
fig = go.Figure(data=traces, layout=layout)

# Show the figure
fig.show()


# In[ ]:





# # 5. End<a id="end"></a>

# In[ ]:




