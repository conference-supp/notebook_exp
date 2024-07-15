# %%
import numpy as np
import pandas as pd

# %%
arr = np.arange(12).reshape(2, 3, 2)
arr

# %%
arr.reshape(
    -1,
)

# %%
arr[0].size
# %%
arr[0, 0].size
# %%
arr[0, 0, 0].size
# %%
