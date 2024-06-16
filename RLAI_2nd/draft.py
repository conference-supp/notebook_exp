def state_val_update2(vnn):
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
                
    vnn
    
    
    \[(i[+\-0-9]*), (j[+\-0-9]*)\]