# mixed_KSG

This is the Python Code for the paper "Estimating Mutual Information for Discrete-Continuous Mixtures", NIPS 2017.

Reference: http://arxiv.org/abs/1709.06212

Sample Usage:

    import mixed 
    MI = mixed.mixed_KSG(x,y)
    
Input

    X -- 2D array of size N by d_x (1D array of size N if d_x = 1)
    
    Y -- 2D array of size N by d_y (1D array of size N if d_y = 1)
    
Output

    An estimate of I(X;Y)
    
See demo.py for an example (Experiment I in Section 5 of the paper)

Contact wgao9@illinois.edu
