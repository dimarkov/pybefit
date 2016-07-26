# -*- coding: utf-8 -*-
from numpy import ones
import nlopt as no
import cma

def isres(f, n, ftol, xtol, bounds, x_init, max_time = 180):
    """Find global maxima of an objective function using isres algorithm.
    
    Args:
        f (scalar function): Objective function
        n (int): Number of parameters
        ftol (double): Tolerance for the change of f function
        xtol (double): Tolerance for the change of paramters
        bound (dict): Absolute value of lower and upper bounds
        x (array_like): Initial parameter value
        max_time (optional): Maximal duration of the optimization
        
    Returns:
        f_opt (double): Function value at the optimum
        x_opt (array_like): Parameter value at the optimum
        res (str): Termination criterion 
    """
    
    #initial the optimizer to GN_ISRES method
    opt = no.opt(no.GN_ISRES, n)
    
    #set bounds
    opt.set_lower_bounds(bounds['lb'])
    opt.set_upper_bounds(bounds['ub'])
    
    #set tolerance
    opt.set_ftol_abs(ftol)
    opt.set_xtol_abs(xtol)
    
    #set objective function to be minimazed
    opt.set_max_objective(f)
    
    #set maximal runtime
    opt.set_maxtime(max_time)
    
    #start optimization
    x_opt = opt.optimize(x_init)
    
    #collect results
    f_opt = opt.last_optimum_value()
    res = opt.last_optimize_result()    
    
    return f_opt, x_opt, res
    

def cmaes(f, n, ftol, xtol, bounds, x):
    """Find global maxima of an objective function using CMA-ES algorithm.
    
    Args:
        f (scalar function): Objective function
        n (int): Number of parameters
        ftol (double): Tolerance for the change of f function
        xtol (double): Tolerance for the change of paramters
        bound (dict): Absolute value of lower and upper bounds
        x (array_like): Initial parameter value
        
    Returns:
        f_opt (double): Function value at the optimum
        x_opt (array_like): Parameter value at the optimum
        res (str): Termination criterion 
    
    """
    
    #set options of the cma-es optimizer
    opts = cma.CMAOptions()
    opts['popsize'] = n*10
    opts['CMA_active'] = False
    opts['tolfun'] = ftol
    opts['tolx'] = xtol    
    opts['verb_disp'] = 0
    opts['verb_log'] = 0
    opts['verbose'] = 0
    if(len(bounds['lb']) == n and len(bounds['ub']) == n):
        opts['bounds'] = [bounds['lb'], bounds['ub']]
    elif type(bounds['lb']) == int or type(bounds['lb']) == float:
        opts['bounds'] = [[bounds['lb']]*n, [bounds['ub']]*n]
    else:
        print('inconsistent definition of parameter bounds')
    
    opts['CMA_stds'] = ones(n)*(bounds['ub'] - bounds['lb'])/4    
    cma_sigma = 1;
    
    #start optimization
    negf = lambda x: -f(x)
    res = cma.fmin(negf, x, cma_sigma, opts)

    #collect results
    x_opt = res[5]
    f_opt = -res[1]
    res = res[-3]    
    
    return f_opt, x_opt, res

def main():
    import numpy as np
    
    def f(x, *args):
        return (-x*x).sum()
    
    n = 2
    bounds = {'lb': -10*np.ones(n), 'ub': 10*np.ones(n)}
    x = np.random.randn(n)    
    print( isres(f, n, 1e-4, 1e-4, bounds, x) )
    print( cmaes(f, n, 1e-4, 1e-4, bounds, x) )
        
if __name__ == '__main__':
    main()