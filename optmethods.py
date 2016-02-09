# -*- coding: utf-8 -*-
import nlopt as no
import numpy as np
import cma

def isres(f, n, ftol, xtol, bounds, x_init):
    """
    f - objective function
    n - number of parameters
    ftol - tolerance for the change of f function
    xtol - tolerance for the change of paramters x
    bound - absolute value of lower and upper bounds
    x_init - initial paramter value
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
    opt.set_maxtime(1200)
    
    #start optimization
    x_opt = opt.optimize(x_init)
    
    #collect results
    f_opt = opt.last_optimum_value()
    res = opt.last_optimize_result()    
    
    return f_opt, x_opt, res
    

def cmaes(f, n, ftol, xtol, bounds, x):
    """
    f - objective function
    n - number of parameters
    ftol - tolerance for the change of f function
    xtol - tolerance for the change of paramters x
    bound - absolute value of lower and upper bounds
    x_init - initial paramter value
    """
    
    #set options of the cma-es optimizer
    opts = cma.CMAOptions()
    opts['popsize'] = n*10
    opts['CMA_active'] = False
    opts['tolfun'] = ftol
    opts['tolx'] = xtol    
    opts['verb_disp'] = 0
    opts['verbose'] = 0
    if(len(bounds['lb']) == n and len(bounds['ub']) == n):
        opts['bounds'] = [bounds['lb'], bounds['ub']]
    elif type(bounds['lb']) == int or type(bounds['lb']) == float:
        opts['bounds'] = [[bounds['lb']]*n, [bounds['ub']]*n]
    else:
        print 'inconsistent definition of parameter bounds'
        
    cma_sigma = 5;
    
    #start optimization
    res = cma.fmin(-f, x, cma_sigma, opts)

    #collect results
    x_opt = res[5]
    f_opt = -res[1]
    res = res[-3]    
    
    return f_opt, x_opt, res