import Parameters as p
import Constants as c
import numpy as np

def get_temperatures(w):
    """
    Gets temperatures given primitives
    """
    
    T = w[p.iP] * p.m_bar / (w[p.irho] * c.K_B_cgs)
    return T

def get_number_density(w):
    """
    Gets number densities given primitives
    """
    
    T = get_temperatures(w)
    P = w[p.iP]
    n = P/(c.K_B_cgs*T)
    return n