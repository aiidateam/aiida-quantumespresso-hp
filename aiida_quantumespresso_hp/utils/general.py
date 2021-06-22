# -*- coding: utf-8 -*-
"""Utilities."""

# Should be a calcfunction ???
def set_tot_magnetization(input_parameters, tot_magnetization):
    """
    Set the tot magnetization input key equal to the round value of tot_magnetization and return TRUE if 
    the latter does not exceed the given threshold from its original value.
    This is needed because 'tot_magnetization' must be an integer in the aiida-quantumespresso input parameters.
    """
    thr = 0.1 # threshold measuring the deviation from integer value
    
    int_tot_magnetization = round(tot_magnetization, 0)
    input_parameters['SYSTEM']['tot_magnetization'] = int_tot_magnetization
    
    if ( abs(tot_magnetization-int_tot_magnetization) < thr ):
        return False
    else:
        return True
    