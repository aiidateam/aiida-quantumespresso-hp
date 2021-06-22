# -*- coding: utf-8 -*-
"""Calculation function to return the sites of a structure for Hubbard_V."""
from aiida.engine import calcfunction
from aiida.orm import List

@calcfunction                     
def create_hubbard_v_from_distance(hubbard_input, structure):
    """Return the proper hubbard_v list from the hubbard_start input."""
    hubbard_out = [] # hubbard_v to return
    row = [] 
    kindname_list = structure.get_site_kindnames()
    for key in hubbard_input.get_list():
        row.append(kindname_list.index(key[0])+1) # in quantumespresso atomic indices start from 1, not 0
        row.append(kindname_list.index(key[1])+1)
        row.append(key[2])
        row.append(key[3])
        hubbard_out.append(row)
        # re-initialize
        row =[]
    return List(list=hubbard_out)

