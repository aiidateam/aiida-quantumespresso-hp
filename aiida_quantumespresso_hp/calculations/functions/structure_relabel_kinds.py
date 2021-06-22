# -*- coding: utf-8 -*-
"""Calculation function to reorder the kinds of a structure with the Hubbard sites first."""
import re

from aiida.engine import calcfunction
from aiida.orm import Dict


@calcfunction
def structure_relabel_kinds(structure, hubbard, magnetization):
    """Create a clone of the given structure but with new kinds, based on the new hubbard sites.

    :param structure: ``StructureData`` node.
    :param hubbard: the ``hubbard`` output node of a ``HpCalculation``.
    """
    relabeled = structure.clone()
    relabeled.clear_kinds()
    relabeled.clear_sites()

    kind_suffix = -1
    hubbard_u = {}
    type_to_kind = {}
    sites = structure.sites
    
    if magnetization != None:
        old_magnetization = magnetization.get_dict()
        new_magnetization = old_magnetization.copy()
        # Removing old Hubbard spin-polarized atom label.
        for index, site in enumerate(hubbard.get_attribute('sites')):
            new_magnetization.pop(site['kind'], None) 

    # First do the Hubbard sites, upping the kind name suffix each time a new type is encountered. We do the suffix
    # generation ourselves, because the indexing done by hp.x contains gaps in the sequence.
    for index, site in enumerate(hubbard.get_attribute('sites')):
        symbol = re.search(r'^([A-za-z]+)[0-9]*$', site['kind']).group(1)
        try:
            spin_type = str( int(site['new_type'])*int(site['spin']) )
            kind_name = type_to_kind[spin_type] 
        except KeyError:
            kind_suffix += 1
            kind_name = f'{symbol}{kind_suffix}'
            hubbard_u[kind_name] = float(site['value'])
            type_to_kind[spin_type] = kind_name
            if magnetization != None:
            # filling 'starting magnetization' with input value but new label;
            # if does not present a starting value, pass.
                try:
                    new_magnetization[kind_name] = old_magnetization[site['kind']] 
                except:
                    pass

        site = sites[index]
        relabeled.append_atom(position=site.position, symbols=symbol, name=kind_name)

    # Now add the non-Hubbard sites
    for site in sites[len(relabeled.sites):]:
        relabeled.append_atom(position=site.position, symbols=structure.get_kind(site.kind_name).symbols)

    return {'structure': relabeled, 'hubbard_u': Dict(dict=hubbard_u), 
            'starting_magnetization': Dict(dict=new_magnetization)}

