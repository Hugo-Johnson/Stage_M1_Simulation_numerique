#%%
'''
auteur: Hugo Johnson
Etude système Hamiltonien
'''
%reload_ext autoreload
%autoreload 2

import numpy as np
import physique 
import solver 
import affichage 

from physique import graphene_FERMION_adim

##############          main         #################################################################
# Parametre pour Trajectoire et FFT
beta_parameter_tab = np.array([ 0., 0.5, 0.8, 0.99])
Px     = 1.6E-26
B      = 1
M_para = 0.0
Hamiltonien = graphene_FERMION_adim

#-- Paramètre pour Energies Graphe
Px_tab = [1.6E-26, 1E-26, 0.5E-26]
beta = 0.4
#------------------------------------------------------------
'''
beta = vd / vf (vf est la vitesse de référence pour vd)
Px : Position initiale sur l'axe y=0, permet de définir une énergie.
B : Champ magnétique (Tesla), entre 0 exclu et 20
Masse particule : M_para * masse électron avec M_para entre 0 et 0.04

Hamitonien disponible :
- graphene_FERMION_adim

Affichage:
FFT et Traj : à partir des listes permet de choisir d'afficher ou non les résultats.

'''

Fourier_list, Traj_list =  solver.resolv_beta_tab(Px, beta_parameter_tab, B, M_para, Hamiltonien)

Energie_list = solver.resolv_Energie_tab(Px_tab, M_para, beta, graphene_FERMION_adim)

#%% 
# Affichage

affichage.Spectre_et_Traj(Fourier_list, Traj_list, Voir_FFT=True, Voir_Traj=True)

affichage.Energies(Energie_list)



