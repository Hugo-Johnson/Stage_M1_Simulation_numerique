#%%
import numpy as np
from pyhamsys import HamSys, solve_ivp_sympext, Parameters
import scipy.constants 
'''
Cette cellule contient les Hamiltoniens et leurs variables 
Les paramètres comme vd_parameter, vf ou M seront écrasés plus tard

Paramètres SI permet d'exprimer les variables dans un contexte physique
Paramètre ADIM sont des facteurs qui permettent d'adimensionner les variables ce qui 
rend le calcul numérique plus facile.

Le paragraphe suivant contient les systèmes hamiltoniens adimensioné utilisés par le solver.
Je le laisse celui de référence en commentaire

Le dernier paragraphe permet de récupérer les bonnes variables pour le solver
'''
### Parametre SI ############################################################
vd_parameter = 3 # 10 étant le cas limite
vd_scale     = 1E5
vd           = vd_parameter * vd_scale
vf           = 1E6
hbar         = scipy.constants.hbar
e            = scipy.constants.e
m_e          = scipy.constants.m_e
E_parameter  = 50
E_scale      = scipy.constants.e * scipy.constants.milli
E            = E_parameter * E_scale
B            = 1 # de 0 à 20 exclu
one_meV      = e * scipy.constants.milli #Pour convertir mon hbar omega_c (qui en en joule) en MeV qui est une meilleur unité

M = 0.066 * m_e # Une masse même si M dans esp est une énergie il faut faire attention aux ajustements.

################# PARAMETRE ADIM ##########################################################

B_ref = 1 # 1 tesla
l     = np.sqrt(hbar / (e * B_ref))
p_ref = hbar / l
W_ref = hbar * vf / l
t_ref = l / vf
M_ref = hbar / (l * vf) 

M_adim  = None
B_adim  = None
vd_adim = None
# Listes des systèmes Hamiltoniens utilisés avec PyHamsys ------------------------------------
# ------ GRAPHENE ------------------------------------------------------------------
# def graphene_FERMION_system(t, y): 
#     G =  (e * B )
#     q, p = np.split(y, 2)
#     esp = np.sqrt(M**2 + vf**2 * (q**2 + p**2))
#     y0 =  G * ((vf**2 * p / esp) + vd) 
#     y1 = -G * ((vf**2 * q / esp))
#     return np.concatenate((y0, y1), axis=None)

def graphene_FERMION_adim(t, y):
    q, p = np.split(y, 2)
    eps_adim = np.sqrt(M_adim**2 + q**2 + p**2 )
    y0_adim =  B_adim * ( p / eps_adim + vd_adim) 
    y1_adim = -B_adim* ( q / eps_adim)
    return np.concatenate((y0_adim,y1_adim), axis=None)

def hbar_Omegac_theorique_graphene(B_adim, Px_adim):
    ''' 
    A partir des variables adimensionnées, donne hbar omega_c en meV
    '''
    W = np.sqrt(M_adim**2 + Px_adim**2)
    beta = vd_adim
    freq = e * B_adim * B_ref * vf**2  / (W * W_ref) * (1 - beta**2)**(3/2)
    hbar_omega_joule = freq * hbar
    return  hbar_omega_joule / one_meV
# -------------------------------------------------------------------------------

######################################################################

def physique_to_adim(vd, B_tab, M, P_tab):
    vd_adim = vd / vf # aussi tab dans certains cas
    B_tab_adim = B_tab / B_ref
    M_adim = M * vf**2 / W_ref 
    # Ici on a corrigé avec W ref et non M_rew parce que ce M doit être homogêne à une Energie (esp est une énergie) d'où le vf**2 qui apparait
    P_adim = P_tab / p_ref
    
    return vd_adim, B_tab_adim, M_adim, P_adim