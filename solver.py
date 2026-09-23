#%% 
# Solver
import numpy as np
from pyhamsys import HamSys, solve_ivp_sympext, Parameters
from scipy.signal import find_peaks
import time
import physique
from physique import e, B_ref, vf, W_ref, t_ref, hbar, one_meV, graphene_FERMION_adim
import affichage
import sys
######################################################

hs = HamSys(ndof=1) # Pour solve_ivp 

# Paramètres solver ##################
def seuil_cycle_FERMION(parametre):
    parametre = parametre / physique.vd_scale
    if parametre <= 5:
        return 30
    elif parametre <= 8:
        return 15
    elif parametre <= 9:
        return 10
    else:
        return 1

def adpat_param_solver(B, Energie, vd):
    '''
    B va de 0 à 20 max
    beta de 0, 1

    B = 1 delta 1 convient toujours
    '''
    seuil = None
    deltaT = None
    if B <5:
        deltaT= 1
        seuil = seuil_cycle_FERMION(vd)
    elif B <10:
        deltaT = 0.1
        seuil = 2 * seuil_cycle_FERMION(vd)
    else:
        if vd/ physique.vd_scale >= 9.5:
            deltaT = 0.05
            seuil = seuil_cycle_FERMION(vd)
        else:
            deltaT = 0.01
            seuil = 10
    if Energie > 1000:
        seuil = 3
        
    return seuil, deltaT


def find_periode(P_adim, vd_adim, B_adim, M_adim, precision_yoshida, systeme_hamiltonien, seuil_periode_pratique):
    '''
    On cherche la période numériquement 
    Pour se faire on part d'une période posée 
    puis on commence à chercher à "boucler" plusieurs fois l'ellipse et ainsi estimer la période.
    '''
   
    Px_adim = P_adim[0]
    W = np.sqrt(M_adim**2 + Px_adim**2)

    T_c_estim = 2 * np.pi * (1 / (e * (B_adim * B_ref) * vf**2 / (W * W_ref) * ( 1 - vd_adim**2)**(3/2))) * 2
    T_long_adim =  T_c_estim / t_ref
    
    vd = vd_adim
    B = B_adim
    nombre_periode = 0
    
    t_peaks = np.array([np.nan])
    sol_long = None
    Detection_first_period = True
    cpt = 0
    
    while (nombre_periode < seuil_periode_pratique) or (np.isnan(t_peaks).any() == True) :
        hs.y_dot = systeme_hamiltonien
        deltaT_Yoshida = precision_yoshida
        sol_long = solve_ivp_sympext(hs, t_span=(0, T_long_adim), y0=P_adim, params=Parameters(step=deltaT_Yoshida, solver='Yos6'))

        py_traj = sol_long.y[1]  
         
        peaks, properties = find_peaks(py_traj, 0.5 * np.max(py_traj))  
        t_peaks = sol_long.t[peaks]
        nombre_periode = len(np.diff(t_peaks))
        # print(np.diff(t_peaks))
        
        if nombre_periode >= 1:
            if Detection_first_period:
                # print(f'Tentatives échouées avant d\'avoir réussi = {cpt}')
                # print(f'La période à été trouvé sur {nombre_periode} cycle(s)')
                Detection_first_period=False
            T_long_adim = np.max(np.diff(t_peaks)) * (seuil_periode_pratique + 1)
        else:
            T_long_adim = T_long_adim * 2
            # print(f'Nouveaux T_long_adim testé : {T_long_adim}')
            # print(f'cpt = {cpt}')
            cpt += 1
    periode = np.mean(np.diff(t_peaks))
    return periode ,sol_long

def calculate_FFT_TRAJ(Px, beta, B, M, Hamiltonien_ADIM):
    '''
    Return Fourier list : infos pour plot spectre FFT et paramètre du plot
    sol_long : infos pour plot la trajectoire
    '''
    ##########################
    vd = beta * vf
    vd_adim, B_adim, M_adim, Px_adim = physique.physique_to_adim(vd, B, M, Px )
    
    physique.vd_adim = vd_adim
    physique.B_adim = B_adim
    physique.M_adim = M_adim
    ###########################
    P_adim = np.array([Px_adim, 0])
    Energie = np.sqrt((M * vf**2)**2 + vf**2 * Px**2) / one_meV
    seuil, deltaT = adpat_param_solver(B, Energie, vd)
    print(f'Seuil = {seuil} et deltaT = {deltaT}')
    start       = time.perf_counter()
    T, sol_long = find_periode(P_adim, vd_adim, B_adim, M_adim, precision_yoshida=deltaT, systeme_hamiltonien=Hamiltonien_ADIM, seuil_periode_pratique=seuil)
    end         = time.perf_counter()
    temps_final, Unit  = affichage.affiche_time(end-start)

    
#     print(f'''Résolution pour :
#     beta = {beta}
#     vf = {vf}
#     B = {B}
#     M = {M}
#     Temps calcul pour {seuil_cycle_FERMION(beta)} cycles : {temps_final} {Unit}
# ''' )

    periode_tab_adim = T

    ############# Partie Fourier ADIM avec Sol_long récupérer via le calul de T #######################################################
    q, p     = sol_long.y
    eps_adim = np.sqrt(M_adim**2 + q**2 + p**2 )
    dt_q     =  B_adim * ( p / eps_adim + vd_adim) 
    # dt_p = -B_adim* ( q / eps_adim) # pour comparer plus tard et voir lequel donne une meilleure transformée de Fourier
    
    Fourier_transofrm = np.fft.rfft(dt_q)
    dt                = sol_long.step
    N                 = len(dt_q)
    freqs_adim        = np.fft.rfftfreq(N, d=dt) # fréquences adimensionnées
    amplitudes        = 2 * np.abs(Fourier_transofrm) / N 
    Fourier_tab = [freqs_adim, amplitudes]
    ####################################################################################################################################
                
    periode_tab     = t_ref * periode_tab_adim
    omega_c         = 2* np.pi / periode_tab
    Energie_adim    = np.sqrt(M_adim**2 + Px_adim**2)
    Energie         = W_ref * Energie_adim / one_meV
    
    frequence_cyclotron = (hbar * omega_c) / one_meV 
    M_para = M / physique.m_e
    Fourier_parameters = [frequence_cyclotron, Energie, M_para , beta, B]
    # print(f'hbar Omega_c : {frequence_cyclotron} meV \n##############################################################\n')
    Fourier_list = [Fourier_tab, Fourier_parameters]
    
    Traj_list = [sol_long, Energie, B_adim, beta, Px_adim, M_para, Hamiltonien_ADIM]
    return Fourier_list, Traj_list

def resolv_beta_tab(Px, beta_parameter_tab, B, M_para, Hamiltonien):
    print('START: resolv FFT et Traj')
    Fourier_list_beta_tab = []
    Traj_list_beta_tab = []
    start = time.perf_counter()
    for _, beta in enumerate(beta_parameter_tab):
        if beta >= 1:
            raise ValueError(f'beta doit être inférieur à 1 mais vaut {beta}')
        M  = M_para * physique.m_e

        Fourier_list, Traj_list = calculate_FFT_TRAJ(Px, beta, B, M, Hamiltonien)

        Fourier_list_beta_tab.append(Fourier_list)
        Traj_list_beta_tab.append(Traj_list)
    end = time.perf_counter()
    temps, unit = affichage.affiche_time(end-start)
    
    print(f'Temps de résolution : {temps} {unit}')
    print('END: resolv FFT et Traj \n')
    return Fourier_list_beta_tab, Traj_list_beta_tab

def calculate_Energie_graphe(Px, beta, B_tab, M, Hamiltonien_ADIM):
    periode_tab_adim = np.zeros_like(B_tab)
    freq_theorique_tab = np.zeros_like(B_tab)

    vd = beta * vf
    vd_adim, B_tab_adim, M_adim, Px_adim  = physique.physique_to_adim(vd, B_tab, M, Px)
    P_adim = np.array([Px_adim, 0])
    physique.vd_adim = vd_adim
    physique.M_adim = M_adim

    for i, B_adim in enumerate(B_tab_adim):
        physique.B_adim = B_adim
        Energie = np.sqrt((M * vf**2)**2 + vf**2 * Px**2)
        seuil, deltaT = adpat_param_solver(B_adim, Energie, vd)
        deltaT= 0.01
        T, sol_long = find_periode(P_adim, vd_adim, B_adim, M_adim , precision_yoshida=deltaT, systeme_hamiltonien=Hamiltonien_ADIM, seuil_periode_pratique=seuil)

        periode_tab_adim[i] = T
        freq_theorique_tab[i] = physique.hbar_Omegac_theorique_graphene(B_adim, Px_adim)
    periode_tab = t_ref * periode_tab_adim
    omega_c = 2* np.pi / periode_tab
    Energie_meV = Energie / one_meV
    frequence_cyclotron = (hbar * omega_c) / one_meV  
    return [[frequence_cyclotron, freq_theorique_tab], Energie_meV]


def resolv_Energie_tab(Px_tab, M_para, beta, Hamiltonien_ADIM):
    print('Start: Resolv graphe Energie')
    B_tab = np.array([10., 14., 20.])
    M = M_para * physique.m_e
    List_Droite_Energie = []
    List_Droite_theorique = []
    List_Energie = []
    List_parameters = []
    start = time.perf_counter()
    for i, Px in enumerate(Px_tab):
        
        Listes, Energie_meV = calculate_Energie_graphe(Px, beta, B_tab, M, Hamiltonien_ADIM)
        List_Droite_Energie.append(Listes[0])
        List_Droite_theorique.append(Listes[1])
        List_Energie.append(Energie_meV)
        print(f'Energie "{i+1}" fait')
    end = time.perf_counter()
    Temps, Unit = affichage.affiche_time(end - start)
    List_parameters = [B_tab, List_Energie, M_para, beta, vf]
    Liste_affichage = [List_Droite_Energie, List_Droite_theorique, List_parameters]
    
    print(f'Temps de calcul: {Temps} {Unit}')
    print('End: Resolv graphe Energie \n')
    return Liste_affichage