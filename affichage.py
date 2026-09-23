#%%
import numpy as np
from physique import e, B_ref, vf, W_ref, t_ref, hbar, one_meV, p_ref, graphene_FERMION_adim
from scipy.signal import find_peaks
import matplotlib.pyplot as plt

############################################################
def affiche_time(time):
    if time < 60:
        time = np.round(time, 3)
        Unit_time = 'secondes'
    elif time < 60 * 60:
        time = np.round(time /60, 2)
        Unit_time = 'minutes'
    else:
        time = np.round(time /(60*60), 1)
        Unit_time = 'heures'
    return time, Unit_time

#### PLOT #################################################
def FFT(Fourier_list, affichage_N_mev=True, N_harmonique=5, axes=None ):
    '''
    Fourier list comprend l'ensemble des donnés à afficher
    N_harmonique correspond au nombre d'harmonique minimal à afficher
    '''
    externe = axes is not None
    if affichage_N_mev:
        print('- Entrer le meV max à afficher')
        N = int(input('meV max = ?'))
    if len(Fourier_list) == 1:      
        Fourier_tab, Fourier_parameters = Fourier_list[0]
        freqs_adim, amplitudes = Fourier_tab
        hbar_omega_c, Energie, M_para, beta, B = Fourier_parameters
        
        freqs = hbar / t_ref * 2 *np.pi * freqs_adim / one_meV
        
        
        ax_fft = axes if externe else plt.figure().gca()
        # ax_fft.plot(freqs, amplitudes)
        # coeffs_fourier, ppt = find_peaks(amplitudes)
        # t_coeffs_fourier = freqs[coeffs_fourier]
        # amp_coeffs_fourier = amplitudes[coeffs_fourier]

        # plt.scatter(t_coeffs_fourier, amp_coeffs_fourier, color='red', marker='o', s=10, zorder=3)
        # plt.axvline(x=frequence_cyclotron[0], color='red', linestyle='--', label=fr'{frequence_cyclotron[0]}$ \ \hbar \omega_c$ ')

        ax_fft.set_xlabel(r"$ \hbar \omega_c \ meV$")
        ax_fft.set_ylabel("Amplitude")
        
        ax_fft.set_title(fr' $\beta = {beta},\ B = {B}\ T,\ E = {np.round(Energie, 2)}\ meV,\ M = {M_para} \times m_e,\ v_f = {vf}\ m.s^{-1} $')
        if affichage_N_mev:
            ax_fft.set_xlim(0, N)
        else: ax_fft.set_xlim(0, ((N_harmonique +1)*1.05 ) * hbar_omega_c )

        ax_fft.grid()
        # ax_fft.legend()
        if not externe:
            plt.suptitle(f"Spectre $FFT$")       

    else:
        if not externe:
            fig, axes = plt.subplots(len(Fourier_list), 1, figsize=(9, 10))
        for i in range(len(Fourier_list)):
            ax_fft  = axes[i]
            Fourier_tab, Fourier_parameters = Fourier_list[i]
            freqs_adim, amplitudes = Fourier_tab
            hbar_omega_c, Energie, M_para, beta, B = Fourier_parameters
            
            freqs = hbar / t_ref * 2 *np.pi * freqs_adim / one_meV
            
            ax_fft.plot(freqs, amplitudes)
            # coeffs_fourier, ppt = find_peaks(amplitudes)
            # t_coeffs_fourier = freqs[coeffs_fourier]
            # amp_coeffs_fourier = amplitudes[coeffs_fourier]

            # ax_fft.scatter(t_coeffs_fourier, amp_coeffs_fourier, color='red', marker='o', s=10, zorder=3)
            # ax_fft.axvline(x=frequence_cyclotron[0], color='red', linestyle='--', label=fr'{frequence_cyclotron[0]}$ \ \hbar \omega_c$ ')

            ax_fft.set_xlabel(r"$ \hbar \omega_c \ meV$")
            ax_fft.set_ylabel("Amplitude")
            
            ax_fft.set_title(fr' $\beta = {beta},\ B = {B}\ T,\ E = {np.round(Energie, 2)}\ meV,\ M = {M_para} \times m_e,\ v_f = {vf}\ m.s^{-1}$')
            if affichage_N_mev:
                ax_fft.set_xlim(0,N)
            else: 
                ax_fft.set_xlim(0, (N_harmonique* 1.2) * hbar_omega_c )
            ax_fft.grid()
            # ax_fft.legend()
        if not externe:
            plt.suptitle(f"Spectre $FFT$")
    if not externe:            
        plt.tight_layout()
        plt.show()
    return
 
def Traj(Traj_list, axes=None):
    externe = axes is not None
    if len(Traj_list) == 1 :
                sol_long, Energie, B_adim, beta, Px_adim, M_para, Hamiltonien_adim = Traj_list[0]
                ax_traj = axes if externe else plt.figure().gca()
                R_theorique = Px_adim * p_ref
                x = np.linspace(-R_theorique, R_theorique, 500)
                ax_traj.plot(sol_long.y[0] * p_ref, sol_long.y[1] * p_ref, label='Solution numérique')
                
                ax_traj.plot(x, np.sqrt(R_theorique**2-x**2), color='black', linestyle='--', label=r'Solution théorique $\beta=0$') 
                ax_traj.legend(shadow=True)
                
                ax_traj.plot(x, -np.sqrt(R_theorique**2-x**2), color='black', linestyle='--')
                ax_traj.set_title(label=fr'$\beta = {beta},\ B={B_adim * B_ref} \ T, \ E={np.round(Energie, 2)} \ meV,\ M = {M_para} \times m_e, \ v_f = {vf}\ m.s^{-1}$')
                ax_traj.set_xlabel(r'$P_x$')
                ax_traj.set_ylabel(r'$P_y$')
                ax_traj.axis('equal')
                ax_traj.grid()
                if not externe:
                    plt.suptitle('Trajectoire')
    else:
            if externe:
                axes_list = axes
            else:
                fig, axes = plt.subplots(len(Traj_list), 1, figsize=(9, 10))
            for i in range(len(Traj_list)):
                sol_long, Energie, B_adim, beta, Px_adim, M_para, Hamiltonien_adim = Traj_list[i]
                ax_traj  = axes[i]
                R_theorique = Px_adim * p_ref
                x = np.linspace(-R_theorique, R_theorique, 500)
                if i == 0 :
                    ax_traj.plot(sol_long.y[0] * p_ref, sol_long.y[1] * p_ref, label='Solution numérique')
                    
                    ax_traj.plot(x, np.sqrt(R_theorique**2-x**2), color='black', linestyle='--', label=r'Solution théorique $\beta=0$') 
                    ax_traj.legend(shadow=True)
                else:
                    ax_traj.plot(sol_long.y[0] * p_ref, sol_long.y[1] * p_ref)
                    ax_traj.plot(x, np.sqrt(R_theorique**2-x**2), color='black', linestyle='--')
                ax_traj.plot(x, -np.sqrt(R_theorique**2-x**2), color='black', linestyle='--')
                ax_traj.set_title(label=fr'$\beta = {beta},\ B={B_adim * B_ref} \ T, \ E={np.round(Energie, 2)} \ meV,\ M = {M_para} \times m_e,\ v_f = {vf}\ m.s^{-1}$')
                ax_traj.set_xlabel(r'$P_x$')
                ax_traj.set_ylabel(r'$P_y$')
                ax_traj.axis('equal')
                ax_traj.grid()
            if not externe:
                plt.suptitle('Trajectoire')
    if not externe:
        plt.tight_layout()
        plt.show()
    return

def Spectre_et_Traj(Fourier_liste, Traj_liste, Voir_FFT = True, Voir_Traj = True):
    # Codé à l'aide d'une ia pour fusionner les deux méthodes précédentes (gestion des axes)
    if not Voir_FFT and not Voir_Traj:
        None
        return
    elif Voir_FFT and not Voir_Traj:
            print('''Choix options affichage spectre FFT:
        [0] = Affichage selon un nombre d'harmonique
        [1] = Affichage en fixant un max meV
            ''')
            choix = int(input('[x] = Choix entre 1 et 0'))
            N = None
            while choix not in (0, 1):
                print('Il faut choisir 1 ou 0 et pas autre chose !')
                choix = int(input('[x] = Choix entre 1 et 0'))
            if choix ==0:
                print('Entrer un nombre d\'harmonique à afficher')
                N = int(input('N = ?'))
            FFT(Fourier_liste, affichage_N_mev=choix, N_harmonique=N)   
    elif Voir_Traj and not Voir_FFT:
        Traj(Traj_liste)
    else:
        # Voir_FFT and Voir_Traj : affichage côte à côte
        n = len(Fourier_liste)
        
        print('''Choix options affichage:
        [0] = Affichage selon un nombre d'harmonique
        [1] = Affichage en fixant un max meV
            ''')
        choix = int(input('[x] = Choix entre 1 et 0'))
        N= None
        while choix not in (0, 1):
            print('Il faut choisir 1 ou 0 et pas autre chose !')
            choix = int(input('[x] = Choix entre 1 et 0'))
        if choix ==0:
            print('Entrer un nombre d\'harmonique à afficher')
            N = int(input('N = ?'))

        fig, axes = plt.subplots(n, 2, figsize=(14, 5 * n), squeeze=False)

        Traj(Traj_liste, axes=axes[0, 0] if n == 1 else axes[:, 0])
        FFT(Fourier_liste, affichage_N_mev=choix, N_harmonique=N, axes=axes[0, 1] if n == 1 else axes[:, 1])

        fig.suptitle("Trajectoire et Spectre FFT ")
        plt.tight_layout(rect=[0, 0, 1, 0.98])
        plt.show()

# --- Affichage tableau des Energies --------------------
def Energies(List_Energies_affichages):
    List_Droite_Energie = List_Energies_affichages[0]
    List_Droite_theorique = List_Energies_affichages[1]
    B_tab, List_Energie, M, beta, vf = List_Energies_affichages[2]

    plt.figure()
    for i in range(len(List_Droite_Energie)):
        plt.plot(B_tab, List_Droite_Energie[i], marker='o', label = fr"$Énergie={np.round(List_Energie[i], 3)} meV$")
        if i==len(List_Droite_Energie) - 1:
            plt.plot(B_tab, List_Droite_theorique[i], marker='o', linestyle='--', label = "Solution théorique Sergei", color='black', alpha=0.75 )
        else: plt.plot(B_tab, List_Droite_theorique[i], marker='o', linestyle='--', color='black', alpha=0.75 )
    plt.xlabel(r"$B (T)$")
    plt.ylabel(r"$ \hbar \omega_c \ meV$")
    plt.suptitle(f"Fréquence en fonction de B")
    plt.title(fr' $\beta = {beta},\ M = {M} \times m_e, v_f = {vf} m.s^{-1}$')
    plt.grid()
    plt.legend()
plt.show()

