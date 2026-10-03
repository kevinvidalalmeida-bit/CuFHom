import numpy as np

def mandel_to_voigt_eng(Cm):
    """
    Mandel (6x6) -> Voigt ingenieril (6x6).
    Voigt ingenieril usa gamma_ij = 2*epsilon_ij, y C44=G.
    """
    T_inv = np.diag([1, 1, 1, 1/np.sqrt(2), 1/np.sqrt(2), 1/np.sqrt(2)])
    return T_inv @ Cm @ T_inv

def engineering_constants_from_Cmandel(Cm):
    Cm = 0.5*(Cm + Cm.T)

    Cv = mandel_to_voigt_eng(Cm)
    S  = np.linalg.inv(Cv)

    E1 = 1.0 / S[0,0]
    E2 = 1.0 / S[1,1]
    E3 = 1.0 / S[2,2]

    # Voigt ingenieril: 4->23, 5->13, 6->12
    G23 = 1.0 / S[3,3]
    G13 = 1.0 / S[4,4]
    G12 = 1.0 / S[5,5]

    nu12 = -S[0,1] / S[0,0]
    nu13 = -S[0,2] / S[0,0]
    nu23 = -S[1,2] / S[1,1]

    return dict(E1=E1,E2=E2,E3=E3, G12=G12,G13=G13,G23=G23, nu12=nu12,nu13=nu13,nu23=nu23)
