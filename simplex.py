"""
=================================================
Exemple d'utilisation de l'algorithme du simplexe
=================================================
"""

import numpy as np

CRED = "\033[91m"  # on met de la couleur pour les sorties
CEND = "\033[0m"


def forme_standard(A, b, c, cont, minmax="min", verbose=True):
    """Fonction pour mettre le problème initial sous forme standard

    Args:
        A (array): matrice originale de taille (p, n)
        b (array): vecteur des contraintes de taille (p) 
        c (array): vecteur des coefficients de la fonction objectif de taille (n)
        cont (list): liste du type des contraintes : 
                        -1 : contrainte <= 
                         0 : contrainte =
                         1 : contrainte >=
        minmax (str, optional): Type de probleme, prend deux valeurs : min ou max. Defaults to "min".
        verbose (bool, optional): Affichage des détails. Defaults to True.

    Returns:
        A,b,c
    """
    p, n = np.shape(A)  # dimensions de A

    if verbose:
        print("########### Forme originale ################")
        pb = [f"{c[i]} x_{i} " for i in range(n)]
        pb = " + ".join(pb)
        if minmax == "min":
            print(f"min {pb}")
        if minmax == "max":
            print(f"max {pb}")
        print("Sous les contraintes :")
        for i in range(p):
            ci = [f"{A[i, j]} x_{j}" for j in range(n)]
            ci = " + ".join(ci)
            if cont[i] == -1:
                ci += f" <= {b[i]}"
            if cont[i] == 0:
                ci += f" = {b[i]}"
            if cont[i] == 1:
                ci += f" >= {b[i]}"

            print(ci)
        print("############################################")

    if minmax == "max":  # si problème de maximisation,
        # on change le signe de l'objectif pour obtenir
        # un problème de minimisation
        c[:] = -c[:]
    for i in range(
        np.shape(b)[0]
    ):  # si des valeurs de b sont negatives, on change leur signe ainsi que le signe de la colonne correspondant dans A
        if b[i] < 0:
            b[i] = -b[i]
            A[:, i] = -A[:, i]
            
    # on ajoute les variables d'ecart
    J_B = []
    for i in range(len(cont)):
        if cont[i] == -1:  # <=
            ecart = np.zeros(p)
            ecart[i] = 1
            A = np.c_[A, ecart]
            c = np.r_[c, 0]
            J_B.append(np.shape(A)[1]-1) 

        elif cont[i] == 1:  # >=
            ecart = np.zeros(p)
            ecart[i] = -1
            A = np.c_[A, ecart] 
            c = np.r_[c, 0] 
    p, n = np.shape(A)  # on réactualise les dimensions

    if verbose:
        print(
            "\n########### Matrices considérées pour la forme standard ########### \n"
            f"{A=} \n"
            f"{b=} \n"
            f"{c=}"
            "\n################################################################### \n"
        )

    return A, b, c, J_B

def simplexe(A, b, c, x0, verbose=True, phase=2, J_B=[], z_k = 0.0, count_iter = False):
    """
    Méthode du simplexe pour des problèmes d'optimisation linéaires. 
    Fonctionne sur des problèmes de minimisation et
    de maximisation avec des contraintes initiales de type "<="
     
    Args:
        A (array): matrice des contraintes de taille (p, n)
        b (array): vecteur des contraites de taille (p)
        c (array): vecteur des coefficients de l'objectif de taille (n)
        x0 (array): sommet initial
        verbose (bool, optionnel): affichage de détails.
            Par défaut : True.
        phase (int, optionnel): Si 1 résolution pour la phase 1, si 2 phase 2 de la méthode.
            Par défaut : 2.
        j_b (list, optional): base initiale. Non indiquée en général
        pour la phase 2, donnée pour la phase 1. Par défaut : [].
        z_k (float, optional): Valeur initiale de l'objectif. Defaults to 0.0.
        count_iter (bool, optional): Si true : on retourne le nombre d'itérations nécessaires. Defaults to False.

    Returns:
        pour la phase 1 :
            xk, Ak, bk, c_k, z_k : sommet initial, matrice des contraintes, 
                vecteur des contraintes, coefficients de l'objectif et valeur de l'objectif
        pour la phase 2 :
            xk, zk (, compteur) : sommet optimal, valeur de l'objectif à l'optimum (, nombre d'itérations)

        N.B. :
            x_k: k-ième itéré formé des variables x_i initiales ainsi que
            des variables d'écart
    """
    if phase == 1 : 
        c_k = A[-1,:] # objectif initial que l'on modifie lors des itérations 
        # de la phase 1 mais n'intervient pas dans les calculs
        A = A[:-1,:]
    p, n = np.shape(A)  # dimension de A

    if verbose:
        # on affiche les données initiales
        print(
            "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ \n"
            "Probleme : \n"
            f"- Coefficients de l'objectif : {c=} \n"
            f"- Matrice des contraintes : \n {A} \n"
            f"- Vecteur des contraintes : {b=} \n"
            f"- Sommet de depart {x0=} \n"
            "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ \n"
        )

    if verbose:
        print("\n########### Méthode ################")

    if phase == 2 or len(J_B) == 0: # si la base n'est pas donnée on la construit (c.f. cours)
        J_B = np.where(x0 > 0)[0]

    B = A[:, J_B]  # on crée la matrice B (matrice de base)

    # on crée les matrices et vecteurs de l'iteration 0
    # (voir cours pour formules)
    Ak = np.linalg.inv(B) @ A
    ck = (c.T - c[J_B].T @ Ak).T
    bk = np.linalg.inv(B) @ b
    xk = np.zeros(n)
    if phase == 2 and z_k != 0.0 :
        zk = z_k 
    else :
        zk = c[J_B].T@bk
    xk[J_B] = bk[:]
    compteur = 0
    if verbose:
        print(
            f"\n {CRED} ### Initialisation {CEND}\n"
            f"j_bk = {J_B} \n"
            f"Ak = \n {Ak} \n"
            f"{bk=} \n {ck=} \n {zk=}\n {xk=}\n"
        )

    while (
        min(ck) < 0
    ):  # on boucle avec comme critère d'arrêt le signe des ck_i
        if phase == 1:
            c_k1 = np.copy(c_k),
            z_k1 = np.copy(z_k)    
        Ak1, bk1, ck1, zk1, = map(np.copy, (Ak, bk, ck, zk))
        j_e = np.argmin(ck)  # variable entrante

        I_S = []  # on détermine la variable sortante
        for i in range(p):
            if Ak[i, j_e] > 0:
                I_S.append(bk[i] / Ak[i, j_e])
            else:
                I_S.append(np.inf)

        i_s = np.argmin(I_S)
        J_Bk = J_B
        # nouvelle base en modifiant la variable sortante
        # par la variable entrante
        J_Bk[i_s] = j_e

        # calculs des différents coefficients
        Ak1[i_s, :] = Ak[i_s, :] / Ak[i_s, j_e]
        bk1[i_s] = bk[i_s] / Ak[i_s, j_e]

        for i in range(p):
            if i != i_s:
                Ak1[i, :] = Ak[i, :] - Ak[i, j_e] * Ak1[i_s, :]
                bk1[i] = bk[i] - Ak[i, j_e] * bk1[i_s]

        ck1 = ck[:] - ck[j_e] * Ak1[i_s, :]
        zk1 = zk + ck[j_e] * bk1[i_s]
        if phase == 1:
            c_k1 = c_k[:] - c_k[j_e] * Ak1[i_s, :]
            z_k1 = z_k + c_k[j_e]*bk1[i_s]
            c_k = c_k1
            z_k = z_k1
        xk = np.zeros(n)
        xk[J_B] = bk1[:]
        # on crée les différents vecteurs et matrices
        # pour l'iteration suivante
        Ak, bk, ck, zk = map(np.copy,(Ak1, bk1, ck1, zk1))
        compteur += 1
        if verbose:
            print(
                f"\n {CRED}### Iteration : {compteur} {CEND} \n"
                f"{I_S=} \n"
                f"{j_e=} \n"
                f"{i_s=} \n"
                f"{J_Bk=} \n"
                f"Ak = \n {Ak} \n"
                f"{bk=} \n"
                f"{ck=} \n"
                f"{zk=} \n"
                f"{xk=}"
            )
    if phase == 1 :
        return xk, Ak, bk, c_k, z_k    
    if phase == 2 :
        if count_iter : 
            return xk, zk, compteur
        else : 
            return xk, zk

def phase_1(A, b, c, cont, J_B):
    """ Fonction pour modifier le problème afin de lui appliquer la phase 1 du simplexe

    Args:
        A (array): matrice des contraintes
        b (array): vecteur des contraintes 
        c (array): vecteur des coefficients de la fonction objectif
        cont (list): liste du type des contraintes : 
                        -1 : contrainte <= 
                         0 : contrainte =
                         1 : contrainte >=

    Returns:
        A[:,:n], x0[:n], c[:n],b,z : Nouvelle matrice des contraintes pour la phase 2, sommet initial de la phase 2, 
                coefficients de l'objectif pour la phase 2 et valeur initiale de l'objectif pour la phase 2
    """

    p, n = np.shape(A)  # nombre lignes, nombre de colonnes de A
    # on crée l'objectif : minimiser la somme des variables artificielles
    c_tilde = np.zeros(n)


    for i in range(len(cont)):       
        if cont[i] == 0 or cont[i] == 1:  # =
            artif = np.zeros(p)
            artif[i] = 1
            A = np.c_[A, artif]
            c_tilde = np.r_[c_tilde, 1]
            c = np.r_[c, 0]
            J_B.append(np.shape(A)[1]-1)
        
    p_tilde, n_tilde = np.shape(A)  # on réactualise les dimensions

    
    x0_tilde = np.zeros(n_tilde)
    x0_tilde[J_B] = b

    A = np.r_[A,[c]]

    x0, A, b, c, z = simplexe(
        A=A,
        b=b,
        c=c_tilde,
        x0=x0_tilde,
        verbose=True,
        phase=1,
        J_B=J_B
    )
    print(f"{A[:,:n]=} \n {x0[:n]=} \n {c[:n]=} \n {b=} \n {z=}")
    return A[:,:n], x0[:n], c[:n],b,z


def resolution_pb(
    A, b, c, x0, cont, minmax="min", verbose=True, count_iter = False
):
    """Résolution d'un problème d'optimisation linéaire par la méthode du simplexe (méthode des 2 phases).

    Args:
        A (array): matrice originale de taille (p, n)
        b (array): vecteur des contraintes de taille (p) 
        c (array): vecteur des composantes de la fonction objectif de taille (n)
        x0 (array): sommet initial
        cont (list): liste du type des contraintes : 
                        -1 : contrainte <= 
                         0 : contrainte =
                         1 : contrainte >=
        minmax (str, optional): Type de probleme, prend deux valeurs : min ou max. Defaults to "min".
        verbose (bool, optional): Affichage des détails. Defaults to True.


    Returns:
        xk, zk : solution et valeur optimale.
    """
    
    # on vérifie d'abord qu'il n'y a pas de problème de dimension pour les données initiales 
    p,n = np.shape(A)
    len_b = len(b)
    len_c = len(c)
    
    if n != len_c or p != len_b or len(cont) != p :
        raise ValueError('Il y a un problème de dimensions dans les données saisies.')

    p_init, n_init = np.shape(A)
    A, b, c, J_B = forme_standard(
        A, b, c, cont, minmax, verbose
    )  # on transforme le problème sous la forme standard
    n = np.shape(A)[1]
    c_init = c
    z = 0.0
    if (len(x0) != len(c) or len(np.where(x0!=0))!=p):  # si le sommet initial est une liste vide, ou s'il n'est pas saisi correctement, on construit un sommet initial via la phase 1.
        if len(x0) != 0 :
            print("Sommet initial de mauvaise dimension, construction d\'un sommet via la phase 1 de la méthode.")
        elif (len(x0) != 0 and len(np.where(x0!=0))!=p):
            print("Mauvais nombre de variables de base, construction d\'un sommet via la phase 1 de la méthode.")
            
        A, x0, c,b,z = phase_1(A, b,c, cont, J_B)
    if count_iter :
        xk, zk, iter = simplexe(
            A, b, c, x0, verbose, phase=2, J_B=[], z_k=z, count_iter=count_iter
        )  # phase 2 de la méthode
    else :
        xk, zk = simplexe(
            A, b, c, x0, verbose, phase=2, J_B=[], z_k=z
        )  # phase 2 de la méthode
    ecart = xk[n_init:]
    xk = xk[: n_init]
    zk = c_init.T[: n_init] @ xk
    if verbose:
        print("\n############ Solution finale ############\n")
        print(f"Solution : {xk=} \n")
        print(f"Variables d\'écart : {ecart=} \n")
        print(f"Valeur optimale : {zk=} \n")
        print("\n#########################################\n")

    if count_iter : 
        return xk, zk, iter 
    else :
        return xk, zk

