import numpy as np 
import simplex 
import matplotlib.pyplot as plt 

def compute_case(case, n = 0):
    """Générer quelques exemples

    Args:
        case (int): Numero de l'exemple :
            - 1 : Exemple principal du cours : (les ceintures)
                    min -2x_0 - 1.5x_1 ; 
                        2x_0 + x_1 <= 1000
                        x_0 + x_1  <=  800
                        x_0        <=  400
                        x_1        <=  700
                        x_0 , x_1  >= 0.
            - 2 : Même problème mais sans donner le sommet initial
            - 3 : Même problème mais sans les contraintes non-saturées à l'optimum : 
                    min -2x_0 - 1.5x_1 ; 
                        2x_0 + x_1 <= 1000
                        x_0 + x_1  <=  800
                        x_0 , x_1  >= 0.
            
            -4 : exemple tiré du contrôle terminal : 
                    min 5x_1 + 3x_2 ;
                        3x_1 + 2x_2 >= 600 000
                        x_1 + x_2   <= 250 000
                        x_1         >=  90 000
                        x_1, x_2    >= 0.
            -5 : problème de Kee-Minty 
                    résolution du problème de Klee-Minty de différentes dimensions, pour vérifier la complexité de l'algorithme
            
            -6 : exemple de la rafinerie :
                    max 524x_1 + 568x_2 ;
                        x_1                 >= 1000
                        x_2                 >= 2000
                        0.4x_1 + 0.5x_2     <= 3400
                        0.24x_1 + 0.18x_2   >= 1800
                        x_1 + x_2           <= 8000
    Returns:
        A, b, c, x0, cont pour le problème à résoudre
    """

    if case == 1:
        A = np.array([[2., 1.], [1., 1.], [1., 0.], [0., 1.]])
        c = np.array([-2., -1.5])
        b = np.array([1000., 800., 400., 700.])
        x0 = np.array([0., 0., 1000., 800., 400.])
        cont = [-1, -1, -1,-1]
        minmax = "min"

    elif case == 2 :
        A = np.array([[2., 1.], [1., 1.], [1., 0.], [0., 1.]])
        c = np.array([-2., -1.5])
        b = np.array([1000., 800., 400., 700.])
        x0 = []
        cont = [-1, -1, -1,-1]
        minmax = "min"
        
    elif case == 3:
        A = np.array([[2., 1.], [1., 1.]])
        c = np.array([-2., -1.5])
        b = np.array([1000., 800.])
        x0 = [] 
        cont = [-1, -1]
        minmax = "min"
        
    elif case == 4:
        A = np.array([[3.,2.],[1.,1.],[1.,0.]])
        b = np.array([600000.,250000.,90000.])
        c = np.array([5.,3.])
        x0 = [] 
        cont = [1,-1,1]
        minmax = "min"      
    
    elif case == 5:
        n = n
        A = np.zeros((n,n))
        b = np.zeros(n)
        c = np.zeros(n)
        for i in range(n):
            c[i] = 10.**(i) # la boucle commence à 0 donc pas besoin du -1
            b[i] = 10.**(2.*n -2.*(i+1.))
            A[i,i] = 1.
            for j in range(i+1,n):
                A[i,j] = 2.*10**(j-i)
        x0 = [] 
        cont = -np.ones(n) # seulement des inegalites du type <=
        minmax = "max"
    
    elif case == 6:
        A = np.array([[1.,0.], [0.,1.],[0.4,0.5],[0.24,0.18],[1.,1.]])
        b = np.array([1000.,1000.,3400.,1800.,8000.])
        c = np.array([524.,568.])
        x0 = []
        cont = [1,1,-1,1,-1]
        minmax = "max"
    else:
        raise NotImplementedError("Seulement 6 exemples : de 1 à 6")
    return A, b, c, x0, cont, minmax


if __name__ == "__main__":
    """ Résolution d'un problème d'optimisation linéaire avec la méthode du simplexe. 
    """
    test_case = 1

    if test_case == 5: 
        # pour le problème de Klee-Minty, on trace le nombre d'itérations en 
        # fonction de la taille du problème considéré 
        dim, iter = [], []
        for i in range(3, 4):

            A, b, c, x0, cont, minmax = compute_case(test_case, i)
            x, z, nb_iter = simplex.resolution_pb(A, b, c, [], cont, minmax, count_iter=True)
            dim.append(i)
            iter.append(nb_iter)
        plt.figure() 
        plt.plot(dim, iter, label='Iterations simplexe')
        plt.plot(dim, [2**d -1 for d in dim], '--+', color = 'purple', label=r'$2^n -1$')
        plt.xlabel('Dimension')
        plt.ylabel('Iterations')
        plt.title('Nombre d\'iterations en fonction de la dimension')
        plt.legend()
        plt.show()
        
    else:
        A, b, c, x0, cont, minmax = compute_case(test_case)
        x, z = simplex.resolution_pb(A, b, c, [], cont, minmax)



