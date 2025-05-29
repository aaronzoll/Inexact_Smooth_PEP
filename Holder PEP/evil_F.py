import numpy as np
import matplotlib.pyplot as plt

def grad_bounds(x, x_l, x_r, g_l, g_r, beta, p):
    min = g_r - beta*((p+1)/(2*p)*np.abs(x-x_r))**p
    max = g_l + beta*((p+1)/(2*p)*np.abs(x-x_l))**p

    return (min,max)

def func_bounds(x, g, x_l, x_r, f_l, f_r, g_l, g_r, beta, p):
    m1 = f_l + g_l*(x-x_l) + beta**(-1/p)*p/(p+1)*np.abs(g_l-g)**(1+1/p)
    m2 = f_r + g_r*(x-x_r) + beta**(-1/p)*p/(p+1)*np.abs(g_r-g)**(1+1/p)
    min = np.max([m1, m2])

    M1 = f_l - g*(x_l-x) - beta**(-1/p)*p/(p+1)*np.abs(g_l-g)**(1+1/p)
    M2 = f_r - g*(x_r-x) - beta**(-1/p)*p/(p+1)*np.abs(g_r-g)**(1+1/p)
    max = np.min([M1, M2])

    return (min, max)

def cocoersive(x, y, f_x, f_y, g_x, g_y, beta, p):
    return np.round(f_y - (f_x + g_x*(y-x)+beta**(-1/p)*(p/(p+1))*np.abs(g_x-g_y)**(1+1/p)),6)

def holder_bound(x,y,g_x,g_y,beta,p):
    return np.round(beta*((p+1)/(2*p))**p*np.abs(x-y)**p-np.abs(g_x-g_y),6)

beta = 5
p = 0.2
m = 3

for p in np.linspace(0.2,0.5,7):
    x = np.array([0., m])
    f = np.array([0, m/2*beta*(m*(p+1)/(2*p))**p])
    g = np.array([0, beta*(m*(p+1)/(2*p))**p])

    for _ in range(5):
        for j in range(len(x)-1,0,-1):
            x = np.insert(x,j,1/2*(x[j-1]+x[j]))  

            g_save = g
            g = np.insert(g,j,0)  
            g_min, g_max =  grad_bounds(x[j], x[j-1], x[j+1], g[j-1], g[j+1], beta, p)
            G_min = np.insert(g_save, j, g_min)
            G_max = np.insert(g_save, j, g_max)

            f_save = f
            f = np.insert(f,j,0)

            alpha = 0.5
            
            G = alpha*(g_min)+(1-alpha)*g_max
            
            f_min, f_max =  func_bounds(x[j], G, x[j-1], x[j+1], f[j-1], f[j+1], g[j-1], g[j+1], beta, p)
            F_min = np.insert(f_save, j, f_min)
            F_max = np.insert(f_save, j, f_max)
        
            g = alpha*G_min+(1-alpha)*G_max
            f = alpha*F_min+(1-alpha)*F_max

    # check cocoercivity

    for i in range(len(f)):
        for j in range(i+1,len(f)):
            if cocoersive(x[i], x[j], f[i], f[j], g[i], g[j], beta, p) < 10**-7:
                print(i,j)
                print(cocoersive(x[i], x[j], f[i], f[j], g[i], g[j], beta, p))
            if holder_bound(x[i],x[j],g[i],g[j],beta,p) < 10**-7:
                print(i,j)
                print(holder_bound(x[i],x[j],g[i],g[j],beta,p))


    #print(f)
    #print(g)
    plt.plot(x,f,label=str(p))
#plt.plot(x,0.5*x**2, label="quadratic")
plt.legend()
plt.show()


