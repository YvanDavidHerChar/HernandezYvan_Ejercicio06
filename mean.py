import numpy as np
import matplotlib.pyplot as plt

x = [4.6 , 6.0 , 2.0 , 5.8]

sigma = [2.0 , 1.5 , 5.0 , 1.0]
#Usando el metodo del ejercicio pasado
mu = np.linspace(0,10,100)
valork = 1/(np.max(mu)-np.min(mu))

mult =  np.ones(100)
posterior = np.zeros(100)
for i in range(len(x)):
    verosimilitud1 = np.log((1/np.sqrt(2*np.pi*sigma[i]))*(np.exp((-1/2*(x[i]-mu)**2)/sigma[i]**2)))
    posterior +=  verosimilitud1

mult = np.exp(posterior)

Evidencia1 = np.trapz(mult,mu)

L =  posterior + np.log(valork) - np.log(Evidencia1)

L_mov = L[1:]

derv_L = (L_mov - L[0:99])/(mu[2]-mu[1])

zero = np.where(np.abs(derv_L)==np.min(np.abs(derv_L)))

derv_L_mov = derv_L[1:]

seg_derv_L = (derv_L_mov - derv_L[0:98])/((mu[2]-mu[1]))

sigma = (-seg_derv_L[zero])**(-1/2)

print(sigma)

mu_zero = mu[zero]

post = np.exp(L)
plt.plot(mu,post*10)
plt.title(r"$\mu$ = {:.2f} $\pm$ {:.2f}".format(float(mu_zero) , float(sigma)))
plt.xlabel('$\mu$')
plt.ylabel('P($\mu$|{obs})')
plt.savefig("mean.png", bbox_inches='tight')
