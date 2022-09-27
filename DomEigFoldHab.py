import numpy as np
from scipy.integrate import *
import matplotlib.pyplot as plt
from numpy import linalg as LA
from numpy import real

time = 1000

evol_step = 0.00001
k = 0.2
dose= .8

vopt=0
st = 5
r = 0.5
m = 0.1
KR = 100
KD = 60
alpha = 1.5
beta = 2
phi = 2
zeta = KR/KD

w1 = 0.43486044
w2 = 0.14807956
w3 = 0.28896748
w4 = 0.12809252

IC = [10,5,0.01] #R,D,v

spectrum = []
t_vec = []

def drug(d,v):
    return d*np.exp(-(v-vopt)**2/st)

def normalize(v):
    tot = sum(v)
    return v / tot

def spec(v,d,R,D):
    R = np.array([[(w1*(r*(KR-R)/KR-drug(d,v)-m)+w2*(r/alpha*(KR-R)/KR-drug(d,v)/beta-phi*m))/(w1+w2),(w3*m+w4*phi*m)/(w3+w4)],[(w1*m+w2*phi*m)/(w1+w2),(w3*(r*(KD-D)/KD-drug(d,v)/zeta-m)+w4*(r/alpha*(KD-D)/KD-drug(d,v)/(beta*zeta)-phi*m))/(w3+w4)]])
    spectrum.append(max(real(LA.eig(R)[0])))
    return max(real(LA.eig(R)[0]))

def evol_change(v,d,R,D):
    R1 = np.array([[(w1*(r*(KR-R)/KR-drug(d,v-evol_step)-m)+w2*(r/alpha*(KR-R)/KR-drug(d,v-evol_step)/beta-phi*m))/(w1+w2),(w3*m+w4*phi*m)/(w3+w4)],[(w1*m+w2*phi*m)/(w1+w2),(w3*(r*(KD-D)/KD-drug(d,v-evol_step)/zeta-m)+w4*(r/alpha*(KD-D)/KD-drug(d,v-evol_step)/(beta*zeta)-phi*m))/(w3+w4)]])
    R2 = np.array([[(w1*(r*(KR-R)/KR-drug(d,v+evol_step)-m)+w2*(r/alpha*(KR-R)/KR-drug(d,v+evol_step)/beta-phi*m))/(w1+w2),(w3*m+w4*phi*m)/(w3+w4)],[(w1*m+w2*phi*m)/(w1+w2),(w3*(r*(KD-D)/KD-drug(d,v+evol_step)/zeta-m)+w4*(r/alpha*(KD-D)/KD-drug(d,v+evol_step)/(beta*zeta)-phi*m))/(w3+w4)]])
    eig1 = max(real(LA.eig(R1)[0]))
    eig2 = max(real(LA.eig(R2)[0]))
    return (eig2-eig1)/(2*evol_step)

def evoLV(X, t):
    R = X[0]
    D = X[1]
    v = X[2]

    if t>200 and t<800:
        d=dose
    else:
        d=0
    
    dRdt = R*((w1*(r*(KR-R)/KR-drug(d,v)-m)+w2*(r/alpha*(KR-R)/KR-drug(d,v)/beta-phi*m))/(w1+w2))+D*(w3*m+w4*phi*m)/(w3+w4)
    dDdt = R*(w1*m+w2*phi*m)/(w1+w2)+D*(w3*(r*(KD-D)/KD-drug(d,v)/zeta-m)+w4*(r/alpha*(KD-D)/KD-drug(d,v)/(beta*zeta)-phi*m))/(w3+w4)
    
    
    dvdt = k*evol_change(v,d,R,D)
    spec(v,d,R,D)
    t_vec.append(t)
    
    dxvdt = np.array([dRdt, dDdt,dvdt])
    return dxvdt

intxv = np.array(IC)
time_sp = np.linspace(0,time,time*10)
pop = odeint(evoLV, intxv, time_sp,hmax=1)

print(pop[:, 0][-1])
print(pop[:, 1][-1])

if dose==0.4:
    title ='Low Dose'
elif dose==0.6:
    title='Medium Dose'
else:
    title='High Dose'


fig = plt.figure()
plt.subplot(211)
plt.title('$F_{Hab}$ '+title)
plt.plot(time_sp,pop[:, 0],label='R',color='b',lw=3)
plt.plot(time_sp,pop[:, 1],label='D',color='r',lw=3)

plt.legend()
plt.xlim(xmin=0)
plt.xlim(xmax=time)
plt.ylim(ymin=0)
ax = plt.gca()
ax.axvspan(200, 800, facecolor='silver')
plt.ylabel('Pop Size, x')
fig.tight_layout()
plt.subplot(212)
plt.plot(time_sp,pop[:, 2],label='Drug Resist',color='k',lw=3)
plt.xlim(xmin=0)
plt.xlim(xmax=time)
plt.ylim(ymin=0)
ax = plt.gca()
ax.axvspan(200, 800, facecolor='silver')
plt.xlabel('Time')
plt.ylabel('Indv Strategy, v')
plt.legend()
fig.tight_layout()
plt.show()