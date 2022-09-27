import numpy as np
from scipy.integrate import *
import matplotlib.pyplot as plt
from numpy import linalg as LA
from numpy import real

time = 1000

evol_step = 0.00001
k = 0.2
dose= .4

vopt = 0
st = 5
r = 0.5
m = 0.1
gN = 0.02
gP = 0.05
c = 1
KR = 100
KD = 60
alpha = 1.5
beta = 2
phi = 1.75
zeta = KR/KD

IC = [10,0,5,0,0.01] #NR,PR,ND,PD,v

spectrum = []
t_vec = []

def drug(d,v):
    return d*np.exp(-(v-vopt)**2/st)

def kth_term(matrix, k=20):
    matrix = LA.matrix_power(matrix, k)
    f_norm = LA.norm(matrix, 'fro')
    term = f_norm**(1.0/k)
    return term

def normalize(v):
    tot = sum(v)
    return v / tot


def spec(v,d,NR,PR,ND,PD):
    R = np.array([[r*(KR-NR-PR)/KR-gN-drug(d,v)-c*drug(d,v)-m, gP, m, 0],[c*drug(d,v)+gN,r/alpha*(KR-NR-PR)/KR-gP-drug(d,v)/beta-phi*m, 0, phi*m],[m,0,r*(KD-ND-PD)/KD-gN-drug(d,v)/zeta-c*drug(d,v)/zeta-m,gP],[gN+c*drug(d,v)/zeta,phi*m,0,r/alpha*(KD-ND-PD)/KD-gP-drug(d,v)/(beta*zeta)-phi*m]])
    eig = max(real(LA.eig(R)[0]))
    spectrum.append(eig)
    ind = np.argmax(eig)
    return eig

def evol_change(v,d,NR,PR,ND,PD):
    R1 = np.array([[r*(KR-NR-PR)/KR-gN-drug(d,v-evol_step)-c*drug(d,v-evol_step)-m, gP, m, 0],[c*drug(d,v-evol_step)+gN,r/alpha*(KR-NR-PR)/KR-gP-drug(d,v-evol_step)/beta-phi*m, 0, phi*m],[m,0,r*(KD-ND-PD)/KD-gN-drug(d,v-evol_step)/zeta-c*drug(d,v-evol_step)/zeta-m,gP],[gN+c*drug(d,v-evol_step)/zeta,phi*m,0,r/alpha*(KD-ND-PD)/KD-gP-drug(d,v-evol_step)/(beta*zeta)-phi*m]])
    R2 = np.array([[r*(KR-NR-PR)/KR-gN-drug(d,v+evol_step)-c*drug(d,v+evol_step)-m, gP, m, 0],[c*drug(d,v+evol_step)+gN,r/alpha*(KR-NR-PR)/KR-gP-drug(d,v+evol_step)/beta-phi*m, 0, phi*m],[m,0,r*(KD-ND-PD)/KD-gN-drug(d,v+evol_step)/zeta-c*drug(d,v+evol_step)/zeta-m,gP],[gN+c*drug(d,v+evol_step)/zeta,phi*m,0,r/alpha*(KD-ND-PD)/KD-gP-drug(d,v+evol_step)/(beta*zeta)-phi*m]])

    eig1 = max(real(LA.eig(R1)[0]))
    eig2 = max(real(LA.eig(R2)[0]))
                    
    return (eig2-eig1)/(2*evol_step)

def evoLV(X, t):
    NR = X[0]
    PR = X[1]
    ND = X[2]
    PD = X[3]
    v = X[4]

    if t>200 and t<800:
        d=dose
    else:
        d=0
    
    if NR<1 and PR<1 and ND<1 and PD<1:
        NR=PR=ND=PD=0
        
    dNRdt = r*NR*(KR-NR-PR)/KR-gN*NR-drug(d,v)*NR-c*drug(d,v)*NR+gP*PR-m*NR+m*ND
    dPRdt = r/alpha*PR*(KR-NR-PR)/KR-gP*PR-drug(d,v)*PR/beta+c*drug(d,v)*NR+gN*NR-phi*m*PR+phi*m*PD
    dNDdt = r*ND*(KD-ND-PD)/KD-gN*ND-drug(d,v)*ND/zeta-c*drug(d,v)*ND/zeta+gP*PD-m*ND+m*NR
    dPDdt = r/alpha*PD*(KD-ND-PD)/KD-gP*PD-drug(d,v)*PD/(zeta*beta)+c*drug(d,v)*ND/zeta+gN*ND-phi*m*PD+phi*m*PR
        
    dvdt = k*evol_change(v,d,NR,PR,ND,PD)
    
    spec(v,d,NR,PR,ND,PD)
    t_vec.append(t)
    
    dxvdt = np.array([dNRdt, dPRdt, dNDdt,dPDdt,dvdt])
    return dxvdt

intxv = np.array(IC)
time_sp = np.linspace(0,time,time*10)
pop = odeint(evoLV, intxv, time_sp,hmax=1)
print(normalize([pop[:, 0][-1],pop[:, 1][-1],pop[:, 2][-1],pop[:, 3][-1]]))
print(pop[:, 0][-1]+pop[:, 2][-1])
print(pop[:, 1][-1]+pop[:, 3][-1])

if dose==0.4:
    title ='Low Dose'
elif dose==0.6:
    title='Medium Dose'
else:
    title='High Dose'
    
fig = plt.figure()
plt.subplot(211)
plt.title('$MPMM(NP)$: '+title)

#plt.plot(time_sp,pop[:, 0],label='NR',color='orange',lw=3)
#plt.plot(time_sp,pop[:, 1],label='PR',color='r',lw=3)
#plt.plot(time_sp,pop[:, 2],label='ND',color='k',lw=3)
#plt.plot(time_sp,pop[:, 3],label='PD',color='b',lw=3)

#plt.plot(time_sp,pop[:, 0]+pop[:, 1],label='R',color='b',lw=3) #Fold_Hab
#plt.plot(time_sp,pop[:, 2]+pop[:, 3],label='D',color='r',lw=3)

plt.plot(time_sp,pop[:, 0]+pop[:, 2],label='N',color='b',lw=3) #Fold_Cell
plt.plot(time_sp,pop[:, 1]+pop[:, 3],label='P',color='r',lw=3)

#plt.plot(time_sp,pop[:, 0]+pop[:, 1]+pop[:, 2]+pop[:, 3],label='Total',color='k',lw=3)


plt.legend()
plt.xlim(xmin=0)
plt.xlim(xmax=time)
plt.ylim(ymin=0)
ax = plt.gca()
ax.axvspan(200, 800, facecolor='silver')
plt.ylabel('Pop Size, x')
fig.tight_layout()
plt.subplot(212)
plt.plot(time_sp,pop[:, 4],label='Drug Resist',color='k',lw=3)
plt.xlim(xmin=0)
plt.xlim(xmax=time)
plt.ylim(ymin=0)
ax = plt.gca()
ax.axvspan(200, 800, facecolor='silver')
plt.xlabel('Time')
plt.ylabel('Indv Strategy, v')
plt.ylim(ymin=0)
plt.legend()
fig.tight_layout()
plt.show()