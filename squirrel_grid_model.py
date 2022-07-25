# -*- coding: utf-8 -*-
"""
Created on Sun Nov 29 16:35:19 2020

@author: Z,S,T,M
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint 

b = 0.4 # Natural mortality rate (per year)
beta = 0.7 # Virus transmission rate
mew = 0.2 # Dispersal fraction to neighbouring tiles (After 1 year)
n = 80 # Number of years
t = np.linspace(0, n, 200) # A grid of time points (in days)

aR = 1.0 # Reds max reproductive rate
KR = 60 # Reds carrying capacity
alpha = 26 # Reds virus mortality rate
sigmaR = 0.61 # Reds competition effect (on Greys)
qR = (aR - b)/KR # Reds rate of susceptibility to crowding

aG = 1.2 # Greys max reproductive rate
KG = 80 # Greys carrying capacity
gamma = 13 # Virus recovery rate
sigmaG = 1.65 # Greys competition effect (on Reds)
qG = (aG - b)/KG # Greys rate of susceptibility to crowding


H = [2,60] # Total population [Grey,Red]
HG, HR = H
RG0 = 0 # Recovered Greys initially 0

IG0 = 2 # Infected Greys initially
SG0 = HG - IG0 - RG0 # Susceptible Greys 

IR0 = 0 # Infected Reds initially 0
SR0 = HR - IR0 # Susceptible Reds 


# The L-V/SIR model differential equations.
def deriv(y, t):
    SG, IG, RG, SR, IR = y
    HG = SG + IG + RG
    HR = SR + IR
    dSGdt = (aG-qG*(HG+sigmaG*HR))*HG - b*SG - beta*SG*(IG+IR)
    dIGdt = beta*SG*(IG+IR)-b*IG-gamma*IG
    dRGdt = gamma*IG - b*RG
    dSRdt = (aR-qR*(HR+sigmaG*HG))*HR - b*SR - beta*SR*(IR+IG)
    dIRdt = beta*SR*(IG+IR) - (alpha+b)*IR 
    return dSGdt, dIGdt, dRGdt, dSRdt, dIRdt

# Initial conditions vector
y0 = SG0, IG0, RG0, SR0, IR0
ret = odeint(deriv, y0, t)
SG, IG, RG, SR, IR = ret.T

grey_total = SG + IG + RG
red_total = SR + IR



'''Grid Model'''

rdim = 10 # 11 rows
cdim = 10

yt = odeint(deriv, (0, 0, 0, 60, 0), [t[0]]).flatten()
 # Creates a yt for each timestamp

yt_grid = np.zeros(((rdim+1),(cdim+1),5))
for i in range(rdim+1):
    for j in range(cdim+1):
        yt_grid[i,j] = yt

yt_grid[0,0] = np.array([0, 2, 0, 60, 0])

SG1, SR1 = [2], [60*121]

for i in range(1,len(t)):        # len(t)
    if np.floor(t[i]) - np.floor(t[i-1]) == 1: # Checks if end of yr
        yt_prop = yt_grid * mew * 0.125
        yt_changeover = np.zeros(((rdim+1),(cdim+1),5))
        print('Year:', int(t[i]))
        for r in range(rdim+1):
            for c in range(cdim+1):
                if r == c == 0:
                    #print("Top Left Corner")
                    yt_changeover[r,c] += - yt_prop[r,c]*3
                    yt_changeover[r,c+1] += yt_prop[r,c]
                    yt_changeover[r+1,c+1] += yt_prop[r,c]
                    yt_changeover[r+1,c] += yt_prop[r,c] 
                elif r==0 and c==rdim:
                    #print("Top Right Corner")
                    yt_changeover[r,c] += - yt_prop[r,c]*3
                    yt_changeover[r,c-1] += yt_prop[r,c]
                    yt_changeover[r+1,c-1] += yt_prop[r,c]
                    yt_changeover[r+1,c] += yt_prop[r,c]                 
                elif r==cdim and c==0:
                    #print("Bottom Left Corner")
                    yt_changeover[r,c] += - yt_prop[r,c]*3
                    yt_changeover[r,c+1] += yt_prop[r,c]
                    yt_changeover[r-1,c+1] += yt_prop[r,c]
                    yt_changeover[r-1,c] += yt_prop[r,c]                  
                elif r == c == rdim:
                    #print("Bottom Right Corner")
                    yt_changeover[r,c] += - yt_prop[r,c]*3
                    yt_changeover[r,c-1] += yt_prop[r,c]
                    yt_changeover[r-1,c-1] += yt_prop[r,c]
                    yt_changeover[r-1,c] += yt_prop[r,c]                    
                elif r == 0:
                    #print("Top Row")
                    yt_changeover[r,c] += -yt_prop[r,c]*5
                    yt_changeover[r,c+1] += yt_prop[r,c]
                    yt_changeover[r+1,c+1] += yt_prop[r,c]
                    yt_changeover[r+1,c] += yt_prop[r,c]
                    yt_changeover[r+1,c-1] += yt_prop[r,c]
                    yt_changeover[r,c-1] += yt_prop[r,c]
                elif c == 0:
                    #print("Left Column")
                    yt_changeover[r,c] += -yt_prop[r,c]*5
                    yt_changeover[r-1,c] += yt_prop[r,c]
                    yt_changeover[r-1,c+1] += yt_prop[r,c]
                    yt_changeover[r,c+1] += yt_prop[r,c]
                    yt_changeover[r+1,c+1] += yt_prop[r,c]
                    yt_changeover[r+1,c] += yt_prop[r,c]                    
                elif r == rdim:
                    #print("Bottom row")
                    yt_changeover[r,c] += -yt_prop[r,c]*5
                    yt_changeover[r,c-1] += yt_prop[r,c]
                    yt_changeover[r-1,c-1] += yt_prop[r,c]
                    yt_changeover[r-1,c] += yt_prop[r,c]
                    yt_changeover[r-1,c+1] += yt_prop[r,c]
                    yt_changeover[r,c+1] += yt_prop[r,c]                    
                elif c == cdim:
                    #print("Right Column")
                    yt_changeover[r,c] += -yt_prop[r,c]*5
                    yt_changeover[r-1,c] += yt_prop[r,c]
                    yt_changeover[r-1,c-1] += yt_prop[r,c]
                    yt_changeover[r,c-1] += yt_prop[r,c]
                    yt_changeover[r+1,c-1] += yt_prop[r,c]
                    yt_changeover[r+1,c] += yt_prop[r,c]                    
                else:
                    #print("Central Squares")
                    yt_changeover[r,c] += -yt_prop[r,c]*8
                    yt_changeover[r-1,c] += yt_prop[r,c]
                    yt_changeover[r-1,c+1] += yt_prop[r,c]
                    yt_changeover[r,c+1] += yt_prop[r,c]
                    yt_changeover[r+1,c+1] += yt_prop[r,c]
                    yt_changeover[r+1,c] += yt_prop[r,c]
                    yt_changeover[r+1,c-1] += yt_prop[r,c]
                    yt_changeover[r,c-1] += yt_prop[r,c]
                    yt_changeover[r-1,c-1] += yt_prop[r,c]
                    
        yt_grid += yt_changeover
        
    for p in range(rdim+1):
        for q in range(cdim+1):
            yt_grid[p,q] = odeint(deriv, yt_grid[p,q].flatten(), [t[i-1], t[i]])[1]
    # print(yt_grid[0,0,3], ret[i,3])
    # Output here compares S reds.
    # Left output is grid model, right is single square
    SG1 = np.append(SG1, [sum(sum(yt_grid[0:,0:,0]) + sum(yt_grid[0:,0:,1]) + sum(yt_grid[0:,0:,2]))])
    SR1 =  np.append(SR1, [sum(sum(yt_grid[0:,0:,3]) + sum(yt_grid[0:,0:,4]))])






fig, ax = plt.subplots(ncols=2,sharey=True)
ax[1].plot(t, SG1/121, 'black', alpha=0.7, lw=2, label='Grey Population')
ax[1].plot(t, SR1/121, 'red', alpha=0.8, lw=2, label='Red Population')
ax[0].plot(t, grey_total, 'black', alpha=0.7, lw=2, label='Grey Population')
ax[0].plot(t, red_total, 'red', alpha=0.8, lw=2, label='Red Population')
ax[0].set_box_aspect(1)
ax[1].set_box_aspect(1)
ax[0].set_xlabel('Time (years)')
ax[1].set_xlabel('Time (years)')
ax[0].set_ylabel('Population')
ax[0].set_title('Simple Square Model')
ax[1].set_title('Grid Model')
fig.suptitle('Graphs comparing the time to red squirrel exclusion \n in the two models.', fontsize=16)
fig.tight_layout()
legend1 = ax[1].legend(loc='upper left', fontsize=9)
legend1.get_frame().set_alpha(0.5)
legend0 = ax[0].legend(loc='right', fontsize=9)
legend0.get_frame().set_alpha(0.5)
plt.show()




