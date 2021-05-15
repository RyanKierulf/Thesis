"""
Author: Ryan Kierulf
Date: February 26th, 2021

This program solves for the potential, electron concentration, and hole 
concentration of a silicon PN junction under non-equilibrium conditions and 
plots the potential, electric field, electron and hole concentrations, and 
conduction and valence band energies across the device. The numerical method 
used is the multiple shooting method, with the same bidirectional approach as 
in the equilibrium case. The key differences are that for the nonequilibrium 
case, the full set of drift diffusion equations must be accounted for, and the 
system is now represented as six first-order equations,rather than two. In 
addition, the boundary conditions are changed to account for a small positive 
applied voltage on the n-side of the device. The initial guess is the result 
of the equilibrium program, scaled to confine to the boundary conditions with 
the applied voltage.
"""

import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import root
from scipy import interpolate
import csv

#Physical constants
Q = 1.60217662E-19 #Charge of electron
KB = 1.380649E-23 #Boltzmann constant
KS = 11.68 #Dialetric constant, silicon
E0 = 8.85418782E-12 #Permittivity of free space
EPS = 1.05E-12 #Semiconductor dialectric constant * permittivity of free space
T = 300 #Room temperature, kelvin
NI = 1.45E10 #Intrinsic carrier concentration, silicon
BAND_GAP = 1.12 #Energy band gap for silicon at 300K
RNC = 2.8E19;  #Silicon effective density of states, condunction band
DIFF_N = 36 #Silicon electron diffusion vconstant
DIFF_P = 12 #Silicon hole diffusion constant

#Physical Quantities
eps = KS * E0
eps = 1.05E-12 #Need to fix this: calculated version of eps in previous line is incorrect
vt  = KB*T/Q #Thermal voltage
ldi = math.sqrt(eps*vt/(Q*NI)) #Intrinsic debye length
dec = vt * math.log(RNC/NI) #Difference between conduction band and fermi potential
MU_N = DIFF_N/vt #Silicon electron mobility
MU_P = DIFF_P/vt #Silicon hole mobility
TAU_N = 1E-6 #Silicon electron recombination lifetime
TAU_P = 1E-6 #Silicon hole recombination lifetime

#Device parameters
n_a = 1E16 #Doping concentration, p-side
n_d = 1E16 #Doping concentration, n-side
x_n = 500E-7 #Length of n-side, cm
x_p = 500E-7 #Length of p-side, cm
v_app = 0.00 #Applied voltage


#Boundary Conditions (scaled)
vb = vt * math.log(n_a * n_d / (NI ** 2)) #Built in voltage
v_boundary_n = (vb - v_app)/vt
n_boundary_n = n_a/NI
p_boundary_n = (1/n_boundary_n)
v_boundary_p = 0
p_boundary_p = n_d/NI
n_boundary_p = (1/p_boundary_p)


def u_n(n_bar, p_bar):
    return ((ldi ** 2) / (DIFF_N)) * (n_bar * p_bar - 1) / (TAU_N*(p_bar+1) + TAU_P*(n_bar+1))
    
    
def u_p(n_bar, p_bar):
    return ((ldi ** 2) / (DIFF_P)) * (n_bar * p_bar - 1) / (TAU_N*(p_bar+1) + TAU_P*(n_bar+1))


def equations_n_side(x, info):
    #************************************************************************#
    #Trajectory of equations on n-side of device, called by solve_ivp
    #x=0 is interface to x=x_n is n_side boundary
    #parameters: x is value of x, info is array of [V(x), dV/dx, n(x), 
    #dn/dx, p(x), dp/dx]
    #returns [dV/dx, d^2V/dx^2, dn/dx, d^2n/dx^2, dp/dx, d^p/dx^2]
    #************************************************************************#
    u1 = info[0]
    u2 = info[1]
    u3 = info[2]
    u4 = info[3]
    u5 = info[4]
    u6 = info[5]
    
    du1 = u2
    du2 = u3 - u5 - (n_d/NI)
    du3 = u4
    du4 = (u4 * u2) + (u3 * (u3 - u5 - (n_d/NI))) #+ u_n(u3, u5)
    du5 = u6
    du6 = (-u6 * u2) - (u5 * (u3 - u5 - (n_d/NI))) #+ u_p(u3, u5)
    return [du1, du2, du3, du4, du5, du6]

def equations_p_side(x, info):
    #************************************************************************#
    #Trajectory of equations on p-side of device, called by solve_ivp
    #x=0 is interface to x=x_p is p_side boundary
    #parameters: x is value of x, info is array of [V(x), dV/dx, n(x), 
    #dn/dx, p(x), dp/dx]
    #returns [dV/dx, d^2V/dx^2, dn/dx, d^2n/dx^2, dp/dx, d^p/dx^2]
    #************************************************************************#
    u1 = info[0]
    u2 = info[1]
    u3 = info[2]
    u4 = info[3]
    u5 = info[4]
    u6 = info[5]
    
    du1 = u2
    du2 = u3 - u5 + (n_a/NI)
    du3 = u4
    du4 = (u4 * u2) + (u3 * (u3 - u5 + (n_a/NI))) #+ u_n(u3, u5)
    du5 = u6
    du6 = (-u6 * u2) - (u5 * (u3 - u5 + (n_a/NI))) #+ u_p(u3, u5)
    return [du1, du2, du3, du4, du5, du6]
    
#****************************************************************************#
#For initial guess, use equilibrium result from PN_eq.py program
#Use numerical differentiation to get dV/dx, dn/dx, and dp/dx and interpolate
#data so that it can be evaluated at any x point
#****************************************************************************#
with open('equilibrium_scaled_result.csv', mode='r') as result_file:
    result_reader = csv.reader(result_file, delimiter=',')
    x_initial = next(result_reader)
    v_initial = next(result_reader)
    n_initial = next(result_reader)
    p_initial = next(result_reader)
 
#Convert data inside lists from string to floating point and add the min value
#of v to every value, since the p-side boundary value is now being used as a
#reference, that is set to zero
for i in range(len(x_initial)):
    x_initial[i] = float(x_initial[i])
    v_initial[i] = float(v_initial[i])
    n_initial[i] = float(n_initial[i])
    p_initial[i] = float(p_initial[i])
v_min = min(v_initial)
for i in range(len(x_initial)):
    v_initial[i] = v_initial[i] - v_min #set p-boundary as reference potential
                                        #equal to zero

#Piecewise linear interpolation, returns evaluatable function
v_initial = interpolate.interp1d(x_initial, v_initial, kind='linear')
n_initial = interpolate.interp1d(x_initial, n_initial, kind='linear')
p_initial = interpolate.interp1d(x_initial, p_initial, kind='linear')

#Allocate x points to define intervals
num_intervals_n_side = 20
num_intervals_p_side = 20

#Chebyshev point allocation: 
points_n = []
points_p = []
for i in range(num_intervals_n_side+1):
    points_n.append(math.cos(i*math.pi/(num_intervals_n_side*2)))
for i in range(num_intervals_p_side+1):
    points_p.append(math.cos(i*math.pi/(num_intervals_p_side*2)))
points_n[-1] = 0
points_p[-1] = 0

points_n_side = []
points_p_side = []
for i in range(len(points_n)):
    points_n_side.append((x_n/ldi) - (points_n[i] * x_n/ldi))
for i in range(len(points_p)):
    points_p_side.append((x_p/ldi) - (points_p[i] * x_p/ldi))

v_points_initial_p = []
dv_points_initial_p = []
n_points_initial_p = []
dn_points_initial_p = []
p_points_initial_p = []
dp_points_initial_p = []

#Initial guess values for dependent variables, p-side of device
for i in range(len(points_p_side) - 1):
    dx = 0.00001
    v_points_initial_p.append(v_initial(points_p_side[i]+(x_n/ldi)))
    dv_points_initial_p.append((v_initial(points_p_side[i]+(x_n/ldi)) - 
        v_initial(points_p_side[i]+(x_n/ldi)-dx))/dx)
    n_points_initial_p.append(n_initial(points_p_side[i]+(x_n/ldi)))
    dn_points_initial_p.append((n_initial(points_p_side[i]+(x_n/ldi)) - 
        n_initial(points_p_side[i]+(x_n/ldi)-dx))/dx)
    p_points_initial_p.append(p_initial(points_p_side[i]+(x_n/ldi)))
    dp_points_initial_p.append((p_initial(points_p_side[i]+(x_n/ldi)) - 
        p_initial(points_p_side[i]+(x_n/ldi)-dx))/dx)    

v_points_initial_n = []
dv_points_initial_n = []
n_points_initial_n = []
dn_points_initial_n = []
p_points_initial_n = []
dp_points_initial_n = []

#Initial guess values for dependent variables, n-side of device
for i in range(len(points_n_side) - 1):
    dx = 0.00001
    v_points_initial_n.append(v_initial((x_p/ldi)-points_n_side[i]))
    dv_points_initial_n.append((v_initial((x_p/ldi)-points_n_side[i]) - 
        v_initial((x_p/ldi)-points_n_side[i]+dx))/dx)
    n_points_initial_n.append(n_initial((x_p/ldi)-points_n_side[i]))
    dn_points_initial_n.append((n_initial((x_p/ldi)-points_n_side[i]) - 
        n_initial((x_p/ldi)-points_n_side[i]+dx))/dx)
    p_points_initial_n.append(p_initial((x_p/ldi)-points_n_side[i]))
    dp_points_initial_n.append((p_initial((x_p/ldi)-points_n_side[i]) - 
        p_initial((x_p/ldi)-points_n_side[i]+dx))/dx)

#****************************************************************************#
#List of all variables that must be solved for: contains values of dependent
#variables on the p-side, followed by their values on the n-side, stored in
#group order from x=0 at the interface to each boundary, where each group
#contains v, dv/dx, n, dn/dx, p, dp/dx
#****************************************************************************#

var_list = np.zeros(6 * (num_intervals_n_side + num_intervals_p_side))
for i in range(num_intervals_p_side):
    var_list[6*i] = v_points_initial_p[i]
    var_list[(6*i)+1] = dv_points_initial_p[i]
    var_list[(6*i)+2] = n_points_initial_p[i]
    var_list[(6*i)+3] = dn_points_initial_p[i]
    var_list[(6*i)+4] = p_points_initial_p[i]
    var_list[(6*i)+5] = dp_points_initial_p[i]
offset = 6 * num_intervals_p_side
for i in range(num_intervals_n_side):
    var_list[(6*i)+offset] = v_points_initial_n[i]
    var_list[(6*i)+1+offset] = dv_points_initial_n[i]
    var_list[(6*i)+2+offset] = n_points_initial_n[i]
    var_list[(6*i)+3+offset] = dn_points_initial_n[i]
    var_list[(6*i)+4+offset] = p_points_initial_n[i]
    var_list[(6*i)+5+offset] = dp_points_initial_n[i]
    
def calculate_f_list(var_list):
    f_list = np.zeros(6 * (num_intervals_n_side + num_intervals_p_side))   
    #Continuity equations p-side:
    #On the p-side, p and dp/dx will be very large values, but all functions
    #to be solved for will be subject to the same tolerance value, so divide
    #the continuity equations for p and dp/dx on the p-side by 1000 (same
    #thing is done on n-side)
    for i in range(num_intervals_p_side-1):
        interval_start_values = [var_list[6*i], var_list[(6*i)+1], var_list[(6*i)+2],
            var_list[(6*i)+3], var_list[(6*i)+4], var_list[(6*i)+5]]
        sol = solve_ivp(equations_p_side, (points_p_side[i], points_p_side[i+1]),
            interval_start_values, method = 'BDF')
        f_list[6*i] = sol.y[0,-1] - var_list[(6*i)+6]
        f_list[(6*i)+1] = sol.y[1,-1] - var_list[(6*i)+7]
        f_list[(6*i)+2] = sol.y[2,-1] - var_list[(6*i)+8]
        f_list[(6*i)+3] = sol.y[3,-1] - var_list[(6*i)+9]
        f_list[(6*i)+4] = (sol.y[4,-1] - var_list[(6*i)+10])/100
        f_list[(6*i)+5] = (sol.y[5,-1] - var_list[(6*i)+11])/100
    #Boundary conditions p-side
    interval_start_values = [var_list[6*(num_intervals_p_side-1)], 
        var_list[6*(num_intervals_p_side-1)+1], var_list[6*(num_intervals_p_side-1)+2],
        var_list[6*(num_intervals_p_side-1)+3], var_list[6*(num_intervals_p_side-1)+4],
        var_list[6*(num_intervals_p_side-1)+5]]
    f_list[6*(num_intervals_p_side-1)] = solve_ivp(equations_p_side, 
        (points_p_side[num_intervals_p_side-1], points_p_side[num_intervals_p_side]), 
        interval_start_values, method = 'BDF').y[0,-1] - v_boundary_p
    f_list[6*(num_intervals_p_side-1)+1] = solve_ivp(equations_p_side, 
        (points_p_side[num_intervals_p_side-1], points_p_side[num_intervals_p_side]), 
        interval_start_values, method = 'BDF').y[2,-1] - n_boundary_p
    f_list[6*(num_intervals_p_side-1)+2] = (solve_ivp(equations_p_side, 
        (points_p_side[num_intervals_p_side-1], points_p_side[num_intervals_p_side]), 
        interval_start_values, method = 'BDF').y[4,-1] - p_boundary_p)/100    
    #Continuity equations at interface
    f_list[6*(num_intervals_p_side-1)+3] = var_list[0] - var_list[6*num_intervals_p_side]
    f_list[6*(num_intervals_p_side-1)+4] = var_list[1] + var_list[6*num_intervals_p_side+1]
    f_list[6*(num_intervals_p_side-1)+5] = var_list[2] - var_list[6*num_intervals_p_side+2]
    f_list[6*(num_intervals_p_side-1)+6] = var_list[3] + var_list[6*num_intervals_p_side+3]
    f_list[6*(num_intervals_p_side-1)+7] = var_list[4] - var_list[6*num_intervals_p_side+4]
    f_list[6*(num_intervals_p_side-1)+8] = var_list[5] + var_list[6*num_intervals_p_side+5]
    #Continuity equations n-side
    f_offset = (6*num_intervals_p_side)+3 #243
    v_offset = (6*num_intervals_p_side) #240
    for i in range(num_intervals_n_side-1):
        interval_start_values = [var_list[(6*i)+v_offset], var_list[(6*i)+1+v_offset], 
            var_list[(6*i)+2+v_offset],var_list[(6*i)+3+v_offset], var_list[(6*i)+4+v_offset], 
            var_list[(6*i)+5+v_offset]]
        sol = solve_ivp(equations_n_side, (points_n_side[i], points_n_side[i+1]),
            interval_start_values, method = 'BDF')
        f_list[(6*i)+f_offset] = sol.y[0,-1] - var_list[(6*i)+6+v_offset]
        f_list[(6*i)+1+f_offset] = sol.y[1,-1] - var_list[(6*i)+7+v_offset]
        f_list[(6*i)+2+f_offset] = (sol.y[2,-1] - var_list[(6*i)+8+v_offset])/100
        f_list[(6*i)+3+f_offset] = (sol.y[3,-1] - var_list[(6*i)+9+v_offset])/100
        f_list[(6*i)+4+f_offset] = sol.y[4,-1] - var_list[(6*i)+10+v_offset]
        f_list[(6*i)+5+f_offset] = sol.y[5,-1] - var_list[(6*i)+11+v_offset]
    #Boundary conditions n-side
    interval_start_values = interval_start_values = [var_list[6*(num_intervals_n_side-1)+v_offset], 
        var_list[6*(num_intervals_p_side-1)+1+v_offset], var_list[6*(num_intervals_n_side-1)+2+v_offset],
        var_list[6*(num_intervals_p_side-1)+3+v_offset], var_list[6*(num_intervals_n_side-1)+4+v_offset],
        var_list[6*(num_intervals_p_side-1)+5+v_offset]]
    f_list[-3] = solve_ivp(equations_n_side,
        (points_n_side[num_intervals_n_side-1], points_n_side[num_intervals_n_side]), 
        interval_start_values, method = 'BDF').y[0,-1] - v_boundary_n
    f_list[-2] = (solve_ivp(equations_n_side,
        (points_n_side[num_intervals_n_side-1], points_n_side[num_intervals_n_side]), 
        interval_start_values, method = 'BDF').y[2,-1] - n_boundary_n)/100
    f_list[-1] = solve_ivp(equations_n_side,
        (points_n_side[num_intervals_n_side-1], points_n_side[num_intervals_n_side]), 
        interval_start_values, method = 'BDF').y[4,-1] - p_boundary_n
    return f_list

#Solve all necessary equations:
solution = root(calculate_f_list, var_list, method='hybr') #Print solution for more info
var_list = solution.x

def plot_solution(var_list, v_app):
    x_p_side = []
    v_p_side = []
    dv_p_side = []
    n_p_side = []
    dn_p_side = []
    p_p_side = []
    dp_p_side = []
    
    for i in range(num_intervals_p_side):
        interval_start_values = [var_list[6*i], var_list[(6*i)+1], var_list[(6*i)+2],
            var_list[(6*i)+3], var_list[(6*i)+4], var_list[(6*i)+5]]
        sol = solve_ivp(equations_p_side, (points_p_side[i], points_p_side[i+1]),
            interval_start_values, method = 'BDF')
        for j in range(sol.t.shape[0]):
            x_p_side.append(sol.t[j])
            v_p_side.append(sol.y[0,j])
            dv_p_side.append(sol.y[1,j])
            n_p_side.append(sol.y[2,j])
            dn_p_side.append(sol.y[3,j])
            p_p_side.append(sol.y[4,j])
            dp_p_side.append(sol.y[5,j])
            
    x_n_side = []
    v_n_side = []
    dv_n_side = []
    n_n_side = []
    p_n_side = []
    dn_n_side = []
    dp_n_side = []
    offset = 6 * num_intervals_p_side
    
    for i in range(num_intervals_n_side):
        interval_start_values = [var_list[(6*i)+offset], var_list[(6*i)+1+offset], 
            var_list[(6*i)+2+offset], var_list[(6*i)+3+offset], 
            var_list[(6*i)+4+offset], var_list[(6*i)+5+offset]]
        sol = solve_ivp(equations_n_side, (points_n_side[i], points_n_side[i+1]),
            interval_start_values, method = 'BDF')
        for j in range(sol.t.shape[0]):
            x_n_side.append(sol.t[j])
            v_n_side.append(sol.y[0,j])
            dv_n_side.append(sol.y[1,j])
            n_n_side.append(sol.y[2,j])
            dn_n_side.append(sol.y[3,j])
            p_n_side.append(sol.y[4,j])
            dp_n_side.append(sol.y[5,j])
            
    #Combine p-side and n-side data
    x_n_side.reverse()
    v_n_side.reverse()
    dv_n_side.reverse()
    n_n_side.reverse()
    p_n_side.reverse()
    dn_n_side.reverse()
    dp_n_side.reverse()
    
    final_x = []
    final_v = []
    final_electric_field = []
    final_p = []
    final_n = []
    final_dn = []
    final_dp = []
    
    for i in range(len(x_n_side)):
        x_n_side[i] = (x_n/ldi) - x_n_side[i]
        final_x.append(x_n_side[i] * ldi)
        final_v.append(v_n_side[i] * vt)
        final_electric_field.append(dv_n_side[i] * vt/ldi)
        final_p.append(p_n_side[i] * NI)
        final_n.append(n_n_side[i] * NI)
        final_dp.append(-dp_n_side[i] * NI/ldi)
        final_dn.append(-dn_n_side[i] * NI/ldi)
    for i in range(len(x_p_side)):
        x_p_side[i] = x_p_side[i] + (x_n/ldi)
        final_x.append(x_p_side[i] * ldi)
        final_v.append(v_p_side[i] * vt)
        final_electric_field.append(-dv_p_side[i] * vt/ldi)
        final_p.append(p_p_side[i] * NI)
        final_n.append(n_p_side[i] * NI)
        final_dp.append(dp_p_side[i] * NI/ldi)
        final_dn.append(dn_p_side[i] * NI/ldi)
            
    fig, ax1 = plt.subplots(3)
    fig.set_figheight(10)
    fig.set_figwidth(15)
    
    ax1[0].plot(final_x, final_p, 'r-', label = 'p')
    ax1[0].plot(final_x, final_n, label = 'n')
    ax1[0].legend(loc = 'upper left')
    ax1[0].set_title('Applied Bias: ' + str(v_app))
    ax1[0].set_xlabel('Position (nm)')
    ax1[0].set_ylabel('Electron and Hole Concentrations (cm^-1)')
    ax1[0].set_yscale('log')
    ax1[1].plot(final_x, final_v)
    ax1[1].set_xlabel('Position (nm)')
    ax1[1].set_ylabel('Electric Potential (V)')
    ax1[2].plot(final_x, final_electric_field)
    ax1[2].set_xlabel('Position (nm)')
    ax1[2].set_ylabel('Electric Field (V/cm)')
      
    
plot_solution(var_list, v_app)
