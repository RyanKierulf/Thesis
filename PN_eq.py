"""
Author: Ryan Kierulf
Date: January 20th, 2021

This program solves for the potential of a silicon PN junction under equilibirum conditions, and plots the potential, electric field, electron and hole concentrations, and conduction and valence band energies 
across the device. The method used to solve the poisson-equation two-point boundary value problem is the multiple shooting method with a bidirectional shooting approach; x=0 is set as the interface between 
the p-side and n-side, and numerical integration is used to go from x=0 to the p-side boundary and from x=0 to the n-side boundary, with a specified number of intervals (I used 20) on each the p and n-side. 
Conditions enforced to form a solution are: continuity of V(x) and dV/dx on each side of the device, continuity of V(x) and dV/dx at the interface, and satisfaction of the dirilecht boundary conditions. 
Scipy uses MINPACK (old fortran code that uses a modified Powell method) to solve the system of non-linear equations.
"""

import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import root

#Physical constants
q   = 1.602E-19 #Charge of electron
kb  = 1.38E-23 #Boltzmann constant
eps = 1.05E-12 #Semiconductor dialectric constant * permittivity of free space
T   = 300 #Room temperature, Kelvin
ni  = 1.5E10 #Intrinsic carrier concentration, silicon
Vt  = kb*T/q #Thermal voltage
Ldi = math.sqrt(eps*Vt/(q*ni)) #Intrinsic debye length
RNc = 2.8E19;  #Silicon effective density of states, condunction band
dEc = Vt * math.log(RNc/ni)
band_gap = 1.12 #Energy band gap for silicon at 300K

#Device parameters
Na = 1E17 #Doping concentration, p-side
Nd = 1E17 #Doping concentration, n-side
x_n = 500E-7 #Length of n-side, cm
x_p = 500E-7 #Length of p-side, cm
total_length = x_n + x_p
v_diff = Vt * math.log(Na * Nd / (ni ** 2)) #Built-in voltage
v_boundary_n = v_diff/2 #Boundary condition on n-side of device
v_boundary_p = -v_diff/2 #Boundary condition on p-side of device

#Standard variable scaling for semiconductor modeling is used: x = x/Ldi, V = V/Vt
v_boundary_n = v_boundary_n / Vt
v_boundary_p = v_boundary_p / Vt
total_length = (x_n + x_p)/Ldi

def poisson_equation_n_side(x_info, V_info): #x=0 is interface to x=x_n is n_side boundary
    V = V_info[0]
    dV = V_info[1]
    V_deriv = dV
    dV_deriv = math.exp(V) - math.exp(-V) - (Nd/ni)
    return [V_deriv, dV_deriv]
   
def poisson_equation_p_side(x_info, V_info): #x=0 is interface to x=x_p is p_side boundary
    V = V_info[0]
    dV = V_info[1]
    V_deriv = dV
    dV_deriv = math.exp(V) - math.exp(-V) + (Na/ni)
    return [V_deriv, dV_deriv]

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
    points_n_side.append((x_n/Ldi) - (points_n[i] * x_n/Ldi))
for i in range(len(points_p)):
    points_p_side.append((x_p/Ldi) - (points_p[i] * x_p/Ldi))

#Initial Guess p-side:
#Guess that potential is zero at interface, exponentially decays to negative boundary value and has decreased by 75% when 1/10 distance across the p_side of the device
#Only two points are needed to fit exponential function y = a * Exp(-b * x)
a_p = -v_boundary_p
b_p = math.log(0.25) / (-x_p / (Ldi * 10))

def initial_guess_V_p_side(x):
    return (a_p * math.exp(-b_p * x) - a_p)

def initial_guess_dV_p_side(x):
    return (-a_p * b_p * math.exp(-b_p * x))

V_points_initial_p = []
dV_points_initial_p = []
for i in range(len(points_p_side)):
    V_points_initial_p.append(initial_guess_V_p_side(points_p_side[i]))
    dV_points_initial_p.append(initial_guess_dV_p_side(points_p_side[i]))
 
#Initial Guess n-side:
#Guess same shape, but decaying from zero to positive boundary value, going from 0 at x=0 to v_boundary_n at x = x_n
a_n = v_boundary_n
b_n = math.log(0.25) / (-x_n / (Ldi * 10))

def initial_guess_V_n_side(x):
    return (-a_n * math.exp(-b_n * x) + a_p)

def initial_guess_dV_n_side(x):
    return (a_n * b_n * math.exp(-b_n * x))

V_points_initial_n = []
dV_points_initial_n = []
for i in range(len(points_n_side)):
    V_points_initial_n.append(initial_guess_V_n_side(points_n_side[i]))
    dV_points_initial_n.append(initial_guess_dV_n_side(points_n_side[i]))
 
#Uncomment to plot initial V and dV, and piecewise IVP solutions resulting from initial guess
#plt.plot(points_p_side, V_points_initial_p)
#plt.plot(points_p_side, dV_points_initial_p)
#plt.plot(points_n_side, V_points_initial_n)
#plt.plot(points_n_side, dV_points_initial_n)
    
#for i in range(num_intervals_p_side):
#   sol = solve_ivp(poisson_equation_p_side, (points_p_side[i], points_p_side[i+1]), [V_points_initial_p[i], dV_points_initial_p[i]], method = 'BDF')
#   plt.plot(sol.t, sol.y[0,:])
    
#for i in range(num_intervals_n_side):
#   sol = solve_ivp(poisson_equation_n_side, (points_n_side[i], points_n_side[i+1]), [V_points_initial_n[i], dV_points_initial_n[i]], method = 'BDF')
#   plt.plot(sol.t, sol.y[0,:])
    
#List of all variables that must be solved for
#var_list[0: 2*num_intervals_p_side] = [V[points_p_side[0]], dV[points_p_side[0]], V[points_p_side[1], dV[points_p_side[1]], ... up to but not including points at boundary]
#var_list[2*num_intervals_p_side: 2*(num_intervals_p_side+num_intervals_n_side)] = [V[points_n_side[0]], dV[points_n_side[0]], V[points_n_side[1], dV[points_n_side[1]], ... up to but not including points at boundary]
#Values of list of variables, based on initial guess:
var_list = np.zeros(2 * (num_intervals_p_side + num_intervals_n_side)) 
for i in range(num_intervals_p_side + 1):
    var_list[(2*i)-2] = V_points_initial_p[i]
    var_list[(2*i)-1] = dV_points_initial_p[i]
offset = num_intervals_p_side * 2
for i in range(1, num_intervals_n_side + 1):
    var_list[(2*i)-2+offset] = V_points_initial_n[i]
    var_list[(2*i)-1+offset] = dV_points_initial_n[i]    

#List of functions that must equal zero, from subtracting RHS from each equation
#Solve for var_list such that calculate_f_list returns vector of zeros    
def calculate_f_list(var_list):
    f_list = np.zeros(2 * (num_intervals_p_side + num_intervals_n_side)) 
    #Continuity equations p-side:
    for i in range(num_intervals_p_side-1):
        f_list[(2*i)] = solve_ivp(poisson_equation_p_side, (points_p_side[i], points_p_side[i+1]), [var_list[(2*i)], var_list[(2*i)+1]], method = 'BDF').y[0,-1] - var_list[(2*i)+2]
        f_list[(2*i)+1] = solve_ivp(poisson_equation_p_side, (points_p_side[i], points_p_side[i+1]), [var_list[(2*i)], var_list[(2*i)+1]], method = 'BDF').y[1,-1] - var_list[(2*i)+3]
    #Boundary condition p-side:
    f_list[(2*num_intervals_p_side)-2] = solve_ivp(poisson_equation_p_side, (points_p_side[num_intervals_p_side-1], points_p_side[num_intervals_p_side]), [var_list[(2*num_intervals_p_side)-2], var_list[(2*num_intervals_p_side)-1]], method = 'BDF').y[0,-1] - v_boundary_p
    #Continuity equations at interface
    f_list[(2*num_intervals_p_side)-1] = var_list[0] - var_list[(2*num_intervals_p_side)]
    f_list[(2*num_intervals_p_side)] = var_list[1] + var_list[(2*num_intervals_p_side)+1]
    #Continuity equations n-side:
    offset = (2*num_intervals_p_side) + 1
    for i in range(num_intervals_n_side-1):
        f_list[(2*i)+offset] = solve_ivp(poisson_equation_n_side, (points_n_side[i], points_n_side[i+1]), [var_list[(2*i)+offset-1], var_list[(2*i)+offset]], method = 'BDF').y[0,-1] - var_list[(2*i)+offset+1]
        f_list[(2*i)+1+offset] = solve_ivp(poisson_equation_n_side, (points_n_side[i], points_n_side[i+1]), [var_list[(2*i)+offset-1], var_list[(2*i)+offset]], method = 'BDF').y[1,-1] - var_list[(2*i)+offset+2]
    #Boundary condition n-side
    f_list[2*(num_intervals_n_side+num_intervals_p_side)-1] = solve_ivp(poisson_equation_n_side, (points_n_side[num_intervals_n_side-1], points_n_side[num_intervals_n_side]), [var_list[2*(num_intervals_p_side+num_intervals_n_side)-2], var_list[2*(num_intervals_p_side+num_intervals_n_side)-1]], method = 'BDF').y[0,-1] - v_boundary_n
    return f_list

#Now solve:
solution = root(calculate_f_list, var_list) #Print solution for more info about method, number of function evaluations, etc.
var_list = solution.x

#Now that all variables have been solved for, solve initial value problems in each interval with correct parameters, combine data, and plot:
x_p_side = []
V_p_side = []
dV_p_side = []
x_n_side = []
V_n_side = []
dV_n_side = []

for i in range(num_intervals_p_side):
    sol = solve_ivp(poisson_equation_p_side, (points_p_side[i], points_p_side[i+1]), [var_list[(2*i)], var_list[(2*i)+1]], method = 'BDF')
    for j in range(sol.t.shape[0]):
        x_p_side.append(sol.t[j])
        V_p_side.append(sol.y[0,j])
        dV_p_side.append(sol.y[1,j])
offset = 2*num_intervals_p_side
for i in range(num_intervals_n_side):
    sol = solve_ivp(poisson_equation_n_side, (points_n_side[i], points_n_side[i+1]), [var_list[(2*i)+offset], var_list[(2*i)+offset+1]], method = 'BDF')
    for j in range(sol.t.shape[0]):
        x_n_side.append(sol.t[j])
        V_n_side.append(sol.y[0,j])
        dV_n_side.append(sol.y[1,j])

x_n_side.reverse()
V_n_side.reverse()
dV_n_side.reverse()

final_x = []
final_V = []
final_dV = []

#Combine data from p-side and n-side
for i in range(len(x_n_side)):
    x_n_side[i] = (x_n/Ldi) - x_n_side[i]
    final_x.append(x_n_side[i])
    final_V.append(V_n_side[i])
    final_dV.append(-dV_n_side[i])
for i in range(len(x_p_side)):
    x_p_side[i] = x_p_side[i] + (x_n/Ldi)
    final_x.append(x_p_side[i])
    final_V.append(V_p_side[i])
    final_dV.append(dV_p_side[i])    

#Unscale variables
for i in range(len(final_x)):
    final_x[i] = final_x[i] * Ldi
    final_x[i] = final_x[i] * 1E7 #convert from cm to nm
    final_V[i] = final_V[i] * Vt
    final_dV[i] = final_dV[i] * (Vt/Ldi)

E = [] #Electric Field = -dV/dx
n = [] #Electron concentration, approximated by ni * e^(V/Vt), which is valid for device in equilibrium
p = [] #Hole concentration, ni * e^(-V/Vt)
c_band = [] #Conduction band
v_band = [] #Valence band
for i in range(len(final_x)):
    E.append(-final_dV[i])
    n.append(ni * math.exp(final_V[i]/Vt))
    p.append(ni * math.exp(-final_V[i]/Vt))    
    c_band.append(dEc - final_V[i])
    v_band.append(c_band[i] - band_gap)


fig, ax1 = plt.subplots(2)
fig.set_figheight(10)
fig.set_figwidth(15)
    
ax1[0].plot(final_x, final_V)
ax1[0].set_xlabel('Position (nm)')
ax1[0].set_ylabel('Electric Potential (V)')
ax1[1].plot(final_x, E)
ax1[1].set_xlabel('Position (nm)')
ax1[1].set_ylabel('Electric Field (V/cm)')

fig, ax2 = plt.subplots(2)
fig.set_figheight(10)
fig.set_figwidth(15)

ax2[0].plot(final_x, n, label = 'n')
ax2[0].plot(final_x, p, 'r-', label = 'p')
ax2[0].legend(loc = 'upper left')
ax2[0].set_yscale('log')
ax2[0].set_xlabel('Position (nm)')
ax2[0].set_ylabel('Electron and Hole Concentrations (cm^-1)')
ax2[1].plot(final_x, c_band, label = 'Conduction Band')
ax2[1].plot(final_x, v_band, 'm-', label = 'Valence Band')
ax2[1].legend(loc = 'upper left')
ax2[1].set_xlabel('Position (nm)')
ax2[1].set_ylabel('Band Energies (eV)')
