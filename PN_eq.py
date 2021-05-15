"""
Author: Ryan Kierulf
Date: January 20th, 2021

This program solves for the potential of a silicon PN junction under 
equilibirum conditions, and plots the potential, electric field, electron and 
hole concentrations, and conduction and valence band energies across the 
device. The method used to solve the poisson-equation two-point boundary value 
problem is the multiple shooting method with a bidirectional shooting 
approach; x=0 is set as the interface between the p-side and n-side, and 
numerical integration is used to go from x=0 to the p-side boundary and from 
x=0 to the n-side boundary, with a specified number of intervals (I used 20) 
on each the p and n-side. Conditions enforced to form a solution are: 
continuity of V(x) and dV/dx on each side of the device, continuity of V(x) 
and dV/dx at the interface, and satisfaction of the dirilecht boundary 
conditions. Scipy uses MINPACK (old fortran code that uses a modified Powell 
method) to solve the system of non-linear equations.
"""

import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import root
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

#Physical Quantities
eps = KS * E0
eps = 1.05E-12 #Need to fix this: calculated version of eps in previous line is incorrect
vt  = KB*T/Q #Thermal voltage
ldi = math.sqrt(eps*vt/(Q*NI)) #Intrinsic debye length
dec = vt * math.log(RNC/NI) 

#Device parameters
n_a = 1E16 #Doping concentration, p-side
n_d = 1E16 #Doping concentration, n-side
x_n = 500E-7 #Length of n-side, cm
x_p = 500E-7 #Length of p-side, cm

#Boundary conditions (scaled)
v_boundary_n = math.log(n_a/NI) #Boundary condition on n-side of device
v_boundary_p = -math.log(n_d/NI) #Boundary condition on p-side of device

def poisson_equation_n_side(x, v_info):
    #***********************************************************************#
    #poisson equation trajectory on n-side of device, called by solve_ivp
    #x=0 is interface to x=x_n is n_side boundary
    #parameters: x is value of x, v_info is [V(x), dV/dx]
    #returns [dV/dx, d^2V/dx^2]
    #***********************************************************************#
    v = v_info[0]
    dV = v_info[1]
    v_deriv = dV
    dv_deriv = math.exp(v) - math.exp(-v) - (n_d/NI)
    return [v_deriv, dv_deriv]
   
def poisson_equation_p_side(x, v_info): 
    #***********************************************************************#
    #poisson equation trajectory on p-side of device, called by solve_ivp
    #x=0 is interface to x=x_p is p_side boundary
    #parameters: x is value of x, v_info is [V(x), dV/dx]
    #returns [dV/dx, d^2V/dx^2]
    #***********************************************************************#
    v = v_info[0]
    dV = v_info[1]
    v_deriv = dV
    dv_deriv = math.exp(v) - math.exp(-v) + (n_a/NI)
    return [v_deriv, dv_deriv]

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

#****************************************************************************#
#Initial Guess p-side:
#Guess that potential is zero at interface, exponentially decays to negative 
#boundary value and has decreased by 75% when 1/10 distance across the p_side 
#of the device 
#Only two points are needed to fit exponential function y = a * Exp(-b * x)
#****************************************************************************#
a_p = -v_boundary_p
b_p = math.log(0.25) / (-x_p / (ldi * 10))

def initial_guess_V_p_side(x):
    return (a_p * math.exp(-b_p * x) - a_p)

def initial_guess_dV_p_side(x):
    return (-a_p * b_p * math.exp(-b_p * x))

v_points_initial_p = []
dV_points_initial_p = []
for i in range(len(points_p_side)):
    v_points_initial_p.append(initial_guess_V_p_side(points_p_side[i]))
    dV_points_initial_p.append(initial_guess_dV_p_side(points_p_side[i]))

#****************************************************************************#
#Initial Guess n-side:
#Guess same shape, but decaying from zero to positive boundary value, going 
#from 0 at x=0 to v_boundary_n at x = x_n
#****************************************************************************#
a_n = v_boundary_n
b_n = math.log(0.25) / (-x_n / (ldi * 10))

def initial_guess_V_n_side(x):
    return (-a_n * math.exp(-b_n * x) + a_p)

def initial_guess_dV_n_side(x):
    return (a_n * b_n * math.exp(-b_n * x))

v_points_initial_n = []
dV_points_initial_n = []
for i in range(len(points_n_side)):
    v_points_initial_n.append(initial_guess_V_n_side(points_n_side[i]))
    dV_points_initial_n.append(initial_guess_dV_n_side(points_n_side[i]))

#****************************************************************************#   
#List of all variables that must be solved for, initialized from initial guess
#var_list[0: 2*num_intervals_p_side] = 
#[V[points_p_side[0]], dV[points_p_side[0]], V[points_p_side[1], 
#dV[points_p_side[1]], ... up to but not including points at boundary]
#var_list[2*num_intervals_p_side: 
#2*(num_intervals_p_side+num_intervals_n_side)] = 
#[V[points_n_side[0]], dV[points_n_side[0]], V[points_n_side[1], 
#dV[points_n_side[1]], ... up to but not including points at boundary]
#****************************************************************************#    
var_list = np.zeros(2 * (num_intervals_p_side + num_intervals_n_side)) 
for i in range(num_intervals_p_side):
    var_list[(2*i)] = v_points_initial_p[i]
    var_list[(2*i)+1] = dV_points_initial_p[i]
offset = num_intervals_p_side * 2
for i in range(num_intervals_n_side):
    var_list[(2*i)+offset] = v_points_initial_n[i]
    var_list[(2*i)+1+offset] = dV_points_initial_n[i]    
                                           
#****************************************************************************#
#List of functions that must be equal to zero, from subtracting RHS from 
#each equation
#Goal is to solve for var_list such that calculate_f_list returns vector of zeros
#Advice for reading code below: solve_ivp has four parameters: the first-order
#system to integrate, the domain of the independent variable over which to
#integrate, the values of the dependent variables at the beginning of the
#interval, and the method to be used. Returns an array, y, of dimensions
#(number of x points, number of dependent variables). solution.y[0,-1] refers
#to the computed value of v at the end of the interval, solution.y[1,-1]
#refers to the computed value of dx/dx at the end of the interval
#****************************************************************************#
def calculate_f_list(var_list):
    f_list = np.zeros(2 * (num_intervals_p_side + num_intervals_n_side)) 
    #Continuity equations p-side:
    for i in range(num_intervals_p_side-1):
        f_list[(2*i)] = solve_ivp(poisson_equation_p_side, (points_p_side[i], 
            points_p_side[i+1]), [var_list[(2*i)], var_list[(2*i)+1]], 
            method = 'BDF').y[0,-1] - var_list[(2*i)+2]
        f_list[(2*i)+1] = solve_ivp(poisson_equation_p_side, (points_p_side[i]
            , points_p_side[i+1]), [var_list[(2*i)], var_list[(2*i)+1]], 
            method = 'BDF').y[1,-1] - var_list[(2*i)+3]
    #Boundary condition p-side:
    f_list[(2*num_intervals_p_side)-2] = solve_ivp(poisson_equation_p_side, 
        (points_p_side[num_intervals_p_side-1], 
        points_p_side[num_intervals_p_side]), 
        [var_list[(2*num_intervals_p_side)-2], 
        var_list[(2*num_intervals_p_side)-1]], method = 'BDF').y[0,-1] - v_boundary_p
    #Continuity equations at interface
    f_list[(2*num_intervals_p_side)-1] = var_list[0] - var_list[(2*num_intervals_p_side)]
    f_list[(2*num_intervals_p_side)] = var_list[1] + var_list[(2*num_intervals_p_side)+1]
    #Continuity equations n-side:
    offset = (2*num_intervals_p_side) + 1
    for i in range(num_intervals_n_side-1):
        f_list[(2*i)+offset] = solve_ivp(poisson_equation_n_side, 
            (points_n_side[i], points_n_side[i+1]), [var_list[(2*i)+offset-1], 
            var_list[(2*i)+offset]], method = 'BDF').y[0,-1] - var_list[(2*i)+offset+1]
        f_list[(2*i)+1+offset] = solve_ivp(poisson_equation_n_side, 
            (points_n_side[i], points_n_side[i+1]), [var_list[(2*i)+offset-1], 
            var_list[(2*i)+offset]], method = 'BDF').y[1,-1] - var_list[(2*i)+offset+2]
    #Boundary condition n-side
    f_list[2*(num_intervals_n_side+num_intervals_p_side)-1] = solve_ivp(poisson_equation_n_side, 
        (points_n_side[num_intervals_n_side-1], points_n_side[num_intervals_n_side]), 
        [var_list[2*(num_intervals_p_side+num_intervals_n_side)-2], 
        var_list[2*(num_intervals_p_side+num_intervals_n_side)-1]], 
        method = 'BDF').y[0,-1] - v_boundary_n
    return f_list

#Solve all necessary equations:
solution = root(calculate_f_list, var_list, method='hybr') #Print solution for more info
var_list = solution.x

#****************************************************************************#
#Now that all variables have been solved for, solve initial value problems in 
#each interval with correct parameters, combine data, and plot:
#****************************************************************************#
x_p_side = []
v_p_side = []
dV_p_side = []
x_n_side = []
v_n_side = []
dV_n_side = []

for i in range(num_intervals_p_side):
    sol = solve_ivp(poisson_equation_p_side, (points_p_side[i], 
        points_p_side[i+1]), [var_list[(2*i)], var_list[(2*i)+1]], 
        method = 'BDF')
    for j in range(sol.t.shape[0]):
        x_p_side.append(sol.t[j])
        v_p_side.append(sol.y[0,j])
        dV_p_side.append(sol.y[1,j])
offset = 2*num_intervals_p_side
for i in range(num_intervals_n_side):
    sol = solve_ivp(poisson_equation_n_side, (points_n_side[i], 
        points_n_side[i+1]), [var_list[(2*i)+offset], var_list[(2*i)+offset+1]]
        , method = 'BDF')
    for j in range(sol.t.shape[0]):
        x_n_side.append(sol.t[j])
        v_n_side.append(sol.y[0,j])
        dV_n_side.append(sol.y[1,j])

x_n_side.reverse()
v_n_side.reverse()
dV_n_side.reverse()

final_x = []
final_V = []
final_dV = []

#Combine data from p-side and n-side
for i in range(len(x_n_side)):
    x_n_side[i] = (x_n/ldi) - x_n_side[i]
    final_x.append(x_n_side[i])
    final_V.append(v_n_side[i])
    final_dV.append(-dV_n_side[i])
for i in range(len(x_p_side)):
    x_p_side[i] = x_p_side[i] + (x_n/ldi)
    final_x.append(x_p_side[i])
    final_V.append(v_p_side[i])
    final_dV.append(dV_p_side[i])    

#Unscale variables
for i in range(len(final_x)):
    final_x[i] = final_x[i] * ldi
    final_x[i] = final_x[i] * 1E7 #convert from cm to nm
    final_V[i] = final_V[i] * vt
    final_dV[i] = final_dV[i] * (vt/ldi)

e_field = [] #Electric Field = -dV/dx
n = [] #Electron concentration = ni * e^(V/Vt) for device in equilibrium
p = [] #Hole concentration, ni * e^(-V/Vt) for device in equilibrium
c_band = [] #Conduction band
v_band = [] #Valence band

for i in range(len(final_x)):
    e_field.append(-final_dV[i])
    n.append(NI * math.exp(final_V[i]/vt))
    p.append(NI * math.exp(-final_V[i]/vt))    
    c_band.append(dec - final_V[i])
    v_band.append(c_band[i] - BAND_GAP)

fig, ax1 = plt.subplots(2)
fig.set_figheight(10)
fig.set_figwidth(15)
    
ax1[0].plot(final_x, final_V)
ax1[0].set_xlabel('Position (nm)')
ax1[0].set_ylabel('Electric Potential (V)')
ax1[1].plot(final_x, e_field)
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

#****************************************************************************#
#Finally, write data to csv files so it can be used for initial guess in the
#nonequilibrium program. Need to write data for x, V, n, and p. Since variable
#scaling is again used in the nonequilibrium program, I will rescale the
#V and x and apply scaling to n and p to make things easier.
#****************************************************************************#
with open('equilibrium_scaled_result.csv', mode='w') as result_file:
    result_writer = csv.writer(result_file, delimiter=',')

    #Change values to arrays so they can be multiplied by a scalar
    result_writer.writerow(np.array(final_x) * ((1E-7)/ldi))
    result_writer.writerow(np.array(final_V) * (1/vt)) 
    result_writer.writerow(np.array(n) * (1/NI))
    result_writer.writerow(np.array(p) * (1/NI))
