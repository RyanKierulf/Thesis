#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import root

q   = 1.602E-19
kb  = 1.38E-23
eps = 1.05E-12
T   = 300
ni  = 1.5E10
Vt  = kb*T/q
Ldi = math.sqrt(eps*Vt/(q*ni))

Na = 1E17
Nd = 1E17
x_n = 500E-7
x_p = 500E-7
total_length = x_n + x_p
v_diff = Vt * math.log(Na * Nd / (ni ** 2))
v_boundary_n = v_diff/2
v_boundary_p = -v_diff/2

v_boundary_n = v_boundary_n / Vt
v_boundary_p = v_boundary_p / Vt
total_length = (x_n + x_p)/Ldi

def poisson_equation_n_side(x_info, V_info): #x=0 is interface to x=x_n is n_side boundary
    V = V_info[0]
    dV = V_info[1]
    V_deriv = -dV
    dV_deriv = -(math.exp(V) - math.exp(-V) - (Nd/ni))
    return [V_deriv, dV_deriv]

#I ended up not using the function below based on assumptions about the symnetry of the device   
def poisson_equation_p_side(x_info, V_info): #x=x_n is interface to x=(x_n + x_p) is p_side boundary
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

#Initial Guess:
#Guess that potential is zero at interface, exponentially decaying and has decreased by 75% when 1/10 distance across the p_side of the device
#Only two points are needed to fit exponential function y = a * Exp(-b * x)
a_p = -v_boundary_p
def get_b_p_side(b): #find root of this function to get b
   return (0.25 - math.exp(-b * (x_p / (Ldi * 10)))) 
b_p = root(get_b_p_side, 1000).x[0] #For a normal PN junction, b is around the order of magnitude of 10^3, so use that as initial guess for b

def initial_guess_V_p_side(x):
    return (a_p * math.exp(-b_p * x) - a_p)

def initial_guess_dV_p_side(x):
    return (-a_p * b_p * math.exp(-b_p * x))

V_points_initial_p = []
dV_points_initial_p = []
for i in range(len(points_p_side)):
    V_points_initial_p.append(initial_guess_V_p_side(points_p_side[i]))
    dV_points_initial_p.append(initial_guess_dV_p_side(points_p_side[i]))
    
#Uncomment to plot initial V and dV
#plt.plot(points_p_side, V_points_initial_p)
#plt.plot(points_p_side, dV_points_initial_p)
    
#for i in range(num_intervals_p_side):
#    sol = solve_ivp(poisson_equation_p_side, (points_p_side[i], points_p_side[i+1]), [V_points_initial_p[i], dV_points_initial_p[i]], method = 'BDF')
#    plt.plot(sol.t, sol.y[0,:])

#Initial list of variables
#var_list is [dV[x0], V[x1], dV[x1], V[x2], dV[x2], ... up to not including V and dV at the final boundary]
var_list_p = np.zeros((num_intervals_p_side * 2) - 1)
var_list_p[0] = dV_points_initial_p[0]
for i in range(1, num_intervals_p_side):
    var_list_p[(2*i)-1] = V_points_initial_p[i]
    var_list_p[(2*i)] = dV_points_initial_p[i]
    
def calculate_f_list(var_list):
    f_list = np.zeros((num_intervals_p_side * 2) - 1)
    f_list[0] = solve_ivp(poisson_equation_p_side, (points_p_side[0], points_p_side[1]), [0, var_list[0]], method = 'BDF').y[0,-1] - var_list[1]
    f_list[1] = solve_ivp(poisson_equation_p_side, (points_p_side[0], points_p_side[1]), [0, var_list[0]], method = 'BDF').y[1,-1] - var_list[2]
    for i in range(1, num_intervals_p_side-1):
        f_list[(2*i)] = solve_ivp(poisson_equation_p_side, (points_p_side[i], points_p_side[i+1]), [var_list[(2*i)-1], var_list[(2*i)]], method = 'BDF').y[0,-1] - var_list[(2*i)+1]
        f_list[(2*i)+1] = solve_ivp(poisson_equation_p_side, (points_p_side[i], points_p_side[i+1]), [var_list[(2*i)-1], var_list[(2*i)]], method = 'BDF').y[1,-1] - var_list[(2*i)+2]
    f_list[(num_intervals_p_side * 2) - 2] = solve_ivp(poisson_equation_p_side, (points_p_side[num_intervals_p_side-1], points_p_side[num_intervals_p_side]), [var_list[(num_intervals_p_side*2)-3], var_list[(num_intervals_p_side*2)-2]], method = 'BDF').y[0,-1] - v_boundary_p
    return f_list

#f_list = calculate_f_list(var_list_p)
solution = root(calculate_f_list, var_list_p) #Print solution for more info about method, number of function evaluations, etc.
var_list = solution.x

x_p_side = []
V_p_side = []
dV_p_side = []

sol1 = solve_ivp(poisson_equation_p_side, (points_p_side[0], points_p_side[1]), [0, var_list[0]], method = 'BDF')
for j in range(sol1.t.shape[0]):
    x_p_side.append(sol1.t[j])
    V_p_side.append(sol1.y[0,j])
    dV_p_side.append(sol1.y[1,j])
for i in range(1, num_intervals_p_side-1):
    sol = solve_ivp(poisson_equation_p_side, (points_p_side[i], points_p_side[i+1]), [var_list[(2*i)-1], var_list[(2*i)]], method = 'BDF')
    for j in range(sol.t.shape[0]):
        x_p_side.append(sol.t[j])
        V_p_side.append(sol.y[0,j])
        dV_p_side.append(sol.y[1,j])
        
x_n_side = []
V_n_side = []
dV_n_side = []
for i in range(len(x_p_side)):
    x_n_side.append(-x_p_side[i])
    V_n_side.append(-V_p_side[i])
    dV_n_side.append(dV_p_side[i])
    
x_n_side.reverse()
V_n_side.reverse()
dV_n_side.reverse()
final_x = []
final_V = []
final_dV = []

for i in range(len(x_n_side)):
    final_x.append(x_n_side[i])
    final_V.append(V_n_side[i])
    final_dV.append(dV_n_side[i])
for i in range(len(x_p_side)):
    final_x.append(x_p_side[i])
    final_V.append(V_p_side[i])
    final_dV.append(dV_p_side[i])

#Now unscale potential
for i in range(len(final_x)):
    final_x[i] = final_x[i] + (x_n/Ldi)
    final_x[i] = final_x[i] * Ldi
    final_V[i] = final_V[i] * Vt
    final_dV[i] = final_dV[i] * (Vt/Ldi)

E = []
n = []
p = []
for i in range(len(final_x)):
    E.append(-final_dV[i])
    n.append(ni * math.exp(final_V[i]/Vt))
    p.append(ni * math.exp(-final_V[i]/Vt))    

fig, ax = plt.subplots(3)
fig.set_figheight(20)
fig.set_figwidth(40)
    
ax[0].plot(final_x, final_V)
ax[0].set_xlabel('Position (cm)')
ax[0].set_ylabel('Electric Potential (V)')
ax[1].plot(final_x, E)
ax[1].set_xlabel('Position (cm)')
ax[1].set_ylabel('Electric Field (V/cm)')
ax[2].plot(final_x, n)
ax[2].plot(final_x, p)
ax[2].set_yscale('log')
ax[2].set_xlabel('Position (cm)')
ax[2].set_ylabel('Electron and Hole Concentrations (cm^-1)')