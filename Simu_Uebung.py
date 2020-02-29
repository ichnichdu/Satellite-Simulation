#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 13:41:49 2019

@author: florian
"""


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import cartopy.crs as ccrs
import cartopy.feature as cf



SMALL_SIZE = 11
MEDIUM_SIZE = 13
BIGGER_SIZE = 16

#Größe der Plots
plt.rc('font', size=MEDIUM_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=BIGGER_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=MEDIUM_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title


plt.close("all")

def converter(x,y,z):
    elevation_func = np.arccos(z/(np.sqrt(x**2 + y**2 + z**2)))
    azimuth_func = np.arctan(y/x)
    return azimuth_func, elevation_func

#%% Input Data

resolution = 1000 # Resolution of the simulation (Multiples of 4) 1000 is good for a smooth 3d plot

speed_min = 0
speed_max = 0

earth_radius = 6371 * 1000

peri_dist = 773. * 10**3  + earth_radius  #Meter
eccentricity = 0.00022
inclination = 8.4 * np.pi / 180 # inclination of 0 would result in a polar orbit


mu = 3.986004418 * 10**14 # Earth Gravitation parameter

circle = np.linspace(0., 2*np.pi, resolution)

earth_speed = 465.1  #Rotation speed at equator in m/s

ground_size_diameter = .5





#%% Calculation

# Simple Calculations for the orbital Parameters
greater_half_axe = peri_dist / (1 - eccentricity)
apo_dist = greater_half_axe * (1 + eccentricity)

mean_dist = (apo_dist + peri_dist) / 2

speed = np.sqrt(mu * ((2 /  mean_dist) - (1/greater_half_axe)))

space_vector = (greater_half_axe * (1 - eccentricity**2))/(1 + eccentricity * np.cos(circle))

speed_array = np.sqrt(mu * ((2 /  space_vector) - (1/greater_half_axe)))

# These Ground speeds are not corrected for Earth rotation
ground_speed_max = max(speed_array) * earth_radius/peri_dist
ground_speed_min = min(speed_array) * earth_radius/peri_dist

ground_speed_array = speed_array * (earth_radius / space_vector) 

# Manual integration of the Orbital time

orbit_time = sum(((2 * np.pi / len(space_vector)) * space_vector) / speed_array)


# Setup of the Transformation for the Orbit into carthesian coordinates
elevation_satellite_min = 0 
elevation_satellite_max =  2*np.pi 
elevation_satellite = np.linspace(elevation_satellite_min, elevation_satellite_max, resolution)

azimuth_satellite_min = 0 
azimuth_satellite_max = 0 
azimuth_satellite = np.linspace(azimuth_satellite_min, azimuth_satellite_max, resolution)

# Rotation Matrix with the X axis as the rotation axis
rot_matrix_x = np.array([[1, 0, 0], 
                        [0, np.cos(inclination), np.sin(inclination)], 
                        [0, -np.sin(inclination), np.cos(inclination)]])

# Transformations
# satellite kartesian
x_satellite = space_vector * np.cos(elevation_satellite) * np.cos(azimuth_satellite)
y_satellite = space_vector * np.cos(elevation_satellite) * np.sin(azimuth_satellite)
z_satellite = space_vector * np.sin(elevation_satellite)

satellite_matrix = np.array([x_satellite, y_satellite, z_satellite])

# satellite rotate
satellite_rot = np.dot(rot_matrix_x, satellite_matrix)


# sphere coordinates
azimuth_angle = circle
elevation_angle = np.linspace(0, np.pi, resolution)

x_earth = np.outer(np.sin(azimuth_angle), np.sin(elevation_angle)) * earth_radius
y_earth = np.outer(np.sin(azimuth_angle), np.cos(elevation_angle)) * earth_radius
z_earth = np.outer(np.cos(azimuth_angle), np.ones_like(elevation_angle)) * earth_radius


# Groundtrack 
x_groundtrack = earth_radius * np.cos(elevation_satellite) * np.cos(azimuth_satellite)
y_groundtrack = earth_radius * np.cos(elevation_satellite) * np.sin(azimuth_satellite)
z_groundtrack = earth_radius * np.sin(elevation_satellite)

groundtrack_matrix = np.array([x_groundtrack, y_groundtrack, z_groundtrack])

ground_rot = np.dot(rot_matrix_x, groundtrack_matrix)


# Groundspeed
ground_speed_simu = np.zeros(resolution)
angle_step = 2*np.pi / (23.9345  * 60 * 60) * orbit_time / resolution

# Earth rotation along the Z Axis
earth_rot_matrix_step = np.array([[np.cos(angle_step), np.sin(angle_step), 0], 
                        [-np.sin(angle_step), np.cos(angle_step), 0], 
                        [0, 0, 1]])

ground_transfrom_angle_azimuth = np.zeros(resolution)
ground_transfrom_angle_elevation = np.zeros(resolution)

# Manual calculation of the Groundspeed. It uses the law of cosines to calculate a side of a triangle
# This side is used to calculate the Groundspeed
for i in range(0, resolution, 1):
    originpoint = ground_rot[:,i-1]
    trackpoint = ground_rot[:,i] - originpoint
    earthpoint = np.dot(earth_rot_matrix_step, originpoint)
    ground_transfrom_angle_azimuth[i], ground_transfrom_angle_elevation[i] = converter(earthpoint[0],
                                                                                 earthpoint[1],
                                                                                 earthpoint[2])
    earthpoint -= originpoint
    vector_speed = np.sqrt(trackpoint**2 
                     + earthpoint**2 
                     - 2*trackpoint*earthpoint*np.cos(inclination - .5*np.pi))
    ground_speed_simu[i] = np.sqrt(vector_speed[0]**2 + vector_speed[1]**2 + vector_speed[2]**2)
    ground_speed_simu[i] = ground_speed_simu[i]  / (orbit_time /resolution)
    
    
    
ground_speed_simu[0] = ground_speed_simu[1]


# Refreshrate of the sensor

refresh_rate = ground_speed_simu / ground_size_diameter
    


#%% Plot


fig = plt.figure(figsize=plt.figaspect(1.))  
ax = fig.add_subplot(111, projection='3d')




ax.plot_wireframe(x_earth, y_earth, z_earth, color = "black")
ax.plot(satellite_rot[0], satellite_rot[1], satellite_rot[2], color = "r", label = "Orbit")
ax.plot(ground_rot[0], ground_rot[1], ground_rot[2], color = "g", label = "Groundtrack")
ax.legend()

ax.set_title("Orbitbahn")

fig, bx = plt.subplots()
fig, cx = plt.subplots()
fig, dx = plt.subplots()
fig, fx = plt.subplots()
fig, ex = plt.subplots()
fig, gx = plt.subplots()

#dx.plot(space_vector, label = "Orbithöhe")
#dx.legend()
#dx.grid()
#dx.set_xlabel("Punkt in der Simulation")
#dx.set_ylabel("Orbithöhe in m")
#dx.set_title("Orbithöhe über der Erdoberfläche")


dx.plot(refresh_rate, label = "Refresh Rate")
dx.legend()
dx.grid()
dx.set_xlabel("Punkt in der Simulation")
dx.set_ylabel("Refresh Rate in Hz")
dx.set_title("Wiederholungsrate des Sensors")


bx.plot(refresh_rate, label = "Refresh Rate")
bx.legend()
bx.grid()
bx.set_xlabel("Punkt in der Simulation")
bx.set_ylabel("Refresh Rate in Hz")
bx.set_title("Wiederholungsrate des Sensors")

cx.plot(ground_speed_simu, label = "Groundspeed")
cx.legend()
cx.grid()
cx.set_xlabel("Punkt in der Simulation")
cx.set_ylabel("Groundspeed in m/s")
cx.set_title("Groundspeed")


# Projection of the Groundtrack on a 2D map
# Correction needed since cartopy seems to have problems with the used frame of reference
lat = ground_transfrom_angle_elevation / np.pi * 180 -90
lon = ground_transfrom_angle_azimuth / np.pi * 180
res_corr = resolution / 4
corr = np.ones(int(res_corr)) * 90
lon[:int(res_corr)] += corr
lon[int(res_corr):2*int(res_corr)] -= corr
lon[2*int(res_corr):3*int(res_corr)] -= corr
lon[3*int(res_corr):] += corr

latlon_array = np.array([lat, lon])



ex.plot(speed_array, label = "Orbitalgeschwindigkeit")
ex.legend()
ex.grid()
ex.set_xlabel("Punkt in der Simulation")
ex.set_ylabel("Orbitalgeschwindigkeit in m/s")
ex.set_title("Orbitalgeschwindigkeit")


fx.plot(space_vector - earth_radius, label = "Orbithöhe")
fx.legend()
fx.grid()
fx.set_xlabel("Punkt in der Simulation")
fx.set_ylabel("Orbithöhe in m")
fx.set_title("Orbithöhe über der Erdoberfläche")


gx = plt.axes(projection = ccrs.PlateCarree())  
gx.add_feature(cf.COASTLINE)  
gx.stock_img()   
plt.scatter(latlon_array[1], latlon_array[0], transform = ccrs.Geodetic())          
gx.set_title("Groundtrack Visualisation")                        
plt.show()




print("Orbital Period: " + str(orbit_time/60/60))




