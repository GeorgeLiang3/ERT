# %%
import numpy as np

import pygimli as pg
import pygimli.meshtools as mt
from pygimli.physics.ert import simulate as simulateERT
import pygimli.physics.ert as ert
import pyvista as pv

from timeit import default_timer
from ertUtil import *
import scipy.io


# %%
# define the geometry
depth = 50
width = 100

#Create a pyGimli configuration
nPerLine = 4
scheme = ert.createERTData(elecs=np.linspace(start=-15, stop=15, num=nPerLine),
                           schemeName='dd')
pos = scheme.sensorPositions().array()

# nLine = 8
# posLine = np.linspace(-15,15,nLine)
# sensors = np.empty((0,3), int)
# for i in posLine:
#     line = pos
#     line[:,1] = i
#     sensors = np.append(sensors,line,axis=0)
# scheme.setSensors(sensors)

## Create the configuration by shifting the reference line config
# total_FourPointData = np.empty((0,4),int)
# refLine = np.array([scheme['a'],scheme['b'],scheme['m'],scheme['n']]).T
# # Shifting lines
# for i in range(nLine):
#     total_FourPointData = np.append(total_FourPointData,refLine + i * nPerLine,axis=0)
# # manually set-up Four-Point Config 
# for i,FourPointer in enumerate(total_FourPointData):
#     scheme.createFourPointData(i,total_FourPointData[i,0],total_FourPointData[i,1],total_FourPointData[i,2],total_FourPointData[i,3])
    

# create a regular mesh for learning
Griddim = (30, 30, 70) # resolution
interpRange = ([-25,25],[-25,25],[-25,-1]) # dimensions
x_ = np.linspace(interpRange[0][0], interpRange[0][1], Griddim[0])
y_ = np.linspace(interpRange[1][0], interpRange[1][1], Griddim[1])
z_ = np.linspace(interpRange[2][0], interpRange[2][1], Griddim[2])
x, y, z = np.meshgrid(x_, y_, z_, indexing='xy')
xyz = np.stack([x.flatten(),y.flatten(),z.flatten()]).T

# x_2d,y_2d = np.meshgrid(x_, y_)

def creatPLC():
    plc = mt.createCube(size=[width, width, depth], pos=[0, 0, -depth/2], boundaryMarker=1)
    # Force finer meshing at at the sensors by create an additional node at 1/2 mm in -z-direction
    for sensor in scheme.sensors().array():
        plc.createNode(sensor - [0.0, 0.0, 0.0005])
        # plc.createNode(sensor - [0.0, 0.0, 0.1001])
        # plc.createNode(sensor - [0.0, 0.0, 0.1])
        # plc.createNode(sensor - [0.0, 0.0, 1])
    return plc

plc = creatPLC()
    
mesh = mt.createMesh(plc, area = 20)
## Visualization
pg.show(mesh)
mesh.exportVTK('mesh')
    
# t00 = default_timer()
# field_hom = simulateERT(mesh, res=1.0, scheme=scheme, sr=False,
#                   calcOnly=True, verbose=True, returnFields=True)

# t01 = default_timer()

# print(' time: ',t01-t00)

# hom.save('homogeneous.ohm', 'a b m n u')
# hom.set('k', 1.0/ (hom('u') / hom('i')))
# hom.set('rhoa', hom('k') * hom('u') / hom('i'))

# hom.save('simulatedhom.dat', 'a b m n rhoa k u i')



    
# # %%

# xyz_list = np.empty((0,3), int)
# a_list = np.empty((0,1), int)
# u_list = np.empty((0,1), int)
xyz_list = []
a_list = []
u_list = []

nSamples = 1

## define the range of the parameters
position_upper = np.array([-2,-2,-20]) # modify manually according to the problem
position_lower = np.array([2,2,-11])
positions = np.random.uniform(position_lower,position_upper,size = (nSamples,3))


dim_upper = np.array([15,15,10]) # modify manually according to the problem
dim_lower = np.array([7,7,5])
dims = np.random.uniform(dim_lower,dim_upper,size = (nSamples,3))

Rho1 = 10*np.ones(shape = (nSamples,1))
Rho2 = 100*np.ones(shape = (nSamples,1))
# Rho2 = np.random.uniform(90,110,size = (nSamples,1))

for i in range(nSamples):
    t1 = default_timer()
    plc = creatPLC()
    # define the geometry of the body
    dim = dims[i]
    position = positions[i]
#     ## force re-meshing around the geobody
#     depth_ = position[2] - dim[2]/2 
#     # create a 2D grid at the top surface of the geo-body
#     # xyd = np.stack([x_2d.flatten(),y_2d.flatten(),np.repeat((depth_),x_2d.flatten().shape)]).T
    
#     # for n in xyd:
#     #     plc.createNode(n + [0.0, 0.0, 0.2])
#     #     plc.createNode(n + [0.0, 0.0, 1])
    
    mesh = defGeom(dim, position, plc)


    res = [[1, Rho1[i]], [2, Rho2[i]]]
    # Truncated mesh range
    R = interpRange
    

    # get resistivity values at regular gird position
    t2 = default_timer()
    meshPoints, meshValues = extractData(mesh,res,R)
    
    resValues = interpData(xyz, meshPoints,meshValues)
    t3 = default_timer()

    
    hetTemp = runSimu(mesh, res, scheme, returnFields=True)

    
    node_position = mesh.positions().array()
    uTem = hetTemp[2].array()
    u = interpData(xyz, node_position,uTem)
    # this implementation is slow
    # a_list = np.append(a_list,resValues)
    # u_list = np.append(u_list,u)
    # xyz_list = np.append(xyz_list, xyz, axis = 0)
    
    a_list.append(resValues)
    u_list.append(u)
    xyz_list.append(xyz)
    
    t4 = default_timer()
#     print('iteration: ', i ,' time: ',t3-t2)
    print('iteration: ', i ,' time: ',t4-t1)

a_list = np.array(a_list)
u_list = np.array(u_list)
xyz_list = np.array(xyz_list)

a_list = np.float32(a_list)
u_list = np.float32(u_list)
xyz_list = np.float32(xyz_list)


# import h5py
# h5f = h5py.File('data5000.h5', 'w')
# h5f.create_dataset('a', data=a_list)
# h5f.create_dataset('coord', data=xyz_list)
# h5f.create_dataset('u', data=u_list)

# h5f.close()

# mesh.exportVTK('mesh')
# scipy.io.savemat('/Users/zhouji/Documents/github/ERT/generateData/Fdata5000.mat', mdict={'coord':xyz_list, 'a': a_list,'u': u_list})

# # %%
# ## Visualization
pg.show(mesh)

# # %%
# import sys
# sys.path.append("..")
# from plot3D import plotData, plotGrid

# # %%
# plotData(hetTemp,nPerLine,1)
# # %%
# values = np.reshape(resValues,[Griddim[1],Griddim[0],Griddim[2]])

# plotGrid(values,Griddim)


# %%
import matplotlib.pyplot as plt
def PlotSection(grid,sec=0,Griddim = (30, 30, 70),interpRange = ([-25,25],[-25,25],[-25,-1])):
    '''plot a section in y direction'''
    g = grid.reshape(Griddim)
    plt.imshow(g[sec].T,origin = 'lower',aspect = 'auto',extent=[interpRange[1][0],interpRange[1][1],interpRange[2][0],interpRange[2][1]])
    plt.colorbar()
    plt.show()

PlotSection(u_list[0],14)

# for i in range(10):
#     PlotSection(u[i],14)