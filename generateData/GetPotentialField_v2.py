## Explicitly define the geobody in a regular grid
##
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
Griddim = (40, 40, 40) # resolution
interpRange = ([-25,25],[-25,25],[-25,-1]) # dimensions
x_ = np.linspace(interpRange[0][0], interpRange[0][1], Griddim[0])
y_ = np.linspace(interpRange[1][0], interpRange[1][1], Griddim[1])
z_ = np.linspace(interpRange[2][0], interpRange[2][1], Griddim[2])
x, y, z = np.meshgrid(x_, y_, z_, indexing='xy')
xyz = np.stack([x.flatten(),y.flatten(),z.flatten()]).T

# for force refining mesh
# x_2d,y_2d = np.meshgrid(x_[::2], y_[::2])

def createPLC():
    plc = mt.createCube(size=[width, width, depth], pos=[0, 0, -depth/2], boundaryMarker=1)
    # Force finer meshing at at the sensors by create an additional node at 1/2 mm in -z-direction
    for sensor in scheme.sensors().array():
        plc.createNode(sensor - [0.0, 0.0, 0.0005])
        # plc.createNode(sensor - [0.0, 0.0, 0.1001])
        # plc.createNode(sensor - [0.0, 0.0, 0.1])
        # plc.createNode(sensor - [0.0, 0.0, 1])
    return plc

plc = createPLC()

mesh = mt.createMesh(plc, area = 20)
## Visualization
pg.show(mesh)
mesh.exportVTK('mesh')
    
# t00 = default_timer()
# define the background field
# field_hom = simulateERT(mesh, res=10, scheme=scheme, sr=False,
#                   calcOnly=True, verbose=True, returnFields=True)
# # extract background field at regular grid
# t40 = default_timer()
# hom_uTemp = field_hom[2].array()
# node_position = mesh.positions().array()
# pub_hom_u = interpData(xyz, node_position,hom_uTemp)
# t41 = default_timer()
# print('field_time_scipy_interp: ',t41-t40)

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

nSamples = 2000

## define the range of the parameters
position_lower= np.array([-7,-7,-10]) # modify manually according to the problem
position_upper = np.array([7,7,-7])
positions = np.random.uniform(position_lower,position_upper,size = (nSamples,3))


dim_upper = np.array([8,8,6]) # modify manually according to the problem
dim_lower = np.array([3,3,5])
dims = np.random.uniform(dim_lower,dim_upper,size = (nSamples,3))

Rho1 = 10*np.ones(shape = (nSamples,1))
Rho2 = 100*np.ones(shape = (nSamples,1))
# Rho2 = np.random.uniform(90,110,size = (nSamples,1))

for i in range(nSamples):
    t1 = default_timer()
    plc = createPLC()
    # define the geometry of the body
    dim = dims[i]
    position = positions[i]
    ## force re-meshing around the geobody
    depth_ = position[2] - dim[2]/2 
    # create a 2D grid at the top surface of the geo-body
    
    ### create a finer mesh around the body
    x_ = np.linspace(position[0]-dim[0]/2, position[0]+dim[0]/2, 5)
    y_ = np.linspace(position[1]-dim[1]/2, position[1]+dim[1]/2, 5)
    z_ = np.linspace(position[2] - dim[2]/2 , position[2] + dim[2]/2 , 5)
    x_r,y_r,z_r = np.meshgrid(x_, y_, z_)
    
    ###
    
    # xyd = np.stack([x_2d.flatten(),y_2d.flatten(),np.repeat((depth_),x_2d.flatten().shape)]).T
    xyd = np.stack([x_r.flatten(),y_r.flatten(),z_r.flatten()]).T
    
    for n in xyd:
        r = 0.5
        plc.createNode(n + [r, r, r])
        plc.createNode(n + [-r, -r, -r])
    
    mesh = defGeom(dim, position, plc, area = 30)


    res = [[1, Rho1[i]], [2, Rho2[i]]]
    ## Truncated mesh range, select part of the date used in the interpolation to save computation time
    # the truncation range should be larger that the query grid
    R = ([-30,30],[-30,30],[-30,-0])
    

    # get resistivity values at regular gird position
    
    meshPoints, meshValues = extractData(mesh,res,R)
    
    # request_range = [[-25, 25, 50/30], [-25, 25, 50/30], [-25, -1, 24/70]]
    t21 = default_timer()
    resValues = interpData(xyz, meshPoints,meshValues)
    # t22 = default_timer()
    # nnresValues = nneighborInter(meshPoints,meshValues.squeeze(),request_range)
    
    t3 = default_timer()
    print('time_scipy_interp: ',t3-t21)
    # print('time_nn_interp: ',t3-t22)
    
    ## both interpolation methods work not quite well
    ## let's try explicitly define the regular grid geometry
    
    ###
    field_hom = simulateERT(mesh, res=10, scheme=scheme, sr=False,
                  calcOnly=True, verbose=True, returnFields=True)
    # extract background field at regular grid
    t40 = default_timer()
    bgField = field_hom[2].array()
    node_position = mesh.positions().array()
    # hom_u = interpData(xyz, node_position,bgField)
    # t41 = default_timer()
    # print('hom_field_time_scipy_interp: ',t41-t40)
    ###
    
    hetTemp = runSimu(mesh, res, scheme, returnFields=True)

    t40 = default_timer()

    uTemp = hetTemp[2].array()
    ## compute the potential field difference with respect to a homogenous background
    u_diff = uTemp-bgField
    u = interpData(xyz, node_position,u_diff)
    t41 = default_timer()
    print('field_time_scipy_interp: ',t41-t40)
    ### this implementation is slow
    # a_list = np.append(a_list,resValues)
    # u_list = np.append(u_list,u)
    # xyz_list = np.append(xyz_list, xyz, axis = 0)
    

    
    a_list.append(resValues)
    u_list.append(u)
    xyz_list.append(xyz)
    
    t6 = default_timer()
#     print('iteration: ', i ,' time: ',t3-t2)
    print('iteration: ', i ,' time: ',t6-t1)
    n = int(np.floor(position[0]-(interpRange[0][0]))/((interpRange[0][1]-(interpRange[0][0]))/40))
    plt.title('sample:'+str(i)+'  section:'+str(n))
    PlotSection(u_list[-1],n,Griddim)
    
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
scipy.io.savemat('/Users/zhouji/Documents/github/ERT/generateData/Fdata5000.mat', mdict={'coord':xyz_list, 'a': a_list,'u': u_list})

# # %%
# ## Visualization
# pg.show(mesh)

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



# PlotSection(nnresValues,14,Griddim)
for i in range(Griddim[0]):
    plt.title(i)
    PlotSection(resValues,i,Griddim)
    
# n = 22
# PlotSection(u_list[-1],n,Griddim)
# PlotSection(hom_u,n,Griddim)
# PlotSection(pub_hom_u,n,Griddim)
# PlotSection(u_list[-1]-pub_hom_u,n,Griddim)
# PlotSection(u_list[-1]-hom_u,n,Griddim)

# for i in range(10):
#     PlotSection(u[i],14)