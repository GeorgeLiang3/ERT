# %%
import numpy as np

import pygimli as pg
import pygimli.meshtools as mt
from pygimli.physics.ert import simulate as simulateERT
import pygimli.physics.ert as ert
import pyvista as pv
import pandas as pd
from scipy.interpolate import griddata
from timeit import default_timer
# import scipy.io


# %%
# define the geometry
depth = 50
width = 100

#Create a pyGimli configuration
nPerLine = 21
scheme = ert.createERTData(elecs=np.linspace(start=-15, stop=15, num=nPerLine),
                           schemeName='dd')
pos = scheme.sensorPositions().array()

nLine = 8
posLine = np.linspace(-15,15,nLine)
sensors = np.empty((0,3), int)
for i in posLine:
    line = pos
    line[:,1] = i
    sensors = np.append(sensors,line,axis=0)
scheme.setSensors(sensors)

## Create the configuration by shifting the reference line config
total_FourPointData = np.empty((0,4),int)
refLine = np.array([scheme['a'],scheme['b'],scheme['m'],scheme['n']]).T
# Shifting lines
for i in range(nLine):
    total_FourPointData = np.append(total_FourPointData,refLine + i * nPerLine,axis=0)
# manually set-up Four-Point Config 
for i,FourPointer in enumerate(total_FourPointData):
    scheme.createFourPointData(i,total_FourPointData[i,0],total_FourPointData[i,1],total_FourPointData[i,2],total_FourPointData[i,3])
    

# create additional nodes in the interests area to force finer meshing
Griddim = (25, 20, 50)
interpRange = ([-25,25],[-25,25],[-25,-1])
x_ = np.linspace(interpRange[0][0], interpRange[0][1], Griddim[0])
y_ = np.linspace(interpRange[1][0], interpRange[1][1], Griddim[1])
z_ = np.linspace(interpRange[2][0], interpRange[2][1], Griddim[2])
x, y, z = np.meshgrid(x_, y_, z_, indexing='xy')
x_2d,y_2d = np.meshgrid(x_, y_)
xyz = np.stack([x.flatten(),y.flatten(),z.flatten()]).T

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
    
mesh = mt.createMesh(plc)
## Visualization
# pg.show(mesh)
# mesh.exportVTK('mesh')
    

# hom = simulateERT(mesh, res=1.0, scheme=scheme, sr=False,
#                   calcOnly=True, verbose=True)


# hom.save('homogeneous.ohm', 'a b m n u')
# hom.set('k', 1.0/ (hom('u') / hom('i')))
# hom.set('rhoa', hom('k') * hom('u') / hom('i'))

# hom.save('simulatedhom.dat', 'a b m n rhoa k u i')


# %%
# Create a cube geobody with low resistivity
def defGeom(dim = [1,2,3], position = [0,0,-10], plc = None):
    

    plc0 = plc
    # c = position[2] # center position
    # h = 2*(depth_-c)
    
    # define the dimention of the geobody
    x_dim = dim[0]
    y_dim = dim[1]
    z_dim = dim[2]
    
    # define the coordinates of the center of the geobody
    x_c = position[0]
    y_c = position[1]
    z_c = position[2]
    
    # cube = mt.createCube(size=[95, 95, h], pos=[0, 0, c], marker=2)
    cube = mt.createCube(size=[x_dim, y_dim, z_dim], pos=[x_c, y_c, z_c], marker=2)
    plc0 += cube 
    
    # xyz_body_grid = getGrid(x_dim, y_dim, z_dim, x_c, y_c, z_c)
    
    # # force finer meshing around the geo-body 
    # meshsize1 = 0.3
    # meshsize2 = 0.5
    
    # # for n in xyz_body_grid:
    #     plc0.createNode(n - [meshsize1, 0.0, 0])
    #     plc0.createNode(n + [meshsize1, 0.0, 0])
    #     plc0.createNode(n - [0.0, meshsize1, 0])
    #     plc0.createNode(n + [0.0, meshsize1, 0])
    #     plc0.createNode(n - [0.0, 0.0, meshsize1])
    #     plc0.createNode(n + [0.0, 0.0, meshsize1])
        
    #     plc0.createNode(n - [meshsize2, 0.0, 0])
    #     plc0.createNode(n + [meshsize2, 0.0, 0])
    #     plc0.createNode(n - [0.0, meshsize2, 0])
    #     plc0.createNode(n + [0.0, meshsize2, 0])
    #     plc0.createNode(n - [0.0, 0.0, meshsize2])
    #     plc0.createNode(n + [0.0, 0.0, meshsize2])
    
    mesh = mt.createMesh(plc0,area = 5)
    return mesh

def getGrid(x_dim, y_dim, z_dim, x_c, y_c, z_c):
    # define a regular grid around the geobody
    x1 = -x_dim/2 + x_c
    x2 = x_dim/2 + x_c
    
    y1 = -y_dim/2 + y_c
    y2 = y_dim/2 + y_c
    
    z1 = -z_dim/2 + z_c
    z2 = z_dim/2 + z_c
    
    xs = np.linspace(x1,x2,5)
    ys = np.linspace(y1,y2,5)
    zs = np.linspace(z1,z2,5)
    x,y,z = np.meshgrid(xs,ys,zs)
    x = x.flatten()
    y = y.flatten()
    z = z.flatten()

    xyz = np.stack((x,y,z)).T
    return xyz

def runSimu(mesh, res):
    res = res  # map markers 1 and 2 to 10 and 100 Ohmm, resp.
    het = simulateERT(mesh, res=res, scheme=scheme, sr=False,calcOnly=True, verbose=True)
    return het

def extractData(mesh, res, Range):
    # Extract data at the regular grid by interpolating the mesh to a regular grid
    points = mesh.cellCenters().array()
    resArray = pg.solver.parseArgToArray(res, mesh.cellCount(), mesh)
    resArray = np.expand_dims(resArray.array(),1)
    R = Range # Define the truncation range 
    selectRange = np.where((points[:,0]>R[0][0])&(points[:,0]<R[0][1])&(points[:,1]>R[1][0])&(points[:,1]<R[1][1])&(points[:,2]>R[2][0])&(points[:,2]<R[2][1]))
    selectPoints = points[selectRange]
    selectValues = resArray[selectRange]
    return selectPoints, selectValues

def interpData(request, selectPoints, selectValues ):
    values = griddata(selectPoints, selectValues, request)
    return values
    
# %%

xyz_list = np.empty((0,3), int)
a_list = np.empty((0,1), int)
u_list = np.empty((0,1), int)


nSamples = 1

## define the range of the parameters
position_upper = np.array([-2,-2,-20]) # modify manually according to the problem
position_lower = np.array([2,2,-11])
positions = np.random.uniform(position_lower,position_upper,size = (nSamples,3))


dim_upper = np.array([15,15,10]) # modify manually according to the problem
dim_lower = np.array([7,7,5])
dims = np.random.uniform(dim_lower,dim_upper,size = (nSamples,3))

Rho1 = 10*np.ones(shape = (nSamples,1))
Rho2 = 10*np.ones(shape = (nSamples,1))
# Rho2 = np.random.uniform(90,110,size = (nSamples,1))

for i in range(nSamples):
    t1 = default_timer()
    plc = creatPLC()
    dim = dims[i]
    position = positions[i]
    ## force re-meshing around the geobody
    depth_ = position[2] - dim[2]/2 
    # create a 2D grid at the top surface of the geo-body
    # xyd = np.stack([x_2d.flatten(),y_2d.flatten(),np.repeat((depth_),x_2d.flatten().shape)]).T
    
    # for n in xyd:
    #     plc.createNode(n + [0.0, 0.0, 0.2])
    #     plc.createNode(n + [0.0, 0.0, 1])
    
    mesh = defGeom(dim, position, plc)


    res = [[1, Rho1[i]], [2, Rho2[i]]]
    # Truncated mesh range
    R = ([-30,30],[-30,30],[-30,0])
    

    # get resistivity values at regular gird position
    t2 = default_timer()
    meshPoints, meshValues = extractData(mesh,res,R)
    
    resValues = interpData(xyz, meshPoints,meshValues)
    t3 = default_timer()
    
    hetTemp = runSimu(mesh, res)
    hetTemp.set('k', 1.0/ (hom('u') / hom('i')))
    hetTemp.set('rhoa', hetTemp('k') * hetTemp('u') / hetTemp('i'))
    
    xyz_list = np.append(xyz_list, xyz, axis = 0)
    a_list = np.append(a_list,resValues)
    u_list = np.append(u_list,hetTemp['rhoa'])
    
    t4 = default_timer()
    print('iteration: ', i ,' time: ',t3-t2)
    print('iteration: ', i ,' time: ',t4-t1)
    # print(hetTemp['rhoa'])

mesh.exportVTK('mesh')
# scipy.io.savemat('/Users/zhouji/Documents/github/ERT/generateData/data1000_6.mat', mdict={'coord':xyz_list, 'a': a_list,'u': u_list})

# %%
## Visualization
pg.show(mesh)

# %%
import sys
sys.path.append("..")
from plot3D import plotData, plotGrid

# %%
plotData(hetTemp,nPerLine,1)
# %%
values = np.reshape(resValues,[Griddim[1],Griddim[0],Griddim[2]])

plotGrid(values,Griddim)

