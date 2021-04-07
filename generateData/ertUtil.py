import pygimli.meshtools as mt
import numpy as np
import pygimli as pg
from pygimli.physics.ert import simulate as simulateERT
from scipy.interpolate import griddata

import naturalneighbor

# %%
# Create a cube geobody with low resistivity
def defGeom(dim = [1,2,3], position = [0,0,-10], plc = None, area = 50):
    

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
    
    mesh = mt.createMesh(plc0,area = area)
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

def runSimu(mesh, res , scheme, returnFields=False):
    # map markers 1 and 2 to 10 and 100 Ohm, resp.
    field_het = simulateERT(mesh, res=res, scheme=scheme, sr=False,calcOnly=True, verbose=True,returnFields=returnFields)
    return field_het

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


def nneighborInter(selectPoints, selectValues, request_range):
    nn_interpolated_values = naturalneighbor.griddata(selectPoints, selectValues, request_range)
    return nn_interpolated_values


def plotVista():
    import pyvsita as pv
    mesh.exportVTK('mesh')
    m = pv.read('mesh.vtk')
    m.point_arrays['u'] = hetTemp[0].array()
    
import matplotlib.pyplot as plt
def PlotSection(grid,sec=0,Griddim = (30, 30, 70),interpRange = ([-25,25],[-25,25],[-25,-1])):
    '''plot a section in y direction, sec = 0'''
    g = grid.reshape(Griddim)
    plt.imshow(g[sec].T,origin = 'lower',aspect = 'auto',extent=[interpRange[1][0],interpRange[1][1],interpRange[2][0],interpRange[2][1]])
    plt.colorbar()
    plt.show()