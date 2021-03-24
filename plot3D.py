from plotUtil import showERTData,drawERTData,midconfERT,generateConfStr
from math import pi
import numpy as np
from numpy import ma

import pygimli as pg
from pygimli.viewer.mpl.dataview import showValMapPatches
import pyvista as pv


    
def plotData(data, ndev, lineToPlot):
    ndev = ndev
    line = lineToPlot
    ndataPerline = np.sum(np.arange(ndev-2))
    Start = line*ndataPerline
    End = (line+1)*ndataPerline

    fig = pg.plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    return drawERTData(ax,data,vals = None,Start = Start,End = End)


def plotGrid(values, dim):
    grid = pv.UniformGrid()
    grid.dimensions = np.array(values.shape) + 1
    grid.spacing = (1/dim[0], 1/dim[1], 1/dim[2])
    grid.cell_arrays["values"] = values.flatten(order="F")  # Flatten the array!
    grid.plot(show_edges=True,opacity=0.4)