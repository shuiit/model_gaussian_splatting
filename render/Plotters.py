import plotly



import plotly.graph_objects as go
import plotly.io as pio
import matplotlib.cm
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from plotly.subplots import make_subplots


pio.renderers.default='browser'

# -*- coding: utf-8 -*-
"""
Created on Mon Jun  5 08:25:02 2023

@author: Roni
"""


def scatter3d(fig,data,color,size,legend,opa = 1,colorscale = 'gray',show_colorbar=True):
    marker_dict = dict(
        color=color,  # Set marker color
        size=size,  # Set marker size
        colorscale=colorscale,
        opacity=opa
    )
    # Conditionally add the colorbar
    if show_colorbar:
        marker_dict["colorbar"] = dict(title="Colorbar")

    
      
    fig.add_trace(go.Scatter3d(
        x=data[:, 0],
        y=data[:, 1],
        z=data[:, 2],
        mode='markers',
        marker=marker_dict,
        name = legend
    ))
    
    # Update layout to set aspectmode to 'cube'
    fig.update_layout(scene=dict(
        aspectmode='data'  # Ensures x, y, z axes have the same scale
    ))
    return fig

def plot_hull(real_hull,size = 3):
    colors = ['green','red','blue']

    fig = go.Figure()
    [scatter3d(fig,data,color,size) for data,color in zip(real_hull.values(),colors)]

    fig.show()

def scatter_projections_from_gs(frames,gs, plot_image = False):
    
    im_name = list(frames.keys())[0]
    fig,axs = plt.subplots(2,2)
    for cam in range(4):
        image = f'{im_name.split("CAM")[0]}CAM{cam+1}.jpg'
        indices = (gs.color[:,0] < 1) &(gs.color[:,1] < 1) & (gs.color[:,2] < 1) & (gs.color[:,0] > 0) & (gs.color[:,1] > 0) &(gs.color[:,2] > 0) 
        colors = gs.color[indices, :]  # Filtered colors (RGB or RGBA)
        homo_voxels_with_idx = frames[image].add_homo_coords(gs.xyz[indices,0:3])
        proj = frames[image].project_on_image(homo_voxels_with_idx,croped_camera_matrix = True)
        if plot_image == True:
            axs[cam // 2,cam % 2].imshow(frames[image].croped_image,'gray')
            proj[:,1] = 800-proj[:,1]
        axs[cam // 2,cam % 2].scatter(proj[:,0],proj[:,1],s = 1,c = colors)

