import plotly



import plotly.graph_objects as go
import plotly.io as pio
import matplotlib.cm
import numpy as np
import matplotlib.pyplot as plt
from plotly.subplots import make_subplots

import matplotlib.pyplot as plt

pio.renderers.default='browser'

# -*- coding: utf-8 -*-
"""
Created on Mon Jun  5 08:25:02 2023

@author: Roni
"""


def scatter3d(fig,data,color,size,legend,opa = 1,colorscale = 'gray',show_colorbar=True,mode='markers'):
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
        mode=mode,
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

def plot_projections(pt_cloud,frames_per_cam,color = 'red', ax = None,size= 2):
    if ax is None:
        fig,ax = plt.subplots(2, 2) 
          
    # pt_cloud =pt_cloud.copy()
    for idx in range(4):
        # vertices_homo = frames_per_cam[idx].homogenize_coordinate(pt_cloud) if homogenize == True else pt_cloud
        # vertices_homo = np.append(cm_point,1)[np.newaxis]
        points2d = frames_per_cam[idx].project_with_proj_mat(pt_cloud)
        ax[idx//2,np.mod(idx,2)].imshow(255-np.array(frames_per_cam[idx].im), cmap = 'gray')
        # ax[idx//2,np.mod(idx,2)].scatter(frames_per_cam[idx].pixels[:,1],frames_per_cam[idx].pixels[:,0] ,color = 'blue', alpha = 0.2, s= 3,cmap = 'gray')
        ax[idx//2,np.mod(idx,2)].scatter(points2d[:,0] ,points2d[:,1] ,color = color, alpha = 1, s= size,cmap = 'gray')
    return  ax

def plot_images(frames_per_cam, ax = None):
    if ax is None:
        fig,ax = plt.subplots(2, 2) 
          
    # pt_cloud =pt_cloud.copy()
    for idx in range(4):
        # vertices_homo = frames_per_cam[idx].homogenize_coordinate(pt_cloud) if homogenize == True else pt_cloud
        # vertices_homo = np.append(cm_point,1)[np.newaxis]
        ax[idx//2,np.mod(idx,2)].imshow(frames_per_cam[idx].im, cmap = 'gray')
        # ax[idx//2,np.mod(idx,2)].scatter(frames_per_cam[idx].pixels[:,1],frames_per_cam[idx].pixels[:,0] ,color = 'blue', alpha = 0.2, s= 3,cmap = 'gray')
    return  ax



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

    
def plot_cones(fig, points, normals,skip = 10,sizeref = 1000,opacity = 0.5):

    fig.add_trace(go.Cone(
    x=points[::skip,0],
    y=points[::skip,1],
    z=points[::skip,2],
    u=normals[::skip,0],
    v=normals[::skip,1],
    w=normals[::skip,2],
    opacity= opacity,
    sizemode="absolute",
    showscale = False,
    sizeref=sizeref))
    fig.update_layout(scene=dict(aspectratio=dict(x=1, y=1, z=0.8),aspectmode = 'data',
                             camera_eye=dict(x=1.2, y=1.2, z=0.6)))


def plot_axis(fig, points, normals,sizeref = 1000,opacity = 0.5,**kwargs):

    fig.add_trace(go.Cone(
    x=[points[0]],
    y=[points[1]],
    z=[points[2]],
    u=[normals[0]],
    v=[normals[1]],
    w=[normals[2]],
    opacity= opacity,
    sizemode="absolute",
    showscale = False,
    sizeref=sizeref,
    **kwargs))
    fig.update_layout(scene=dict(aspectratio=dict(x=1, y=1, z=0.8),aspectmode = 'data',
                             camera_eye=dict(x=1.2, y=1.2, z=0.6)))




def plot_interest_points_hist(width,hight,hist_points,points_to_plot, title):
    fig,ax = plt.subplots(width,hight,sharex = True)
    for idx,points in enumerate(points_to_plot):
        row  = idx // width
        col = idx % width
        ax[row][col].hist(hist_points[:,:,points].flatten())
        ax[row,col].set_title(f'{title[idx]} mean {np.mean(points):.2f} std{np.std(points):.2f})')
        ax[row,col].set_xlabel(f'Reprojection error [pixels]')

    plt.tight_layout()

def plot_cameras_points_hist(width,hight,hist_points):
    fig,ax = plt.subplots(width,hight,sharex = True)
    for idx,points in enumerate(np.hstack(hist_points[:,:,:])):
        
        ax[idx//2,np.mod(idx,width)].hist(points) 
        ax[idx//2,np.mod(idx,width)].set_title(f'Camera{idx + 1} mean {np.mean(points):.2f} std {np.std(points):.2f}')
        ax[idx//2,np.mod(idx,width)].set_xlabel(f'Reprojection error [pixels]')


    plt.tight_layout()


def plot_subplot_hist_wing(width,height,hist_points):
    fig, ax = plt.subplots(height, width, sharex=True, figsize=(12, 6))
    ax = ax.reshape(height, width)  # Ensures consistent 2D indexing
    for idx,hist_data in enumerate(hist_points):
        row  = idx // width
        col = idx % width
        mean = np.mean(hist_data)
        std = np.std(hist_data)
        ax[row][col].hist(hist_data)
        title = f'wing {idx} mean = {mean:.2f}, std = {std:.2f}' 
        ax[row,col].set_title(title)
        ax[row,col].set_xlabel(f'3D distance [mm]')

    plt.tight_layout()



def plot_body_hist(width,height,body_points,hist_points,title):
    fig,ax = plt.subplots(width,height,sharex = True)
    for idx,point in enumerate(body_points):
        histogram_data = hist_points[:,:,body_points[idx]].flatten()
        mean = np.mean(histogram_data)
        std = np.std(histogram_data)
        ax[idx].hist(hist_points[:,:,body_points[idx]].flatten())
        title_str = f'{title[idx]}  mean = {mean:.2f}, std = {std:.2f}' 
        ax[idx].set_title(title_str)