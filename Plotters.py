import plotly



import plotly.graph_objects as go
import plotly.io as pio
import matplotlib.cm
import numpy as np
import matplotlib.pyplot as plt
from plotly.subplots import make_subplots
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pickle
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

def plot_projections(pt_cloud,frames_per_cam,color = 'red', ax = None,size= 2,alpha = 1):
    if ax is None:
        fig,ax = plt.subplots(2, 2) 
          
    # pt_cloud =pt_cloud.copy()
    for idx in range(4):
        # vertices_homo = frames_per_cam[idx].homogenize_coordinate(pt_cloud) if homogenize == True else pt_cloud
        # vertices_homo = np.append(cm_point,1)[np.newaxis]
        points2d = frames_per_cam[idx].project_with_proj_mat(pt_cloud)
        ax[idx//2,np.mod(idx,2)].imshow(255-np.array(frames_per_cam[idx].im), cmap = 'gray')
        # ax[idx//2,np.mod(idx,2)].scatter(frames_per_cam[idx].pixels[:,1],frames_per_cam[idx].pixels[:,0] ,color = 'blue', alpha = 0.2, s= 3,cmap = 'gray')
        ax[idx//2,np.mod(idx,2)].scatter(points2d[:,0] ,points2d[:,1] ,color = color, alpha = alpha, s= size,cmap = 'gray')
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


def boxplot2(delta_angles, chamfer_3d, angle_name,ax, showfliers=False,cmap_name = 'viridis', xdata = False,**kwargs):
    n_samples = chamfer_3d.shape[0]

    # Flatten data for seaborn
    data_long = {
        f'delta_{angle_name}': np.repeat(delta_angles, n_samples),
        'chamfer_dist': chamfer_3d.T.flatten()
    }
    df_long = pd.DataFrame(data_long)

    # Get unique sorted delta values
    unique_deltas = np.sort(np.unique(delta_angles))

    # Create gradient color list from colormap
    cmap = cm.get_cmap(cmap_name)  # You can change to 'plasma', 'Blues', etc.
    colors = [cmap(i) for i in np.linspace(0.3, 1, len(unique_deltas))]

    # Create a mapping from delta value to color
    palette = {delta: color for delta, color in zip(unique_deltas, colors)}

    # Plot

    xdata = f'delta_{angle_name}' if xdata == False else xdata

    sns.boxplot(
        data=df_long,
        x=xdata,
        y='chamfer_dist',
        palette=palette,
        showfliers=showfliers,
        ax = ax,**kwargs
    )
    # sns.stripplot(
    #     data=df_long,
    #     x=xdata,
    #     y='chamfer_dist',
    #     color='black',
    #     size=4,
    #     jitter=True,
    #     ax = ax
    # )



def boxplot(delta_angles, chamfer_3d, angle_name,ax, showfliers=False,cmap_name = 'viridis'):
    n_samples = chamfer_3d.shape[0]

    # Flatten data for seaborn
    data_long = {
        f'delta_{angle_name}': np.repeat(delta_angles, n_samples),
        'chamfer_dist': chamfer_3d.T.flatten()
    }
    df_long = pd.DataFrame(data_long)

    # Get unique sorted delta values
    unique_deltas = np.sort(np.unique(delta_angles))

    # Create gradient color list from colormap
    cmap = cm.get_cmap(cmap_name)  # You can change to 'plasma', 'Blues', etc.
    colors = [cmap(i) for i in np.linspace(0.3, 1, len(unique_deltas))]

    # Create a mapping from delta value to color
    palette = {delta: color for delta, color in zip(unique_deltas, colors)}

    # Plot


    sns.boxplot(
        data=df_long,
        x=xdata,
        y='chamfer_dist',
        palette=palette,
        showfliers=showfliers,
        ax = ax
    )
    sns.stripplot(
        data=df_long,
        x=xdata,
        y='chamfer_dist',
        color='black',
        size=4,
        jitter=True,
        ax = ax
    )



def boxplot_v2(df_long, data_for_colors, xtick, ax, showfliers=False,cmap_name = 'viridis',**kwargs):
   
    # Create gradient color list from colormap
    cmap = cm.get_cmap(cmap_name)  # You can change to 'plasma', 'Blues', etc.
    colors = [cmap(i) for i in np.linspace(0.3, 1, len(data_for_colors))]

    # Create a mapping from delta value to color
    palette = {delta: color for delta, color in zip(data_for_colors, colors)}

    # Plot


    sns.boxplot(
        data=df_long,
        x=xtick,
        y='chamfer_dist',
        palette=palette,
        showfliers=showfliers,
        ax = ax,**kwargs
    )
    sns.stripplot(
        data=df_long,
        x=xtick,
        y='chamfer_dist',
        color='black',
        size=4,
        jitter=True,
        ax = ax
    )


    

def plot_chamfer_frames_th(angle_name,chamfer_dist,chamfer_stats,x,colors,ax):
    sort_lines = np.argsort(chamfer_stats[f'delta {angle_name}'].to_numpy())
    chamfer_dist = chamfer_dist[:,sort_lines]
    chamfer_stats = chamfer_stats.iloc[sort_lines]

    chamfer_stats_th = [
        np.sum(chamfer_dist < th_3d, axis=0) / np.vstack(chamfer_dist).shape[0] * 100 
        for th_3d in x
    ]
    chamfer_stats_th = np.vstack(chamfer_stats_th)  # shape: (len(x), num_angles)
    
    num_lines = chamfer_stats_th.shape[1]
    



    for i in range(num_lines):
        ax.plot(x.astype(np.int16), chamfer_stats_th[:, i], color=colors[i], label=chamfer_stats[f'delta {angle_name}'].iloc[i])
    
    ax.set_ylabel('Frames [%]')
    ax.set_xlabel('CD threshold [μm]')
    ax.set_ylim(0)
    ax.set_xlim(0,x.max()*1.05)
    ax.legend(title=f'$\delta ^\circ $')



def wing_subplot_box(delta_angles,path_output,plot_body_wing,angle_name, ax, yticks, cmap = 'turbo',**kwargs ):
    camfer = {}
    angle = ['phi','theta','psi']
    for idx,angle in enumerate(angle):
        wing_side = ['right', 'left']
        for wing_side in wing_side:
            phi_names = f'fly_{angle}_{wing_side}_delta10_sweep_m40_40_try'
            with open(f'{path_output}/{phi_names}/chamfer.pkl', 'rb') as f: 
                camfer[wing_side] = pickle.load(f)
        wings_phi_chamfer = np.vstack((camfer['right'][plot_body_wing][0],camfer['left'][plot_body_wing][0]))
        chamfer_df = pd.DataFrame(wings_phi_chamfer)
        

        boxplot2(delta_angles.astype(np.int16),chamfer_df.fillna(np.nan).to_numpy(),angle_name,ax[idx],showfliers = False, cmap_name=cmap,**kwargs)

        ax[idx].set(xlabel=None)
        if idx != 0:
            ax[idx].set(ylabel=None)
        ax[idx].set_yticks(yticks)
        ax[idx].set_ylim([min(yticks),max(yticks)])
        ax[idx].set_title(f"$\{angle}$")
            # Labeling the axis directly
        ax[idx].set_ylabel("CD [µm]")
        xlabel = f"$\delta ^\circ ${angle_name}"
        ax[idx].set_xlabel(xlabel.capitalize())

def body_subplot_box(delta_angles,path_output,plot_body_wing,angle_name, ax, yticks, cmap = 'turbo' ):

    angle = ['yaw','pitch','roll']
    for idx,angle in enumerate(angle):
        body_names = f'fly_{angle}_delta10_sweep_m40_40_try'
        with open(f'{path_output}/{body_names}/chamfer.pkl', 'rb') as f: 
            chamfer_body = pickle.load(f)
        chamfer_df = pd.DataFrame(chamfer_body[plot_body_wing][0])
        boxplot2(delta_angles.astype(np.int16),chamfer_df.fillna(np.nan).to_numpy(),angle_name,ax[idx],showfliers = False, cmap_name=cmap)
        if idx != 0:
            ax[idx].set(ylabel=None)
        ax[idx].set_yticks(yticks)
        ax[idx].set_ylim([min(yticks),max(yticks)])
        # ax[idx].set_title(angle)
        ax[idx].set_ylabel("CD [µm]")
        xlabel = f"$\delta ^\circ ${angle_name}"

        ax[idx].set_xlabel(xlabel.capitalize())


def wing_subplot_th(path_output,plot_body_wing,ax,colors):

    camfer = {}
    angle = ['phi','theta','psi']
    for idx,angle in enumerate(angle):
        wing_side = ['right', 'left']
        for wing_side in wing_side:
            phi_names = f'fly_{angle}_{wing_side}_delta10_sweep_m40_40_try'
            with open(f'{path_output}/{phi_names}/chamfer.pkl', 'rb') as f: 
                camfer[wing_side] = pickle.load(f)
        wings_phi_chamfer = np.vstack((camfer['right'][plot_body_wing][0],camfer['left'][plot_body_wing][0]))
        chamfer_df = pd.DataFrame(wings_phi_chamfer)
        stats3d = camfer[wing_side][plot_body_wing][2]
        plot_chamfer_frames_th(angle,chamfer_df.fillna(np.nan).to_numpy(),stats3d,np.arange(0.01*1000,0.3*1000,5),colors,ax[idx])
        ax[idx].set(xlabel=None)
        # 
        if idx != 0:
            ax[idx].get_legend().remove()
            ax[idx].set(ylabel=None)
        if idx == 0:
            ax[idx].legend(title='$\Delta$ angle',
                title_fontsize='x-small',
                )
        ax[idx].set_title(f"$\{angle}$")


def body_subplot_th(path_output,plot_body_wing,ax,colors):
    angle = ['yaw','pitch','roll']
    for idx,angle in enumerate(angle):
        body_names = f'fly_{angle}_delta10_sweep_m40_40_try'
        with open(f'{path_output}/{body_names}/chamfer.pkl', 'rb') as f: 
            chamfer_body = pickle.load(f)
        chamfer_df = pd.DataFrame(chamfer_body[plot_body_wing][0])
        stats3d = chamfer_body[plot_body_wing][2]
        plot_chamfer_frames_th(angle,chamfer_df.fillna(np.nan).to_numpy(),stats3d,np.arange(0.01*1000,0.3*1000,5),colors,ax[idx])
        ax[idx].get_legend().remove()
        if idx != 0:
            ax[idx].set(ylabel=None)
        ax[idx].set_title(angle)






