import plotly.graph_objects as go
import plotly.io as pio
import numpy as np
import matplotlib.pyplot as plt
from plotly.subplots import make_subplots



pio.renderers.default='browser'

# -*- coding: utf-8 -*-
"""
Created on Mon Jun  5 08:25:02 2023

@author: Roni
"""
    
def plot_prop_movie(xdata,ydata,color,group_name,
                    marker_size = 7,fig = False,showlegend = True,
                    line_dict = False,mode = 'lines',plot_points = False,
                    yaxis = 'y1',name = False,symbol = 'circle',marker_color = False,idx = False,fig_subplots = False):
    fig = go.Figure() if fig == False else fig
    name = group_name if name == False else name + group_name
    marker_color =  f'rgba{str(tuple(np.append(color,0.5)))}' if marker_color == False else marker_color
    line_dict = {'width': 5,'dash':'solid','color' : f'rgba{str(tuple(np.append(color,0.5)))}'} if line_dict == False else line_dict
    traces = go.Scattergl(
        x=xdata,
        y=ydata,
        legendgroup = group_name,
        name = name,
        showlegend = showlegend,
        mode=mode,
        line  = line_dict,
        marker=dict(color = marker_color,size = marker_size, symbol = symbol),
        yaxis = yaxis)
    if idx != False: 
        fig_subplots.add_trace(traces, row=idx + 1, col=1)
    else:
        fig.add_traces(traces)


    
    if plot_points != False:
        fig.add_scatter(x=[plot_points[0]],
            y=[plot_points[1]],
            marker=dict(
                color= f'rgba{str(tuple(np.append(color,0.5)))}',
                size=10
            ))

    return fig

def only_scatter3d(fig,data, name,skip = 1,**kwargs):
    fig.add_trace(go.Scatter3d(
    x=data[0:-1:skip,0],
    y=data[0:-1:skip,1],
    z=data[0:-1:skip,2],
    **kwargs,
    name = name,
    ))
    return fig

def scatter_3d(fig,data,hover_data, name,**kwargs):
    """3d plotly scatter

    Args:
        fig (plotly): plotly figure
        data (np array): contins x,y,z (axis per column)
        hover_data (_type_): data to show when hoverinb, such as time
        name (string): name of poins

    Returns:
        fig (plotly): plotly figure
    """
    hover_data = hover_data.astype(int) if isinstance(hover_data,int) else hover_data
    fig.add_trace(go.Scatter3d(
    x=data[:,0],
    y=data[:,1],
    z=data[:,2],
    customdata = hover_data,**kwargs,
    name = name,
    ))
    fig.update_traces( hovertemplate='<b>time</b>: %{customdata:,.f}<br>') 
    return fig

def matplotlib_hist(data,xlabel,title,bins = 20):
    
    plt.figure(),plt.hist(data, bins = bins)
    plt.axvline(np.nanmean(data), color='k', linestyle='dashed', linewidth=1)
    plt.title(title)
    plt.xlabel(xlabel)

def plot_3d_traj(data,plot_cofnig,mov_name,exp_name,color_prop):
    """plot 3d trajectory

    Args:
        data (dict): dictionary contining cm, time, x_vector, y_vector, start_pert_endpert, color_prop
        plot_cofnig (dict):fly_samples - delta sample of the fly axes .
                                    traj_samples - delta samle of cm 
                                    size_x - scale of body axis
                                    size_y - scale of y axis
                                    delta_y_on_x - location of y axis on x axis (make it look like a cross)

        mov_name (str): name of movie
        exp_name (str): name of experiment
        color_prop (str): name of property to color cm 

    Returns:
        fig
    """
    fig = go.Figure()

    scene=dict(camera=dict(eye=dict(x=1., y=1, z=1.25)),
    xaxis=dict(nticks=10),
    yaxis=dict(nticks=10),
    zaxis=dict(nticks=10),
    aspectmode='data' 
    )

    fig.update_scenes(scene)


    vector_kwargs = {'marker':dict(size=5,opacity=0.8,color = 'red'),'line':dict(color='black' ,width=8)}
    cm_kwargs = {'marker':dict(size=8,opacity=0.8,color =data[color_prop][::plot_cofnig['traj_samples']],colorbar=dict(thickness=10,title = color_prop),colorscale='jet',colorbar_x = -0.1),'mode' : 'markers'}
    pertubations = {'marker':dict(size=13,opacity=0,color = ['pink','green','red']),'mode' : 'markers'}

    start_pert_endpert = data['start_pert_endpert'] 

    fig = scatter_3d(fig,data['cm'][::plot_cofnig['traj_samples'],:],
                    data['time'][::plot_cofnig['traj_samples']],name = 'CM',**cm_kwargs)

    fig = scatter_3d(fig,data['x_vector'],data['time'][::plot_cofnig['fly_samples']],name = 'X vector',**vector_kwargs)
    fig = scatter_3d(fig,data['y_vector'],data['time'][::plot_cofnig['fly_samples']],name = 'Y vector',**vector_kwargs)
    
    if 'forces' in data:
        forces_kwargs = {'marker':dict(size=5,opacity=0.8,color = 'blue'),'line':dict(color='blue' ,width=5)}
        fig = scatter_3d(fig,data['forces'],data['time'][::plot_cofnig['fly_samples']],name = 'forces',**forces_kwargs)
    fig = scatter_3d(fig,data['cm'][start_pert_endpert,:],data['time'][start_pert_endpert],name =  'pertubations',**pertubations)
    fig.update_layout( title = f'{mov_name} {exp_name}',coloraxis_colorbar_y=0.5,coloraxis_colorscale='jet' )
    return fig



def add_point_to_plot(x,y,legend_group_name,color,fig,showlegend = False,yaxis = 'y1'):
    fig.add_trace(go.Scattergl(x=[x],
                    y=[y],
                    marker=dict(
                        color=f'rgba{str(tuple(np.append(color,1)))}',
                        size=10,
                    ),
            legendgroup = legend_group_name,
            showlegend = showlegend,
            yaxis = yaxis))
    return fig









@staticmethod
def disconnect_line_add_none(array1,array2):
    """
    Combines two 2D arrays such that the rows alternate between the two arrays,
    and inserts `None` (represented as `np.nan`) every two rows.

    Args:
        array1 (numpy.ndarray): The first input array with shape (n, 3).
        array2 (numpy.ndarray): The second input array with shape (n, 3).

    Returns:
        numpy.ndarray: The combined array with alternating rows from `array1` and `array2`,
        and `None` (np.nan) inserted every two rows.

    """
    combined_array = np.empty((2 * array1.shape[0], array1.shape[1]))
    combined_array[::2] =array1[:array1.shape[0]]
    combined_array[1::2] =  array2[:array1.shape[0]]
    combined_array = np.insert(combined_array,range(2, combined_array.shape[0], 2),np.nan,axis = 0)
    return combined_array


def histogram(data,name,xaxis, title,opacity = 0.5,histnorm = 'percent',xbins = dict(start=0,end=100, size=5),
              yaxis = ''):
    
    hist_trace = go.Histogram(
            x=data,
            opacity=opacity,
            name=name,
            histnorm=histnorm,  # To normalize to a probability density
            xbins=xbins
    )
    # Create layout
    layout = go.Layout(
        title=title,
        xaxis=dict(title=xaxis),
        yaxis=dict(title=yaxis),
        barmode='overlay',  # To overlay histograms
    )
    return hist_trace,layout


def subplot_histograms_delta_prop(time_vec,experiments,prop,color_map,xbins,wing_body):
    fig = make_subplots(rows=len(time_vec), cols=len(experiments))

    for (i,tfin) in enumerate(time_vec):
        for (j,exp) in enumerate(experiments.values()):
            color = color_map[j]
            delta_v = exp.get_delta_prop_movies(prop,wing_body, time_prop = 'time', t_fin = tfin, delta_frames = 300) 
            fig.add_trace(
                go.Histogram(x=delta_v,legendgroup=exp.experiment_name,
                            marker_color = f'rgba{str(tuple(np.append(color,1)))}',
                            showlegend= i==0,
                            name = exp.experiment_name,
                            xbins = xbins),
                row=i+1,
                col=j+1,
                )
          
        fig.update_yaxes(title_text=f'Time Fin: {tfin}', row=i + 1, col=1)
        fig.for_each_xaxis(lambda x: x.update( {"range":[xbins['start'], xbins['end']]}))

    fig.update_layout(title = prop)
    return fig
            



def vplot(data,fig,propx,propy,name):

        # Creating violin plot using graph objects
    fig.add_trace(go.Violin(x=data[propx], y=data[propy],
                            box_visible=True,
                            points='all',
                            name = name, 
                            ))
    fig.update_layout(xaxis_title=propx,yaxis_title=propy)

        

    