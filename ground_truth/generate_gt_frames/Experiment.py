import plotly.graph_objects as go
import plotly.io as pio
import plotly.express as px
import matplotlib.cm
import numpy as np
import os
import plotly
import matplotlib.pyplot as plt
import pandas as pd
from plotly.subplots import make_subplots
import h5py
from Movie import Movie
import matplotlib.cm as colormap
import Plotters 

pio.renderers.default='browser'

class Experiment():
    def __init__(self,loadir,exp_name, movie_name_list = False, movie_length = 3000):    
        self.experiment = h5py.File(f'{loadir}/{exp_name}.hdf5', "r")
        self.experiment_name = exp_name
        self.pertubation_name = self.experiment_name.split('_')[-1]
        self.color_map = colormap.datad["tab10"]['listed']
        self.mov_names = list(self.experiment.keys()) if movie_name_list == False else movie_name_list
        self.pertubation = int(self.pertubation_name.split('ms')[0]) if self.pertubation_name.split('ms')[0].isdigit() == True else False
        self.loadir =loadir
        time_idx = np.where(self.experiment[self.mov_names[0]]['body'].attrs['header'] == 'time')[0][0]
        self.exp_dict = {mov : Movie(self.experiment,mov,pertubation = self.pertubation) for mov in self.mov_names if self.del_initial_tim_and_length(self.experiment[mov],movie_length,time_idx) !=True}
        self.mov_names = list(self.exp_dict.keys())
        self.body_header = self.exp_dict[self.mov_names[0]].header['body']
        self.wing_header = self.exp_dict[self.mov_names[0]].header['wing']
        self.interest_points = {}

        self.figures_path = f'{self.loadir}/figures/{self.experiment_name.split("manipulated_")[1]}'
        if not os.path.exists(self.figures_path): os.makedirs(self.figures_path)

    
        self.csv_path = f'{self.loadir}/csv/{self.experiment_name.split("manipulated_")[1]}'
        if not os.path.exists(self.csv_path): os.makedirs(self.csv_path)

        self.experiment.close()

    def pqr_movies(self, mov = False):
        mov_names = self.mov_names if mov == False else mov
        [self.get_mov(mov_name).calculate_pqr_update_data_header() for mov_name in mov_names]



    def plot_3d_traj_movies(self,color_prop,save_plot = False, mov = False,**kwargs):
        mov_names = self.mov_names if mov == False else mov
        for mov in  list(mov_names):
            try:
                fig = self.exp_dict[mov].plot_3d_traj_movie(color_prop,**kwargs)
                plotly.offline.plot(fig, filename=f'{self.figures_path}/traj_3d_{mov}.html',auto_open=False) if save_plot == True else fig.show()
            except:
                continue
            

    def mean_mean_props_movies(self,prop1,prop2,wing_body,header_name, mov = False):
        mov_names = self.mov_names if mov == False else mov
        [self.get_mov(mov_name).mean_props(prop1,prop2,wing_body,header_name) for mov_name in mov_names]

    def mean_prop_time_vector_movies(self,prop,delta_t,t_vec, mov = False,**kwargs):
        mov_names = self.mov_names if mov == False else mov
        return pd.DataFrame(np.vstack([self.get_mov(mov_name).mean_prop_time_vector(prop,delta_t,t_vec,**kwargs) for mov_name in mov_names]), columns = [prop,'t0'])

    def calculate_model_movies(self,add_drag = True,**kwargs):
        
        header = 'no_drag' if add_drag == False else 'drag'
        self.calculate_model_nog(add_drag = add_drag,**kwargs) 
        self.calculate_model(add_drag = add_drag,**kwargs) 
        self.project_prop_all_axes_movies(f'model_x_{header}',header_name = f'model_x_{header}',three_col = 2,ax_to_proj = 'X_x_body_projected')  
        self.project_prop_all_axes_movies(f'model_x_{header}',header_name = f'model_y_{header}',three_col = 2,ax_to_proj = 'Y_x_body_projected')
        self.project_prop_all_axes_movies(f'model_gamma_x_{header}',header_name = f'model_gamma_x_{header}',three_col = 2,ax_to_proj = 'X_x_body_projected',wing_body = 'mean_body') 
        self.project_prop_all_axes_movies(f'model_gamma_x_{header}',header_name = f'model_gamma_y_{header}',three_col = 2,ax_to_proj = 'Y_x_body_projected',wing_body = 'mean_body')

    
    def min_max_point_movies(self,prop,**kwargs):
        return np.hstack([self.exp_dict[mov_name].min_max_point(prop,**kwargs) for mov_name in self.mov_names])
     
    def calc_drag_movies(self,**kwargs):
        [self.get_mov(mov_name).calculate_drag(**kwargs) for mov_name in self.mov_names]
 
    def calc_cone_angle_movies(self,**kwargs):
        [self.get_mov(mov_name).calc_cone_angle(**kwargs) for mov_name in self.mov_names]
        

    def th_for_response_time(self,th_mean):
        # th_mean - mean of all frames from the beginning of the movie to the t = 0. the purpose is to add to the analysis only flies that do not accelerate before
        #
        mean_acc = []
        for mov_name in self.mov_names:
            mov = self.get_mov(mov_name)
            acc_norm  = mov.get_prop('acc_norm',wing_body='body')

            if np.nanmean(mov.get_prop('acc_norm',wing_body='body')[0:mov.ref_frame]) < th_mean:
                mean_acc.append(np.nanmean(acc_norm[0:mov.ref_frame]))
        return np.nanmean(mean_acc) + np.nanstd(mean_acc)*3

    
    def th_for_velocity_dec(self, time_to_sample = 80):
        # th_mean - mean of all frames from the beginning of the movie to the t = 0. the purpose is to add to the analysis only flies that do not accelerate before
        mean_acc = []
        for mov_name in self.mov_names:
                mov = self.get_mov(mov_name)
                acc_norm  = mov.get_prop('CM_dot_x_projected',wing_body='body')
                time = mov.get_prop('time','body')
                row,col = np.where(time > time_to_sample)
                if len(row) > 0:
                    if acc_norm[mov.ref_frame] > acc_norm[row[0]]:
                        mean_acc.append(np.nanmean(acc_norm[mov.ref_frame] - acc_norm[row[0]]))
        # mean_std = np.nanmean(mean_acc) + np.nanstd(mean_acc) if len(mean_acc) > 0 else 10
        
        return mean_acc

    def rotate_prop_movies(self,prop,header,**kwargs):
        [self.get_mov(mov_name).rotate_prop(prop,header,**kwargs) for mov_name in self.mov_names]
 

    def calc_force_movies(self,**kwargs):
        [self.get_mov(mov_name).calc_force(**kwargs) for mov_name in self.mov_names]

    def diff_model_exp_movies(self,model_name,exp_name,**kwargs):
        rms_list = []
        diff_list = []
        for  mov_name in self.mov_names:
            diff,diff_exp,rms = self.get_mov(mov_name).diff_model_exp(model_name,exp_name,**kwargs) 
            rms_list.append(rms)
            diff_list.append(np.hstack((diff,diff_exp)))
        return np.vstack(diff_list),rms_list

    def zero_velocity_movies(self,prop):
        zero_v_list = [self.exp_dict[mov_name].zero_velocity(prop) for mov_name in self.mov_names]
        return [velocity for velocity in zero_v_list if np.min(velocity) != None]
    
    def add_mean_prop_movies(self,prop_name,wing_body_prop,wing_body_mean_save,mov = False,phi_idx_to_mean = 'phi_rw_min_idx'):
        mov_names = self.mov_names if mov == False else mov
        [self.get_mov(mov_name).mean_prop_stroke(prop_name,wing_body_prop,wing_body_mean_save,phi_idx_to_mean = phi_idx_to_mean) for mov_name in mov_names]

    def plot_prop_movies(self,prop,wing_body,fig,mov = False,case = 'plot_mov',prop_x = 'time',
                         add_horizontal_line = 0,color = False,legend = False,**kwargs):
        mov_names = self.mov_names if mov == False else mov
        legend = mov_names if legend == False else legend

        color = [self.color_map[idx%len(self.color_map)] for idx,mov_name in enumerate(mov_names)] if color == False else color 
        if 'plot_exp' == case: [self.get_mov(mov_name).plot_prop(prop,wing_body,color,legend,fig,showlegend = idx == 0,prop_x = prop_x,**kwargs) for idx,mov_name in enumerate(mov_names)]
        if 'plot_mov' == case: [self.get_mov(mov_name).plot_prop(prop,wing_body,color[idx],legend[idx],fig,prop_x = prop_x,**kwargs) for idx,mov_name in enumerate(mov_names)]

        if add_horizontal_line != None: fig.add_hline(y=add_horizontal_line, line_width=3, line_color="lime")
        fig.add_vline(x=0, line_width=3, line_color="lime")
        if self.pertubation != False: fig.add_vline(x=self.pertubation, line_width=3, line_color="red")
        fig.update_layout( xaxis_title = prop_x, yaxis_title = prop)    
        # fig.layout.yaxis.scaleanchor="x"
        # fig.update_layout(autosize=False )
    def get_allexp_props(self,prop,wing_body = 'body'):
        return [self.get_mov(mov_name).get_prop(prop,wing_body=wing_body) for mov_name in self.mov_names]
    

    def norm_prop_movies(self,prop,header,**kwargs):
        [self.get_mov(mov_name).norm_prop(prop,header,**kwargs) for  mov_name in self.mov_names]



    def smooth_prop_movies(self,prop,derives,wing_body):
        [self.get_mov(mov_name).smooth_and_derive(prop,derives,wing_body) for  mov_name in self.mov_names]

    def get_peaks_movies(self,prop,case,**kwargs):
        self.interest_points[f'{prop}_{case}'] =  np.vstack([self.get_mov(mov_name).get_peaks_min_max(prop,case = case,**kwargs) for  mov_name in self.mov_names])
    
    def subtract_interest_time_from_time(self,prop,header):
        time_idx = self.body_header['time']
        [self.get_mov(mov_name).add_time_m_t0( self.interest_points[prop][idx][time_idx],header,wing_body = 'body') for  idx,mov_name in enumerate(self.mov_names)]
    
    def add_point_to_plot_movies(self,name_point,ydata,fig,**kwargs):
        [self.get_mov(mov_name).add_point_to_plot(self.interest_points[name_point][idx,:],ydata,fig,self.color_map[idx%len(self.color_map)],wing_body = 'body',**kwargs) for idx,mov_name in enumerate(self.mov_names)]
          
    def interest_point_hist(self,point_name,prop = 'time',**kwargs):
        return Plotters.histogram(self.interest_points[point_name][:,self.body_header[prop]],self.experiment_name,prop, point_name,**kwargs)

    def get_delta_prop_movies(self,prop,wing_body,**kwargs):
        return [self.get_mov(mov_name).get_delta_prop(prop,wing_body,**kwargs) for  mov_name in self.mov_names if self.get_mov(mov_name).get_delta_prop(prop,wing_body,**kwargs) != None]   



    def get_prop_on_time_movies(self,prop,wing_body,**kwargs):
        return [self.get_mov(mov_name).get_prop_on_time(prop,wing_body,**kwargs) for  mov_name in self.mov_names if self.get_mov(mov_name).get_delta_prop(prop,wing_body,**kwargs) != None]   

    def delta_ang_all_time_movies(self,prop1_name,prop2_name,header,**kwargs):
        [self.get_mov(mov_name).delta_ang_all_time(prop1_name,prop2_name,header,**kwargs) for  mov_name in self.mov_names]

    def calculate_freq_movies(self, idx_prop, mean_wing_body):
        [self.get_mov(mov_name).calculate_freq(idx_prop, mean_wing_body) for  mov_name in self.mov_names]
    
    def calculate_phi_amp_movies(self, wing,mean_wing_body):
        [self.get_mov(mov_name).calculate_phi_amp(wing,mean_wing_body) for  mov_name in self.mov_names]

    def mean_by_stroke_movies(self, prop,mean_wing_body,wing_body):
        [self.get_mov(mov_name).mean_by_stroke(prop,mean_wing_body,wing_body) for  mov_name in self.mov_names]


    def acc_dir_movies(self, t,prop):
        return np.vstack([self.get_mov(mov_name).acc_dir(t,prop) for  mov_name in self.mov_names])

    def vel_dir_movies(self,prop):
        return np.vstack([self.get_mov(mov_name).vel_dir(prop) for  mov_name in self.mov_names])


    def angles_between_vector_and_ref_frame_movies(self,prop,header):
        [self.get_mov(mov_name).angles_between_vector_and_ref_frame(prop,header,wing_body = 'body') for  mov_name in self.mov_names]


    def project_axes_movies(self,header,**kwargs):
        [self.get_mov(mov_name).project_axes_xy(header,**kwargs) for  mov_name in self.mov_names]

    def project_prop_movies(self,prop_to_project,**kwargs):
        [self.get_mov(mov_name).project_prop(prop_to_project,**kwargs) for  mov_name in self.mov_names]

    def project_prop_all_axes_movies(self,prop_to_project,**kwargs):
        [self.get_mov(mov_name).project_prop_all_axes(prop_to_project,**kwargs) for  mov_name in self.mov_names]

    def sub_two_props_movies(self,prop1,prop2,wing_body,header):
        [self.get_mov(mov_name).sub_two_props(prop1,prop2,wing_body,header) for  mov_name in self.mov_names]

    def from_vector_to_wing_body(self,prop,add_to_vectors):
        [self.get_mov(mov_name).from_wing_body_to_vectors(prop,add_to_vectors[0],add_to_vectors[1]) for  mov_name in self.mov_names]


    def substruct_first_frame(self,prop,wing_body):
        [self.get_mov(mov_name).sub_ref_frame(prop,wing_body) for  mov_name in self.mov_names]

    def velocity_amplitude(self,prop_x,prop_y,wing_body):
        [self.get_mov(mov_name).v_size(prop_x,prop_y,wing_body) for  mov_name in self.mov_names]

    def delta_ref_ang_movies(self,ref_frame_axis,header,**kwargs):
        [self.get_mov(mov_name).delta_ang_ref_frame(ref_frame_axis,header,**kwargs) for  mov_name in self.mov_names]


    def calculate_model_nog(self,**kwargs):
        [self.get_mov(mov_name).calculate_model_nog(**kwargs) for  mov_name in self.mov_names]


    def calculate_model(self,**kwargs):
        [self.get_mov(mov_name).calculate_model(**kwargs) for  mov_name in self.mov_names]

    def xy_body_on_sp_movies(self,**kwargs):
        [self.get_mov(mov_name).xy_body_on_sp(**kwargs) for  mov_name in self.mov_names]


    def save_to_csv_movies(self,mov = False,**kwargs):
        mov_names = self.mov_names if mov == False else mov
        [self.get_mov(mov_name).save_to_csv(self.csv_path,**kwargs) for  mov_name in mov_names]



    def get_mov(self,mov_name):
        return self.exp_dict[mov_name]
    
    def mean_time_series_prop(self,prop,wing_body,window = (73*7)//2,dt = 1/16000):
        
        t_list = np.arange(-40,400,dt*1000)
        cmlist = np.full((len(self.mov_names),len(t_list)),np.nan)
        for idx_mov,mov_name in enumerate(self.mov_names):
            mov = self.get_mov(mov_name)
            time = mov.get_prop('time',wing_body)[window:,0]

            cm_dot = mov.get_prop(prop,wing_body)[window:,0]
            # cm_dot = cm_dot - np.mean(cm_dot[window:mov.ref_frame + 1])
            time_inter,tlist_idx,time_idx = np.intersect1d(t_list,time,return_indices = True )
            cmlist[idx_mov,tlist_idx] = cm_dot[time_idx]
            mean_prop = np.nanmean(cmlist,axis = 0)
        return mean_prop,t_list
    

    
    @staticmethod
    def del_initial_tim_and_length(mov,movie_length,time_idx):
        if ('body' in mov) & ('vectors' in mov):
            if ((mov['body'].shape[0] < movie_length) | (mov['body'][0,time_idx] > 0)):
                return True

    
    
    

    
    