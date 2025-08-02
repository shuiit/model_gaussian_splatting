
import pickle
import os
from Evaluation import Evaluation
from tqdm import tqdm
import Utils
from FlyOutput import FlyOutput
import numpy as np
import pandas as pd
class EvaluateAngleSweep():
    def __init__(self,frames,nominal_initial_angles,input_dir,input_path_for_image,iterations,path_frame,angle_name,model_name,delta_angles =  np.array([  0., -30., -20., -10.,  10.,  20.,  30.])):

        
        self.model_name = model_name
        self.delta_angles = delta_angles
        self.zbuff_hull,self.zbuff_model = {},{}
        res_dir = os.listdir(f'{input_dir}/results') 
        # self.results_dir = [mov_name for res_dir,mov_name in zip(res_dir, nominal_initial_angles) if len(os.listdir(f'{input_dir}/results/{res_dir}')) == len(self.delta_angles)]

        results_dir = [res_dir for res_dir in res_dir if (len(os.listdir(f'{input_dir}/results/{res_dir}')) == len(delta_angles))]



        self.results_dir = [mov_name for mov_name in nominal_initial_angles.keys() if mov_name.split('_')[3] in results_dir]


        self.wing_hull,self.wing_model = {'right':{},'left':{}},{'right':{},'left':{}}
        self.body_hull = {}
        self.input_dir = input_dir
        self.iterations = iterations
        dirs = os.listdir(f'{input_dir}/{list(nominal_initial_angles.keys())[0].split("_")[3]}') 
        self.sweep_size = len([dir for dir in dirs if 'fly' in dir.split('_')])
        dir_names = [name for name in os.listdir(path_frame) if os.path.isfile(os.path.join(path_frame, name))]
        self.ini_angles = {idx:self.open_file(f'{path_frame}/{dir}') for idx,dir in enumerate(dir_names)}
        self.input_path_for_image = input_path_for_image
        self.frames = frames
        self.nominal_initial_angles = nominal_initial_angles
        self.angle_name = angle_name

    def open_file(self,path,ang_dict = {}):
        with open(path,'rb') as f:
            ini_angles = pickle.load(f)
        ang_dict = {ang_name: angles for ang_name,angles in ini_angles.items()}
        return ang_dict
    


    
    def load_frame_all_sweep(self,idx_iter,mov_name,iteration,letedict,frames):
        mov = int(mov_name.split('_')[1]) 
        frame0 = int(mov_name.split('_')[3]) 
        image_path =  f'{self.input_path_for_image}/mov{mov}_2023_08_09_60ms/'
        file_name = f'fly_model_scale_iter{idx_iter}'
        interest_points_path = f'G:/My Drive/Research/gaussian_splatting/gaussian_splatting_input/evaluation/points/mov{mov}'
        
        with open(f'{self.input_dir}/results/{frame0}/{file_name}_results.pkl', 'rb') as handle:
            output_angles_weights = pickle.load(handle)
        
        frame_eval = Evaluation(interest_points_path,image_path,frame0,self.input_dir,output_angles_weights,frame0,iteration,file_name,letedict = letedict,frames_dict = frames)
        for source_attr, target_attr, output_attr in frame_eval.projection_tasks:
            frame_eval.get_projected_and_store(frame_eval, source_attr, target_attr, output_attr)
        return frame_eval
    

    def load_all_sweep(self,letedict):
        # generate frames file if it doesnt exist
        self.sweep = {}
        for mov_name in tqdm(self.results_dir):
            self.sweep[mov_name] = []
            for idx_iter in range(self.sweep_size):
                try:
                    self.sweep[mov_name].append(self.load_frame_all_sweep(idx_iter,mov_name,self.iterations,letedict,self.frames))
                except:
                    self.sweep[mov_name].append('Fail')




    def calculate_wing_chamfer_v2(self):
        for mov_name in tqdm(list(self.sweep.keys())):
                [frame.calculate_chamfer_v2(self.model_name) for frame in self.sweep[mov_name] if frame != 'Fail']  



    def calculate_wing_chamfer(self):
        for mov_name in tqdm(list(self.sweep.keys())):
                [frame.calculate_chamfler(self.model_name) for frame in self.sweep[mov_name] if frame != 'Fail']  


    def hull_calc_zbuff_hull_xbody(self,file_path_save_hull):

        # generate hull file if it doesnt exist (body hull for ground truth)
        if os.path.isfile(file_path_save_hull):
            with open(file_path_save_hull, "rb") as input_file:
                hull_movs = pickle.load(input_file)
        else:
            Utils.make_body_hull_file(self.nominal_initial_angles,file_path_save_hull)
        self.zbuff_hull,self.zbuff_model = {},{}
        for mov_name in tqdm(list(self.sweep.keys())):
            zbuff_hull = Utils.load_body_hull_calc_xbody(self.sweep[mov_name][0], hull_movs[mov_name]) 
            [frame.load_hull_calc_xbody_dot_per_idx(zbuff_hull) for frame in self.sweep[mov_name] if frame != 'Fail']
            [frame.calculate_chamfler_body() for frame in self.sweep[mov_name] if frame != 'Fail']
            [frame.calculate_chamfler_body_2d() for frame in self.sweep[mov_name] if frame != 'Fail']




    def load_sweep(self,sweep_path,file_path_save_hull,letedict = None):
      if os.path.isfile(sweep_path):
        with open(sweep_path, "rb") as input_file:
            self.sweep = pickle.load(input_file)
      else:
          self.load_all_sweep(letedict)
          self.calculate_wing_chamfer_v2()
          self.hull_calc_zbuff_hull_xbody(file_path_save_hull)     
      


          Utils.pickle_file(self.sweep,sweep_path)
          print(f'frames saved: {sweep_path}')

    def get_dist_ptclouds(self,points1,points2):
        closest_to_hull_body = Utils.find_closest_points_inptclouds(points1,points2)
        return np.sqrt(np.sum((closest_to_hull_body - points1)**2, axis = 1))

    def get_wing_body_from_hull(self,hull_points,gt_body,gt_wing):

        closest_to_hull_body = self.get_dist_ptclouds(hull_points,gt_body)
        closest_to_hull_wing = self.get_dist_ptclouds(hull_points,gt_wing)
        wing = closest_to_hull_wing < closest_to_hull_body
        body = closest_to_hull_body < closest_to_hull_wing
        return hull_points[wing],hull_points[body]


    def calculate_zbuffs_hull_model(self,mov_name,path_hull):
            mov = int(mov_name.split('_')[1]) 
            frame = int(mov_name.split('_')[3]) 
            image_path =  f'G:/My Drive/Research/gaussian_splatting/gaussian_splatting_input/mov{mov}_2023_08_09_60ms/'
            hull_to_compare = FlyOutput(image_path,frame,path_hull,None,frame,self.iterations,f'',frames_dict = self.frames)
            self.zbuff_hull[mov_name] = self.sweep[mov_name][0].homog_and_zbuff( hull_to_compare.xyz)
            self.zbuff_model[mov_name] = self.sweep[mov_name][0].homog_and_zbuff(self.sweep[mov_name][0].xyz) 


    def load_hull_for_wing_width(self,path_hull):
        movs_list = []
        for mov_name in list(self.sweep.keys()):
            mov = int(mov_name.split('_')[1]) 
            frame = int(mov_name.split('_')[3]) 
            hull_file = f'{path_hull}/{frame}/point_cloud/iteration_{self.iterations}/point_cloud.ply'
            if os.path.isfile(hull_file):
                movs_list.append(mov_name)
        movs_to_compare = [mov_name for mov_name in  movs_list if (mov_name != 'mov_36_frame_223') & (mov_name != 'mov_121_frame_2939') & 
                           (mov_name != 'mov_40_frame_4690') & (mov_name != 'mov_59_frame_2027') & (mov_name in self.sweep.keys())]

        [self.calculate_zbuffs_hull_model(mov_name,path_hull) for mov_name in tqdm(movs_to_compare)]

    def get_wing_body_from_hull(self,hull_points,gt_body,gt_wing):

        closest_to_hull_body = self.get_dist_ptclouds(hull_points,gt_body)
        closest_to_hull_wing = self.get_dist_ptclouds(hull_points,gt_wing)
        wing = closest_to_hull_wing < closest_to_hull_body
        body = closest_to_hull_body < closest_to_hull_wing
        return hull_points[wing],hull_points[body]



    def get_parts_wing(self,wing,devided_wing,span):
        projected_on_span = np.dot(wing,span)
        devided_bool = [(projected_on_span >= devided_wing[0]) & (projected_on_span <= devided_wing[1]) for devided_wing in devided_wing]
        return  [wing[devided_bool] for devided_bool in devided_bool]
    

    def find_wings_hull(self,mov_name,side_wing):
        wing_gt = getattr(self.sweep[mov_name][0],f'{side_wing}_wing_tagged')
        gt_wing = np.vstack((wing_gt))
        gt_body = np.vstack((self.sweep[mov_name][0].hull_ew))

        wing_hull,body_hull_1 = self.get_wing_body_from_hull(self.zbuff_hull[mov_name],gt_body,gt_wing)
        wing_model,body_model = self.get_wing_body_from_hull(self.zbuff_model[mov_name],gt_body,gt_wing)
        # both_wings = np.vstack([self.sweep[mov_name][0].right_wing_tagged,self.sweep[mov_name][0].left_wing_tagged])
        # wing_hull2,body_hull = self.get_wing_body_from_hull(self.zbuff_hull[mov_name],gt_body,both_wings)

        self.wing_hull[side_wing][mov_name] = wing_hull
        self.wing_model[side_wing][mov_name] = wing_model
        # self.body_hull[mov_name] = body_hull


    def find_wings_from_hull(self,side_wing):

        [self.find_wings_hull(mov_name,side_wing) for mov_name in tqdm(list(self.zbuff_hull.keys()))]


    def devide_wings_to_parts(self,wing,span,num_of_parts = 8):
        projected_on_span = np.dot(wing,span)
        len_span = max(projected_on_span) - min(projected_on_span)
        projected_parts = [projected_on_span[((projected_on_span - min(projected_on_span)) <= (len_span*idx/num_of_parts)) & ((projected_on_span - min(projected_on_span)) >= (len_span*(idx-1)/num_of_parts))]  for idx in range(1,num_of_parts + 1)]
        parts_bound = []
        for idx in range(len(projected_parts) - 1): 
            parts_bound.append([min(projected_parts[idx]),min(projected_parts[idx + 1])])
        parts_bound.append([min(projected_parts[-1]),max(projected_parts[-1])])
        return parts_bound

    

    def devide_wing_hull(self,mov_name,side_wing,num_of_parts = 8):
        span = getattr(self.sweep[mov_name][0],f'{side_wing}_wing_span')
        devided_wing = self.devide_wings_to_parts(self.wing_hull[side_wing][mov_name],span,num_of_parts = num_of_parts)
        
        wing_hull_parts = self.get_parts_wing(self.wing_hull[side_wing][mov_name],devided_wing,span)
        wing_model_parts = self.get_parts_wing(self.wing_model[side_wing][mov_name],devided_wing,span)
        
        wing_model_parts = [wing_part for wing_part in wing_model_parts if len(wing_part) > 5]
        wing_hull_parts = [wing_hull_part for wing_model_part,wing_hull_part in zip(wing_model_parts,wing_hull_parts) if len(wing_model_part) > 5]
        return wing_hull_parts,wing_model_parts

    def calculate_projection_on_z(self,points,mov_name ):
        z_wing = self.sweep[mov_name][0].get_principle_axes(points)[2]  # shape (3,)
        centered = points - np.mean(points, axis=0)  # shape (N, 3)
        return np.abs(np.dot(centered, z_wing) )
    

    def project_on_z(self,wing_parts,mov_name):
    
        return np.hstack([self.calculate_projection_on_z(wing_parts[part_num],mov_name ) for part_num in range(len(wing_parts))])
        

    def devide_wing_project_z(self,mov_name):
        wing_hull_parts_right,wing_model_parts_right = self.devide_wing_hull(mov_name,'right',num_of_parts = 8)
        wing_hull_parts_left,wing_model_parts_left = self.devide_wing_hull(mov_name,'right',num_of_parts = 8)

        hull_wings_project_z = np.hstack((self.project_on_z(wing_hull_parts_right,mov_name),self.project_on_z(wing_hull_parts_left,mov_name)))
        model_wings_project_z = np.hstack((self.project_on_z(wing_model_parts_right,mov_name),self.project_on_z(wing_model_parts_left,mov_name)))
        return np.std(hull_wings_project_z*1000),np.std(model_wings_project_z*1000)
    
    def get_chamfer(self):
        for mov_name in tqdm(list(self.sweep.keys())):
            [frame.calculate_chamfler() for frame in self.sweep[mov_name] if frame != 'Fail']
            
                


    
    def load_1d_chamfer_nominal_to_df(self,xname,model,plot_body_wing = 'wing'):
            path = os.path.dirname(self.input_dir)
            with open(f'{path}/{model}/chamfer.pkl', 'rb') as f: 
                chamfer_body = pickle.load(f)
            data_long = {f'xtick': np.repeat(xname, len(chamfer_body[plot_body_wing][0][:,0])),'chamfer_dist': chamfer_body[plot_body_wing][0][:,0]}
            return pd.DataFrame(data_long)


    
    
    def generate_dflong_chamfer(self,xname,model,plot_body_wing = 'wing'):
        path = os.path.dirname(self.input_dir)
        with open(f'{path}/{model}/chamfer.pkl', 'rb') as f: 
                chamfer = pickle.load(f)
        chamfer_df = pd.DataFrame(chamfer[plot_body_wing][0])
        chamfer_3d = chamfer_df.fillna(np.nan).to_numpy()
            # Flatten data for seaborn
        data_long = {
            f'xtick': np.repeat(xname, chamfer_3d.shape[0]),
            'chamfer_dist': chamfer_3d.T.flatten()
        }
        return pd.DataFrame(data_long)

