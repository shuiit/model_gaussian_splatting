
import pickle
import os
from Evaluation import Evaluation
from tqdm import tqdm
import Utils

class EvaluateAngleSweep():
    def __init__(self,frames,nominal_initial_angles,input_dir, ini_path,input_path_for_image):



        

        dirs = os.listdir(f'{input_dir}/{list(nominal_initial_angles.keys())[0].split("_")[3]}') 
        self.sweep_size = len([dir for dir in dirs if 'fly' in dir.split('_')])
        dir_names = [name for name in os.listdir(ini_path) if os.path.isfile(os.path.join(ini_path, name))]
        self.ini_angles = {idx:self.open_file(f'{ini_path}/{dir}') for idx,dir in enumerate(dir_names)}
        self.input_path_for_image = input_path_for_image
        self.frames = frames
        self.nominal_initial_angles = nominal_initial_angles

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
    

    def load_all_sweep(self):
        # generate frames file if it doesnt exist
        self.sweep = {}
        for mov_name in tqdm(list(self.nominal_initial_angles.keys())[1:]):
            try:
                self.sweep[mov_name] = [self.load_frame_all_sweep(idx_iter,mov_name,iteration,letedict,self.frames) for idx_iter in range(self.sweep_size)]   
            except:
                continue


    def calculate_wing_chamfer(self):
        for mov_name in tqdm(list(self.sweep.keys())):
                [frame.calculate_chamfler() for frame in self.sweep[mov_name]]  
                [frame.calculate_chamfler_body() for frame in self.sweep[mov_name]]


    def hull_calc_zbuff_hull_xbody(self,file_path_save_hull):

        # generate hull file if it doesnt exist (body hull for ground truth)
        if os.path.isfile(file_path_save_hull):
            with open(file_path_save_hull, "rb") as input_file:
                hull_movs = pickle.load(input_file)
        else:
            Utils.make_body_hull_file(self.nominal_initial_angles,file_path_save_hull)

        self.zbuff_hull = {movname: Utils.load_body_hull_calc_xbody(self.sweep[movname][0], hull_movs[movname]) for movname in tqdm(list(self.sweep.keys())[1:])}
        self.xbody_xbody_hull_dot_vec = [[frame.load_hull_calc_xbody_dot_per_idx(self.zbuff_hull[mov_name]) for frame in self.sweep[mov_name]] for mov_name in tqdm(list(self.sweep.keys())[1:])]


    def load_sweep(self,sweep_path,file_path_save_hull):
      if os.path.isfile(sweep_path):
        with open(sweep_path, "rb") as input_file:
            self.sweep = pickle.load(input_file)
      else:
          self.load_all_sweep()
          self.calculate_wing_chamfer()
          self.hull_calc_zbuff_hull_xbody(file_path_save_hull)     



