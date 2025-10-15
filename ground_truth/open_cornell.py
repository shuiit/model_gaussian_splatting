


from EpiPolar import EpiPolar
import pickle
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import os
matplotlib.use('TkAgg')
plt.ion()

mov_frame = 'mov_132_frame_2195'#'mov_78_frame_3698'
mov = int(mov_frame.split('_')[1]) 
frame = int(mov_frame.split('_')[3]) 


frame = 1765

image_path = 'G:/My Drive/Research/gaussian_splatting/gaussian_splatting_input/mov1_2023_08_09_60ms/'
dict_path  = f'G:/My Drive/Research/gaussian_splatting/gaussian_splatting_input/{mov}_2023_08_09_60ms/dict/frames_model.pkl'
dict_path  = 'G:/My Drive/Research/gaussian_splatting/gaussian_splatting_input/evaluation/dict/frames_model_evaluation.pkl'


image_path = 'G:/My Drive/Research/gaussian_splatting/gaussian_splatting_input/mov9_cornell/'
dict_path = 'G:/My Drive/Research/gaussian_splatting/gaussian_splatting_input/mov9_cornell/dict/frames_model_cornell.pkl'


# frame = 439
width = 2
with open(dict_path,'rb') as f:
    frames_dict = pickle.load(f)


for wing in ['wing1','wing2']:
    
    # mov = frames_dict[frame][-1]['mov_name']
    # image_path = f'G:/My Drive/Research/gaussian_splatting/gaussian_splatting_input/mov{mov}_2023_08_09_60ms/'
    # save_dir = f'G:/My Drive/Research/gaussian_splatting/gaussian_splatting_input/evaluation/points/mov{mov}/'
    # save_file_name = f'{wing}_gt_points_frame{frame}.pkl'

    # os.makedirs(save_dir, exist_ok=True)

    epi = EpiPolar(image_path,frame,frames_dict)
    fig, axs = epi.plot_frame()
    if wing == 'wing2':
        file_to_plot = f'G:/My Drive/Research/gaussian_splatting/gaussian_splatting_input/evaluation/points/mov{mov}/wing1_gt_points_frame{frame}.pkl'

        with open(file_to_plot,'rb') as f:
            interest_points = pickle.load(f) 
        [axs[idx // width,idx % width].scatter(np.vstack(pixels)[:,0], np.vstack(pixels)[:,1], c='red',s = 3) for idx,pixels in enumerate(interest_points.values())]


    epi.get_n_colors(80)


    # fig.show()
    fig.canvas.mpl_connect('button_press_event', lambda event: epi.on_click(event, axs))
    plt.show(block = True)   
    
