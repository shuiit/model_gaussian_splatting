


from EpiPolar import EpiPolar
import pickle

image_path = 'I:/My Drive/Research/gaussian_splatting/gaussian_splatting_input/mov1_2023_08_09_60ms/'
dict_path  = 'I:/My Drive/Research/gaussian_splatting/gaussian_splatting_input/mov1_2023_08_09_60ms/dict/frames_model.pkl'
frame = 370

with open(dict_path,'rb') as f:
    frames_dict = pickle.load(f)

epi = EpiPolar(image_path,frame,frames_dict)
epi.get_n_colors(80)
fig, axs = epi.plot_frame()
fig.show()
fig.canvas.mpl_connect('button_press_event', lambda event: epi.on_click(event, axs))
fig.show()