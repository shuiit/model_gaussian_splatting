import numpy as np
import pickle
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import collections
from matplotlib import colors as mcolors
import itertools as it
import os
import sys
parent_dir = os.path.abspath(os.path.join(os.getcwd()))
sys.path.insert(0, parent_dir)
from Frame import Frame
import random



import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import collections
from matplotlib import colors as mcolors

# %matplotlib qt  # You can uncomment if you want external window

def get_n_colors(n):
    colors = [mcolors.to_rgba(c)
                  for c in plt.rcParams['axes.prop_cycle'].by_key()['color']]
    colors = colors * math.ceil(n / len(colors))
    return colors[:n]

def fundamental_matrix(P1, P2, K1, K2):
    R1 = P1[:, 0:3]
    R2 = P2[:, 0:3]
    t1 = P1[:, 3]
    t2 = P2[:, 3]

    R = R2.T @ R1
    t = R1.T @ (t2 - t1)

    tx = np.array([[0, -t[2], t[1]],
                   [t[2], 0, -t[0]],
                   [-t[1], t[0], 0]])
    Y = np.linalg.solve(K2.T, R) @ tx
    F = np.linalg.solve(K1.T, Y.T).T
    return F

def lineToBorderPoints(lines, imageSize):
    nPts = lines.shape[0]
    pts = -np.ones((nPts, 4))
    firstRow = 0.5
    firstCol = 0.5
    lastRow = firstRow + imageSize[0]
    lastCol = firstCol + imageSize[1]
    eps = np.finfo(np.float32).eps

    for iLine in range(nPts):
        a = lines[iLine, 1]
        b = lines[iLine, 0]
        c = lines[iLine, 2]
        endPoints = np.zeros((4))
        iPoint = 0

        if abs(a) > eps:
            row = -(b * firstCol + c) / a
            if firstRow <= row <= lastRow:
                endPoints[iPoint:iPoint+2] = [row, firstCol]
                iPoint += 2
            row = -(b * lastCol + c) / a
            if firstRow <= row <= lastRow:
                endPoints[iPoint:iPoint+2] = [row, lastCol]
                iPoint += 2
        if abs(b) > eps:
            if iPoint < 3:
                col = -(a * firstRow + c) / b
                if firstCol <= col <= lastCol:
                    endPoints[iPoint:iPoint+2] = [firstRow, col]
                    iPoint += 2
            if iPoint < 3:
                col = -(a * lastRow + c) / b
                if firstCol <= col <= lastCol:
                    endPoints[iPoint:iPoint+2] = [lastRow, col]
                    iPoint += 2

        for i in range(iPoint, 4):
            endPoints[i] = -1
        pts[iLine, :] = endPoints[[1, 0, 3, 2]]

    return pts



image_path = 'I:/My Drive/Research/gaussian_splatting/gaussian_splatting_input/mov1_2023_08_09_60ms/'
dict_path  = 'I:/My Drive/Research/gaussian_splatting/gaussian_splatting_input/mov1_2023_08_09_60ms/dict/frames_model.pkl'
dict_path  = 'G:/My Drive/Research/gaussian_splatting/gaussian_splatting_input/evaluation/frames_model_evaluation.pkl'

frame = 804



with open(dict_path,'rb') as f:
    frames = pickle.load(f)

mov = frames[frame][-1]['mov_name']
image_path = f'G:/My Drive/Research/gaussian_splatting/gaussian_splatting_input/mov{mov}_2023_08_09_60ms/'
frames = [Frame(image_path,frame,cam,frames_dict = frames) for cam in range(4)] 

intrinsic = [frames[cam].K for cam in range(4)]
rotation_translation = [np.hstack((frames[cam].R.T,frames[cam].X0)) for cam in range(4)]

all_intrinsic = [(a,b) for a,b in it.combinations(intrinsic, 2)]
all_rotation_translation = [(a,b) for a,b in it.combinations(rotation_translation, 2)]
cam_num = [(a,b) for a,b in it.combinations(list(range(4)), 2)]


all_intrinsic = [(a, b) for a, b in it.product(intrinsic, repeat=2) if a is not b]
all_rotation_translation = [(a, b) for a, b in it.product(rotation_translation, repeat=2) if a is not b]
cam_num = [(a, b) for a, b in it.product(range(4), repeat=2) if a != b]

F = [fundamental_matrix(P1[0], P1[1], K1[0], K1[1]) for P1, K1 in zip(all_rotation_translation, all_intrinsic)]

dict_cameras = {f'cam{idx + 1}':[] for idx in range(4)}
for idx,cam in enumerate(cam_num):
    cam_from = cam[0]
    cam_to = cam[1]
    dict_cameras[f'cam{cam_from + 1}'].append(F[idx])

fig, axs = plt.subplots(2, 2)
im_size = [160,160]
from IPython.display import display
width = 2
for idx in range(4):        
    axs[idx // width,idx % width].imshow(255-np.array(frames[idx].im),cmap = 'gray')

colors = get_n_colors(80)
plotted_artists = []  # list to store all scatter and line collections
wing_pixels = {f'cam{idx + 1}':[] for idx in range(4)}
list_click = []
wing_pixels_result = {}
def on_click(event,dict_cameras, axs, im_size,wing_pixels,list_click):

    # Check if the click happened inside the plot
    # if event.xdata is None or event.ydata is None:
    #     return  # Don't do anything if the click was outside the image

    clicked_pixel = np.array([event.xdata, event.ydata])
    
    list_click.append(clicked_pixel)
    
    clicked_axis = event.inaxes
    if clicked_axis == axs[0][0]:
        cam_idx = 0
    elif clicked_axis == axs[0][1]:
        cam_idx = 1
    elif clicked_axis == axs[1][0]:
        cam_idx = 2
    elif clicked_axis == axs[1][1]:
        cam_idx = 3
    if event.button == 1:  # Left click: ADD POINT
        wing_pixels[f'cam{cam_idx +1}'].append(clicked_pixel)
        fundamental = dict_cameras[f'cam{cam_idx +1}'] 
        rand_idx = random.randint(0, len(colors))
        color = colors[rand_idx]
        
        cam_to = [idx for idx in range(4) if idx != cam_idx]  
        for idx, cam_to in enumerate(cam_to):

            fund = fundamental[idx]
            pix_h = np.append(clicked_pixel, 1)
            epiLines = pix_h @ fund.T
            pts = lineToBorderPoints(np.atleast_2d(epiLines), im_size)

            fig.suptitle('Epipolar lines')

            scatter = axs[cam_idx // width,cam_idx % width].scatter(clicked_pixel[0], clicked_pixel[1], c=color)
            lines = np.stack([pts[:, :2], pts[:, 2:]], axis=1)
            collection = collections.LineCollection(lines, colors=color)
            axs[cam_to // width,cam_to % width].add_collection(collection)

            # Save the artists for later removal
            plotted_artists.append(scatter)
            plotted_artists.append(collection)

            
            plt.pause(0.01)

            if len(list_click) == 4:
                # Remove only the scatter and lines
                
                list_click.clear()
                wing_pixels_result.update(wing_pixels)
                print(wing_pixels)
                for artist in plotted_artists:
                    artist.remove()
                plotted_artists.clear()  # Clear the list after removing
                plt.draw()
                with open(file_name, 'wb') as f:
                    pickle.dump(wing_pixels_result, f)
                    print(f"Saved to {file_name}")
                
                [axs[idx // width,idx % width].scatter(np.vstack(pixels)[:,0], np.vstack(pixels)[:,1], c='red',s = 3) for idx,pixels in enumerate(wing_pixels.values())]
            
           
fig.canvas.mpl_connect('button_press_event', lambda event: on_click(event, dict_cameras, axs, im_size,wing_pixels,list_click))
file_name = 'G:/My Drive/Research/gaussian_splatting/gaussian_splatting_input/evaluation/gt_points.pkl'

# Show the plot
plt.show()