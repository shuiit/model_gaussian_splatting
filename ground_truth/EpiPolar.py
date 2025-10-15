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


class EpiPolar:
    def __init__(self,image_path,frame,frames_dict, save_file_name = None ):
        self.frames = [Frame(image_path,frame,cam,frames_dict = frames_dict) for cam in range(4)] 


        intrinsic = [self.frames[cam].K for cam in range(4)]
        rotation_translation = [np.hstack((self.frames[cam].R.T,self.frames[cam].X0)) for cam in range(4)]

        self.intrinsic = self.get_product(intrinsic)
        self.all_rotation_translation = self.get_product(rotation_translation)
        self.cam_num = [(a, b) for a, b in it.product(range(4), repeat=2) if a != b]
        self.fundamental = [self.fundamental_matrix(P1[0], P1[1], K1[0], K1[1]) for P1, K1 in zip(self.all_rotation_translation, self.intrinsic)]
        self.dict_cameras = {f'cam{idx + 1}':[] for idx in range(4)}
        for idx,cam in enumerate(self.cam_num):
            self.dict_cameras[f'cam{cam[0] + 1}'].append(self.fundamental[idx])

        self.wing_pixels = {f'cam{idx + 1}':[] for idx in range(4)}
        self.wing_pixels_result = {}
        self.list_click = []
        self.plotted_artists = []
        self.save_file_name = save_file_name

    def plot_frame(self, width = 2, hight = 2):
        
        fig, axs = plt.subplots(width, hight)
        for idx in range(4):        
            axs[idx // width,idx % width].imshow(255-np.array(self.frames[idx].im),cmap = 'gray')
        

        return fig, axs



    def fundamental_matrix(self,P1, P2, K1, K2):
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



    def get_product(self,data):
        return  [(a, b) for a, b in it.product(data, repeat=2) if a is not b]
    

    def lineToBorderPoints(self,lines, imageSize):
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
    
    
    def get_n_colors(self,n):
        colors = [mcolors.to_rgba(c) for c in plt.rcParams['axes.prop_cycle'].by_key()['color']]
        self.colors = colors * math.ceil(n / len(colors))


    def remove_epi_lines(self,axs,width = 2):
    
        self.list_click.clear()
        
        for artist in self.plotted_artists:
            artist.remove()
        self.plotted_artists.clear()  # Clear the list after removing
        plt.draw()
        with open(self.save_file_name, 'wb') as f:
            pickle.dump(self.wing_pixels, f)
        [axs[idx // width,idx % width].scatter(np.vstack(pixels)[:,0], np.vstack(pixels)[:,1], c='red',s = 3) for idx,pixels in enumerate(self.wing_pixels.values()) if len(pixels) > 0]
        
    def calculate_epi_lines(self,fundamental,clicked_pixel,idx):

        fund = fundamental[idx]
        pix_h = np.append(clicked_pixel, 1)
        epiLines = pix_h @ fund.T
        return self.lineToBorderPoints(np.atleast_2d(epiLines), [160,160])



    def plot_epi(self,fundamental,clicked_pixel,cam_idx,axs,cam_to,color,idx,width = 2):
    

        pts = self.calculate_epi_lines(fundamental,clicked_pixel,idx)

        scatter = axs[cam_idx // width,cam_idx % width].scatter(clicked_pixel[0], clicked_pixel[1], c=color)
        lines = np.stack([pts[:, :2], pts[:, 2:]], axis=1)
        collection = collections.LineCollection(lines, colors=color)
        axs[cam_to // width,cam_to % width].add_collection(collection)

        # Save the artists for later removal
        self.plotted_artists.append(scatter)
        self.plotted_artists.append(collection)
        

    def on_click(self,event,axs, width = 2):

        if event.inaxes is not None:
            cam_idx = np.argwhere(axs == event.inaxes)[0]
            cam_idx = cam_idx[0] * axs.shape[1] + cam_idx[1]
        else:
            return

        clicked_pixel = np.array([event.xdata, event.ydata])
        self.list_click.append(clicked_pixel)
        

        if event.button == 1:  # Left click: ADD POINT

            self.wing_pixels[f'cam{cam_idx +1}'].append(clicked_pixel)
            fundamental = self.dict_cameras[f'cam{cam_idx +1}'] 
            rand_idx = random.randint(0, len(self.colors))
            color = self.colors[rand_idx]
            cam_to = [idx for idx in range(4) if idx != cam_idx]  
            [self.plot_epi(fundamental,clicked_pixel,cam_idx,axs,cam_to,color,idx,width = 2) for idx, cam_to in enumerate(cam_to)]
            plt.pause(0.01)
            if len(self.list_click) == 4:
                self.remove_epi_lines(axs)

        if event.button == 3:  # Right click: REMOVE LAST POINT
            for wing_pixels_key in self.wing_pixels.keys(): 
                if len(self.wing_pixels[wing_pixels_key]) > 0:
                    array = np.vstack(self.wing_pixels[wing_pixels_key])
                    idx_to_remove = [np.where((clicked == array).all(axis = 1))[0]  for clicked in self.list_click if len(np.where((clicked == array).all(axis = 1))[0])>0 ]
                    if len(idx_to_remove) > 0:
                        self.wing_pixels[wing_pixels_key] = [pt for i, pt in enumerate(self.wing_pixels[wing_pixels_key]) if i not in idx_to_remove]
            self.remove_epi_lines(axs)
                
                    
                    
