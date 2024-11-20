import time

import ultralytics

from PIL import Image, ImageDraw
import numpy as np
import os

def linear_fit(X, y):
    XT = X.transpose()
    beta = np.linalg.inv(XT @ X) @ (XT @ y)
    
    beta=beta.reshape(-1)
    intercept, slope = beta[0], beta[1]
    return intercept, slope

def projection_error2d(x, y, intercept, slope):
    # get projection distances
    a = np.array([[1], [slope]])
    B = np.concatenate((x, y), axis=-1).transpose()
    B[-1] -= intercept
    P = a@(a.transpose()@B)/(a.transpose()@a)
    B[-1], P[-1] = B[-1] + intercept, P[-1] + intercept
    distances = ((B-P)**2).sum(axis=0)**.5
    return distances, P


def calculate_widths(inference_results, img_path=None):
    widths = []
    if img_path:
        img = Image.open(img_path) 
        draw = ImageDraw.Draw(img) 
        w, h = img.size
    for inference_result in inference_results:
        overall_width = 0
        assert type(inference_result) == ultralytics.engine.results.Results, "Inference result should be of type 'ultralytics.engine.results.Results'!"
        if not inference_result.masks:
            #widths.append(0)
            continue
        for coos in inference_result.masks.xy:
            start = time.time()
            # get LMS estimation
            x, y = coos[:,0:1], coos[:,1:]
            X = np.concatenate((np.ones_like(x), x), axis=-1)
            intercept, slope = linear_fit(X,y)
            distances, P = projection_error2d(x, y, intercept, slope)  # get projection distances
            intercept_r, slope_r = linear_fit(np.concatenate((np.ones_like(y), y), axis=-1),x)
            distances_r, P_r = projection_error2d(y, x, intercept_r, slope_r)  # get projection distances
            if distances_r.sum() <= distances.sum():  # fit x from y if line is better explained with vertical direction fitting
                distances, P = distances_r, P_r
                x, y = y, x
            selection = np.logical_and(np.less_equal(np.percentile(distances, 25), distances), np.greater_equal( np.percentile(distances, 100), distances))
            #distances = distances[selection]
            # robust fit
            x, y = x[selection], y[selection]
            X = np.concatenate((np.ones_like(x), x), axis=-1)
            intercept, slope = linear_fit(X,y)
            distances, _ = projection_error2d(x, y, intercept, slope)  # get projection distances
            selection = np.logical_and(np.less_equal(np.percentile(distances, 25), distances), np.greater_equal( np.percentile(distances, 100), distances))
            distances = distances[selection]
            estimated_widths = distances.mean()*2
            #print('Elapsed time: {:.5e}'.format(time.time()-start))
            overall_width += estimated_widths
            #print(estimated_widths)
        
        overall_width /= len(inference_result.masks.xy)
        overall_width = overall_width / sum(inference_result.orig_shape) * 2
        widths.append(overall_width)
    #print(widths)
    return np.stack(widths)[...,None]
    #img.show()

def draw_kernels(inference_results, img_path=None):
    if img_path:
        img = Image.open(img_path) 
        draw = ImageDraw.Draw(img) 
        w, h = img.size
    for inference_result in inference_results:
        if not inference_result.masks:
            continue
        for coos in inference_result.masks.xy:
            start = time.time()
            # get LMS estimation
            x, y = coos[:,0:1], coos[:,1:]
            X = np.concatenate((np.ones_like(x), x), axis=-1)
            X = np.concatenate((np.ones_like(x), x), axis=-1)
            intercept, slope = linear_fit(X,y)
            distances, P = projection_error2d(x, y, intercept, slope)  # get projection distances
            intercept_r, slope_r = linear_fit(np.concatenate((np.ones_like(y), y), axis=-1),x)
            distances_r, P_r = projection_error2d(y, x, intercept_r, slope_r)  # get projection distances
            if distances_r.sum() <= distances.sum():  # fit x from y if line is better explained with vertical direction fitting
                intercept, slope, P = intercept_r, slope_r, P_r[::-1]
            # draw polygons
            B = np.concatenate((x, y), axis=-1).transpose()
            draw.line(B.T.flatten())
            # draw projections
            
            # draw kernel
            draw_points = [(0, int(intercept)), (w, int(intercept+slope*w))]
            
            # draw examples of projection
            choice = np.random.choice(range(coos.shape[0]))
            random_coo, projection = B.transpose()[choice], P.transpose()[choice]
            draw_points_projection = [tuple(random_coo), tuple(projection)]
            if distances_r.sum() <= distances.sum():
                draw_points = [(int(intercept), 0), (int(intercept+slope*h), h)]
                draw_points_projection = draw_points_projection[::-1]
            draw.line(draw_points, fill='green', width=10)
            draw.line(draw_points_projection, fill='blue', width=5)
            
    return img

def get_image_widths_by_path_names(inference_results:list):
    """
    return the relative width of the generated lines in the image by looking up the file name in the YOLO inference results
    """
    original_size = 1000
    widths=[]
    for inference_result in inference_results:
        path = inference_result.path
        #img_shape = Image.open(path).size
        filename =os.path.splitext(os.path.split(path)[-1])[0]
        width_pos = filename.find('Width') + 5
        width_digits = filename[width_pos:].find('_')
        width = int(filename[width_pos:width_pos+width_digits]) / original_size
        widths.append(width)
    
    return np.stack(widths)[...,None]
        

from skimage.draw import line
from skimage import color
from scipy.signal import medfilt2d, convolve2d
import time
import sys
def get_width_by_Hough_lines(nms_lines:list, black_white_mask:np.ndarray, rotation_diff_threshold=5):
    """
    _arguments:

    black_white_mask: a 2d array taking values of 0 or 255, with 0 being background substrate and 255 being the printed pattern
    """
    #
    rotation_diff_threshold *= (np.pi/180)
    height, width = black_white_mask.shape[:2]
    #
    mask_x = np.array([[-1,0,1]])
    mask_y = mask_x.T
    dx, dy = convolve2d(black_white_mask, mask_x, mode='same'),  convolve2d(black_white_mask, mask_y, mode='same')
    # set window size for ambient gradient computation
    neiborhood_size = 13
    neighborhood_mask = np.ones((neiborhood_size, neiborhood_size))
    dx, dy = convolve2d(dx, neighborhood_mask, mode='same')[...,None],  convolve2d(dy, neighborhood_mask, mode='same')[...,None]
    dx, dy = dx/neiborhood_size**2, dy/neiborhood_size**2
    nabla_I = np.concatenate((dx, dy), axis=-1)

    starting_rho, starting_theta, widths=None, None, []
    for rho,theta in nms_lines:
        # get coordinates of pixels over the line
        a, b = np.cos(theta), np.sin(theta)
        #print(a,b)
        x0, y0 = a*rho,  b*rho
        x1, y1 = int(x0 + width*(-b)), int(y0 + height*(a)) 
        x2, y2 = int(x0 - width*(-b)), int(y0 - height*(a))

        rr, cc = line(x1,y1,x2,y2)
        #print(x1,y1,x2,y2)

        coords = np.array(list(zip(cc,rr)))
        # select coordinates within image
        coords = coords[(coords >= 0).all(axis=-1)]
        coords = coords[coords[:,0] < height]
        coords = coords[coords[:,1] < width]

        print(nabla_I.shape, coords.shape)
        # compute dominant orientation of gradient
        grad_x, grad_y = nabla_I[tuple(coords.T)].mean(axis=0)
        d = (grad_x**2 + grad_y**2)**.5 + 1e-7
        rotation = np.arcsin(grad_y/d)
        theta = theta + np.pi if rho<0 else theta
        print("Theta: {}; line gradient angle: {}".format(theta, rotation))
        difference_1, difference_2 = abs((rotation + theta)%np.pi), abs((-rotation + theta)%np.pi)
        difference_1, difference_2 = min(difference_1, np.pi-difference_1),  min(difference_2, np.pi-difference_2)
        #print(np.arctan(grad_y/grad_x))
        # if the gradient direction is roughly the same as the line (polar representation), the line is the starting edge(left to right) of the pattern
        if difference_1 < difference_2 and difference_1 <= rotation_diff_threshold:
            print('Start')
            starting_rho = rho
        elif difference_2 <= rotation_diff_threshold and starting_rho:
            print('End')
            widths.append(rho-starting_rho)
            starting_rho=None
    # classify image as invalid if too much differences in width
    if np.array(widths).std() > 30:
        return []
    return widths


import cv2
from skimage import color
from scipy.signal import medfilt2d, convolve2d
import time

def get_nms_Hough_lines(img:np.ndarray):
    height, width = img.shape[:2]
    gray = np.array(color.rgb2gray(img)*255, dtype=np.uint8)
    # Compute a mask
    gray = medfilt2d(gray, 11)
    # 
    # Otsu's thresholding after Gaussian filtering
    blur = cv2.GaussianBlur(gray,(5,5),0)
    _, otsu_th = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    # adaptive threshold
    thresh = cv2.adaptiveThreshold(gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 5)
    #thresh = medfilt2d(thresh, 5)
    # visualize
    #cv2.imshow("medfilter", gray)
    #cv2.imshow("Mean Adaptive Thresholding", thresh)

    lines = cv2.HoughLines(thresh,1,np.pi/180,200)
    if lines is None:
        return None, thresh
    nms_lines, threshold_rho, threshold_theta = [lines[0][0]], 40, np.pi/180*3
    rho,theta = lines[0][0]
    # compute endpoints of the line
    a, b = np.cos(theta), np.sin(theta)
    x0, y0 = a*rho,  b*rho
    x1, y1 = int(x0 + width*(-b)), int(y0 + height*(a)) 
    x2, y2 = int(x0 - width*(-b)), int(y0 - height*(a))

    cv2.line(img,(x1,y1),(x2,y2),(0,0,255),2)
    for line in lines:
        # select lines of highest confidence with different rho (theta are the same)
        rho,theta = line[0]
        # compare with every lines in nms lines
        new_strong_line = True
        for prev_line in nms_lines:
            rho_0, theta_0 = prev_line
            # skip this line if it is within threshold range with any previous strong lines
            if abs(rho_0 - rho) < threshold_rho and abs(theta_0 - theta) < threshold_theta:
                new_strong_line = False
                break
        if new_strong_line:
            nms_lines.append([rho,theta])
            # compute endpoints of the line
            a, b = np.cos(theta), np.sin(theta)
            x0, y0 = a*rho,  b*rho
            x1, y1 = int(x0 + width*(-b)), int(y0 + height*(a)) 
            x2, y2 = int(x0 - width*(-b)), int(y0 - height*(a))
            cv2.line(img,(x1,y1),(x2,y2),(0,0,255),2)
    
    nms_lines = np.array(nms_lines)
    # sort the nms lines by rho
    index = np.argsort(nms_lines[:,0])
    nms_lines = nms_lines[index]
    Image.fromarray(img).show()
    return nms_lines, otsu_th