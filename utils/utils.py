import os
import numpy as np
import cv2
import shutil
from pathlib import Path

def list_file_r(path:os.PathLike, extension:list=None) -> list:
    """
    Args

        path: PathLike object

    Return

        list of files under path, excluding directories
    """
    listed_files = []
    for root, dirs, files in os.walk(path):
        for name in files:
            # if file formats are provided, only return files with the given formats
            if extension and os.path.splitext(name)[-1] in extension:
                pathfile = os.path.join(root, name)
                listed_files.append(os.path.normpath(pathfile))
            elif not extension:
                pathfile = os.path.join(root, name)
                listed_files.append(os.path.normpath(pathfile))
    return listed_files

def mask_to_contour(mask_path:os.PathLike):
    msk = cv2.imread(mask_path)
    msk = cv2.cvtColor(msk, cv2.COLOR_BGR2GRAY)
    _, im = cv2.threshold(msk, 100, 255, type=cv2.THRESH_BINARY)
    contours, hierachy = cv2.findContours(im, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours

def contour_to_YOLO_annotation(contours, width, height, destination):
    annotation_text = ''
    for contour in contours[:-1]:
        annotation_text += '0'
        for d in range(len(contour)):
            annotation_text += ' '
            X, Y = contour[d][0]
            X, Y = X/width, Y/height
            annotation_text += '{} {}'.format(X, Y)
        annotation_text += '\n' # newline
    #with open()
    return

def copy_image_to_sub_dir(image_pathfiles:list, sub_dir:str, copy_bbox=False):
    image_data_path = os.path.split(image_pathfiles[0])[0]
    last_images_folder_index = image_data_path.rfind('images')
    annotation_path = image_data_path[:last_images_folder_index] + 'labels' +  image_data_path[last_images_folder_index+6:]
    if copy_bbox:
        bbox_path = image_data_path[:last_images_folder_index] + 'bbox' +  image_data_path[last_images_folder_index+6:]
    for img_pf in image_pathfiles:
        #print(img_pf)
        annotation_pf = os.path.splitext(os.path.split(img_pf)[-1])[0] + '.txt'
        #print(annotation_pf)
        #annotation_pf = 'MASK' + annotation_pf[3:]
        annotation_pf = os.path.normpath(os.path.join(annotation_path, annotation_pf))
        annotation_filename = os.path.split(annotation_pf)[-1]
        annotation_dst_pf = os.path.normpath(os.path.join(annotation_path, sub_dir, annotation_filename))

        img_filename = os.path.split(img_pf)[-1]
        img_dst_pf = os.path.normpath(os.path.join(image_data_path, sub_dir, img_filename))
        #print(img_filename)
        #print(img_dst_pf)
        img_dir, ann_dir=os.path.join(image_data_path, sub_dir), os.path.join(annotation_path, sub_dir)
        
        if not os.path.exists(img_dir):
            os.makedirs(img_dir)
        if not os.path.exists(ann_dir):
            os.makedirs(ann_dir)

        if copy_bbox:
            bbox_pf = os.path.normpath(os.path.join(bbox_path, os.path.splitext(os.path.split(img_pf)[-1])[0] + '.txt'))
            bbox_filename = os.path.split(bbox_pf)[-1]
            bbox_dst_pf = os.path.normpath(os.path.join(bbox_path, sub_dir, bbox_filename))
            bbox_dir = os.path.join(bbox_path, sub_dir)
            if not os.path.exists(bbox_dir):
                os.makedirs(bbox_dir)
            shutil.copy2(bbox_pf, bbox_dst_pf)

        shutil.copy2(img_pf, img_dst_pf)
        shutil.copy2(annotation_pf, annotation_dst_pf)
    return

def create_yolo_annotation_from_mask(mask:os.PathLike, out:os.PathLike, add_border_on_edge = False):
    if os.path.isfile(mask):
        pass
    elif os.path.isdir(mask):
        mask_data_path = mask
        annotation_dst = out
        pathparts=list(Path(os.path.normpath(annotation_dst)).parts[:-1])
        bbox_dst = os.sep.join(pathparts+['bbox'])
        mask_pfs = list_file_r(mask_data_path, extension=['.tif'])
        # loop over provided masks
        for pf in mask_pfs:
            msk = cv2.imread(pf)
            height, width, _ = msk.shape
            msk = cv2.cvtColor(msk, cv2.COLOR_BGR2GRAY)
            msk = cv2.bitwise_not(msk)  # invert mask
            _, im = cv2.threshold(msk, 100, 255, type=cv2.THRESH_BINARY)
            if add_border_on_edge:
                black_border = [0 for _ in range(1000)]
                im[0], im[-1] = black_border, black_border
            contours, hierachy = cv2.findContours(im, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # preprocess contours and bounding boxes
            annotation_text = ''
            #bbox_text = ''
            for contour in contours:
                # b-box
                #bbox_text += '0 '
                x,y,w,h = cv2.boundingRect(contour)
                x,y = x+w/2, y+h/2
                x,y,w,h = x/width, y/height, w/width, h/height
                #bbox_text += '{} {} {} {}\n' .format(x,y,w,h)
                # mask
                annotation_text += '0'
                for d in range(len(contour)):
                    annotation_text += ' '
                    #contour[d][0][1] -= border_height
                    X, Y = contour[d][0]
                    #Y = Y - border_height
                    X, Y = X/width, Y/height
                    annotation_text += '{} {}'.format(X, Y)
                annotation_text += '\n' # newline

            file_name = os.path.splitext(os.path.split(pf)[-1])[0]
            annotation_fp = os.path.normpath(os.path.join(annotation_dst, file_name)+'.txt')
            bbox_fp = os.path.normpath(os.path.join(bbox_dst, file_name)+'.txt')
            if not os.path.exists(annotation_dst):
                os.makedirs(annotation_dst)
            with open(annotation_fp, 'w') as mask_annotation_file:
                mask_annotation_file.write(annotation_text)
                mask_annotation_file.close()

            if not os.path.exists(bbox_dst):
                os.makedirs(bbox_dst)
            """ with open(bbox_fp, 'w') as mask_bbox_file:
                mask_bbox_file.write(bbox_text)
                mask_bbox_file.close() """

def save_mask_from_yolo_results(results):
    for test_result in results:
        if not test_result.masks:
            continue

        masks = test_result.masks.xy
        h, w = test_result.orig_shape
        blank_mask = np.zeros([h, w, 3],dtype=np.uint8)
        blank_mask.fill(255)
        # path
        img_path = test_result.path
        img_name = os.path.split(img_path)[1]
        folder = os.path.split(os.path.splitext(img_path)[0])[0]
        dst_folter = os.path.join(folder, 'mask')
        if not os.path.exists(dst_folter):
            os.makedirs(dst_folter)
        dst_fp = os.path.join(dst_folter, img_name)
        for mask in masks:
            cv2.fillPoly(blank_mask, pts=[mask.astype(np.int32)], color=(0, 0, 0))
        
        # save mask
        cv2.imwrite(dst_fp, blank_mask)

#def yolo_to_
