import os

import cv2
import shutil
import numpy as np
import albumentations as A

from utils.utils import list_file_r
from utils.fft import inverseFDA
from PIL import Image


def batch_augment_with_bbox(images_path:os.PathLike, bbox_path:os.PathLike, augmentor:A.Compose=None, target_folder:os.PathLike='augmented', random_choice:float=0.2) -> None:
    """

    """
    # get img path files
    img_extension = '.tif'
    img_pfs = [f for f in os.listdir(images_path) if os.path.splitext(f)[-1] == img_extension]
    if random_choice:
        rng = np.random.default_rng(seed=12345)
        sample_size = int(len(img_pfs)*random_choice)
        img_pfs = rng.choice(img_pfs, size=sample_size, replace=False)
    
    for img_pf in img_pfs:
        filename = os.path.splitext(os.path.split(img_pf)[-1])[0]
        img_filename = filename + '.tif'
        img_dst_pf = os.path.normpath(os.path.join(images_path, target_folder, img_filename))
        # get corresponding bbox path files
        bbox_filename = filename + '.txt'
        bbox_pf = os.path.normpath(os.path.join(bbox_path, bbox_filename))
        bbox_dst_pf = os.path.normpath(os.path.join(bbox_path, target_folder, bbox_filename))
        # put class label to the end
        bboxes = np.loadtxt(bbox_pf, delimiter=' ')
        bboxes = np.concatenate((bboxes, bboxes[:,0:1]), axis=-1)[:,1:]
        # get img
        img_pf = os.path.normpath(os.path.join(images_path, img_filename))
        img = cv2.imread(img_pf)
        # augment
        augmented = augmentor(image=img,bboxes=bboxes)
        # swap back class label
        bboxes = np.array(augmented['bboxes'])
        bboxes = np.concatenate((bboxes[:,-1:], bboxes), axis=-1)[:,:-1]
        # save augmented image and bounding boxes
        img = Image.fromarray(augmented['image'])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
        if not os.path.exists(os.path.split(img_dst_pf)[0]):
            os.makedirs(os.path.split(img_dst_pf)[0])
        if not os.path.exists(os.path.split(bbox_dst_pf)[0]):
            os.makedirs(os.path.split(bbox_dst_pf)[0])
        img.save(img_dst_pf)
        num_of_points=4
        np.savetxt(bbox_dst_pf, bboxes, fmt=' '.join(['%i']+['%1.4f']*num_of_points))

    return


def batch_augment_with_mask(images_path:os.PathLike, mask_path:os.PathLike, transforms:list=[], target_folder:os.PathLike='augmented', random_choice:float=0.2, FDA_targets=None) -> None:
    """

    """
    # get img path files
    img_extensions = ['.tif', '.bmp']
    img_pfs = [f for f in os.listdir(images_path) if os.path.splitext(f)[-1] in img_extensions]
    if random_choice:
        rng = np.random.default_rng(seed=12345)
        sample_size = int(len(img_pfs)*random_choice)
        if type(random_choice)==int:
            sample_size = random_choice
        img_pfs = rng.choice(img_pfs, size=sample_size)
    # if using FDA
    if FDA_targets:
        # target is a single image
        if os.path.isfile(FDA_targets):
            FDA_targets = [FDA_targets]
        # target is a folder
        elif os.path.isdir(FDA_targets):
            #FDA_targets = next(os.walk(FDA_targets), (None, None, []))[2]  # [] if no file
            FDA_targets = list_file_r(FDA_targets, extension=['.tif'])
        else:
            Exception()
        # add FDA to augmentor
        transforms += [A.FDA(reference_images=FDA_targets, read_fn=cv2.imread, beta_limit=0.05, p=1)]
    
    for img_pf in img_pfs:
        filename = os.path.splitext(os.path.split(img_pf)[-1])[0]
        img_filename = filename + os.path.splitext(img_pf)[-1]
        img_dst_pf = os.path.normpath(os.path.join(images_path, target_folder, img_filename))
        # get corresponding mask path files
        
        #mask_filename = filename + '.tif'
        mask_pf = os.path.normpath(os.path.join(mask_path, img_filename))
        mask_dst_pf = os.path.normpath(os.path.join(mask_path, target_folder, img_filename))
        # get img
        img_pf = os.path.normpath(os.path.join(images_path, img_filename))
        img = cv2.imread(img_pf)
        # mask img
        mask_pf = os.path.normpath(os.path.join(mask_path, img_filename))
        mask = cv2.imread(mask_pf)

        # augment
        augmentor = A.Sequential(transforms=transforms,
                                )
        augmented = augmentor(image=img,mask=mask)
        # save augmented image and masks
        img = Image.fromarray(augmented['image']).convert('L')
        #img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
        mask = Image.fromarray(augmented['mask']).convert('L')
        #mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY) 
        if not os.path.exists(os.path.split(img_dst_pf)[0]):
            os.makedirs(os.path.split(img_dst_pf)[0])
        if not os.path.exists(os.path.split(mask_dst_pf)[0]):
            os.makedirs(os.path.split(mask_dst_pf)[0])
        
        temp = 0
        while os.path.isfile(img_dst_pf):
            temp+=1
            img_dst_pf = os.path.normpath(os.path.join(images_path, target_folder, filename+str(temp)+'.tif'))
            mask_dst_pf = os.path.normpath(os.path.join(mask_path, target_folder, filename+str(temp)+'.tif'))
        img.save(img_dst_pf)
        mask.save(mask_dst_pf)

    return


def batch_invers_FDA(src:os.PathLike, FDA_targets:os.PathLike, beta:float=0.4, p=0.2, in_place=True):
    img_extensions = ['.tif']
    img_pfs = [f for f in os.listdir(src) if os.path.splitext(f)[-1] in img_extensions]
    targets = list_file_r(FDA_targets, extension=['.tif'])
    rng = np.random.default_rng()
    for img_path in img_pfs:
        # only apply to come percent of the images
        if np.random.rand() >= p:
            continue
        tgt_path = rng.choice(targets)
        source_img, target_img = Image.open(os.path.join(src, img_path)), Image.open(tgt_path)
        target_img = target_img.resize(source_img.size)
        source_img, target_img = np.array(source_img), np.array(target_img)
        random_beta = rng.random()*beta
        result = Image.fromarray(inverseFDA(source_img, target_img, beta=random_beta))
        if in_place:
            result.save(os.path.join(src, img_path))
        
    return







