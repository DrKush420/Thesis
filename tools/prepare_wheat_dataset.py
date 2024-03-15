import numpy as np
import argparse
import glob
import cv2
import os
import shutil


def roi_to_rect(roi):
    """Return the rectangular shape (x, y, w, h) that fits tightly around the trapezoid ROI
    roi:    ROI coords, start top left and go counter clockwise: [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]
    """
    x1 = min(roi[0][0], roi[1][0])
    y1 = min(roi[0][1], roi[3][1])
    x2 = max(roi[2][0], roi[3][0])
    y2 = max(roi[2][1], roi[2][1])
    return x1, y1, (x2 - x1), (y2 - y1)


def rect_to_roi(rect):
    """x, y, w, h to ((x1, y1), (x2, y2), (x3, y3), (x4, y4))
    """
    x, y, w, h = rect
    roi = (
        (x, y),
        (x, y+h),
        (x+w, y+h),
        (x+w, y)
    )
    return roi


# matrix determined by `read_trapezoid2.py`
matrix = np.array([
    [ 1.51130629e+00,  1.92260258e+00, -4.89180994e+02],
    [ 1.33971426e-16,  5.14071420e+00, -9.87483229e+02],
    [ 1.10457464e-19,  2.00326720e-03,  1.00000000e+00],
])
inv_matrix = np.linalg.inv(matrix)


def warp_image(image, matrix):
    image = cv2.warpPerspective(image, matrix, (1920, 1080))
    return image


def warp_rois(rois, matrix):
    orig_shape = rois.shape
    rois = rois.astype('float32').reshape(1, -1, 2)
    rois = cv2.perspectiveTransform(rois, matrix)
    rois = np.round(rois.reshape(*orig_shape)).astype('int32')
    return rois


def draw_polygon_on_im(image, polys, thickness=3, color=(0, 0, 255)):
    for poly in polys:
        image = cv2.polylines(image, [poly], True, color, thickness=thickness)
    return image


parser = argparse.ArgumentParser(description='Prepare dataset for training with trapezoid input shapes')
parser.add_argument("dataset", help="Path to original unzipped wheat dataset, needed for extracting labels")
parser.add_argument("framesdataset", help="Path to frames dataset, this is the image source")
parser.add_argument("--trapezoids", help="Use trapezoids instead of squares", action="store_true")
parser.add_argument("output", help="Path to store the prepared dataset")
args = parser.parse_args()

# output folder structure:
# output:
#   - spray
#       - <basename>_x1_x2_x3_x4.jpg
#   - dont
#       - <basename>_x1_x2_x3_x4.jpg
#
# where x1, x2, x3, x4 are x-coordinates indicating the shape of the trapezoid mask that the training framework
# need to overlay during pre-processing

# extract label data from original dataset
info = {} # fileid: {crop1: label1, crop2: label2, crop3: label3, crop4: label4}
for filename in glob.glob(os.path.join(args.dataset, '*/*.jpg')):
    label = filename.split('/')[-2]
    fileid, crop, _ = os.path.basename(filename).split('.jpg')
    crop = crop[1:] # remove leasing '_'
    crop = tuple([int(x) for x in crop.split('_')]) # convert to tuple coords

    if fileid not in info.keys():
        info[fileid] = {crop: label}
    else:
        info[fileid][crop] = label

# obtain trapezoid rois
rects = tuple(info[list(info.keys())[0]].keys())        # get the four crop rects from an arbitrary value in `info`
rois = np.array([rect_to_roi(rect) for rect in rects])  # x, y, w, h to ((x1, y1), (x2, y2), (x3, y3), (x4, y4))
if args.trapezoids:
    rois = warp_rois(rois, inv_matrix)                  # rect rois to trapezoids

rois_map = {k: v for k, v in zip(rects, rois)}          # make a lookup table
#img = np.zeros((1080, 1920, 3), dtype='uint8')
#img = draw_polygon_on_im(img, rois)
#cv2.imshow('out', img)
#cv2.waitKey(0)

if not os.path.exists(args.output):
    os.makedirs(args.output)

# iterate through image frames
for filename in glob.glob(os.path.join(args.framesdataset, '*.jpg')):
    fileid = os.path.splitext(os.path.basename(filename))[0]
    crop_label_data = info[fileid]
    img = cv2.imread(filename)
    img = warp_image(img, inv_matrix)
    if not args.trapezoids:
        img = warp_image(img, matrix)

    # iterate through each ROI area in the image, crop out a rectangle and save this to disk
    for crop, label in crop_label_data.items():
        roi = rois_map[crop]
        x, y, w, h = roi_to_rect(roi)
        img_crop = img[y:y+h, x:x+w]
        crop_coords = "_".join([f'{v}' for v in [x, y, w, h]])
        mask_coords = "_".join([f'{x_-x}' for x_, _ in roi])
        dirname = os.path.join(args.output, label)
        if not os.path.exists(dirname):
            os.makedirs(dirname)
        dst_filename = os.path.join(dirname, fileid + '_' + crop_coords + '_' + mask_coords + '.jpg')
        print("Img size =", img_crop.shape, ", img loc =", dst_filename)
        cv2.imwrite(dst_filename, img_crop)
