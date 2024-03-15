import cv2
import numpy as np


# define square rois: taken from wheat dataset
crops = [
    [1216, 200, 448, 448],
    [256, 200, 448, 448],
    [456, 350, 672, 672],
    [791, 350, 672, 672],
]

# define homography matrix
# matrix determined by `read_trapezoid2.py`
matrix = np.array([
    [ 1.51130629e+00,  1.92260258e+00, -4.89180994e+02],
    [ 1.33971426e-16,  5.14071420e+00, -9.87483229e+02],
    [ 1.10457464e-19,  2.00326720e-03,  1.00000000e+00],
])
inv_matrix = np.linalg.inv(matrix)

# calculate trapezoid rois by transforming the square rois using the homography matrix
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


def warp_rois(rois, matrix):
    orig_shape = rois.shape
    rois = rois.astype('float32').reshape(1, -1, 2)
    rois = cv2.perspectiveTransform(rois, matrix)
    rois = np.round(rois.reshape(*orig_shape)).astype('int32')
    return rois


rois = np.array([rect_to_roi(rect) for rect in crops])  # x, y, w, h to ((x1, y1), (x2, y2), (x3, y3), (x4, y4))
rois = warp_rois(rois, inv_matrix)                      # rectangular rois to trapezoids

# calculate tight rectangles around trapezoids and store them to disk
def roi_to_rect(roi):
    """Return the rectangular shape (x, y, w, h) that fits tightly around the trapezoid ROI
    roi:    ROI coords, start top left and go counter clockwise: [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]
    """
    x1 = min(roi[0][0], roi[1][0])
    y1 = min(roi[0][1], roi[3][1])
    x2 = max(roi[2][0], roi[3][0])
    y2 = max(roi[2][1], roi[2][1])
    return x1, y1, (x2 - x1), (y2 - y1)

rects = np.array([roi_to_rect(roi) for roi in rois])
np.save("rects.npy", rects)

# create relative rois: use (h, w) of each rectangle in `rects` as the new reference frame
new_rois = []
for roi, rect in zip(rois, rects):
    x, y, _, _ = rect
    new_rois.append([(_x-x, _y-y) for _x, _y in roi])
new_rois = np.array(new_rois)


# get homography matrix for each trapezoid roi to 224x224 squares
out_shape = (224, 224)
matrices = []
for roi in new_rois:
    pts1 = roi.astype("float32")
    h, w = out_shape
    pts2 = np.array([(0, 0), (0, h), (w, h), (w, 0)], dtype="float32")
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    matrices.append(matrix)


# calculate map_x and map_y pixel maps for cv2.remap and store them to disk
mapxs = []
mapys = []
for matrix in matrices:
    mapx, mapy = cv2.initUndistortRectifyMap(np.eye(3, 3), (), None, matrix, out_shape, cv2.CV_32FC1)
    mapx, mapy = cv2.convertMaps(mapx, mapy, cv2.CV_16SC2, nninterpolation=False)
    mapxs.append(mapx)
    mapys.append(mapy)

mapxs = np.array(mapxs)
mapys = np.array(mapys)
np.save("mapxs.npy", mapxs)
np.save("mapys.npy", mapys)


# read an image, crop our rectangles, apply remap and display the result
frame = cv2.imread('../data/Wheat/Frames/Wheat_20210827142622_1_10206.jpg')
frame = cv2.warpPerspective(frame, inv_matrix, (1920, 1080))

# draw trapezoids
def draw_polygon_on_im(image, polys, thickness, color=(0, 255, 255)):
    for poly in polys:
        image = cv2.polylines(image, [poly], True, color, thickness=thickness)
    return image

for roi, rect, mapx, mapy in zip(new_rois, rects, mapxs, mapys):
    x, y, w, h = rect
    crop = frame[y:y+h, x:x+w]
    crop_vis = crop.copy()
    crop_vis = draw_polygon_on_im(crop_vis, roi.reshape(1, 4, 2), 2)
    cv2.imshow('crop', crop_vis)
    result = cv2.remap(crop, mapx, mapy, cv2.INTER_LINEAR)
    cv2.imshow('result', result)
    cv2.waitKey(0)
