import cv2
import sys
import numpy as np
import datetime
import os
import glob
from retinaface import RetinaFace
import sys

seq = sys.argv[1]
image_root = sys.argv[2]
res_root = sys.argv[3]

if not os.path.exists(res_root):
    os.makedirs(res_root, exist_ok=True)

thresh = 0.8
scales = [1024, 1980]

count = 1

gpuid = 0
detector = RetinaFace('./model/R50', 0, gpuid, 'net3')

# res_root = "/work/yongxinw/SocialIQ/raw/raw/vision/results/"
# image_root = "/work/yongxinw/SocialIQ/raw/raw/vision/results/extracted_frames"
# sequences = ["zzERcgFrTrA_trimmed-out", "zzWaf1z9kgk_trimmed-out", "ztlOyCk5pcg_trimmed-out"]

# image_root = "/projects/dataset_processed/abagherz/to_richard_temp"
# res_root = "/work/yongxinw/SocialIQ/raw/raw/vision/results/yuying/retinaface_yuying/"

# seq_len = len(os.listdir(os.path.join(image_root, "images", seq)))
seq_len = len(os.listdir(os.path.join(image_root, "frames", seq)))

res_dir = os.path.join(res_root, seq)
os.makedirs(res_dir, exist_ok=True)

res = []
for frame in range(1, seq_len + 1):
    # print("Processing {:s}".format(os.path.join(image_root, "images", seq, "{:06d}.jpg".format(frame))))
    print("Processing {:s}".format(os.path.join(image_root, "frames", seq, "{}-{:04d}.jpg".format(seq, frame))))
    # img = cv2.imread(os.path.join(image_root, "images", seq, "{:06d}.jpg".format(frame)))
    img = cv2.imread(os.path.join(image_root, "frames", seq, "{}-{:04d}.jpg".format(seq, frame)))
    # print(img.shape)
    im_shape = img.shape
    target_size = scales[0]
    max_size = scales[1]
    im_size_min = np.min(im_shape[0:2])
    im_size_max = np.max(im_shape[0:2])
    # im_scale = 1.0
    # if im_size_min>target_size or im_size_max>max_size:
    im_scale = float(target_size) / float(im_size_min)
    # prevent bigger axis from being more than max_size:
    if np.round(im_scale * im_size_max) > max_size:
        im_scale = float(max_size) / float(im_size_max)

    print('im_scale', im_scale)

    _scales = [im_scale]
    flip = False

    for c in range(count):
        faces, landmarks = detector.detect(img, thresh, scales=_scales, do_flip=flip)
        print(c, faces.shape, landmarks.shape)

    if faces is not None:
        print('find', faces.shape[0], 'faces')
        line = []
        for i in range(faces.shape[0]):
            # print('score', faces[i][4])
            box = faces[i].astype(np.int)
            # color = (255,0,0)
            color = (0, 0, 255)
            cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), color, 2)
            line.extend([frame, box[0], box[1], box[2], box[3]])  # frame_no, x1, y1, x2, y2
            if landmarks is not None:
                landmark5 = landmarks[i].astype(np.int)
                # print(landmark.shape)
                for l in range(landmark5.shape[0]):
                    color = (0, 0, 255)
                    if l == 0 or l == 3:
                        color = (0, 255, 0)
                    cv2.circle(img, (landmark5[l][0], landmark5[l][1]), 1, color, 2)
                    line.extend([landmark5[l][0], landmark5[l][1]])
            res.append(line)
            line = []
    # filename = os.path.join(res_dir, 'retinaface_{:06d}.jpg'.format(frame))
    filename = os.path.join(res_dir, '{}-{:06d}-retinaface.jpg'.format(seq, frame))
    print('writing', filename)
    cv2.imwrite(filename, img)

res = np.array(res)
np.savetxt(os.path.join(res_dir, "results-{}.txt".format(seq)), res, fmt="%12.3f")
