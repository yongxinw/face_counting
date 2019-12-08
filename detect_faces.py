# Created by yongxinwang at 2019-09-27 16:32
# from mmsdk import mmdatasdk

# Download social iq datasets
# sociaiq = mmdatasdk.mmdataset(mmdatasdk.socialiq.highlevel, destination="/work/yongxinw/SocialIQ")

from mtcnn import MTCNN
from Learner import face_learner
from config import get_config
from utils import draw_box_name

import os
import os.path as osp
import argparse
from PIL import Image, ImageDraw
import numpy as np
import cv2


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='for face verification')
    parser.add_argument("-f", "--file_name", help="video file name", default='video.mp4', type=str)
    parser.add_argument("-s", "--save_name", help="output file name", default='recording', type=str)
    parser.add_argument('-th', '--threshold', help='threshold to decide identical faces', default=1.54, type=float)
    parser.add_argument("-u", "--update", help="whether perform update the facebank", action="store_true")
    parser.add_argument("-tta", "--tta", help="whether test time augmentation", action="store_true")
    parser.add_argument("-c", "--score", help="whether show the confidence score", action="store_true")
    parser.add_argument("-b", "--begin", help="from when to start detection(in seconds)", default=0, type=int)
    parser.add_argument("-d", "--duration", help="perform detection for how long(in seconds)", default=0, type=int)

    args = parser.parse_args()

    conf = get_config(False)

    sequences = ["ZYzql-Y1sP4_trimmed-out"]
    image_root = "/work/yongxinw/SocialIQ/raw/raw/vision/"

    mtcnn = MTCNN()
    print('mtcnn loaded')

    learner = face_learner(conf, True)
    learner.threshold = args.threshold

    for seq in sequences:
        image_paths = sorted(os.listdir(osp.join(image_root, "images", seq)))
        os.makedirs(osp.join(image_root, "results", seq), exist_ok=True)
        face_detections = []
        face_landmarks = []
        for i, path in enumerate(image_paths):
            im = Image.open(osp.join(image_root, "images", seq, path))
            draw = ImageDraw.Draw(im)
            print(i)
            try:
                # bboxes, faces = mtcnn.align_multi(im, conf.face_limit, 16)
                bboxes, faces, landmarks = mtcnn.align_multi_with_landmarks(im, conf.face_limit, 16)
            except:
                bboxes = []
                faces = []

            if len(bboxes) == 0:
                print('no face')
                continue
            else:
                bboxes = bboxes[:,:-1] #shape:[10,4],only keep 10 highest possibiity faces
                bboxes = bboxes.astype(int)
                bboxes = bboxes + [-1,-1,1,1] # personal choice
                for idx, bbox in enumerate(bboxes):
                    draw.rectangle(((bbox[0], bbox[1]), (bbox[2], bbox[3])))
                    for ld in range(len(landmarks[idx]) // 2):
                        draw.point((landmarks[idx][ld], landmarks[idx][ld+5]))
                    face_detections.append([i+1, bbox[0], bbox[1], bbox[2], bbox[3]])
                    face_landmarks.append(list(landmarks[idx]))

            im.save(osp.join(image_root, "results", seq, path))
        np.savetxt(osp.join(image_root, "results", seq, "detections.txt"), np.array(face_detections), fmt="%.3f")
        np.savetxt(osp.join(image_root, "results", seq, "landmarks.txt"), np.array(face_landmarks), fmt="%.3f")
