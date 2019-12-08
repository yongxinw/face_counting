from mtcnn import MTCNN
from Learner import face_learner
from config import get_config
from utils import draw_box_name

# from RetinaFace import test

import os
import os.path as osp
import argparse
from PIL import Image, ImageDraw
import numpy as np
import torch
import cv2
import json
import random

from mtcnn_pytorch.src.align_trans import warp_and_crop_face, get_reference_facial_points


# detect.detect()


def initialize_face_bank_mtcnn(image, mtcnn, learner, conf):
    embeddings = []
    names = ["Unknown"]
    emb_counts = []
    bboxes, faces = mtcnn.align_multi(image, conf.face_limit, 16)
    for i, img in enumerate(faces):
        emb = learner.model(conf.test_transform(img).to(conf.device).unsqueeze(0))
        embeddings.append(emb)
        names.append("{:02d}".format(i))
        emb_counts.append(1)
    embeddings = torch.cat(embeddings)
    return embeddings, names, emb_counts, bboxes[:, 0:4]


def initialize_face_bank_retinaface(img, retinaface_res_path):
    embeddings = []
    names = ["Unknown"]
    emb_counts = []

    # get retinaface detections
    refrence = get_reference_facial_points(default_square=True)
    retinaface_res = np.loadtxt(retinaface_res_path)
    info = retinaface_res[retinaface_res[:, 0] == 1]
    if info.shape[0] > 0:
        bboxes = info[:, 1:5]
        landmarks = info[:, 5:]

        # Warp faces
        faces = []
        for i, landmark in enumerate(landmarks):
            facial5points = [[landmark[j], landmark[j + 1]] for j in range(0, 10, 2)]
            warped_face = warp_and_crop_face(np.array(img), facial5points, refrence, crop_size=(112, 112))
            faces.append(Image.fromarray(warped_face))

        # Extracting face embeddings
        for i, img in enumerate(faces):
            emb = learner.model(conf.test_transform(img).to(conf.device).unsqueeze(0))
            embeddings.append(emb)
            names.append("{:02d}".format(i))
            emb_counts.append(1)
        embeddings = torch.cat(embeddings)
        return embeddings, names, emb_counts, bboxes
    return None


def update_face_bank(embeddings, source_embs, min_idx, emb_counts, names):
    # print(min_idx)
    for i, idx in enumerate(min_idx):
        if idx != -1:  # we find a match, smooth the previous embeddings
            embeddings[idx] = (emb_counts[idx] * embeddings[idx] + source_embs[i]) / (emb_counts[idx] + 1)
            emb_counts[idx] += 1
        else:
            embeddings = torch.cat((embeddings, source_embs[i].unsqueeze(0)), dim=0)
            emb_counts.append(1)
            names.append("{:02d}".format(int(names[-1]) + 1))
            min_idx[i] = len(embeddings) - 1
    return embeddings


def keypoints2box(keypoints):
    keypoints = keypoints[keypoints[:, 2] != 0]
    if len(keypoints) >= 34:
        xmax, xmin, ymax, ymin = np.max(keypoints[:, 0]), np.min(keypoints[:, 0]), \
                                 np.max(keypoints[:, 1]), np.min(keypoints[:, 1])

        box = [int(xmin), int(ymin), int(xmax), int(ymax)]
        return box
    return [0, 0, 0, 0]

def process_openpose(openpose_dir, path):
    person_data = json.load(
        open(osp.join(openpose_dir,
                      "{:06d}_keypoints.json".format(int(path.replace(".jpg", "")))
                      ), 'r'
             )
    )["people"]

    body_bboxes = []
    face_bboxes = []
    for person in person_data:
        keypoints = np.array(person["pose_keypoints_2d"]).reshape(-1, 3)
        facekeypoints = np.array(person["face_keypoints_2d"]).reshape(-1, 3)
        lh_keypoints = np.array(person["hand_left_keypoints_2d"]).reshape(-1, 3)
        rh_keypoints = np.array(person["hand_right_keypoints_2d"]).reshape(-1, 3)
        # print(keypoints)
        body_box = keypoints2box(np.concatenate((keypoints, facekeypoints, lh_keypoints, rh_keypoints), axis=0))
        body_bboxes.append(body_box)
        face_bboxes.append(keypoints2box(facekeypoints))

    return np.array(body_bboxes), np.array(face_bboxes)


def np_vec_no_jit_iou(boxes1, boxes2):
    def run(bboxes1, bboxes2):
        x11, y11, x12, y12 = np.split(bboxes1, 4, axis=1)
        x21, y21, x22, y22 = np.split(bboxes2, 4, axis=1)

        # determine the (x, y)-coordinates of the intersection rectangle
        xA = np.maximum(x11, np.transpose(x21))
        yA = np.maximum(y11, np.transpose(y21))
        xB = np.minimum(x12, np.transpose(x22))
        yB = np.minimum(y12, np.transpose(y22))

        # compute the area of intersection rectangle
        interArea = np.maximum((xB - xA + 1), 0) * np.maximum((yB - yA + 1), 0)

        # compute the area of both the prediction and ground-truth rectangles
        boxAArea = (x12 - x11 + 1) * (y12 - y11 + 1)
        boxBArea = (x22 - x21 + 1) * (y22 - y21 + 1)

        iou = interArea / (boxAArea + np.transpose(boxBArea) - interArea)

        return iou

    return run(boxes1, boxes2)


def merge_box(box1, box2):
    return np.array([min(box1[0], box2[0]),
                     min(box1[1], box2[1]),
                     max(box1[2], box2[2]),
                     max(box1[3], box2[3])])


def find_retinaface_detections(img, retinaface_res_path, im_path):
    print("loading from {:s}".format(retinaface_res_path))
    refrence = get_reference_facial_points(default_square= True)
    frame = int(im_path.replace(".jpg", ""))

    retinaface_res = np.loadtxt(retinaface_res_path)
    info = retinaface_res[retinaface_res[:, 0] == frame]
    if info.shape[0] > 0:
        bboxes = info[:, 1:5]
        landmarks = info[:, 5:]

        faces = []
        for i, landmark in enumerate(landmarks):
            facial5points = [[landmark[j], landmark[j + 1]] for j in range(0, 10, 2)]
            warped_face = warp_and_crop_face(np.array(img), facial5points, refrence, crop_size=(112, 112))

            im_warped = Image.fromarray(warped_face)
            os.makedirs(osp.join(image_root, "debug"), exist_ok=True)
            im_warped.save(osp.join(image_root, "debug", "face_warped_{}_{}.jpg".format(frame, i)))

            faces.append(Image.fromarray(warped_face))
    else:
        bboxes = []
        faces = []
        landmarks = []
    print(bboxes)
    return bboxes, faces, landmarks


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='for face verification')
    parser.add_argument('-th', '--threshold', help='threshold to decide identical faces', default=1.00, type=float)
    parser.add_argument("-c", "--score", help="whether show the confidence score", action="store_true")
    parser.add_argument("-d", "--detector", help="what face detector to use", default="retinaface", type=str)

    args = parser.parse_args()

    conf = get_config(False)


    image_root = "/work/yongxinw/SocialIQ/raw/raw/vision/"
    all_sequences = np.array(os.listdir(osp.join(image_root, "raw")))
    # sequences = [
    #     # "UUukBV82P9A_trimmed-out.mp4",
    #     "uB3FdWmnFZU_trimmed-out",
    #     "ztlOyCk5pcg_trimmed-out",
    #     # "ZYzql-Y1sP4_trimmed-out.mp4",
    #     "zzERcgFrTrA_trimmed-out",
    #     "zzWaf1z9kgk_trimmed-out"
    # ]

    sequences = [
        "urYGhhMOToU_trimmed-out",
        "iBL0FcUTFT8_trimmed-out",
        "DZsBei4nCkU_trimmed-out",
        "g8D-LyfTrRs_trimmed-out",
        "qDpGgd4oTQ8_trimmed-out",
        "c67D5bP0Hg4_trimmed-out",
        "pMGhqE76kQA_trimmed-out",
        "mA402F5K47o_trimmed-out",
        "SAgYiERRDPY_trimmed-out",
        "L49M8C4wjVU_trimmed-out"
    ]
        # load face detector
    mtcnn = MTCNN()
    print('mtcnn loaded')

    # initialize face verification module
    learner = face_learner(conf, True)
    learner.threshold = args.threshold
    learner.load_state(conf, fixed_str="ir_se50.pth", from_save_folder=False, model_only=True)
    learner.model.eval()
    print('arcface model loaded')

    for seq in sequences:
        image_dir = osp.join(image_root, "images", seq)
        openpose_dir = osp.join("/home", "yongxinw", "data", "images", seq)

        # extract video to frames and create openpose dir
        if not osp.exists(image_dir):
            os.makedirs(image_dir, exist_ok=True)
            video_path = osp.join(image_root, "raw", "{}.mp4".format(seq))
            os.system("ffmpeg -i {} {}/%06d.jpg".format(video_path, image_dir))

        image_paths = sorted(os.listdir(osp.join(image_root, "images", seq)))
        os.makedirs(osp.join(image_root, "results", "tracking_noid2", args.detector, seq), exist_ok=True)

        openpose_dir = osp.join("/home", "yongxinw", "data", "images", seq)

        print("Processsing {}".format(seq))
        init_frame = 0
        names0 = []
        while len(names0) == 0:
            # Initialize identities (i.e. face bank) using the first valid frame
            image0 = Image.open(osp.join(image_root, "images", seq, image_paths[init_frame]))

            if args.detector == "mtcnn":
                try:
                    embeddings, names, emb_counts, bboxes0 = initialize_face_bank_mtcnn(image0, mtcnn, learner, conf)
                except:
                    init_frame += 1
                    continue
                names0.extend(names[1:])
            elif args.detector == "retinaface":
                # retinaface_result_path0 = osp.join(image_root, "results", "detection", "RetinaFace", seq, "results.txt")
                retinaface_result_path0 = osp.join(image_root, "results", "RetinaFace", seq, "results.txt")
                try:
                    embeddings, names, emb_counts, bboxes0 = initialize_face_bank_retinaface(image0, retinaface_result_path0)
                except:
                    init_frame += 1
                    continue
                names0.extend(names[1:])
            else:
                raise NotImplementedError

        res_file = osp.join(image_root, "results", "tracking_noid2", args.detector, seq, "tracks.txt")
        # clear buffer
        f = open(res_file, 'w')
        f.close()

        for path in image_paths[init_frame:]:

            frame = cv2.imread(osp.join(image_root, "images", seq, path))
            # im = Image.fromarray(frame)
            im = Image.open(osp.join(image_root, "images", seq, path))
            print(path)
            if args.detector == "mtcnn":
                try:
                    detected_bboxes, faces, landmarks = mtcnn.align_multi_with_landmarks(im, conf.face_limit, 16,
                                                                                         thresholds=[0.6, 0.7, 0.8])
                except:
                    detected_bboxes = []
                    faces = []
            elif args.detector == "retinaface":
                # retinaface_result_path = osp.join(image_root, "results", "detection", "RetinaFace", seq, "results.txt")
                retinaface_result_path = osp.join(image_root, "results", "RetinaFace", seq, "results.txt")
                detected_bboxes, faces, landmarks = find_retinaface_detections(img=im,
                                                                               retinaface_res_path=retinaface_result_path,
                                                                               im_path=path)
            else:
                raise NotImplementedError

            if len(detected_bboxes) == 0:
                print('no face')
            else:
                with torch.no_grad():
                    results, score, source_embs = learner.infer_embeddings(conf, faces, embeddings, True)

                embeddings = update_face_bank(embeddings, source_embs, results, emb_counts, names)

                if args.detector == "mtcnn":
                    detected_bboxes = detected_bboxes[:, :-1]  # shape:[10,4],only keep 10 highest possibiity faces
                detected_bboxes = detected_bboxes.astype(int)
                detected_bboxes = detected_bboxes + [-1, -1, 1, 1]  # personal choice

                openpose_body_bboxes, openpose_face_bboxes = process_openpose(openpose_dir, path)

                if len(openpose_face_bboxes) == 0:
                    openpose_face_bboxes = detected_bboxes
                    openpose_body_bboxes = detected_bboxes
                # for idx, bbox in enumerate(detected_bboxes):
                #     if args.score:
                #         frame = draw_box_name(bbox, names[results[idx] + 1] + '_{:.2f}'.format(score[idx]), frame)
                #     else:
                #         frame = draw_box_name(bbox, names[results[idx] + 1], frame)

                # iou between face detector and openpose face boxes
                face_iou = np_vec_no_jit_iou(openpose_face_bboxes, detected_bboxes)
                # matches
                max_ious, max_inds = np.max(face_iou, axis=1), np.argmax(face_iou, axis=1)
                max_inds[max_ious <= 0.1] = -1

                curr_names = []
                curr_boxes = []
                out = []
                for j, bodybbox in enumerate(openpose_body_bboxes):
                    matched_detection = max_inds[j]

                    # Draw boxes only if there's a match between the detector faces and the openpose faces
                    if matched_detection != -1:
                        detected_box = detected_bboxes[matched_detection]
                        # merge the body box with the matched detection
                        merged_box = merge_box(bodybbox, detected_box)

                        # match for the previous frame using IOU
                        # iou = np_vec_no_jit_iou(np.array([detected_box]), bboxes0)
                        # max_iou, max_ind = np.max(iou, axis=1), np.argmax(iou, axis=1)
                        # print(iou, max_ind)
                        # if max_iou >= 0.5:
                        #     # we have an IoU over 0.5, then there's match. We use the previous frame information
                        #     identity = names0[max_ind[0]]
                        #     frame = draw_box_name(merged_box, identity, frame)
                        #     frame = draw_box_name(detected_box, identity, frame)
                        #     curr_names.append(identity)
                        # else:  # otherwise, look up in the face bank
                        identity = names[results[matched_detection] + 1]
                        frame = draw_box_name(merged_box, "", frame)
                        frame = draw_box_name(detected_box, "", frame)

                        curr_names.append(identity)
                        curr_boxes.append(detected_box)
                        out.append([int(path.replace(".jpg", "")), identity,
                                    detected_box[0], detected_box[1], detected_box[2], detected_box[3],
                                    merged_box[0], merged_box[1], merged_box[2], merged_box[3]])
                    else:
                        face_bbox = openpose_face_bboxes[j]
                        merged_box = merge_box(bodybbox, face_bbox)
                        frame = draw_box_name(merged_box, "", frame)
                        out.append([int(path.replace(".jpg", "")), -1,
                                    face_bbox[0], face_bbox[1], face_bbox[2], face_bbox[3],
                                    merged_box[0], merged_box[1], merged_box[2], merged_box[3]])

                with open(res_file, 'a') as res:
                    np.savetxt(res, np.array(out), fmt="%s")
                    res.close()
                # Update the previous detections
                if len(curr_names) != 0:
                    names0 = curr_names
                if len(curr_boxes) != 0:
                    bboxes0 = np.array(curr_boxes)

                print(results, score, embeddings.shape, emb_counts, names)

            cv2.imwrite(osp.join(image_root, "results", "tracking_noid2", args.detector, seq, path), frame)
