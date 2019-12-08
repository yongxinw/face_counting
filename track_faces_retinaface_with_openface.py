# Created by yongxinwang at 2019-10-21 02:55
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
import pandas as pd

from mtcnn_pytorch.src.align_trans import warp_and_crop_face, get_reference_facial_points


def merge_box(box1, box2):
    return np.array([min(box1[0], box2[0]),
                     min(box1[1], box2[1]),
                     max(box1[2], box2[2]),
                     max(box1[3], box2[3])])


def keypoints2box_openpose(keypoints):
    keypoints = keypoints[keypoints[:, 2] != 0]
    if len(keypoints) >= 34:
        xmax, xmin, ymax, ymin = np.max(keypoints[:, 0]), np.min(keypoints[:, 0]), \
                                 np.max(keypoints[:, 1]), np.min(keypoints[:, 1])

        box = [int(xmin), int(ymin), int(xmax), int(ymax)]
        return box
    return [0, 0, 0, 0]


def keypoints2box(keypoints):
    if len(keypoints) >= 34:
        xmax, xmin, ymax, ymin = np.max(keypoints[:, 0]), np.min(keypoints[:, 0]), \
                                 np.max(keypoints[:, 1]), np.min(keypoints[:, 1])

        box = [int(xmin), int(ymin), int(xmax), int(ymax)]
        return box
    return [0, 0, 0, 0]


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


def process_openpose(openpose_dir, seq, frame_num):
    # person_data = json.load(
    #     open(osp.join(openpose_dir,
    #                   "{}_{:012d}_keypoints.json".format(seq, int(path.replace(".jpg", "")))
    #                   ), 'r'
    #          )
    # )["people"]
    # person_data = json.load(
    #     open(osp.join(openpose_dir,
    #                   "{}_{:012d}_keypoints.json".format(seq, frame_num)
    #                   ), 'r'
    #          )
    # )["people"]

    person_data = json.load(
        open(osp.join(openpose_dir,
                      "{}-{:04d}_keypoints.json".format(seq, frame_num)
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
        body_box = keypoints2box_openpose(np.concatenate((keypoints, facekeypoints, lh_keypoints, rh_keypoints), axis=0))
        body_bboxes.append(body_box)
        face_bboxes.append(keypoints2box(facekeypoints))

    return np.array(body_bboxes), np.array(face_bboxes)


def construct_face_bank(seq: str, retina_face_root: str, image_root: str, openface_root: str, learner: face_learner,
                        data_root: str):
    """
    :param seq: sequence id
    :param retina_face_root: root dir for retinaface detections. (frame, face_bbox (4), landmarks (10))
    :param image_root: root dir for images
    :param openface_root: root dir for openface results
    :param learner: a face verification network
    :return:
    """
    seq_images_dir = osp.join(image_root, seq)
    seq_images = sorted(os.listdir(seq_images_dir))
    seq_openface_dir = osp.join(openface_root, seq)
    seq_retina_face_result = np.loadtxt(osp.join(retina_face_root, seq, "results-{}.txt".format(seq)))

    embeddings = []
    names = ["Unknown"]
    emb_counts = []

    # get default reference face points
    reference = get_reference_facial_points(default_square=True)

    # get openface results
    # openface_result = pd.read_csv(osp.join(seq_openface_dir, "{}.csv".format(seq)), delimiter=",\s+")

    for frame_idx, frame in enumerate(seq_images):
        print("Processing {} {}".format(frame_idx, frame))
        print("Names: {}".format(names))
        print("emb_counts: {}".format(emb_counts))
        # frame_num = int(frame.replace(".jpg", ""))
        frame_num = int(frame.replace(".jpg", "").split("-")[-1])

        # 1. load data
        # 1.1. read image
        img = Image.open(osp.join(seq_images_dir, frame))
        # 1.2. get the retinaface detections
        retinaface_result = seq_retina_face_result[seq_retina_face_result[:, 0] == frame_num]
        # skip if no detection
        if len(retinaface_result) == 0:
            print("No retinaface")
            continue
        retinaface_bboxes = retinaface_result[:, 1:5]

        # 1.3. get the openface results
        # openface_result_frame = openface_result[openface_result['frame'] == frame_num]
        # if len(openface_result_frame) == 0:
        #     print("No OpenFace detection at frame {}".format(frame_num))
        #     continue
        # xs = ["x_{}".format(i) for i in range(68)]
        # ys = ["y_{}".format(i) for i in range(68)]
        #
        # landmark_x, landmark_y = np.array(openface_result_frame[xs]), np.array(openface_result_frame[ys])
        # confidences = np.array(openface_result_frame['confidence'])

        # skip if no results
        if not osp.exists(osp.join(seq_openface_dir, "{}-{:04d}.csv".format(seq, frame_num))):
            # print("Openface file {} does not exist".format(osp.join(seq_openface_dir, "{:06d}.csv".format(frame_num))))
            print("Openface file {} does not exist".format(osp.join(seq_openface_dir,  "{}-{:04d}.csv".format(seq, frame_num))))
            continue

        # openface_result = pd.read_csv(osp.join(seq_openface_dir, "{:06d}.csv".format(frame_num)), delimiter=",\s+")
        openface_result = pd.read_csv(osp.join(seq_openface_dir, "{}-{:04d}.csv".format(seq, frame_num)),
                                      delimiter=",\s+")
        xs = ["x_{}".format(i) for i in range(68)]
        ys = ["y_{}".format(i) for i in range(68)]

        landmark_x, landmark_y = np.array(openface_result[xs]), np.array(openface_result[ys])
        confidences = np.array(openface_result['confidence'])

        # 1.4. only keep the confident openface detections (as profile faces)
        openface_bboxes = []
        for i in range(len(landmark_x)):
            if confidences[i] > 0.5:
                landmarks = np.concatenate((landmark_x[i].reshape(-1, 1), landmark_y[i].reshape(-1, 1)), axis=1)
                openface_bboxes.append(keypoints2box(landmarks))
        openface_bboxes = np.array(openface_bboxes)
        # skip if no confident boxes
        if len(openface_bboxes) == 0:
            print("No openface confident faces")
            continue

        # 2. match between the retinaface detection and openface detection
        ious = np_vec_no_jit_iou(retinaface_bboxes, openface_bboxes)
        max_ious = np.max(ious, axis=1)
        max_inds = np.argmax(ious, axis=1)
        max_inds[max_ious <= 0.5] = -1

        # 2.1. filter retina_face detection based on openface confidence
        keep_inds = []
        for i, max_i in enumerate(max_inds):
            # only select those retinaface detections with a max_ious larger than 0.8 compared to
            # openface confident faces
            if max_i != -1:
                keep_inds.append(i)

        # skip if no match
        if len(keep_inds) == 0:
            print("No Match")
            continue
        landmarks = retinaface_result[keep_inds, 5:]
        retinaface_bboxes = retinaface_bboxes[keep_inds, :]

        # Warp faces: preparing input for learner
        faces = []
        for i, landmark in enumerate(landmarks):
            facial5points = [[landmark[j], landmark[j + 1]] for j in range(0, 10, 2)]
            warped_face = warp_and_crop_face(np.array(img), facial5points, reference, crop_size=(112, 112))
            faces.append(Image.fromarray(warped_face))

        # if the very first frame, we initialize face bank
        if len(embeddings) == 0:

            # Extracting face embeddings
            for i, img in enumerate(faces):
                with torch.no_grad():
                    emb = learner.model(conf.test_transform(img).to(conf.device).unsqueeze(0))
                embeddings.append(emb)
                names.append("{:02d}".format(i))
                emb_counts.append(1)
            embeddings = torch.cat(embeddings)
        else:  # otherwise we try to match, and update the face bank
            with torch.no_grad():
                print(len(faces))
                results, score, source_embs = learner.infer_embeddings(conf, faces, embeddings, True)
            # udpate facebank
            for i, idx in enumerate(results):
                if idx != -1:  # we find a match, smooth the previous embeddings
                    embeddings[idx] = (emb_counts[idx] * embeddings[idx] + source_embs[i]) / (emb_counts[idx] + 1)
                    emb_counts[idx] += 1
                else:  # otherwise this is a new face
                    embeddings = torch.cat((embeddings, source_embs[i].unsqueeze(0)), dim=0)
                    emb_counts.append(1)
                    names.append("{:02d}".format(int(names[-1]) + 1))
                    results[i] = len(embeddings) - 1

    # Only keep the embeddings that appeared 3 times or more.
    keep_inds_by_count = np.where(np.array(emb_counts) > 2)[0]
    if len(keep_inds_by_count) > 0:
        embeddings = embeddings[keep_inds_by_count, :]
        names = [names[0]] + list(np.array(names)[keep_inds_by_count])
        emb_counts = list(np.array(emb_counts)[keep_inds_by_count])

    save_dir = osp.join(data_root, "results", "tracking_openface_yuying", args.detector, "npzs")
    os.makedirs(save_dir, exist_ok=True)
    np.savez(osp.join(save_dir, "{}.npz".format(seq)),
             embeddings=embeddings.cpu().numpy(), names=names, emb_counts=emb_counts)
    return embeddings, names, emb_counts


def match_faces(seq, retina_face_root, image_root, openpose_root, learner, face_bank, names, data_root, emb_counts):
    """

    :param seq: sequence id
    :param retina_face_root: root dir for retinaface detections. (frame, face_bbox (4), landmarks (10))
    :param image_root: root dir for images
    :param openpose_root: root dir for openpose results
    :param learner: a face verification network
    :param face_bank: the constructed facebank
    :param names: the identity names
    :param data_root: root dir for image data
    :param emb_counts: embedding counts
    :return:
    """
    seq_images_dir = osp.join(image_root, seq)
    seq_images = sorted(os.listdir(seq_images_dir))
    seq_openpose_dir = osp.join(openpose_root, seq)
    seq_retina_face_result = np.loadtxt(osp.join(retina_face_root, seq, "results-{}.txt".format(seq)))

    # get default reference face points
    reference = get_reference_facial_points(default_square=True)
    res = []
    for frame in seq_images:
        print("Processing {}".format(frame))
        # frame_num = int(frame.replace(".jpg", ""))
        frame_num = int(frame.replace(".jpg", "").split("-")[-1])

        # 1. process data
        # 1.1. read image
        img = Image.open(osp.join(seq_images_dir, frame))
        img_cv2 = cv2.imread(osp.join(seq_images_dir, frame))

        # 1.2. get the retinaface detections
        retinaface_result = seq_retina_face_result[seq_retina_face_result[:, 0] == frame_num]
        # skip if no detection
        if len(retinaface_result) == 0:
            os.makedirs(osp.join(data_root, "results", "tracking_openface_yuying", args.detector, seq), exist_ok=True)
            cv2.imwrite(osp.join(data_root, "results", "tracking_openface_yuying", args.detector, seq, frame), img_cv2)
            continue

        retinaface_bboxes = retinaface_result[:, 1:5]
        landmarks = retinaface_result[:, 5:]

        # 1.3. load openpose as boxes
        openpose_body_bboxes, openpose_face_bboxes = process_openpose(seq_openpose_dir, seq, frame_num)
        if len(openpose_face_bboxes) == 0 or len(openpose_body_bboxes) == 0:
            os.makedirs(osp.join(data_root, "results", "tracking_openface_yuying", args.detector, seq), exist_ok=True)
            cv2.imwrite(osp.join(data_root, "results", "tracking_openface_yuying", args.detector, seq, frame), img_cv2)
            continue

        # 2. classify retinaface bboxes into identities
        # 2.1. Warp faces: preparing input for learner
        faces = []
        for i, landmark in enumerate(landmarks):
            facial5points = [[landmark[j], landmark[j + 1]] for j in range(0, 10, 2)]
            warped_face = warp_and_crop_face(np.array(img), facial5points, reference, crop_size=(112, 112))
            faces.append(Image.fromarray(warped_face))
        # 2.2. forward pass
        with torch.no_grad():
            results, score, source_embs = learner.infer_embeddings(conf, faces, face_bank, True)

        # 2.3. match retinaface with openpose
        # iou between face detector and openpose face boxes
        face_iou = np_vec_no_jit_iou(openpose_face_bboxes, retinaface_bboxes)
        # matches
        max_ious, max_inds = np.max(face_iou, axis=1), np.argmax(face_iou, axis=1)
        max_inds[max_ious <= 0.5] = -1

        out = []
        for j, bodybbox in enumerate(openpose_body_bboxes):
            matched_detection = max_inds[j]
            detected_box = retinaface_bboxes[matched_detection]
            # merge the body box with the matched detection
            merged_box = merge_box(bodybbox, detected_box)
            # Draw boxes only if there's a match between the detector faces and the openpose faces
            if matched_detection != -1:
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
                freq = emb_counts[results[matched_detection]]
                img_cv2 = draw_box_name(merged_box.astype(int), identity, img_cv2)
                img_cv2 = draw_box_name(detected_box.astype(int), identity, img_cv2)

                out.append([frame_num, identity,
                            detected_box[0], detected_box[1], detected_box[2], detected_box[3],
                            merged_box[0], merged_box[1], merged_box[2], merged_box[3], freq])
            else:
                # draw the body box only
                img_cv2 = draw_box_name(bodybbox.astype(int), "", img_cv2)
                out.append([frame_num, -1, -1, -1, -1, -1,
                            bodybbox[0], bodybbox[1], bodybbox[2], bodybbox[3], -1])

        res.extend(out)
        os.makedirs(osp.join(data_root, "results", "tracking_openface_yuying", args.detector, seq), exist_ok=True)
        cv2.imwrite(osp.join(data_root, "results", "tracking_openface_yuying", args.detector, seq, frame), img_cv2)
    res = np.array(res)
    np.savetxt(osp.join(data_root, "results", "tracking_openface_yuying", args.detector, seq, "tracks.txt"), res,
               fmt="%s")
    return res


def run(args, conf):
    # image_root = "/work/yongxinw/SocialIQ/raw/raw/vision/results/extracted_frames/images"
    # image_root = "/work/yuyingz/socialIQ_raw/vision/frames/"
    # retina_face_root = "/work/yongxinw/SocialIQ/raw/raw/vision/results/yuying/retinaface_yuying/"
    # openpose_root = "/work/yongxinw/SocialIQ/raw/raw/vision/results/yuying/openpose_yuying/"
    # openface_root = "/work/yongxinw/SocialIQ/raw/raw/vision/results/yuying/openface_yuying/"
    # data_root = "/work/yongxinw/SocialIQ/raw/raw/vision/"
    image_root = "/hdd/yongxinw/SocialIQ/frames/"
    retina_face_root = "/hdd/yongxinw/SocialIQ/retinaface_yuying/"
    openpose_root = "/hdd/yongxinw/SocialIQ/openpose_yuying/"
    openface_root = "/hdd/yongxinw/SocialIQ/openface_yuying/"
    data_root = "/hdd/yongxinw/SocialIQ/"

    learner = face_learner(conf, True)
    learner.threshold = args.threshold
    learner.load_state(conf, fixed_str="ir_se50.pth", from_save_folder=False, model_only=True)
    learner.model.eval()
    sequences = [
        # "00m9ssEAnU4_trimmed-out",
        # "02A2a-aEvmI_trimmed-out",
        # "0_8zVtd5VLY_trimmed-out",
        # "09j-Mg5guGg_trimmed-out",
        # "0a2lv4IwZFY_trimmed-out",
        # "0aNFS1cLs4k_trimmed-out",
        # "_0at8kXKWSw_trimmed-out",
        # "0B7UgDEzcUM_trimmed-out",
        # "0B9VJfzqwhM_trimmed-out",
        # "0DBfvvIVmqY_trimmed-out",
        # "0djStqAuc_E_trimmed-out",
        # "0gcuigYJ2cw_trimmed-out",
        # "0hCihQ5Kep0_trimmed-out",
        # "0LDAgvbRtk4_trimmed-out",
        # "-0REX0yx4QA_trimmed-out",
        # "0SuGC2ifkIA_trimmed-out",
        # "11BN2alqrFU_trimmed-out",
        # "13HM_96pUIo_trimmed-out",
        # "13tG38-Ojn4_trimmed-out",
        # "17PtS1Qx8kU_trimmed-out",
        # "1a4Gx6UHdI8_trimmed-out",
        # "1A7dqFxx8wU_trimmed-out",
        # "1akNUksCpRU_trimmed-out",
        # "1B_swZA4Hqk_trimmed-out",
        # "1CjUHNYzW1E_trimmed-out",
        # "1GJqfyzfCWU_trimmed-out",
        # "1iH8ajlHJ1M_trimmed-out",
        # "1IHDvzYwqgE_trimmed-out",
        # "1j1C3dlwh7E_trimmed-out",
        # "1mHjMNZZvFo_trimmed-out",
        # "1MwN5nDajWs_trimmed-out",
        # "1YFLBjR-swo_trimmed-out",
        # "1YOzwKKbuPo_trimmed-out",
        # "1Za8BtLgKv8_trimmed-out",
        # "25Qq8k7V83g_trimmed-out",
        # "29rvfxBQBQA_trimmed-out",
        # "29vnZjb39u0_trimmed-out",
        # "2a01Rg2g2Z8_trimmed-out",
        # "2c_nei61mQQ_trimmed-out",
        # "2DTwXdqmRiM_trimmed-out",
        # "2G-B5upjLjM_trimmed-out",
        # "2GRzO0p9uVU_trimmed-out",
        # "2gy8H-wYzic_trimmed-out",
        # "2hLcCnZzRtY_trimmed-out",
        # "2ihOXaU0I8o_trimmed-out",
        # "2im0kvBEIrg_trimmed-out",
        # "2jMvc5VoavE_trimmed-out",
        # "2K09qUoN1Oo_trimmed-out",
        # "2MrFWB__GIA_trimmed-out",
        # "2nDh8MQuS-Y_trimmed-out",
        # "2NE4KCCfutk_trimmed-out",
        # "-2NF882DbaM_trimmed-out",
        # "2SIhhPzrjVA_trimmed-out",
        # "2Wk9JY6ic0k_trimmed-out",
        # "2XFVnzr4Vho_trimmed-out",
        # "2YGrrsKs-Xg_trimmed-out",
        # "2ZkzLAQszHA_trimmed-out",
        # "31ktAMJocw8_trimmed-out",
        # "34XCuNsQ7O8_trimmed-out",
        # "3_-Bjqs6AxA_trimmed-out",
        # "3d--LpVQxDo_trimmed-out",
        # "3eaASCCJB4U_trimmed-out",
        # "3eCZ8haia58_trimmed-out",
        # "3esHlM0cBx4_trimmed-out",
        # "3EwNcTzx-Bs_trimmed-out",
        # "3m-54UiEqzg_trimmed-out",
        # "3nUKwvFsjA4_trimmed-out",
        # "3oj7mCSydoM_trimmed-out",
        # "3qp3AeWmt38_trimmed-out",
        # "3udl28R3yIA_trimmed-out",
        # "3uk6rKXbG1M_trimmed-out",
        # "3vjA8sDxDuQ_trimmed-out",
        # "3wIejfT9l30_trimmed-out",
        # "3xdQIW24aj0_trimmed-out",
        # "3yovMKR__4Q_trimmed-out",
        # "3zjz6ryPvIg_trimmed-out",
        # "40mpZRU47T4_trimmed-out",
        # "43fC9xuQRCY_trimmed-out",
        # "44MVdpDEQJs_trimmed-out",
        # "460_-P8pK8E_trimmed-out",
        # "47lUTksozNI_trimmed-out",
        # "47U9SVOiw4o_trimmed-out",
        # "4AmVjblOvy4_trimmed-out",
        # "4An2AF2rWGk_trimmed-out",
        # "4_BudcPRi7E_trimmed-out",
        # "4EZxURAhU6U_trimmed-out",
        # "4HHR_3HJdEQ_trimmed-out",
        # "4HN0caXjW3s_trimmed-out",
        # "4HxgOizA2Ow_trimmed-out",
        # "4iw1jTY-X3A_trimmed-out",
        # "4_jXi0nzuow_trimmed-out",
        # "4KAvDyGzz4E_trimmed-out",
        # "4LGe265pwvU_trimmed-out",
        # "4oh_PsdY-W0_trimmed-out",
        # "4pYqIEQow2s_trimmed-out",
        # "4Ry2bE-WRqA_trimmed-out",
        # "4tLBy9FGS5A_trimmed-out",
        # "4VA4kqMnEqA_trimmed-out",
        # "4Vic0qKl64Y_trimmed-out",
        # "4vPrTC5qMh0_trimmed-out",
        # "4W192A7g5KY_trimmed-out",
        # "4wdeBJ39Cuw_trimmed-out",
        # "4yr_etbfZtQ_trimmed-out",
        # "56EbHWYK_q0_trimmed-out",
        # "58DqoE56OWc_trimmed-out",
        # "5BLjOCK2SlM_trimmed-out",
        # "5fy7S3jCyAg_trimmed-out",
        # "5h-SslT--8E_trimmed-out",
        # "5NHofFOpsDU_trimmed-out",
        # "5OUcvUDMMWE_trimmed-out",
        # "5RS1CKa3JVg_trimmed-out",
        # "5slZuaphzVs_trimmed-out",
        # "5_uSZcXMV7s_trimmed-out",
        # "5WgFDQUjg5s_trimmed-out",
        # "5XEQ8rYl1qs_trimmed-out",
        # "64mWOoj68qo_trimmed-out",
        # "66ojfophGys_trimmed-out",
        # "6AzXxhPKh8U_trimmed-out",
        # "6b1QbKtmaZ0_trimmed-out",
        # "6dCClwMqJK8_trimmed-out",
        # "6I7Ktp4dV_s_trimmed-out",
        # "6kYu7-5EyU8_trimmed-out",
        # "6qNawyzVGbc_trimmed-out",
        # "6rkV4QRcVnk_trimmed-out",
        # "6tAfdCTnToY_trimmed-out",
        # "6V0UfejAo_E_trimmed-out",
        # "6W77wcXg2no_trimmed-out",
        # "6xQv6ozrz90_trimmed-out",
        # "72ltfGTYqpQ_trimmed-out",
        # "79I7_vkwaeg_trimmed-out",
        # "7bc_qfRmPK0_trimmed-out",
        # "7doQf8xjFVg_trimmed-out",
        # "7FYHA728nBI_trimmed-out",
        # "7grGUUPbEbo_trimmed-out",
        # "7GRTyxc4uMU_trimmed-out",
        # "7GRWqlKfgmg_trimmed-out",
        # "7_lpdZhf28E_trimmed-out",
        # "7Oum_c5Seis_trimmed-out",
        # "7wLDCFduiLY_trimmed-out",
        # "87yBSfgwoUI_trimmed-out",
        # "8ACAI_Z7aLM_trimmed-out",
        # "8_Av3cDcoR8_trimmed-out",
        # "8fN6D1VOHlo_trimmed-out",
        # "-8GAQpsV4Qo_trimmed-out",
        # "8-Hi9NmF4rM_trimmed-out",
        # "8i0Vr6DiBCQ_trimmed-out",
        # "8Kv4F0D210A_trimmed-out",
        # "8m_3eBsy22Y_trimmed-out",
        # "8MK9frCMoWA_trimmed-out",
        # "8NL5jXoa-Jc_trimmed-out",
        # "8Rk4sGEBJlM_trimmed-out",
        # "8SGQ0VdXvAg_trimmed-out",
        # "8TDAP0KNIIw_trimmed-out",
        # "8w41NfRyWqE_trimmed-out",
        # "8wLCmDtCDAM_trimmed-out",
        # "8xFtIsyRvNE_trimmed-out",
        # "8y-N6UDxTxQ_trimmed-out",
        # "90P3VEbzUK0_trimmed-out",
        # "96YOZOU7ggo_trimmed-out",
        # "97AUfvzQ_1E_trimmed-out",
        # "-99aZZhUgRk_trimmed-out",
        # "9cFEh0aaOOo_trimmed-out",
        # "9eqze5JWNjY_trimmed-out",
        # "9hn6Z1o-IYI_trimmed-out",
        # "9jRkACywckE_trimmed-out",
        # "9kLNVTm3Z90_trimmed-out",
        # "9L1tM3fOb80_trimmed-out",
        # "9l2W_GDiNyE_trimmed-out",
        # "9m0d0RaWpfY_trimmed-out",
        # "-9NhaKWMtWU_trimmed-out",
        # "9PJb4cFWOfY_trimmed-out",
        # "9QdaNUrq1EQ_trimmed-out",
        # "9qK9VQDELpc_trimmed-out",
        # "a0tn33wZGVo_trimmed-out",
        # "A3WbCRfad-w_trimmed-out",
        # "A48AJ_5nWsc_trimmed-out",
        # "A4gVxvYFA3M_trimmed-out",
        # "a6Ke1YThz4o_trimmed-out",
        # "A6Pz9V6LzcU_trimmed-out",
        # "a80o4DGxt7Q_trimmed-out",
        # "aai7dDBNXBs_trimmed-out",
        # "abOuBvUfQk4_trimmed-out",
        # "AciwXaRfh3k_trimmed-out",
        # "ACPPfJtYCVc_trimmed-out",
        # "aDJJBMXiwiI_trimmed-out",
        # "afXewnGZXKs_trimmed-out",
        # "ahcAFnY6iAY_trimmed-out",
        # "AHiA9hohKr8_trimmed-out",
        # "AHXwnFvqYDk_trimmed-out",
        # "AiIrjf-s128_trimmed-out",
        # "ajFmgmUSYAc_trimmed-out",
        # "ajVTImleJlk_trimmed-out",
        # "AKAtC7easns_trimmed-out",
        # "ALbnaCezgdM_trimmed-out",
        # "alg7qHta0Sk_trimmed-out",
        # "Am6NHDbj6XA_trimmed-out",
        # "aNOuoSVlunM_trimmed-out",
        # "ap9vRY_Vdwc_trimmed-out",
        # "ApExci9PnNM_trimmed-out",
        # "APshm-9gPgI_trimmed-out",
        # "AQ7wbfX_av0_trimmed-out",
        # "aqGNOsZFdBU_trimmed-out",
        # "AQX2Q-V2Uh8_trimmed-out",
        # "aRQLU3IwNYs_trimmed-out",
        # "aS01LwpC23g_trimmed-out",
        # "ASqnnZpsX1M_trimmed-out",
        # "aSZ_eLxuLAs_trimmed-out",
        # "Ate-1815RNA_trimmed-out",
        # "atEkAkPfpUY_trimmed-out",
        # "_AuZO31q62g_trimmed-out",
        # "aw-fKJhcQE4_trimmed-out",
        # "awpHn196aVs_trimmed-out",
        # "aXiMaioTUkg_trimmed-out",
        # "AZCs9VoHeBo_trimmed-out",
        # "b0yONlMjxjs_trimmed-out",
        # "b1OedrPQ464_trimmed-out",
        # "B1VB7vVQNQg_trimmed-out",
        # "B2V9PFGQBH4_trimmed-out",
        # "b3I1tK1Iyzc_trimmed-out",
        # "B5ltukfhtw8_trimmed-out",
        # "B6p6X1LSjiA_trimmed-out",
        # "B6PpxrnttDg_trimmed-out",
        # "B7Nbbxh3m1Q_trimmed-out",
        # "B7XIUxyTi_8_trimmed-out",
        # "b9aeM__20E8_trimmed-out",
        # "badtXoOJaf8_trimmed-out",
        # "bb08nFwfoxA_trimmed-out",
        # "bBRWF0wju-c_trimmed-out",
        # "BC0dD13bwEw_trimmed-out",
        # "bC9hc4cqHGY_trimmed-out",
        # "bCKOVlsSluU_trimmed-out",
        # "bCWEOlvi5fY_trimmed-out",
        # "BDEUrfqlwcg_trimmed-out",
        # "Bd_vAawM9LA_trimmed-out",
        # "BEOdicifuqM_trimmed-out",
        # "b-FX9NOVQOM_trimmed-out",
        # "bgczomH1kLk_trimmed-out",
        # "B-gHVjv4_c4_trimmed-out",
        # "Bg_tJvCA8zw_trimmed-out",
        # "BH8FUBW4IIE_trimmed-out",
        # "BiV9eJU8Gsw_trimmed-out",
        # "bJ-G8xiLB6o_trimmed-out",
        # "Bks4JX95dD8_trimmed-out",
        # "bLVm1vfXRw8_trimmed-out",
        # "bMuoPr5-Yt4_trimmed-out",
        # "br0mu7r-ak0_trimmed-out",
        # "-bSM6iswghE_trimmed-out",
        # "bT_DEZz99VQ_trimmed-out",
        # "BUumpYIgVg4_trimmed-out",
        # "bwzH7ceQX8Y_trimmed-out",
        # "C08WmKiwcSs_trimmed-out",
        # "C0g5RjQ7cRE_trimmed-out",
        # "C2PneBztZ3g_trimmed-out",
        # "c2pwnHLaYTQ_trimmed-out",
        # "c67D5bP0Hg4_trimmed-out",
        # "C6RMS4F6LDc_trimmed-out",
        # "caOaW604Tqc_trimmed-out",
        # "CbMVjQV9b40_trimmed-out",
        # "CggDN9EIuNY_trimmed-out",
        # "cGTFuTIgc88_trimmed-out",
        # "cGU1Pepn1hU_trimmed-out",
        # "cHpER0dG1o8_trimmed-out",
        # "CNHBsxOZd80_trimmed-out",
        # "Cn_Mlwouwng_trimmed-out",
        # "cOlibbx5sx0_trimmed-out",
        # "CoMz3JOnZFo_trimmed-out",
        # "COYJC6dvB8I_trimmed-out",
        # "cq1er8IWz1U_trimmed-out",
        # "cQREa5Y-jqk_trimmed-out",
        # "Csy2RxzkbaM_trimmed-out",
        # "ctHj7R35dL0_trimmed-out",
        # "cuR-l2qCxBc_trimmed-out",
        # "Cv4Xj4fIkRo_trimmed-out",
        # "CwanEycyH_8_trimmed-out",
        # "cwoR3fkcJ9g_trimmed-out",
        # "CXmRmrBPDII_trimmed-out",
        # "cXTjL-f-msU_trimmed-out",
        # "CY2D1L1JtKU_trimmed-out",
        # "D0a2KWuL4S0_trimmed-out",
        # "D1Cil5n_-zs_trimmed-out",
        # "D1FXpqUivtU_trimmed-out",
        # "D2g3gTRkv0U_trimmed-out",
        # "D2VcClclMbs_trimmed-out",
        # "d43n4njmxcE_trimmed-out",
        # "D56yCIgqqgk_trimmed-out",
        # "d89i7OY2yTw_trimmed-out",
        # "dACF-Mz-X8M_trimmed-out",
        # "-daGjyKKNio_trimmed-out",
        # "DB7de4nC2rc_trimmed-out",
        # "DClIawJYpHs_trimmed-out",
        # "Ddbyb8zVKG0_trimmed-out",
        # "DE5S7W8ZfnI_trimmed-out",
        # "deKPBy_uLkg_trimmed-out",
        # "DelU5tQ4grw_trimmed-out",
        # "dI5D3aTgjZk_trimmed-out",
        # "DiaDblUd-lw_trimmed-out",
        # "DK8s_btC8F8_trimmed-out",
        # "dKxXtOyMmYc_trimmed-out",
        # "dONZkRDs4k4_trimmed-out",
        # "DpTB4TDKIa0_trimmed-out",
        # "-DTqvzmUw74_trimmed-out",
        # "dU7L1hvMx9Y_trimmed-out",
        # "DuXGDE6tolY_trimmed-out",
        # "dvisqlHIKpM_trimmed-out",
        # "dVJAvMbb8H4_trimmed-out",
        # "DW2umNQrQU0_trimmed-out",
        # "DWmUHNpOJxI_trimmed-out",
        # "DXyaQVlRVkY_trimmed-out",
        # "dZPwXsbohK4_trimmed-out",
        # "DZsBei4nCkU_trimmed-out",
        # "E0TBOKN8J2E_trimmed-out",
        # "E2IdU5lgaH4_trimmed-out",
        # "E4MUXs4IHtY_trimmed-out",
        # "e4mvg9r6_cI_trimmed-out",
        # "e6ppqFNBkLo_trimmed-out",
        # "e6zn4UlO0fU_trimmed-out",
        # "e8v9i_ksUyY_trimmed-out",
        # "EaPaLCuXjT8_trimmed-out",
        # "EC77tcJZIdU_trimmed-out",
        # "ecALuiFDRT0_trimmed-out",
        # "eDqEcrIRxgQ_trimmed-out",
        # "EeClqsYITso_trimmed-out",
        # "EEDZjwA1wM8_trimmed-out",
        # "Eg307HcpbJE_trimmed-out",
        # "EGK2P1cOJJc_trimmed-out",
        # "egw67gXKK3A_trimmed-out",
        # "EJdboFptQ3o_trimmed-out",
        # "eKQKEi2-0Ws_trimmed-out",
        # "ElghrCC2Rbs_trimmed-out",
        # "epy3Dy2FUOI_trimmed-out",
        # "EqXKrS3gPN4_trimmed-out",
        # "erOpqmubBL4_trimmed-out",
        # "eS8SpCRASr0_trimmed-out",
        # "eS9U1QO0F7M_trimmed-out",
        # "eTnuG394AcY_trimmed-out",
        # "eTph1-CG280_trimmed-out",
        # "EUIIWsgDpZY_trimmed-out",
        # "EwAb8ZW5Eiw_trimmed-out",
        # "EWUfDU8TWn4_trimmed-out",
        # "F0wIBTfLnE8_trimmed-out",
        # "F2mIH0vlI9c_trimmed-out",
        # "F2Xul-ihUVc_trimmed-out",
        # "F2YbeTjcpfs_trimmed-out",
        # "f3Ch2aIlXWo_trimmed-out",
        # "F4rSKCXqEw0_trimmed-out",
        # "FAaWqJLCCd0_trimmed-out",
        # "FaLaPEnjeqY_trimmed-out",
        # "f-BbAnnQVtY_trimmed-out",
        # "fce2RtEvPr8_trimmed-out",
        # "fC_Z5HlK9Pw_trimmed-out",
        # "fDe50VbOU44_trimmed-out",
        # "FgnO3muvvoM_trimmed-out",
        # "fhqu7ve9MA4_trimmed-out",
        # "fHW461eiQp8_trimmed-out",
        # "FiLWlqVE9Fg_trimmed-out",
        # "FJF56lmDqQo_trimmed-out",
        # "FkLblGfWAvY_trimmed-out",
        # "fL3_AauvjJ4_trimmed-out",
        # "fmuEMg2fh_8_trimmed-out",
        # "FositxHjuUk_trimmed-out",
        # "fpxPstb2DAU_trimmed-out",
        # "fsBzpr4k3rY_trimmed-out",
        # "fV1o_g6uzuI_trimmed-out",
        # "FWBCTZiijEM_trimmed-out",
        # "Fy3Mi8rOB3U_trimmed-out",
        # "Fy6BOTB4sXw_trimmed-out",
        # "FybhAns3or8_trimmed-out",
        # "fz0q7YKjp48_trimmed-out",
        # "fZuk-TaECZo_trimmed-out",
        # "G1wsCworwWk_trimmed-out",
        # "G22mJGndp14_trimmed-out",
        # "G3xzem7HSME_trimmed-out",
        # "G4heS2754l4_trimmed-out",
        # "G4ROcoq32rQ_trimmed-out",
        # "g67e0hDT1oQ_trimmed-out",
        # "G7bkiEh7_AM_trimmed-out",
        # "g7OMgsD7T74_trimmed-out",
        # "g8cyMIFcC_g_trimmed-out",
        # "g8D-LyfTrRs_trimmed-out",
        # "gAPPzmRb4r0_trimmed-out",
        # "gBs-CkxGXy8_trimmed-out",
        # "gbVOyKifrAo_trimmed-out",
        # "GbYGoWvJpwI_trimmed-out",
        # "gcDnKQul_c8_trimmed-out",
        # "GcImUUGmZ3I_trimmed-out",
        # "GCZ5aagOddY_trimmed-out",
        # "gDUFvLWl-Oc_trimmed-out",
        # "gDVmHsYgJUA_trimmed-out",
        # "geaSpx-R4Kc_trimmed-out",
        # "GeZIBgX7vkg_trimmed-out",
        # "gf5mfpSDFKM_trimmed-out",
        # "gfA1xa-BMCg_trimmed-out",
        # "GGEXxniRfWQ_trimmed-out",
        # "ggLOXOiq7WE_trimmed-out",
        # "GHBKZZuA314_trimmed-out",
        # "GI8LoYEYKI0_trimmed-out",
        # "gImYPbTTZko_trimmed-out",
        # "gjX78p5tvfo_trimmed-out",
        # "GK4_G33fXFU_trimmed-out",
        # "gKuBUQVcDJM_trimmed-out",
        # "GMidefrr1MM_trimmed-out",
        # "grTg3dzQDZI_trimmed-out",
        # "GxYimeaoea0_trimmed-out",
        # "GzPIbX1pzDg_trimmed-out",
        # "H0Qdz8bSkv0_trimmed-out",
        # "h35dZhHkuFM_trimmed-out",
        # "h7YTPuEMgaE_trimmed-out",
        # "h9hAaQOanZY_trimmed-out",
        # "Ham3IQQzoU8_trimmed-out",
        # "hBdsfj0YPO8_trimmed-out",
        # "HCgv_HNoJrY_trimmed-out",
        # "hcu4zY2HUQY_trimmed-out",
        # "hd8bXHCvZME_trimmed-out",
        # "HDhwReMUBsA_trimmed-out",
        # "HEke6Dlhqtw_trimmed-out",
        # "hezHSFSwa08_trimmed-out",
        # "HFPGeaEPy9o_trimmed-out",
        # "h-H1LddWxo8_trimmed-out",
        # "hjGsGtihBpc_trimmed-out",
        # "hkAFdIrTR00_trimmed-out",
        # "hKfNp8NU82o_trimmed-out",
        # "hL2u93brqiA_trimmed-out",
        # "HMkOO15nye8_trimmed-out",
        # "HN75tPziZAo_trimmed-out",
        # "-hnBHBN8p5A_trimmed-out",
        # "hnfkZx4jmpA_trimmed-out",
        # "HPszYa77CkM_trimmed-out",
        # "H-q0zh-XGVc_trimmed-out",
        # "hqzF4IDaIYE_trimmed-out",
        # "hRcSU9-krNU_trimmed-out",
        # "hrhX40bQYY0_trimmed-out",
        # "hRkl5WhbQLc_trimmed-out",
        # "HSEi0RXGVq8_trimmed-out",
        # "Hv0PyGfxbpw_trimmed-out",
        # "HwIRCeUCyxw_trimmed-out",
        # "HzH0cBmHg5k_trimmed-out",
        # "I0izJOlMJiM_trimmed-out",
        # "i3itBKdwE7M_trimmed-out",
        # "I4mItsGR3uI_trimmed-out",
        # "i5YckMkwmm4_trimmed-out",
        # "I7GvG0WYfOo_trimmed-out",
        # "i7JGn06gsnA_trimmed-out",
        # "IbkyL0pDtEc_trimmed-out",
        # "iBL0FcUTFT8_trimmed-out",
        # "ICBNX0i855Q_trimmed-out",
        # "_Ice5RkbWUY_trimmed-out",
        # "iDgaqD7CWXU_trimmed-out",
        # "IE6r8Pk91T0_trimmed-out",
        # "iEENyD0JiRE_trimmed-out",
        # "iFgOoTRflnw_trimmed-out",
        # "ifxLhziWjm0_trimmed-out",
        # "igH3ixpts2g_trimmed-out",
        # "ihP926ccYDw_trimmed-out",
        # "IHU9Jc_NUuk_trimmed-out",
        # "iiDdvLrSUG4_trimmed-out",
        # "Ilf38Achvzk_trimmed-out",
        # "IlLFTI24Qkw_trimmed-out",
        # "ilotZqzaZgU_trimmed-out",
        # "imUigBNF-TE_trimmed-out",
        # "im_uLJKzs-4_trimmed-out",
        # "iNr9xdc6cVA_trimmed-out",
        # "iOdEJMNEzyI_trimmed-out",
        # "iovKlisBCzU_trimmed-out",
        # "ipnGPeRIy2k_trimmed-out",
        # "IsEykad2V9U_trimmed-out",
        # "IsgFVkMnqJc_trimmed-out",
        # "Iu6_k2ok00U_trimmed-out",
        # "iwCdv9iR8P8_trimmed-out",
        # "IXbAYg5pp9M_trimmed-out",
        # "ixQbCXLUUj8_trimmed-out",
        # "izCiPuiGe9E_trimmed-out",
        # "j1CTHVQ8Z3k_trimmed-out",
        # "j3pLDghHKyc_trimmed-out",
        # "j5SKmUoL9Tg_trimmed-out",
        # "j7cUdCFEpeU_trimmed-out",
        # "J94uO-urSTg_trimmed-out",
        # "jbHKeVsI35M_trimmed-out",
        # "-jDIwv4wCsU_trimmed-out",
        # "jdrVAQrzt9Y_trimmed-out",
        # "jE1tmy-g6tw_trimmed-out",
        # "jfKDozXX_Uo_trimmed-out",
        # "jh5PklItWjA_trimmed-out",
        # "jiMUoVjQ5uI_trimmed-out",
        # "jK08J1811uA_trimmed-out",
        # "jKguXsNkJ4w_trimmed-out",
        # "JMzilFvwNXE_trimmed-out",
        # "jMZLrPxp31E_trimmed-out",
        # "JNGcJEZ6rwA_trimmed-out",
        # "jP2HuCwfKFA_trimmed-out",
        # "Jp_KHLvQcuw_trimmed-out",
        # "jrv2BW_c8jA_trimmed-out",
        # "JRYjFh_hHBs_trimmed-out",
        # "jtl5XK7QP38_trimmed-out",
        # "jvpj4qdPRE0_trimmed-out",
        # "JW2HHfQiGVs_trimmed-out",
        # "JW3OfSCZlhc_trimmed-out",
        # "JxbV5wGpXc8_trimmed-out",
        # "JXYyQamYw84_trimmed-out",
        # "jYL4gMsGZgE_trimmed-out",
        # "jYSuKn09_e4_trimmed-out",
        # "K25zmgYg5s4_trimmed-out",
        # "k5bk1efKBSI_trimmed-out",
        # "K6v2QYiMdCE_trimmed-out",
        # "KaIzZrMb2og_trimmed-out",
        # "KBj3TocgvOA_trimmed-out",
        # "KBMAUGQpHBU_trimmed-out",
        # "KB_p2QvvLGw_trimmed-out",
        # "K-bZQJ3P9N0_trimmed-out",
        # "KCbXRRvnFj8_trimmed-out",
        # "_KE_-EdMhDI_trimmed-out",
        # "kefqa4xre2I_trimmed-out",
        # "kf9RtsXQPWU_trimmed-out",
        # "kGoON1J872w_trimmed-out",
        # "KHgGasfOFPg_trimmed-out",
        # "kIfsmKj42XE_trimmed-out",
        # "Kjlt_FgKPgA_trimmed-out",
        # "k-kyKv_kKGU_trimmed-out",
        # "kmgsC68hIL8_trimmed-out",
        # "kmi_liqBsdU_trimmed-out",
        # "KoH4OKx4x1Y_trimmed-out",
        # "kOIWYXDRR7s_trimmed-out",
        # "KpljzfIVBDM_trimmed-out",
        # "kpMLnFxvi6Y_trimmed-out",
        # "kRh1zXFKC_o_trimmed-out",
        # "ks66e-O4YCQ_trimmed-out",
        # "ktdgC1dJkOA_trimmed-out",
        # "kU6YY2z7z0I_trimmed-out",
        # "KvbeKlGeNRU_trimmed-out",
        # "kVh1Jw2D9NY_trimmed-out",
        # "KWnV1Aa6VQ8_trimmed-out",
        # "KWS6OcNbvLA_trimmed-out",
        # "KWSDwS4S6Ss_trimmed-out",
        # "kYnmc_AVfMs_trimmed-out",
        # "kzhJb5jcH58_trimmed-out",
        # "KzN6XWDEmXI_trimmed-out",
        # "kztkcj-WAvw_trimmed-out",
        # "l1jW3OMXUzs_trimmed-out",
        # "L3uDQ0S1Iis_trimmed-out",
        # "L49M8C4wjVU_trimmed-out",
        # "L75hdqt98nw_trimmed-out",
        # "L9U9hvUf42g_trimmed-out",
        # "lacVKwbsE7Q_trimmed-out",
        # "lB0mkAa1vfM_trimmed-out",
        # "LBd5x_fe4Jc_trimmed-out",
        # "lC8nM75AqUk_trimmed-out",
        # "LchHXKL6xZY_trimmed-out",
        # "LcHtLypALog_trimmed-out",
        # "LcMBahfo0NA_trimmed-out",
        # "lcmI9_aypI0_trimmed-out",
        # "LD1RAPiT_7A_trimmed-out",
        # "lickge5rPdc_trimmed-out",
        # "licUm-aEaCY_trimmed-out",
        # "LiqyOoGW-I8_trimmed-out",
        # "lJ83ILGA8yI_trimmed-out",
        # "lkeVfgI0eEk_trimmed-out",
        # "lKPRa_4hnlE_trimmed-out",
        # "LmCJIBsQjOY_trimmed-out",
        # "lmcuesVvLf0_trimmed-out",
        # "lo0R1mvjDT8_trimmed-out",
        # "LoMhBo8ATBM_trimmed-out",
        # "-Lp96hoSUC8_trimmed-out",
        # "lpAMi2lwjo0_trimmed-out",
        # "LTEiibVnRgI_trimmed-out",
        # "lTfsqoo-jKg_trimmed-out",
        # "lTk5zkc0i8M_trimmed-out",
        # "LTUojzYVUUI_trimmed-out",
        # "LtxWWUBt7ro_trimmed-out",
        # "luJceOt47UM_trimmed-out",
        # "_LUX70mXcEE_trimmed-out",
        # "lUyKpfbB9M8_trimmed-out",
        # "m075kb3aV04_trimmed-out",
        # "M4blAdS6r3Q_trimmed-out",
        # "mA402F5K47o_trimmed-out",
        # "MBDZbACupsc_trimmed-out",
        # "MCAH-zHLTLE_trimmed-out",
        # "Md4QnipNYqM_trimmed-out",
        # "MdG9Lkk8VWo_trimmed-out",
        # "MDQTH8WkAvQ_trimmed-out",
        # "Me-A3eOhgXs_trimmed-out",
        # "Mek_AQ5DbUs_trimmed-out",
        # "Mf76yyTY7Ss_trimmed-out",
        # "MHVrwCEWLPI_trimmed-out",
        # "MIFz86h5nEA_trimmed-out",
        # "mJwWxbSR6r0_trimmed-out",
        # "MlteErDn4to_trimmed-out",
        # "MM0YOB-cSWA_trimmed-out",
        # "mN1Z5xEOy10_trimmed-out",
        # "mNcdlLIOdNw_trimmed-out",
        # "mO8TK-QaIf8_trimmed-out",
        # "mPAESQEQoms_trimmed-out",
        # "mpHoYhIFKNI_trimmed-out",
        # "MqiOBIxouw4_trimmed-out",
        # "mr88Ud5V9uE_trimmed-out",
        # "MsjnLoTKAXo_trimmed-out",
        # "muuW2EU8nrA_trimmed-out",
        # "mVnqP-vLpuo_trimmed-out",
        # "mxa4KXSz9rw_trimmed-out",
        # "N188QSyfmeQ_trimmed-out",
        # "N1fVL4AQEW8_trimmed-out",
        # "N3QuD26DuCI_trimmed-out",
        # "N4fvzhsdTio_trimmed-out",
        # "n5_HdNzf03Q_trimmed-out",
        # "N5sGxcAJFCA_trimmed-out",
        # "n5V7TVYiVas_trimmed-out",
        # "n6-ef_YHeJU_trimmed-out",
        # "N-6zVmVuTs0_trimmed-out",
        # "n71IBjXHnrQ_trimmed-out",
        # "N7K5DQvMWXM_trimmed-out",
        # "n8Cn_KhcR7o_trimmed-out",
        # "nAkqWGO1T-c_trimmed-out",
        # "nblf7Yw4jys_trimmed-out",
        # "ncHMwblapdI_trimmed-out",
        # "NcHQjL_WlxQ_trimmed-out",
        # "Nck6BZga7TQ_trimmed-out",
        # "nCQB2AaVgOY_trimmed-out",
        # "ndsKiHu63oM_trimmed-out",
        # "nEpJaJyeRy8_trimmed-out",
        # "NFKdaj1Qsek_trimmed-out",
        # "nGah7qST1dI_trimmed-out",
        # "ngHnomR-114_trimmed-out",
        # "NgPP6UXVkYU_trimmed-out",
        # "nLu13aVrNcQ_trimmed-out",
        # "n_mTiDeQvWg_trimmed-out",
        # "NmWjnYUkT_s_trimmed-out",
        # "nmWplIQhvoA_trimmed-out",
        # "NnhpG0nEC84_trimmed-out",
        # "-nO3dBeh5n0_trimmed-out",
        # "No6mB6V1wL4_trimmed-out",
        # "noHN3H3gWPQ_trimmed-out",
        # "Nqf15ViHSmQ_trimmed-out",
        # "nQqY3j8btbI_trimmed-out",
        # "nqY3tv-y62A_trimmed-out",
        # "NR9v3PBJw8g_trimmed-out",
        # "Nra5F1tQQww_trimmed-out",
        # "nRfU_c4QlaQ_trimmed-out",
        # "NTpj7quCIqQ_trimmed-out",
        # "NTySVAtrdQc_trimmed-out",
        # "Nx5VK6DUUEY_trimmed-out",
        # "nyqh7vhU3X0_trimmed-out",
        # "NysITFb_Wbs_trimmed-out",
        # "nZiIVvEJ8m0_trimmed-out",
        # "nzj7Wg4DAbs_trimmed-out",
        # "NZtIGzAzJZM_trimmed-out",
        # "O24d_rJ4uDo_trimmed-out",
        # "o4CKGkaTn-A_trimmed-out",
        # "O5rTU5EA1C8_trimmed-out",
        # "o6DEc3OYQfU_trimmed-out",
        # "o6hMDs4rBmw_trimmed-out",
        # "o7Ax6SRTGks_trimmed-out",
        # "O8iOYngEUBU_trimmed-out",
        # "o92pxWhZomM_trimmed-out",
        # "O951zoCuU7Q_trimmed-out",
        # "Ob3hr2H3VD4_trimmed-out",
        # "Obde542xA9I_trimmed-out",
        # "ocg0MbBfCS0_trimmed-out",
        # "OFia3dWgaoI_trimmed-out",
        # "ofvOXABptcY_trimmed-out",
        # "ogIbkhMeJFI_trimmed-out",
        # "OLw7cIJApMI_trimmed-out",
        # "OMkfujDPpwc_trimmed-out",
        # "ON0rtQyNS20_trimmed-out",
        # "ON45DvDMSk4_trimmed-out",
        # "onbBEbC24SQ_trimmed-out",
        # "ooUw7Hq8YLg_trimmed-out",
        # "Op2e8_JURSQ_trimmed-out",
        "OPdbdjctx2I_trimmed-out",
        "_oqc_t0mbsQ_trimmed-out",
        "OqL_u-uGIi0_trimmed-out",
        "oRBPxefAHTY_trimmed-out",
        "ory7fcHYXhQ_trimmed-out",
        "OsQdzsIMFPQ_trimmed-out",
        "Ot1aZoxt9PU_trimmed-out",
        "OT3MVRv0TT4_trimmed-out",
        "OTdFPlXfFj4_trimmed-out",
        "otna1VHHCow_trimmed-out",
        "oW3UPfXHSUs_trimmed-out",
        "OWsT2rkqCk8_trimmed-out",
        "O_xHNbx5jqI_trimmed-out",
        "oxZYBldwpP4_trimmed-out",
        "oY6BfHkNFPY_trimmed-out",
        "oYM9bFGb8B8_trimmed-out",
        "P2rLv-vjSOs_trimmed-out",
        "P3QEOJWxXI8_trimmed-out",
        "P3UP62l7g6Q_trimmed-out",
        "P5ZOwNK6n9U_trimmed-out",
        "p6CRBNmX_gY_trimmed-out",
        "p6IwwFevo7o_trimmed-out",
        "P8YRuxu8dQU_trimmed-out",
        "p9GrpGa8Uys_trimmed-out",
        "pawEXwLjTlo_trimmed-out",
        "pc3__b1uF4Q_trimmed-out",
        "PCVvpS7w-2w_trimmed-out",
        "PFA-RmV_wG0_trimmed-out",
        "pFUXcA1fp6g_trimmed-out",
        "PfwwCpAy0-0_trimmed-out",
        "pG8zAQGgL1o_trimmed-out",
        "PIqmZBVVH0c_trimmed-out",
        "PiYhxJzWLzo_trimmed-out",
        "PjBxYw4SJVg_trimmed-out",
        "pK1WAx4jJzE_trimmed-out",
        "pMGhqE76kQA_trimmed-out",
        "PnqJALSs6so_trimmed-out",
        "PQW7zw2URes_trimmed-out",
        "PR04klFMPYQ_trimmed-out",
        "PR0y7IlMEOs_trimmed-out",
        "PR2fDvbCoAU_trimmed-out",
        "PsOnuG1EBjQ_trimmed-out",
        "PVD-SzVd_UQ_trimmed-out",
        "pXvVnvQOMQ0_trimmed-out",
        "pZCz9-b5Grs_trimmed-out",
        "q0r-dKImVLk_trimmed-out",
        "Q25kn317a2M_trimmed-out",
        "Q3PJuGz0Njs_trimmed-out",
        "q45sJ2n2XPg_trimmed-out",
        "q53HUAKB9oU_trimmed-out",
        "Q5YJ7mh8zWc_trimmed-out",
        "qAs7UWhTyaE_trimmed-out",
        "QC_4iR0tyvE_trimmed-out",
        "QCR7uyowjhM_trimmed-out",
        "qDpGgd4oTQ8_trimmed-out",
        "qe--y0apjcU_trimmed-out",
        "Qf2fBuiddNI_trimmed-out",
        "QFTvmsHgFYM_trimmed-out",
        "qh8JWHlIgcE_trimmed-out",
        "QHeNKMxsqYE_trimmed-out",
        "QHGeDg6XX6U_trimmed-out",
        "qhlmvp3VDzI_trimmed-out",
        "qN-ZCqaSHHk_trimmed-out",
        "QoIsjc4-GIg_trimmed-out",
        "qow_n-oNEt8_trimmed-out",
        "-QQ0Lv4_--4_trimmed-out",
        "QQKG6cd-sBI_trimmed-out",
        "QQpIBRLlJzo_trimmed-out",
        "qQPl5ySv3Fk_trimmed-out",
        "QqWrqHC8wjs_trimmed-out",
        "QR43TImr0dQ_trimmed-out",
        "qr69jLeQdRA_trimmed-out",
        "qRM1D4jE09w_trimmed-out",
        "QRRzvOvG21E_trimmed-out",
        "QtANwxed36M_trimmed-out",
        "QUd09jZIQEk_trimmed-out",
        "qXG69NfiePE_trimmed-out",
        "QytntXWJIe8_trimmed-out",
        "qzCjToACNrU_trimmed-out",
        "R4QbJL80Ri4_trimmed-out",
        "r7rpWLLPQSA_trimmed-out",
        "RAUSVB_Nu9k_trimmed-out",
        "rB6WmWvvyxg_trimmed-out",
        "RFcuAhpQdXU_trimmed-out",
        "rFJIgAmT8dE_trimmed-out",
        "RgGLnaS7smE_trimmed-out",
        "RHRPK2O9R4w_trimmed-out",
        "RJnquSNVuuw_trimmed-out",
        "rMRWJEvKopk_trimmed-out",
        "rmZGPAXE0oA_trimmed-out",
        "rnbtRiLamsw_trimmed-out",
        "rnj_bAv7Hyo_trimmed-out",
        "-RpZEe4w4fY_trimmed-out",
        "rq-7zoXQ69Y_trimmed-out",
        "RS5x5GW5bEE_trimmed-out",
        "RsqRaVYQ6Cc_trimmed-out",
        "RtjJBtFXJaI_trimmed-out",
        "rVWZuXDkc4A_trimmed-out",
        "R-xtt7w9Do4_trimmed-out",
        "rZLwPui5TQk_trimmed-out",
        "RZPBl5-cu3c_trimmed-out",
        "RZzaa_PqNJI_trimmed-out",
        "S0xTq15pzJU_trimmed-out",
        "s3Czwcz3E-o_trimmed-out",
        "_s42gOg2WSU_trimmed-out",
        "s5ak_9z3Cp8_trimmed-out",
        "s5jrdHASx04_trimmed-out",
        "S63inYNCsCM_trimmed-out",
        "S8vC5FNRDTU_trimmed-out",
        "SADub7W22Zg_trimmed-out",
        "SAgYiERRDPY_trimmed-out",
        "SBp5KuQC_lA_trimmed-out",
        "sGAj25JPwcg_trimmed-out",
        "sheEL099ADM_trimmed-out",
        "shL1gWm9qdg_trimmed-out",
        "Sir2QeCh4B0_trimmed-out",
        "sJR9QjezRGg_trimmed-out",
        "SjrMprYa608_trimmed-out",
        "SJSmbF9W9PQ_trimmed-out",
        "sjzVFt59eds_trimmed-out",
        "-sp3Q524oO4_trimmed-out",
        "spdhjKK6zCo_trimmed-out",
        "sPxoGNvnVzg_trimmed-out",
        "sQ3pkACdX84_trimmed-out",
        "SQ8aRKG9660_trimmed-out",
        "sqDIGuzt38w_trimmed-out",
        "sqIR2N5izHo_trimmed-out",
        "SrMN3rwvCsE_trimmed-out",
        "srWtQnseRyE_trimmed-out",
        "SstGaYWCNoQ_trimmed-out",
        "SsuLW2JpjEA_trimmed-out",
        "STbJ8hJFMAg_trimmed-out",
        "sty54JE9DHw_trimmed-out",
        "SWNXZbasPvQ_trimmed-out",
        "SxtSW5HFkvg_trimmed-out",
        "s_y9h4hK2PM_trimmed-out",
        "sZ9gSPUdZIA_trimmed-out",
        "SzCgZK1KMio_trimmed-out",
        "T0jM37coZPo_trimmed-out",
        "T1icdxz0TNs_trimmed-out",
        "T3cwJlY8OYk_trimmed-out",
        "_T3uHSwRgdU_trimmed-out",
        "t4-ZulW5np4_trimmed-out",
        "T6x-kDiQsWM_trimmed-out",
        "t7qDtNdJAlk_trimmed-out",
        "T86T9wMn77s_trimmed-out",
        "T8JwNZBJ_wI_trimmed-out",
        "TA3Xy53hMMY_trimmed-out",
        "tafBLziHJ0g_trimmed-out",
        "tbwPN9fZb2Q_trimmed-out",
        "_tBXwqueWcg_trimmed-out",
        "TCCYCSACA2Y_trimmed-out",
        "t-dmFuH7TyM_trimmed-out",
        "TDxlMelzl10_trimmed-out",
        "teg6qTE9Hjs_trimmed-out",
        "TewPhK6CZ-Q_trimmed-out",
        "tfVvLs379Oo_trimmed-out",
        "Tfw2mq2wJls_trimmed-out",
        "ThpV4osXuCk_trimmed-out",
        "tjogri9eYzs_trimmed-out",
        "tmkqp3VpDCE_trimmed-out",
        "tNami65DCyE_trimmed-out",
        "TpH0DD3MCUQ_trimmed-out",
        "tth90qiKXgY_trimmed-out",
        "TTY4WKXoPac_trimmed-out",
        "tvcrwTcA5iw_trimmed-out",
        "tv-hgegVq0k_trimmed-out",
        "tVvGairu3Hs_trimmed-out",
        "tXG-qPZJj-8_trimmed-out",
        "TyFWoohX0bo_trimmed-out",
        "_tYPnaK5gkI_trimmed-out",
        "TyR3GMJJVhA_trimmed-out",
        "U4r9ePA6RkQ_trimmed-out",
        "uANIooMR9a0_trimmed-out",
        "uB3FdWmnFZU_trimmed-out",
        "UCW_UH-k-ec_trimmed-out",
        "ucYhn366lts_trimmed-out",
        "uH5JkfBKD9M_trimmed-out",
        "uivjORoSW0k_trimmed-out",
        "ujj3zYdBd0k_trimmed-out",
        "UJLb8PfQYl8_trimmed-out",
        "_UJNNySGM6Q_trimmed-out",
        "U_JrsRlieNQ_trimmed-out",
        "uJUP839M2WM_trimmed-out",
        "uk0Ntd8lJ98_trimmed-out",
        "ukn4Yw4mB1Y_trimmed-out",
        "ullXTZ0jjZw_trimmed-out",
        "uNU6H62jNxA_trimmed-out",
        "u_pE5O64Q9I_trimmed-out",
        "urYGhhMOToU_trimmed-out",
        "US7hym404bQ_trimmed-out",
        "utkUkvYq-zM_trimmed-out",
        "UUukBV82P9A_trimmed-out",
        "UUuXfyfCaL4_trimmed-out",
        "UwPd_D3y3Pk_trimmed-out",
        "uXlJ3ezSZpg_trimmed-out",
        "U-Y_HkD_0O4_trimmed-out",
        "UzzPYlIeSvo_trimmed-out",
        "Va54WZgPTdY_trimmed-out",
        "vaHpVjqCnNE_trimmed-out",
        "Vbp7qtUXQFA_trimmed-out",
        "vC63_bemI2I_trimmed-out",
        "vE0R-Gw_GEA_trimmed-out",
        "VeV7vvfipA0_trimmed-out",
        "VIVkYG31Oas_trimmed-out",
        "vjZDdaFTubE_trimmed-out",
        "vKkpvQlHEG8_trimmed-out",
        "Vlh0sMMjU04_trimmed-out",
        "vlnKSMw5v1o_trimmed-out",
        "vLS1o1F6KqE_trimmed-out",
        "VNM7Z7hir_I_trimmed-out",
        "-VNNzqqnaew_trimmed-out",
        "VOHCh8U9XQY_trimmed-out",
        "VopLr9b_IgM_trimmed-out",
        "VP4rHzYyuL0_trimmed-out",
        "VPoNcAeOycw_trimmed-out",
        "VPTjROKYhlU_trimmed-out",
        "vqFmKLl2hq4_trimmed-out",
        "V-qtKSEGAac_trimmed-out",
        "vsNVU7y6dUE_trimmed-out",
        "vSQvk9P1dus_trimmed-out",
        "VsSGubvfPiA_trimmed-out",
        "vTLkSpY_aYg_trimmed-out",
        "VTPz9z7M6B8_trimmed-out",
        "vVaZzurwxeI_trimmed-out",
        "Vwn_QS9vB1g_trimmed-out",
        "VX671ftXpMU_trimmed-out",
        "VxS7HHz0mRI_trimmed-out",
        "wA4i3eHKsTQ_trimmed-out",
        "waE2GdoBW68_trimmed-out",
        "wB_hjqZQ1UY_trimmed-out",
        "wbmW_Z1oPJw_trimmed-out",
        "WBOT0Tqpbag_trimmed-out",
        "WczPTuyEt5c_trimmed-out",
        "wDgVPV_t0AE_trimmed-out",
        "wf1LTMcVFEc_trimmed-out",
        "Wg7ppxAgcuw_trimmed-out",
        "W_GrHtwZez8_trimmed-out",
        "WhsZxUWSHaQ_trimmed-out",
        "WI9QnhX7vcI_trimmed-out",
        "WLBB9CqfMCk_trimmed-out",
        "Wo1RxRjXyYw_trimmed-out",
        "Wp0xufdpjqc_trimmed-out",
        "WQxM_EK5aiM_trimmed-out",
        "WR0hwJp9AOA_trimmed-out",
        "wRMF9hqdcTI_trimmed-out",
        "WSqbeVe4jXo_trimmed-out",
        "wu9D9hdwOQ8_trimmed-out",
        "W-V0sdbQA-M_trimmed-out",
        "WV1ncOsZjog_trimmed-out",
        "WVnNoiQKhPc_trimmed-out",
        "w_WAB7TXtXQ_trimmed-out",
        "WXuRTQPKwqY_trimmed-out",
        "wYLmgQy6zq8_trimmed-out",
        "wz3nVBPVgIA_trimmed-out",
        "wZoXjg1x4Yg_trimmed-out",
        "X0jMZoxUL2A_trimmed-out",
        "x6yr8W3mc38_trimmed-out",
        "x6YzIqb6Cas_trimmed-out",
        "X9BE2oUSOXM_trimmed-out",
        "x9lr8OomuJ4_trimmed-out",
        "xaQYLAM1LHE_trimmed-out",
        "xbhXd8lj2s0_trimmed-out",
        "Xbx0Dl90wO0_trimmed-out",
        "_xChA_vIuxE_trimmed-out",
        "xdiFzcpmmJc_trimmed-out",
        "XE4ifwZqvEs_trimmed-out",
        "XFGAQrEUaeU_trimmed-out",
        "XFP-qBKzDBY_trimmed-out",
        "XHBZkek8OSU_trimmed-out",
        "xITR-FIi_oA_trimmed-out",
        "XjAmvLVbE3E_trimmed-out",
        "XKyumlBmix8_trimmed-out",
        "xNxw4rbwh68_trimmed-out",
        "xpIFaKnNvco_trimmed-out",
        "xpXCQkb4LUI_trimmed-out",
        "xQwrPjLUwmo_trimmed-out",
        "Xs1sMqpc1oM_trimmed-out",
        "xs8O-MFu4yU_trimmed-out",
        "xWjiz7ZrHkA_trimmed-out",
        "xWLmbDdg9tY_trimmed-out",
        "xwQBg9pVxrQ_trimmed-out",
        "xWRNBOXoLf8_trimmed-out",
        "XwUcQU9BFDI_trimmed-out",
        "xXIq7YPkdUQ_trimmed-out",
        "XYBx__4iAUw_trimmed-out",
        "XYviM5xevC8_trimmed-out",
        "XZn3wQhEXWQ_trimmed-out",
        "y1Y02_oZP8U_trimmed-out",
        "Y6fVeuHTx0w_trimmed-out",
        "Y7eYfHIyn3M_trimmed-out",
        "YAYfMkxhRiQ_trimmed-out",
        "yDDRstyk9Dg_trimmed-out",
        "yDrkv0iVtsk_trimmed-out",
        "YgfaKTH9RbY_trimmed-out",
        "_yHLgEDXZbI_trimmed-out",
        "YI5l5ltMl8g_trimmed-out",
        "yIhH-_BXPr8_trimmed-out",
        "YIJ0mZYi1O4_trimmed-out",
        "YIZpD3fEfgs_trimmed-out",
        "yK1RQv3S8KE_trimmed-out",
        "yKGfDlxvR_Y_trimmed-out",
        "ykTHGJLCstU_trimmed-out",
        "YlVkBzXo0YM_trimmed-out",
        "yMC03qoBlE0_trimmed-out",
        "YNmgvI44N5Q_trimmed-out",
        "YnqZfVCUSiA_trimmed-out",
        "yq1cX1lAyQg_trimmed-out",
        "YRmjALBdTdE_trimmed-out",
        "yRtPan09Ek0_trimmed-out",
        "YsBSR_z9BR8_trimmed-out",
        "ytEgqN-BdKA_trimmed-out",
        "YTwEFUfX4zA_trimmed-out",
        "YuU1QV7N9yo_trimmed-out",
        "YVHJpAROBvQ_trimmed-out",
        "Yz4NWtsQm1w_trimmed-out",
        "Yzq6fNeXf6w_trimmed-out",
        "YZR6LEk3doM_trimmed-out",
        "_Z1tcdf6qkI_trimmed-out",
        "z6qavchy4Ho_trimmed-out",
        "Z7MknXjNJSg_trimmed-out",
        "Z86hdiW6l6c_trimmed-out",
        "ZeYarFkvoGw_trimmed-out",
        "zf2tZ0HIiPQ_trimmed-out",
        "Z_FMSWtuU6A_trimmed-out",
        "ZfZhi8nnnLI_trimmed-out",
        "zHqGiZQTxfw_trimmed-out",
        "ziwYbVx_-qg_trimmed-out",
        "ZKTIOHRjlHM_trimmed-out",
        "ZMJw752P7z8_trimmed-out",
        "zoE6RKgvBlI_trimmed-out",
        "ZoInHw_M_H4_trimmed-out",
        "-ZOjNsOHtOU_trimmed-out",
        "ZP8ACbJ677I_trimmed-out",
        "zqgqVF6lWB0_trimmed-out",
        "zT-1nFMmt9Q_trimmed-out",
        "ztlOyCk5pcg_trimmed-out",
        "ZuBzOV1ADkk_trimmed-out",
        "_zuODgCQ6O8_trimmed-out",
        "ZuYTtKZUkJc_trimmed-out",
        "ZVR5lhbmGw0_trimmed-out",
        "zVrWAxxLuA0_trimmed-out",
        "zWaV_PTlCxk_trimmed-out",
        "Zx7c8huezqY_trimmed-out",
        "zXbortaLKDE_trimmed-out",
        "zYeBxnAm2wI_trimmed-out",
        "ZyIQkgixS_o_trimmed-out",
        "ZYzql-Y1sP4_trimmed-out",
        "zzERcgFrTrA_trimmed-out",
        "zzWaf1z9kgk_trimmed-out"
    ]
    for seq in sequences:
        # try:
        ############################################################
        # 1. Construct face bank only using the confident faces
        # from openface
        ############################################################
        # if not osp.exists(osp.join(data_root, "results", "tracking_openface", args.detector, "npzs", "{}.npz".format(seq))):
        if True:
            face_bank, names, emb_counts = construct_face_bank(seq, retina_face_root, image_root, openface_root, learner, data_root)
            names, emb_counts = np.array(names), np.array(emb_counts)
        # else:
        #     data = np.load(osp.join(data_root, "results", "tracking_openface", args.detector, "npzs", "{}.npz".format(seq)))
        #     face_bank = torch.tensor(data['embeddings']).cuda()
        #     names = data['names']
        #     emb_counts = data['emb_counts']
        print("face_bank： {}".format(face_bank.shape))
        print("names： {}".format(names))
        print("emb_counts： {}".format(emb_counts))

        # Note: abandoned. This is too harsh
        # # Select the top N faces (N = 5) ?
        # emb_counts = np.array(emb_counts)
        # names = np.array(names)
        # topN = emb_counts.argsort()[-5:][::-1]
        # print(topN)
        # # topN = range(len(emb_counts))
        # face_bank = face_bank[list(topN), :]
        # names = np.array([names[0]] + list(names[1:][topN]))
        # emb_counts = np.array(emb_counts[topN])

        # print("Filtered face_bank： {}".format(face_bank.shape))
        # print("Filtered names： {}".format(names))
        # print("Filtered emb_counts： {}".format(emb_counts))
        ############################################################
        # 2. Use the constructed face bank to do face
        # re-identification throughout the video and match body
        # detected from openpose
        ############################################################
        results = match_faces(seq, retina_face_root, image_root, openpose_root, learner, face_bank, names, data_root, emb_counts)
        # except:
        #     continue
        # break


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='for face verification')
    parser.add_argument('-th', '--threshold', help='threshold to decide identical faces', default=1.00, type=float)
    parser.add_argument("-c", "--score", help="whether show the confidence score", action="store_true")
    parser.add_argument("-d", "--detector", help="what face detector to use", default="retinaface", type=str)

    args = parser.parse_args()

    conf = get_config(False)

    run(args, conf)
