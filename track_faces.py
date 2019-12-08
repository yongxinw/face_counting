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

    sequences = ["ZYzql-Y1sP4_trimmed-out"]
    # sequences = ["zzWaf1z9kgk_trimmed-out"]
    # sequences = ["zzWaf1z9kgk_trimmed-out"]
    # sequences = ["ztlOyCk5pcg_trimmed-out"]
    image_root = "/work/yongxinw/SocialIQ/raw/raw/vision/"

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
        image_paths = sorted(os.listdir(osp.join(image_root, "images", seq)))
        os.makedirs(osp.join(image_root, "results", "tracking", args.detector, seq), exist_ok=True)

        openpose_dir = osp.join("/home", "yongxinw", "data", "images", seq)
        # Initialize identities (i.e. face bank) using the first frame
        image0 = Image.open(osp.join(image_root, "images", seq, image_paths[0]))

        if args.detector == "mtcnn":
            embeddings, names, emb_counts, bboxes0 = initialize_face_bank_mtcnn(image0, mtcnn, learner, conf)
            names0 = []
            names0.extend(names[1:])
        elif args.detector == "retinaface":
            retinaface_result_path0 = osp.join(image_root, "results", "detection", "RetinaFace", seq, "results.txt")
            embeddings, names, emb_counts, bboxes0 = initialize_face_bank_retinaface(image0, retinaface_result_path0)
            names0 = []
            names0.extend(names[1:])
        else:
            raise NotImplementedError

        for i, path in enumerate(image_paths[1:]):

            frame = cv2.imread(osp.join(image_root, "images", seq, path))
            # im = Image.fromarray(frame)
            im = Image.open(osp.join(image_root, "images", seq, path))
            print(i)
            if args.detector == "mtcnn":
                try:
                    bboxes, faces, landmarks = mtcnn.align_multi_with_landmarks(im, conf.face_limit, 16)
                except:
                    bboxes = []
                    faces = []
            elif args.detector == "retinaface":
                retinaface_result_path = osp.join(image_root, "results", "detection", "RetinaFace", seq, "results.txt")
                bboxes, faces, landmarks = find_retinaface_detections(img=im,
                                                                      retinaface_res_path=retinaface_result_path,
                                                                      im_path=path)
            else:
                raise NotImplementedError

            if len(bboxes) == 0:
                print('no face')
            else:
                with torch.no_grad():
                    results, score, source_embs = learner.infer_embeddings(conf, faces, embeddings, True)



                # embeddings = update_face_bank(embeddings, source_embs, results, emb_counts, names)

                if args.detector == "mtcnn":
                    bboxes = bboxes[:, :-1]  # shape:[10,4],only keep 10 highest possibiity faces
                bboxes = bboxes.astype(int)
                bboxes = bboxes + [-1, -1, 1, 1]  # personal choice

                body_bboxes, face_bboxes = process_openpose(openpose_dir, path)
                # for idx, bbox in enumerate(face_bboxes):
                #     if args.score:
                #         frame = draw_box_name(bbox, names[results[idx] + 1] + '_{:.2f}'.format(score[idx]), frame)
                #     else:
                #         frame = draw_box_name(bbox, names[results[idx] + 1], frame)
                print(bboxes)
                print(bboxes0)
                iou_curr_prev = np_vec_no_jit_iou(bboxes, bboxes0)

                # print(face_bboxes.shape, bboxes.shape)
                face_iou = np_vec_no_jit_iou(face_bboxes, bboxes)
                # print(face_iou)

                max_ious, max_inds = np.max(face_iou, axis=1), np.argmax(face_iou, axis=1)
                max_inds[max_ious <= 0.1] = -1

                curr_names = []
                curr_boxes = []
                print(names0)
                # First update the face banks
                for i, idx in enumerate(results):
                    if idx != -1:  # we find a match, smooth the previous embeddings
                        embeddings[idx] = (emb_counts[idx] * embeddings[idx] + source_embs[i]) / (
                                emb_counts[idx] + 1)
                        emb_counts[idx] += 1
                    else:  # The similarity did not pass the threshold, so we are unsure
                        # We check whether this box in the previous frame has a match
                        iou = np_vec_no_jit_iou(np.array([bboxes[i]]), bboxes0)
                        max_iou, max_ind = np.max(iou, axis=1), np.argmax(iou, axis=1)
                        if max_iou >= 0.6:  # There is a match in the previous frame
                            # Then we update the face bank
                            embeddings[idx] = (emb_counts[idx] * embeddings[idx] + source_embs[i]) / (
                                    emb_counts[idx] + 1)
                            emb_counts[idx] += 1
                        else:  # There is no match, so we add a new embedding
                            embeddings = torch.cat((embeddings, source_embs[i].unsqueeze(0)), dim=0)
                            emb_counts.append(1)
                            names.append("{:02d}".format(int(names[-1]) + 1))

                for j, bodybbox in enumerate(body_bboxes):
                    merged_box = merge_box(bodybbox, bboxes[max_inds[j]])

                    if max_inds[j] != -1:

                        # match for the previous frame using IOU
                        iou = np_vec_no_jit_iou(np.array([bboxes[max_inds[j]]]), bboxes0)
                        max_iou, max_ind = np.max(iou, axis=1), np.argmax(iou, axis=1)
                        print(iou, max_ind)
                        if max_iou >= 0.5:
                            frame = draw_box_name(merged_box, names0[max_ind[0]], frame)
                            frame = draw_box_name(bboxes[max_inds[j]], names0[max_ind[0]], frame)
                            curr_names.append(names0[max_ind[0]])
                        else:
                            frame = draw_box_name(merged_box, names[results[max_inds[j]] + 1], frame)
                            frame = draw_box_name(bboxes[max_inds[j]], names[results[max_inds[j]] + 1], frame)

                            curr_names.append(names[results[max_inds[j]] + 1])
                        curr_boxes.append(bboxes[max_inds[j]])
                if len(curr_names) != 0:
                    names0 = curr_names
                if len(curr_boxes) != 0:
                    bboxes0 = np.array(curr_boxes)
                    # frame = cv2.rectangle(frame, (bodybbox[0], bodybbox[1]), (bodybbox[2], bodybbox[3]), (0, 0, 255), 6)


                print(results, score, embeddings.shape, emb_counts, names)

            cv2.imwrite(osp.join(image_root, "results", "tracking", args.detector, seq, path), frame)
