# Created by yongxinwang at 2019-10-21 02:55
from Learner import face_learner
from config import get_config

import os
import os.path as osp
import argparse
from PIL import Image
import numpy as np
import torch

from mtcnn_pytorch.src.align_trans import warp_and_crop_face, get_reference_facial_points


def construct_face_bank(seq: str, retina_face_root: str, image_root: str, data_root: str, learner: face_learner, verbose: bool):
    """
    loop through all images under the seq folder and count the number of distinct face embeddings
    :param seq: sequence id
    :param retina_face_root: root dir for retinaface detections. (frame, face_bbox (4), landmarks (10))
    :param image_root: root dir for images
    :param openface_root: root dir for openface results
    :param learner: a face verification network
    :return:
    """
    seq_images = sorted(os.listdir(image_root))
    seq_retina_face_result = np.loadtxt(retina_face_root, dtype=str)

    embeddings = []
    names = ["Unknown"]
    emb_counts = []

    # get default reference face points
    reference = get_reference_facial_points(default_square=True)

    for frame_idx, frame in enumerate(seq_images):
        if verbose:
            print("Processing {} {}".format(frame_idx, frame))
            print("Names: {}".format(names))
            print("emb_counts: {}".format(emb_counts))

        # 1. load data
        # 1.1. read image
        img = Image.open(osp.join(image_root, frame))
        # 1.2. get the retinaface detections
        retinaface_result = seq_retina_face_result[seq_retina_face_result[:, 0] == frame]

        # skip if no detection
        if len(retinaface_result) == 0:
            print("No retinaface")
            continue

        retinaface_bboxes = retinaface_result[:, 1:5].astype(np.int)

        landmarks = retinaface_result[:, 5:].astype(np.float32)
        retinaface_bboxes = retinaface_bboxes

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

    np.savez(osp.join(data_root, "{}.npz".format(seq)),
             embeddings=embeddings.cpu().numpy(), names=names, emb_counts=emb_counts)
    return embeddings, names, emb_counts


def run(args, conf):
    image_root = args.image_root
    retina_face_root = args.retinaface
    result_root = args.result_root
    seq = os.path.basename(image_root)
    verbose = args.verbose
    if not osp.exists(result_root):
        os.makedirs(result_root, exist_ok=True)

    learner = face_learner(conf, True)
    learner.threshold = args.threshold
    learner.load_state(conf, fixed_str="ir_se50.pth", from_save_folder=False, model_only=True)
    learner.model.eval()

    ############################################################
    # 1. Construct face bank only using the confident faces
    # from openface
    ############################################################
    face_bank, names, emb_counts = construct_face_bank(seq, retina_face_root, image_root, result_root, learner, verbose)
    names, emb_counts = np.array(names), np.array(emb_counts)
    print("Sequence: {}, found {} unique faces".format(seq, len(emb_counts)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='for face verification')
    parser.add_argument('-th', '--threshold', help='threshold to decide identical faces', default=1.00, type=float)
    parser.add_argument("-v", "--verbose", help="whether to print while processing", action="store_true")
    parser.add_argument("-d", "--detector", help="what face detector to use", default="retinaface", type=str)
    parser.add_argument("-r", "--result_root", help="root dir to store results", type=str)
    parser.add_argument("-f", "--retinaface", help="retinaface detection results", type=str)
    parser.add_argument("-i", "--image_root", help="image dir", type=str)
    parser.add_argument("-m", "--model_path", default="/work/yongxinw/InsightFace/",
                        help="directory containing the pretrained model from insightface", type=str)

    args = parser.parse_args()

    conf = get_config(False)
    conf.model_path = args.model_path

    run(args, conf)
