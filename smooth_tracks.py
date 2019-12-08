# Created by yongxinwang at 2019-10-13 19:56
import numpy as np
import argparse
import os.path as osp
import os
import cv2

from utils import draw_box_name


def smooth_tracks(tracks, args):
    # get the unique ids in the raw tracking results
    unique_ids, unique_counts = np.unique(tracks[:, 1], return_counts=True)
    # print(unique_ids)
    # print(unique_counts)

    # filter out the ids that are too short, likely to be noises
    unique_ids = unique_ids[(unique_counts >= 100) & (unique_ids != -1)]
    # exit()

    smoothed = np.zeros(shape=(0, 11))
    smoothed_file = osp.join(image_root, "results", "tracking", args.detector, seq, "smoothed.txt")
    # clear buffer
    f = open(smoothed_file, 'w')
    f.close()

    for identity in unique_ids:
        # get all data for this identity
        track_i = tracks[tracks[:, 1] == identity]
        for j in range(len(track_i) - 1):
            # get the data between two detections
            curr_data = track_i[j]
            next_data = track_i[j + 1]

            # reformat data points
            curr = list(curr_data)
            curr.append(0)
            curr = np.array(curr).reshape(1, -1)

            next = list(next_data)
            next.append(0)
            next = np.array(next).reshape(1, -1)

            # see if they need to be smoothed
            if args.window >= next_data[0] - curr_data[0] > 1:
                curr_face_box = curr_data[2:6]
                curr_body_box = curr_data[6:]
                next_face_box = next_data[2:6]
                next_body_box = next_data[6:]

                interps = np.linspace(0, 1, num=next_data[0] - curr_data[0] + 1)
                interps = interps[1:-1]  # get rid of the head and tail
                face_box_interp = curr_face_box.reshape(1, -1) * (1 - interps.reshape(-1, 1)) + \
                    next_face_box.reshape(1, -1) * interps.reshape(-1, 1)

                body_box_interp = curr_body_box.reshape(1, -1) * (1 - interps.reshape(-1, 1)) + \
                    next_body_box.reshape(1, -1) * interps.reshape(-1, 1)

                frames = np.arange(next_data[0] - curr_data[0] - 1) + curr_data[0] + 1
                frames = frames.reshape(-1, 1)

                identities = np.array([identity] * (next_data[0] - curr_data[0] - 1)).reshape(-1, 1)

                smooth_flag = np.ones_like(identities)

                interpolated = np.concatenate((frames, identities, face_box_interp, body_box_interp, smooth_flag), axis=1)

                interpolated = np.concatenate((curr, interpolated), axis=0)
            else:
                interpolated = curr

            with open(smoothed_file, 'a') as sm:
                np.savetxt(sm, interpolated, fmt="%s")

            smoothed = np.concatenate((smoothed, interpolated), axis=0)

    # for s in smoothed:
    #     print(list(s))

    return smoothed


def visualize_tracks(tracks, image_root, seq, args):
    frames = np.unique(tracks[:, 0])
    start_frame = min(frames)
    save_dir = osp.join(image_root, "results", "tracking_noid", args.detector, seq)
    for f in range(1, int(max(frames))):
        data = tracks[tracks[:, 0] == int(f)]
        image_path = osp.join(image_root, "images", seq, "{:06d}.jpg".format(int(f)))
        frame = cv2.imread(image_path)

        if len(data) > 0:
            for box_data in data:
                identity = box_data[1]
                face_box = box_data[2:6].astype(int)
                body_box = box_data[6:].astype(int)
                # frame = draw_box_name(face_box, "{:02d}".format(int(identity)), frame)
                # frame = draw_box_name(body_box, "{:02d}".format(int(identity)), frame)
                frame = draw_box_name(face_box, "", frame)
                frame = draw_box_name(body_box, "", frame)

        save_path = osp.join(save_dir, "{:06d}_smoothed.jpg".format(int(f)))
        cv2.imwrite(save_path, frame)
    os.system("ffmpeg -framerate 20 -start_number {} -i {}/%06d_smoothed.jpg {}/{}_smoothed.mp4".format(
        start_frame, save_dir, save_dir, seq))

    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='for smoothing face tracks')
    parser.add_argument('-w', '--window', help='smoothing window length', default=20, type=int)
    parser.add_argument("-d", "--detector", help="what face detector to use", default="mtcnn", type=str)

    args = parser.parse_args()

    sequences = ["ZYzql-Y1sP4_trimmed-out"]
    # sequences = ["zzWaf1z9kgk_trimmed-out"]
    # sequences = ["zzWaf1z9kgk_trimmed-out"]
    # sequences = ["ztlOyCk5pcg_trimmed-out"]

    image_root = "/work/yongxinw/SocialIQ/raw/raw/vision/"
    # sequences = [
    #     "UUukBV82P9A_trimmed-out",
    #     "uB3FdWmnFZU_trimmed-out",
    #     "ztlOyCk5pcg_trimmed-out",
    #     "ZYzql-Y1sP4_trimmed-out",
    #     "zzERcgFrTrA_trimmed-out",
    #     "zzWaf1z9kgk_trimmed-out"
    # ]
    sequences = [
        "UUukBV82P9A_trimmed-out",
        "ztlOyCk5pcg_trimmed-out",
        "ZYzql-Y1sP4_trimmed-out",
        "zzERcgFrTrA_trimmed-out",
        "zzWaf1z9kgk_trimmed-out",
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

    for seq in sequences:
        print("Smoothing {}".format(seq))
        # raw_result = np.loadtxt(osp.join(image_root, "results", "tracking", args.detector, seq, "tracks.txt")).astype(int)
        raw_result = np.loadtxt(osp.join(image_root, "results", "tracking_noid", args.detector, seq, "tracks.txt")).astype(int)

        # smoothed = smooth_tracks(raw_result, args)
        # debug = np.array([["8", "00", "505", "82", "719", "361", "382", "82", "968", "715"],
        #                   ["17", "00", "499", "90", "725", "388", "384", "90", "919", "515"],
        #                   ["18", "00", "499", "90", "725", "388", "384", "90", "919", "515"]]).astype(np.int)
        # smoothed_tracks = smooth_tracks(raw_result, args)
        # visualize_tracks(smoothed_tracks, image_root, seq, args)
        visualize_tracks(raw_result, image_root, seq, args)
