# Created by yongxinwang at 2019-10-21 12:12
# Created by yongxinwang at 2019-10-13 19:56
import numpy as np
import argparse
import os.path as osp
import os
import cv2

from utils import draw_box_name


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


def smooth_tracks(tracks, args):
    start_frame = tracks[:, 0].astype(int).min()
    max_frame = tracks[:, 0].astype(int).max()

    # smoothed = np.zeros(shape=(0, 11))
    smoothed = tracks[tracks[:, 0].astype(int) == start_frame].copy()
    smoothed = smoothed[smoothed[:, 1] != "-1"]
    smoothed_file = osp.join(image_root, "results", "tracking_openface", args.detector, seq, "smoothed.txt")
    # clear buffer
    f = open(smoothed_file, 'w')
    f.close()
    print(smoothed)
    print(tracks)
    # exit()
    for frame in range(start_frame + 1, max_frame + 1):
        cur_data = tracks[tracks[:, 0].astype(int) == frame].copy()
        cur_data = cur_data[cur_data[:, 1] != "-1"]
        pre_frame = smoothed[:, 0].astype(int).max()
        pre_data = smoothed[smoothed[:, 0].astype(int) == pre_frame].copy()
        print(pre_data)
        print(cur_data)
        print("===========================================")
        if len(cur_data) == 0 or len(pre_data) == 0:
            print(frame)
            print("No data")
            continue

        cur_unknowns_inds = np.where(cur_data[:, 1] == 'Unknown')[0]
        cur_unknowns = cur_data[cur_unknowns_inds]
        if len(cur_unknowns) == 0:
            print(frame)
            print("No unknowns")
            smoothed = np.concatenate((smoothed, cur_data), axis=0)
            continue

        # import ipdb
        # ipdb.set_trace()

        if len(cur_unknowns) == 1:
            cur_unknowns = cur_unknowns.reshape(1, -1)

        ious = np_vec_no_jit_iou(cur_unknowns[:, 2:6].astype(float), pre_data[:, 2:6].astype(float))
        max_ious, max_inds = np.max(ious, axis=1), np.argmax(ious, axis=1)

        max_inds[max_ious < 0.8] = -1

        for i, max_idx in enumerate(max_inds):
            if max_idx != -1:
                pre_id = pre_data[max_idx, 1]
                cur_data[cur_unknowns_inds[i], 1] = pre_id

        smoothed = np.concatenate((smoothed, cur_data), axis=0)

    np.savetxt(smoothed_file, smoothed, fmt="%s")
    return smoothed

    # for identity in unique_ids:
    #     # get all data for this identity
    #     track_i = tracks[tracks[:, 1] == identity]
    #     for j in range(len(track_i) - 1):
    #         # get the data between two detections
    #         curr_data = track_i[j]
    #         next_data = track_i[j + 1]
    #
    #         # reformat data points
    #         curr = list(curr_data)
    #         curr.append(0)
    #         curr = np.array(curr).reshape(1, -1)
    #
    #         next = list(next_data)
    #         next.append(0)
    #         next = np.array(next).reshape(1, -1)
    #
    #         # see if they need to be smoothed
    #         if args.window >= next_data[0] - curr_data[0] > 1:
    #             curr_face_box = curr_data[2:6]
    #             curr_body_box = curr_data[6:]
    #             next_face_box = next_data[2:6]
    #             next_body_box = next_data[6:]
    #
    #             interps = np.linspace(0, 1, num=next_data[0] - curr_data[0] + 1)
    #             interps = interps[1:-1]  # get rid of the head and tail
    #             face_box_interp = curr_face_box.reshape(1, -1) * (1 - interps.reshape(-1, 1)) + \
    #                 next_face_box.reshape(1, -1) * interps.reshape(-1, 1)
    #
    #             body_box_interp = curr_body_box.reshape(1, -1) * (1 - interps.reshape(-1, 1)) + \
    #                 next_body_box.reshape(1, -1) * interps.reshape(-1, 1)
    #
    #             frames = np.arange(next_data[0] - curr_data[0] - 1) + curr_data[0] + 1
    #             frames = frames.reshape(-1, 1)
    #
    #             identities = np.array([identity] * (next_data[0] - curr_data[0] - 1)).reshape(-1, 1)
    #
    #             smooth_flag = np.ones_like(identities)
    #
    #             interpolated = np.concatenate((frames, identities, face_box_interp, body_box_interp, smooth_flag), axis=1)
    #
    #             interpolated = np.concatenate((curr, interpolated), axis=0)
    #         else:
    #             interpolated = curr
    #
    #         with open(smoothed_file, 'a') as sm:
    #             np.savetxt(sm, interpolated, fmt="%s")
    #
    #         smoothed = np.concatenate((smoothed, interpolated), axis=0)

    # return smoothed


def visualize_tracks(tracks, image_root, seq, args):
    print(tracks)
    frames = np.unique(tracks[:, 0].astype(int))
    print(frames)
    start_frame = min(frames)
    save_dir = osp.join(image_root, "results", "tracking_openface", args.detector, seq)
    for f in range(1, int(max(frames))):
        # data = tracks[tracks[:, 0] == int(f)]
        data = tracks[tracks[:, 0] == str(f)]
        image_path = osp.join(image_root, "results", "extracted_frames", "images", seq, "{:06d}.jpg".format(int(f)))
        frame = cv2.imread(image_path)

        if len(data) > 0:
            for box_data in data:
                identity = box_data[1]
                face_box = box_data[2:6].astype(float).astype(int)
                body_box = box_data[6:].astype(float).astype(int)
                if identity != "Unknown":
                    frame = draw_box_name(face_box, "{:02d}".format(int(identity)), frame)
                    frame = draw_box_name(body_box, "{:02d}".format(int(identity)), frame)
                else:
                    frame = draw_box_name(face_box, "{:s}".format(identity), frame)
                    frame = draw_box_name(body_box, "{:s}".format(identity), frame)
                # frame = draw_box_name(face_box, "", frame)
                # frame = draw_box_name(body_box, "", frame)

        save_path = osp.join(save_dir, "{:06d}_smoothed.jpg".format(int(f)))
        cv2.imwrite(save_path, frame)
    os.system("ffmpeg -y -framerate 20 -start_number {} -i {}/%06d_smoothed.jpg {}/{}_smoothed.mp4".format(
        start_frame, save_dir, save_dir, seq))

    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='for smoothing face tracks')
    parser.add_argument('-w', '--window', help='smoothing window length', default=20, type=int)
    parser.add_argument("-d", "--detector", help="what face detector to use", default="retinaface", type=str)

    args = parser.parse_args()

    # sequences = ["ZYzql-Y1sP4_trimmed-out"]
    # sequences = ["zzWaf1z9kgk_trimmed-out"]
    # sequences = ["zzWaf1z9kgk_trimmed-out"]
    # sequences = ["ztlOyCk5pcg_trimmed-out"]

    image_root = "/work/yongxinw/SocialIQ/raw/raw/vision/"
    # sequences = [
    #     "00m9ssEAnU4_trimmed-out",
    #     "02A2a-aEvmI_trimmed-out",
    #     "0_8zVtd5VLY_trimmed-out",
    #     "09j-Mg5guGg_trimmed-out",
    #     "0a2lv4IwZFY_trimmed-out",
    #     "0aNFS1cLs4k_trimmed-out",
    #     "_0at8kXKWSw_trimmed-out",
    #     "0B7UgDEzcUM_trimmed-out",
    #     "0B9VJfzqwhM_trimmed-out",
    #     "0DBfvvIVmqY_trimmed-out",
    #     "0djStqAuc_E_trimmed-out",
    #     "0gcuigYJ2cw_trimmed-out",
    #     "0hCihQ5Kep0_trimmed-out",
    #     "0LDAgvbRtk4_trimmed-out",
    #     "-0REX0yx4QA_trimmed-out",
    #     "0SuGC2ifkIA_trimmed-out",
    #     "11BN2alqrFU_trimmed-out",
    #     "13HM_96pUIo_trimmed-out",
    #     "13tG38-Ojn4_trimmed-out",
    #     "17PtS1Qx8kU_trimmed-out",
    #     "1a4Gx6UHdI8_trimmed-out",
    #     "1A7dqFxx8wU_trimmed-out",
    #     "1akNUksCpRU_trimmed-out",
    #     "1B_swZA4Hqk_trimmed-out",
    #     "1CjUHNYzW1E_trimmed-out",
    #     "1GJqfyzfCWU_trimmed-out",
    #     "1iH8ajlHJ1M_trimmed-out",
    #     "1IHDvzYwqgE_trimmed-out",
    #     "1j1C3dlwh7E_trimmed-out",
    #     "1mHjMNZZvFo_trimmed-out",
    #     "1MwN5nDajWs_trimmed-out",
    #     "1YFLBjR-swo_trimmed-out",
    #     "1YOzwKKbuPo_trimmed-out",
    #     "1Za8BtLgKv8_trimmed-out",
    #     "25Qq8k7V83g_trimmed-out",
    #     "29rvfxBQBQA_trimmed-out",
    #     "29vnZjb39u0_trimmed-out",
    #     "2a01Rg2g2Z8_trimmed-out",
    #     "2c_nei61mQQ_trimmed-out",
    #     "2DTwXdqmRiM_trimmed-out",
    #     "2G-B5upjLjM_trimmed-out",
    #     "2GRzO0p9uVU_trimmed-out",
    #     "2gy8H-wYzic_trimmed-out",
    #     "2hLcCnZzRtY_trimmed-out",
    #     "2ihOXaU0I8o_trimmed-out",
    #     "2im0kvBEIrg_trimmed-out",
    #     "2jMvc5VoavE_trimmed-out",
    #     "2K09qUoN1Oo_trimmed-out",
    #     "2MrFWB__GIA_trimmed-out",
    #     "2nDh8MQuS-Y_trimmed-out",
    #     "2NE4KCCfutk_trimmed-out",
    #     "-2NF882DbaM_trimmed-out",
    #     "2SIhhPzrjVA_trimmed-out",
    #     "2Wk9JY6ic0k_trimmed-out",
    #     "2XFVnzr4Vho_trimmed-out",
    #     "2YGrrsKs-Xg_trimmed-out",
    #     "2ZkzLAQszHA_trimmed-out",
    #     "31ktAMJocw8_trimmed-out",
    #     "34XCuNsQ7O8_trimmed-out",
    #     "3_-Bjqs6AxA_trimmed-out",
    #     "3d--LpVQxDo_trimmed-out",
    #     "3eaASCCJB4U_trimmed-out",
    #     "3eCZ8haia58_trimmed-out",
    #     "3esHlM0cBx4_trimmed-out",
    #     "3EwNcTzx-Bs_trimmed-out",
    #     "3m-54UiEqzg_trimmed-out",
    #     "3nUKwvFsjA4_trimmed-out",
    #     "3oj7mCSydoM_trimmed-out",
    #     "3qp3AeWmt38_trimmed-out",
    #     "3udl28R3yIA_trimmed-out",
    #     "3uk6rKXbG1M_trimmed-out",
    #     "3vjA8sDxDuQ_trimmed-out",
    #     "3wIejfT9l30_trimmed-out",
    #     "3xdQIW24aj0_trimmed-out",
    #     "3yovMKR__4Q_trimmed-out",
    #     "3zjz6ryPvIg_trimmed-out",
    #     "40mpZRU47T4_trimmed-out",
    #     "43fC9xuQRCY_trimmed-out",
    #     "44MVdpDEQJs_trimmed-out",
    #     "460_-P8pK8E_trimmed-out",
    #     "47lUTksozNI_trimmed-out",
    #     "47U9SVOiw4o_trimmed-out",
    #     "4AmVjblOvy4_trimmed-out",
    #     "4An2AF2rWGk_trimmed-out",
    #     "4_BudcPRi7E_trimmed-out",
    #     "4EZxURAhU6U_trimmed-out",
    #     "4HHR_3HJdEQ_trimmed-out",
    #     "4HN0caXjW3s_trimmed-out",
    #     "4HxgOizA2Ow_trimmed-out",
    #     "4iw1jTY-X3A_trimmed-out",
    #     "4_jXi0nzuow_trimmed-out",
    #     "4KAvDyGzz4E_trimmed-out",
    #     "4LGe265pwvU_trimmed-out",
    #     "4oh_PsdY-W0_trimmed-out",
    #     "4pYqIEQow2s_trimmed-out",
    #     "4Ry2bE-WRqA_trimmed-out",
    #     "4tLBy9FGS5A_trimmed-out",
    #     "4VA4kqMnEqA_trimmed-out",
    #     "4Vic0qKl64Y_trimmed-out",
    #     "4vPrTC5qMh0_trimmed-out",
    #     "4W192A7g5KY_trimmed-out",
    #     "4wdeBJ39Cuw_trimmed-out",
    #     "4yr_etbfZtQ_trimmed-out",
    #     "56EbHWYK_q0_trimmed-out",
    #     "58DqoE56OWc_trimmed-out",
    #     "5BLjOCK2SlM_trimmed-out",
    #     "5fy7S3jCyAg_trimmed-out",
    #     "5h-SslT--8E_trimmed-out",
    #     "5NHofFOpsDU_trimmed-out",
    #     "5OUcvUDMMWE_trimmed-out",
    #     "5RS1CKa3JVg_trimmed-out",
    #     "5slZuaphzVs_trimmed-out",
    #     "5_uSZcXMV7s_trimmed-out",
    #     "5WgFDQUjg5s_trimmed-out",
    #     "5XEQ8rYl1qs_trimmed-out",
    #     "64mWOoj68qo_trimmed-out",
    #     "66ojfophGys_trimmed-out",
    #     "6AzXxhPKh8U_trimmed-out",
    #     "6b1QbKtmaZ0_trimmed-out",
    #     "6dCClwMqJK8_trimmed-out",
    #     "6I7Ktp4dV_s_trimmed-out",
    #     "6kYu7-5EyU8_trimmed-out",
    #     "6qNawyzVGbc_trimmed-out",
    #     "6rkV4QRcVnk_trimmed-out",
    #     "6tAfdCTnToY_trimmed-out",
    #     "6V0UfejAo_E_trimmed-out",
    #     "6W77wcXg2no_trimmed-out",
    #     "6xQv6ozrz90_trimmed-out",
    #     "72ltfGTYqpQ_trimmed-out",
    #     "79I7_vkwaeg_trimmed-out",
    #     "7bc_qfRmPK0_trimmed-out",
    #     "7doQf8xjFVg_trimmed-out",
    #     "7FYHA728nBI_trimmed-out",
    #     "7grGUUPbEbo_trimmed-out",
    #     "7GRTyxc4uMU_trimmed-out",
    #     "7GRWqlKfgmg_trimmed-out",
    #     "7_lpdZhf28E_trimmed-out",
    #     "7Oum_c5Seis_trimmed-out",
    #     "7wLDCFduiLY_trimmed-out",
    #     "87yBSfgwoUI_trimmed-out",
    #     "8ACAI_Z7aLM_trimmed-out",
    #     "8_Av3cDcoR8_trimmed-out",
    #     "8fN6D1VOHlo_trimmed-out",
    #     "-8GAQpsV4Qo_trimmed-out",
    #     "8-Hi9NmF4rM_trimmed-out",
    #     "8i0Vr6DiBCQ_trimmed-out",
    #     "8Kv4F0D210A_trimmed-out",
    #     "8m_3eBsy22Y_trimmed-out",
    #     "8MK9frCMoWA_trimmed-out",
    #     "8NL5jXoa-Jc_trimmed-out",
    #     "8Rk4sGEBJlM_trimmed-out",
    #     "8SGQ0VdXvAg_trimmed-out",
    #     "8TDAP0KNIIw_trimmed-out",
    #     "8w41NfRyWqE_trimmed-out",
    #     "8wLCmDtCDAM_trimmed-out",
    #     "8xFtIsyRvNE_trimmed-out",
    #     "8y-N6UDxTxQ_trimmed-out",
    #     "90P3VEbzUK0_trimmed-out",
    #     "96YOZOU7ggo_trimmed-out",
    #     "97AUfvzQ_1E_trimmed-out",
    #     "-99aZZhUgRk_trimmed-out",
    #     "9cFEh0aaOOo_trimmed-out",
    #     "9eqze5JWNjY_trimmed-out",
    #     "9hn6Z1o-IYI_trimmed-out",
    #     "9jRkACywckE_trimmed-out",
    #     "9kLNVTm3Z90_trimmed-out",
    #     "9L1tM3fOb80_trimmed-out",
    #     "9l2W_GDiNyE_trimmed-out",
    #     "9m0d0RaWpfY_trimmed-out",
    #     "-9NhaKWMtWU_trimmed-out",
    #     "9PJb4cFWOfY_trimmed-out",
    #     "9QdaNUrq1EQ_trimmed-out",
    #     "9qK9VQDELpc_trimmed-out",
    #     "a0tn33wZGVo_trimmed-out",
    #     "A3WbCRfad-w_trimmed-out",
    #     "A48AJ_5nWsc_trimmed-out",
    #     "A4gVxvYFA3M_trimmed-out",
    #     "a6Ke1YThz4o_trimmed-out",
    #     "A6Pz9V6LzcU_trimmed-out",
    #     "a80o4DGxt7Q_trimmed-out",
    #     "aai7dDBNXBs_trimmed-out",
    #     "abOuBvUfQk4_trimmed-out",
    #     "AciwXaRfh3k_trimmed-out",
    #     "ACPPfJtYCVc_trimmed-out",
    #     "aDJJBMXiwiI_trimmed-out",
    #     "afXewnGZXKs_trimmed-out",
    #     "ahcAFnY6iAY_trimmed-out",
    #     "AHiA9hohKr8_trimmed-out",
    #     "AHXwnFvqYDk_trimmed-out",
    #     "AiIrjf-s128_trimmed-out",
    #     "ajFmgmUSYAc_trimmed-out",
    #     "ajVTImleJlk_trimmed-out",
    #     "AKAtC7easns_trimmed-out",
    #     "ALbnaCezgdM_trimmed-out",
    #     "alg7qHta0Sk_trimmed-out",
    #     "Am6NHDbj6XA_trimmed-out",
    #     "aNOuoSVlunM_trimmed-out",
    #     "ap9vRY_Vdwc_trimmed-out",
    #     "ApExci9PnNM_trimmed-out",
    #     "APshm-9gPgI_trimmed-out",
    #     "AQ7wbfX_av0_trimmed-out",
    #     "aqGNOsZFdBU_trimmed-out",
    #     "AQX2Q-V2Uh8_trimmed-out",
    #     "aRQLU3IwNYs_trimmed-out",
    #     "aS01LwpC23g_trimmed-out",
    #     "ASqnnZpsX1M_trimmed-out",
    #     "aSZ_eLxuLAs_trimmed-out",
    #     "Ate-1815RNA_trimmed-out",
    #     "atEkAkPfpUY_trimmed-out",
    #     "_AuZO31q62g_trimmed-out",
    #     "aw-fKJhcQE4_trimmed-out",
    #     "awpHn196aVs_trimmed-out",
    #     "aXiMaioTUkg_trimmed-out",
    #     "AZCs9VoHeBo_trimmed-out",
    #     "b0yONlMjxjs_trimmed-out",
    #     "b1OedrPQ464_trimmed-out",
    #     "B1VB7vVQNQg_trimmed-out",
    #     "B2V9PFGQBH4_trimmed-out",
    #     "b3I1tK1Iyzc_trimmed-out",
    #     "B5ltukfhtw8_trimmed-out",
    #     "B6p6X1LSjiA_trimmed-out",
    #     "B6PpxrnttDg_trimmed-out",
    #     "B7Nbbxh3m1Q_trimmed-out",
    #     "B7XIUxyTi_8_trimmed-out",
    #     "b9aeM__20E8_trimmed-out",
    #     "badtXoOJaf8_trimmed-out",
    #     "bb08nFwfoxA_trimmed-out",
    #     "bBRWF0wju-c_trimmed-out",
    #     "BC0dD13bwEw_trimmed-out",
    #     "bC9hc4cqHGY_trimmed-out",
    #     "bCKOVlsSluU_trimmed-out",
    #     "bCWEOlvi5fY_trimmed-out",
    #     "BDEUrfqlwcg_trimmed-out"
    # ]
    # sequences = [
    #     "Bd_vAawM9LA_trimmed-out",
    #     "BEOdicifuqM_trimmed-out",
    #     "b-FX9NOVQOM_trimmed-out",
    #     "bgczomH1kLk_trimmed-out",
    #     "B-gHVjv4_c4_trimmed-out",
    #     "Bg_tJvCA8zw_trimmed-out",
    #     "BH8FUBW4IIE_trimmed-out",
    #     "BiV9eJU8Gsw_trimmed-out",
    #     "bJ-G8xiLB6o_trimmed-out",
    #     "Bks4JX95dD8_trimmed-out",
    #     "bLVm1vfXRw8_trimmed-out",
    #     "bMuoPr5-Yt4_trimmed-out",
    #     "br0mu7r-ak0_trimmed-out",
    #     "-bSM6iswghE_trimmed-out",
    #     "bT_DEZz99VQ_trimmed-out",
    #     "BUumpYIgVg4_trimmed-out",
    #     "bwzH7ceQX8Y_trimmed-out",
    #     "C08WmKiwcSs_trimmed-out",
    #     "C0g5RjQ7cRE_trimmed-out",
    #     "C2PneBztZ3g_trimmed-out",
    #     "c2pwnHLaYTQ_trimmed-out",
    #     "c67D5bP0Hg4_trimmed-out",
    #     "C6RMS4F6LDc_trimmed-out",
    #     "caOaW604Tqc_trimmed-out",
    #     "CbMVjQV9b40_trimmed-out",
    #     "CggDN9EIuNY_trimmed-out",
    #     "cGTFuTIgc88_trimmed-out",
    #     "cGU1Pepn1hU_trimmed-out",
    #     "cHpER0dG1o8_trimmed-out",
    #     "CNHBsxOZd80_trimmed-out",
    #     "Cn_Mlwouwng_trimmed-out",
    #     "cOlibbx5sx0_trimmed-out",
    #     "CoMz3JOnZFo_trimmed-out",
    #     "COYJC6dvB8I_trimmed-out",
    #     "cq1er8IWz1U_trimmed-out",
    #     "cQREa5Y-jqk_trimmed-out",
    #     "Csy2RxzkbaM_trimmed-out",
    #     "ctHj7R35dL0_trimmed-out",
    #     "cuR-l2qCxBc_trimmed-out",
    #     "Cv4Xj4fIkRo_trimmed-out",
    #     "CwanEycyH_8_trimmed-out",
    #     "cwoR3fkcJ9g_trimmed-out",
    #     "CXmRmrBPDII_trimmed-out",
    #     "cXTjL-f-msU_trimmed-out",
    #     "CY2D1L1JtKU_trimmed-out",
    #     "D0a2KWuL4S0_trimmed-out",
    #     "D1Cil5n_-zs_trimmed-out",
    #     "D1FXpqUivtU_trimmed-out",
    #     "D2g3gTRkv0U_trimmed-out",
    #     "D2VcClclMbs_trimmed-out",
    #     "d43n4njmxcE_trimmed-out",
    #     "D56yCIgqqgk_trimmed-out",
    #     "d89i7OY2yTw_trimmed-out",
    #     "dACF-Mz-X8M_trimmed-out",
    #     "-daGjyKKNio_trimmed-out",
    #     "DB7de4nC2rc_trimmed-out",
    #     "DClIawJYpHs_trimmed-out",
    #     "Ddbyb8zVKG0_trimmed-out",
    #     "DE5S7W8ZfnI_trimmed-out",
    #     "deKPBy_uLkg_trimmed-out",
    #     "DelU5tQ4grw_trimmed-out",
    #     "dI5D3aTgjZk_trimmed-out",
    #     "DiaDblUd-lw_trimmed-out",
    #     "DK8s_btC8F8_trimmed-out",
    #     "dKxXtOyMmYc_trimmed-out",
    #     "dONZkRDs4k4_trimmed-out",
    #     "DpTB4TDKIa0_trimmed-out",
    #     "-DTqvzmUw74_trimmed-out",
    #     "dU7L1hvMx9Y_trimmed-out",
    #     "DuXGDE6tolY_trimmed-out",
    #     "dvisqlHIKpM_trimmed-out",
    #     "dVJAvMbb8H4_trimmed-out",
    #     "DW2umNQrQU0_trimmed-out",
    #     "DWmUHNpOJxI_trimmed-out",
    #     "DXyaQVlRVkY_trimmed-out",
    #     "dZPwXsbohK4_trimmed-out",
    #     "DZsBei4nCkU_trimmed-out",
    #     "E0TBOKN8J2E_trimmed-out",
    #     "E2IdU5lgaH4_trimmed-out",
    #     "E4MUXs4IHtY_trimmed-out",
    #     "e4mvg9r6_cI_trimmed-out",
    #     "e6ppqFNBkLo_trimmed-out",
    #     "e6zn4UlO0fU_trimmed-out",
    #     "e8v9i_ksUyY_trimmed-out",
    #     "EaPaLCuXjT8_trimmed-out",
    #     "EC77tcJZIdU_trimmed-out",
    #     "ecALuiFDRT0_trimmed-out",
    #     "eDqEcrIRxgQ_trimmed-out",
    #     "EeClqsYITso_trimmed-out",
    #     "EEDZjwA1wM8_trimmed-out",
    #     "Eg307HcpbJE_trimmed-out",
    #     "EGK2P1cOJJc_trimmed-out",
    #     "egw67gXKK3A_trimmed-out",
    #     "EJdboFptQ3o_trimmed-out",
    #     "eKQKEi2-0Ws_trimmed-out",
    #     "ElghrCC2Rbs_trimmed-out",
    #     "epy3Dy2FUOI_trimmed-out",
    #     "EqXKrS3gPN4_trimmed-out",
    #     "erOpqmubBL4_trimmed-out",
    #     "eS8SpCRASr0_trimmed-out",
    #     "eS9U1QO0F7M_trimmed-out",
    #     "eTnuG394AcY_trimmed-out",
    #     "eTph1-CG280_trimmed-out",
    #     "EUIIWsgDpZY_trimmed-out",
    #     "EwAb8ZW5Eiw_trimmed-out",
    #     "EWUfDU8TWn4_trimmed-out",
    #     "F0wIBTfLnE8_trimmed-out",
    #     "F2mIH0vlI9c_trimmed-out",
    #     "F2Xul-ihUVc_trimmed-out",
    #     "F2YbeTjcpfs_trimmed-out",
    #     "f3Ch2aIlXWo_trimmed-out",
    #     "F4rSKCXqEw0_trimmed-out",
    #     "FAaWqJLCCd0_trimmed-out",
    #     "FaLaPEnjeqY_trimmed-out",
    #     "f-BbAnnQVtY_trimmed-out",
    #     "fce2RtEvPr8_trimmed-out",
    #     "fC_Z5HlK9Pw_trimmed-out",
    #     "fDe50VbOU44_trimmed-out",
    #     "FgnO3muvvoM_trimmed-out",
    #     "fhqu7ve9MA4_trimmed-out",
    #     "fHW461eiQp8_trimmed-out",
    #     "FiLWlqVE9Fg_trimmed-out",
    #     "FJF56lmDqQo_trimmed-out",
    #     "FkLblGfWAvY_trimmed-out",
    #     "fL3_AauvjJ4_trimmed-out",
    #     "fmuEMg2fh_8_trimmed-out",
    #     "FositxHjuUk_trimmed-out",
    #     "fpxPstb2DAU_trimmed-out",
    #     "fsBzpr4k3rY_trimmed-out",
    #     "fV1o_g6uzuI_trimmed-out",
    #     "FWBCTZiijEM_trimmed-out",
    #     "Fy3Mi8rOB3U_trimmed-out",
    #     "Fy6BOTB4sXw_trimmed-out",
    #     "FybhAns3or8_trimmed-out",
    #     "fz0q7YKjp48_trimmed-out",
    #     "fZuk-TaECZo_trimmed-out",
    #     "G1wsCworwWk_trimmed-out",
    #     "G22mJGndp14_trimmed-out",
    #     "G3xzem7HSME_trimmed-out",
    #     "G4heS2754l4_trimmed-out",
    #     "G4ROcoq32rQ_trimmed-out",
    #     "g67e0hDT1oQ_trimmed-out",
    #     "G7bkiEh7_AM_trimmed-out",
    #     "g7OMgsD7T74_trimmed-out",
    #     "g8cyMIFcC_g_trimmed-out",
    #     "g8D-LyfTrRs_trimmed-out",
    #     "gAPPzmRb4r0_trimmed-out",
    #     "gBs-CkxGXy8_trimmed-out",
    #     "gbVOyKifrAo_trimmed-out",
    #     "GbYGoWvJpwI_trimmed-out",
    #     "gcDnKQul_c8_trimmed-out",
    #     "GcImUUGmZ3I_trimmed-out",
    #     "GCZ5aagOddY_trimmed-out",
    #     "gDUFvLWl-Oc_trimmed-out",
    #     "gDVmHsYgJUA_trimmed-out",
    #     "geaSpx-R4Kc_trimmed-out",
    #     "GeZIBgX7vkg_trimmed-out",
    #     "gf5mfpSDFKM_trimmed-out",
    #     "gfA1xa-BMCg_trimmed-out",
    #     "GGEXxniRfWQ_trimmed-out",
    #     "ggLOXOiq7WE_trimmed-out",
    #     "GHBKZZuA314_trimmed-out",
    #     "GI8LoYEYKI0_trimmed-out",
    #     "gImYPbTTZko_trimmed-out",
    #     "gjX78p5tvfo_trimmed-out",
    #     "GK4_G33fXFU_trimmed-out",
    #     "gKuBUQVcDJM_trimmed-out",
    #     "GMidefrr1MM_trimmed-out",
    #     "grTg3dzQDZI_trimmed-out",
    #     "GxYimeaoea0_trimmed-out",
    #     "GzPIbX1pzDg_trimmed-out",
    #     "H0Qdz8bSkv0_trimmed-out",
    #     "h35dZhHkuFM_trimmed-out",
    #     "h7YTPuEMgaE_trimmed-out",
    #     "h9hAaQOanZY_trimmed-out",
    #     "Ham3IQQzoU8_trimmed-out",
    #     "hBdsfj0YPO8_trimmed-out",
    #     "HCgv_HNoJrY_trimmed-out",
    #     "hcu4zY2HUQY_trimmed-out",
    #     "hd8bXHCvZME_trimmed-out",
    #     "HDhwReMUBsA_trimmed-out",
    #     "HEke6Dlhqtw_trimmed-out",
    #     "hezHSFSwa08_trimmed-out",
    #     "HFPGeaEPy9o_trimmed-out",
    #     "h-H1LddWxo8_trimmed-out",
    #     "hjGsGtihBpc_trimmed-out",
    #     "hkAFdIrTR00_trimmed-out",
    #     "hKfNp8NU82o_trimmed-out",
    #     "hL2u93brqiA_trimmed-out",
    #     "HMkOO15nye8_trimmed-out",
    #     "HN75tPziZAo_trimmed-out",
    #     "-hnBHBN8p5A_trimmed-out",
    #     "hnfkZx4jmpA_trimmed-out",
    #     "HPszYa77CkM_trimmed-out",
    #     "H-q0zh-XGVc_trimmed-out",
    #     "hqzF4IDaIYE_trimmed-out",
    #     "hRcSU9-krNU_trimmed-out",
    #     "hrhX40bQYY0_trimmed-out",
    #     "hRkl5WhbQLc_trimmed-out",
    #     "HSEi0RXGVq8_trimmed-out",
    #     "Hv0PyGfxbpw_trimmed-out",
    #     "HwIRCeUCyxw_trimmed-out",
    #     "HzH0cBmHg5k_trimmed-out",
    #     "I0izJOlMJiM_trimmed-out",
    #     "i3itBKdwE7M_trimmed-out",
    #     "I4mItsGR3uI_trimmed-out",
    #     "i5YckMkwmm4_trimmed-out",
    #     "I7GvG0WYfOo_trimmed-out",
    #     "i7JGn06gsnA_trimmed-out",
    #     "IbkyL0pDtEc_trimmed-out",
    #     "iBL0FcUTFT8_trimmed-out",
    #     "ICBNX0i855Q_trimmed-out",
    #     "_Ice5RkbWUY_trimmed-out",
    #     "iDgaqD7CWXU_trimmed-out",
    #     "IE6r8Pk91T0_trimmed-out",
    #     "iEENyD0JiRE_trimmed-out",
    #     "iFgOoTRflnw_trimmed-out",
    #     "ifxLhziWjm0_trimmed-out",
    #     "igH3ixpts2g_trimmed-out",
    #     "ihP926ccYDw_trimmed-out",
    #     "IHU9Jc_NUuk_trimmed-out",
    #     "iiDdvLrSUG4_trimmed-out",
    #     "Ilf38Achvzk_trimmed-out",
    #     "IlLFTI24Qkw_trimmed-out",
    #     "ilotZqzaZgU_trimmed-out",
    #     "imUigBNF-TE_trimmed-out",
    #     "im_uLJKzs-4_trimmed-out",
    #     "iNr9xdc6cVA_trimmed-out",
    #     "iOdEJMNEzyI_trimmed-out",
    #     "iovKlisBCzU_trimmed-out",
    #     "ipnGPeRIy2k_trimmed-out",
    #     "IsEykad2V9U_trimmed-out",
    #     "IsgFVkMnqJc_trimmed-out"
    # ]
    # sequences = [
    #     "Iu6_k2ok00U_trimmed-out",
    #     "iwCdv9iR8P8_trimmed-out",
    #     "IXbAYg5pp9M_trimmed-out",
    #     "ixQbCXLUUj8_trimmed-out",
    #     "izCiPuiGe9E_trimmed-out",
    #     "j1CTHVQ8Z3k_trimmed-out",
    #     "j3pLDghHKyc_trimmed-out",
    #     "j5SKmUoL9Tg_trimmed-out",
    #     "j7cUdCFEpeU_trimmed-out",
    #     "J94uO-urSTg_trimmed-out",
    #     "jbHKeVsI35M_trimmed-out",
    #     "-jDIwv4wCsU_trimmed-out",
    #     "jdrVAQrzt9Y_trimmed-out",
    #     "jE1tmy-g6tw_trimmed-out",
    #     "jfKDozXX_Uo_trimmed-out",
    #     "jh5PklItWjA_trimmed-out",
    #     "jiMUoVjQ5uI_trimmed-out",
    #     "jK08J1811uA_trimmed-out",
    #     "jKguXsNkJ4w_trimmed-out",
    #     "JMzilFvwNXE_trimmed-out",
    #     "jMZLrPxp31E_trimmed-out",
    #     "JNGcJEZ6rwA_trimmed-out",
    #     "jP2HuCwfKFA_trimmed-out",
    #     "Jp_KHLvQcuw_trimmed-out",
    #     "jrv2BW_c8jA_trimmed-out",
    #     "JRYjFh_hHBs_trimmed-out",
    #     "jtl5XK7QP38_trimmed-out",
    #     "jvpj4qdPRE0_trimmed-out",
    #     "JW2HHfQiGVs_trimmed-out",
    #     "JW3OfSCZlhc_trimmed-out",
    #     "JxbV5wGpXc8_trimmed-out",
    #     "JXYyQamYw84_trimmed-out",
    #     "jYL4gMsGZgE_trimmed-out",
    #     "jYSuKn09_e4_trimmed-out",
    #     "K25zmgYg5s4_trimmed-out",
    #     "k5bk1efKBSI_trimmed-out",
    #     "K6v2QYiMdCE_trimmed-out",
    #     "KaIzZrMb2og_trimmed-out",
    #     "KBj3TocgvOA_trimmed-out",
    #     "KBMAUGQpHBU_trimmed-out",
    #     "KB_p2QvvLGw_trimmed-out",
    #     "K-bZQJ3P9N0_trimmed-out",
    #     "KCbXRRvnFj8_trimmed-out",
    #     "_KE_-EdMhDI_trimmed-out",
    #     "kefqa4xre2I_trimmed-out",
    #     "kf9RtsXQPWU_trimmed-out",
    #     "kGoON1J872w_trimmed-out",
    #     "KHgGasfOFPg_trimmed-out",
    #     "kIfsmKj42XE_trimmed-out",
    #     "Kjlt_FgKPgA_trimmed-out",
    #     "k-kyKv_kKGU_trimmed-out",
    #     "kmgsC68hIL8_trimmed-out",
    #     "kmi_liqBsdU_trimmed-out",
    #     "KoH4OKx4x1Y_trimmed-out",
    #     "kOIWYXDRR7s_trimmed-out",
    #     "KpljzfIVBDM_trimmed-out",
    #     "kpMLnFxvi6Y_trimmed-out",
    #     "kRh1zXFKC_o_trimmed-out",
    #     "ks66e-O4YCQ_trimmed-out",
    #     "ktdgC1dJkOA_trimmed-out",
    #     "kU6YY2z7z0I_trimmed-out",
    #     "KvbeKlGeNRU_trimmed-out",
    #     "kVh1Jw2D9NY_trimmed-out",
    #     "KWnV1Aa6VQ8_trimmed-out",
    #     "KWS6OcNbvLA_trimmed-out",
    #     "KWSDwS4S6Ss_trimmed-out",
    #     "kYnmc_AVfMs_trimmed-out",
    #     "kzhJb5jcH58_trimmed-out",
    #     "KzN6XWDEmXI_trimmed-out",
    #     "kztkcj-WAvw_trimmed-out",
    #     "l1jW3OMXUzs_trimmed-out",
    #     "L3uDQ0S1Iis_trimmed-out",
    #     "L49M8C4wjVU_trimmed-out",
    #     "L75hdqt98nw_trimmed-out",
    #     "L9U9hvUf42g_trimmed-out",
    #     "lacVKwbsE7Q_trimmed-out",
    #     "lB0mkAa1vfM_trimmed-out",
    #     "LBd5x_fe4Jc_trimmed-out",
    #     "lC8nM75AqUk_trimmed-out",
    #     "LchHXKL6xZY_trimmed-out",
    #     "LcHtLypALog_trimmed-out",
    #     "LcMBahfo0NA_trimmed-out",
    #     "lcmI9_aypI0_trimmed-out",
    #     "LD1RAPiT_7A_trimmed-out",
    #     "lickge5rPdc_trimmed-out",
    #     "licUm-aEaCY_trimmed-out",
    #     "LiqyOoGW-I8_trimmed-out",
    #     "lJ83ILGA8yI_trimmed-out",
    #     "lkeVfgI0eEk_trimmed-out",
    #     "lKPRa_4hnlE_trimmed-out",
    #     "LmCJIBsQjOY_trimmed-out",
    #     "lmcuesVvLf0_trimmed-out",
    #     "lo0R1mvjDT8_trimmed-out",
    #     "LoMhBo8ATBM_trimmed-out",
    #     "-Lp96hoSUC8_trimmed-out",
    #     "lpAMi2lwjo0_trimmed-out",
    #     "LTEiibVnRgI_trimmed-out",
    #     "lTfsqoo-jKg_trimmed-out",
    #     "lTk5zkc0i8M_trimmed-out",
    #     "LTUojzYVUUI_trimmed-out",
    #     "LtxWWUBt7ro_trimmed-out",
    #     "luJceOt47UM_trimmed-out",
    #     "_LUX70mXcEE_trimmed-out",
    #     "lUyKpfbB9M8_trimmed-out",
    #     "m075kb3aV04_trimmed-out",
    #     "M4blAdS6r3Q_trimmed-out",
    #     "mA402F5K47o_trimmed-out",
    #     "MBDZbACupsc_trimmed-out",
    #     "MCAH-zHLTLE_trimmed-out",
    #     "Md4QnipNYqM_trimmed-out",
    #     "MdG9Lkk8VWo_trimmed-out",
    #     "MDQTH8WkAvQ_trimmed-out",
    #     "Me-A3eOhgXs_trimmed-out",
    #     "Mek_AQ5DbUs_trimmed-out",
    #     "Mf76yyTY7Ss_trimmed-out",
    #     "MHVrwCEWLPI_trimmed-out",
    #     "MIFz86h5nEA_trimmed-out",
    #     "mJwWxbSR6r0_trimmed-out",
    #     "MlteErDn4to_trimmed-out",
    #     "MM0YOB-cSWA_trimmed-out",
    #     "mN1Z5xEOy10_trimmed-out",
    #     "mNcdlLIOdNw_trimmed-out",
    #     "mO8TK-QaIf8_trimmed-out",
    #     "mPAESQEQoms_trimmed-out",
    #     "mpHoYhIFKNI_trimmed-out",
    #     "MqiOBIxouw4_trimmed-out",
    #     "mr88Ud5V9uE_trimmed-out",
    #     "MsjnLoTKAXo_trimmed-out",
    #     "muuW2EU8nrA_trimmed-out",
    #     "mVnqP-vLpuo_trimmed-out",
    #     "mxa4KXSz9rw_trimmed-out",
    #     "N188QSyfmeQ_trimmed-out",
    #     "N1fVL4AQEW8_trimmed-out",
    #     "N3QuD26DuCI_trimmed-out",
    #     "N4fvzhsdTio_trimmed-out",
    #     "n5_HdNzf03Q_trimmed-out",
    #     "N5sGxcAJFCA_trimmed-out",
    #     "n5V7TVYiVas_trimmed-out",
    #     "n6-ef_YHeJU_trimmed-out",
    #     "N-6zVmVuTs0_trimmed-out",
    #     "n71IBjXHnrQ_trimmed-out",
    #     "N7K5DQvMWXM_trimmed-out",
    #     "n8Cn_KhcR7o_trimmed-out",
    #     "nAkqWGO1T-c_trimmed-out",
    #     "nblf7Yw4jys_trimmed-out",
    #     "ncHMwblapdI_trimmed-out",
    #     "NcHQjL_WlxQ_trimmed-out",
    #     "Nck6BZga7TQ_trimmed-out",
    #     "nCQB2AaVgOY_trimmed-out",
    #     "ndsKiHu63oM_trimmed-out",
    #     "nEpJaJyeRy8_trimmed-out",
    #     "NFKdaj1Qsek_trimmed-out",
    #     "nGah7qST1dI_trimmed-out",
    #     "ngHnomR-114_trimmed-out",
    #     "NgPP6UXVkYU_trimmed-out",
    #     "nLu13aVrNcQ_trimmed-out",
    #     "n_mTiDeQvWg_trimmed-out",
    #     "NmWjnYUkT_s_trimmed-out",
    #     "nmWplIQhvoA_trimmed-out",
    #     "NnhpG0nEC84_trimmed-out",
    #     "-nO3dBeh5n0_trimmed-out",
    #     "No6mB6V1wL4_trimmed-out",
    #     "noHN3H3gWPQ_trimmed-out",
    #     "Nqf15ViHSmQ_trimmed-out",
    #     "nQqY3j8btbI_trimmed-out",
    #     "nqY3tv-y62A_trimmed-out",
    #     "NR9v3PBJw8g_trimmed-out",
    #     "Nra5F1tQQww_trimmed-out",
    #     "nRfU_c4QlaQ_trimmed-out",
    #     "NTpj7quCIqQ_trimmed-out",
    #     "NTySVAtrdQc_trimmed-out",
    #     "Nx5VK6DUUEY_trimmed-out",
    #     "nyqh7vhU3X0_trimmed-out",
    #     "NysITFb_Wbs_trimmed-out",
    #     "nZiIVvEJ8m0_trimmed-out",
    #     "nzj7Wg4DAbs_trimmed-out",
    #     "NZtIGzAzJZM_trimmed-out",
    #     "O24d_rJ4uDo_trimmed-out",
    #     "o4CKGkaTn-A_trimmed-out",
    #     "O5rTU5EA1C8_trimmed-out",
    #     "o6DEc3OYQfU_trimmed-out",
    #     "o6hMDs4rBmw_trimmed-out",
    #     "o7Ax6SRTGks_trimmed-out",
    #     "O8iOYngEUBU_trimmed-out",
    #     "o92pxWhZomM_trimmed-out",
    #     "O951zoCuU7Q_trimmed-out",
    #     "Ob3hr2H3VD4_trimmed-out",
    #     "Obde542xA9I_trimmed-out",
    #     "ocg0MbBfCS0_trimmed-out",
    #     "OFia3dWgaoI_trimmed-out",
    #     "ofvOXABptcY_trimmed-out",
    #     "ogIbkhMeJFI_trimmed-out",
    #     "OLw7cIJApMI_trimmed-out",
    #     "OMkfujDPpwc_trimmed-out",
    #     "ON0rtQyNS20_trimmed-out",
    #     "ON45DvDMSk4_trimmed-out",
    #     "onbBEbC24SQ_trimmed-out",
    #     "ooUw7Hq8YLg_trimmed-out",
    #     "Op2e8_JURSQ_trimmed-out",
    #     "OPdbdjctx2I_trimmed-out",
    #     "_oqc_t0mbsQ_trimmed-out",
    #     "OqL_u-uGIi0_trimmed-out",
    #     "oRBPxefAHTY_trimmed-out",
    #     "ory7fcHYXhQ_trimmed-out",
    #     "OsQdzsIMFPQ_trimmed-out",
    #     "Ot1aZoxt9PU_trimmed-out",
    #     "OT3MVRv0TT4_trimmed-out",
    #     "OTdFPlXfFj4_trimmed-out",
    #     "otna1VHHCow_trimmed-out",
    #     "oW3UPfXHSUs_trimmed-out",
    #     "OWsT2rkqCk8_trimmed-out"
    # ]
    sequences = [
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
        "sPxoGNvnVzg_trimmed-out"
    ]
    for seq in sequences:
        # print("Smoothing {}".format(seq))
        # # raw_result = np.loadtxt(osp.join(image_root, "results", "tracking", args.detector, seq, "tracks.txt")).astype(int)
        # raw_result = np.loadtxt(
        #     osp.join(image_root, "results", "tracking_openface", args.detector, seq, "tracks.txt"), dtype=str)
        #
        # # smoothed = smooth_tracks(raw_result, args)
        # # debug = np.array([["8", "00", "505", "82", "719", "361", "382", "82", "968", "715"],
        # #                   ["17", "00", "499", "90", "725", "388", "384", "90", "919", "515"],
        # #                   ["18", "00", "499", "90", "725", "388", "384", "90", "919", "515"]]).astype(np.int)
        # smoothed_tracks = smooth_tracks(raw_result, args)
        # visualize_tracks(smoothed_tracks, image_root, seq, args)
        try:
            print("Smoothing {}".format(seq))
            # raw_result = np.loadtxt(osp.join(image_root, "results", "tracking", args.detector, seq, "tracks.txt")).astype(int)
            raw_result = np.loadtxt(
                osp.join(image_root, "results", "tracking_openface", args.detector, seq, "tracks.txt"), dtype=str)

            # smoothed = smooth_tracks(raw_result, args)
            # debug = np.array([["8", "00", "505", "82", "719", "361", "382", "82", "968", "715"],
            #                   ["17", "00", "499", "90", "725", "388", "384", "90", "919", "515"],
            #                   ["18", "00", "499", "90", "725", "388", "384", "90", "919", "515"]]).astype(np.int)
            smoothed_tracks = smooth_tracks(raw_result, args)
            visualize_tracks(smoothed_tracks, image_root, seq, args)
            # visualize_tracks(raw_result, image_root, seq, args)
            # break
        except:
            print("error")
            continue
        # break
