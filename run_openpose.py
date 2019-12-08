# Created by yongxinwang at 2019-10-13 21:05
import os.path as osp
import os
import numpy as np


def run_openpose(image_path, openpose_dir):
    pass


if __name__ == "__main__":
    image_root = "/work/yongxinw/SocialIQ/raw/raw/vision/"
    # all_sequences = np.array(os.listdir(osp.join(image_root, "raw")))
    # sequences = all_sequences[np.random.choice(len(all_sequences), size=10, replace=False)]
    # sequences = [
    #     # "UUukBV82P9A_trimmed-out.mp4",
    #     "uB3FdWmnFZU_trimmed-out.mp4",
    #     "ztlOyCk5pcg_trimmed-out.mp4",
    #     "ZYzql-Y1sP4_trimmed-out.mp4",
    #     "zzERcgFrTrA_trimmed-out.mp4",
    #     "zzWaf1z9kgk_trimmed-out.mp4"
    # ]
    # sequences = ["UUukBV82P9A_trimmed-out.mp4"]

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

    for seq in sequences:
        # image_dir = osp.join(image_root, "images", seq)
        openpose_dir = osp.join("/home", "yongxinw", "data", "images", seq.replace(".mp4", ""))

        os.chdir("/opt/openpose/")
        # os.system("cd /home/yongxinw/openpose/")
        os.system("./build/examples/openpose/openpose.bin --image_dir {} "
                  "--face --hand --body --render_pose 0 --display 0 "
                  "--write_json {}".format(openpose_dir, openpose_dir))
