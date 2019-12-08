# Created by yongxinwang at 2019-10-13 21:05
import os.path as osp
import os
import numpy as np


def run_openpose(image_path, openpose_dir):
    pass


if __name__ == "__main__":
    image_root = "/work/yongxinw/SocialIQ/raw/raw/vision/"
    all_sequences = np.array(os.listdir(osp.join(image_root, "raw")))
    sequences = all_sequences[np.random.choice(len(all_sequences), size=10, replace=False)]
    # sequences = ['UUukBV82P9A_trimmed-out']
    # sequences = ["UUukBV82P9A_trimmed-out.mp4",
    #              "uB3FdWmnFZU_trimmed-out.mp4",
    #              "ztlOyCk5pcg_trimmed-out.mp4",
    #              "ZYzql-Y1sP4_trimmed-out.mp4",
    #              "zzERcgFrTrA_trimmed-out.mp4",
    #              "zzWaf1z9kgk_trimmed-out.mp4"]
    sequences = ["urYGhhMOToU_trimmed-out",
                 "iBL0FcUTFT8_trimmed-out",
                 "DZsBei4nCkU_trimmed-out",
                 "g8D-LyfTrRs_trimmed-out",
                 "qDpGgd4oTQ8_trimmed-out",
                 "c67D5bP0Hg4_trimmed-out",
                 "pMGhqE76kQA_trimmed-out",
                 "mA402F5K47o_trimmed-out",
                 "SAgYiERRDPY_trimmed-out",
                 "L49M8C4wjVU_trimmed-out"]
    print(sequences)
    for seq in sequences:
        print("Processing {}".format(seq))
        image_dir = osp.join(image_root, "images", seq.replace(".mp4", ""))
        openpose_dir = osp.join("/home", "yongxinw", "data", "images", seq.replace(".mp4", ""))

        # extract video to frames and copy to openpose dir
        os.makedirs(image_dir, exist_ok=True)
        video_path = osp.join(image_root, "raw", "{}".format(seq))
        os.system("ffmpeg -i {} -q:v 1 -qmin 1 {}/%06d.jpg".format(video_path, image_dir))

        # copy to home dir
        os.makedirs(openpose_dir, exist_ok=True)
        os.system("cp -r {}/*.jpg {}".format(image_dir, openpose_dir))

            # os.system("cd /home/yongxinw/openpose/")
            # os.system("./build/examples/openpose/openpose.bin --image_dir {} "
            #           "--face --hand --render_pose 0 --display 0 "
            #           "--write_json {}".format(openpose_dir, openpose_dir))
