#!/usr/bin/env bash
#for seq in "qDpGgd4oTQ8_trimmed-out" "c67D5bP0Hg4_trimmed-out" "pMGhqE76kQA_trimmed-out" "mA402F5K47o_trimmed-out" "SAgYiERRDPY_trimmed-out" "L49M8C4wjVU_trimmed-out" "UUukBV82P9A_trimmed-out" "ztlOyCk5pcg_trimmed-out"
#do
#    ffmpeg -i $seq/openface/%06d.jpg processed/$seq.avi
#done

python track_faces_retinaface_with_openface.py --detector retinaface
#python smooth_tracks_iou.py

#for seq in "c67D5bP0Hg4_trimmed-out" "DZsBei4nCkU_trimmed-out" "g8D-LyfTrRs_trimmed-out" "L49M8C4wjVU_trimmed-out" "pMGhqE76kQA_trimmed-out" "qDpGgd4oTQ8_trimmed-out" "urYGhhMOToU_trimmed-out" "ztlOyCk5pcg_trimmed-out" "ZYzql-Y1sP4_trimmed-out"
#do
#    ffmpeg -i /work/yongxinw/SocialIQ/raw/raw/vision/results/tracking_openface/retinaface/$seq/%06d_smoothed.jpg /work/yongxinw/SocialIQ/raw/raw/vision/results/tracking_openface/retinaface/$seq/$seq_smoothed.mp4
#done