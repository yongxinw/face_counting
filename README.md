# Face counting

Pytorch0.4.1 codes for counting the number of unique faces under a directory containing images.

## 1. Intro

- This repo is heavily borrowed from [Insightface_Pytorch](https://github.com/TreB1eN/InsightFace_Pytorch)
- This code also save the 
identity embeddings of the unique faces found as an ```.npz``` file, where the keys are 
    
    ```
    embeddings: the identity embeddings, PyTorch tensor of shape (num_embeddings, embed_dim)  
    names: the names of the embeddings, array of shape (num_embeddings + 1,) where the first being "Unknown"
    emb_counts: number of occurances of each embeddings, array of shape (num_embeddings,)
    ```
- Pretrained models are listed below. (Note: I used IR-SE50) 

    For InsightFace [IR-SE50 @ Onedrive](https://1drv.ms/u/s!AhMqVPD44cDOhkPsOU2S_HFpY9dC)
    For RetinaFace [RetinaFace-R50](https://www.dropbox.com/s/53ftnlarhyrpkg2/retinaface-R50.zip?dl=0)
    
    | LFW(%) | CFP-FF(%) | CFP-FP(%) | AgeDB-30(%) | calfw(%) | cplfw(%) | vgg2_fp(%) |
    | ------ | --------- | --------- | ----------- | -------- | -------- | ---------- |
    | 0.9952 | 0.9962    | 0.9504    | 0.9622      | 0.9557   | 0.9107   | 0.9386     |
    
    [Mobilefacenet @ BaiduNetDisk](https://pan.baidu.com/s/1hqNNkcAjQOSxUjofboN6qg), [Mobilefacenet @ OneDrive](https://1drv.ms/u/s!AhMqVPD44cDOhkSMHodSH4rhfb5u)
    
    | LFW(%) | CFP-FF(%) | CFP-FP(%) | AgeDB-30(%) | calfw(%) | cplfw(%) | vgg2_fp(%) |
    | ------ | --------- | --------- | ----------- | -------- | -------- | ---------- |
    | 0.9918 | 0.9891    | 0.8986    | 0.9347      | 0.9402   | 0.866    | 0.9100     |

## 2. How to use

### 2.1 Pre-requisites

- clone this repo
  ```
  git clone https://github.com/yongxinw/face_counting.git
  ```

- install requirements by (requires python 3.6)
    ```
    pip install -r requirements.txt
    ```
- set environmental variables
    ```
    export LD_LIBRARY_PATH=<path_to_cuda_installation>/lib64:$LD_LIBRARY_PATH
    export LD_LIBRARY_PATH=<path_to_cuda_installation>:$LD_LIBRARY_PATH
    ```
- download pretrained weights by clicking the links from section 1


### 2.2 Run
#### 2.2.1 RetinaFace for detection
- go to the RetinaFace directory, and compile RetinaFace
    ```
    cd <path_to_this_repo>/RetinaFace
    make
    ```
- run RetinaFace detection
    ```
    python test_folder.py \
    --image_root <directory containing images> \
    --res_root <directory to save detection visualizations>
    ```

#### 2.2.2 InsightFace for verification
- run the following command to count the number of faces under a directory
    ```
    python count_faces.py \
    --result_root <directory to save the .npz file> \
    --retinaface <path to .txt file containing the RetinaFace detection> \ 
    --image_root <directory of the images> \
    --model_path <path to the insightface pretrained model> \
    --verbose
    ```

## 3. References 

- This repo is heavily borrowed from [Insightface_Pytorch](https://github.com/TreB1eN/InsightFace_Pytorch)
- See also [deepinsight/insightface](https://github.com/deepinsight/insightface)