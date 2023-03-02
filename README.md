<div align="center">

<h2> 😭 SadTalker： <span style="font-size:12px">Learning Realistic 3D Motion Coefficients for  Stylized Audio-Driven Single Image Talking Face Animation </span> </h2> 

  <a href='https://arxiv.org/abs/2211.12194'><img src='https://img.shields.io/badge/ArXiv-2211.14758-red'></a> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a href='https://sadtalker.github.io'><img src='https://img.shields.io/badge/Project-Page-Green'></a>

<div>
    <a target='_blank'>Wenxuan Zhang <sup>*,1,2</sup> </a>&emsp;
    <a href='https://vinthony.github.io/' target='_blank'>Xiaodong Cun <sup>*,2</a>&emsp;
    <a href='https://xuanwangvc.github.io/' target='_blank'>Xuan Wang <sup>2</sup></a>&emsp;
    <a href='https://yzhang2016.github.io/' target='_blank'>Yong Zhang <sup>2</sup></a>&emsp;
    <a href='https://xishen0220.github.io/' target='_blank'>Xi Shen <sup>2</sup></a>&emsp; </br>
    <a href='https://yuguo-xjtu.github.io/' target='_blank'>Yu Guo<sup>1</sup> </a>&emsp;
    <a target='_blank'>Ying Shan <sup>2</sup> </a>&emsp;
    <a target='_blank'>Fei Wang <sup>1</sup> </a>&emsp;
</div>
<br>
<div>
    <sup>1</sup> Xi'an Jiaotong University &emsp; <sup>2</sup> Tencent AI Lab &emsp; 
</div>
<br>
<i><strong><a href='https://arxiv.org/abs/2211.12194' target='_blank'>CVPR 2023</a></strong></i>
<br>
<br>

![sadtalker](https://user-images.githubusercontent.com/4397546/222490039-b1f6156b-bf00-405b-9fda-0c9a9156f991.gif)

TL;DR: A realistic and stylized talking head video generation method from a single image and audio

<br>

</div>


## Changelog
- 2023.03.03 Release the test code for audio-driven single image animation!
- 2023.02.28 SadTalker has been accepted by CVPR 2023!


## Pipeline
![main_of_sadtalker](https://user-images.githubusercontent.com/4397546/222490596-4c8a2115-49a7-42ad-a2c3-3bb3288a5f36.png)


## **TODO**

- [x] Generating 2D face from a single Image.
- [ ] Generating 3D face from Audio.
- [x] Generating 4D free-view talking examples from audio and a single image.
- [ ] Gradio/Colab Demo.
- [ ] integrade with stable-diffusion-web-ui.
- [ ] training code of each componments.




## **Test**

#### **Requirements**

  * Python >= 3.6
  * PyTorch  
  * ffmpeg

#### **Conda Installation**

```
git clone https://github.com/Winfredy/SadTalker.git
cd SadTalker 
cd test
conda create -n sadtalker python=3.7
source activate sadtalker
pip install torch==1.12.0+cu113 torchvision==0.13.0+cu113 torchaudio==0.12.0 --extra-index-url https://download.pytorch.org/whl/cu113
conda config --add channels conda-forge
conda install ffmpeg
pip install ffmpy
pip install Cmake
pip install boost
conda install dlib
pip install -r requirements.txt
```  

#### **Models**

Please download our [pre-trained model](https://drive.google.com/drive/folders/1Wd88VDoLhVzYsQ30_qDVluQr_Xm46yHT?usp=sharing) and put it in ./checkpoints.

| Model | Description
| :--- | :----------
|checkpoints/auido2exp_00300-model.pth | Pre-trained ExpNet in Sadtalker.
|checkpoints/auido2pose_00140-model.pth | Pre-trained PoseVAE in Sadtalker.
|checkpoints/mapping_00229-model.pth.tar | Pre-trained MappingNet in Sadtalker.
|checkpoints/facevid2vid_00189-model.pth.tar | Pre-trained face-vid2vid model from [the reappearance of face-vid2vid](https://github.com/zhanglonghao1992/One-Shot_Free-View_Neural_Talking_Head_Synthesis).
|checkpoints/epoch_20.pth | Pre-trained 3DMM extractor in [Deep3DFaceReconstruction](https://github.com/microsoft/Deep3DFaceReconstruction).
|checkpoints/wav2lip.pth | Highly accurate lip-sync model in [Wav2lip](https://github.com/Rudrabha/Wav2Lip).
|checkpoints/shape_predictor_68_face_landmarks.dat | Face landmark model used in [dilb](http://dlib.net/). 
|checkpoints/BFM | 3DMM library file.  
|checkpoints/hub | Face detection models used in [face alignment](https://github.com/1adrianb/face-alignment).

#### **Generating 2D face from a single Image**

```
python inference.py --driven_audio <audio.wav> --source_image <video.mp4 or picture.png> --result_dir <a file to store results>
```

#### **Generating 3D face from Audio**

To do ...

#### **Generating 4D free-view talking examples from audio and a single image**

We use `camera_yaw`, `camera_pitch`, `camera_roll` to control camera pose. For example, `--camera_yaw -20 30 10` means the camera yaw degree changes from -20 to 30 and then changes from 30 to 10.
```
python inference.py --driven_audio <audio.wav> \
                    --source_image <video.mp4 or picture.png> \
                    --result_dir <a file to store results> \
                    --camera_yaw -20 30 10
```




## **Citation**

If you find our work useful in your research, please consider citing:

```
@article{zhang2022sadtalker,
  title={SadTalker: Learning Realistic 3D Motion Coefficients for Stylized Audio-Driven Single Image Talking Face Animation},
  author={Zhang, Wenxuan and Cun, Xiaodong and Wang, Xuan and Zhang, Yong and Shen, Xi and Guo, Yu and Shan, Ying and Wang, Fei},
  journal={arXiv preprint arXiv:2211.12194},
  year={2022}
}
```

Acknowledgements
----------
Facerender code borrows heavily from [zhanglonghao](https://github.com/zhanglonghao1992/One-Shot_Free-View_Neural_Talking_Head_Synthesis) and [PIRender](https://github.com/RenYurui/PIRender). We thank the author for sharing their wonderful code. In training process, We also use the model from [Deep3DFaceReconstruction](https://github.com/microsoft/Deep3DFaceReconstruction) and [Wav2lip](https://github.com/Rudrabha/Wav2Lip). We thank for their wonderful work.

