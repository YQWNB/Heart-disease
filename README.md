# AttentionConvLSTM
运行：
先下载数据（尽量下载到此文件夹，如果没有，需要修改loadtxt文件的root变量为training文件夹所在的地址并运行，以得到data.txt文件，其包含所有数据的信息）

打开文件training_res3d_aclstm_mobilenet.py，运行该文件。
如果报错，可能是版本不一致，下面的项目介绍中有提到。需要TensorFlow-1.2
然后需要将补丁（patchs文件夹）里面的文件复制到相应的目录下面（里面有readme）。
应该就不会报错了

我设置的nb_epoch=1只训练了一次可以在training_res3d_aclstm_mobilenet.py里面修改。

以下是原项目介绍
## Prerequisites
1) Python 2.7
2) Tensorflow-1.2 <br/>
3) The implementation files of the variants of ConvLSTM are in the local dir "patchs". You need merge them with the corresponding files of TF-1.2. <br/> <br/>
   
## Get the pretrained models
The trained models can be obtained from the below link:  <br/>
    Link: https://pan.baidu.com/s/1O-U_Q-5i9wxOA0MDyi3Idg Code: immi

## How to use the code
### Prepare the data
1) Convert each video files into images.
2) Replace the path "/ssd/dataset" in the files under "dataset_splits" 
### Training 
1) Use training_*.py to train the networks for different datasets and different modalities. <br/>
### Testing 
1) Use testing_*.py to evaluate the trained networks on the valid or test subsets of Jester or IsoGD. <br/>

## Citation
Please cite the following paper if you feel this repository useful. <br/>
http://papers.nips.cc/paper/7465-attention-in-convolutional-lstm-for-gesture-recognition
http://openaccess.thecvf.com/content_ICCV_2017_workshops/w44/html/Zhang_Learning_Spatiotemporal_Features_ICCV_2017_paper.html
http://ieeexplore.ieee.org/abstract/document/7880648/
```
@article{ZhuNIPS2018,
  title={Attention in Convolutional LSTM for Gesture Recognition},
  author={Liang Zhang and Guangming Zhu and Lin Mei and Peiyi Shen and Syed Afaq Shah and Mohammed Bennamoun},
  journal={NIPS},
  year={2018}
}
@article{ZhuICCV2017,
  title={Learning Spatiotemporal Features using 3DCNN and Convolutional LSTM for Gesture Recognition},
  author={Liang Zhang and Guangming Zhu and Peiyi Shen and Juan Song and Syed Afaq Shah and Mohammed Bennamoun},
  journal={ICCV},
  year={2017}
}
@article{Zhu2017MultimodalGR,
  title={Multimodal Gesture Recognition Using 3-D Convolution and Convolutional LSTM},
  author={Guangming Zhu and Liang Zhang and Peiyi Shen and Juan Song},
  journal={IEEE Access},
  year={2017},
  volume={5},
  pages={4517-4524}
}
```

## Contact
For any question, please contact
```
  gmzhu@xidian.edu.cn
```

