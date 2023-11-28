# Model Calibration in Dense Classification with Adaptive Label Perturbation<br><sub><sub>Official Pytorch Implementation</sub></sub>
[<a target="_blank" href="https://openaccess.thecvf.com/content/ICCV2023/papers/Liu_Model_Calibration_in_Dense_Classification_with_Adaptive_Label_Perturbation_ICCV_2023_paper.pdf">Paper</a>] [<a href="#bibtex">BibTex</a>]


**[Abstract]** *For safety-related applications, it is crucial to produce trustworthy deep neural networks whose prediction is associated with confidence that can represent the likelihood of correctness for subsequent decision-making. Existing dense binary classification models are prone to being over-confident. To improve model calibration, we propose Adaptive Stochastic Label Perturbation (ASLP) which
learns a unique label perturbation level for each training image. ASLP employs our proposed Self-Calibrating Binary Cross Entropy (SC-BCE) loss, which unifies label perturbation processes including stochastic approaches (like DisturbLabel), and label smoothing, to correct calibration while maintaining classification rates. ASLP follows Maximum Entropy Inference of classic statistical mechanics to maximise prediction entropy with respect to missing information. It performs this while: (1) preserving classification accuracy on known data as a conservative solution, or (2) specifically improves model calibration degree by minimising the gap between the prediction accuracy and expected confidence of the target training label Extensive results demonstrate that ASLP can significantly improve calibration degrees of dense binary classification models on both in-distribution and out-of-distribution data.*


## Environment

- python 3.8.12
- cuda 11.3
- pytorch 1.11.0
- torchvision 0.12.0


## Prepare the Data
Download the training dataset: <a target="_blank" href="https://www.kaggle.com/datasets/balraj98/duts-saliency-detection-dataset">DUTS-TR</a> and the six SOD testing datasets: <a target="_blank" href="https://www.kaggle.com/datasets/balraj98/duts-saliency-detection-dataset">DUTS-TE</a> (same link as DUTS-TR), <a target="_blank" href="http://saliencydetection.net/dut-omron/">DUT-OMRON</a>, <a target="_blank" href="http://cbi.gatech.edu/salobj/">PASCAL-S</a>, <a target="_blank" href="https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/">SOD</a>, <a target="_blank" href="https://www.elderlab.yorku.ca/resources/salient-objects-dataset-sod/">HKU-IS</a> and <a target="_blank" href="https://www.cse.cuhk.edu.hk/leojia/projects/hsaliency/dataset.html">ECSSD</a>. The 500 Out-of-Distribution texture images are selected from the <a target="_blank" href="https://www.robots.ox.ac.uk/~vgg/data/dtd/">Describable Texture Dataset</a>.

**The structure of dataset directory is illustrated as below:**
```
├── Dataset
│   ├── DUTS-TR-Train.txt
│   ├── DUTS-TR-Validation.txt
│   ├── Train
│   │   ├── DUTS-TR
│   │   │   ├── Image
│   │   │   ├── GT
│   ├── Test
│   │   ├── DUTS-TE
│   │   ├── DUT-OMRON
│   │   ├── PASCAL-S
│   │   ├── SOD
│   │   ├── HKU-IS
│   │   ├── ECSSD
│   │   ├── DTD_Texture_500
```
The subdirectory structure of the testing dataset follows that of the training dataset.


## Train, Test and Evaluate
To train, test and evaluate the model consecutively, run the following line of code:
```
CUDA_VISIBLE_DEVICES=GOU_ID python main_ASLP.py
```


## Pretrained Model
Weight and predicted results of the pretrained model can be downloaded from the [<a target="_blank" href="https://drive.google.com/drive/folders/1Wjn9tdAj4v69zfFc9lqq8xnJ4arXxQnd?usp=share_link">Google Drive</a>].


## <a name="bibtex">Citing ASLP</a>

If you find this work useful to your research, please consider citing it:

```BibTex
@InProceedings{Liu_2023_ICCV,
    author    = {Liu, Jiawei and Ye, Changkun and Wang, Shan and Cui, Ruikai and Zhang, Jing and Zhang, Kaihao and Barnes, Nick},
    title     = {Model Calibration in Dense Classification with Adaptive Label Perturbation},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    month     = {October},
    year      = {2023},
    pages     = {1173-1184}
}
```

