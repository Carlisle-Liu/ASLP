# Model Calibration in Dense Classification with Adaptive Label Perturbation<br>
# <sub><sub>Official Pytorch Implementation</sub></sub><br>
[<a target="_blank" href="https://openaccess.thecvf.com/content/ICCV2023/papers/Liu_Model_Calibration_in_Dense_Classification_with_Adaptive_Label_Perturbation_ICCV_2023_paper.pdf">Paper</a>] [<a href="#bibtex">BibTex</a>]


**[Abstract]** *For safety-related applications, it is crucial to produce trustworthy deep neural networks whose prediction is associated with confidence that can represent the likelihood of correctness for subsequent decision-making. Existing dense binary classification models are prone to being over-confident. To improve model calibration, we propose Adaptive Stochastic Label Perturbation (ASLP) which
learns a unique label perturbation level for each training image. ASLP employs our proposed Self-Calibrating Binary Cross Entropy (SC-BCE) loss, which unifies label perturbation processes including stochastic approaches (like DisturbLabel), and label smoothing, to correct calibration while maintaining classification rates. ASLP follows Maximum Entropy Inference of classic statistical mechanics to maximise prediction entropy with respect to missing information. It performs this while: (1) preserving classification accuracy on known data as a conservative solution, or (2) specifically improves model calibration degree by minimising the gap between the prediction accuracy and expected confidence of the target training label Extensive results demonstrate that ASLP can significantly improve calibration degrees of dense binary classification models on both in-distribution and out-of-distribution data.*






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

