# Underwater Signal Classifier
In this repository, we present two new deep learning architectures
based on spatio-temporal modeling for underwater signal classification.
And tested on two real datasets: DeepShip and ShipsEar.

* The first architecture is based on a static feature extraction (MFCC features).
* The second one, is an improvement of the first by adding a convolutional bloc to generate an artificial spectrum and makes the network takes the original signal as input directly.

## About the model
* `model.py`: DL models definition in Keras
* `params.py`: Hyperparameters
* `features.py`: Dataset preparation and features extraction
* `inference.py`: Testing the model
* `file.wav`: wav file for test

## Results

### Results obtained on ShipsEar data-set compared with other architectures.
| **Model**            | **Accuracy** | **Precision** | **Recall** | **F1-score** |
|----------------------|--------------|---------------|------------|--------------|
| **Hybrid model**     | 0.9878       | 0.9839        | 0.9949     | 0.9878       |
| **End-to-end model** | 0.9856       | 0.9730        | 0.9964     | 0.9843       |

### Results obtained on DeepShip data-set compared with other architectures.
| **Model**            | **Accuracy** | **Precision** | **Recall** | **F1-score** |
|----------------------|--------------|---------------|------------|--------------|
| **Hybrid model**     | 0.9727       | 0.9721        | 0.9731     | 0.9725       |
| **End-to-end model** | 0.9107       | 0.9010        | 0.9103     | 0.9097       |

## Citation
If you use this code in your research, please consider citing this work via the following:

**`Plain Text`**
Z. Alouani, Y. Hmamouche, B. E. Khamlichi and A. E. F. Seghrouchni, "A Spatio-temporal Deep Learning Approach for Underwater Acoustic Signals Classification," 2022 18th IEEE International Conference on Advanced Video and Signal Based Surveillance (AVSS), 2022, pp. 1-7, doi: 10.1109/AVSS56176.2022.9959247.

**`BibTex`**
@INPROCEEDINGS{9959247,  author={Alouani, Zakaria and Hmamouche, Youssef and Khamlichi, Btissam El and Seghrouchni, Amal El Fallah},  booktitle={2022 18th IEEE International Conference on Advanced Video and Signal Based Surveillance (AVSS)},   title={A Spatio-temporal Deep Learning Approach for Underwater Acoustic Signals Classification},   year={2022},  volume={},  number={},  pages={1-7},  doi={10.1109/AVSS56176.2022.9959247}}
