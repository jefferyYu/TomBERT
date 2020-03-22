# TomBERT

## Adapting BERT for Target-Oriented Multimodal Sentiment Classification
- Dataset and codes for our IJCAI 2019 paper "Adapting BERT for Target-Oriented Multimodal Sentiment Classification"
https://www.ijcai.org/proceedings/2019/0751.pdf

Author

Jianfei YU

jfyu@njust.edu.cn

Mar 02, 2020

> Target-oriented Multimodal Sentiment Classification (TMSC), PyTorch Implementations.

## Requirement

* PyTorch 1.0.0
* Python 3.7


## Download tweet images and set up image path
- Step 1: Download each tweet's associated image via this link (https://drive.google.com/file/d/1PpvvncnQkgDNeBMKVgG2zFYuRhbL873g/view)
- Step 2: Change the image path in line 553 and line 555 of the "run_multimodal_classifier.py" file
- Step 3: Download the pre-trained ResNet-152 via this link (https://download.pytorch.org/models/resnet152-b121ed2d.pth)
- Setp 4: Put the pre-trained ResNet-152 model under the folder named "resnet"



## Code Usage

### (Optional) Preprocessing
- This is optional, because I have provided the pre-processed data under the folder named "absa_data"

```sh
python process_absa_data.py
```

### Training for TomBERT
- This is the training code of tuning parameters on the dev set, and testing on the test set. Note that you can change "CUDA_VISIBLE_DEVICES=6" based on your available GPUs.

```sh
sh run_multimodal_classifier.sh
```

### Testing for TomBERT
- After training the model, the following code is used for directly loading the trained model and testing it on the test set

```sh
sh run_multimodal_classifier_test.sh
```


## Implemented models

### BERT and BERT+BL ([run_classifier.py](./run_classifier.py))
- You can run the following code to perform training and testing.

```sh
sh run_classifier.sh
```

### TomBERT, mBERT, Res-BERT ([run_multimodal_classifier.py](./run_multimodal_classifier.py))
- You can choose different models in the "run_multimodal_classifier.sh" file.

### BERT and TomBERT trained by me
- You can download the BERT and TomBERT models trained by me. You can find the results we report in our paper from the "eval_result" files.
https://drive.google.com/open?id=1e3rL3G1ojaDWZnrkmZX-uLudPbQo7tVe

## Acknowledgements

- Most of the codes are based on the codes provided by huggingface: https://github.com/huggingface/transformers.
