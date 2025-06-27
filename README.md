# Joint Imbalance Adaptation for Radiology Report Generation (JIMA)

Source codes for our paper "Joint Imbalance Adaptation for Radiology Report Generation" which addresses the data imbalance challenge in medical report generation.

## Citation

Please cite our work as:

```bibtex
@article{li2024jima,
  title={Joint Imbalance Adaptation for Radiology Report Generation},
  author={Li, Wang and Han, Guangzeng and Wu, Yuexin and Huang, I-Chan and Huang, Xiaolei},
  journal={[Journal Name]},
  year={2024},
  note={Under Review}
}
```

## Data Imbalance Challenge

![Data Imbalance Effects](git_images/label_im_performance.png)

Radiology report generation faces two critical imbalance challenges:
- **Token Imbalance**: Medical tokens appear less frequently than regular tokens, but contain crucial clinical information
- **Label Imbalance**: Normal cases dominate datasets (>85% in MIMIC-CXR), leading to poor performance on abnormal cases

This causes models to overfit on frequent patterns while underperforming on rare but clinically important cases.

## JIMA: A Joint Imbalance Adaptation Approach

We propose **J**oint **Im**balance **A**daptation (JIMA), a curriculum learning-based approach:
  
## Method Overview

![JIMA Architecture](./git_images/crop_acl2024_overview.png)

JIMA employs a two-stage curriculum learning approach:
- **Entity Distribution Prediction**: Extracts clinical entities to guide report generation
- **Joint Feature Fusion**: Cross-concatenation and element-wise multiplication of image and entity features
- **Adaptive Training**: Dynamic sample selection based on difficulty assessment

## Experimental Results

### Key Performance Gains
- **IU X-ray**: 16.75%-50.50% average improvement, 72.10% clinical F1 improvement
- **MIMIC-CXR**: 9.59%-16.26% average improvement, 31.29% clinical F1 improvement
- **Imbalance Handling**: Significant improvements on low-frequency tokens and abnormal cases
- **Human Evaluation**: Medical experts prefer JIMA for clinical accuracy (32 vs 21 votes overall)

## Test Platform
Python 3.7+, PyTorch 1.8+, CUDA-enabled GPU recommended

## Experiment Preparation
1. **Environment Setup**:
   ```bash
   aiohappyeyeballs==2.6.1
   aiohttp==3.11.14
   aiosignal==1.3.2
   async-timeout==5.0.1
   attrs==25.3.0
   certifi @ file:///home/conda/feedstock_root/build_artifacts/certifi_1739515848642/work/certifi
   charset-normalizer==3.4.1
   datasets==3.4.1
   dill==0.3.8
   filelock==3.18.0
   frozenlist==1.5.0
   fsspec==2024.12.0
   huggingface-hub==0.29.3
   idna==3.10
   Jinja2==3.1.6
   joblib==1.4.2
   MarkupSafe==3.0.2
   mpmath==1.3.0
   multidict==6.2.0
   multiprocess==0.70.16
   networkx==3.4.2
   numpy==2.2.4
   nvidia-cublas-cu12==12.4.5.8
   nvidia-cuda-cupti-cu12==12.4.127
   nvidia-cuda-nvrtc-cu12==12.4.127
   nvidia-cuda-runtime-cu12==12.4.127
   nvidia-cudnn-cu12==9.1.0.70
   nvidia-cufft-cu12==11.2.1.3
   nvidia-curand-cu12==10.3.5.147
   nvidia-cusolver-cu12==11.6.1.9
   nvidia-cusparse-cu12==12.3.1.170
   nvidia-cusparselt-cu12==0.6.2
   nvidia-nccl-cu12==2.21.5
   nvidia-nvjitlink-cu12==12.4.127
   nvidia-nvtx-cu12==12.4.127
   opencv-python==4.11.0.86
   packaging==24.2
   pandas==2.2.3
   pillow==11.1.0
   propcache==0.3.1
   pyarrow==19.0.1
   python-dateutil==2.9.0.post0
   pytz==2025.2
   PyYAML==6.0.2
   regex==2024.11.6
   requests==2.32.3
   safetensors==0.5.3
   scikit-learn==1.6.1
   scipy==1.15.2
   six==1.17.0
   sympy==1.13.1
   threadpoolctl==3.6.0
   tokenizers==0.21.1
   torch==2.6.0
   torchvision==0.21.0
   tqdm==4.67.1
   transformers==4.50.1
   triton==3.2.0
   typing_extensions==4.13.0
   tzdata==2025.2
   urllib3==2.3.0
   xxhash==3.5.0
   yarl==1.18.3
   ```

2. **Data Availability**:
   - Download IU X-ray dataset from [OpenI](https://openi.nlm.nih.gov/)
   - Download MIMIC-CXR dataset from [PhysioNet](https://physionet.org/content/mimic-cxr/)
   - Follow data preprocessing steps in `modules/datasets.py`

3. **Model Training**:
   ```bash
   # Joint training (recommended)
   python main_train.py --joint --dataset_name iu_xray --epochs 100
   
   # Alternating training
   python main_train.py --dataset_name iu_xray --epochs 100
   ```

4. **Evaluation**:
   ```bash
   python main_plot.py --dataset_name iu_xray --model_path results/best_model.pth
   ```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

---
