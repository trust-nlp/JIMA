# Joint Imbalance Adaptation for Radiology Report Generation (JIMA)

Source codes for our paper "Joint Imbalance Adaptation for Radiology Report Generation" which addresses the data imbalance challenge in medical report generation.

## Data Imbalance Challenge

![Data Imbalance Effects](git_images/label_im_performance.png)

Radiology report generation faces two critical imbalance challenges:
- **Token Imbalance**: Medical tokens appear less frequently than regular tokens, but contain crucial clinical information
- **Label Imbalance**: Normal cases dominate datasets (>85% in MIMIC-CXR), leading to poor performance on abnormal cases

This causes models to overfit on frequent patterns while underperforming on rare but clinically important cases.

## JIMA: A Joint Imbalance Adaptation Approach

We propose **J**oint **Im**balance **A**daptation (JIMA), a curriculum learning-based approach that:

### Key Innovation: Dual-Task Architecture
1. **Task 1**: Entity prediction from images to handle label imbalance
2. **Task 2**: Report generation from images and entity distributions to handle token imbalance

### Core Components
- **Difficulty Measurer**: Novel ranking-based scoring mechanism using equation:
  ```
  k = Rank(p, p[z]) / |V|
  ```
- **Training Scheduler**: Adaptive sample selection based on performance changes:
  ```
  c(s_t) = min(1, [1-(s_t-s_{t-1})/s_{t-1}] × c(s_{t-1}))
  ```
  
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
   pip install torch torchvision
   pip install numpy pandas tqdm
   pip install transformers
   ```

2. **Data Preparation**:
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

## Data Analysis

### Imbalance Statistics
| Dataset | Abnormal % | Normal % | Avg Length | Medical Token Ratio |
|---------|------------|----------|------------|-------------------|
| IU X-ray | 32.96% | 67.04% | 35.99 | Higher in abnormal |
| MIMIC-CXR | 13.97% | 86.03% | 59.70 | Higher in abnormal |

### Key Findings
1. **Token Distribution**: Top 12.5% frequent tokens account for >80% of all tokens
2. **Medical Relevance**: Infrequent tokens contain higher ratios of medical terminology
3. **Length Correlation**: Abnormal reports are significantly longer and more complex

## Baselines

We compare against state-of-the-art methods:
- **R2Gen**: Transformer-based with relational memory
- **CMN**: Cross-modal memory network
- **WCL**: Weakly-supervised contrastive learning
- **CMM+RL**: Cross-modal memory with reinforcement learning
- **RRG**: RadGraph-based reinforcement learning
- **TIMER**: Token imbalance-aware method
- **RGRG**: GPT2-based generation model

## Ablation Analysis

Our curriculum learning strategy shows:
- **Minimal impact** on highly frequent tokens and labels
- **Significant improvement** (6.49% average) on moderately frequent samples
- **Limited enhancement** for extremely rare tokens
- **Consistent gains** across different imbalance severity levels

## Code Availability

The complete implementation is available at: [https://github.com/trust-nlp/JIMA/](https://github.com/trust-nlp/JIMA/)

This repository includes:
- Model implementation (`models/`, `modules/`)
- Training scripts (`main_train.py`)
- Evaluation tools (`main_plot.py`)
- Data processing utilities
- Comprehensive documentation

## Data Availability

- **MIMIC-CXR**: Available through PhysioNet ([link](https://physionet.org/content/mimic-cxr/2.0.0/))
- **IU X-ray**: Available through OpenI ([link](https://openi.nlm.nih.gov/faq#collection))

Both datasets require appropriate data use agreements and ethical compliance.

## Contact

For questions, requests, or collaborations, please contact:
- **Xiaolei Huang**: [xiaolei.huang@memphis.edu](mailto:xiaolei.huang@memphis.edu)
- **Wang Li**: [wli5@memphis.edu](mailto:wli5@memphis.edu)

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

## Acknowledgments

This work was supported by:
- National Science Foundation Award IIS-2245920
- National Cancer Institute Award R01CA258193

## License

This project is licensed under the MIT License - see the LICENSE file for details.

---

**Note**: This research contributes to the advancement of AI-assisted medical diagnosis and report generation. The methods developed here aim to improve healthcare quality while addressing critical data imbalance challenges in medical AI systems.