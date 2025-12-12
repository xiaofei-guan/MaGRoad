<h1 align="center">Beyond Endpoints: Path-Centric Reasoning for Vectorized Off-Road Network Extraction</h1>

<div align="center">

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Framework: PyTorch](https://img.shields.io/badge/Framework-PyTorch-orange.svg)](https://pytorch.org/)

</div>

<div align="center">

[**[ArXiv Paper]**](#) | [**[Interactive Tool]**](#) | [**[WildRoad Dataset]**](#)

</div>

<div align="center">
  <img src="assets/author.png" width="100%" alt="MagRoad Results"/>
</div>


## 📖 Introduction

**MagRoad** introduces a novel *path-centric* reasoning approach for extracting vectorized road networks, specifically designed to handle the challenges of off-road and wild environments. Unlike *node-centric* methods (sam_road series), our model focuses on the connectivity and topology of paths, enabling robust extraction in complex terrains.



<div align="center">
  <img src="assets/intro.png" width="75%" alt="MagRoad Results"/>
</div>

*Node-centric* models suffer from path ambiguity due to sparse features, *path-centric* sampling resolves path ambiguity by leveraging evidence along the entire edge.

**Key Features:**

- 🤖Robust Extraction Pipeline: Path-centric reasoning for reliability in complex terrains.

- 🖱️Interactive Annotation: Lightweight tool for faster, low-effort annotation and refinement.
  
- 🌍 WildRoad Dataset: A new benchmark for challenging scenarios (To be released).

## 📊 Demo

### 🤖 Automated Extraction

<div align="center">
  <img src="assets/demo.png" width="85%" alt="MagRoad Results"/>
</div>

MaGRoad demonstrates effective road extraction results across four diverse datasets: City-Scale, Global-Scale, SpaceNet, and WildRoad.

### 🖱️ Interactive Annotation Tool


| **Manual Annotation (QGIS)** | **Interactive Annotation (Ours)** |
|:----------------------------:|:---------------------------------:|
| <img src="assets/qgis_manual.gif" width="91%" alt="Manual"/> | <img src="assets/interactive.gif" width="100%" alt="Interactive"/> |

To address the bottleneck of creating large-scale vectorized datasets, we developed the **first interactive road extraction algorithm** and integrated it into a seamless **Web Application**. It transforms the workflow: instead of tedious manual plotting (e.g., QGIS), our model intelligently automates path connectivity based on sparse user clicks, drastically reducing annotation time.

> 📢 **Open Source:** The full codebase for both the interactive algorithm and the annotation tool will be released to the community.

---

## 🛠️ Quick Start

1. **Clone the repository**
   ```bash
   git clone https://github.com/xiaofei-guan/MaGRoad.git
   cd MaGRoad
   ```

2. **Environment Setup**
   ```bash
   conda create -n magroad python=3.8 # we use python 3.8.19
   conda activate magroad
   pip install -r requirements.txt
   ```
   *Note: Ensure you have PyTorch and CUDA installed compatible with your system.*

### 📥 Model Preparation

Please download the **ViT-B** checkpoint from the [official SAM repository](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth) and place it under `sam/ckpt`:

```bash
mkdir -p sam/ckpt
# Save sam_vit_b_01ec64.pth here
```

### 📂 Data Preparation

> **Note for WildRoad and Global-Scale:**
> This branch (`cityspace`) is dedicated to reproducing results on standard benchmarks. For **WildRoad** and **Global-Scale** datasets, please refer to the `main` branch.

**Data Preparation for City-Scale & SpaceNet:**

1. **Download Datasets**
   Follow instructions in [RNGDet++](https://github.com/TonyXuQAQ/RNGDetPlusPlus) or use these direct links:
   - **SpaceNet**: [Download Link](https://drive.google.com/uc?id=1FiZVkEEEVir_iUJpEH5NQunrtlG0Ff1W)
   - **City-Scale**: [Download Link](https://drive.google.com/uc?id=1R8sI1RmFe3rUfWMQaOfsYlBDHpQxFH-H)

2. **Organize Directory Structure**
   Ensure your data directory looks like this:
   ```
   MagRoad/
   ├── cityscale/
   │   └── 20cities/
   ├── spacenet/
   │   └── RGB_1.0_meter/
   ```

3. **Generate Labels**
   Run the generation script inside both directories:
   ```bash
   # In cityscale/
   cd cityscale
   python generate_labes.py

   # In spacenet/
   cd spacenet
   python generate_labes.py
   ```

*For more details on data preparation, please refer to the original [sam_road](https://github.com/htcr/sam_road) repository.*

## 🚀 Usage

### 1. Training
Train the model with the specified configuration.
```bash
python train.py --config=config/toponet_vitb_512_cityscale.yaml
# Example with resume
python train.py --config=config/toponet_vitb_512_cityscale.yaml --resume=<your_ckpt_path>
```

### 2. Threshold
Compute the mask and topo connectivity threshold.
```bash
python test.py --config=config/toponet_vitb_512_cityscale.yaml --checkpoint=<your_ckpt_path>
```

### 3. Inference
Update mask and topo threshold
- ITSC_THRESHOLD: 0.133
  
- ROAD_THRESHOLD: 0.839
  
- TOPO_THRESHOLD: 0.373

```bash
python inferencer.py --config=config/toponet_vitb_512_cityscale.yaml --checkpoint=<your_ckpt_path>
```

### 4. Evaluation
Follow the evaluation process from [sam_road](https://github.com/htcr/sam_road):
Navigate to the evaluation directory
```bash
cd cityscale_metrics  # or cd spacenet_metrics
```
Run the evaluation script
```bash
bash eval_schedule.bash
```

The script automatically runs both APLS and TOPO metrics
Evaluation scores will be saved to your output directory


💡 Tip: See eval_schedule.bash for detailed configuration and parameters.

## 📝 To-Do List

- [x] Release automated extraction code (Training & Inference).
- [x] Organize and clean up dataset preparation scripts.
- [ ] Release **WildRoad** Dataset.
- [ ] Release pre-trained model checkpoints.
- [ ] Release Interactive Annotation Tool and GUI.

## 📍 Citation

If you find this work useful for your research, please cite our paper:

```bibtex
@article{magroad2025,
  title={Beyond Endpoints: Path-Centric Reasoning for Vectorized Off-Road Network Extraction},
  author={Guan, Wenfei and Mei, Jilin and Shen, Tong and Wu, Xumin and Wang, Shuo and Min, Cheng and Hu, Yu},
  journal={arXiv preprint # comming soon},
  year={2025}
}
```

## 🤝 Acknowledgements

We sincerely thank the authors of the following open-source projects for their contributions, which served as important foundations for our work:

- [**Sam_Road**](https://github.com/htcr/sam_road)
- [**Sam_Road++**](https://github.com/earth-insights/samroadplus)
- [**Sat2Graph**](https://github.com/songtaohe/Sat2Graph)
- [**Segment Anything Model**](https://github.com/facebookresearch/segment-anything)
