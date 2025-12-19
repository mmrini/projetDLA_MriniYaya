# Fine-tuning GoogLeNet on CIFAR-10 (Project DLA — GM5)

This repository contains code used to fine-tune a pretrained GoogLeNet (Inception V1) (*Going deeper with convolutions - Christian Szegedy et al. (2014)*) model on the CIFAR-10 dataset, and to explore explainability, adversarial attacks, and membership-inference privacy analyses.

This project was produced as part of the **Advanced Deep Learning course** at *INSA Rouen Normandie* and *University of Rouen* (Master 2 Data Science). 

**Authors**: Il-hême Yaya and Malak Mrini (GM5, M2 SD).

Contents
--------
- `finetune_googlenet.py` — Core functions to load a pretrained GoogLeNet, adapt it for CIFAR-10, training loop helpers, and multi-crop evaluation.
- `finetune_test.ipynb` — Notebook to run the full fine-tuning experiments, dataset preparation, training loop, validation and plotting. This is the primary entrypoint to reproduce experiments.
- `tools/xai.py` — Explainability utilities (Grad-CAM, Occlusion Sensitivity, feature extraction and t-SNE visualization).
- `tools/attacks.py` — Adversarial attack implementations (FGSM, PGD) and evaluation helper.
- `tools/privacy.py` — Membership inference attack utilities and plotting helpers.
- `data/` — Expected location for CIFAR-10 data (the notebook uses torchvision.datasets to download into `./data`).
- `results/` — Directory used for saving trained checkpoints (an example checkpoint exists in `results/finetune_20251213_224800/best_model.pth`).

Purpose
-------
The code demonstrates how to fine-tune a pretrained GoogLeNet on the CIFAR-10 dataset and to analyze model behaviour using explainability techniques, adversarial robustness evaluation, and a simple membership-inference attack pipeline.

Reproducibility 
----------------------------
**Recommended environment:** Python 3.8+ and a CUDA-capable GPU for reasonable training speed. A CPU will still work but training will be slow.


1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Launch the Jupyter notebook.

Alternatively, open the notebook in Google Colab (the notebook contains optional mounting code for Google Drive). In Colab, ensure the runtime is set to GPU.

3. Configure the notebook (Cell 2 and config):
- Set `config['data_dir']` (default `./data`) — torchvision will download CIFAR-10 here.
- Set `config['save_dir']` (default `./results`) — checkpoints will be saved here.
- Adjust `config['batch_size']`, `config['num_epochs']`, and `config['val_num_crops']`.

4. Run the notebook cells sequentially. The notebook performs:
- Data download and augmentation setup for CIFAR-10
- Model loading (pretrained GoogLeNet via torchvision) and adaptation for 10 classes
- Training loop with validation using multi-crop evaluation (1, 10, or 144 crops)
- Saving the best model to `results/<experiment_name>/best_model.pth`

Notes on outputs
----------------
- Trained model checkpoints are saved to `results/<experiment_name>/best_model.pth`.
- The notebook records training history (loss/accuracy) and shows plots after each epoch.
- The `tools/` modules provide utilities to:
  - Visualize explanations (Grad-CAM, occlusion sensitivity)
  - Generate adversarial examples (FGSM, PGD) and measure robustness
  - Run a simple threshold-based membership inference attack and compute ROC/AUC

Implementation details and assumptions
--------------------------------------
- The GoogLeNet model is loaded using `torchvision.models.googlenet` (the code handles both newer and older torchvision call signatures).
- Input preprocessing in the notebook resizes images to 256 then crops to 224×224, matching the input size expected by GoogLeNet.
- The code uses ImageNet normalization statistics.
- For fine-tuning on small datasets (CIFAR-10) the notebook defaults to freezing backbone parameters and training only the classifier.

Contact / Authors
-----------------
Il-hême Yaya and Malak Mrini (GM5, Master 2 Data Science)

**Course:** Advanced Deep Learning — INSA Rouen Normandie & University of Rouen

License
-------
This repository is provided for educational use within the course.

