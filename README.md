# 👀 Stenosis Detection using CNN

This project implements a Convolutional Neural Network (CNN) for detecting stenosis from medical images such as angiographic or spinal scans. The model is trained to identify key patterns in images that may indicate narrowing of vessels or spinal canals.

---

## 📁 Project Structure

```
stenosis-detection-using-cnn-model/
🗀️ data/
│   🗀️ test/              # Input images for inference
│   🗀️ train/             # Images for training (optional)
│
🗀️ model/
│   🗀️ old_best.pt        # Pretrained model (use Git LFS)
│
🗀️ src/
│   🗀️ inference.py       # Run model on input images
│   🗀️ train.py           # Script to train/retrain the model
│
🗀️ requirements.txt       # List of Python dependencies
🗀️ README.md              # Project documentation
```

---

## ⚙️ Installation & Setup

### 1. Clone the repository

```bash
git clone https://github.com/24siddhartha/Stenosis-detection-using-cnn-model.git
cd Stenosis-detection-using-cnn-model
```

### 2. Install dependencies

Make sure you have Python 3.8+ installed.

```bash
pip install -r requirements.txt
```

### 3. Setup Git Large File Storage (LFS)

GitHub does not allow files larger than 100MB in regular Git.

```bash
git lfs install
git lfs pull
```

> 🗱️ If `old_best.pt` is not downloaded, manually place it inside the `model/` folder or download from external link (if provided).

---

## ▶️ How to Run Inference

Use the provided script to predict stenosis from test images.

```bash
python src/inference.py \
    --model_path model/old_best.pt \
    --input_dir data/test \
    --output_dir data/results
```

### Arguments:

* `--model_path`: Path to the pre-trained model
* `--input_dir`: Folder containing input `.jpg` or `.png` images
* `--output_dir`: Folder to save the predictions/results

---

## 🏋️‍♂️ How to Train the Model (Optional)

If you wish to fine-tune or retrain the model:

```bash
python src/train.py \
    --train_dir data/train \
    --val_dir data/val \
    --epochs 50 \
    --batch_size 16 \
    --pretrained_model model/old_best.pt \
    --output_model model/retrained.pt
```

### Arguments:

* `--train_dir`: Training data directory
* `--val_dir`: Validation data directory
* `--pretrained_model`: (Optional) path to a model to start from
* `--output_model`: Output path to save the trained model

> Make sure `data/train` and `data/val` are structured correctly, typically with labeled subfolders for classification.

---

## 🧪 Sample Results

> *(Add example prediction images or a summary of output format here)*
> You can include:

* Before/after annotated images
* Accuracy, precision, recall metrics
* Visual explanation (e.g., Grad-CAM heatmaps)

---

## 🗞️ Dependencies

Key Python packages:

```
torch
torchvision
matplotlib
numpy
opencv-python
```

All are listed in [`requirements.txt`](./requirements.txt).

---

## 🔍 Dataset Information

> *(Optional – Mention source of dataset used for training, format, license, etc.)*

---

## 🛑 Limitations

* Accuracy depends on image quality and dataset diversity
* No clinical validation
* Model size exceeds GitHub limit (Git LFS used)

---

## 👨‍💼 Author

**Siddhartha Guguloth**
[GitHub](https://github.com/24siddhartha) | [LinkedIn](https://www.linkedin.com/in/siddhartha-guguloth/)

---

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## 💡 Future Improvements

* Add a Streamlit-based web demo
* Integrate Grad-CAM visualizations
* Improve dataset labeling and evaluation
