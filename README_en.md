# Implementing Multi-Class Image Recognition on Android with TensorFlow Lite (TFLite or MNN)

## Project Overview

This project demonstrates a complete workflow for image classification:

- **Python Side:** Train a deep learning model supporting multi-class image classification.
- **Android Side:** Deploy and run the model using two inference engines: **TensorFlow Lite (TFLite)** and **MNN**.

💡 **Note:**

- The model training code was primarily generated with the assistance of AI (Qwen & Gemini 3 Pro).
- The dataset was collected by our team: `dog` and `cat` categories are from Kaggle; other categories were scraped via Baidu and Bing Image Search.
- The default training script enables **GPU acceleration**. You must have CUDA and cuDNN compatible with your graphics card installed. If you only support CPU, please ask an AI to generate the corresponding CPU training version.

------

## 🧪 Training Environment Configuration

### Basic Requirements

- **Python Version:** ≥ 3.8.20 (Recommended 3.8.x)
- **Virtual Environment:** Strongly recommended (e.g., `venv` or `conda`) to avoid polluting the system environment.

### Install Dependencies

Bash

```
# Create and activate virtual environment (Example)
python -m venv tf_env
source tf_env/bin/activate   # Linux/macOS
# or tf_env\Scripts\activate # Windows

# Install dependencies
pip install -r requirements.txt
```

### Download Dataset

Please visit the link below to download the dataset:

🔗 [123 Pan Download Link](https://www.123865.com/s/vcnRVv-Q0U1h?pwd=T5qo '123 Pan Download') (Access Code: T5qo)

After unzipping the archive, ensure the directory structure looks like this:

Plaintext

```
your_project/
└── Python/
    └── datasets/         ← Place the dataset here
```

------

## 📁 Training Files Description

| **File/Directory**                            | **Function Description**                                     |
| --------------------------------------------- | ------------------------------------------------------------ |
| **`datasets/`**                               | Stores raw training and validation image data.               |
| **`img/`**                                    | Stores images for training, prediction, and confusion matrices. |
| **`predict_error_image/`**                    | Stores filenames of images that failed prediction.           |
| **`batch_convert_to_jpg.py`**                 | Converts untrainable images into a trainable format (does more than just changing the extension). |
| **`change_file_name.py`**                     | Batch renames image files in `datasets/train_1/xxx/` to unify the format. |
| **`check_img.py`**                            | Checks if images can be loaded normally (filters out corrupt or unsupported formats). |
| **`split_test_dataset.py`**                   | Splits an independent test set from the training set (used for subsequent evaluation). |
| **`predict_test_model.py`**                   | Uses the trained model to predict on the test set and evaluate performance. |
| **`train_main.py`** / **`train_finetune.py`** | Main training scripts responsible for model construction, training, and export (`.tflite`). You can use either; the second one is significantly faster. |
| **`models/`**                                 | Automatically created. Saves generated model files (e.g., `.tflite`). |
| **`labels/`**                                 | Automatically created. Saves category label mapping files (e.g., `labels.txt`). |

✅ **Note:** The scripts will automatically create the `models/` and `labels/` directories; you do not need to create them manually.

------

## 📱 Deploying TFLite / MNN Models on Android

### System Requirements

| **Inference Framework**      | **Minimum Android API** | **Java Version** | **Other Dependencies**              |
| ---------------------------- | ----------------------- | ---------------- | ----------------------------------- |
| **TensorFlow Lite (TFLite)** | API 34 (Android 14)     | 1.8              | No NDK required                     |
| **MNN**                      | API 29 (Android 10)     | 1.8              | Requires NDK (Version 20.0.5594570) |

⚠️ **Note:** If using MNN, you must first convert the `.tflite` model to `.mnn` format (see below).

### Integration Steps

1. Create New Project:

   In Android Studio, select the "No Activity" template to create a new project.

2. Import Reference Structure:

   Copy the corresponding project structure based on your target framework:

   - **TFLite** → Refer to `MyApplication3/`
   - **MNN** → Refer to `MNNClassification/` (Source: GitHub open source project, author info currently missing)

3. **Configure Gradle:**

   - Modify `applicationId` in `app/build.gradle` to your actual package name.
   - **Important:** Complete this step first, then click **"Sync Now"**. Wait for dependencies to download before adding other files.

4. Update Manifest:

   Modify the package attribute in AndroidManifest.xml to ensure it matches your applicationId.

5. (MNN Only) Model Conversion:

   Use the MNN official tool to convert .tflite to .mnn:

   Bash

   ```
   ./MNNConvert -f TFLITE \
     --modelFile model.tflite \
     --MNNModel model.mnn \
     --bizCode your_app
   ```

6. Place Resource Files:

   Place the following files into app/src/main/assets/:

   - Model file (`model.tflite` or `model.mnn`)
   - Label file (`labels.txt`)

------

## ✅ Tips & Tricks

- **Image Preprocessing:** Ensure that preprocessing steps (resizing, normalization, etc.) in the Android app match exactly what was used during training.

- **CPU Training:** If your training environment does not have a GPU, add the following code to the beginning of `train_main.py` to force CPU usage:

  Python

  ```
  import tensorflow as tf
  tf.config.set_visible_devices([], 'GPU')
  ```

- **Verification:** It is recommended to use `predict_test_model.py` to verify model accuracy before deployment.