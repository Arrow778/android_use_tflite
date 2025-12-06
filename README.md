# 在 Android 上使用 TensorFlow Lite（TFLite）模型实现多类别图像识别

## 项目简介

本项目包含完整的流程：  
- **Python 端**：训练一个支持多类别图像分类的深度学习模型  
- **Android 端**：部署并运行该模型，支持 **TensorFlow Lite（TFLite）** 和 **MNN** 两种推理引擎  

> 💡 **说明**：  
> - 模型训练代码主要由 AI（通义千问 & Gemini 3 Pro）辅助生成。  
> - 数据集由我们小组自行收集：其中 `dog` 和 `cat` 类别来自 [Kaggle](https://www.kaggle.com)，其余类别通过百度、Bing 图片搜索获取。  
> - 默认训练脚本启用 **GPU 加速**，需提前安装与显卡兼容的 **CUDA** 和 **cuDNN**。若仅支持 CPU，请让 AI 生成对应的 CPU 训练版本。

---

## 🧪 训练环境配置

### 基础要求
- **Python 版本 ≥ 3.8.20**（推荐使用 3.8.x）
- 强烈建议使用虚拟环境（如 `venv` 或 `conda`）避免污染系统环境

### 安装依赖
```bash
# 创建并激活虚拟环境（示例）
python -m venv tf_env
source tf_env/bin/activate  # Linux/macOS
# 或 tf_env\Scripts\activate  # Windows

# 安装依赖
pip install -r requirements.txt
```

### 下载数据集
请访问以下链接下载数据集：  
🔗 [123网盘下载地址](https://www.123865.com/s/vcnRVv-Q0U1h?pwd=T5qo)（提取码：`T5qo`）

将压缩包解压后，确保目录结构如下：
```
your_project/
└── Python/
    └── datasets/          ← 放在此处
```

---

## 📁 训练相关文件说明

| 文件/目录                             |                           功能说明                           |
| ------------------------------------- | :----------------------------------------------------------: |
| `datasets/`                           |                  存放原始训练与验证图像数据                  |
| `img/`                                |                存放训练、预测、混淆矩阵的图片                |
| `predict_error_image/`                |         `predict_test_model.py`预测失败的文件名列表          |
| `batch_convert_to_jpg.py`             | 将不能训练的图片尽可能的转为可以训练的格式（不仅仅是改后缀名） |
| `change_file_name.py`                 |  批量重命名 `datasets/train_1/xxx/` 中的图片文件，统一格式   |
| `check_img.py`                        |       检查图片是否可正常加载（过滤损坏或不支持的格式）       |
| `split_test_dataset.py`               |         从训练集中划分出独立的测试集（用于后续评估）         |
| `predict_test_model.py`               |          使用训练好的模型对测试集进行预测并评估性能          |
| `train_main.py` / `train_finetune.py` | **主训练脚本**，负责模型构建、训练与导出（.tflite），使用任意一个都行，第二个要快很多。 |
| `models/`                             |       自动创建，保存训练生成的模型文件（如 `.tflite`）       |
| `labels/`                             |      自动创建，保存类别标签映射文件（如 `labels.txt`）       |

> ✅ 脚本会自动创建 `models/` 和 `labels/` 目录，无需手动新建。

---

## 📱 在 Android 中部署 TFLite / MNN 模型

### 系统要求

| 推理框架                     | 最低 Android API    | Java 版本 | 其他依赖                      |
| ---------------------------- | ------------------- | --------- | ----------------------------- |
| **TensorFlow Lite (TFLite)** | API 34 (Android 14) | 1.8       | 无需 NDK                      |
| **MNN**                      | API 29 (Android 10) | 1.8       | 需 NDK（版本 `20.0.5594570`） |

> ⚠️ 若使用 MNN，需先将 `.tflite` 模型转换为 `.mnn` 格式（见下文）。

---

### 集成步骤

1. **创建新项目**  
   在 Android Studio 中选择 **“No Activity”** 模板新建项目。

2. **导入参考结构**  
   根据目标框架，复制对应项目结构：
   - TFLite → 参考 `MyApplication3/`
   - MNN → 参考 `MNNClassification/`（源自 GitHub 开源项目，作者信息暂缺）

3. **配置 Gradle**  
   - 修改 `app/build.gradle` 中的 `applicationId` 为你的实际包名  
   - **务必先完成此步，再点击 “Sync Now”**，等待依赖下载完毕后再继续补充其他文件

4. **更新 Manifest**  
   修改 `AndroidManifest.xml` 中的 `package` 属性，确保与 `applicationId` 一致。

5. **（仅 MNN）模型转换**  
   使用 MNN 官方工具将 `.tflite` 转换为 `.mnn`：
   ```bash
   ./MNNConvert -f TFLITE \
     --modelFile model.tflite \
     --MNNModel model.mnn \
     --bizCode your_app
   ```

6. **放置资源文件**  
   将以下文件放入 `app/src/main/assets/`：
   - 模型文件（`model.tflite` 或 `model.mnn`）
   - 标签文件（`labels.txt`）

---

## ✅ 温馨提示

- 图像预处理（尺寸、归一化等）需与训练时保持一致。
- 若训练环境无 GPU，可在 `train_main.py` 开头添加以下代码强制使用 CPU：
  ```python
  import tensorflow as tf
  tf.config.set_visible_devices([], 'GPU')
  ```
- 部署前建议用 `predict_test_model.py` 验证模型准确率。

