import os
import tensorflow as tf
from tensorflow.keras import layers, models, applications, regularizers
import matplotlib.pyplot as plt
import os.path as path
from datetime import datetime

# ======================
# 配置常量（建议集中管理）
# ======================
CONFIG: dict = {
    "IMG_SIZE": (224, 224),  # 输入图像尺寸（宽, 高），MobileNetV2 默认使用 224x224
    "BATCH_SIZE": 16,  # 每个训练批次包含的样本数量；若 GPU 显存充足（>4GB），可适当增大（如 32）
    "EPOCHS": 37,  # 训练总轮数（遍历整个训练集的次数）
    "LEARNING_RATE": 0.000060,  # Adam 优化器的学习率；适合微调预训练模型
    "SEED": 100,  # 随机种子，用于确保数据划分和数据增强的可复现性
    "VAL_RATE": 0.30,  # 验证集占总训练数据的比例（此处为 30%）
    "MODEL_DIR_ROOT": "models",  # 保存训练后 TFLite 模型的根目录
    "LABEL_DIR_ROOT": "labels",  # 保存类别标签文件（如 label-mutil.txt）的根目录
    "DATASET_PATH": "datasets/train_1"  # 训练数据集的根路径，应包含以类别命名的子文件夹（每个子文件夹存放对应类别的图片）
}


def ensure_dirs_exist(model_dir_root: str, label_dir_root: str):
    """确保模型和标签目录存在"""
    for d in [model_dir_root, label_dir_root]:
        if not path.exists(d):
            print(f"{d} folder does not exist, making...")
            os.makedirs(d)


def calculate_class_weight(source_path: str) -> dict:
    """计算类别权重、总样本数和类别数"""
    class_count = {}
    total_count = 0
    class_index = 0

    for class_dir in sorted(os.listdir(source_path)):
        class_path = path.join(source_path, class_dir)
        if not path.isdir(class_path):
            continue
        count = len([f for f in os.listdir(class_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        class_count[class_index] = count
        total_count += count
        class_index += 1

    num_classes = len(class_count)
    class_weights = {}
    if num_classes > 0:
        for idx, count in class_count.items():
            class_weights[idx] = (1.0 / count) * (total_count / num_classes)

    return {
        "class_weights": class_weights,
        "total_count": total_count,
        "num_classes": num_classes
    }


def setup_gpu():
    """配置 GPU 内存增长"""
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print("GPU settings error:", e)
    print(f"✅ TF version: {tf.__version__}, GPU available: {len(gpus) > 0}")


def load_datasets(data_path: str, img_size, batch_size, seed, val_rate):
    """加载训练与验证数据集，并应用 cache + prefetch"""
    print("🔍 loading dataset...(enable shuffle=True)...")
    train_ds_raw = tf.keras.utils.image_dataset_from_directory(
        data_path,
        validation_split=val_rate,
        subset="training",
        seed=seed,
        image_size=img_size,
        batch_size=batch_size,
        label_mode="categorical",
        shuffle=True
    )
    val_ds_raw = tf.keras.utils.image_dataset_from_directory(
        data_path,
        validation_split=val_rate,
        subset="validation",
        seed=seed,
        image_size=img_size,
        batch_size=batch_size,
        label_mode="categorical",
        shuffle=True
    )

    # ✅ 关键：在 .cache() 之前保存 class_names！
    class_names = train_ds_raw.class_names

    # 缓存与预取优化
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds_raw.cache().prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds_raw.cache().prefetch(buffer_size=AUTOTUNE)
    return train_ds, val_ds, class_names


def build_training_model(num_classes, img_size):
    """构建带数据增强的训练模型"""
    preprocess_input = applications.mobilenet_v2.preprocess_input

    data_augmentation = tf.keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.16),
        layers.RandomZoom(0.21),
        layers.RandomBrightness(0.16),
    ], name="data_augmentation")

    base_model = applications.MobileNetV2(
        input_shape=(*img_size, 3),
        include_top=False,
        weights="imagenet"
    )

    base_model.trainable = True  # 开启全微调

    model = models.Sequential([
        layers.Lambda(preprocess_input, input_shape=(*img_size, 3)),
        data_augmentation,
        base_model,
        layers.BatchNormalization(),
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.4),
        layers.Dense(
            num_classes,
            activation="softmax",
            kernel_regularizer=regularizers.l2(1e-3)
        )
    ])
    return model


def build_inference_model(num_classes, img_size):
    """构建用于 TFLite 导出的推理模型（无数据增强）"""
    preprocess_input = applications.mobilenet_v2.preprocess_input

    base_model = applications.MobileNetV2(
        input_shape=(*img_size, 3),
        include_top=False,
        weights="imagenet"
    )
    base_model.trainable = True
    # fline_tune_at = len(base_model.layers) // 2
    # for layer in base_model.layers[:fline_tune_at]:
    #     layer.trainable = False
    # for layer in base_model.layers[fline_tune_at:]:
    #     layer.trainable = True

    model = models.Sequential([
        layers.Lambda(preprocess_input, input_shape=(*img_size, 3)),
        base_model,
        layers.BatchNormalization(),
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.0),
        layers.Dense(num_classes, activation="softmax")
    ])
    return model


def plot_training_history(history, save_path="img\\training_curves.png"):
    """绘制并保存训练曲线"""
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='train_accuracy')
    plt.plot(history.history['val_accuracy'], label='val_accuracy')
    plt.title('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='train_loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.title('Loss')
    plt.legend()
    plt.savefig(save_path)
    print(f"✅ The training curve has been saved as {save_path}")


def save_labels(class_names, label_dir):
    """保存标签文件"""
    with open(label_dir, 'w', encoding='utf-8') as f:
        for name in class_names:
            f.write(name + '\n')
    print(f"✅ label file saved as  {label_dir}")


def visualize_validation_samples(val_dataset, class_names, save_path="img\\validation_samples.png"):
    """可视化验证集样本"""
    for images, labels in val_dataset.take(1):
        plt.figure(figsize=(12, 8))
        for i in range(min(9, len(images))):
            plt.subplot(3, 3, i + 1)
            plt.imshow(images[i].numpy().astype("uint8"))
            true_label = int(tf.argmax(labels[i]))
            plt.title(f"Label: {class_names[true_label]}")
            plt.axis('off')
        plt.suptitle("Random samples from the validation set (9 images)", fontsize=16)
        plt.tight_layout()
        plt.savefig(save_path)
        print(f"✅ Validation set sample images have been saved as {save_path}")
        break


def export_tflite_model(keras_model, model_save_path):
    """导出 TFLite 模型（float16 量化）"""
    converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float16]
    tflite_model = converter.convert()
    with open(model_save_path, 'wb') as f:
        f.write(tflite_model)
    print(f"✅ TFLite model saved as  {model_save_path}")


def main():
    # === 初始化 ===
    now_str = datetime.now().strftime('%m-%d-%H-%M-%S')
    MODEL_NAME = f"model-mutil-{now_str}.tflite"
    MODEL_DIR = path.join(CONFIG["MODEL_DIR_ROOT"], MODEL_NAME)
    LABEL_DIR = path.join(CONFIG["LABEL_DIR_ROOT"], "label-mutil.txt")

    ensure_dirs_exist(CONFIG["MODEL_DIR_ROOT"], CONFIG["LABEL_DIR_ROOT"])
    setup_gpu()

    # === 数据准备 ===
    weight_info = calculate_class_weight(CONFIG["DATASET_PATH"])
    class_weights = weight_info["class_weights"]
    num_classes = weight_info["num_classes"]
    total_count = weight_info["total_count"]

    print(f"num_classes: {num_classes}, total_count: {total_count}")
    for k, v in class_weights.items():
        print(f'class_index: {k}, weight: {v}')

    train_ds, val_ds, class_names = load_datasets(
        data_path=CONFIG["DATASET_PATH"],
        img_size=CONFIG["IMG_SIZE"],
        batch_size=CONFIG["BATCH_SIZE"],
        seed=CONFIG["SEED"],
        val_rate=CONFIG["VAL_RATE"]
    )

    # === 模型构建与训练 ===
    train_model = build_training_model(num_classes, CONFIG["IMG_SIZE"])
    train_model.compile(
        optimizer=tf.keras.optimizers.Adam(CONFIG["LEARNING_RATE"]),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    callbacks = [
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-7),
        tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True, mode='max')
    ]

    print("\n🚀 training..... ")
    history = train_model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=CONFIG["EPOCHS"],
        callbacks=callbacks,
        class_weight=class_weights,
        verbose=1
    )

    # === 可视化与保存 ===
    plot_training_history(history)
    visualize_validation_samples(val_ds, class_names)

    # 构建推理模型并复制权重
    inference_model = build_inference_model(num_classes, CONFIG["IMG_SIZE"])
    inference_model.set_weights(train_model.get_weights())

    export_tflite_model(inference_model, MODEL_DIR)
    save_labels(class_names, LABEL_DIR)


if __name__ == "__main__":
    main()
