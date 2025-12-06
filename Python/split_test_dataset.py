import os
import shutil
import random
import os.path as path


def create_test_set_from_train(train_root: str, test_root: str, sample_ratio: float = 0.42, seed: int = 100):
    """
    从 train_root 的每个子目录中随机抽取 sample_ratio 比例的图片，
    复制到 test_root 对应的子目录中，保留原始文件名。

    :param train_root: 训练集根目录（如 'datasets/train_1'）
    :param test_root: 测试集输出根目录（如 'datasets/test'）
    :param sample_ratio: 抽样比例，如 0.1 表示 10%
    :param seed: 随机种子，确保可复现
    """
    random.seed(seed)

    if path.exists(test_root):
        shutil.rmtree(test_root)

    if not path.exists(train_root):
        raise ValueError(f"训练集路径不存在: {train_root}")

    # 获取所有类别子目录
    class_dirs = [d for d in os.listdir(train_root) if path.isdir(path.join(train_root, d))]
    class_dirs.sort()  # 确保顺序一致

    total_train = 0
    total_test = 0

    for class_name in class_dirs:
        src_dir = path.join(train_root, class_name)
        dst_dir = path.join(test_root, class_name)

        # 获取该类别下所有图片文件（支持常见格式）
        all_files = [
            f for f in os.listdir(src_dir)
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))
        ]

        if not all_files:
            print(f"⚠️ 警告：类别 '{class_name}' 下无有效图片，跳过。")
            continue

        n_total = len(all_files)
        n_sample = max(1, int(n_total * sample_ratio))  # 至少抽1张

        sampled_files = random.sample(all_files, n_sample)

        # 创建目标目录
        os.makedirs(dst_dir, exist_ok=True)

        # 复制文件
        for f in sampled_files:
            shutil.copy2(path.join(src_dir, f), path.join(dst_dir, f))

        total_train += n_total
        total_test += n_sample
        print(f"📁 类别 '{class_name}': {n_total} 张 → 抽取 {n_sample} 张 到测试集")

    print("\n✅ 测试集创建完成！")
    print(f"📊 总训练样本数: {total_train}")
    print(f"📊 总测试样本数: {total_test} (约 {sample_ratio * 100:.1f}%)")
    print(f"📂 测试集保存路径: {path.abspath(test_root)}")


if __name__ == "__main__":
    create_test_set_from_train(
        train_root="datasets/train_1",
        test_root="datasets/test",
        sample_ratio=0.42,
        seed=100
    )