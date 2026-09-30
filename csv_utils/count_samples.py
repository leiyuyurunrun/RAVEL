"""
统计 data/csv/v4 下各类别攻击的正负样本数量
"""

import os
import pandas as pd
from pathlib import Path

# ============ 可配置参数 ============
# 输入文件夹路径
INPUT_DIR = Path("data/csv/v4")

# 8类攻击类别的文件夹名称
CATEGORIES = [
    "Access control",
    "Flashloans",
    "Insufficient validation",
    "Market manipulation",
    "Price manipulation",
    "Protocol accounting exploitation",
    "Reentrancy",
    "Token semantic exploitation",
]

# 正样本文件的后缀模式 (不含类别前缀)
POS_PATTERNS = [
    "_evaluation_hard_pos.csv",
    "_evaluation_pos_new.csv",
    "_Fewshot_pos.csv",
]

# 负样本文件的后缀模式
NEG_PATTERNS = [
    "_evaluation_hard_neg.csv",
    "_evaluation_neg_new.csv",
    "_Fewshot_neg.csv",
]

# 输出统计表的文件名
OUTPUT_FILE = INPUT_DIR / "sample_statistics.csv"


def get_category_prefix(category: str) -> str:
    """根据文件夹名称获取文件前缀"""
    # 将空格替换为下划线，并小写
    return category.lower().replace(" ", "_")


def count_rows_in_file(file_path: Path) -> int:
    """统计CSV文件的行数（不含表头）"""
    if not file_path.exists():
        return 0
    try:
        df = pd.read_csv(file_path)
        return len(df)
    except Exception as e:
        print(f"  警告: 读取文件 {file_path} 失败: {e}")
        return 0


def count_category_samples(category: str) -> tuple:
    """
    统计某个类别的正负样本数量
    返回: (pos_count, neg_count)
    """
    prefix = get_category_prefix(category)
    pos_count = 0
    neg_count = 0

    # 统计正样本
    for pattern in POS_PATTERNS:
        file_name = f"{prefix}{pattern}"
        file_path = INPUT_DIR / category / file_name
        count = count_rows_in_file(file_path)
        pos_count += count
        print(f"  + {file_name}: {count}")

    # 统计负样本
    for pattern in NEG_PATTERNS:
        file_name = f"{prefix}{pattern}"
        file_path = INPUT_DIR / category / file_name
        count = count_rows_in_file(file_path)
        neg_count += count
        print(f"  - {file_name}: {count}")

    return pos_count, neg_count


def main():
    print("=" * 60)
    print("样本统计开始")
    print(f"输入目录: {INPUT_DIR}")
    print("=" * 60)

    results = []

    for category in CATEGORIES:
        print(f"\n处理类别: {category}")
        pos_num, neg_num = count_category_samples(category)
        results.append({
            "category": category,
            "pos_num": pos_num,
            "neg_num": neg_num
        })
        print(f"  总计: pos={pos_num}, neg={neg_num}")

    # 创建统计表
    df = pd.DataFrame(results)

    # 添加总计行
    total_row = pd.DataFrame([{
        "category": "TOTAL",
        "pos_num": df["pos_num"].sum(),
        "neg_num": df["neg_num"].sum()
    }])
    df = pd.concat([df, total_row], ignore_index=True)

    # 保存统计表
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False)

    print("\n" + "=" * 60)
    print("统计完成!")
    print(f"统计表已保存至: {OUTPUT_FILE}")
    print("=" * 60)

    # 打印统计表
    print("\n统计表预览:")
    print(df.to_string(index=False))

    return df


if __name__ == "__main__":
    main()
