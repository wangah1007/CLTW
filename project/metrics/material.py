import cv2
import numpy as np
import json
from pathlib import Path


COLOR_T = 30.0
TEXTURE_T = 20.0


def _clip01(x: float) -> float:
    return float(np.clip(x, 0.0, 1.0))


# ===== 笔画mask =====
def get_text_mask(patch):
    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)

    th = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        15, 5
    )

    kernel = np.ones((3, 3), np.uint8)
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel)

    return th > 0


# ===== 单字符评分 =====
def compute_char_score(patch):
    h, w = patch.shape[:2]
    if h < 5 or w < 5:
        return 0

    mask_text = get_text_mask(patch)
    mask_bg = ~mask_text

    if np.sum(mask_text) < 10 or np.sum(mask_bg) < 10:
        return 0

    # ===== Lab颜色 =====
    lab = cv2.cvtColor(patch, cv2.COLOR_BGR2LAB)

    text_color = np.mean(lab[mask_text], axis=0)
    bg_color   = np.mean(lab[mask_bg], axis=0)

    color_dist = np.linalg.norm(text_color - bg_color)
    color_score = np.exp(-color_dist / COLOR_T)

    # ===== 梯度 =====
    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    grad = cv2.Sobel(gray, cv2.CV_32F, 1, 1, ksize=3)
    grad = np.abs(grad)

    grad_text = np.mean(grad[mask_text])
    grad_bg   = np.mean(grad[mask_bg])

    texture_score = np.exp(-abs(grad_text - grad_bg) / TEXTURE_T)

    score = 0.6 * color_score + 0.4 * texture_score

    return score


# ===== 单张图片（不保存可视化，仅返回分数）=====
def process_single(image_path, json_path):
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"无法读取图片: {image_path}")

    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    scores = []

    for char in data["characters"]:
        pts = np.array(char["points"], dtype=np.int32)

        x, y, w, h = cv2.boundingRect(pts)
        patch = img[y:y+h, x:x+w]

        if patch.size == 0:
            continue

        score = compute_char_score(patch)
        scores.append(score)

    # ===== 整体分数 =====
    scores = np.array(scores)

    if len(scores) > 0:
        scores = np.sort(scores)
        scores = scores[int(len(scores)*0.2):]  # 去掉最低20%
        final_score = np.mean(scores)
    else:
        final_score = 0

    return _clip01(float(final_score))


def process_single_from_characters(image_path, characters):
    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"无法读取图片: {image_path}")

    scores = []
    for char in characters:
        if "points" not in char:
            continue
        pts = np.array(char["points"], dtype=np.int32)
        x, y, w, h = cv2.boundingRect(pts)
        patch = img[y:y+h, x:x+w]
        if patch.size == 0:
            continue
        scores.append(compute_char_score(patch))

    scores = np.array(scores)
    if len(scores) > 0:
        scores = np.sort(scores)
        scores = scores[int(len(scores)*0.2):]
        return _clip01(float(np.mean(scores)))
    return 0.0


# ===== 批量处理 =====
def process_batch(image_folder, json_folder, output_json_path="material_scores.json"):
    image_folder = Path(image_folder)
    json_folder = Path(json_folder)

    if not image_folder.exists():
        raise FileNotFoundError(f"图片文件夹不存在: {image_folder}")
    if not json_folder.exists():
        raise FileNotFoundError(f"JSON文件夹不存在: {json_folder}")

    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    results = {}

    for img_path in image_folder.iterdir():
        if img_path.suffix.lower() not in image_extensions:
            continue

        json_path = json_folder / (img_path.stem + ".json")
        if not json_path.exists():
            print(f"警告: 未找到对应的JSON文件 {json_path.name}，跳过图片 {img_path.name}")
            continue

        try:
            score = process_single(str(img_path), str(json_path))
            results[img_path.name] = score
            print(f"✓ {img_path.name} -> 分数: {score:.3f}")
        except Exception as e:
            print(f"✗ 处理失败 {img_path.name}: {e}")
            results[img_path.name] = None

    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n汇总分数已保存到: {output_json_path}")
    return results


if __name__ == "__main__":
    image_folder = input("请输入图片文件夹路径: ").strip().strip('"').strip("'")
    json_folder = input("请输入JSON文件夹路径: ").strip().strip('"').strip("'")
    output_json_path = input("请输入输出JSON文件路径(回车使用 material_scores.json): ").strip()
    if not output_json_path:
        output_json_path = "material_scores.json"

    process_batch(image_folder, json_folder, output_json_path)