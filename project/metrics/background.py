import cv2
import numpy as np
import json
import os
from pathlib import Path


def _clip01(x: float) -> float:
    return float(np.clip(x, 0.0, 1.0))


# =========================================================
# 1️⃣ 背景复杂度计算（只计算mask区域）
# =========================================================
def compute_complexity(gray, mask):

    if np.sum(mask) == 0:
        return 0.0

    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    gradient = np.sqrt(sobelx**2 + sobely**2)

    laplacian = cv2.Laplacian(gray, cv2.CV_64F, ksize=3)
    texture = np.abs(laplacian)

    values = (gradient + texture)[mask == 255]

    if len(values) == 0:
        return 0.0

    return np.mean(values)


# =========================================================
# 2️⃣ 笔画提取
# =========================================================
def extract_stroke_mask(roi_gray):

    blur = cv2.GaussianBlur(roi_gray, (5, 5), 0)

    _, otsu = cv2.threshold(
        blur, 0, 255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    edges = cv2.Canny(blur, 50, 150)

    stroke = cv2.bitwise_and(otsu, edges)

    kernel = np.ones((3, 3), np.uint8)
    stroke = cv2.dilate(stroke, kernel, iterations=1)

    stroke = (stroke > 0).astype(np.uint8) * 255

    return stroke


def _score_from_image_and_characters(image: np.ndarray, characters):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    char_scores = []

    for char in characters:
        pts = np.array(char["points"], dtype=np.int32)
        x, y, w_box, h_box = cv2.boundingRect(pts)
        roi_gray = gray[y:y+h_box, x:x+w_box]

        stroke_mask = extract_stroke_mask(roi_gray)

        inv_stroke = (stroke_mask == 0).astype(np.uint8)
        dist_map = cv2.distanceTransform(inv_stroke, cv2.DIST_L2, 5)

        d = max(2, min(w_box, h_box) // 20)
        bg_mask = (dist_map > d).astype(np.uint8) * 255

        score = compute_complexity(roi_gray, bg_mask)
        char_scores.append(score)

    char_scores = np.array(char_scores, dtype=np.float32)
    raw = float(np.mean(char_scores)) if len(char_scores) else 0.0
    return _clip01(raw)


# =========================================================
# 3️⃣ 主流程
# =========================================================
def process_single(image_path, json_path):
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"无法读取图片: {image_path}")

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    characters = data.get("characters", [])
    return _score_from_image_and_characters(image, characters)


def process_single_from_characters(image_path, characters):
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"无法读取图片: {image_path}")
    return _score_from_image_and_characters(image, characters)


def process_batch(image_folder, json_folder, output_json_path="background_scores.json"):
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
            print(f"警告: 未找到对应JSON文件 {json_path.name}，跳过 {img_path.name}")
            continue

        try:
            score = process_single(str(img_path), str(json_path))
            results[img_path.name] = score
            print(f"✓ {img_path.name} -> 背景复杂度: {score:.6f}")
        except Exception as e:
            print(f"✗ 处理失败 {img_path.name}: {e}")
            results[img_path.name] = None

    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n汇总已保存: {output_json_path}")
    return results


# =========================================================
# 5️⃣ 执行
# =========================================================
if __name__ == "__main__":

    image_folder = input("请输入图片文件夹路径: ").strip().strip('"').strip("'")
    json_folder = input("请输入JSON文件夹路径: ").strip().strip('"').strip("'")
    output_json_path = input("请输入输出JSON文件路径(回车使用 background_scores.json): ").strip()
    if not output_json_path:
        output_json_path = "background_scores.json"

    process_batch(image_folder, json_folder, output_json_path)