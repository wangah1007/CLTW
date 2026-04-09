import json
import cv2
import numpy as np
import os
from pathlib import Path


def _clip01(x: float) -> float:
    return float(np.clip(x, 0.0, 1.0))


# =========================
# ROI 裁剪
# =========================

def crop_by_polygon(image, points):
    pts = np.array(points, dtype=np.int32)
    x, y, w, h = cv2.boundingRect(pts)
    roi = image[y:y+h, x:x+w]
    return roi, (x, y, w, h)


def normalize(x, xmin, xmax):
    if xmax - xmin < 1e-6:
        return 0.0
    return np.clip((x - xmin) / (xmax - xmin), 0, 1)


# =========================
# 单字模糊度
# =========================

def compute_char_blur(roi_gray):
    h, w = roi_gray.shape
    if h < 20 or w < 20:
        return None

    # Laplacian
    lap = cv2.Laplacian(roi_gray, cv2.CV_64F)
    lap_score = lap.var() / (h * w)

    # FFT 高频
    f = np.fft.fft2(roi_gray)
    fshift = np.fft.fftshift(f)
    mag = np.abs(fshift)

    cy, cx = h // 2, w // 2
    center = mag[cy-5:cy+5, cx-5:cx+5]
    fft_score = (mag.sum() - center.sum()) / (mag.sum() + 1e-6)

    # Sobel 梯度
    gx = cv2.Sobel(roi_gray, cv2.CV_64F, 1, 0, 3)
    gy = cv2.Sobel(roi_gray, cv2.CV_64F, 0, 1, 3)
    grad_score = np.mean(np.sqrt(gx**2 + gy**2))

    return lap_score, fft_score, grad_score


# =========================
# 主流程
# =========================

def process(image_path, json_path):
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"无法读取图片: {image_path}")
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    with open(json_path, "r", encoding="utf-8") as f:
        anno = json.load(f)

    records = []

    for char in anno["characters"]:
        roi, bbox = crop_by_polygon(image, char["points"])
        roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        res = compute_char_blur(roi_gray)
        if res is None:
            continue

        lap, fft, grad = res
        records.append({
            "char": char["transcription"],
            "bbox": bbox,
            "lap": lap,
            "fft": fft,
            "grad": grad,
            "area": bbox[2] * bbox[3]
        })

    # 归一化
    lap_vals = [r["lap"] for r in records]
    fft_vals = [r["fft"] for r in records]
    grad_vals = [r["grad"] for r in records]

    for r in records:
        lap_n = normalize(r["lap"], min(lap_vals), max(lap_vals))
        fft_n = normalize(r["fft"], min(fft_vals), max(fft_vals))
        grad_n = normalize(r["grad"], min(grad_vals), max(grad_vals))

        r["blur"] = 1.0 - (0.5 * lap_n + 0.3 * fft_n + 0.2 * grad_n)

    # 整图模糊度（面积加权）
    image_blur = sum(r["blur"] * r["area"] for r in records) / sum(r["area"] for r in records)

    return image_rgb, records, _clip01(float(image_blur))


def process_from_characters(image_path, characters):
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"无法读取图片: {image_path}")
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    records = []
    for char in characters:
        if "points" not in char:
            continue
        roi, bbox = crop_by_polygon(image, char["points"])
        roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        res = compute_char_blur(roi_gray)
        if res is None:
            continue

        lap, fft, grad = res
        records.append({
            "char": char.get("transcription", ""),
            "bbox": bbox,
            "lap": lap,
            "fft": fft,
            "grad": grad,
            "area": bbox[2] * bbox[3]
        })

    if not records:
        return image_rgb, records, 0.0

    lap_vals = [r["lap"] for r in records]
    fft_vals = [r["fft"] for r in records]
    grad_vals = [r["grad"] for r in records]

    for r in records:
        lap_n = normalize(r["lap"], min(lap_vals), max(lap_vals))
        fft_n = normalize(r["fft"], min(fft_vals), max(fft_vals))
        grad_n = normalize(r["grad"], min(grad_vals), max(grad_vals))
        r["blur"] = 1.0 - (0.5 * lap_n + 0.3 * fft_n + 0.2 * grad_n)

    image_blur = sum(r["blur"] * r["area"] for r in records) / sum(r["area"] for r in records)
    return image_rgb, records, _clip01(float(image_blur))


def process_batch(image_folder, json_folder, output_json_path="blur_scores.json"):
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
            _, _, image_blur = process(str(img_path), str(json_path))
            results[img_path.name] = float(image_blur)
            print(f"✓ {img_path.name} -> blur: {image_blur:.6f}")
        except Exception as e:
            print(f"✗ 处理失败 {img_path.name}: {e}")
            results[img_path.name] = None

    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n汇总已保存: {output_json_path}")
    return results


# =========================
# 入口
# =========================

if __name__ == "__main__":
    image_folder = input("请输入图片文件夹路径: ").strip().strip('"').strip("'")
    json_folder = input("请输入JSON文件夹路径: ").strip().strip('"').strip("'")
    output_json_path = input("请输入输出JSON文件路径(回车使用 blur_scores.json): ").strip()
    if not output_json_path:
        output_json_path = "blur_scores.json"

    process_batch(image_folder, json_folder, output_json_path)