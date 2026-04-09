import cv2
import numpy as np
import json
from pathlib import Path
import csv


def _clip01(x: float) -> float:
    return float(np.clip(x, 0.0, 1.0))


# =========================
# 读取JSON
# =========================

def load_json(json_path):

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    boxes = []

    for item in data["characters"]:
        pts = np.array(item["points"], dtype=np.int32)
        boxes.append(pts)

    return boxes


# =========================
# 裁剪文字框
# =========================

def crop_polygon(img, pts):

    x,y,w,h = cv2.boundingRect(pts)

    patch = img[y:y+h, x:x+w]

    return patch


# =========================
# FFT周期纹理
# =========================

def fft_period_score(gray):

    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)

    mag = np.abs(fshift)
    mag /= (np.max(mag) + 1e-6)

    h,w = mag.shape
    cy,cx = h//2, w//2

    Y,X = np.ogrid[:h,:w]

    dist = np.sqrt((Y-cy)**2+(X-cx)**2)

    r = min(h,w)*0.15

    high = mag[dist>r]

    peak = np.percentile(high,95)
    mean = np.mean(mag)

    score = min(1.0, peak/(mean+1e-6)/5)

    return score


# =========================
# 高频能量
# =========================

def high_freq_score(gray):

    lap = cv2.Laplacian(gray, cv2.CV_64F)

    energy = np.mean(np.abs(lap))

    score = min(1.0, energy/30)

    return score


# =========================
# 方向纹理
# =========================

def direction_score(gray):

    sx = cv2.Sobel(gray,cv2.CV_64F,1,0,ksize=3)
    sy = cv2.Sobel(gray,cv2.CV_64F,0,1,ksize=3)

    ex = np.mean(np.abs(sx))
    ey = np.mean(np.abs(sy))

    return abs(ex-ey)/(ex+ey+1e-6)


# =========================
# 字符STI
# =========================

def compute_char_sti(patch):

    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)

    if gray.shape[0] < 10 or gray.shape[1] < 10:
        return 0

    p = fft_period_score(gray)
    h = high_freq_score(gray)
    d = direction_score(gray)

    return 0.5*p + 0.3*h + 0.2*d


# =========================
# 单图处理
# =========================

def process_single(image_path, json_path):
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"无法读取图片: {image_path}")

    boxes = load_json(str(json_path))

    scores = []

    for pts in boxes:

        patch = crop_polygon(image, pts)

        sti = compute_char_sti(patch)

        scores.append(sti)

    overall = float(np.mean(scores)) if scores else 0.0

    return _clip01(overall)


def process_single_from_characters(image_path, characters):
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"无法读取图片: {image_path}")

    boxes = [np.array(item["points"], dtype=np.int32) for item in characters if "points" in item]
    scores = []
    for pts in boxes:
        patch = crop_polygon(image, pts)
        scores.append(compute_char_sti(patch))
    return _clip01(float(np.mean(scores)) if scores else 0.0)


# =========================
# 批量处理
# =========================

def process_batch(image_folder, json_folder, output_dir):
    image_folder = Path(image_folder)
    json_folder = Path(json_folder)
    output_dir = Path(output_dir)

    if not image_folder.exists():
        raise FileNotFoundError(f"图片文件夹不存在: {image_folder}")
    if not json_folder.exists():
        raise FileNotFoundError(f"JSON文件夹不存在: {json_folder}")

    output_dir.mkdir(exist_ok=True, parents=True)

    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    rows = []
    results = {}

    for img_path in image_folder.iterdir():
        if img_path.suffix.lower() not in image_extensions:
            continue

        json_path = json_folder / (img_path.stem + ".json")
        if not json_path.exists():
            print(f"警告: 未找到对应JSON文件 {json_path.name}，跳过 {img_path.name}")
            continue

        try:
            score = process_single(img_path, json_path)
            results[img_path.name] = score
            rows.append([img_path.name, score])
            print(f"✓ {img_path.name} -> STI: {score:.6f}")
        except Exception as e:
            print(f"✗ 处理失败 {img_path.name}: {e}")
            results[img_path.name] = None
            rows.append([img_path.name, None])

    csv_file = output_dir / "screen_texture_index.csv"
    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image", "screen_texture_index"])
        writer.writerows(rows)

    json_file = output_dir / "screen_texture_index.json"
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"结果保存: {csv_file}")
    print(f"结果保存: {json_file}")
    return results


# =========================
# 主程序
# =========================

if __name__ == "__main__":

    image_folder = input("请输入图片文件夹路径: ").strip().strip('"').strip("'")
    json_folder = input("请输入JSON文件夹路径: ").strip().strip('"').strip("'")
    output_dir = input("请输入输出结果文件夹(回车使用 output_screen): ").strip().strip('"').strip("'")
    if not output_dir:
        output_dir = "output_screen"

    process_batch(image_folder, json_folder, output_dir)