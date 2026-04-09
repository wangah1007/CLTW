import json
import os
from pathlib import Path

# CLTW 目录约定（实测）：
# <root>/<split>/<类别>/<类别>/
#   - 图片: *.jpg
#   - 标注: Label.txt（仅 angle 依赖）
#   - JSON标注：可能与图片同目录（img_stem.json），也可能在 ocr_json/ 等子目录

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}


def _infer_category_from_folder(folder: Path) -> str | None:
    # folder 形如 .../<类别>/<类别>
    if folder.parent.name == folder.name:
        return folder.name
    return folder.name if folder.name in {"扭曲斜角度", "小图虚图", "反光", "遮挡", "光线不足", "屏幕", "特殊材质", "复杂背景"} else None


def _load_label_map(label_path: Path):
    """
    Label.txt 格式：<相对路径>\t<json数组>
    返回 dict: {basename.jpg: characters(list)}
    """
    mapping = {}
    with open(label_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t", 1)
            if len(parts) != 2:
                continue
            img_rel, json_str = parts
            try:
                chars = json.loads(json_str)
            except Exception:
                continue
            mapping[os.path.basename(img_rel)] = chars
    return mapping


def _find_json_folder(folder: Path) -> Path | None:
    # 优先使用图片同目录（json 与图片放一起）
    if any(folder.glob("*.json")):
        return folder

    # 其次 ocr_json / json / ocr 等子目录
    for name in ("ocr_json", "json", "ocr", "ocr-json", "ocrJson"):
        p = folder / name
        if p.exists() and p.is_dir():
            return p
    return None


def run_one_folder(folder: Path):
    """
    对一个“图片文件夹”（CLTW 的 .../<类>/<类>）输出 result.json
    """
    category = _infer_category_from_folder(folder)
    if not category:
        return

    label_path = folder / "Label.txt"
    json_folder = _find_json_folder(folder)
    output_path = folder / "result.json"

    # 延迟 import，避免 runner 在缺依赖时报错
    # 通过相对包导入，便于直接运行本脚本
    from metrics import angle, blur, reflection, occlusion, lowlight, screen, material, background

    if category == "扭曲斜角度":
        if not label_path.exists():
            print(f"[SKIP] 缺少 Label.txt: {label_path}")
            return
        angle.process(str(folder), str(label_path), str(output_path))
        return

    # 非 angle：仅使用 json_folder（如 ocr_json）。不存在则跳过，不再使用 Label.txt。
    scores = {}

    if json_folder and any(json_folder.glob("*.json")):
        if category == "反光":
            reflection.analyze_images_with_json_folders(str(folder), str(json_folder), str(output_path))
            return
        if category == "光线不足":
            lowlight.analyze_images_with_json_folders(str(folder), str(json_folder), str(output_path))
            return
        if category == "遮挡":
            occlusion.process_batch(str(folder), str(json_folder), str(output_path))
            return
        if category == "小图虚图":
            # blur.py 已有 process_batch，但它要求 json_folder
            # 这里直接复用 blur.process_batch 的输出路径
            blur.process_batch(str(folder), str(json_folder), str(output_path))
            return
        if category == "屏幕":
            # screen.py 的 batch 输出为目录；这里按要求输出到同目录 json
            tmp_dir = folder  # 直接输出到图片文件夹
            screen.process_batch(str(folder), str(json_folder), str(tmp_dir))
            # 统一命名：把 screen 输出 json 复制/重写到 result.json
            screen_json = tmp_dir / "screen_texture_index.json"
            if screen_json.exists():
                with open(screen_json, "r", encoding="utf-8") as f:
                    scores = json.load(f)
                with open(output_path, "w", encoding="utf-8") as f:
                    json.dump(scores, f, ensure_ascii=False, indent=2)
            return
        if category == "特殊材质":
            material.process_batch(str(folder), str(json_folder), str(output_path))
            return
        if category == "复杂背景":
            background.process_batch(str(folder), str(json_folder), str(output_path))
            return

    print(f"[SKIP] {category} 未找到JSON标注文件夹或无.json文件: {folder}")
    return


def run_cltw(root_dir: str):
    root = Path(root_dir)
    if not root.exists():
        raise FileNotFoundError(f"CLTW根目录不存在: {root_dir}")

    # 找到所有包含 Label.txt 的“类别图片文件夹”
    label_files = list(root.rglob("Label.txt"))
    if not label_files:
        print("[WARN] 未找到任何 Label.txt")
        return

    for label_path in label_files:
        folder = label_path.parent
        category = _infer_category_from_folder(folder)
        if not category:
            continue
        print(f"\n=== 处理: {folder} (类别: {category}) ===")
        run_one_folder(folder)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, required=True, help="CLTW根目录，例如 D:/study/experiment/照片文件夹/CLTW/CLTW")
    args = parser.parse_args()

    run_cltw(args.root)
