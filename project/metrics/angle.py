import os
import cv2
import json
import numpy as np


# =========================
# 解决 Windows 中文路径
# =========================

def cv_imread(path):
    if not os.path.exists(path):
        return None
    data = np.fromfile(path, dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    return img


def _clip01(x: float) -> float:
    return float(np.clip(x, 0.0, 1.0))


def _resolve_image_path(img_dir: str, img_rel_path: str) -> str:
    """
    CLTW 的 Label.txt 第一列通常是类似 “扭曲斜角度/xxx.jpg” 的相对路径。
    这里做多级兜底：
    - 先用 img_dir + rel_path
    - 不存在则回退到 img_dir 下的 basename
    - 仍不存在则按 stem 在 img_dir 下尝试匹配常见图片扩展名
    """
    img_dir = str(img_dir)
    rel = img_rel_path.strip().lstrip("/\\")

    cand1 = os.path.normpath(os.path.join(img_dir, rel))
    if os.path.exists(cand1):
        return cand1

    base = os.path.basename(rel)
    cand2 = os.path.normpath(os.path.join(img_dir, base))
    if os.path.exists(cand2):
        return cand2

    stem = os.path.splitext(base)[0]
    for ext in (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"):
        cand3 = os.path.normpath(os.path.join(img_dir, stem + ext))
        if os.path.exists(cand3):
            return cand3

    # 保底返回 cand2（用于打印排查）
    return cand2


# =========================
# 角度计算
# =========================

def angle(p1, p2):
    return np.arctan2(p2[1]-p1[1], p2[0]-p1[0])


# =========================
# 透视指标计算
# =========================

def compute_perspective(data):

    widths=[]
    angles=[]
    traps=[]
    paras=[]

    for item in data:

        pts=np.array(item["points"])

        if pts.shape[0]!=4:
            continue

        p1,p2,p3,p4=pts

        top=np.linalg.norm(p1-p2)
        bottom=np.linalg.norm(p3-p4)

        widths.append((top+bottom)/2)

        traps.append(abs(top-bottom)/(top+bottom+1e-6))

        angles.append(angle(p1,p2))

        phi1=angle(p1,p4)
        phi2=angle(p2,p3)

        paras.append(abs(phi1-phi2))


    widths=np.array(widths)
    angles=np.array(angles)
    traps=np.array(traps)
    paras=np.array(paras)

    if len(angles)<2:
        return 0,angles


    # --------------------
    # 去除整体旋转
    # --------------------

    angles=angles-np.mean(angles)


    # --------------------
    # 角度区间
    # --------------------

    angle_range=np.max(angles)-np.min(angles)

    angle_range_score=np.tanh(angle_range*2)


    # --------------------
    # 连续变化检测
    # --------------------

    diffs=np.abs(np.diff(angles))

    smoothness=np.mean(diffs)

    angle_score=angle_range_score*np.exp(-smoothness)


    # --------------------
    # 宽度透视
    # --------------------

    width_range=(np.max(widths)-np.min(widths))/(np.mean(widths)+1e-6)

    width_score=np.tanh(width_range*3)


    # --------------------
    # 梯形
    # --------------------

    trap_score=np.tanh(np.mean(traps)*3)


    # --------------------
    # 平行四边形
    # --------------------

    para_score=np.tanh(np.mean(paras)*2)


    score=(
        0.45*angle_score+
        0.25*width_score+
        0.20*trap_score+
        0.10*para_score
    )

    score=score**1.5

    return score,angles


# =========================
# 读取TXT标注
# =========================

def load_txt(txt_path):

    dataset = {}

    with open(txt_path, "r", encoding="utf-8") as f:

        for line in f:

            line = line.strip()

            if not line:
                continue

            parts = line.split("\t")

            if len(parts) < 2:
                continue

            img_path = parts[0]
            json_str = parts[1]

            try:
                data = json.loads(json_str)
            except:
                continue

            dataset[img_path] = data

    return dataset


# =========================
# 主程序（无可视化，输出得分JSON）
# =========================

def process(img_dir, txt_path, output_json_path):
    dataset = load_txt(txt_path)
    print(f"共找到 {len(dataset)} 张图片记录")

    results = {}
    for img_rel_path, data in dataset.items():
        img_path = _resolve_image_path(img_dir, img_rel_path)
        img_name = os.path.basename(img_rel_path)
        print(f"处理: {img_rel_path}")
        img = cv_imread(img_path)
        if img is None:
            print(f"  ⚠️ 图片读取失败，跳过: {img_path}")
            results[img_name] = None
            continue
        score, angles = compute_perspective(data)
        results[img_name] = _clip01(float(score))
        print(f"  ✅ 得分: {score:.4f}")

    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n透视得分已保存到: {output_json_path}")
    return results


# =========================
# 运行入口
# =========================

if __name__ == "__main__":

    img_dir = r"images"          # 图片文件夹
    txt_path = r"Label.txt" # 标注文件
    output_json_path = r"perspective_scores.json"   # 输出JSON

    process(img_dir, txt_path, output_json_path)