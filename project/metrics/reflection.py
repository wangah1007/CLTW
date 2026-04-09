import cv2
import numpy as np
from scipy import ndimage
#from skimage import filters, feature, morphology
import os
import json
from pathlib import Path
from typing import List, Dict, Tuple, Optional

# ==================== 原始 AccurateReflectionAnalyzer 类定义 ====================
class AccurateReflectionAnalyzer:
    def __init__(self, window_size=8, brightness_threshold=1.5):
        self.window_size = window_size
        self.brightness_threshold = brightness_threshold
        
    def safe_imread(self, image_path):
        """安全读取图片"""
        try:
            path = Path(image_path)
            with open(path, 'rb') as f:
                image_array = np.frombuffer(f.read(), np.uint8)
            image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError("无法解码图片")
            return image
        except Exception as e:
            raise ValueError(f"无法读取图片 {image_path}: {str(e)}")
    
    def calculate_robust_features(self, image):
        """计算鲁棒的特征，准确识别反光"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            saturation = hsv[:, :, 1].astype(np.float32)
            value = hsv[:, :, 2].astype(np.float32)
        else:
            gray = image.astype(np.float32)
            saturation = value = np.zeros_like(gray)
        
        height, width = gray.shape
        
        # 1. 多尺度亮度分析
        local_means = np.zeros_like(gray, dtype=np.float32)
        local_maxs = np.zeros_like(gray, dtype=np.float32)
        
        for i in range(0, height - self.window_size + 1, self.window_size//2):
            for j in range(0, width - self.window_size + 1, self.window_size//2):
                window = gray[i:i+self.window_size, j:j+self.window_size]
                local_means[i:i+self.window_size, j:j+self.window_size] = np.mean(window)
                local_maxs[i:i+self.window_size, j:j+self.window_size] = np.max(window)
        
        # 2. 纹理特征 - 使用多种方法
        # 梯度幅值
        gradient_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        gradient_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
        
        # 局部熵
        entropy_map = np.zeros_like(gray, dtype=np.float32)
        for i in range(3, gray.shape[0]-3, 2):
            for j in range(3, gray.shape[1]-3, 2):
                window = gray[i-3:i+3, j-3:j+3]
                hist, _ = np.histogram(window, bins=8, range=(0, 255))
                hist = hist[hist > 0] / hist.sum()
                entropy_map[i-3:i+3, j-3:j+3] = -np.sum(hist * np.log2(hist))
        
        # 3. 颜色特征（如果是彩色图）
        if len(image.shape) == 3:
            # 饱和度异常检测：高亮度但低饱和度的区域
            saturation_anomaly = np.zeros_like(saturation)
            bright_regions = value > 200
            low_sat_regions = saturation < 50
            saturation_anomaly[bright_regions & low_sat_regions] = 1.0
        else:
            saturation_anomaly = np.zeros_like(gray)
        
        return gray, local_means, local_maxs, gradient_magnitude, entropy_map, saturation_anomaly, saturation, value
    
    def detect_reflection_with_confidence(self, image):
        """使用置信度机制检测反光区域"""
        gray, local_means, local_maxs, gradient_magnitude, entropy_map, saturation_anomaly, saturation, value = \
            self.calculate_robust_features(image)
        
        height, width = gray.shape
        global_mean = np.mean(gray)
        global_std = np.std(gray)
        
        # 初始化掩码和置信度图
        reflection_mask = np.zeros_like(gray, dtype=bool)
        confidence_map = np.zeros_like(gray, dtype=np.float32)
        
        # 检测反光候选区域
        for i in range(0, height - self.window_size + 1, self.window_size//2):
            for j in range(0, width - self.window_size + 1, self.window_size//2):
                window_coords = (slice(i, i+self.window_size), slice(j, j+self.window_size))
                
                # 特征提取
                window_mean = local_means[i, j]
                window_max = local_maxs[i, j]
                window_gradient = np.mean(gradient_magnitude[window_coords])
                window_entropy = np.mean(entropy_map[window_coords])
                
                # 多条件评分
                brightness_score = 0
                if window_mean > global_mean + self.brightness_threshold * global_std:
                    brightness_score = min(1.0, (window_mean - global_mean) / (2 * global_std))
                
                texture_score = 0
                if window_entropy < 2.0 and window_gradient < 25:
                    texture_score = 1.0 - min(1.0, window_entropy / 2.0)
                
                color_score = 0
                if len(image.shape) == 3:
                    window_sat_anomaly = np.mean(saturation_anomaly[window_coords])
                    color_score = window_sat_anomaly
                
                # 综合置信度
                total_confidence = (brightness_score * 0.5 + 
                                  texture_score * 0.3 + 
                                  color_score * 0.2)
                
                confidence_map[window_coords] = total_confidence
                
                # 动态阈值：根据区域特性调整
                dynamic_threshold = 0.3
                if window_max > 240:  # 非常亮的区域，降低阈值
                    dynamic_threshold = 0.2
                
                if total_confidence > dynamic_threshold:
                    reflection_mask[window_coords] = True
        
        # 后处理：连接相邻区域并去除噪声
        kernel = np.ones((3, 3), np.uint8)
        reflection_mask_uint8 = reflection_mask.astype(np.uint8) * 255
        reflection_mask_uint8 = cv2.morphologyEx(reflection_mask_uint8, cv2.MORPH_OPEN, kernel)
        reflection_mask_uint8 = cv2.morphologyEx(reflection_mask_uint8, cv2.MORPH_CLOSE, kernel)
        reflection_mask = reflection_mask_uint8 > 0
        
        # 计算反光严重程度
        severity_scores = []
        contours, _ = cv2.findContours(reflection_mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            if cv2.contourArea(contour) > 50:  # 只处理足够大的区域
                # 创建单个区域的掩码 - 修复：使用uint8类型
                single_mask = np.zeros_like(gray, dtype=np.uint8)
                cv2.fillPoly(single_mask, [contour], 255)
                single_mask_bool = single_mask > 0
                
                # 计算该区域的严重程度
                region_brightness = np.mean(gray[single_mask_bool])
                region_entropy = np.mean(entropy_map[single_mask_bool])
                region_gradient = np.mean(gradient_magnitude[single_mask_bool])
                
                # 严重程度评分
                brightness_severity = min(1.0, (region_brightness - global_mean) / (255 - global_mean))
                texture_severity = 1.0 - min(1.0, region_entropy / 3.0)  # 纹理越简单，反光越严重
                gradient_severity = 1.0 - min(1.0, region_gradient / 50.0)  # 梯度越小，反光越严重
                
                region_severity = (brightness_severity * 0.5 + 
                                 texture_severity * 0.3 + 
                                 gradient_severity * 0.2)
                
                severity_scores.append(region_severity)
        
        # 计算整体指标
        reflection_ratio = np.sum(reflection_mask) / (height * width)
        
        if np.sum(reflection_mask) > 0:
            # 亮度异常指数
            reflection_brightness = np.mean(gray[reflection_mask])
            brightness_anomaly = (reflection_brightness - global_mean) / global_std if global_std > 0 else 0
            brightness_index = min(1.0, reflection_ratio * np.log1p(abs(brightness_anomaly)))
            
            # 纹理丢失指数
            background_entropy = np.mean(entropy_map[~reflection_mask])
            reflection_entropy = np.mean(entropy_map[reflection_mask])
            if background_entropy > 0:
                texture_index = (background_entropy - reflection_entropy) / background_entropy
            else:
                texture_index = 0
            
            # 反光严重程度指数（基于区域评分）
            if severity_scores:
                severity_index = np.mean(severity_scores)
            else:
                severity_index = 0
        else:
            brightness_index = texture_index = severity_index = 0
        
        # 计算综合反光指数
        comprehensive_index = (severity_index * 0.6 + 
                             brightness_index * 0.3 + 
                             texture_index * 0.1)
        
        components = {
            'comprehensive_reflection_index': float(comprehensive_index),
            'brightness_anomaly_index': float(brightness_index),
            'texture_loss_index': float(texture_index),
            'reflection_severity_index': float(severity_index),
            'reflection_area_ratio': float(reflection_ratio),
            'max_reflection_brightness': float(np.max(gray[reflection_mask]) if np.sum(reflection_mask) > 0 else 0),
            'avg_reflection_brightness': float(np.mean(gray[reflection_mask]) if np.sum(reflection_mask) > 0 else 0),
            'global_brightness': float(global_mean)
        }
        
        features = {
            'local_means': local_means,
            'entropy_map': entropy_map,
            'gradient_magnitude': gradient_magnitude,
            'saturation': saturation,
            'confidence_map': confidence_map,
            'global_mean': global_mean,
            'global_std': global_std
        }
        
        return reflection_mask, components, features


def _clip01(x: float) -> float:
    return float(np.clip(x, 0.0, 1.0))

# ==================== TextRegionReflectionAnalyzer 类定义 ====================
class TextRegionReflectionAnalyzer:
    def __init__(self, window_size=8, brightness_threshold=1.5):
        self.window_size = window_size
        self.brightness_threshold = brightness_threshold
        self.base_analyzer = AccurateReflectionAnalyzer(window_size, brightness_threshold)
        
    def safe_imread(self, image_path):
        """安全读取图片"""
        return self.base_analyzer.safe_imread(image_path)
    
    def extract_text_regions(self, image, annotations):
        """
        根据标注信息提取文本区域
        使用原始图像的坐标点提取区域，然后缩放区域进行分析
        annotations: 标注列表，每个元素包含'points'字段
        """
        text_regions = []
        extraction_stats = {'total': 0, 'success': 0, 'failed': 0, 'reasons': {}}
        
        extraction_stats['total'] = len(annotations)
        
        # 检查图像尺寸
        h, w = image.shape[:2]
        
        for ann_idx, ann in enumerate(annotations):
            try:
                points = np.array(ann['points'], dtype=np.float32)
                
                if len(points) < 3:  # 至少需要3个点构成多边形
                    extraction_stats['failed'] += 1
                    extraction_stats['reasons'][ann_idx] = '点数不足'
                    continue
                
                # 获取文本区域边界框
                x_coords = points[:, 0]
                y_coords = points[:, 1]
                x_min, x_max = np.min(x_coords), np.max(x_coords)
                y_min, y_max = np.min(y_coords), np.max(y_coords)
                
                # 检查边界框有效性
                if x_max <= x_min or y_max <= y_min:
                    extraction_stats['failed'] += 1
                    extraction_stats['reasons'][ann_idx] = '无效边界框'
                    continue
                
                # 确保边界框在图像范围内
                x_min = max(0, int(x_min))
                y_min = max(0, int(y_min))
                x_max = min(w, int(x_max))
                y_max = min(h, int(y_max))
                
                # 再次检查调整后的边界框
                if x_max <= x_min or y_max <= y_min:
                    extraction_stats['failed'] += 1
                    extraction_stats['reasons'][ann_idx] = '调整后无效'
                    continue
                
                # 提取区域
                text_region = image[y_min:y_max, x_min:x_max]
                
                # 检查提取的区域是否有效
                if text_region.size == 0:
                    extraction_stats['failed'] += 1
                    extraction_stats['reasons'][ann_idx] = '空区域'
                    continue
                
                # 如果区域太小，跳过
                if text_region.shape[0] < 5 or text_region.shape[1] < 5:
                    extraction_stats['failed'] += 1
                    extraction_stats['reasons'][ann_idx] = '区域太小'
                    continue
                
                # 缩放区域到合理大小进行分析（保持宽高比）
                region_h, region_w = text_region.shape[:2]
                max_region_size = 200  # 最大区域尺寸
                
                if max(region_h, region_w) > max_region_size:
                    scale = max_region_size / max(region_h, region_w)
                    new_w = int(region_w * scale)
                    new_h = int(region_h * scale)
                    if new_w > 0 and new_h > 0:
                        text_region = cv2.resize(text_region, (new_w, new_h))
                    else:
                        extraction_stats['failed'] += 1
                        extraction_stats['reasons'][ann_idx] = '缩放后尺寸无效'
                        continue
                
                region_info = {
                    'region': text_region,
                    'bbox': (x_min, y_min, x_max, y_max),
                    'original_bbox': (x_min, y_min, x_max, y_max),
                    'scaled': max(region_h, region_w) > max_region_size,
                    'points': points,
                    'transcription': ann.get('transcription', ''),
                    'difficult': ann.get('difficult', False)
                }
                text_regions.append(region_info)
                extraction_stats['success'] += 1
                
            except Exception as e:
                extraction_stats['failed'] += 1
                extraction_stats['reasons'][ann_idx] = str(e)
        
        return text_regions, extraction_stats
    
    def analyze_region_reflection(self, region_image):
        """分析单个文本区域的反光情况"""
        if region_image.size == 0:
            return None, None
        
        # 使用基础分析器分析区域
        try:
            reflection_mask, components, features = self.base_analyzer.detect_reflection_with_confidence(region_image)
            return components, reflection_mask
        except Exception as e:
            print(f"区域分析失败: {str(e)}")
            return None, None
    
    def calculate_overall_reflection_index(self, region_analyses, image_size):
        """
        计算整张图片的综合反光指数
        region_analyses: 每个区域的分析结果列表
        image_size: (height, width) 图片尺寸
        """
        if not region_analyses:
            return {
                'overall_index': 0.0,
                'weighted_index': 0.0,
                'max_region_index': 0.0,
                'min_region_index': 0.0,
                'avg_region_index': 0.0,
                'reflective_region_count': 0,
                'total_regions': 0,
                'reflective_region_ratio': 0.0,
                'total_reflection_area_ratio': 0.0
            }
        
        valid_analyses = [a for a in region_analyses if a is not None]
        total_regions = len(valid_analyses)
        
        if total_regions == 0:
            return {
                'overall_index': 0.0,
                'weighted_index': 0.0,
                'max_region_index': 0.0,
                'min_region_index': 0.0,
                'avg_region_index': 0.0,
                'reflective_region_count': 0,
                'total_regions': 0,
                'reflective_region_ratio': 0.0,
                'total_reflection_area_ratio': 0.0
            }
        
        region_indices = []
        region_weights = []
        total_reflection_area = 0
        
        for analysis in valid_analyses:
            # 获取区域的反光指数
            region_index = analysis['comprehensive_reflection_index']
            region_indices.append(region_index)
            
            # 基于区域面积计算权重
            region_area_ratio = analysis['reflection_area_ratio']
            region_weights.append(region_area_ratio)
            
            # 累加反光面积比例
            total_reflection_area += region_area_ratio
        
        # 计算平均反光面积比例
        avg_reflection_area_ratio = total_reflection_area / total_regions if total_regions > 0 else 0
        
        # 计算各种统计量
        max_index = max(region_indices)
        min_index = min(region_indices)
        avg_index = np.mean(region_indices)
        
        # 计算加权平均（考虑区域重要性）
        if sum(region_weights) > 0:
            weighted_index = np.average(region_indices, weights=region_weights)
        else:
            weighted_index = avg_index
        
        # 统计反光区域数量
        reflection_threshold = 0.4  # 中度反光以上算反光区域
        reflective_regions = sum(1 for idx in region_indices if idx >= reflection_threshold)
        reflective_region_ratio = reflective_regions / total_regions if total_regions > 0 else 0
        
        # 整体指数：考虑最差区域、加权平均和反光面积比例
        overall_index = max(
            weighted_index * 0.5 + 
            max_index * 0.3 + 
            avg_reflection_area_ratio * 0.2, 
            avg_index
        )
        
        return {
            'overall_index': float(overall_index),
            'weighted_index': float(weighted_index),
            'max_region_index': float(max_index),
            'min_region_index': float(min_index),
            'avg_region_index': float(avg_index),
            'reflective_region_count': int(reflective_regions),
            'total_regions': int(total_regions),
            'reflective_region_ratio': float(reflective_region_ratio),
            'avg_reflection_area_ratio': float(avg_reflection_area_ratio)
        }
    
    # 已移除 create_visualization（可视化）相关逻辑


def parse_label_file(label_file_path):
    """解析Label.txt文件，返回图片路径到标注的映射"""
    image_to_annotations = {}
    
    with open(label_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
                
            parts = line.split('\t')
            if len(parts) >= 2:
                image_path = parts[0]
                try:
                    annotations = json.loads(parts[1])
                    image_to_annotations[image_path] = annotations
                except json.JSONDecodeError:
                    print(f"无法解析标注信息: {image_path}")
                    continue
    
    return image_to_annotations


def analyze_images_with_annotations(image_folder, label_file_path):
    """分析带有标注信息的图片"""
    analyzer = TextRegionReflectionAnalyzer()
    
    # 解析标注文件
    print(f"正在解析标注文件: {label_file_path}")
    image_to_annotations = parse_label_file(label_file_path)
    print(f"找到 {len(image_to_annotations)} 张图片的标注信息")
    
    results = []
    
    for image_path, annotations in image_to_annotations.items():
        try:
            # 构建完整图片路径
            full_image_path = os.path.join(image_folder, os.path.basename(image_path))
            
            if not os.path.exists(full_image_path):
                print(f"图片不存在: {full_image_path}")
                continue
            
            print(f"\n正在分析: {os.path.basename(image_path)}")
            print(f"找到 {len(annotations)} 个文本区域标注")
            
            # 读取原始图片
            image = analyzer.safe_imread(full_image_path)
            
            # 记录原始尺寸
            original_h, original_w = image.shape[:2]
            
            # 提取文本区域（使用原始坐标）
            text_regions, extraction_stats = analyzer.extract_text_regions(image, annotations)
            print(f"成功提取 {extraction_stats['success']} 个文本区域")
            print(f"提取失败 {extraction_stats['failed']} 个区域")
            
            if extraction_stats['failed'] > 0 and extraction_stats['failed'] < 10:
                # 打印前几个失败的原因
                print("失败原因示例:")
                for i, (idx, reason) in enumerate(list(extraction_stats['reasons'].items())[:3]):
                    print(f"  区域{idx+1}: {reason}")
            
            # 分析每个文本区域
            region_analyses = []
            for i, region_info in enumerate(text_regions):
                components, _ = analyzer.analyze_region_reflection(region_info['region'])
                if components:
                    region_analyses.append(components)
                else:
                    region_analyses.append(None)
            
            # 计算整体结果
            overall_result = analyzer.calculate_overall_reflection_index(
                region_analyses, image.shape[:2]
            )
            overall_index = overall_result['overall_index']
            
            # 收集结果
            result_info = {
                'image_name': os.path.basename(image_path),
                'overall_index': overall_index,
                'region_based_result': overall_result,
                'total_regions': len(annotations),
                'analyzed_regions': extraction_stats['success'],
                'failed_regions': extraction_stats['failed'],
                'original_size': (original_w, original_h)
            }
            results.append(result_info)
            
            # 打印简要结果
            print(f"整体反光指数（基于文本区域）: {overall_index:.3f}")
            print(f"反光文本区域: {overall_result['reflective_region_count']}/{overall_result['total_regions']}")
            print(f"平均反光面积比例: {overall_result['avg_reflection_area_ratio']:.1%}")
            
            # 打印提取失败率
            if extraction_stats['failed'] > 0 and extraction_stats['total'] > 0:
                fail_rate = extraction_stats['failed']/extraction_stats['total']
                print(f"提取失败率: {fail_rate:.1%}")
                
        except Exception as e:
            print(f"分析 {image_path} 时出错: {str(e)}")
            import traceback
            traceback.print_exc()
            continue
    
    return results


def analyze_images_with_json_folders(image_folder: str, json_folder: str, output_json_path: str = "reflection_scores.json"):
    """
    按“图片文件夹 + JSON文件夹”批量分析。
    JSON格式需包含字段: characters: [{transcription, points}, ...]
    """
    analyzer = TextRegionReflectionAnalyzer()
    image_folder_p = Path(image_folder)
    json_folder_p = Path(json_folder)

    if not image_folder_p.exists():
        raise FileNotFoundError(f"图片文件夹不存在: {image_folder}")
    if not json_folder_p.exists():
        raise FileNotFoundError(f"JSON文件夹不存在: {json_folder}")

    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    results = {}

    for img_path in image_folder_p.iterdir():
        if img_path.suffix.lower() not in image_extensions:
            continue

        json_path = json_folder_p / (img_path.stem + ".json")
        if not json_path.exists():
            print(f"警告: 未找到对应JSON文件 {json_path.name}，跳过 {img_path.name}")
            continue

        try:
            image = analyzer.safe_imread(str(img_path))
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            annotations = data.get("characters", [])

            # 提取文本区域（使用原始坐标）
            text_regions, extraction_stats = analyzer.extract_text_regions(image, annotations)

            # 分析每个文本区域
            region_analyses = []
            for region_info in text_regions:
                components, _ = analyzer.analyze_region_reflection(region_info['region'])
                region_analyses.append(components if components else None)

            overall_result = analyzer.calculate_overall_reflection_index(region_analyses, image.shape[:2])
            overall_index = overall_result['overall_index']

            results[img_path.name] = _clip01(float(overall_index))
            print(f"✓ {img_path.name} -> 反光指数: {overall_index:.6f} (有效区域: {extraction_stats['success']}/{extraction_stats['total']})")
        except Exception as e:
            print(f"✗ 处理失败 {img_path.name}: {e}")
            results[img_path.name] = None

    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n汇总已保存: {output_json_path}")
    return results


def analyze_images_with_labeltxt(image_folder: str, label_txt_path: str, output_json_path: str = "reflection_scores.json"):
    """
    使用 CLTW 的 Label.txt（格式: 相对路径\\tJSON数组）进行批量分析。
    """
    analyzer = TextRegionReflectionAnalyzer()
    image_folder_p = Path(image_folder)
    if not image_folder_p.exists():
        raise FileNotFoundError(f"图片文件夹不存在: {image_folder}")
    if not os.path.exists(label_txt_path):
        raise FileNotFoundError(f"Label.txt不存在: {label_txt_path}")

    image_to_annotations = parse_label_file(label_txt_path)
    results = {}

    for rel_path, annotations in image_to_annotations.items():
        img_name = os.path.basename(rel_path)
        img_path = image_folder_p / img_name
        if not img_path.exists():
            # 兜底：按 rel_path 拼
            img_path = image_folder_p / rel_path
            if not img_path.exists():
                print(f"图片不存在，跳过: {img_name}")
                continue

        try:
            image = analyzer.safe_imread(str(img_path))
            text_regions, extraction_stats = analyzer.extract_text_regions(image, annotations)

            region_analyses = []
            for region_info in text_regions:
                components, _ = analyzer.analyze_region_reflection(region_info['region'])
                region_analyses.append(components if components else None)

            overall_result = analyzer.calculate_overall_reflection_index(region_analyses, image.shape[:2])
            overall_index = overall_result['overall_index']

            results[img_name] = _clip01(float(overall_index))
            print(f"✓ {img_name} -> 反光指数: {overall_index:.6f} (有效区域: {extraction_stats['success']}/{extraction_stats['total']})")
        except Exception as e:
            print(f"✗ 处理失败 {img_name}: {e}")
            results[img_name] = None

    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n汇总已保存: {output_json_path}")
    return results


def print_summary(results):
    """打印分析总结"""
    if not results:
        print("没有分析结果")
        return
    
    print("\n" + "=" * 80)
    print("基于文本区域的反光分析总结")
    print("=" * 80)
    
    # 按整体指数排序
    results.sort(key=lambda x: x['overall_index'], reverse=True)
    
    print(f"{'序号':<4} {'图片名称':<25} {'区域指数':<8} {'反光区域':<12} {'提取统计':<15} {'级别':<10}")
    print("-" * 80)
    
    for i, result in enumerate(results, 1):
        idx = result['overall_index']
        
        # 确定级别
        if idx < 0.2:
            level = "无"
        elif idx < 0.4:
            level = "轻微"
        elif idx < 0.6:
            level = "中度"
        elif idx < 0.8:
            level = "严重"
        else:
            level = "极重"
        
        # 提取统计
        analyzed = result['analyzed_regions']
        total = result['total_regions']
        failed = result.get('failed_regions', 0)
        extract_rate = analyzed/total if total > 0 else 0
        
        reflective_count = result['region_based_result']['reflective_region_count']
        
        print(f"{i:<4} {result['image_name'][:22]:<25} {idx:.3f}     "
              f"{reflective_count}/{analyzed:<11} "
              f"{analyzed}/{total} ({extract_rate:.0%})  "
              f"{level:<10}")
    
    # 统计信息
    print("\n" + "=" * 80)
    print("统计信息:")
    print(f"- 分析图片总数: {len(results)}张")
    
    if results:
        # 计算各种平均值
        avg_region_index = np.mean([r['overall_index'] for r in results])
        avg_reflective_ratio = np.mean([r['region_based_result']['reflective_region_ratio'] for r in results])
        avg_extract_rate = np.mean([r['analyzed_regions']/r['total_regions'] for r in results if r['total_regions'] > 0])
        avg_reflection_area = np.mean([r['region_based_result']['avg_reflection_area_ratio'] for r in results])
        
        print(f"- 平均区域指数: {avg_region_index:.3f}")
        print(f"- 平均反光区域比例: {avg_reflective_ratio:.1%}")
        print(f"- 平均反光面积比例: {avg_reflection_area:.1%}")
        print(f"- 平均提取率: {avg_extract_rate:.1%}")
        
        # 按级别分类
        levels = {'无反光': 0, '轻微反光': 0, '中度反光': 0, '严重反光': 0, '极严重反光': 0}
        for result in results:
            idx = result['overall_index']
            if idx < 0.2:
                levels['无反光'] += 1
            elif idx < 0.4:
                levels['轻微反光'] += 1
            elif idx < 0.6:
                levels['中度反光'] += 1
            elif idx < 0.8:
                levels['严重反光'] += 1
            else:
                levels['极严重反光'] += 1
        
        print("\n反光级别分布:")
        total_images = len(results)
        for level, count in levels.items():
            if count > 0:
                percentage = count / total_images * 100
                print(f"  {level}: {count}张 ({percentage:.1f}%)")
        
        # 最差的3张图片
        print("\n反光最严重的3张图片:")
        for i in range(min(3, len(results))):
            result = results[i]
            print(f"  {i+1}. {result['image_name']}: 指数={result['overall_index']:.3f}, "
                  f"反光区域={result['region_based_result']['reflective_region_count']}/{result['analyzed_regions']}")
        
        # 最好的3张图片
        print("\n反光最轻微的3张图片:")
        for i in range(min(3, len(results))):
            result = results[-i-1] if i < len(results) else results[-1]
            print(f"  {i+1}. {result['image_name']}: 指数={result['overall_index']:.3f}, "
                  f"反光区域={result['region_based_result']['reflective_region_count']}/{result['analyzed_regions']}")


# ==================== 主函数 ====================
if __name__ == "__main__":
    print("基于文本区域的反光分析系统")
    print("=" * 60)
    
    folder_path = input("请输入图片文件夹路径: ").strip().strip('"').strip("'")
    json_folder = input("请输入JSON文件夹路径: ").strip().strip('"').strip("'")
    output_json_path = input("请输入输出JSON文件路径(回车使用 reflection_scores.json): ").strip()
    if not output_json_path:
        output_json_path = "reflection_scores.json"

    analyze_images_with_json_folders(folder_path, json_folder, output_json_path)