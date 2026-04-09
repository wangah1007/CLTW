"""
文本遮挡程度测量系统（原始版本 - 不翻转结果）
专注于计算每个检测框内被遮挡面积与框总面积的比例，并综合得到整张图片的遮挡程度
使用原始的、未翻转的检测结果
"""

import cv2
import numpy as np
import json
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
from pathlib import Path
import os
import warnings
warnings.filterwarnings('ignore')

def _clip01(x: float) -> float:
    return float(np.clip(x, 0.0, 1.0))

# 尝试导入深度学习相关库
try:
    import torch
    import torchvision.transforms as transforms
    from torchvision.models.segmentation import deeplabv3_resnet50, DeepLabV3_ResNet50_Weights
    from PIL import Image
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("警告: PyTorch未安装，将使用传统方法。建议安装PyTorch以获得更好的检测效果。")

@dataclass
class TextBox:
    """文本框数据结构"""
    transcription: str
    points: np.ndarray  # 4个点的坐标，形状为(4, 2)
    bbox: Tuple[int, int, int, int] = None  # (x_min, y_min, x_max, y_max)
    
    def __post_init__(self):
        # 将点列表转换为numpy数组
        if isinstance(self.points, list):
            self.points = np.array(self.points, dtype=np.int32)
        # 计算边界框
        self.bbox = (
            int(np.min(self.points[:, 0])),
            int(np.min(self.points[:, 1])),
            int(np.max(self.points[:, 0])),
            int(np.max(self.points[:, 1]))
        )


@dataclass
class BoxOcclusionResult:
    """单个框的遮挡结果"""
    transcription: str
    bbox_coords: Tuple[int, int, int, int]
    bbox_area: int  # 框的总面积（像素数）
    text_area: int  # 文本区域面积（像素数）
    occluded_area: int  # 被遮挡的面积（像素数）= 框总面积 - 文本区域面积
    occlusion_ratio: float  # 遮挡比例 = 被遮挡面积 / 框总面积
    occlusion_mask: np.ndarray  # 遮挡掩膜（仅在框内）
    text_mask: np.ndarray  # 文本区域掩膜（仅在框内）


@dataclass
class ImageOcclusionResult:
    """整张图片的遮挡结果"""
    image_path: str
    total_boxes: int
    total_bbox_area: int  # 所有框的总面积
    total_occluded_area: int  # 所有框中被遮挡的总面积
    global_occlusion_ratio: float  # 全局遮挡比例
    box_results: List[BoxOcclusionResult]  # 每个框的详细结果
    average_box_occlusion_ratio: float  # 所有框的平均遮挡比例
    severity_level: str  # 严重程度等级


class OcclusionMeasurer:
    """遮挡程度测量器"""
    
    def __init__(self, 
                 occlusion_color_ranges: Dict[str, Tuple] = None,
                 min_occlusion_size: int = 10,
                 use_adaptive_detection: bool = True,
                 use_model: bool = True,
                 model_device: str = 'auto',
                 invert_ratio: bool = False):  # 默认不翻转
        """
        初始化测量器
        
        参数:
            occlusion_color_ranges: 遮挡物颜色范围（HSV格式）
            min_occlusion_size: 最小遮挡物尺寸（像素）
            use_adaptive_detection: 是否使用自适应检测方法（传统方法）
            use_model: 是否使用深度学习模型（推荐）
            model_device: 模型运行设备 ('auto', 'cuda', 'cpu')
            invert_ratio: 是否翻转遮挡比例（默认False，使用原始结果）
        """
        # 默认遮挡物颜色范围（HSV格式）
        if occlusion_color_ranges is None:
            self.occlusion_colors = {
                'green_leaves': ((35, 50, 50), (85, 255, 255)),  # 绿色树叶
                'brown_leaves': ((10, 50, 50), (25, 255, 255)),  # 棕色树叶
                'yellow_leaves': ((20, 50, 50), (35, 255, 255)),  # 黄色树叶
                'dark_objects': ((0, 0, 0), (180, 255, 100)),    # 深色物体
            }
        else:
            self.occlusion_colors = occlusion_color_ranges
        
        self.min_occlusion_size = min_occlusion_size
        self.use_adaptive_detection = use_adaptive_detection
        self.use_model = use_model and TORCH_AVAILABLE
        self.invert_ratio = invert_ratio  # 是否翻转遮挡比例
        
        # 初始化模型
        self.model = None
        self.model_transform = None
        self.device = None
        
        if self.use_model:
            self._init_model(model_device)
    
    def _init_model(self, device: str = 'auto'):
        """初始化深度学习模型"""
        if not TORCH_AVAILABLE:
            print("警告: PyTorch未安装，将使用传统方法")
            self.use_model = False
            return
        
        try:
            # 设置设备
            if device == 'auto':
                self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            else:
                self.device = torch.device(device)
            
            print(f"正在加载DeepLabV3模型 (设备: {self.device})...")
            
            # 加载预训练的DeepLabV3模型
            weights = DeepLabV3_ResNet50_Weights.DEFAULT
            self.model = deeplabv3_resnet50(weights=weights)
            self.model.eval()
            self.model.to(self.device)
            
            # 设置图像预处理
            self.model_transform = weights.transforms()
            
            print("模型加载成功！")
            
        except Exception as e:
            print(f"模型加载失败: {e}")
            print("将使用传统方法")
            self.use_model = False
    
    def parse_txt_detection_results(self, txt_path: str, image_filename: str = None) -> List[TextBox]:
        """解析txt格式的检测结果文件"""
        with open(txt_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        text_boxes = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # 分割文件名和检测结果
            if '\t' in line:
                parts = line.split('\t', 1)
                current_filename = parts[0].strip()
                json_str = parts[1].strip()
            else:
                # 如果没有制表符，假设整行都是JSON
                current_filename = ""
                json_str = line
            
            # 如果指定了图片文件名，只处理该文件的数据
            if image_filename and os.path.basename(current_filename) != image_filename:
                continue
                
            try:
                # 解析JSON数据
                detection_data = json.loads(json_str)
                
                for item in detection_data:
                    text_box = TextBox(
                        transcription=item.get('transcription', ''),
                        points=item['points']
                    )
                    text_boxes.append(text_box)
                
                # 如果指定了文件名，找到后立即返回
                if image_filename and current_filename == image_filename:
                    return text_boxes
                    
            except json.JSONDecodeError as e:
                print(f"警告：解析JSON时出错: {e}")
                continue
        
        return text_boxes
    
    def load_detection_results(self, file_path: str, image_filename: str = None) -> List[TextBox]:
        """加载检测结果文件（自动识别格式）"""
        file_ext = Path(file_path).suffix.lower()
        
        if file_ext == '.json':
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # 兼容两种JSON格式：
            # 1) 直接是list: [{points, transcription}, ...]
            # 2) OCR JSON: {filename, image_size, ..., characters:[{points, transcription}, ...]}
            if isinstance(data, dict) and "characters" in data:
                data_list = data.get("characters", [])
            elif isinstance(data, list):
                data_list = data
            else:
                data_list = []

            text_boxes = []
            for item in data_list:
                if not isinstance(item, dict) or "points" not in item:
                    continue
                text_box = TextBox(
                    transcription=item.get('transcription', ''),
                    points=item['points']
                )
                text_boxes.append(text_box)

            return text_boxes
            
        elif file_ext == '.txt':
            return self.parse_txt_detection_results(file_path, image_filename)
            
        else:
            raise ValueError(f"不支持的文件格式: {file_ext}。支持.json和.txt格式")
    
    def create_box_mask(self, bbox: TextBox, image_shape: Tuple[int, int]) -> np.ndarray:
        """
        创建文本框的掩膜
        
        参数:
            bbox: 文本框信息
            image_shape: 图像形状 (height, width)
            
        返回:
            mask: 框的掩膜
        """
        mask = np.zeros(image_shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, [bbox.points], 255)
        return mask
    
    def extract_text_region_with_model(self, box_region: np.ndarray, full_image: np.ndarray, bbox: TextBox) -> np.ndarray:
        """
        使用深度学习模型辅助提取文本区域
        结合语义分割和传统方法，更准确地识别文本
        
        参数:
            box_region: 框内的图像区域
            full_image: 完整图像
            bbox: 文本框信息
            
        返回:
            text_mask: 文本区域的二值掩膜
        """
        if not self.use_model or self.model is None:
            return self.extract_text_region_improved(box_region)
        
        try:
            # 首先使用改进的传统方法
            text_mask_traditional = self.extract_text_region_improved(box_region)
            
            # 使用模型进行辅助验证和优化
            x_min, y_min, x_max, y_max = bbox.bbox
            
            # 扩展区域以获得更好的上下文
            margin = 30
            h, w = full_image.shape[:2]
            x_min_exp = max(0, x_min - margin)
            y_min_exp = max(0, y_min - margin)
            x_max_exp = min(w, x_max + margin)
            y_max_exp = min(h, y_max + margin)
            
            expanded_region = full_image[y_min_exp:y_max_exp, x_min_exp:x_max_exp]
            
            # 转换为RGB
            if len(expanded_region.shape) == 3:
                rgb_image = cv2.cvtColor(expanded_region, cv2.COLOR_BGR2RGB)
            else:
                rgb_image = cv2.cvtColor(expanded_region, cv2.COLOR_GRAY2RGB)
            
            # 将numpy数组转换为PIL Image（model_transform需要PIL Image）
            pil_image = Image.fromarray(rgb_image)
            
            # 预处理和推理
            input_tensor = self.model_transform(pil_image).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                output = self.model(input_tensor)['out'][0]
                predictions = output.argmax(0).cpu().numpy()
            
            # 使用语义分割结果来过滤误检
            # 提取结构区域（非天空、非背景等）
            structure_mask = (predictions > 0).astype(np.uint8) * 255
            
            # 裁剪回原始框区域
            crop_y_min = y_min - y_min_exp
            crop_y_max = crop_y_min + (y_max - y_min)
            crop_x_min = x_min - x_min_exp
            crop_x_max = crop_x_min + (x_max - x_min)
            
            structure_mask_crop = structure_mask[crop_y_min:crop_y_max, crop_x_min:crop_x_max]
            
            # 确保尺寸一致
            if structure_mask_crop.shape != text_mask_traditional.shape:
                structure_mask_crop = cv2.resize(structure_mask_crop, 
                                               (text_mask_traditional.shape[1], text_mask_traditional.shape[0]),
                                               interpolation=cv2.INTER_NEAREST)
            
            # 结合传统方法和模型结果
            # 如果传统方法检测到文本，使用模型结果进行优化
            # 如果传统方法没有检测到文本，直接使用传统方法结果（不过滤）
            if np.sum(text_mask_traditional) > 0:
                # 有文本检测结果，使用模型进行优化
                text_mask_refined = cv2.bitwise_and(text_mask_traditional, structure_mask_crop)
                # 如果优化后文本太少，回退到原始结果
                if np.sum(text_mask_refined) < np.sum(text_mask_traditional) * 0.3:
                    return text_mask_traditional
                return text_mask_refined
            else:
                # 传统方法没有检测到文本，直接返回（不过滤）
                return text_mask_traditional
            
        except Exception as e:
            print(f"模型推理出错: {e}，使用改进的传统方法")
            return self.extract_text_region_improved(box_region)
    
    def extract_text_region_improved(self, box_region: np.ndarray) -> np.ndarray:
        """
        改进的文本区域提取方法
        使用多种技术组合，更准确地识别文本，避免将标点符号误判为遮挡
        
        参数:
            box_region: 框内的图像区域
            
        返回:
            text_mask: 文本区域的二值掩膜
        """
        if box_region.size == 0:
            return np.zeros((0, 0), dtype=np.uint8)
        
        # 转换为灰度图
        if len(box_region.shape) == 3:
            gray = cv2.cvtColor(box_region, cv2.COLOR_BGR2GRAY)
        else:
            gray = box_region
        
        h, w = gray.shape
        
        # 首先尝试简单方法（更宽松）
        text_mask_simple = self._extract_text_simple(gray)
        
        # 如果简单方法检测到足够的文本，直接返回
        if np.sum(text_mask_simple) > h * w * 0.01:  # 至少检测到框面积的1%
            return text_mask_simple
        
        # 方法1：多尺度自适应阈值
        # 使用不同大小的窗口来适应不同大小的文字
        binary1 = cv2.adaptiveThreshold(
            gray, 255, 
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, 11, 2
        )
        
        binary2 = cv2.adaptiveThreshold(
            gray, 255, 
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, 15, 3
        )
        
        # 方法2：OTSU阈值
        _, otsu_binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # 方法3：局部对比度增强 + 阈值
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        _, enhanced_binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # 合并多种方法
        combined = cv2.bitwise_or(binary1, binary2)
        combined = cv2.bitwise_or(combined, otsu_binary)
        combined = cv2.bitwise_or(combined, enhanced_binary)
        
        # 形态学操作：先闭运算连接笔画，再开运算去除噪点
        kernel_close = np.ones((2, 2), np.uint8)
        combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel_close, iterations=2)
        
        kernel_open = np.ones((2, 2), np.uint8)
        combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel_open)
        
        # 连通域分析，过滤小噪点和保留文本
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(combined, connectivity=8)
        text_mask = np.zeros_like(combined)
        
        if num_labels > 1:
            areas = [stats[i, cv2.CC_STAT_AREA] for i in range(1, num_labels)]
            if areas:
                # 使用更宽松的阈值：基于面积分布
                areas_sorted = sorted(areas)
                median_area = np.median(areas)
                q1 = np.percentile(areas, 25)
                q3 = np.percentile(areas, 75)
                
                # 更宽松的阈值：保留更多区域作为文本
                # 最小面积：非常小，只过滤明显的噪点
                min_area = max(2, int(q1 * 0.1))  # 更宽松：只过滤非常小的噪点
                # 最大面积：允许更大的区域（可能是大字符或连笔）
                max_area = min(h * w * 0.8, int(q3 * 5))  # 更宽松：允许更大的区域
                
                # 如果过滤后没有文本，进一步放宽条件
                filtered_count = 0
                for i in range(1, num_labels):
                    area = stats[i, cv2.CC_STAT_AREA]
                    width = stats[i, cv2.CC_STAT_WIDTH]
                    height = stats[i, cv2.CC_STAT_HEIGHT]
                    aspect_ratio = width / height if height > 0 else 0
                    
                    # 更宽松的条件
                    if min_area <= area <= max_area:
                        # 宽高比范围更宽松：0.05到20
                        if 0.05 <= aspect_ratio <= 20:
                            text_mask[labels == i] = 255
                            filtered_count += 1
                
                # 如果过滤后文本太少（少于总连通域的10%），使用更宽松的策略
                if filtered_count < max(1, (num_labels - 1) * 0.1):
                    # 回退到更简单的策略：只过滤非常小的噪点
                    min_area_simple = max(1, int(median_area * 0.01))  # 只保留大于中位数1%的区域
                    text_mask = np.zeros_like(combined)
                    for i in range(1, num_labels):
                        if stats[i, cv2.CC_STAT_AREA] >= min_area_simple:
                            text_mask[labels == i] = 255
        
        # 如果仍然没有检测到文本，直接使用合并结果（不过滤）
        if np.sum(text_mask) == 0:
            # 最后的备选方案：使用原始合并结果，只去除极小的噪点
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(combined, connectivity=8)
            text_mask = np.zeros_like(combined)
            min_area_final = max(1, h * w * 0.001)  # 至少是框面积的0.1%
            for i in range(1, num_labels):
                if stats[i, cv2.CC_STAT_AREA] >= min_area_final:
                    text_mask[labels == i] = 255
        
        return text_mask
    
    def _extract_text_simple(self, gray: np.ndarray) -> np.ndarray:
        """
        简单的文本提取方法（更宽松，作为备选）
        
        参数:
            gray: 灰度图像
            
        返回:
            text_mask: 文本区域的二值掩膜
        """
        h, w = gray.shape
        
        # 使用自适应阈值
        binary = cv2.adaptiveThreshold(
            gray, 255, 
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, 11, 2
        )
        
        # 使用OTSU阈值
        _, otsu_binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # 合并
        combined = cv2.bitwise_or(binary, otsu_binary)
        
        # 轻微的形态学操作
        kernel = np.ones((2, 2), np.uint8)
        combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel, iterations=1)
        
        # 只过滤极小的噪点（面积小于5像素）
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(combined, connectivity=8)
        text_mask = np.zeros_like(combined)
        
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] >= 3:  # 只过滤小于3像素的噪点
                text_mask[labels == i] = 255
        
        return text_mask
    
    def detect_occlusion_objects(self, box_region: np.ndarray, box_mask_region: np.ndarray) -> np.ndarray:
        """
        直接检测框内的遮挡物（绿色、棕色等自然物体）
        
        参数:
            box_region: 框内的图像区域
            box_mask_region: 框的掩膜区域
            
        返回:
            occlusion_mask: 遮挡物掩膜
        """
        if box_region.size == 0:
            return np.zeros_like(box_mask_region)
        
        # 转换为HSV颜色空间
        hsv = cv2.cvtColor(box_region, cv2.COLOR_BGR2HSV)
        
        # 创建遮挡物掩膜
        occlusion_mask = np.zeros(box_mask_region.shape, dtype=np.uint8)
        
        # 检测绿色遮挡物（树叶等）
        lower_green = np.array([35, 50, 50])
        upper_green = np.array([85, 255, 255])
        green_mask = cv2.inRange(hsv, lower_green, upper_green)
        occlusion_mask = cv2.bitwise_or(occlusion_mask, green_mask)
        
        # 检测棕色遮挡物
        lower_brown = np.array([10, 50, 50])
        upper_brown = np.array([25, 255, 255])
        brown_mask = cv2.inRange(hsv, lower_brown, upper_brown)
        occlusion_mask = cv2.bitwise_or(occlusion_mask, brown_mask)
        
        # 检测黄色遮挡物
        lower_yellow = np.array([20, 50, 50])
        upper_yellow = np.array([35, 255, 255])
        yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
        occlusion_mask = cv2.bitwise_or(occlusion_mask, yellow_mask)
        
        # 检测深色遮挡物（阴影、深色物体）
        gray = cv2.cvtColor(box_region, cv2.COLOR_BGR2GRAY)
        _, dark_mask = cv2.threshold(gray, 60, 255, cv2.THRESH_BINARY_INV)
        # 只保留足够大的深色区域（避免误检文字）
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(dark_mask, connectivity=8)
        dark_mask_cleaned = np.zeros_like(dark_mask)
        h, w = gray.shape
        min_dark_area = max(10, h * w * 0.01)  # 至少是框面积的1%
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] >= min_dark_area:
                dark_mask_cleaned[labels == i] = 255
        occlusion_mask = cv2.bitwise_or(occlusion_mask, dark_mask_cleaned)
        
        # 形态学操作：连接和清理
        kernel = np.ones((3, 3), np.uint8)
        occlusion_mask = cv2.morphologyEx(occlusion_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        occlusion_mask = cv2.morphologyEx(occlusion_mask, cv2.MORPH_OPEN, kernel, iterations=1)
        
        # 确保只在框内
        occlusion_mask = cv2.bitwise_and(occlusion_mask, box_mask_region)
        
        return occlusion_mask
    
    def extract_text_region_traditional(self, box_region: np.ndarray) -> np.ndarray:
        """
        使用传统方法提取框内的文本区域（文字笔画）
        
        参数:
            box_region: 框内的图像区域
            
        返回:
            text_mask: 文本区域的二值掩膜
        """
        if box_region.size == 0:
            return np.zeros((0, 0), dtype=np.uint8)
        
        # 转换为灰度图
        if len(box_region.shape) == 3:
            gray = cv2.cvtColor(box_region, cv2.COLOR_BGR2GRAY)
        else:
            gray = box_region
        
        # 方法1：使用自适应阈值提取文字
        binary = cv2.adaptiveThreshold(
            gray, 255, 
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, 11, 2
        )
        
        # 方法2：使用OTSU阈值
        _, otsu_binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # 方法3：使用形态学梯度检测文字边缘
        kernel = np.ones((3, 3), np.uint8)
        gradient = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, kernel)
        _, gradient_binary = cv2.threshold(gradient, 30, 255, cv2.THRESH_BINARY)
        
        # 合并多种方法的结果
        combined = cv2.bitwise_or(binary, otsu_binary)
        combined = cv2.bitwise_or(combined, gradient_binary)
        
        # 形态学操作增强文字笔画
        kernel_close = np.ones((2, 2), np.uint8)
        combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel_close)
        
        kernel_open = np.ones((2, 2), np.uint8)
        combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel_open)
        
        # 去除小连通域（噪点）
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(combined, connectivity=8)
        text_mask = np.zeros_like(combined)
        
        areas = [stats[i, cv2.CC_STAT_AREA] for i in range(1, num_labels)]
        if areas:
            median_area = np.median(areas)
            min_area = max(5, int(median_area * 0.1))
        else:
            min_area = 5
        
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] >= min_area:
                text_mask[labels == i] = 255
        
        return text_mask
    
    def detect_occlusion_in_box(self, image: np.ndarray, bbox: TextBox) -> Tuple[np.ndarray, np.ndarray, int, int, int]:
        """
        检测单个框内的遮挡物
        新方法：先检测文本区域，剩余部分为遮挡区域
        
        参数:
            image: 原始图像
            bbox: 文本框信息
            
        返回:
            occlusion_mask: 框内的遮挡掩膜（完整图像大小）
            text_mask: 文本区域掩膜（完整图像大小）
            bbox_area: 框的总面积
            text_area: 文本区域面积
            occluded_area: 被遮挡的面积
        """
        # 创建框的掩膜
        box_mask = self.create_box_mask(bbox, image.shape[:2])
        
        # 提取框内区域
        x_min, y_min, x_max, y_max = bbox.bbox
        box_region = image[y_min:y_max, x_min:x_max].copy()
        box_mask_region = box_mask[y_min:y_max, x_min:x_max]
        
        if box_region.size == 0:
            return np.zeros_like(box_mask_region), 0, 0
        
        # 恢复之前的逻辑：遮挡面积 = 框总面积 - 文本区域面积
        # 但改进文本检测，避免将遮挡物误判为文本
        
        # 提取文本区域（文字笔画）
        if self.use_model:
            text_mask_region = self.extract_text_region_with_model(box_region, image, bbox)
        else:
            text_mask_region = self.extract_text_region_improved(box_region)
        
        # 确保文本掩膜和框掩膜尺寸一致
        if text_mask_region.shape != box_mask_region.shape:
            text_mask_region = cv2.resize(text_mask_region, 
                                         (box_mask_region.shape[1], box_mask_region.shape[0]),
                                         interpolation=cv2.INTER_NEAREST)
        
        # 确保文本掩膜只在框内
        text_mask_region = cv2.bitwise_and(text_mask_region, box_mask_region)
        
        # 关键改进：从文本掩膜中移除遮挡物区域
        # 避免将绿色、棕色等遮挡物误判为文本
        occlusion_objects_mask = self.detect_occlusion_objects(box_region, box_mask_region)
        text_mask_region = cv2.bitwise_and(text_mask_region, cv2.bitwise_not(occlusion_objects_mask))
        
        # 计算面积
        bbox_area = int(np.sum(box_mask_region > 0))  # 框的总面积
        text_area = int(np.sum(text_mask_region > 0))  # 文本区域面积（已过滤遮挡物）
        
        # 遮挡面积 = 框总面积 - 文本区域面积
        occluded_area = bbox_area - text_area
        
        # 确保遮挡面积不为负
        occluded_area = max(0, occluded_area)
        
        # 创建遮挡掩膜：框内非文本区域
        occlusion_mask_region = cv2.bitwise_and(
            box_mask_region, 
            cv2.bitwise_not(text_mask_region)
        )
        
        # 创建完整图像大小的掩膜
        full_occlusion_mask = np.zeros(image.shape[:2], dtype=np.uint8)
        full_occlusion_mask[y_min:y_max, x_min:x_max] = occlusion_mask_region
        
        full_text_mask = np.zeros(image.shape[:2], dtype=np.uint8)
        full_text_mask[y_min:y_max, x_min:x_max] = text_mask_region
        
        return full_occlusion_mask, full_text_mask, bbox_area, text_area, occluded_area
    
    def measure_image_occlusion(self, 
                                image_path: str,
                                detection_file: str) -> ImageOcclusionResult:
        """
        测量整张图片的遮挡程度
        
        参数:
            image_path: 图像路径
            detection_file: 检测结果文件路径
            
        返回:
            ImageOcclusionResult: 遮挡测量结果
        """
        # 加载图像
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"无法加载图像: {image_path}")
        
        # 从文件路径提取图片文件名
        image_filename = Path(image_path).name
        
        # 加载检测结果
        bboxes = self.load_detection_results(detection_file, image_filename)
        
        if not bboxes:
            print(f"警告：未找到图片 {image_filename} 的检测结果")
            return ImageOcclusionResult(
                image_path=image_path,
                total_boxes=0,
                total_bbox_area=0,
                total_occluded_area=0,
                global_occlusion_ratio=0.0,
                box_results=[],
                average_box_occlusion_ratio=0.0,
                severity_level="无检测结果"
            )
        
        # 分析每个框
        box_results = []
        total_bbox_area = 0
        total_occluded_area = 0
        
        for bbox in bboxes:
            # 检测框内遮挡
            occlusion_mask, text_mask, bbox_area, text_area, occluded_area = self.detect_occlusion_in_box(image, bbox)
            
            # 计算遮挡比例
            if bbox_area > 0:
                occlusion_ratio = occluded_area / bbox_area
            else:
                occlusion_ratio = 0.0
            
            # 如果结果反了，翻转遮挡比例（默认不翻转）
            if self.invert_ratio:
                occlusion_ratio = 1.0 - occlusion_ratio
            
            # 保存结果
            box_result = BoxOcclusionResult(
                transcription=bbox.transcription,
                bbox_coords=bbox.bbox,
                bbox_area=bbox_area,
                text_area=text_area,
                occluded_area=occluded_area,
                occlusion_ratio=occlusion_ratio,
                occlusion_mask=occlusion_mask,
                text_mask=text_mask
            )
            box_results.append(box_result)
            
            total_bbox_area += bbox_area
            total_occluded_area += occluded_area
        
        # 计算全局指标
        if total_bbox_area > 0:
            global_occlusion_ratio = total_occluded_area / total_bbox_area
        else:
            global_occlusion_ratio = 0.0
        
        # 如果结果反了，翻转全局遮挡比例（默认不翻转）
        if self.invert_ratio:
            global_occlusion_ratio = 1.0 - global_occlusion_ratio
        global_occlusion_ratio = _clip01(float(global_occlusion_ratio))
        
        # 计算平均遮挡比例
        if box_results:
            average_box_occlusion_ratio = np.mean([r.occlusion_ratio for r in box_results])
        else:
            average_box_occlusion_ratio = 0.0
        
        # 判断严重程度
        if global_occlusion_ratio >= 0.5:
            severity_level = "严重遮挡"
        elif global_occlusion_ratio >= 0.3:
            severity_level = "中度遮挡"
        elif global_occlusion_ratio >= 0.1:
            severity_level = "轻微遮挡"
        else:
            severity_level = "几乎无遮挡"
        
        return ImageOcclusionResult(
            image_path=image_path,
            total_boxes=len(bboxes),
            total_bbox_area=total_bbox_area,
            total_occluded_area=total_occluded_area,
            global_occlusion_ratio=global_occlusion_ratio,
            box_results=box_results,
            average_box_occlusion_ratio=average_box_occlusion_ratio,
            severity_level=severity_level
        )


def main():
    """主函数"""
    print("=" * 60)
    print("文本遮挡程度测量系统（原始版本 - 不翻转结果）")
    print("计算每个检测框内被遮挡面积与框总面积的比例")
    print("=" * 60)
    
    # 设置文件路径
    IMAGE_PATH = "7.jpg"
    DETECTION_FILE = "7_result.txt"
    
    # 检查文件是否存在
    import os
    if not os.path.exists(IMAGE_PATH):
        print(f"错误：图片文件不存在 - {IMAGE_PATH}")
        return
    
    if not os.path.exists(DETECTION_FILE):
        print(f"错误：检测文件不存在 - {DETECTION_FILE}")
        return
    
    print(f"正在分析: {IMAGE_PATH}")
    print(f"使用检测文件: {DETECTION_FILE}")
    print("=" * 60)
    
    # 初始化测量器
    # use_model=True 使用深度学习模型（需要PyTorch）
    # use_model=False 使用改进的传统方法
    # invert_ratio=False 不翻转遮挡比例（使用原始结果）
    measurer = OcclusionMeasurer(
        min_occlusion_size=10,
        use_adaptive_detection=True,
        use_model=True,  # 尝试使用模型，如果不可用会自动回退到传统方法
        model_device='auto',  # 'auto', 'cuda', 'cpu'
        invert_ratio=False  # 不翻转遮挡比例（使用原始结果）
    )
    
    try:
        # 进行测量
        result = measurer.measure_image_occlusion(
            image_path=IMAGE_PATH,
            detection_file=DETECTION_FILE
        )
        
        # 显示结果
        print("\n" + "=" * 60)
        print("测量结果（原始结果，未翻转）:")
        print("=" * 60)
        print(f"总检测框数: {result.total_boxes}")
        print(f"所有框的总面积: {result.total_bbox_area} 像素")
        print(f"所有框中被遮挡的总面积: {result.total_occluded_area} 像素")
        print(f"全局遮挡比例: {result.global_occlusion_ratio:.4f} ({result.global_occlusion_ratio*100:.2f}%)")
        print(f"平均框遮挡比例: {result.average_box_occlusion_ratio:.4f} ({result.average_box_occlusion_ratio*100:.2f}%)")
        print(f"严重程度: {result.severity_level}")
        
        print("\n" + "-" * 60)
        print("各框详细结果:")
        print("-" * 60)
        print(f"{'文字':<6} {'遮挡比例':<12} {'文本面积':<12} {'遮挡面积':<12} {'框总面积':<12} {'状态':<8}")
        print("-" * 80)
        
        for r in result.box_results:
            if r.occlusion_ratio >= 0.5:
                status = "严重"
            elif r.occlusion_ratio >= 0.3:
                status = "中度"
            elif r.occlusion_ratio >= 0.1:
                status = "轻微"
            else:
                status = "无遮挡"
            
            print(f"{r.transcription:<6} {r.occlusion_ratio:<12.4f} "
                  f"{r.text_area:<12} {r.occluded_area:<12} {r.bbox_area:<12} {status:<8}")
        
        image_filename = os.path.basename(IMAGE_PATH)
        base_name = os.path.splitext(image_filename)[0]
        
        # 保存详细结果到JSON
        output_data = {
            "image_path": IMAGE_PATH,
            "detection_file": DETECTION_FILE,
            "note": "原始结果，未翻转",
            "global_metrics": {
                "total_boxes": result.total_boxes,
                "total_bbox_area": result.total_bbox_area,
                "total_occluded_area": result.total_occluded_area,
                "global_occlusion_ratio": float(result.global_occlusion_ratio),
                "average_box_occlusion_ratio": float(result.average_box_occlusion_ratio),
                "severity_level": result.severity_level
            },
            "box_details": [
                {
                    "transcription": r.transcription,
                    "bbox_coords": list(r.bbox_coords),
                    "bbox_area": r.bbox_area,
                    "text_area": r.text_area,
                    "occluded_area": r.occluded_area,
                    "occlusion_ratio": float(r.occlusion_ratio)
                }
                for r in result.box_results
            ]
        }
        
        output_json = f"{base_name}_occlusion_measurement_first.json"
        with open(output_json, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        
        print(f"✓ 详细结果已保存: {output_json}")
        print("\n" + "=" * 60)
        
    except Exception as e:
        print(f"分析过程中出现错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()


def process_batch(image_folder: str, json_folder: str, output_json_path: str = "occlusion_scores.json",
                  use_model: bool = True, model_device: str = "auto", invert_ratio: bool = False):
    """
    按“图片文件夹 + JSON文件夹”批量计算遮挡指标。
    JSON格式需包含字段: characters: [{transcription, points}, ...]
    输出: {image_filename: global_occlusion_ratio}
    """
    image_folder_p = Path(image_folder)
    json_folder_p = Path(json_folder)

    if not image_folder_p.exists():
        raise FileNotFoundError(f"图片文件夹不存在: {image_folder}")
    if not json_folder_p.exists():
        raise FileNotFoundError(f"JSON文件夹不存在: {json_folder}")

    measurer = OcclusionMeasurer(
        min_occlusion_size=10,
        use_adaptive_detection=True,
        use_model=use_model,
        model_device=model_device,
        invert_ratio=invert_ratio
    )

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
            result = measurer.measure_image_occlusion(
                image_path=str(img_path),
                detection_file=str(json_path)
            )
            results[img_path.name] = _clip01(float(result.global_occlusion_ratio))
            print(f"✓ {img_path.name} -> 遮挡比例: {result.global_occlusion_ratio:.6f}")
        except Exception as e:
            print(f"✗ 处理失败 {img_path.name}: {e}")
            results[img_path.name] = None

    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n汇总已保存: {output_json_path}")
    return results
