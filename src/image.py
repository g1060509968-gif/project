import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def extract_precise_data(image_path, output_csv='spectrum_data.csv'):
    # 1. 读取图片
    img = cv2.imread(image_path)
    if img is None:
        print("Error: Image not found.")
        return

    # --- 精确参数设置 (基于图像分析) ---
    # 绘图框裁剪范围 (y_start:y_end, x_start:x_end)
    roi_y_start, roi_y_end = 69, 852
    roi_x_start, roi_x_end = 217, 1241
    
    # 物理坐标校准点 (绝对像素坐标 -> 物理数值)
    # 点1: (像素X, 像素Y) -> (波长nm, 强度)
    p1_pixel = (279, 848)  # 对应 1062 nm, 0.0
    p1_phys  = (1062.0, 0.0)
    
    # 点2: (像素X, 像素Y) -> (波长nm, 强度)
    p2_pixel = (1162, 141) # 对应 1068 nm, 1.0
    p2_phys  = (1068.0, 1.0)
    # --------------------------------

    # 2. 裁剪 ROI
    roi = img[roi_y_start:roi_y_end, roi_x_start:roi_x_end]
    
    # 3. 颜色分离 (提取红色)
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    # 红色在 HSV 中有两个区间: 0-10 和 170-180
    mask1 = cv2.inRange(hsv, np.array([0, 100, 100]), np.array([10, 255, 255]))
    mask2 = cv2.inRange(hsv, np.array([170, 100, 100]), np.array([180, 255, 255]))
    mask = mask1 | mask2
    
    # 形态学去噪 (去除孤立噪点)
    kernel = np.ones((2,2), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # 4. 提取数据点
    data_points = []
    height, width = mask.shape
    
    for x_rel in range(width):
        # 寻找当前列所有的白色像素
        y_indices = np.where(mask[:, x_rel] > 0)[0]
        
        if len(y_indices) > 0:
            # 计算平均 Y 值 (亚像素精度)
            y_avg_rel = np.mean(y_indices)
            
            # 转换回整图的绝对坐标
            x_abs = x_rel + roi_x_start
            y_abs = y_avg_rel + roi_y_start
            
            data_points.append([x_abs, y_abs])
            
    df = pd.DataFrame(data_points, columns=['px_x', 'px_y'])
    
    # 5. 坐标变换 (线性插值)
    # 计算缩放比例 (slope) 和 截距 (intercept)
    # Formula: value = slope * pixel + intercept
    
    # X轴映射
    slope_x = (p2_phys[0] - p1_phys[0]) / (p2_pixel[0] - p1_pixel[0])
    intercept_x = p1_phys[0] - slope_x * p1_pixel[0]
    
    # Y轴映射
    slope_y = (p2_phys[1] - p1_phys[1]) / (p2_pixel[1] - p1_pixel[1])
    intercept_y = p1_phys[1] - slope_y * p1_pixel[1]
    
    # 应用转换
    df['Wavelength (nm)'] = df['px_x'] * slope_x + intercept_x
    df['Normalized Intensity'] = df['px_y'] * slope_y + intercept_y
    
    # 6. 保存和展示
    final_data = df[['Wavelength (nm)', 'Normalized Intensity']]
    final_data.to_csv(output_csv, index=False)
    print(f"提取完成! 数据已保存至 {output_csv}")
    
    # 简单验证绘图
    plt.figure(figsize=(10, 6))
    plt.plot(final_data['Wavelength (nm)'], final_data['Normalized Intensity'], 'r-', linewidth=1)
    plt.title("Extracted Spectrum Data")
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Normalized Intensity")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig('extracted_plot_preview.png')
    plt.show()

# 运行提取
extract_precise_data('image.png')