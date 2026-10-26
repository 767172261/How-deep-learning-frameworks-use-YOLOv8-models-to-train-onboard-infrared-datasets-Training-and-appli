深度学习框架 如何使用YOLOv8模型训练车载红外数据集FLIR ADAS 数据集的训练及应用   道路车辆红外车辆行人检测数据集的训练及应用
@[toc]

车载红外数据集FLIR ADAS v1和FLIR ADAS v2 ，其中版本训练集8862张，版本训练集10742张，均已转为yolo格式，可直接用于训练。![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/8e7c04d6f4064e58bb11646144f89419.png)
**数据集描述：**

车载红外数据集FLIR ADAS v1和FLIR ADAS v2 ，其中版本训练集8862张，版本训练集10742张，均已转为yolo格式，可直接用于训练。

**版本1类别有3类:person bicycle car**

**版本2类别有16类:0: person 1: bike 2: car 3: motor 4: bus 5: train 6: truck 7: light 8: hydrant 9: sign 10: dog 11: deer 12: skateboard 13: stroller 14: scooter 15: other vehicle。**
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/31cf2eb9e2c947418c96c17a9dbfce7c.png)
使用YOLOv8训练FLIR ADAS v1和v2数据集，并进行推理和评估，需要完成以下步骤。将从环境设置开始，一直到模型训练、推理及评估。
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/78163c6126cf441aa9507a58a5e5b075.png)

### 一、环境搭建

#### 1. CUDA驱动安装
确保系统已经安装了NVIDIA显卡驱动以及CUDA Toolkit。可以通过运行`nvidia-smi`命令来检查是否正确安装CUDA。

#### 2. Anaconda安装与Python虚拟环境创建
如果还没有Anaconda，请从[Anaconda官网](https://www.anaconda.com/products/distribution)下载并安装。然后，创建一个新的虚拟环境：

```bash
conda create -n flir_yolov8 python=3.9
conda activate flir_yolov8
```

#### 3. 安装必要的依赖项
在激活的虚拟环境中安装YOLOv8和其他必要的库：

```bash
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu118
pip install ultralytics opencv-python matplotlib numpy
```

### 二、数据集准备

确保数据集已按YOLO格式组织如下：

```
flir_dataset/
├── images/
│   ├── train/
│   └── val/
├── labels/
│   ├── train/
│   └── val/
└── data.yaml
```

对于两个版本的数据集，分别创建相应的`data.yaml`文件：

**版本1 (`data_v1.yaml`):**

```yaml
train: ./flir_dataset/images/train
val: ./flir_dataset/images/val

nc: 3
names: ['person', 'bicycle', 'car']
```

**版本2 (`data_v2.yaml`):**

```yaml
train: ./flir_dataset/images/train
val: ./flir_dataset/images/val

nc: 16
names: ['person', 'bike', 'car', 'motor', 'bus', 'train', 'truck', 'light', 'hydrant', 'sign', 'dog', 'deer', 'skateboard', 'stroller', 'scooter', 'other vehicle']
```

### 三、调用官方预训练模型并训练

#### 1. 下载官方预训练模型

```bash
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s.pt
```

#### 2. 开始训练

对于版本1：

```python
from ultralytics import YOLO

model = YOLO('yolov8s.pt')  # 加载预训练模型

results = model.train(
    data='path/to/data_v1.yaml',  # 数据集配置文件路径
    epochs=100,                    # 训练周期数
    imgsz=640,                     # 输入图像大小
    batch=16,                      # 每批次样本数量
    device=0,                      # 使用第0号GPU
    project='runs/flir_v1',       # 结果保存位置
    name='exp',                    # 实验名称
)
```

对于版本2，只需将`data='path/to/data_v1.yaml'`替换为`data='path/to/data_v2.yaml'`，并根据需要调整其他参数。

### 四、推理代码

```python
import cv2
from ultralytics import YOLO

model = YOLO('runs/flir_v1/exp/weights/best.pt')  # 根据实际情况加载最佳权重

def detect_image(model, img_path):
    results = model(img_path)
    
    for r in results:
        im_array = r.plot()
        im = cv2.cvtColor(im_array, cv2.COLOR_RGB2BGR)
        cv2.imshow('Detection Result', im)
        cv2.waitKey(0)
    cv2.destroyAllWindows()

img_path = 'path/to/test_image.jpg'
detect_image(model, img_path)
```

### 五、评估代码

```python
from ultralytics import YOLO

model = YOLO('runs/flir_v1/exp/weights/best.pt')

metrics = model.val(data='path/to/data_v1.yaml')

print("mAP50:", metrics.box.map50)
print("mAP50-95:", metrics.box.map)
print("Precision:", metrics.box.precision)
print("Recall:", metrics.box.recall)
```

构建一个**目标检测系统**（如基于YOLOv8的车载红外目标检测系统），需要从数据准备、模型训练、推理部署、结果可视化到最终的系统集成等多个环节进行系统化设计。以下是完整的实现流程和代码示例，适用于 **FLIR ADAS v1/v2 数据集**。

---

## 🧱 一、系统架构设计（整体流程）

| 阶段 | 内容 |
|------|------|
| 1️⃣ 数据准备 | 将FLIR数据集转换为YOLO格式，划分训练集、验证集 |
| 2️⃣ 模型训练 | 使用YOLOv8训练目标检测模型 |
| 3️⃣ 模型评估 | 验证模型性能，分析mAP、Precision、Recall等指标 |
| 4️⃣ 模型推理 | 对图像、视频或实时摄像头进行目标检测 |
| 5️⃣ 系统封装 | 可选：构建可视化界面、Web API、边缘设备部署等 |
| 6️⃣ 持续优化 | 调整模型结构、数据增强、类别不平衡问题处理等 |

---

## 📦 二、数据准备（FLIR ADAS v1/v2）

### ✅ 数据集结构要求（YOLO格式）

```
flir_dataset/
├── images/
│   ├── train/
│   └── val/
├── labels/
│   ├── train/
│   └── val/
└── data.yaml
```

### ✅ `data.yaml` 示例（v2）

```yaml
train: ./flir_dataset/images/train
val: ./flir_dataset/images/val

nc: 16
names: ['person', 'bike', 'car', 'motor', 'bus', 'train', 'truck', 'light', 'hydrant', 'sign', 'dog', 'deer', 'skateboard', 'stroller', 'scooter', 'other vehicle']
```

---

## 🧠 三、模型训练（YOLOv8）

### 1. 下载YOLOv8预训练模型

```bash
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s.pt
```

### 2. 训练代码（Python API）

```python
from ultralytics import YOLO

# 加载预训练模型
model = YOLO('yolov8s.pt')

# 开始训练
results = model.train(
    data='flir_dataset/data.yaml',  # 替换为你的data.yaml路径
    epochs=100,
    imgsz=640,
    batch=16,
    device=0,  # 使用GPU 0
    project='runs/flir',  # 保存路径
    name='exp',  # 实验名称
    save=True,
    save_period=10,  # 每10个epoch保存一次
    pretrained=True,
    verbose=True
)
```

### 或使用命令行训练：

```bash
yolo task=detect mode=train model=yolov8s.pt data=flir_dataset/data.yaml epochs=100 imgsz=640 batch=16 device=0
```

---

## 📊 四、模型评估

### 1. 使用Python API评估模型性能

```python
from ultralytics import YOLO

# 加载训练好的模型
model = YOLO('runs/flir/exp/weights/best.pt')

# 执行验证
metrics = model.val(data='flir_dataset/data.yaml')

# 输出关键指标
print("mAP50:", metrics.box.map50)
print("mAP50-95:", metrics.box.map)
print("Precision:", metrics.box.precision)
print("Recall:", metrics.box.recall)
```

### 2. 可视化结果（PR曲线、混淆矩阵等）

YOLOv8 会自动保存评估结果到 `runs/flir/exp/val/` 目录下，包括：

- `confusion_matrix.png`
- `precision_recall_curve.png`
- `results.csv`（包含各类指标）

---

## 🔍 五、推理与检测（图像/视频/摄像头）

### 1. 单张图像推理

```python
from ultralytics import YOLO
import cv2

model = YOLO('runs/flir/exp/weights/best.pt')
img_path = 'flir_dataset/images/val/FLIR_00001.jpg'

results = model(img_path)

for r in results:
    im_array = r.plot()
    im = cv2.cvtColor(im_array, cv2.COLOR_RGB2BGR)
    cv2.imshow('Detection Result', im)
    cv2.waitKey(0)
cv2.destroyAllWindows()
```

### 2. 视频流推理（本地视频或摄像头）

```python
cap = cv2.VideoCapture('video.mp4')  # 或使用 0 表示摄像头

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)
    for r in results:
        im_array = r.plot()
        im = cv2.cvtColor(im_array, cv2.COLOR_RGB2BGR)
        cv2.imshow('FLIR Detection', im)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
```

---

## 🧩 六、系统集成与部署建议

### ✅ 1. 构建可视化界面（可选）

可以使用 `PyQt5` 或 `Tkinter` 创建一个简单的GUI界面，展示检测结果。

### ✅ 2. 构建RESTful API服务（推荐）

使用 `FastAPI` 或 `Flask` 提供HTTP接口：

```bash
pip install fastapi uvicorn
```

```python
from fastapi import FastAPI, File, UploadFile
from ultralytics import YOLO
import cv2
import numpy as np

app = FastAPI()
model = YOLO('runs/flir/exp/weights/best.pt')

@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    results = model(img)
    for r in results:
        im_array = r.plot()
        im = cv2.cvtColor(im_array, cv2.COLOR_RGB2BGR)
        _, buffer = cv2.imencode('.jpg', im)
        return {"result": buffer.tobytes().hex()}
```

启动服务：

```bash
uvicorn app:app --host 0.0.0.0 --port 8000
```

---

### ✅ 3. 部署到边缘设备（如 NVIDIA Jetson）

- 安装Jetson系统（JetPack SDK）
- 安装CUDA、TensorRT、OpenCV等依赖
- 使用TensorRT优化YOLO模型
- 部署推理代码，实现实时红外目标检测

---

## 🛠 七、优化与持续改进

### 1. 数据增强建议

- 对稀有类别（如deer、scooter）进行数据增强（旋转、翻转、亮度扰动）
- 使用 `albumentations` 或 YOLOv8内置增强

```python
model.train(..., augment=True)
```

### 2. 类别不平衡处理

- 在训练时设置类别权重：

```python
model.train(..., class_weights=[1.0, 1.5, 1.0, ...])  # 根据标注数量调整
```


