# 多模态图像生成系统

一个基于AI的图像生成系统，支持文本生成图像、提示词增强优化、CLIP语义相似度评估和BLIP图像描述。

## 功能特性

- 🎨 **双模式图像生成**
  - **SD 2.1 基础生成** - 使用 Stable Diffusion 2.1 将文本描述转化为图像
  - **SD 1.5 + ControlNet** - 使用 ControlNet 基于 Canny 边缘图精确控制生成
- ✨ **提示词增强** - 使用 DeepSeek API 优化提示词
- 📊 **CLIP 语义相似度评估** - 使用 CLIP 模型评估生成图像与提示词的语义匹配度
- 🤖 **BLIP 图像描述** - 自动生成详细的图像描述
- 🌐 **多语言支持** - 支持中文提示词自动翻译为英文
- 🔊 **语音播报** - 使用浏览器原生 Web Speech API 朗读图像描述

## 项目结构

```
├── api_server.py             # FastAPI 后端服务（所有功能集成于此）
├── index.html                # 前端页面
├── frontend/                 # React 前端项目（备用）
├── .env.example              # 环境变量配置示例
├── requirements.txt          # Python 依赖
└── README.md                 # 项目说明文档
```

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置环境变量

复制 `.env.example` 为 `.env` 并填入你的 API Key：

```bash
copy .env.example .env
```

编辑 `.env` 文件：
```env
HF_TOKEN=your-huggingface-token-here
DEEPSEEK_API_KEY=your-deepseek-api-key-here
```

### 3. 启动服务

```bash
python api_server.py
```

然后打开 **http://localhost:8000** 即可使用！

## 技术栈

### 后端

- **FastAPI** - Web 框架
- **PyTorch** - 深度学习框架
- **Diffusers** - Stable Diffusion 模型加载
- **Transformers** - BLIP、CLIP、翻译模型

### 前端

- **HTML/CSS/JavaScript** - 原生前端

### API 服务

- **DeepSeek API** - 提示词增强

## 模型说明

### 1. Stable Diffusion 2.1 (图像生成)
- **函数**: `get_generator()` 
- **模型ID**: `sd2-community/stable-diffusion-2-1-base`
- **用途**: 根据文本提示生成图像
- **特性**: 启用 VAE Tiling 减少边缘接缝

### 2. Stable Diffusion 1.5 + ControlNet (边缘控制生成)
- **函数**: `get_controlnet()`
- **模型ID**: `runwayml/stable-diffusion-v1-5`
- **ControlNet**: `lllyasviel/sd-controlnet-canny`
- **用途**: 基于 Canny 边缘图精确控制图像生成
- **特点**: 更好地保持边缘细节

### 3. CLIP (语义相似度评估)
- **函数**: `get_clip()`
- **模型ID**: `openai/clip-vit-base-patch32`
- **用途**: 在图片生成后评估提示词与生成图像的语义匹配度
- **功能**:
  - 编码提示词获取 text\_features
  - 编码生成图像获取 image\_features
  - 计算两者的余弦相似度作为语义匹配度

### 4. BLIP (图像描述)
- **函数**: `get_captioner()`
- **模型ID**: `Salesforce/blip-image-captioning-large`
- **用途**: 生成详细图像文字描述
- **特点**: 升级到 Large 版本，描述更详细准确

### 5. 翻译模型 (中文→英文)
- **函数**: `get_translator()`
- **模型ID**: `Helsinki-NLP/opus-mt-zh-en`
- **用途**: 将中文提示词翻译为英文

### 6. TTS 语音播报
- **实现方式**: 浏览器原生 Web Speech API (`SpeechSynthesis`)
- **用途**: 朗读生成的图像描述
- **特点**: 无需后端模型，跨浏览器支持

## 安装部署

### 1. 克隆项目

```bash
git clone <your-repo-url>
cd <project-folder>
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 3. 配置 API Key

在 `api_server.py` 中配置以下 API Key：

```python
HF_TOKEN = 'your-huggingface-token'      # HuggingFace Token
DEEPSEEK_API_KEY = 'your-deepseek-key'    # DeepSeek API Key
```

### 4. 首次运行

首次运行时会自动从 HuggingFace 下载模型到本地缓存：

- Stable Diffusion 2.1 Base (\~4GB)
- Stable Diffusion 1.5 + ControlNet (\~4GB)
- BLIP Image Captioning Large (\~1GB)
- Opus MT 中文→英文 翻译模型 (\~300MB)
- CLIP 提示词评估模型 (\~500MB)

### 5. 启动服务

```bash
python api_server.py
```

服务启动后访问: **<http://localhost:8000>**

## 工作流程

```
用户输入提示词
      ↓
  ↓ 扩充提示词 (DeepSeek API)
      ↓
Stable Diffusion 生成图像
      ↓
  ↓ CLIP 计算提示词与图像的语义相似度
      ↓
  ↓ BLIP 生成图像描述
      ↓
语音播报描述
```

## API 接口

### 1. SD 2.1 图像生成

```bash
POST /generate
Content-Type: application/json

{
    "prompt": "一只可爱的猫咪",
    "enhanced_prompt": "扩展后的提示词",
    "num_inference_steps": 50,
    "guidance_scale": 10.0
}
```

返回结果：

```json
{
    "image": "data:image/png;base64,...",
    "original_prompt": "一只可爱的猫咪",
    "translated_prompt": "A lovely cat...",
    "caption": "a cat sitting on...",
    "clip_evaluation": {
        "similarity_score": 0.85,
        "interpretation": "相似度越高表示生成的图像与提示词语义匹配度越好"
    },
    "used_num_steps": 50,
    "used_guidance": 10.0
}
```

### 2. SD 1.5 + ControlNet 图像生成

```bash
POST /generate-controlnet
Content-Type: application/json

{
    "prompt": "一只可爱的猫咪",
    "enhanced_prompt": "扩展后的提示词",
    "control_image": "base64 encoded canny image",
    "num_inference_steps": 50,
    "guidance_scale": 10.0,
    "controlnet_conditioning_scale": 1.0
}
```

### 3. 提示词增强

```bash
POST /enhance
Content-Type: application/json

{
    "prompt": "一只猫"
}
```

### 4. Canny 边缘图处理

```bash
POST /process-canny
Content-Type: multipart/form-data

file: <image file>
```

### 5. CLIP 语义相似度评估（可选）

```bash
POST /clip-evaluate
Content-Type: application/json

{
    "prompt": "A lovely cat...",
    "image": "base64 encoded image"
}
```

## 前端使用

1. 打开 `http://localhost:8000`
2. 选择生成模式：
   - **SD 2.1 基础生成**: 直接文本生成图像
   - **SD 1.5 + ControlNet**: 上传参考图片，生成基于Canny边缘图的图像
3. 在文本框中输入描述（如"一只可爱的猫咪"）
4. 点击"✨ 扩充提示词"获得更好的效果
5. 如果使用 ControlNet 模式，上传参考图片
6. 点击"🚀 生成图像"
7. 系统会显示生成的图像、BLIP描述和CLIP语义相似度评分
8. 可点击"🔊 语音播报"朗读图像描述

## CLIP 语义相似度说明

| 相似度分数   | 评价              |
| ------- | --------------- |
| > 80%   | 优秀 - 图像与提示词高度匹配 |
| 60%-80% | 良好 - 图像基本符合提示词  |
| 40%-60% | 一般 - 图像部分符合提示词  |
| < 40%   | 较差 - 图像与提示词差异较大 |

## 常见问题

### Q: 生成图像有边缘接缝怎么办？

A: 系统已启用 VAE Tiling，可以有效减少边缘接缝。如仍有明显接缝，可尝试增加推理步数。

### Q: 图像不符合预期怎么办？

A:

1. 使用"扩充提示词"功能生成更详细的提示词
2. 如果使用 ControlNet 模式，调整参考图片的边缘检测参数
3. 增加推理步数和引导系数

### Q: 模型加载失败怎么办？

A:

1. 检查网络连接
2. 确认 HuggingFace Token 正确
3. 确保有足够的磁盘空间

### Q: 生成速度慢怎么办？

A:

1. 使用 GPU 加速（需要 CUDA）
2. 减少推理步数

### Q: BLIP 描述太简单怎么办？

A: 系统已升级到 BLIP Large 版本，默认生成更详细的描述。

## 环境要求

- Python 3.8+
- PyTorch 2.0+
- 8GB+ RAM
- 15GB+ 磁盘空间
- 推荐: NVIDIA GPU with 8GB+ VRAM

## License

MIT License
