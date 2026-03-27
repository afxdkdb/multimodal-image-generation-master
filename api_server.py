import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['CUDA_VISIBLE_DEVICES'] = ''

import torch
import base64
import io
import numpy as np
from PIL import Image
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import threading

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

HF_TOKEN = os.getenv('HF_TOKEN', '')
DEEPSEEK_API_KEY = os.getenv('DEEPSEEK_API_KEY', '')
LOCAL_FILES_ONLY = os.getenv('LOCAL_FILES_ONLY', 'false').lower() == 'true'

if not HF_TOKEN:
    raise ValueError("HF_TOKEN 环境变量未设置！请在 .env 文件中配置或设置环境变量")
if not DEEPSEEK_API_KEY:
    raise ValueError("DEEPSEEK_API_KEY 环境变量未设置！请在 .env 文件中配置或设置环境变量")

print(f"[API] HF_TOKEN={'OK' if HF_TOKEN else 'MISSING'}, DEEPSEEK={'OK' if DEEPSEEK_API_KEY else 'MISSING'}")
print(f"[CONFIG] LOCAL_FILES_ONLY: {LOCAL_FILES_ONLY}")

app = FastAPI(title="Multimodal API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

image_generator = None
captioner = None
translator = None
clip_alignment = None
controlnet = None

class GenerateRequest(BaseModel):
    prompt: str
    enhanced_prompt: str | None = None
    num_inference_steps: int = 50
    guidance_scale: float = 10.0
    controlnet_image: str | None = None

class ControlNetGenerateRequest(BaseModel):
    prompt: str
    control_image: str  # base64 encoded image
    num_inference_steps: int = 50
    guidance_scale: float = 10.0
    controlnet_conditioning_scale: float = 1.0
    enhanced_prompt: str | None = None

class EnhanceRequest(BaseModel):
    prompt: str

class CLIPEvaluateRequest(BaseModel):
    prompt: str
    image: str | None = None

def enhance_prompt(text: str) -> str:
    import requests
    
    url = "https://api.deepseek.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json"
    }
    
    system_prompt = """你是一个专业的AI绘画提示词生成器。请将用户输入扩展成简洁的中文图像生成提示词。

重要规则（必须遵守）：
1. 输出控制在50-80字左右，不要太长
2. 只输出中文提示词
3. 核心内容放前面：主体 + 风格 + 光线 + 氛围
4. 用逗号或顿号分隔关键词
5. 不要写完整句子
6. 包含：主体描述、艺术风格、光线、氛围、构图"""

    data = {
        "model": "deepseek-chat",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"请将以下文本扩展成简洁的中文AI绘画提示词：{text}"}
        ],
        "temperature": 0.7,
        "max_tokens": 300
    }
    
    response = requests.post(url, headers=headers, json=data)
    result = response.json()
    
    enhanced = result['choices'][0]['message']['content'].strip()
    print(f"Enhanced: '{text}' -> '{enhanced}'")
    return enhanced

def get_translator():
    global translator
    if translator is None:
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        print("Loading Translator (zh->en)...")
        tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-zh-en", token=HF_TOKEN, local_files_only=LOCAL_FILES_ONLY)
        model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-zh-en", token=HF_TOKEN, local_files_only=LOCAL_FILES_ONLY)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)
        translator = {"tokenizer": tokenizer, "model": model, "device": device}
        print(f"Translator loaded on {device}!")
    return translator

def translate_to_english(text: str) -> str:
    has_chinese = any('\u4e00' <= char <= '\u9fff' for char in text)
    if has_chinese:
        trans = get_translator()
        inputs = trans["tokenizer"](text, return_tensors="pt", padding=True).to(trans["device"])
        with torch.no_grad():
            outputs = trans["model"].generate(**inputs, max_length=512)
        translated = trans["tokenizer"].decode(outputs[0], skip_special_tokens=True)
        print(f"Translated: '{text}' -> '{translated}'")
        return translated
    return text

def get_generator():
    global image_generator
    if image_generator is None:
        import logging
        logging.getLogger("diffusers").setLevel(logging.WARNING)
        logging.getLogger("torch").setLevel(logging.WARNING)
        
        from diffusers import StableDiffusionPipeline, DDIMScheduler
        import sys
        from io import StringIO
        
        print("Loading Stable Diffusion 2.1...")
        image_generator = StableDiffusionPipeline.from_pretrained(
            "sd2-community/stable-diffusion-2-1-base",
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            safety_checker=None,
            requires_safety_checker=False,
            token=HF_TOKEN,
            local_files_only=LOCAL_FILES_ONLY
        )
        image_generator.scheduler = DDIMScheduler.from_config(image_generator.scheduler.config)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        image_generator = image_generator.to(device)
        print(f"SD loaded on {device}!")
    return image_generator

def get_captioner():
    global captioner
    if captioner is None:
        from transformers import BlipProcessor, BlipForConditionalGeneration
        print("Loading BLIP Large...")
        processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large", token=HF_TOKEN, local_files_only=LOCAL_FILES_ONLY)
        model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large", token=HF_TOKEN, local_files_only=LOCAL_FILES_ONLY)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cuda":
            model.enable_cpu_offload()
        else:
            model = model.to(device)
        captioner = {"processor": processor, "model": model, "device": device}
        print(f"BLIP loaded on {device}!")
    return captioner

def get_clip():
    global clip_alignment
    if clip_alignment is None:
        from transformers import CLIPProcessor, CLIPModel
        print("Loading CLIP for image-text similarity...")
        clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32", token=HF_TOKEN, local_files_only=LOCAL_FILES_ONLY)
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32", token=HF_TOKEN, local_files_only=LOCAL_FILES_ONLY)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cuda":
            clip_model.enable_cpu_offload()
        else:
            clip_model = clip_model.to(device)
        clip_model.eval()
        clip_alignment = {"model": clip_model, "processor": processor, "device": device}
        print(f"CLIP loaded on {device}!")
    return clip_alignment

def compute_image_text_similarity(prompt: str, image: Image.Image):
    """
    正确的 CLIP 语义相似度计算方法：
    1. 用 CLIP 编码提示词 -> text_features
    2. 用 CLIP 编码生成的图像 -> image_features
    3. 计算两者的余弦相似度 -> 这才是真正的语义匹配度
    """
    clip = get_clip()
    
    inputs = clip["processor"](text=[prompt], images=image, return_tensors="pt", padding=True)
    inputs = {k: v.to(clip["device"]) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = clip["model"](**inputs)
        text_features = outputs.text_embeds
        image_features = outputs.image_embeds
        
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    
    similarity = (text_features * image_features).sum(dim=-1).item()
    
    return {
        "similarity_score": float(similarity),
        "interpretation": "相似度越高表示生成的图像与提示词语义匹配度越好"
    }

def get_controlnet():
    global controlnet
    if controlnet is None:
        from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
        print("Loading ControlNet (Canny)...")
        device = "cpu"
        
        controlnet_model = ControlNetModel.from_pretrained(
            "lllyasviel/sd-controlnet-canny",
            torch_dtype=torch.float32,
            token=HF_TOKEN,
            local_files_only=LOCAL_FILES_ONLY
        )
        
        print("Loading SD 1.5 base model...")
        pipe = StableDiffusionControlNetPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            controlnet=controlnet_model,
            torch_dtype=torch.float32,
            safety_checker=None,
            requires_safety_checker=False,
            token=HF_TOKEN,
            local_files_only=LOCAL_FILES_ONLY
        )
        
        pipe = pipe.to(device)
        
        controlnet = {
            "pipeline": pipe,
            "device": device
        }
        print(f"ControlNet loaded on {device}!")
    return controlnet

@app.get("/")
def root():
    return {"message": "Multimodal API is running", "status": "ok"}

@app.get("/status")
def status():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return {
        "device": device,
        "cuda_available": torch.cuda.is_available(),
        "models_loaded": {
            "sd": image_generator is not None,
            "blip": captioner is not None,
            "clip": clip_alignment is not None
        }
    }

@app.post("/enhance")
def enhance_prompt_api(req: EnhanceRequest):
    try:
        original_prompt = req.prompt
        enhanced = enhance_prompt(original_prompt)
        return {
            "original_prompt": original_prompt,
            "enhanced_prompt": enhanced
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/process-canny")
async def process_canny_image(file: UploadFile = File(...)):
    try:
        import cv2
        import numpy as np
        
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        
        # 转换为 numpy 数组
        img_array = np.array(image)
        
        # 转为灰度图
        if len(img_array.shape) == 3:
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_array
        
        # Canny 边缘检测
        low_threshold = 100
        high_threshold = 200
        edges = cv2.Canny(gray, low_threshold, high_threshold)
        
        # 转为3通道
        edges = edges[:, :, None]
        edges = np.concatenate([edges, edges, edges], axis=2)
        
        # 转回 PIL Image
        canny_image = Image.fromarray(edges)
        
        # 返回 base64
        img_buffer = io.BytesIO()
        canny_image.save(img_buffer, format="PNG")
        img_str = base64.b64encode(img_buffer.getvalue()).decode()
        
        return {
            "canny_image": f"data:image/png;base64,{img_str}",
            "original_size": {"width": image.width, "height": image.height}
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/clip-evaluate")
def clip_evaluate_api(req: CLIPEvaluateRequest):
    try:
        translated = translate_to_english(req.prompt)
        
        if req.image:
            image_data = base64.b64decode(req.image)
            image = Image.open(io.BytesIO(image_data)).convert("RGB")
            result = compute_image_text_similarity(translated, image)
        else:
            result = {"error": "请提供图片以计算语义相似度"}
        
        return result
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate")
def generate_image(req: GenerateRequest):
    try:

        if req.enhanced_prompt:
            translated_prompt = translate_to_english(req.enhanced_prompt)
            final_prompt = translated_prompt
        else:
            final_prompt = translate_to_english(req.prompt)
            translated_prompt = final_prompt

        num_steps = req.num_inference_steps
        guidance = req.guidance_scale
        
        pipe = get_generator()
        
        import os
        os.environ['TOKENIZERS_PARALLELISM'] = 'false'
        
        with torch.inference_mode():
            image = pipe(
                final_prompt,
                num_inference_steps=num_steps,
                guidance_scale=guidance,
                height=768,
                width=768,
                enable_vae_tiling=True
            ).images[0]
        
        img_buffer = io.BytesIO()
        image.save(img_buffer, format="PNG")
        img_str = base64.b64encode(img_buffer.getvalue()).decode()
        
        cap = get_captioner()
        inputs = cap["processor"](image, return_tensors="pt").to(cap["model"].device)
        with torch.no_grad():
            output = cap["model"].generate(**inputs, max_length=100)
        caption = cap["processor"].decode(output[0], skip_special_tokens=True)
        
        clip_evaluation = compute_image_text_similarity(final_prompt, image)
        print(f"CLIP Image-Text Similarity: {clip_evaluation}")
        
        return {
            "image": f"data:image/png;base64,{img_str}",
            "original_prompt": req.prompt,
            "translated_prompt": translated_prompt,
            "caption": caption,
            "clip_evaluation": clip_evaluation,
            "used_num_steps": num_steps,
            "used_guidance": guidance
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/caption")
async def caption_image(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        
        cap = get_captioner()
        inputs = cap["processor"](image, return_tensors="pt").to(cap["model"].device)
        with torch.no_grad():
            output = cap["model"].generate(**inputs, max_length=100)
        caption = cap["processor"].decode(output[0], skip_special_tokens=True)
        
        return {"caption": caption}
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate-controlnet")
async def generate_with_controlnet(req: ControlNetGenerateRequest):
    try:
        import os
        os.environ['TOKENIZERS_PARALLELISM'] = 'false'
        
        controlnet_data = get_controlnet()
        pipe = controlnet_data["pipeline"]
        
        image_data = base64.b64decode(req.control_image)
        control_image = Image.open(io.BytesIO(image_data)).convert("RGB")
        control_image = control_image.resize((768, 768))
        
        if req.enhanced_prompt:
            translated = translate_to_english(req.enhanced_prompt)
        else:
            translated = translate_to_english(req.prompt)
        
        with torch.inference_mode():
            image = pipe(
                translated,
                image=control_image,
                num_inference_steps=req.num_inference_steps,
                guidance_scale=req.guidance_scale,
                controlnet_conditioning_scale=req.controlnet_conditioning_scale,
                height=768,
                width=768,
            ).images[0]
        
        img_buffer = io.BytesIO()
        image.save(img_buffer, format="PNG")
        img_str = base64.b64encode(img_buffer.getvalue()).decode()
        
        cap = get_captioner()
        inputs = cap["processor"](image, return_tensors="pt").to(cap["model"].device)
        with torch.no_grad():
            output = cap["model"].generate(**inputs, max_length=100)
        caption = cap["processor"].decode(output[0], skip_special_tokens=True)
        
        clip_evaluation = compute_image_text_similarity(translated, image)
        
        return {
            "image": f"data:image/png;base64,{img_str}",
            "original_prompt": req.prompt,
            "translated_prompt": translated,
            "caption": caption,
            "clip_evaluation": clip_evaluation,
            "used_num_steps": req.num_inference_steps,
            "used_guidance": req.guidance_scale
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    print("Starting FastAPI server on http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)
