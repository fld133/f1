import cv2
import torch
from PIL import Image
from transformers import (
    BlipProcessor, BlipForConditionalGeneration,
    CLIPProcessor, CLIPModel,
    AutoTokenizer, AutoModelForCausalLM
)

# ================== 设备 ==================
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

# ================== 路径 ==================
CLIP_MODEL_DIR = r"D:\pythondata\biye1\models\clip\models--openai--clip-vit-base-patch32\snapshots\3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268"
QWEN_MODEL_DIR = r"D:\pythondata\biye1\models\qwen2.5\models--Qwen--Qwen2.5-1.5B-Instruct\snapshots\989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
BLIP_MODEL_DIR = r"D:\pythondata\biye1\models\blip-image-captioning-base"

# ================== 加载模型 ==================
print("Loading models...")

blip_processor = BlipProcessor.from_pretrained(BLIP_MODEL_DIR)
blip_model = BlipForConditionalGeneration.from_pretrained(
    BLIP_MODEL_DIR,
    torch_dtype=torch.float16 if device == "cuda" else torch.float32
).to(device)

clip_processor = CLIPProcessor.from_pretrained(CLIP_MODEL_DIR)
clip_model = CLIPModel.from_pretrained(
    CLIP_MODEL_DIR,
    torch_dtype=torch.float16 if device == "cuda" else torch.float32
).to(device)

tokenizer = AutoTokenizer.from_pretrained(QWEN_MODEL_DIR, trust_remote_code=True)
qwen_model = AutoModelForCausalLM.from_pretrained(
    QWEN_MODEL_DIR,
    torch_dtype=torch.float32,
    device_map="auto",
    trust_remote_code=True
)
qwen_model.eval()

print("Models loaded!")

# ================== 1. CLIP直接匹配帧 ==================
def clip_select_frames(video_path, question, max_frames=8, threshold=0.2):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    best_frames = []
    best_times = []

    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 每0.5秒采一帧
        if idx % max(int(fps/2), 1) == 0:
            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            inputs = clip_processor(images=img, text=question, return_tensors="pt").to(device)
            with torch.no_grad():
                image_feat = clip_model.get_image_features(pixel_values=inputs['pixel_values'])
                text_feat = clip_model.get_text_features(
                    input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask']
                )
                image_feat /= image_feat.norm(dim=-1, keepdim=True)
                text_feat /= text_feat.norm(dim=-1, keepdim=True)
                sim = (image_feat @ text_feat.T).item()

            if sim >= threshold:
                best_frames.append(img)
                best_times.append(round(idx/fps, 2))

        idx += 1

    cap.release()
    # 最多取 max_frames
    return best_frames[:max_frames], best_times[:max_frames]

# ================== 2. BLIP ==================
def generate_captions(frames, times):
    captions = []
    for img, t in zip(frames, times):
        inputs = blip_processor(images=img, return_tensors="pt").to(device)
        with torch.no_grad():
            out = blip_model.generate(**inputs, max_new_tokens=30)
        cap = blip_processor.decode(out[0], skip_special_tokens=True)
        # 添加时间信息，但不在答案中显示 frame
        captions.append(f"[{t}s] {cap}")
    return captions

# ================== 3. Summary ==================
def generate_summary(captions):
    # 简单摘要，只取前三条
    summary = " ".join(captions[:3])
    return summary

# ================== 4. Prompt构建 ==================
def build_prompt(summary, question):
    # 对who问题的规则
    who_keywords = ["who", "人", "人物"]
    is_who = any(k in question.lower() for k in who_keywords)

    prompt = f"""
You are answering a question about a video.

Video summary:
{summary}

Question:
{question}

Rules:
- Answer with ONLY 1-3 words
- Do NOT explain
- Do NOT mention frames
- If unsure, you can guess
"""
    if is_who:
        prompt += "- For 'who' questions, answer as: baby, man, woman, old man, old woman\n"

    prompt += "\nAnswer:\n"
    return prompt

# ================== 5. Qwen问答 ==================
def answer_question(prompt):
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(qwen_model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = qwen_model.generate(
            **inputs,
            max_new_tokens=20,
            do_sample=False
        )

    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    if "Answer:" in text:
        text = text.split("Answer:")[-1]
    text = text.strip().split("\n")[0]
    text = text.split(".")[0]
    return text.strip()

# ================== 主流程 ==================
def video_qa(video_path, question):
    print("Selecting key frames with CLIP...")
    frames, times = clip_select_frames(video_path, question)

    print(f"Selected {len(frames)} frames.")
    print("Generating captions with BLIP...")
    captions = generate_captions(frames, times)

    summary = generate_summary(captions)
    print("Video summary:", summary)

    prompt = build_prompt(summary, question)
    print("Prompt built. Asking Qwen...")

    answer = answer_question(prompt)
    return answer

# ================== 测试 ==================
if __name__ == "__main__":
    video_path = input("请输入视频路径: ").strip()
    question = input("请输入你的问题: ").strip()

    answer = video_qa(video_path, question)
    print("\n===== ANSWER =====\n")
    print(answer)