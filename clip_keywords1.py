#clip_keywords.py
import cv2
import torch
import numpy as np
from PIL import Image
from transformers import CLIPModel, CLIPProcessor
from dictionary import OBJECT_KEYWORDS, ACTION_KEYWORDS, OVERALL_KEYWORDS,COLOR_KEYWORDS, COUNT_KEYWORDS, SIZE_KEYWORDS, AGE_KEYWORDS, LOCATION_KEYWORDS

CLIP_MODEL_DIR = r"D:\pythondata\biye1\models\clip\models--openai--clip-vit-base-patch32\snapshots\3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

processor = CLIPProcessor.from_pretrained(CLIP_MODEL_DIR)
model = CLIPModel.from_pretrained(CLIP_MODEL_DIR).to(DEVICE)
model.eval()
print("clip模型加载成功")

TYPE_KEYWORDS_MAP = {
    "overall": OVERALL_KEYWORDS,
    "action": ACTION_KEYWORDS,
    "object": OBJECT_KEYWORDS,
    "color": COLOR_KEYWORDS,
    "count": COUNT_KEYWORDS,
    "size": SIZE_KEYWORDS,
    "age": AGE_KEYWORDS,
    "location": LOCATION_KEYWORDS,
}

TOP_N_PER_TYPE = 2  # 每个子类选两个关键词

# 视频抽帧
def sample_video_frames(video_path, num_frames=8):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(frame))
    cap.release()
    return frames

# 视频 → CLIP 向量
def video_clip_embedding(frames):
    inputs = processor(images=frames, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        image_features = model.get_image_features(**inputs)
    video_embedding = image_features.mean(dim=0, keepdim=True)
    return video_embedding

# 针对单个类型匹配关键词，返回 [(keyword, similarity)]
def clip_keywords_for_type(video_embedding, keywords, top_k=TOP_N_PER_TYPE):
    if len(keywords) == 0:
        return []

    text_inputs = processor(text=keywords, return_tensors="pt", padding=True).to(DEVICE)
    with torch.no_grad():
        text_features = model.get_text_features(**text_inputs)

    # 归一化
    video_embedding = video_embedding / video_embedding.norm(dim=-1, keepdim=True)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    similarity = (video_embedding @ text_features.T).squeeze(0)

    actual_top_k = min(top_k, similarity.shape[0])
    top_sim, top_indices = similarity.topk(actual_top_k)

    return [(keywords[i], top_sim[j].item()) for j, i in enumerate(top_indices)]

# 总体函数：多类型匹配，返回 [(keyword, similarity)]
def get_clip_keywords_from_video(video_path, matched_types, num_frames=5):
    frames = sample_video_frames(video_path, num_frames=num_frames)
    video_emb = video_clip_embedding(frames)
    all_keywords = []

    for qtype in matched_types:
        type_keywords = TYPE_KEYWORDS_MAP.get(qtype, [])
        if qtype in {"count", "size", "age", "location"}:
            top_size = 1
        else:
            top_size = TOP_N_PER_TYPE
        selected = clip_keywords_for_type(video_emb, type_keywords, top_k=top_size)
        all_keywords.extend(selected)

    return all_keywords
