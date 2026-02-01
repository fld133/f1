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



TOP_N_PER_TYPE = 3  # 每个子类选两个关键词

# 视频抽帧
def sample_video_frames(video_path, num_frames=200):
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
def get_clip_keywords_from_video(video_path, matched_types, num_frames=200):
    frames = sample_video_frames(video_path, num_frames=num_frames)
    video_emb = video_clip_embedding(frames)

    all_keywords = []

    ATTRIBUTE_TYPES = {"color", "count", "size", "age","location"}
    COMPOSABLE_OBJECTS = ["man", "woman", "child", "animal", "person","elephant","people"]

    # --- 先处理 object（所有情况都取） ---
    obj_candidates = []
    if "object" in matched_types:
        obj_candidates = clip_keywords_for_type(
            video_emb,
            TYPE_KEYWORDS_MAP["object"],
            top_k=2  # 取最相关的一个 object
        )
        all_keywords.extend(obj_candidates)

    # --- 属性类问题：object + attribute 二次组合 ---
    for qtype in matched_types:
        if qtype in ATTRIBUTE_TYPES:
            composed_keywords = []
            if obj_candidates:
                for obj_kw, _ in obj_candidates:
                    if obj_kw in COMPOSABLE_OBJECTS:
                        attr_candidates = clip_keywords_for_type(
                            video_emb,
                            [f"{kw} {obj_kw}" for kw in TYPE_KEYWORDS_MAP[qtype]],
                            top_k=1
                        )
                        composed_keywords.extend(attr_candidates)
            else:
                # 没有 object 时直接 top 1
                attr_candidates = clip_keywords_for_type(
                    video_emb,
                    TYPE_KEYWORDS_MAP[qtype],
                    top_k=1
                )
                composed_keywords.extend(attr_candidates)
            all_keywords.extend(composed_keywords)

    # --- 非属性类问题（overall, action, object 已处理 object）
    for qtype in matched_types:
        if qtype not in ATTRIBUTE_TYPES and qtype != "object":
            type_keywords = TYPE_KEYWORDS_MAP.get(qtype, [])
            selected = clip_keywords_for_type(
                video_emb,
                type_keywords,
                top_k=TOP_N_PER_TYPE
            )
            all_keywords.extend(selected)

    # 去重，按 keyword 名称去重，保留第一个出现的
    seen = set()
    unique_keywords = []
    for kw, sim in all_keywords:
        if kw not in seen:
            seen.add(kw)
            unique_keywords.append((kw, sim))

    return unique_keywords

