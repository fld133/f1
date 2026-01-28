# qwen_classifier.py
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# 标签列表，去掉重复
LABELS = ["overall", "action", "object", "color", "count", "size", "age", "location"]

def load_qwen_model(model_dir, device="cuda"):
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        torch_dtype=torch.float16 if device=="cuda" else torch.float32,
        device_map=None
    )
    model.to(device)
    model.eval()
    return tokenizer, model

def qwen_classify(question: str, tokenizer, model, top_k=3):
    """
    使用 Qwen 对问题进行分类，返回 top_k 个最可能的类别
    """
    prompt = f"""
You are a question classifier.

Classify the following question into one or more of these categories:
{', '.join(LABELS)}

Question: {question}

Only output the category names separated by commas. If unsure, list the top 1-3 most likely categories.
"""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=20,
            do_sample=True,   # 用采样可以获取多个可能类别
            temperature=0.7,
            top_p=0.9,
            top_k=50,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id
        )
    result = tokenizer.decode(outputs[0], skip_special_tokens=True).lower()
    # 清理输出，拆分成类别列表
    predicted_types = []
    for label in LABELS:
        if label in result and label not in predicted_types:
            predicted_types.append(label)
        if len(predicted_types) >= top_k:
            break
    if not predicted_types:
        predicted_types = ["overall"]  # fallback
    return {"question_type": predicted_types, "source": "qwen"}
