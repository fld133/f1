# =========================
# video_qa_qwen.py
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from clip_keywords import get_clip_keywords_from_video
from confidence_router import route_question
from qwen_classifier import qwen_classify

QWEN_MODEL_DIR = r"D:\pythondata\biye1\models\qwen2.5\models--Qwen--Qwen2.5-1.5B-Instruct\snapshots\989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ====== Load Qwen ======
print("Loading Qwen tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(QWEN_MODEL_DIR, trust_remote_code=True)
print("Loading Qwen model...")
model = AutoModelForCausalLM.from_pretrained(
    QWEN_MODEL_DIR,
    torch_dtype=torch.float16 if DEVICE=="cuda" else torch.float32,
    device_map="auto",
    trust_remote_code=True
)
model.eval()
model = model.float()
print("Qwen model loaded successfully! Device:", next(model.parameters()).device)

# ====== Step 0: Determine Question Type ======
def determine_question_type(user_question: str):
    route_info = route_question(user_question)
    print("[DEBUG] Confidence routing info:")
    print(route_info)

    if route_info["use_rule"]:
        matched_types = route_info["matched_types"]
        source = "rule"
    else:
        qwen_result = qwen_classify(user_question, tokenizer, model)
        matched_types = [qwen_result["question_type"]]
        source = qwen_result["source"]

    print(f"[Router] Final matched types: {matched_types} (source={source})")
    return matched_types

# ====== Step 1: Build Prompt ======
# ====== Step 1: Build Prompt (Strictly based on keywords) ======
def build_prompt(clip_keywords_with_sim, user_question, question_types):
    """
    clip_keywords_with_sim: list of tuples [(keyword, similarity), ...]
    """
    # 按相似度降序排列
    clip_keywords_sorted = sorted(clip_keywords_with_sim, key=lambda x: x[1], reverse=True)
    keywords_text = ", ".join([f"{kw} ({sim:.2f})" for kw, sim in clip_keywords_sorted])

    prompt = f"""
You are a video question answering assistant.

The following visual keywords were extracted from the video (keyword: similarity):
{keywords_text}

Prioritize keywords with higher similarity when generating your answer.
Answer the question strictly based on these visual cues.
Do NOT introduce any entities, actions, or objects that are not listed in the keywords.
Do NOT mention the keywords explicitly in your answer.
Do NOT explain or justify your answer.
If an article is needed, ONLY use "the"，do not use "a" or "an".
Do NOT start your answer with phrases like 'Based on the visual cues provided'.
Answer in one concise sentence.

If the keywords include "group" and "many", interpret it as a small number of items, preferably between 1 and 10, unless another number keyword is present. 
Use "the" instead of "a" or "an" for objects.

Question types: {', '.join(question_types)}

Question: {user_question}

Give a concise factual answer.
If unsure, give the most likely answer based on the visual evidence.
"""
    return prompt.strip()


# ====== Step 2: Video QA ======
def video_qa(video_path, user_question):
    print("[Step 1] Determining question type...")
    question_types = determine_question_type(user_question)

    print(f"[Step 2] Extracting CLIP keywords for question types {question_types}...")
    clip_keywords = get_clip_keywords_from_video(video_path, question_types)
    print("[DEBUG] CLIP 自动关键词 (keyword, similarity):")
    for k, s in clip_keywords:
        print(f"- {k}: {s:.3f}")

    print("[Step 3] Building prompt...")
    prompt = build_prompt(clip_keywords, user_question, question_types)
    print("[DEBUG] Prompt sent to Qwen:\n", prompt)

    messages = [{"role": "system", "content": "You are Qwen, a helpful multimodal assistant."},
                {"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs, max_new_tokens=100, do_sample=True, temperature=0.7,
            top_p=0.9, top_k=50, eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id
        )
    output_ids = outputs[0][inputs["input_ids"].shape[1]:]
    answer = tokenizer.decode(output_ids, skip_special_tokens=True)
    return answer.strip()

# ====== CLI ======
if __name__ == "__main__":
    video_path = input("请输入视频路径: ").strip()
    question = input("请输入你的问题: ").strip()
    print("\n=== 系统回答 ===")
    answer = video_qa(video_path, question)
    print("\n" + answer)
