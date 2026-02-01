# confidence_router.py
import re
from typing import List

QUESTION_KEYWORDS = {
    "overall": {
        "video", "scene", "happening", "happen", "about", "content",
        "overall", "summary", "situation", "context", "going on","describe the scene",
    },

    "action": {
        "do", "doing", "does", "did", "happening", "happen",
        "action", "activity", "perform", "work", "move",
        "running", "walking", "playing", "cooking", "working",
    },

    "object": {
        "what", "object", "thing", "item", "person", "people",
        "man", "woman", "child", "children", "boy", "girl",
        "someone", "something","device", "tool", "food", "car", "phone",
        "animal",
    },

    "color":{
        "color", "colour", "what color", "which color",
    },
    "count": {
        "how many", "number of", "count", "quantity",
    },
    "size":{
        "how big", "how large", "how small", "size",
    },
    "age":{
        "how old", "age", "young or old","young","middle-aged","look like","looks like","look"
    },
    "location":{
        "where", "location", "place", "position",
    },
}

CONF_THRESHOLD = 0.4

# -----------------------------
# Tokenization
# -----------------------------
def tokenize(question: str):
    question = question.lower()
    tokens = re.findall(r"\b[a-z]+\b", question)
    return set(tokens)

# -----------------------------
# Phrase matching (for multi-word keywords)
# -----------------------------
def phrase_match(question: str, keywords: set):
    question_lc = question.lower()
    matched = set()
    for kw in keywords:
        if " " in kw:  # multi-word phrase
            if kw in question_lc:
                matched.add(kw)
    return matched

# -----------------------------
# Compute multi-class confidence
# -----------------------------
def compute_confidence(question: str):
    tokens = tokenize(question)
    scores = {}
    for qtype, keywords in QUESTION_KEYWORDS.items():
        # 单词匹配
        matched_tokens = tokens & set(kw for kw in keywords if " " not in kw)
        # 短语匹配
        matched_phrases = phrase_match(question, set(kw for kw in keywords if " " in kw))
        total_matched = len(matched_tokens) + len(matched_phrases)
        scores[qtype] = total_matched / max(len(tokens), 1)  # 避免除0
    best_type = max(scores, key=scores.get)
    best_conf = scores[best_type]
    overall_conf = sum(scores.values()) / len(scores)
    return scores, best_type, best_conf, overall_conf

# -----------------------------
# Route question
# -----------------------------
def route_question(question: str):
    scores, best_type, best_conf, overall_conf = compute_confidence(question)

    # Step 0: 获取匹配的子类
    matched_types = [t for t, s in scores.items() if s > 0]
    ATTRIBUTE_TYPES = {"count", "color", "size", "age","location"}

    # 整体问题自动加动作+对象
    # -----------------------------
    if "overall" in matched_types:
        if "action" not in matched_types:
            matched_types.append("action")
        if "object" not in matched_types:
            matched_types.append("object")

    #属性问题
    if any(t in ATTRIBUTE_TYPES for t in matched_types):
        if "object" not in matched_types:
            matched_types.append("object")

    # Step 1: 决策逻辑
    use_rule = False
    if 0 < len(matched_types) <= 3:
        use_rule = True
        selected_types = matched_types
    elif len(matched_types) > 3:
        use_rule = True
        # 取置信度最高的前3类
        sorted_types = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        selected_types = [t for t, s in sorted_types[:3]]
    else:
        # 没有匹配时，交给 Qwen 兜底
        use_rule = False
        selected_types = []

    return {
        "use_rule": use_rule,
        "matched_types": selected_types,
        "overall_confidence": overall_conf,
        "scores": scores
    }

# -----------------------------
# CLI 测试
# -----------------------------
if __name__ == "__main__":
    q = input("Enter your question: ")
    info = route_question(q)
    print(info)
