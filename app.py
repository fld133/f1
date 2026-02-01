import os
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import sqlite3



# 已有的函数
from video_qa_qwen import video_qa

app = Flask(__name__)
CORS(app)
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)
DB_PATH = "history.db"

# 数据库工具函数
def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def insert_history(video_path, question, answer):
    conn = get_db()
    cursor = conn.cursor()

    cursor.execute(
        """
        INSERT INTO history (video_path, question, answer)
        VALUES (?, ?, ?)
        """,
        (video_path, question, answer)
    )

    # 最多保留 20 条
    cursor.execute("SELECT COUNT(*) FROM history")
    count = cursor.fetchone()[0]

    if count > 20:
        cursor.execute(
            """
            DELETE FROM history
            WHERE id = (
                SELECT id FROM history
                ORDER BY created_at ASC
                LIMIT 1
            )
            """
        )

    conn.commit()
    conn.close()

# 常见问题
COMMON_QUESTIONS = [
    "Describe the scene.",
    "What is going on in the video?",
    "What is happening?",

    "What is the man doing?",
    "What are the women doing?",
    "What is the child doing?",
    "What are the animals doing?",
    "What is the person doing?",
    "What is people doing?",

    "What animal is in the video?",
    "What objects are visible in the scene?",
    "Who is in the video?",

    "How many people are in the pool?",
    "How many cars are on the road?",
    "How many people are in the video?",
    "How many animals are there?",

    "What color is the man's clothes?",
    "What color is the baby's clothes?",
    "What color is the pool water?",
    "What color is the animal?",

    "Is the man young or middle-aged?",
    "Is the child very young?",
    "Is the animal large or small?",
    "Is the object big or small?",
    "What does this person look like?",
    "How old does this person look?",

    "Where is the man ?",
    "Where is the child?",
    "Where is people?",
    "Where is the person running?",
    "where is it？",
    "What is the location of the people",
    "What is the position of the people",
]

@app.route("/", methods=["GET", "POST"])
def index():
    answer = ""
    video_path = ""
    question = ""

    if request.method == "POST":
        video_file = request.files["video"]
        question = request.form.get("question", "")  # ✅ 这一行是关键

        video_path = os.path.join("uploads", video_file.filename)
        video_file.save(video_path)

        if video_path and question:
            try:
                answer = video_qa(video_path, question)
            except Exception as e:
                answer = f"Error: {str(e)}"
            try:
                insert_history(video_path, question, answer)
            except Exception as e:
                print("Error saving history:", e)

    return render_template(
        "index.html",
        answer=answer,
        video_path=video_path,
        question=question,
        common_questions=COMMON_QUESTIONS
    )

# 历史记录接口（给网页 / 前端用）
@app.route("/history", methods=["GET"])
def get_history():
    conn = get_db()
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    cursor.execute(
        """
        SELECT id, video_path, question, answer, created_at
        FROM history
        ORDER BY created_at DESC
        """
    )

    rows = cursor.fetchall()
    print("DEBUG rows:", rows)  # 🔹 查看是否真的有数据
    conn.close()
    return jsonify([dict(row) for row in rows])

@app.route("/history/<int:record_id>", methods=["DELETE"])
def delete_history(record_id):
    conn = get_db()
    cursor = conn.cursor()

    cursor.execute("DELETE FROM history WHERE id = ?", (record_id,))
    conn.commit()
    conn.close()

    return jsonify({"status": "deleted"})

if __name__ == "__main__":
    app.run(debug=True)

