import os
import yt_dlp
import whisper
import subprocess
from together import Together
import re
from flask import Flask, render_template, request, jsonify
import tempfile

app = Flask(__name__)

# =========================
# TOGETHER AI CLIENT
# =========================
client = Together(api_key="9bed3ed14a70a199a753547a90c53da4f45468ce3c7b9f7ffa9a3983b4610735")

# =========================
# UTILITY FUNCTIONS
# =========================


def download_from_url(url, output_path="downloads"):
    os.makedirs(output_path, exist_ok=True)
    ydl_opts = {
        "format": "bestvideo+bestaudio/best",
        "outtmpl": os.path.join(output_path, "%(title)s.%(ext)s"),
        "merge_output_format": "mp4",
        "restrictfilenames": True
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        try:
            info = ydl.extract_info(url, download=True)
            filename = ydl.prepare_filename(info)
            return filename
        except yt_dlp.utils.DownloadError as e:
            print(f"❌ Download Error: {e}")
            return None

def convert_to_audio(input_file):
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmpfile:
        output_file = tmpfile.name
        try:
            subprocess.run(
                [
                    "ffmpeg", "-y",
                    "-i", input_file,
                    "-vn",
                    "-acodec", "pcm_s16le",
                    "-ar", "16000",
                    "-ac", "1",
                    output_file
                ],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            print(f"🎶 Extracted audio: {output_file}")
            return output_file
        except subprocess.CalledProcessError:
            print("❌ Error converting video to audio")
            return None

def transcribe_audio(file_path):
    ext = os.path.splitext(file_path)[1].lower()
    if ext in [".mp4", ".mkv", ".avi", ".mov"]:
        file_path = convert_to_audio(file_path)
    
    # Handle cases where conversion to audio fails
    if not file_path:
        raise Exception("Audio conversion failed, cannot transcribe.")

    model = whisper.load_model("base")
    # 👇 force silent mode
    result = model.transcribe(file_path, verbose=False, fp16=False)

    transcript = result["text"]
    
    # This print statement has been removed to stop printing the transcript to the console.
    # print("✅ Transcript generated.")
    
    return transcript

def summarize_transcript(transcript):
    prompt = f"""
    Summarize the following lecture in a clear, simple, and student-friendly way.
    {transcript}
    """
    response = client.chat.completions.create(
        model="mistralai/Mixtral-8x7B-Instruct-v0.1",
        messages=[
            {"role": "system", "content": "You are a helpful AI teacher."},
            {"role": "user", "content": prompt},
        ],
        max_tokens=800,
        temperature=0.4,
    )
    return response.choices[0].message.content

def generate_quiz(summary, num_questions=10):
    prompt = f"""
    Based on the following summary, generate {num_questions} multiple-choice quiz questions.

    Format STRICTLY like this:
    1. Question?
        A) Option 1
        B) Option 2
        C) Option 3
        D) Option 4
        Correct Answer: B

    Summary: {summary}
    """

    response = client.chat.completions.create(
        model="meta-llama/Llama-3-8b-chat-hf",
        messages=[{"role": "user", "content": prompt}],
    )

    quiz_text = response.choices[0].message.content.strip()

    q_and_a = re.findall(
        r"(\d+)\.\s*(.*?)\n\s*A\)(.*?)\n\s*B\)(.*?)\n\s*C\)(.*?)\n\s*D\)(.*?)\n\s*Correct Answer:\s*([A-D])",
        quiz_text, re.S
    )

    quiz = []
    answer_key = {}
    for q in q_and_a:
        q_num, question, A, B, C, D, correct = q
        quiz.append({
            "question": question.strip(),
            "options": {
                "A": A.strip(),
                "B": B.strip(),
                "C": C.strip(),
                "D": D.strip(),
            }
        })
        answer_key[int(q_num)] = correct.strip()

    return quiz, answer_key

def generate_extended_notes(mistakes, transcript):
    if not mistakes:
        return "🎉 Great job! You mastered all topics."

    prompt = f"""
    The student made mistakes in the following quiz questions/topics:
    {mistakes}

    Based on the original lecture transcript below, explain these topics
    in detail with examples and clear explanations so the student can
    understand better.

    Transcript: {transcript}
    """
    try:
        response = client.chat.completions.create(
            model="mistralai/Mixtral-8x7B-Instruct-v0.1",
            messages=[
                {"role": "system", "content": "You are a helpful AI teacher."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=800,
            temperature=0.5,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error generating extended notes: {e}")
        return "❌ Could not generate extended notes."

# =========================
# FLASK ROUTES
# =========================
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process_data():
    data_type = request.form.get('type')
    transcript = None
    
    try:
        if data_type == 'upload':
            if 'file' not in request.files:
                return jsonify({"error": "No file part"}), 400
            file = request.files['file']
            if file.filename == '':
                return jsonify({"error": "No selected file"}), 400
            
            with tempfile.NamedTemporaryFile(delete=False) as tmpfile:
                file.save(tmpfile.name)
                transcript = transcribe_audio(tmpfile.name)
            os.remove(tmpfile.name)

        elif data_type == 'url':
            url = request.form.get('url')
            downloaded_file = download_from_url(url)
            
            if downloaded_file:
                transcript = transcribe_audio(downloaded_file)
                os.remove(downloaded_file)
            else:
                return jsonify({"error": "Failed to download the video from the provided URL."}), 500
        
        else:
            return jsonify({"error": "Invalid data type"}), 400
        
        summary = summarize_transcript(transcript)
        quiz, answer_key = generate_quiz(summary, num_questions=10)

        return jsonify({
            "transcript": transcript,
            "summary": summary,
            "quiz": quiz,
            "answer_key": answer_key
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/submit_quiz', methods=['POST'])
def submit_quiz():
    user_answers = request.json.get('user_answers', {})
    answer_key = request.json.get('answer_key', {})
    transcript = request.json.get('transcript', "")
    
    score = 0
    mistakes = []
    
    for q_num, user_ans in user_answers.items():
        if user_ans == answer_key.get(q_num):
            score += 1
        else:
            mistakes.append(f"Question {q_num}")

    extended_notes = generate_extended_notes(mistakes, transcript)
    
    return jsonify({
        "score": score,
        "total": len(answer_key),
        "notes": extended_notes
    })

if __name__ == '__main__':
    if not os.path.exists('downloads'):
        os.makedirs('downloads')
    app.run(debug=True)
