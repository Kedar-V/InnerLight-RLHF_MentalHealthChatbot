from google import genai
from google.genai import types
from flask import Flask, request, jsonify
from policy import Policy
from database import add_interaction
import os
import json
import logging
import sys
import subprocess

# -------------------------------
# LOGGING CONFIG
# -------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

logger = logging.getLogger(__name__)

# -------------------------------
# FLASK APP
# -------------------------------

app = Flask(__name__)

# -------------------------------
# LOAD LOCAL PPO POLICY
# -------------------------------
policy = Policy(model_path="/models/policy.pt")

# -------------------------------
# GEMINI CLIENT
# -------------------------------
client = genai.Client(api_key="AIzaSyACAvQyTMC5_JtHmjaT0-ceozhowgmMFMA")

# -------------------------------
# OPENAI CALL (MODERN + SAFE)
# -------------------------------

def _call_gemini(prompt: str):

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        config=types.GenerateContentConfig(
            system_instruction='''
            You are InnerLight, a calm, empathetic, and non-judgmental mental health support assistant. 
            Your role is to provide emotional support, gentle guidance, and reflective listening to users who may be experiencing stress, anxiety, sadness, loneliness, or emotional difficulty.

            Core principles you must always follow:
            - Be warm, compassionate, and validating. Acknowledge the user's feelings without judgment.
            - Never diagnose medical or psychiatric conditions.
            - Never prescribe medication or medical treatment.
            - Do not present yourself as a therapist, doctor, or replacement for professional care.
            - Encourage healthy coping strategies such as grounding, journaling, rest, breathing, self-reflection, and reaching out for social support.
            - Ask gentle, open-ended questions to understand the user's emotional state better.
            - Keep responses clear, calm, and emotionally supportive, not overly clinical.
            - Avoid moralizing, blaming, or dismissing emotions.

            Crisis handling rules:
            - If the user expresses suicidal thoughts, self-harm intent, or imminent danger:
            - Respond with high empathy and seriousness.
            - Encourage them to seek immediate help from local emergency services, a crisis hotline, or a trusted person.
            - If in the U.S., recommend calling or texting 988. Otherwise, suggest contacting local emergency numbers.
            - Do not attempt to handle a suicide crisis alone.
            - Never normalize or encourage self-harm.

            Boundaries:
            - You may offer coping techniques and emotional reframing.
            - You may provide general mental wellness education.
            - You may not provide diagnosis, therapy plans, or legal/medical advice.
            - You must respect privacy and never pressure the user to disclose more than they choose.

            Tone and style:
            - Gentle, human, calm, reassuring.
            - Use short, grounded sentences.
            - Avoid excessive emojis or casual slang.
            - Speak like a compassionate listener, not an authority figure.

            Goal:
            Your goal is to help users feel heard, supported, emotionally safer, and gently guided toward healthier coping and, when needed, professional support.
            '''),
        contents=prompt
    )

    return response.text

# -------------------------------
# GENERATION LOGIC
# -------------------------------

def generate_response(prompt: str) -> dict:
    use_openai = bool(os.getenv("GEMINI_API_KEY"))

    if use_openai:
        resp = _call_gemini(prompt)
        src = "gemini"
    else:
        resp = policy.generate(prompt)
        src = "local_policy"

    try:
        add_interaction(prompt, resp)
    except Exception as e:
        logger.warning("Failed to save interaction: %s", e)

    return {"response": resp, "source": src}

# -------------------------------
# ROUTES
# -------------------------------

@app.route("/generate", methods=["POST"])
def generate():
    data = request.get_json(silent=True) or {}
    prompt = data.get("prompt")

    if not prompt:
        return jsonify({"error": "missing 'prompt' in JSON body"}), 400

    try:
        logger.info("Request received at /generate")
        logger.info("Prompt: %s", prompt)

        out = generate_response(prompt)
        return jsonify(out)

    except Exception as e:
        logger.exception("Generation failed")
        return jsonify({"error": str(e)}), 500


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})

# -------------------------------
# LOCAL DEV RUNNER
# -------------------------------

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "serve":
        app.run(host="0.0.0.0", port=8000, debug=True)
    else:
        try:
            user_prompt = input("Enter prompt (leave blank for default): ").strip()
        except Exception:
            user_prompt = ""

        if not user_prompt:
            user_prompt = "Hello from local test"

        try:
            out = generate_response(user_prompt)
            print("\n=== Generation Result ===")
            print(out)
        except Exception as e:
            print("Error during local test:", e)
