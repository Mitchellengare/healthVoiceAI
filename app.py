from flask import Flask, request, jsonify, render_template
from twilio.twiml.voice_response import VoiceResponse, Gather
from twilio.twiml.messaging_response import MessagingResponse
from openai import OpenAI
import os
from dotenv import load_dotenv
import logging
import json
import phonenumbers

load_dotenv()

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Phone Number Helpers 
def normalize_e164(number: str) -> str:
    """Ensure phone number is in international E.164 format."""
    number = (number or "").strip()
    if not number:
        return ""
    try:
        parsed = phonenumbers.parse(number, None)
        if phonenumbers.is_valid_number(parsed):
            return phonenumbers.format_number(
                parsed,
                phonenumbers.PhoneNumberFormat.E164
            )
    except Exception:
        pass
    return number


def infer_country_code(number_e164: str) -> str:
    """Return ISO country code like 'NG', 'GH', 'ZA' from phone number."""
    try:
        parsed = phonenumbers.parse(number_e164, None)
        region = phonenumbers.region_code_for_number(parsed)
        return region or "UNKNOWN"
    except Exception:
        return "UNKNOWN"


openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"), max_retries=0)
USER_LANG_PREF = {}

#Prompts

CLASSIFIER_PROMPT = """You are a classifier for a health education service used across Africa.

Given a user message, do THREE things:
1. Detect if it is health/medical related (true) or not (false).
2. Detect if it describes an emergency (life-threatening situation).
3. If NOT health related, write a short polite refusal IN THE EXACT SAME LANGUAGE the user wrote in.

An emergency includes: chest pain, difficulty breathing, severe bleeding, loss of consciousness, 
stroke symptoms, poisoning, severe injury, suicidal intent — in ANY language.

Health topics (is_health = true): symptoms, medications, lab results, diagnoses, medical terms, 
mental health, nutrition related to illness, chronic conditions, maternal health, HIV, malaria, 
tuberculosis, anything a doctor or pharmacist would discuss.

NOT health (is_health = false): coding, technology, finance, entertainment, beauty/grooming, sports.

Respond ONLY in this exact JSON format:
{
  "is_health": true or false,
  "is_emergency": true or false,
  "is_understandable": true or false,
  "language_code": "ISO 639-3 if possible (e.g., 'guz', 'rw', 'yo', 'ig', 'ha'); otherwise 'unknown'",
  "language_name": "language name in English or 'Unknown'",
  "language_confidence": 0.0 to 1.0,
  "refusal_message": "only include if is_health is false AND language_confidence >= 0.7 AND is_understandable is true"
}

Rules:
- If the message is too short, garbled, or ambiguous, set is_understandable=false.
- If you are not confident about the language, set language_code='unknown', language_name='Unknown', language_confidence < 0.7.
- Only generate refusal_message when you are confident you are matching the user's language.
"""

SYSTEM_PROMPT = """You are HealthVoice AI, a compassionate health educator serving patients across Africa.

Your ONLY role is to explain medical topics in simple, clear language.
Always respond in the same language the user writes in — including Swahili, Hausa, Yoruba, 
Amharic, Zulu, Somali, Wolof, Lingala, Shona, Igbo, Twi, Xhosa, French, Portuguese, Arabic, 
or any other language or language mix.

Health topics you cover:
- Medications, dosages, and side effects
- Lab results and what they mean
- Symptoms and when to seek care
- Maternal and child health
- HIV, malaria, tuberculosis, typhoid, and other common conditions in Africa
- Chronic conditions: diabetes, hypertension, sickle cell disease
- Mental health

Rules:
1. ONLY answer health/medical questions. If a question is not medical, politely refuse in the user's language.
2. Use very simple language — as if explaining to someone with no medical background.
3. Be culturally sensitive. If a patient mentions traditional or herbal remedies, acknowledge respectfully, then provide medical context.
4. NEVER say "call 911". Instead say: "call your local emergency number or go to the nearest hospital immediately."
5. If you don't know something, say so clearly and suggest seeing a doctor.
6. End every response with: "This is health education only. Please see a doctor or nurse for personal medical advice."
7. Keep responses under 300 words."""

EMERGENCY_RESPONSE_PROMPT = """The user has sent a message that may be a medical emergency. 

Write a SHORT, CALM emergency response in the EXACT same language as the user's message.
The response must:
- Acknowledge what they said
- Tell them to call their local emergency number or go to the nearest hospital immediately
- Be under 50 words
- NOT include any disclaimers or health education

User message: {user_input}
Detected language: {language}"""


def classify_message(user_input, country="UNKNOWN"):
    """
    Classify the message in one GPT call.
    Returns dict with: is_health, is_emergency, detected_language, refusal_message (optional)
    """
    try:
        response = openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": CLASSIFIER_PROMPT},
                {"role": "user", "content": f"[Caller country: {country}]\n{user_input}"}
            ],
            max_tokens=200,
            temperature=0,
            response_format={"type": "json_object"}
        )
        raw = response.choices[0].message.content
        if not raw:
            raise ValueError("Classifier returned empty content")
        result = json.loads(raw)

        return {
            "is_health": result.get("is_health", False),
            "is_emergency": result.get("is_emergency", False),
            "is_understandable": result.get("is_understandable", True),
            "language_code": result.get("language_code", "unknown"),
            "language_name": result.get("language_name", "Unknown"),
            "language_confidence": float(result.get("language_confidence", 0.0)),
            "refusal_message": result.get("refusal_message", None),
        }
    except Exception as e:
        logger.error(f"Classifier error: {str(e)}")
        # Fail CLOSED — prevents becoming a general-purpose bot if classification fails
        return {
                "is_health": False,
                "is_emergency": False,
                "is_understandable": True,
                "language_code": "unknown",
                "language_name": "Unknown",
                "language_confidence": 0.0,
                "refusal_message": "I can only help with health and medical questions."
        }

def generate_emergency_response(user_input, language):
    """Generate a localized emergency response in the user's language."""
    try:
        prompt = EMERGENCY_RESPONSE_PROMPT.format(user_input=user_input, language=language)
        response = openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=80,
            temperature=0
        )
        raw = response.choices[0].message.content
        if not raw:
            raise ValueError("Empty response from OpenAI")
        
        return raw.strip()
    except Exception as e:
        logger.error(f"Emergency response error: {str(e)}")
        # Hardcoded fallback
        return "⚠️ This sounds like an emergency. Please call your local emergency number or go to the nearest hospital immediately."


def generate_ai_response(user_input, country="UNKNOWN", user_id = None):
    """Main entry point. Classifies then responds appropriately."""
    try:
        # Step 1: Classify (handles emergency, language, and topic in one call)
        classification = classify_message(user_input, country=country)

        # Step 2: If user is setting a preferred language
        if classification.get("intent") == "set_language":
            requested = (classification.get("requested_language") or "").strip()
            if requested and user_id:
                USER_LANG_PREF[user_id] = requested
                return f"Okay. I will reply in {requested}. Now ask your health question."
            return "Okay. Tell me your health question, and I will reply in that language."

        # Step 3: Emergency takes priority
        if classification["is_emergency"]:
            return generate_emergency_response(user_input, classification.get("language_name", "Unknown"))

        # Step 4: If we can't understand the message OR we can't confidently identify the language,
        if (not classification.get("is_understandable", True)) and (classification.get("language_confidence", 1.0) < 0.7):
            return (
                    "I didn’t understand your message clearly. Please retype your health question in your preferred language, "
                    "or tell me the language name you want me to use."
                    )

        # Step 5: Off-topic refusal
        if not classification["is_health"]:
            return classification["refusal_message"] or "I can only help with health and medical questions."

        # Step 6: Generate health response
        preferred = USER_LANG_PREF.get(user_id) if user_id else None
        system = SYSTEM_PROMPT + f"\n\nContext: Caller country is {country}."
        if preferred:
            system += f"\nUser prefers responses in: {preferred}. Always respond in that language."

        response = openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT + f"\n\nContext: Caller country is {country}."},
                {"role": "user", "content": user_input}
            ],
            max_tokens=300,
            temperature=0.3  # Lower = more consistent for medical info
        )
        raw = response.choices[0].message.content

        if not raw:
            raise ValueError("Empty response from OpenAI")
        return raw.strip()

    except Exception as e:
        logger.error(f"Error generating AI response: {str(e)}")
        return "I'm having trouble right now. Please contact your healthcare provider directly."


#Routes 
@app.route('/')
def home():
    return render_template('index.html')


@app.route('/voice', methods=['POST'])
def voice():
    from_number_raw = request.form.get("From", "")
    from_number = normalize_e164(from_number_raw)
    country = infer_country_code(from_number)

    logger.info(f"VOICE call from {from_number} (country={country})")

    response = VoiceResponse()


    response.say(
        "Welcome to HealthVoice AI. I am your health educator. Please ask your health question.",
        voice='alice',
        language='en-US'  
        # Note: Twilio alice voice supports limited languages.
        # For broader African language support, consider Twilio's Amazon Polly 
        # or Google TTS voices which support more languages.
    )
    response.pause(length=1)

    gather = Gather(
        input='speech',
        action='/voice/process',
        method='POST',
        speechTimeout='auto',
        language='en-US',  # Twilio will attempt to transcribe; swap for user's language if known
        hints='medication, lab results, blood pressure, malaria, diabetes, HIV, tuberculosis, pregnancy'
    )
    gather.say("Please ask your question now.")
    response.append(gather)

    response.redirect('/voice')
    return str(response)


@app.route('/voice/process', methods=['POST'])
def process_voice():
    speech_result = request.form.get('SpeechResult', '')

    if not speech_result:
        response = VoiceResponse()
        response.say("I did not catch that. Please try again.", voice='alice')
        response.redirect('/voice')
        return str(response)

    logger.info(f"Voice query: {speech_result}")
    country = request.args.get("country", "UNKNOWN")
    ai_response = generate_ai_response(speech_result, country=country)

    response = VoiceResponse()
    response.say(ai_response, voice='alice', language='en-US')
    response.pause(length=2)

    gather = Gather(
        input='speech dtmf',
        action='/voice/continue?country={country}',
        method='POST',
        speechTimeout=3
    )
    gather.say("Say yes or press 1 to ask another question. Say no or press 2 to end.")
    response.append(gather)

    response.redirect('/voice/end')
    return str(response)


@app.route('/voice/continue', methods=['POST'])
def voice_continue():
    response = VoiceResponse()
    user_choice = request.form.get('SpeechResult', '').lower() or request.form.get('Digits', '')

    if user_choice in ['yes', '1', 'continue', 'ndiyo', 'oui', 'naam']:  # yes in Swahili, French, Arabic
        response.redirect('/voice')
    else:
        response.say("Thank you for using HealthVoice AI. Please see a doctor for personal advice. Goodbye!", voice='alice')
        response.hangup()
    return str(response)


@app.route('/voice/end')
def voice_end():
    response = VoiceResponse()
    response.say("Thank you for using HealthVoice AI. Goodbye!", voice='alice')
    response.hangup()
    return str(response)


@app.route('/sms', methods=['POST'])
def sms():
    incoming_msg = request.form.get('Body', '').strip()
    from_number_raw = request.form.get('From', '')

    from_number = normalize_e164(from_number_raw)
    country = infer_country_code(from_number)

    logger.info(f"SMS from {from_number} (country={country}): {incoming_msg}")

    ai_response = generate_ai_response(incoming_msg, country=country, user_id=from_number)

    # No hardcoded English footer — keep it clean for multilingual users
    resp = MessagingResponse()
    resp.message(ai_response)
    return str(resp)


@app.route('/test', methods=['GET'])
def test():
    return jsonify({
        "status": "healthy",
        "server": "running",
        "openai_key_loaded": bool(os.getenv("OPENAI_API_KEY")),
    })


@app.route("/api/ask", methods=["POST"])
def api_ask():
    data = request.get_json(silent=True) or {}
    question = (data.get("question") or "").strip()

    if not question:
        return jsonify({"error": "Missing 'question'"}), 400

    answer = generate_ai_response(question)
    return jsonify({"question": question, "answer": answer})


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8000))
    app.run(host='0.0.0.0', port=port, debug=os.getenv('DEBUG', 'False') == 'True')