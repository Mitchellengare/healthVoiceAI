from flask import Flask, request, jsonify, render_template
from twilio.twiml.voice_response import VoiceResponse, Gather
from twilio.twiml.messaging_response import MessagingResponse
from openai import OpenAI
import os
from dotenv import load_dotenv
import logging

# Load environment variables
load_dotenv()

app = Flask(__name__)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize clients
twilio_client = None  # We'll initialize only if credentials exist
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"), max_retries=0)


# Medical safety keywords for emergency detection
EMERGENCY_KEYWORDS = [
    'chest pain', 'heart attack', 'stroke', 'can\'t breathe', 'difficulty breathing',
    'severe bleeding', 'unconscious', 'suicide', 'kill myself', '911', 'emergency'
]

# Medical disclaimer
DISCLAIMER = "I am an AI health educator, not a doctor. This information is for educational purposes only. Always consult with a healthcare professional for medical advice."

# System prompt for GPT
SYSTEM_PROMPT = """You are a compassionate, patient health educator named HealthVoice AI. 
Your role is to explain medical terms, lab results, and medication instructions in simple, 
clear language for someone with an 8th-grade reading level.

Rules:
1. Always be empathetic and calm
2. Break down complex medical jargon into everyday language
3. If explaining lab results, mention what's normal and when to worry
4. If explaining medications, include how to take them and common side effects
5. If you don't know something, say so and suggest consulting a doctor
6. Always include this disclaimer at the end: "Remember: I'm an AI assistant for education, not a replacement for your doctor's advice."

Language:
- Default to English.
- If the user explicitly asks for another language, reply in that language.

Keep responses under 300 words. Speak slowly and clearly as if explaining to a grandparent."""

def is_emergency(user_input):
    """Check if the user input contains emergency keywords"""
    user_input_lower = user_input.lower()
    for keyword in EMERGENCY_KEYWORDS:
        if keyword in user_input_lower:
            return True, keyword
    return False, None

def generate_ai_response(user_input):
    """Generate a safe, helpful medical explanation using GPT"""
    try:
        # First check for emergencies
        emergency, keyword = is_emergency(user_input)
        if emergency:
            return f"⚠️ EMERGENCY DETECTED: You mentioned '{keyword}'. This sounds serious. Please hang up and call 911 immediately or go to the nearest emergency room. Do not wait."
        
        # Generate AI response
        response = openai_client.chat.completions.create(
            model="gpt-3.5-turbo",  # Use gpt-4 if available
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_input}
            ],
            max_tokens=300,
            temperature=0.7
        )
        
        ai_response = response.choices[0].message.content
        return f"{ai_response}\n\n{DISCLAIMER}"
    
    except Exception as e:
        logger.error(f"Error generating AI response: {str(e)}")
        return "I'm having trouble processing your question right now. Please try again in a few moments or contact your healthcare provider directly."


@app.route('/')
def home():
    """Home page with instructions"""
    return render_template('index.html')

@app.route('/voice', methods=['POST'])
def voice():
    """Handle incoming voice calls"""
    response = VoiceResponse()
    
    # Welcome message
    response.say("Welcome to HealthVoice AI. I'm your friendly health educator.", 
                 voice='alice', language='en-US')
    response.pause(length=1)
    
    # Gather user speech
    gather = Gather(
        input='speech',
        action='/voice/process',
        method='POST',
        speechTimeout='auto',
        language='en-US',
        hints='what does, explain, how to, medication, lab, test, blood, sugar, cholesterol, A1C, diabetes, blood pressure'
    )
    gather.say("Please ask your health question after the beep. For example: What does high blood pressure mean? Or: How do I take my medication?")
    
    response.append(gather)
    
    # If no speech detected
    response.redirect('/voice')
    
    return str(response)

@app.route('/voice/process', methods=['POST'])
def process_voice():
    """Process speech input from voice call"""
    speech_result = request.form.get('SpeechResult', '')
    
    if not speech_result:
        response = VoiceResponse()
        response.say("I didn't catch that. Please try asking your question again.", voice='alice')
        response.redirect('/voice')
        return str(response)
    
    logger.info(f"Voice query: {speech_result}")
    
    # Generate AI response
    ai_response = generate_ai_response(speech_result)
    
    # Convert to speech
    response = VoiceResponse()
    response.say(ai_response, voice='alice', language='en-US')
    response.pause(length=2)
    
    # Offer to continue
    response.say("Would you like to ask another question?", voice='alice')
    gather = Gather(
        input='speech dtmf',
        action='/voice/continue',
        method='POST',
        speechTimeout=3
    )
    gather.say("Say yes or press 1 to continue. Say no or press 2 to end the call.")
    response.append(gather)
    
    # Timeout handler
    response.redirect('/voice/end')
    
    return str(response)

@app.route('/voice/continue', methods=['POST'])
def voice_continue():
    """Handle continuation choice"""
    response = VoiceResponse()
    user_choice = request.form.get('SpeechResult', '').lower() or request.form.get('Digits', '')
    
    if user_choice in ['yes', '1', 'continue']:
        response.redirect('/voice')
    else:
        response.say("Thank you for using HealthVoice AI. Remember to always consult with your doctor for medical advice. Goodbye!", voice='alice')
        response.hangup()
    
    return str(response)

@app.route('/voice/end')
def voice_end():
    """End call handler"""
    response = VoiceResponse()
    response.say("I didn't hear a response. Thank you for using HealthVoice AI. Goodbye!", voice='alice')
    response.hangup()
    return str(response)

@app.route('/sms', methods=['POST'])
def sms():
    """Handle incoming SMS/WhatsApp messages"""
    incoming_msg = request.form.get('Body', '').strip()
    from_number = request.form.get('From', '')
    
    logger.info(f"SMS from {from_number}: {incoming_msg}")
    
    ai_response = generate_ai_response(incoming_msg)

    footer = "\n\n📞 Need voice help? Call our toll-free number: [Your Number Here]\n💡 Tip: Ask about medications, lab results, or medical terms!"
    
    resp = MessagingResponse()
    resp.message(ai_response + footer)
    return str(resp)


@app.route('/test', methods=['GET'])
def test():
    """Health check endpoint (no OpenAI calls)."""
    return jsonify({
        "status": "healthy",
        "server": "running",
        "openai_key_loaded": bool(os.getenv("OPENAI_API_KEY")),
        "note": "No OpenAI call here. Use POST /api/ask to query the AI."
    })

@app.route("/api/ask", methods=["POST"])
def api_ask():
    """Web demo endpoint: call OpenAI once per question."""
    data = request.get_json(silent=True) or {}
    question = (data.get("question") or "").strip()

    if not question:
        return jsonify({"error": "Missing 'question'"}), 400

    answer = generate_ai_response(question)
    return jsonify({"question": question, "answer": answer})


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8000))
    app.run(host='0.0.0.0', port=port, debug=os.getenv('DEBUG', 'False') == 'True')