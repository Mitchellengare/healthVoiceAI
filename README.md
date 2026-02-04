# HealthVoice AI

**Bridging the healthcare digital divide with voice-first AI explanations**

## The Problem 
40% of elderly and low-income patients don't use smartphones regularly, leaving them behind in the healthcare AI revolution. Complex medical jargon creates confusion and poor health outcomes.

##  Solution
HealthVoice AI provides AI-powered medical explanations through:
- **Toll-free voice calls** (works on any phone)
- **SMS** (works on basic mobile phones)
- No app installation required
- No smartphone needed

##  Features
- Voice call interface with speech recognition
- WhatsApp/SMS text interface
- AI-powered medical explanations (GPT-4)
- Emergency keyword detection (routes to 911)
- Multilingual support
- HIPAA-aware design

## Tech Stack
- **Backend:** Python + Flask
- **Voice/SMS:** Twilio API
- **AI:** OpenAI GPT + Whisper
- **Hosting:** Render/Railway/Heroku

##  How It Works
1. User calls toll-free number or texts to WhatsApp
2. Asks medical question in plain language
3. AI provides clear, simple explanation
4. Emergency symptoms trigger 911 directive
5. Conversation ends with doctor consultation reminder

##  Setup Instructions

### 1. Clone Repository
```bash
git clone https://github.com/yourusername/healthvoice-ai.git
cd healthvoice-ai



# pip install -r requirements.txt
# create virtual env
# pip install flask twilio openai python-dotenv gunicorn - install independecies
# .env has the api keys
# brew install ngrok/ngrok/ngrok for voice/sms
