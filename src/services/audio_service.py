from gtts import gTTS
from groq import Groq
#import os
from config.settings import GROQ_API_KEY, TTS_LANGUAGE

class AudioService:
    def __init__(self):
        self.groq_client = Groq(api_key=GROQ_API_KEY)
    
    def transcribe_audio(self, filename):
        with open(filename, "rb") as file:
            transcription = self.groq_client.audio.transcriptions.create(
                file=(filename, file.read()),
                model="whisper-large-v3-turbo",
                response_format="verbose_json",
            )
        return transcription.text
    
    def text_to_speech(self, text, output_file="response.mp3"):
        tts = gTTS(text, lang=TTS_LANGUAGE)
        tts.save(output_file)
        return output_file
