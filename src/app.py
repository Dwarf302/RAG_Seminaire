import gradio as gr
from models.embeddings import EmbeddingsModel
from services.audio_service import AudioService
from services.chat_service import ChatService

class VirtualAssistant:
    def __init__(self):
        self.embeddings_model = EmbeddingsModel()
        self.audio_service = AudioService()
        self.chat_service = ChatService()

    def process_input(self, audio, text, history):
        # Determine input source
        if text:
            user_message = text
        elif audio:
            user_message = self.audio_service.transcribe_audio(audio)
        else:
            return history, None, None, False

        # Get similar responses and generate response
        contexts = self.embeddings_model.get_similar_responses(user_message)
        prompt = self.chat_service.create_prompt(user_message, contexts)
        bot_response = self.chat_service.get_response(prompt, history)
        
        # Generate audio response
        audio_response = self.audio_service.text_to_speech(bot_response)
        
        # Update history
        history = history + [(user_message, bot_response)]
        
        return history, user_message, audio_response, True

    def create_ui(self):
        with gr.Blocks(theme=gr.themes.Soft(primary_hue="blue", secondary_hue="indigo")) as demo:
            gr.Markdown("# 🎙️ Your AI Assistant in Car Specs")
            
            with gr.Row():
                with gr.Column(scale=2):
                    chatbot = gr.Chatbot(show_label=False, height=500)
                    
                    with gr.Row():
                        audio_input = gr.Audio(sources=["microphone"], type="filepath")
                        text_input = gr.Textbox(label="Message")
                    
                    with gr.Row():
                        submit = gr.Button("📤 Send", variant="primary")
                        clear = gr.Button("🗑️ Delete", variant="secondary")
                
                with gr.Column(scale=1):
                    text_output = gr.Textbox(label="Transcription")
                    audio_output = gr.Audio(label="Answer", autoplay=True)
            
            should_autoplay = gr.State(False)
            
            # Event handlers
            submit.click(
                self.process_input,
                inputs=[audio_input, text_input, chatbot],
                outputs=[chatbot, text_output, audio_output, should_autoplay],
            ).then(
                lambda: (None, None),
                outputs=[audio_input, text_input],
            )
            
            clear.click(
                lambda: ([], None, None, False),
                outputs=[chatbot, text_output, audio_output, should_autoplay]
            )
            
            return demo

def main():
    assistant = VirtualAssistant()
    demo = assistant.create_ui()
    demo.launch(share=True)

if __name__ == "__main__":
    main()