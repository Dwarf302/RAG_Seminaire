from config.settings import LLM_MODEL, GROQ_API_KEY
from groq import Groq

class ChatService:
    def __init__(self):
        self.client = Groq(api_key=GROQ_API_KEY)
    
    def format_conversation_history(self, history):
        messages = [{"role": "system", "content": "You are a helpful and friendly voice assistant."}]
        for human, assistant in history:
            messages.append({"role": "user", "content": human})
            messages.append({"role": "assistant", "content": assistant})
        return messages

    def create_prompt(self, query, contexts):
        prompt_template = """
        You need to answer the given question using the provided contexts to find the most relevant response.

        Notes:
        - If you don't have a relevant answer, simply respond with: "Sorry, I don't have an answer to your request!"
        - Keep your response concise, with a maximum of four sentences.
        - If the user simply says "Hello," respond with: "Hello! Welcome to the Car Specifications Assistant. How can I help you today?"
        - If the question is unrelated to car specifications, simply respond with: "Sorry, I don't have an answer to your request!"

        Here is the question: {query}
        Here is the first context: {context1}
        Here is the second context: {context2}
        """
        return prompt_template.format(query=query, context1=contexts[0], context2=contexts[1])

    def get_response(self, prompt, history):
        messages = self.format_conversation_history(history)
        messages.append({"role": "user", "content": prompt})
        
        chat_completion = self.client.chat.completions.create(
            messages=messages,
            model=LLM_MODEL,
        )
        return chat_completion.choices[0].message.content
