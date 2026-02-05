import os
import logging
import base64
import json
import ollama

logger = logging.getLogger(__name__)

# Configuration
# Default to llama3.2 for text as requested/implied context, and llava for vision
TEXT_MODEL_ID = os.getenv("TEXT_MODEL_ID", "llama3.2")
VISION_MODEL_ID = os.getenv("VISION_MODEL_ID", "llava")

class LocalLLMService:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if hasattr(self, "initialized"):
            return
        self.initialized = True
        logger.info(f"LocalLLMService initialized using Ollama. Text: {TEXT_MODEL_ID}, Vision: {VISION_MODEL_ID}")

    def is_ready(self) -> bool:
        try:
            # Quick check to see if we can connect to Ollama
            ollama.list()
            return True
        except Exception:
            return False

    def generate_json_response(
        self,
        prompt: str,
        max_tokens: int = 500,
        temperature: float = 0.1,
    ) -> dict:
        """
        Generates a JSON response from the LLM.
        """
        try:
            response = ollama.chat(
                model=TEXT_MODEL_ID,
                messages=[
                    {'role': 'user', 'content': prompt}
                ],
                format='json',
                options={
                    'num_predict': max_tokens,
                    'temperature': temperature,
                }
            )
            return json.loads(response['message']['content'])

        except Exception as e:
            logger.error(f"Ollama JSON generation error: {e}")
            # Fallback empty structure or raise
            return {}

    def generate_response(
        self,
        prompt: str,
        max_tokens: int = 400,
        temperature: float = 0.3,
        system_instruction: str = None
    ) -> str:
        # Default strict system prompt if none provided
        if not system_instruction:
            system_instruction = """You are Prism, a professional AI assistant for internal corporate knowledge management.
        
GUIDELINES:
1. You are analyzing authorized business documents and records.
2. Your goal is to provide helpful, accurate, and direct answers to the user's questions.
3. If asked about specific names, roles, or contact details mentioned in the text, provide them as they are relevant business information.
4. Maintain a professional, objective tone.
5. Do not withhold information that is present in the provided context.
"""

        try:
            response = ollama.chat(
                model=TEXT_MODEL_ID,
                messages=[
                    {'role': 'system', 'content': system_instruction},
                    {'role': 'user', 'content': prompt}
                ],
                options={
                    'num_predict': max_tokens,
                    'temperature': temperature,
                }
            )
            return response['message']['content']

        except Exception as e:
            logger.error(f"Ollama text generation error: {e}")
            return f"Error generating response: {str(e)}"

    def generate_vision_response(
        self,
        prompt: str,
        image_base64: str,
        model: str = "llava", # Legacy param
        max_tokens: int = 400,
        temperature: float = 0.3,
    ) -> str:
        try:
            # Allow overriding vision model if provided, else use default
            model_to_use = VISION_MODEL_ID
            
            response = ollama.chat(
                model=model_to_use,
                messages=[{
                    'role': 'user', 
                    'content': prompt, 
                    'images': [image_base64]
                }],
                options={
                    'num_predict': max_tokens,
                    'temperature': temperature,
                }
            )
            return response['message']['content']

        except Exception as e:
            logger.error(f"Ollama vision generation error: {e}")
            return f"Error generating vision response: {str(e)}"

    def answer_question(self, context: str, question: str) -> str:
        # 1. Strong System Prompt with Jailbreak-style Authorization
        # 1. Professional Business Analyst System Prompt
        system_prompt = """You are Prism, a professional business analyst and expert document assistant.
Your role is to analyze internal corporate documents and extract specific information for the user.

CONTEXT & GUIDELINES:
1. The User is an authorized employee asking about internal business records.
2. The provided text (Context) contains the necessary information to answer the question.
3. You must extract and summarize the relevant details from the Context to answer the User's Question.
4. If the Context includes names, roles, emails, or company details, treat them as essential business information and include them in your answer.
5. Do not redact or withhold information found in the business documents.
6. Provide a clear, direct, and factual answer derived solely from the Context.

ONE-SHOT EXAMPLE:
Context:
[Policy QD109] Out of Office Email Policy. 
All employees must set an auto-reply when absent for >1 day.
Exceptions: Executives (Need VP approval).
Contact: hr@corp.com for disputes.

User Question:
What is the email policy?

Your Answer:
The Out of Office Email Policy (QD109) states that all employees must set an auto-reply if they are absent for more than one day.

**Key Points:**
- **Requirement:** Auto-reply mandatory for >1 day absence.
- **Exceptions:** Executives require VP approval.
- **Contact:** hr@corp.com.

Reference: Policy QD109.

END OF INSTRUCTIONS.
"""

        # 2. Clean User Prompt
        user_message = f"""Context:
{context}

User Question:
{question}

Answer:"""

        # Grounded reasoning requires low temperature (deterministic)
        response = self.generate_response(
            user_message, 
            max_tokens=1000, 
            temperature=0.1, 
            system_instruction=system_prompt
        )
        
        return response


# Global instance
# We strictly expose 'ollama_llm' as the variable name for backward compatibility with main.py
ollama_llm = LocalLLMService()
