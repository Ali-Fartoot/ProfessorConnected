from abc import ABC, abstractmethod
from openai import OpenAI

class LLMAgent(ABC):
    """
    Abstract base class for LLM agents that handle image processing and inference.
    """
    def __init__(self, message = None):
        self.message_template = message or []
        self.client = OpenAI(base_url="http://localhost:5333/v1", api_key="llama.cpp")

    @abstractmethod
    def infer(self, 
              temperature: float = 0.9,
              max_token: int = 500,
              n: int = 1,
              stop: str = None) -> any:
        """
        Abstract method for model inference.
        
        Args:
            temperature (float): Sampling temperature
            max_token (int): Maximum number of tokens to generate
            n (int): Number of completions to generate
            stop (str): Stop sequence
            
        Returns:
            Implementation dependent return type
        """
        pass


class SummarizerAgent(LLMAgent):
    def __init__(self, message: list[dict] = None):
        super().__init__(message)
        self.message_template = message or [
            {
                "role": "system",
                "content": """You are a assitant text summarization assistant. Your task is to create concise, 
                accurate summaries. Please Keep the summary length to a minimum but point at techniques."""
            },
            {
                "role": "user",
                "content": None
            }
        ]

    def infer(self, 
              text: str, 
              temperature: float = 0.6, 
              max_token: int = 8096, 
              n: int = 1, 
              stop: str = None) -> str:

        self.message_template[1]["content"]= str(text)
        response = self.client.chat.completions.create(
            model="local-model",
            messages=self.message_template,
            temperature=temperature,
            max_tokens=max_token,
            n=n,
            stop=stop    
        )
        return response.choices[0].message.content



class SummarizerAgent(LLMAgent):
    def __init__(self, message: list[dict] = None):
        super().__init__(message)
        self.message_template = message or [
            {
                "role": "system",
                "content": """You are a assitant text summarization assistant. Your task is to create concise, 
                accurate summaries. Please Keep the summary length to a minimum but point at techniques."""
            },
            {
                "role": "user",
                "content": None
            }
        ]

    def infer(self, 
              text: str, 
              temperature: float = 0.6, 
              max_token: int = 8096, 
              n: int = 1, 
              stop: str = None) -> str:

        self.message_template[1]["content"]= str(text)
        response = self.client.chat.completions.create(
            model="local-model",
            messages=self.message_template,
            temperature=temperature,
            max_tokens=max_token,
            n=n,
            stop=stop    
        )
        return response.choices[0].message.content


class ExpandKeywordsAgent(LLMAgent):
    def __init__(self, message: list[dict] = None):
        super().__init__(message)
        self.message_template = message or [
            {
                "role": "system",
                "content": "I gave a bunch of keywords. Can you expand on them by Merging or lingual and generate new keywords?"

            },
            {
             "role": "user",
             "content":None
            }
        ]
    
    def infer(self, 
              keywords: list, 
              temperature: float = 0.6, 
              max_token: int = 8091, 
              n: int = 1, 
              stop: str = None) -> str:
        
        keyword_str = "\n".join([f"{i+1}. {keyword}" for i, keyword in enumerate(keywords)])
        self.message_template[1]["content"]= keyword_str
        response = self.client.chat.completions.create(
            model="local-model",
            messages=self.message_template,
            temperature=temperature,
            max_tokens=max_token,
            n=n,
            stop=stop    
        )
        return response.choices[0].message.content
