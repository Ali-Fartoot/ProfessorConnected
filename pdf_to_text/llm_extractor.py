from abc import ABC, abstractmethod
from openai import OpenAI
from keybert.llm import OpenAI as keybert_openai
from keybert import KeyLLM

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
                "content": """You are an assistant text summarization assistant."""
            },
            {
                "role": "user",
                "content": None
            }
        ]

    def infer(self, 
              text: str, 
              temperature: float = 0.5, 
              stop: str = None) -> str:

        self.message_template[1]["content"] = f'''Your task is to create concise, accurate summaries. Please point at techniques by considering keywords. The summerization should have two part introduction and conclusion. but don't mention that by exact world
                                                The text: {text}, The summarization:'''
        response = self.client.chat.completions.create(
            model="local-model",
            messages=self.message_template,
            temperature=temperature,
            stop=stop    
        )
        return response.choices[0].message.content.strip()


class KeyExtractorLLM(LLMAgent):
    def __init__(self, message=None):
        super().__init__(message)
        DEFAULT_PROMPT = """
        The following is a list of documents. Please extract the top keywords, separated by a comma, that describe the topic of the texts.
        Note Please remove specific name such as Person, Oragazonation, Country etc.

        Document:
        - Traditional diets in most cultures were primarily plant-based with a little meat on top, but with the rise of industrial style meat production and factory farming, meat has become a staple food.

        Keywords: Traditional diets, Plant-based, Meat, Industrial style meat production, Factory farming, Staple food, Cultural dietary practices

        Document:
        - The website mentions that it only takes a couple of days to deliver but I still have not received mine.

        Keywords: Website, Delivery, Mention, Timeframe, Not received, Waiting, Order fulfillment

        Document:
        - [DOCUMENT]

        Keywords:"""

        self.key_extractor = KeyLLM(keybert_openai(self.client, prompt=DEFAULT_PROMPT))

    def infer(self, text: str):
        # Use the KeyLLM instance to extract keywords
        keywords = self.key_extractor.extract_keywords(text)
        return keywords