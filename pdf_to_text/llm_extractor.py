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

class KeyExtractorAgent(LLMAgent):
    def __init__(self, message: list[dict] = None):
        super().__init__(message)
        self.message_template = message or [
            {
                "role": "system",
                "content": """You are a specialized keyword extraction assistant. Your task is to extract relevant keywords from the given text while following these guidelines:

                EXCLUDE the following types of words:
                - Proper nouns (names of people, authors, universities, organizations)
                - Common stop words and articles (the, a, an, etc.)
                - Dates and numbers
                - Generic terms without specific meaning
                
                INCLUDE:
                - Technical terms and concepts
                - Important domain-specific vocabulary
                - Action words and significant descriptors
                - Key themes and topics
                
                Format your response as:
                - Return keywords in lowercase and a Python list format like ["keyword1", "keyword2", ...]. 
                - Focus on meaningful terms that capture the essence of the content
                - Each keyword should be a string in the list
                
                Example:
                Input: "Professor Smith from Harvard University discussed machine learning algorithms"
                Output: ["machine learning", "algorithms"]
                """
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": None
                    }
                ]
            }
        ]

    def infer(self, 
              text: str, 
              temperature: float = 0.7, 
              max_token: int = 2048, 
              n: int = 1, 
              stop: str = None) -> list[str]:

        # Update the message template with the input text
        self.message_template[1]["content"][0]["text"] = text
        
        response = self.client.chat.completions.create(
            model="local-model",
            messages=self.message_template,
            temperature=temperature,
            max_tokens=max_token,
            n=n,
            stop=stop    
        )
        return response.choices[0].message.content
        # try:
        #     keywords = eval(response.choices[0].message.content)
        #     if not isinstance(keywords, list):
        #         return []
        #     return [str(k).lower() for k in keywords if isinstance(k, str)]
        # except:
        #     raise TypeError(f"The Parser for {self.__class__.__name__} failed. Agent didn't return as a list!")
        


class SummarizerAgent(LLMAgent):
    def __init__(self, message: list[dict] = None):
        super().__init__(message)
        self.message_template = message or [
            {
                "role": "system",
                "content": """You are a specialized text summarization assistant. Your task is to create concise, 
                accurate summaries while following these guidelines:

                SUMMARY REQUIREMENTS:
                - Maintain the core message and key points of the original text
                - Use clear and concise language
                - Preserve important facts and relationships
                - Remove redundant information
                - Keep the original meaning intact
                
                STRUCTURE:
                - Create a summary of 5-6 sentences
                - Focus on the main ideas and key findings
                - Maintain logical flow and coherence
                - Use active voice when possible
                
                FORMAT:
                - Return a single string containing the summary
                - Do not include phrases like "This text discusses..." or "This article is about..."
                - Start directly with the main content
                - Use proper punctuation and grammar
                
                Example:
                Input: "Professor Smith from Harvard University conducted a comprehensive study on climate change effects 
                in urban areas over the past decade. The research involved 20 major cities and found significant temperature 
                increases in city centers. The study also revealed that green spaces can reduce urban heat island effects by up to 30%."
                
                Output: "A comprehensive study on climate change effects in urban areas examined 20 major cities over a decade, 
                revealing significant temperature increases in city centers. Green spaces were found to reduce urban heat island 
                effects by up to 30%."
                """
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": None
                    }
                ]
            }
        ]

    def infer(self, 
              text: str, 
              temperature: float = 0.7, 
              max_token: int = 2048, 
              n: int = 1, 
              stop: str = None) -> str:

        self.message_template[1]["content"][0]["text"] = text
        response = self.client.chat.completions.create(
            model="local-model",
            messages=self.message_template,
            temperature=temperature,
            max_tokens=max_token,
            n=n,
            stop=stop    
        )
        return response.choices[0].message.content

        # try:
        #     summary = response.choices[0].message.content
        #     if not isinstance(summary, str):
        #         raise TypeError("Summary must be a string")
        #     return summary.strip()
        # except Exception as e:
        #     raise TypeError(f"The Parser for {self.__class__.__name__} failed: {str(e)}")