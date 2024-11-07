import os
from typing import Optional
import google.generativeai as genai
from dotenv import load_dotenv
from rich.console import Console

console = Console()

class GeminiClient:
    def __init__(self, model_name: str = "gemini-1.5-flash"):
        """Initialize Gemini client with specified model."""
        # Load API key from .env file
        load_dotenv()
        genai.configure(api_key=os.getenv('GEMINI_API_KEY'))
        self.model = genai.GenerativeModel(model_name)
    
    def generate_text(self, prompt: str) -> str:
        """Generate text response from a prompt."""
        with console.status("Generating response..."):
            response = self.model.generate_content(prompt)
            return response.text
    
    def analyze_image(self, image_path: str, prompt: Optional[str] = None) -> str:
        """Analyze an image with optional prompt."""
        with console.status("Analyzing image..."):
            # Upload and process image
            image = genai.upload_file(image_path)
            
            # Prepare content for generation
            content = [image]
            if prompt:
                content.extend(["\n\n", prompt])
            
            result = self.model.generate_content(content)
            return result.text


    

if __name__ == "__main__":
    # Initialize client
    client = GeminiClient()
    
    # Example 1: Text generation
    response = client.generate_text("Hello!")
    console.print("[green]Text Response:[/green]")
    console.print(response)
    
    # Example 2: Image analysis
    image_path = "./images/tram.jpg"
    prompt = "Describe this photo to a blind person to help them understand where they are and for navigation purposes."
    
    result = client.analyze_image(image_path, prompt)
    console.print("\n[green]Image Analysis:[/green]")
    console.print(result)