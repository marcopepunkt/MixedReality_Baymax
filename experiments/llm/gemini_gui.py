import os
import tkinter as tk
from tkinter import ttk, scrolledtext, filedialog, messagebox
from typing import Optional
import google.generativeai as genai
from dotenv import load_dotenv
from PIL import Image, ImageTk

class GeminiClient:
    def __init__(self, model_name: str = "gemini-1.5-flash"):
        """Initialize Gemini client with specified model."""
        load_dotenv()
        genai.configure(api_key="os.getenv('GEMINI_API_KEY')")
        self.model = genai.GenerativeModel(model_name)
    
    def generate_text(self, prompt: str, conversation: Optional[str] = None) -> str:
        """Generate text response from a prompt, optionally using conversation history."""
        if conversation:
            prompt = f"{conversation}\nUser: {prompt}"
        response = self.model.generate_content(prompt)
        return response.text
    
    def analyze_image(self, image_path: str, prompt: Optional[str] = None, conversation: Optional[str] = None) -> str:
        """Analyze an image with optional prompt and conversation context."""
        image = genai.upload_file(image_path)
        content = [image]
        if prompt:
            content.extend(["\n\n", prompt])
        if conversation:
            content.append(f"\n\nConversation so far:\n{conversation}")
        result = self.model.generate_content(content)
        return result.text

class GeminiGUI:
    def __init__(self):
        self.client = GeminiClient()
        self.setup_gui()
        self.current_image_path = None
        self.photo_image = None  # Store PhotoImage object
        self.conversation_history = ""  # Maintain conversation history
        
    def setup_gui(self):
        # Create main window
        self.root = tk.Tk()
        self.root.title("Gemini AI Interface")
        self.root.geometry("1000x800")
        
        # Create notebook for tabs
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(expand=True, fill='both', padx=10, pady=5)
        
        # Text Generation Tab
        self.text_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.text_tab, text='Text Generation')
        self.setup_text_tab()
        
        # Image Analysis Tab
        self.image_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.image_tab, text='Image Analysis')
        self.setup_image_tab()
        
    def setup_text_tab(self):
        # Prompt input
        prompt_frame = ttk.LabelFrame(self.text_tab, text="Enter Prompt")
        prompt_frame.pack(fill='both', expand=True, padx=5, pady=5)
        
        self.text_prompt = scrolledtext.ScrolledText(prompt_frame, height=5)
        self.text_prompt.pack(fill='both', expand=True, padx=5, pady=5)
        
        # Generate button
        generate_btn = ttk.Button(self.text_tab, text="Generate Response", 
                                command=self.generate_text_response)
        generate_btn.pack(pady=5)
        
        # Response output
        response_frame = ttk.LabelFrame(self.text_tab, text="Conversation")
        response_frame.pack(fill='both', expand=True, padx=5, pady=5)
        
        self.text_response = scrolledtext.ScrolledText(response_frame, height=10)
        self.text_response.pack(fill='both', expand=True, padx=5, pady=5)
        
    def setup_image_tab(self):
        # Top frame for image selection
        select_frame = ttk.Frame(self.image_tab)
        select_frame.pack(fill='x', padx=5, pady=5)
        
        self.image_path_var = tk.StringVar()
        path_entry = ttk.Entry(select_frame, textvariable=self.image_path_var)
        path_entry.pack(side='left', fill='x', expand=True, padx=(0, 5))
        
        select_btn = ttk.Button(select_frame, text="Select Image", 
                              command=self.select_image)
        select_btn.pack(side='right')
        
        # Create a frame for the image preview and analysis
        content_frame = ttk.Frame(self.image_tab)
        content_frame.pack(fill='both', expand=True, padx=5, pady=5)
        
        # Left side - Image preview
        preview_frame = ttk.LabelFrame(content_frame, text="Image Preview")
        preview_frame.pack(side='left', fill='both', expand=True, padx=(0, 5))
        
        self.image_label = ttk.Label(preview_frame)
        self.image_label.pack(padx=5, pady=5)
        
        # Right side - Analysis
        analysis_frame = ttk.Frame(content_frame)
        analysis_frame.pack(side='right', fill='both', expand=True)
        
        # Prompt input
        prompt_frame = ttk.LabelFrame(analysis_frame, text="Enter Prompt (Optional)")
        prompt_frame.pack(fill='x', padx=5, pady=5)
        
        self.image_prompt = scrolledtext.ScrolledText(prompt_frame, height=3)
        self.image_prompt.pack(fill='both', expand=True, padx=5, pady=5)
        
        # Analyze button
        analyze_btn = ttk.Button(analysis_frame, text="Analyze Image", 
                               command=self.analyze_image)
        analyze_btn.pack(pady=5)
        
        # Response output
        response_frame = ttk.LabelFrame(analysis_frame, text="Analysis Result")
        response_frame.pack(fill='both', expand=True, padx=5, pady=5)
        
        self.image_response = scrolledtext.ScrolledText(response_frame, height=10)
        self.image_response.pack(fill='both', expand=True, padx=5, pady=5)
    
    def load_and_resize_image(self, image_path, max_size=(400, 400)):
        """Load and resize image while maintaining aspect ratio"""
        image = Image.open(image_path)
        
        # Calculate aspect ratio
        aspect_ratio = min(max_size[0]/image.width, max_size[1]/image.height)
        new_size = (int(image.width * aspect_ratio), int(image.height * aspect_ratio))
        
        # Resize image
        resized_image = image.resize(new_size, Image.Resampling.LANCZOS)
        
        # Convert to PhotoImage
        return ImageTk.PhotoImage(resized_image)
            
    def generate_text_response(self):
        prompt = self.text_prompt.get("1.0", tk.END).strip()
        if not prompt:
            messagebox.showwarning("Warning", "Please enter a prompt first!")
            return
            
        try:
            self.text_response.insert(tk.END, f"User: {prompt}\n")
            self.text_response.insert(tk.END, "Generating response...\n")
            self.root.update_idletasks()
            
            response = self.client.generate_text(prompt, conversation=self.conversation_history)
            
            # Update conversation history
            self.conversation_history += f"User: {prompt}\nAI: {response}\n"
            
            # Display response
            self.text_response.insert(tk.END, f"AI: {response}\n")
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {str(e)}")
            
    def select_image(self):
        filetypes = (
            ('Image files', '*.jpg *.jpeg *.png *.gif *.bmp'),
            ('All files', '*.*')
        )
        
        filename = filedialog.askopenfilename(
            title='Select an image',
            filetypes=filetypes
        )
        
        if filename:
            self.current_image_path = filename
            self.image_path_var.set(filename)
            
            # Load and display the image
            try:
                self.photo_image = self.load_and_resize_image(filename)
                self.image_label.configure(image=self.photo_image)
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load image: {str(e)}")
            
    def analyze_image(self):
        if not self.current_image_path:
            messagebox.showwarning("Warning", "Please select an image first!")
            return
            
        prompt = self.image_prompt.get("1.0", tk.END).strip()
        
        try:
            self.image_response.insert(tk.END, "Analyzing image...\n")
            self.root.update_idletasks()
            
            response = self.client.analyze_image(self.current_image_path, prompt, conversation=self.conversation_history)
            
            # Update conversation history
            self.conversation_history += f"Image Analysis: {self.current_image_path}\nAI: {response}\n"
            
            # Display response
            self.image_response.insert(tk.END, f"AI: {response}\n")
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {str(e)}")
            
    def run(self):
        self.root.mainloop()

def main():
    app = GeminiGUI()
    app.run()

if __name__ == "__main__":
    main()
