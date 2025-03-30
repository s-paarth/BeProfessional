import os
import requests
import json
import google.generativeai as genai
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.prompt import Prompt, Confirm
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.rule import Rule
from rich import box
import pyperclip
import time
import traceback
import random
from functools import lru_cache

# Load environment variables
load_dotenv(override=True)

# Initialize console
console = Console()

# Flag to control debug output
SHOW_DEBUG = False

# Cache for storing previous transformations to reduce API calls
RESPONSE_CACHE = {}

# Common transformations for fallback
COMMON_TRANSFORMATIONS = {
    "please share your thoughts": [
        "I would appreciate your insights on this matter.",
        "Could you please provide your perspective on this topic?",
        "Your input would be valuable in this discussion.",
        "I welcome your feedback on this subject.",
    ],
    "what do you think": [
        "What are your thoughts on this?", 
        "I'd appreciate your perspective on this matter.",
        "May I ask for your professional opinion on this?"
    ],
    "i don't like this": [
        "I have some concerns about this approach.",
        "I'd like to suggest an alternative solution.",
        "This approach may benefit from some adjustments."
    ],
    "this is stupid": [
        "I believe this approach could be reconsidered.",
        "I have some reservations about the effectiveness of this strategy.",
        "This solution might not be optimal for our objectives."
    ],
    "i'm angry about": [
        "I'm concerned about",
        "I feel strongly regarding",
        "I'd like to address my concerns about"
    ],
    "that's not my job": [
        "This falls outside my current responsibilities.",
        "This may require expertise from another department.",
        "This task might align better with a different team's objectives."
    ],
    "i quit": [
        "I would like to tender my resignation.",
        "I've decided to pursue opportunities elsewhere.",
        "I am giving my notice of resignation."
    ]
}

# Expanded fallback for informational questions
INFO_FALLBACKS = {
    "what is": "I'm inquiring about",
    "who is": "I'm seeking information regarding",
    "where is": "Could you provide information on the location of",
    "when is": "I'd like to know the timing of",
    "why is": "I'm interested in understanding the reason behind",
    "how to": "Could you provide guidance on how to"
}

def show_header():
    """Display a beautiful header for the app"""
    console.print(Rule(style="bright_blue"))
    text = Text()
    text.append("How To ", style="cyan bold")
    text.append("Professionally ", style="bright_cyan bold")
    text.append("Say", style="bright_white bold")
    text.append(" 🤵", style="bright_yellow")
    
    panel = Panel(
        text,
        box=box.ROUNDED,
        expand=False,
        border_style="bright_blue",
        padding=(1, 10)
    )
    console.print(panel)
    console.print("Transform your blunt thoughts into corporate-approved language", 
                  style="italic bright_black", justify="center")
    console.print(Rule(style="bright_blue"))

def format_output(input_text, output_text):
    """Format the output with original and transformed text"""
    original_content = Text(input_text, style="bright_white")
    professional_content = Text(output_text, style="bright_green")
    
    original_panel = Panel(
        original_content,
        title="Original Text",
        title_align="left",
        box=box.ROUNDED,
        border_style="yellow",
        padding=(1, 2)
    )
    
    pro_panel = Panel(
        professional_content,
        title="Professional Version",
        title_align="left",
        box=box.ROUNDED,
        border_style="green", 
        padding=(1, 2)
    )
    
    console.print("\n")
    console.print(original_panel)
    console.print("\n")
    console.print(pro_panel)
    
    # Copy to clipboard
    try:
        pyperclip.copy(output_text)
        console.print("\n[dim italic]✓ Copied to clipboard[/dim italic]", style="green")
    except:
        console.print("\n[dim italic]❌ Failed to copy to clipboard[/dim italic]", style="red")

@lru_cache(maxsize=100)
def get_cached_response(text):
    """Get cached response if available"""
    return RESPONSE_CACHE.get(text.lower().strip())

def save_to_cache(text, response):
    """Save response to cache"""
    RESPONSE_CACHE[text.lower().strip()] = response

def call_gemini_api(text, api_key):
    """Call the Gemini API using the official Python module"""
    # Check cache first
    cached = get_cached_response(text)
    if cached:
        if SHOW_DEBUG:
            console.print("[yellow]Using cached response...[/yellow]")
        return cached
    
    # Configure the Gemini API with the API key
    genai.configure(api_key=api_key)
    
    # Try the main API call
    response = try_with_module(text)
    
    # If successful, cache the result
    if response:
        save_to_cache(text, response)
        
    return response

def try_with_module(text):
    """Use the Google Generative AI module to call Gemini"""
    if SHOW_DEBUG:
        console.print("[yellow]Using Gemini module with gemini-2.0-flash...[/yellow]")
    
    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[yellow]Generating professional response...[/yellow]"),
            transient=True,
        ) as progress:
            progress.add_task("", total=None)
            
            prompt = f"""Transform this casual text into professional workplace language:
            '{text}'
            
            Make it:
            1. Professional and workplace-appropriate
            2. Clear and concise
            3. Diplomatic and respectful
            4. Solution-oriented
            5. Constructive rather than negative
            
            Provide only the transformed text, without any additional explanation or quotes."""
            
            # Create a Gemini model instance
            model = genai.GenerativeModel('gemini-2.0-flash')
            
            # Generate content
            response = model.generate_content(prompt)
            
            if response.text:
                return response.text.strip()
            else:
                # Try fallback if response is empty
                if SHOW_DEBUG:
                    console.print("[yellow]Empty response from primary model, trying fallback...[/yellow]")
                return try_fallback_with_module(text)
            
    except Exception as e:
        if SHOW_DEBUG:
            console.print(Panel(f"Error with primary model: {str(e)}", 
                            title="Exception", 
                            border_style="red"))
        return try_fallback_with_module(text)

def try_fallback_with_module(text):
    """Try with a different model using the module"""
    if SHOW_DEBUG:
        console.print("[yellow]Trying fallback model with module...[/yellow]")
    
    try:
        # Use gemini-pro as fallback
        model = genai.GenerativeModel('gemini-pro')
        prompt = f"Transform this text into professional language: '{text}'. Provide only the transformed text without quotes or explanations."
        
        response = model.generate_content(prompt)
        
        if response.text:
            return response.text.strip()
        
        # If we get here, all module attempts have failed
        return get_smart_fallback(text)
    
    except Exception as e:
        if SHOW_DEBUG:
            console.print(Panel(f"Error with fallback model: {str(e)}", 
                          title="Exception", 
                          border_style="red"))
        return get_smart_fallback(text)

def process_user_input(text):
    """Process user input to handle any special characters"""
    # Remove leading colons that might be causing issues
    if text.startswith(':'):
        text = text.lstrip(':').strip()
    return text

def get_smart_fallback(text):
    """Provide smarter fallback responses when API fails"""
    text_lower = text.lower().strip()
    
    # Check for exact matches in our common transformations
    if text_lower in COMMON_TRANSFORMATIONS:
        return random.choice(COMMON_TRANSFORMATIONS[text_lower])
    
    # Check for partial matches in our common transformations
    for key, responses in COMMON_TRANSFORMATIONS.items():
        if key in text_lower or text_lower in key:
            return random.choice(responses)
    
    # Check for question prefixes and informational queries
    for prefix, replacement in INFO_FALLBACKS.items():
        if text_lower.startswith(prefix):
            return f"{replacement} {text[len(prefix):].strip()}."
    
    # Use different fallback transformations based on content patterns
    if any(word in text_lower for word in ["i need", "i want", "give me"]):
        return f"I would like to request {text_lower.replace('i need', '').replace('i want', '').replace('give me', '').strip()}."
    
    if any(word in text_lower for word in ["not good", "bad", "terrible", "awful"]):
        return f"I believe there may be room for improvement regarding {text_lower.replace('not good', '').replace('bad', '').replace('terrible', '').replace('awful', '').strip()}."
    
    if "?" in text:
        return f"I would appreciate your insights on {text}"
    
    # Most generic fallback, but still better than the previous one
    return f"I would like to professionally communicate: {text}"

def main():
    # Clear screen for better UX
    console.clear()
    
    # Show app header
    show_header()
    
    # Get API key
    api_key = os.environ.get("GEMINI_API_KEY")
    
    if not api_key:
        console.print(Panel("No Gemini API key found. Please check your .env file", 
                           title="API Key Missing", 
                           border_style="red"))
        return
    
    console.print(f"[green]✓[/green] API key loaded successfully")
    
    # Main processing loop
    while True:
        console.print("\n[bright_blue]Enter your casual text[/bright_blue] [dim](or 'exit' to quit)[/dim]:")
        casual_text = Prompt.ask("> ", console=console)
        
        if casual_text.lower() in ('exit', 'quit', 'q'):
            console.print("\n[bright_blue]Thank you for using How To Professionally Say! Goodbye! 👋[/bright_blue]")
            break
            
        if casual_text.lower() == 'debug':
            global SHOW_DEBUG
            SHOW_DEBUG = not SHOW_DEBUG
            console.print(f"[yellow]Debug mode: {'ON' if SHOW_DEBUG else 'OFF'}[/yellow]")
            continue
            
        if not casual_text:
            console.print(Panel("No input provided. Please try again.", 
                               title="Empty Input", 
                               border_style="yellow"))
            continue
        
        # Process the input to handle any special characters
        casual_text = process_user_input(casual_text)
        
        # Call the API
        try:
            professional_text = call_gemini_api(casual_text, api_key)
            
            if professional_text:
                format_output(casual_text, professional_text)
            else:
                console.print(Panel("Could not get response from Gemini API.", 
                                title="API Error", 
                                border_style="red"))
                
                # Use smart fallback
                smart_text = get_smart_fallback(casual_text)
                console.print("\n[yellow]Using smart fallback transformation:[/yellow]")
                format_output(casual_text, smart_text)
        except Exception as e:
            if SHOW_DEBUG:
                console.print(Panel(f"Error: {str(e)}\n{traceback.format_exc()}", 
                                title="Exception", 
                                border_style="red"))
            else:
                console.print(Panel("An error occurred while processing your request.", 
                                title="Error", 
                                border_style="red"))
                
            # Use a smart fallback
            smart_text = get_smart_fallback(casual_text)
            console.print("\n[yellow]Using smart fallback transformation:[/yellow]")
            format_output(casual_text, smart_text)
            
        console.print("\nPress Enter for another transformation or type 'exit' to quit.", 
                     style="bright_black italic")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        console.print("\n\n[bright_blue]Program interrupted. Goodbye! 👋[/bright_blue]")