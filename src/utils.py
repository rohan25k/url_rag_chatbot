from bs4 import BeautifulSoup
import re

def clean_webpage_content(content):
    """
    Clean up the webpage content by removing extra whitespace, scripts, styles, and other unwanted elements.
    
    Args:
        content (str): The raw content to clean
        
    Returns:
        str: The cleaned content
    """
    # Use BeautifulSoup to parse the HTML content
    soup = BeautifulSoup(content, 'html.parser')
    
    # Remove script and style elements
    for script_or_style in soup(['script', 'style', 'header', 'footer', 'nav']):
        script_or_style.decompose()
    
    # Extract text from the remaining HTML
    text = soup.get_text()
    
    # Remove extra whitespace and normalize line breaks
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\n+', '\n', text)
    
    # Remove leading/trailing whitespace
    text = text.strip()
    
    return text