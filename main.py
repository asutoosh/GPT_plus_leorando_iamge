#!/usr/bin/env python3
"""
python -m streamlit run a.py

Complete YouTube Script & Image Generation Automation
Combines script generation, image prompt creation, and Leonardo AI image generation
Enhanced with batched context approach, retry loops, and detailed logging
"""

import streamlit as st
import openai
import os
import time
import re
import json
import requests
import logging
from pathlib import Path
from tqdm import tqdm
from datetime import datetime

# === ENHANCED LOGGING SETUP ===
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('youtube_automation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# === CONFIGURATION ===
OPENAI_API_KEY = ""   # paste GPT KEY
LEONARDO_API_KEY = ""    #PASTE LEONARDO API KEY
LEONARDO_API_URL = "https://cloud.leonardo.ai/api/rest/v1/generations"
LEONARDO_MODEL_ID = "b2614463-296c-462a-9586-aafdb8f00e36"
LEONARDO_USER_LORA_ID = 65553    #paste your trained leornado model id

MODEL = "gpt-4o"
TRIGGER = "--AUTO"

# Enhanced Configuration
BATCH_SIZE = 5  # Process 5 images per chat context
MAX_RETRIES = 3  # Maximum retry attempts for API calls

# File paths
TOPIC_FILE = "topic.txt"
FIRST_PROMPT_FILE = "Input_prompts/firstprompt.txt"

# Directories
BASE_DIR = Path(".")
INDIVIDUAL_SCRIPTS_DIR = BASE_DIR / "individual_scripts"
IMAGES_DIR = BASE_DIR / "Images"
SCRIPT_IMAGES_DIR = IMAGES_DIR / "Script_Images"
INTRO_IMAGES_DIR = IMAGES_DIR / "intro_Images"

# Create directories
for dir_path in [INDIVIDUAL_SCRIPTS_DIR, IMAGES_DIR, SCRIPT_IMAGES_DIR, INTRO_IMAGES_DIR]:
    dir_path.mkdir(exist_ok=True)


#Create first prompt
def load_and_fill_first_prompt(topic_name):
    """
    Load firstprompt.txt template and insert the dynamic topic name.
    Replace [TOPIC_PLACEHOLDER] with the actual topic.
    """
    try:
        with open(FIRST_PROMPT_FILE, "r", encoding="utf-8") as f:
            template = f.read().strip()
            if "[TOPIC_PLACEHOLDER]" not in template:
                logger.warning("Template missing [TOPIC_PLACEHOLDER] — fallback to old manual replace.")
                # fallback: replace by matching square brackets if needed
                template = re.sub(r"\[.*?\]", f"[{topic_name}]", template, count=1)
            else:
                template = template.replace("[TOPIC_PLACEHOLDER]", topic_name)
            logger.info("First prompt template loaded and filled with topic name.")
            return template
    except FileNotFoundError:
        logger.error(f"File not found: {FIRST_PROMPT_FILE}")
        st.error(f"File not found: {FIRST_PROMPT_FILE}")
        return ""

# === SETUP ===
client = openai.OpenAI(api_key=OPENAI_API_KEY)

LEONARDO_HEADERS = {
    "Authorization": f"Bearer {LEONARDO_API_KEY}",
    "Content-Type": "application/json"
}

# === EMBEDDED IMAGE PROMPT TEMPLATE ===
IMAGE_PROMPT_TEMPLATE = """NOTE: Your reply should not exceed 200 words. You are tasked with creating multiple image prompts based on a specific segment of a script. Each image should visually represent the ideas, themes, or concepts from the segment in a way that depicts a scene in the life of a medieval character mentioned in the script title. The images should align with the tone and style of the script (e.g. medieval setting, folk style art, scene in the life of). Use the art style below for all the prompts.

Medieval folk art style, sombre and depressing mood, usage of contrasting colors in the figures to stand out from the background, medieval painting style. Detailed description of the characters or figures doing the profession mentioned in the video title, what they are doing, objects in the image, background setting, color usage, mood of the characters or figures in the image, actions being performed by the characters.

Input from User:
Video title: "Why It Sucked to Be a Medieval Peasant"
Segment Paragraph: {para}
Number of Image Prompts: 1

Output Format (Image Prompts):
NOTE: the output word length should not exceed 200 words.

Image Prompt 1:
Description: Describe the visual elements, mood, and style of the image.
Style: Specify one of the predefined styles above or suggest an alternative style that fits the concept.
Key Details: Highlight any specific objects, characters, or abstract elements that should be included to convey the idea segment concept.

[Repeat for the remaining image prompts.]

Guidelines for Generating Image Prompts
Match Tone and Theme: Ensure each image aligns with the medieval setting, scene in the life of the character in video title, and the feeling depicted in the script segment.
Styles: Use the same image style for all the prompts you'll be generating.
Maintain Variety: Avoid redundancy across image prompts; ensure each one offers a unique perspective or interpretation of the segment.
Balance Simplicity and Complexity: Some images may benefit from minimalistic designs, while others might require intricate details to fully convey the idea.
Aspect Ratio: Each image to be in 16:9 aspect ratio."""

# === ENHANCED IMAGE PROCESSOR CLASS ===
class EnhancedImageProcessor:
    def __init__(self, api_key, batch_size=5, max_retries=3):
        """Initialize the enhanced image processor with batching capability"""
        self.client = openai.OpenAI(api_key=api_key)
        self.batch_size = batch_size
        self.max_retries = max_retries
        self.current_batch = []
        self.batch_count = 0
        self.chat_contexts = []
        
        logger.info(f"Enhanced ImageProcessor initialized with batch_size={batch_size}, max_retries={max_retries}")

    def retry_api_call(self, func, *args, **kwargs):
        """Retry mechanism for API calls with exponential backoff"""
        for attempt in range(self.max_retries):
            try:
                logger.info(f"API call attempt {attempt + 1}/{self.max_retries}")
                result = func(*args, **kwargs)
                logger.info("API call successful")
                return result
            
            except Exception as e:
                wait_time = (2 ** attempt)  # Exponential backoff: 1s, 2s, 4s
                logger.warning(f"API call failed (attempt {attempt + 1}/{self.max_retries}): {str(e)}")
                
                if attempt < self.max_retries - 1:
                    logger.info(f"Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                else:
                    logger.error(f"All {self.max_retries} attempts failed for API call")
                    
        return None

    def create_new_chat_context(self):
        """Create a new chat context for the next batch"""
        self.batch_count += 1
        new_context = {
            'batch_id': self.batch_count,
            'messages': [],
            'processed_images': 0,
            'created_at': datetime.now().isoformat()
        }
        self.chat_contexts.append(new_context)
        logger.info(f"Created new chat context - Batch #{self.batch_count}")
        st.info(f"🔄 Starting new batch #{self.batch_count} (5 images per batch)")
        return new_context

    def get_current_context(self):
        """Get the current active chat context"""
        if not self.chat_contexts or self.chat_contexts[-1]['processed_images'] >= self.batch_size:
            return self.create_new_chat_context()
        return self.chat_contexts[-1]

    def generate_image_prompt_with_batching(self, paragraph):
        """Generate image prompt with batched context approach"""
        try:
            logger.info(f"Generating image prompt for paragraph (length: {len(paragraph)} chars)")
            
            # Get current chat context
            context = self.get_current_context()
            
            # Create the message with the embedded prompt
            prompt_with_paragraph = IMAGE_PROMPT_TEMPLATE.format(para=paragraph)
            
            message = {
                "role": "user",
                "content": prompt_with_paragraph
            }
            
            # Add message to current context
            context['messages'].append(message)
            
            # Make API call with retry mechanism
            def make_chat_completion():
                return self.client.chat.completions.create(
                    model=MODEL,
                    messages=[{
                        "role": "system", 
                        "content": "You are an expert at creating detailed image prompts for medieval-themed artwork."
                    }] + context['messages'],
                    max_tokens=300
                )
            
            response = self.retry_api_call(make_chat_completion)
            
            if response is None:
                logger.error(f"Failed to generate image prompt after all retries")
                return ""
            
            # Add assistant response to context
            assistant_message = {
                "role": "assistant",
                "content": response.choices[0].message.content
            }
            context['messages'].append(assistant_message)
            context['processed_images'] += 1
            
            logger.info(f"Successfully generated image prompt in batch #{context['batch_id']}")
            logger.info(f"Batch progress: {context['processed_images']}/{self.batch_size}")
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"Unexpected error generating image prompt: {str(e)}")
            return ""

# Initialize global image processor
image_processor = EnhancedImageProcessor(OPENAI_API_KEY, BATCH_SIZE, MAX_RETRIES)

# === UTILITY FUNCTIONS ===
def load_file(filepath):
    """Load content from a text file"""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read().strip()
            logger.info(f"Loaded file: {filepath} ({len(content)} characters)")
            return content
    except FileNotFoundError:
        logger.error(f"File not found: {filepath}")
        st.error(f"File not found: {filepath}")
        return ""

def save_file(filepath, content):
    """Save content to a text file"""
    try:
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)
        logger.info(f"Saved file: {filepath} ({len(content)} characters)")
    except Exception as e:
        logger.error(f"Error saving file {filepath}: {str(e)}")

def get_topic_name():
    """Get topic name from topic.txt"""
    return load_file(TOPIC_FILE)

def clean_filename(name, max_length=50):
    """Clean and truncate filename"""
    name = re.sub(r'[^\w\s-]', '', name.strip())
    name = re.sub(r'\s+', ' ', name)
    return name[:max_length].strip()

def count_words(text):
    """Count words in text"""
    return len(text.split())

def split_into_paragraphs(text):
    """Split text into paragraphs"""
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
    logger.info(f"Split text into {len(paragraphs)} paragraphs")
    return paragraphs

# === ENHANCED GPT FUNCTIONS ===
def send_gpt_message_with_retry(messages, content):
    """Send message to GPT with retry mechanism"""
    messages.append({"role": "user", "content": content})
    
    def make_api_call():
        return client.chat.completions.create(
            model=MODEL,
            messages=messages
        )
    
    # Use retry mechanism
    for attempt in range(MAX_RETRIES):
        try:
            logger.info(f"GPT API call attempt {attempt + 1}/{MAX_RETRIES}")
            response = make_api_call()
            reply = response.choices[0].message.content
            messages.append({"role": "assistant", "content": reply})
            logger.info("GPT API call successful")
            return reply
        except Exception as e:
            wait_time = (2 ** attempt)
            logger.warning(f"GPT API call failed (attempt {attempt + 1}/{MAX_RETRIES}): {str(e)}")
            
            if attempt < MAX_RETRIES - 1:
                logger.info(f"Retrying GPT call in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                logger.error(f"All {MAX_RETRIES} GPT attempts failed")
                st.error(f"GPT API Error after {MAX_RETRIES} attempts: {e}")
                
    return ""

def generate_script_segment(messages, segment_num):
    """Generate a script segment with retry mechanism"""
    prompt = f"keep 200 words in one paragraphs and skip small paras. dont give small paragraphs like containing 1 -2 lines , Remember it and expand the part <{segment_num}> in 1000 words."
    logger.info(f"Generating script segment {segment_num}")
    return send_gpt_message_with_retry(messages, prompt)

def generate_intro_prompts_with_retry(intro_text):
    """Generate 6 image prompts for intro with retry mechanism"""
    prompt = f"Generate 6 image prompts for this intro text that convey a flowing style. Each prompt should be separate and numbered. Intro text: {intro_text}"
    
    messages = [
        {"role": "system", "content": "You are an expert at creating image prompts that flow together visually."},
        {"role": "user", "content": prompt}
    ]
    
    def make_api_call():
        return client.chat.completions.create(
            model=MODEL,
            messages=messages
        )
    
    for attempt in range(MAX_RETRIES):
        try:
            logger.info(f"Intro prompts API call attempt {attempt + 1}/{MAX_RETRIES}")
            response = make_api_call()
            logger.info("Intro prompts API call successful")
            return response.choices[0].message.content
        except Exception as e:
            wait_time = (2 ** attempt)
            logger.warning(f"Intro prompts API call failed (attempt {attempt + 1}/{MAX_RETRIES}): {str(e)}")
            
            if attempt < MAX_RETRIES - 1:
                logger.info(f"Retrying intro prompts call in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                logger.error(f"All {MAX_RETRIES} intro prompts attempts failed")
                st.error(f"Error generating intro prompts after {MAX_RETRIES} attempts: {e}")
                
    return ""

# === LEONARDO AI FUNCTIONS ===
def sanitize_prompt_for_leonardo(prompt):
    """Clean and prepare prompt for Leonardo AI"""
    # Remove markdown formatting
    prompt = re.sub(r'\*\*.*?\*\*', '', prompt)
    prompt = re.sub(r'Image Prompt \d+:', '', prompt)
    prompt = re.sub(r'Description:', '', prompt)
    prompt = re.sub(r'Style:.*', '', prompt, flags=re.DOTALL)
    
    # Clean up
    prompt = prompt.strip()
    return prompt[:1800]  # Leonardo has character limits

def generate_leonardo_image_with_retry(prompt, filename, folder_path):
    """Generate image using Leonardo AI with retry mechanism"""
    if not prompt.strip():
        logger.warning(f"Empty prompt for {filename}, skipping...")
        st.warning(f"Empty prompt for {filename}, skipping...")
        return False

    clean_prompt = sanitize_prompt_for_leonardo(prompt)
    logger.info(f"Generating Leonardo image: {filename}")
    
    payload = {
        "modelId": LEONARDO_MODEL_ID,
        "prompt": clean_prompt,
        "width": 1024,
        "height": 576,
        "num_images": 1,
        "contrast": 3.5,
        "enhancePrompt": True,
        "userElements": [
            {
                "userLoraId": LEONARDO_USER_LORA_ID,
                "weight": 0.8
            }
        ]
    }

    for attempt in range(MAX_RETRIES):
        try:
            logger.info(f"Leonardo API call attempt {attempt + 1}/{MAX_RETRIES} for {filename}")
            
            # Start generation
            response = requests.post(LEONARDO_API_URL, headers=LEONARDO_HEADERS, json=payload)
            response.raise_for_status()
            data = response.json()
            
            generation_id = data.get('sdGenerationJob', {}).get('generationId')
            if not generation_id:
                logger.error(f"No generation ID returned for {filename}")
                st.error(f"No generation ID returned for {filename}")
                continue

            st.info(f"Generating image: {filename} (ID: {generation_id}) - Attempt {attempt + 1}")

            # Poll for completion
            status_url = f"{LEONARDO_API_URL}/{generation_id}"
            for poll_attempt in range(30):  # 60 seconds max wait
                time.sleep(2)
                status_resp = requests.get(status_url, headers=LEONARDO_HEADERS)
                status_resp.raise_for_status()
                status_data = status_resp.json()
                
                images = status_data.get("generations_by_pk", {}).get("generated_images", [])
                if images:
                    img_url = images[0]['url']
                    img_data = requests.get(img_url).content
                    
                    # Save image
                    filepath = folder_path / filename
                    with open(filepath, "wb") as f:
                        f.write(img_data)
                    
                    logger.info(f"Successfully saved image: {filepath}")
                    st.success(f"✅ Saved image: {filepath}")
                    return True
                    
            logger.warning(f"Timeout waiting for image: {filename} (attempt {attempt + 1})")
            
        except Exception as e:
            wait_time = (2 ** attempt)
            logger.warning(f"Leonardo API error for {filename} (attempt {attempt + 1}/{MAX_RETRIES}): {str(e)}")
            
            if attempt < MAX_RETRIES - 1:
                logger.info(f"Retrying Leonardo call in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                logger.error(f"All {MAX_RETRIES} Leonardo attempts failed for {filename}")
                st.error(f"Error generating image {filename} after {MAX_RETRIES} attempts: {e}")
                
    return False

# === MAIN STREAMLIT APP ===
def main():
    st.title("🎬 Enhanced YouTube Video Script & Image Generator")
    st.markdown("*With Batched Processing, Retry Loops & Advanced Logging*")
    st.markdown("---")

    # Display logging info
    st.sidebar.header("📊 Processing Info")
    st.sidebar.info(f"Batch Size: {BATCH_SIZE} images/context")
    st.sidebar.info(f"Max Retries: {MAX_RETRIES} attempts")
    st.sidebar.info(f"Active Batches: {len(image_processor.chat_contexts)}")

    # Load configuration
    topic_name = get_topic_name()
    if not topic_name:
        st.error("Please add a topic name to topic.txt file")
        return

    st.info(f"**Topic:** {topic_name}")

    # Load first prompt
    # first_prompt = load_file(FIRST_PROMPT_FILE)
    first_prompt = load_and_fill_first_prompt(topic_name)
    
    if not first_prompt:
        st.error("Please ensure firstprompt.txt exists in Input_prompts folder")
        return

    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "system", "content": "You are an expert script writer for educational YouTube videos. Your style is consistent, calm, and detailed."},
            {"role": "user", "content": first_prompt}
        ]
        
        # Get initial response
        with st.spinner("Getting initial response from GPT..."):
            response = client.chat.completions.create(
                model=MODEL,
                messages=st.session_state.messages
            )
            reply = response.choices[0].message.content
            st.session_state.messages.append({"role": "assistant", "content": reply})
            logger.info("Initial GPT response received and saved")
        
        st.session_state.auto_mode = False
        st.session_state.current_segment = 1
        st.session_state.image_counter = 0
        st.session_state.scripts_generated = False
        st.session_state.intro_generated = False

    # Display chat history
    st.subheader("💬 Chat with GPT")
    for msg in st.session_state.messages[1:]:  # Skip system message
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Manual chat or automation trigger
    if not st.session_state.get("auto_mode", False):
        user_input = st.chat_input("Chat with GPT or type --AUTO to start automation")
        
        if user_input:
            if user_input.strip() == TRIGGER:
                st.session_state.auto_mode = True
                logger.info("Automation mode activated")
                st.rerun()
            else:
                # Manual chat
                with st.spinner("Getting response..."):
                    response = send_gpt_message_with_retry(st.session_state.messages, user_input)
                st.rerun()

    # Automation mode
    if st.session_state.get("auto_mode", False):
        st.markdown("---")
        st.header("🤖 Enhanced Automation Mode")
        
        # Generate 15 script segments
        if not st.session_state.get("scripts_generated", False):
            st.subheader("📝 Generating Script Segments (1-15)")
            
            progress_bar = st.progress(0)
            for i in range(1, 16):
                progress_bar.progress(i / 15)
                
                with st.spinner(f"Generating segment {i}/15..."):
                    script_content = generate_script_segment(st.session_state.messages, i)
                    
                    if script_content:
                        # Save individual script
                        script_file = INDIVIDUAL_SCRIPTS_DIR / f"{i}.txt"
                        save_file(script_file, script_content)
                        st.success(f"✅ Saved: {script_file}")
                        
                        # Process paragraphs for images (only first 40 paragraphs total)
                        if st.session_state.image_counter < 40:
                            paragraphs = split_into_paragraphs(script_content)
                            
                            for j, paragraph in enumerate(paragraphs):
                                if st.session_state.image_counter >= 40:
                                    break
                                    
                                if count_words(paragraph) >= 50:  # Skip short paragraphs
                                    st.session_state.image_counter += 1
                                    
                                    # Show batch info
                                    current_context = image_processor.get_current_context()
                                    batch_progress = current_context['processed_images']
                                    st.info(f"🖼️ Processing image {st.session_state.image_counter}/40 (Batch #{current_context['batch_id']}, Image {batch_progress + 1}/5)")
                                    
                                    with st.spinner(f"Generating image prompt {st.session_state.image_counter}/40..."):
                                        # Generate image prompt using enhanced processor
                                        img_prompt = image_processor.generate_image_prompt_with_batching(paragraph)
                                        
                                        if img_prompt:
                                            # Create filename
                                            first_words = clean_filename(' '.join(paragraph.split()[:10]))
                                            filename = f"{st.session_state.image_counter}_{first_words}.png"
                                            
                                            # Generate image with retry
                                            generate_leonardo_image_with_retry(img_prompt, filename, SCRIPT_IMAGES_DIR)
                    
                    time.sleep(1)  # Rate limiting
            
            st.session_state.scripts_generated = True
            st.success("🎉 All 15 script segments generated!")
            logger.info("All script segments generation completed")


        if st.session_state.get("scripts_generated", False) and not st.session_state.get("intro_generated", False):
            st.subheader("🎬 Generating Intro")
    
            intro_prompt = "Now generate a quick hook intro for the TOPIC [ "+ (get_topic_name()) +" ]in 100 words and also add some interactive lines with viewers like comment and like and sleep good as this is a video to give better sleep . and if possible tell users to comment their location and what time is it . add unique style"
            

    
            # IMPORTANT: Fresh short message list
            intro_messages = [
                {"role": "system", "content": "You are an expert script writer for educational YouTube videos."},
                {"role": "user", "content": intro_prompt}
                ]
        # #  Generate intro
        # if st.session_state.get("scripts_generated", False) and not st.session_state.get("intro_generated", False):
        #     st.subheader("🎬 Generating Intro")
            
        #     intro_prompt = "Now generate a quick hook intro for the script in 100 words and also add some interactive lines with viewers like comment and like and sleep good as this is a video to give better sleep . and if possible tell users to comment their location and what time is it . add unique style"

            with st.spinner("Generating intro..."):
                intro_content = send_gpt_message_with_retry(intro_messages, intro_prompt)           


            # with st.spinner("Generating intro..."):
            #     intro_content = send_gpt_message_with_retry(st.session_state.messages, intro_prompt)
                
                if intro_content:
                    # Save intro
                    intro_file = INDIVIDUAL_SCRIPTS_DIR / "intro.txt"
                    save_file(intro_file, intro_content)
                    st.success(f"✅ Saved intro: {intro_file}")
                    
                    # Generate 6 intro images
                    st.subheader("🖼️ Generating Intro Images (1-6)")
                    intro_prompts = generate_intro_prompts_with_retry(intro_content)
                    
                    if intro_prompts:
                        # Split prompts (assuming they're numbered)
                        prompt_sections = re.split(r'\d+\.', intro_prompts)[1:]  # Skip first empty split
                        
                        for k, prompt_section in enumerate(prompt_sections[:6], 1):
                            with st.spinner(f"Generating intro image {k}/6..."):
                                filename = f"{k}_intro_image.png"
                                generate_leonardo_image_with_retry(prompt_section.strip(), filename, INTRO_IMAGES_DIR)
            
            st.session_state.intro_generated = True
            logger.info("Intro generation completed")

        # Combine final script
        if st.session_state.get("intro_generated", False):
            st.subheader("📄 Creating Final Combined Script")
            
            with st.spinner("Combining intro and script segments..."):
                # Load intro
                intro_content = load_file(INDIVIDUAL_SCRIPTS_DIR / "intro.txt")
                
                # Load all segments
                combined_content = intro_content + "\n\n"
                
                for i in range(1, 16):
                    segment_file = INDIVIDUAL_SCRIPTS_DIR / f"{i}.txt"
                    segment_content = load_file(segment_file)
                    combined_content += segment_content + "\n\n"
                
                # Save final script
                clean_topic = clean_filename(topic_name)
                final_script_file = BASE_DIR / f"{clean_topic}(generated_script).txt"
                save_file(final_script_file, combined_content)
                
                st.success(f"🎉 Final script saved: {final_script_file}")
                logger.info(f"Final combined script saved: {final_script_file}")
                
                # Show enhanced summary
                st.markdown("---")
                st.header("📊 Enhanced Generation Summary")
                st.success("✅ All tasks completed successfully!")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.info(f"📝 Scripts: 15 + 1 intro")
                    st.info(f"🖼️ Script images: {min(st.session_state.image_counter, 40)}")
                    st.info(f"🎬 Intro images: 6")
                
                with col2:
                    st.info(f"🔄 Chat contexts used: {len(image_processor.chat_contexts)}")
                    st.info(f"⚡ Images per batch: {BATCH_SIZE}")
                    st.info(f"🔁 Max retries: {MAX_RETRIES}")
                
                with col3:
                    st.info(f"📁 Scripts: {INDIVIDUAL_SCRIPTS_DIR}")
                    st.info(f"🖼️ Script images: {SCRIPT_IMAGES_DIR}")
                    st.info(f"🎬 Intro images: {INTRO_IMAGES_DIR}")
                
                # Log file info
                st.markdown("---")
                st.info("📋 **Detailed logs saved to:** `youtube_automation.log`")

if __name__ == "__main__":
    main()
