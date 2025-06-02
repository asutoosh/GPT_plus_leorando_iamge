#!/usr/bin/env python3
"""
Setup script to create required directories and example files
Run this first to set up the project structure
"""

import os
from pathlib import Path

def create_directory_structure():
    """Create the required directory structure"""
    
    # Base directories
    dirs_to_create = [
        "Input_prompts",
        "individual_scripts", 
        "Images/Script_Images",
        "Images/intro_Images"
    ]
    
    for dir_path in dirs_to_create:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        print(f"✅ Created directory: {dir_path}")

def create_example_files():
    """Create example configuration files"""
    
    # Example topic.txt
    topic_content = "3 Hours of Real Life Cheat Codes to Fall Asleep To"
    with open("topic.txt", "w", encoding="utf-8") as f:
        f.write(topic_content)
    print("✅ Created topic.txt")
    
    # Example firstprompt.txt
    firstprompt_content = """Create a detailed structure for a long-form YouTube video script titled [ 3 Hours of Real Life Cheat Codes to Fall Asleep To ]. The script will be divided into [ 15 ] segments, each focusing on a specific idea, paradox, theory, or concept related to the title/topic. The first segment should briefly introduce the concept in an engaging way, using storytelling, analogies, or thought-provoking questions.

Each segment should contain 1500 words total. Each segment should follow this format:

Segment Title:
Provide a concise, intriguing title for the segment that reflects the concept being explored.

Catchy Hook or Relatable Scenario:
One or two sentences that set up the concept with a real-world feeling.

Definition in Simple Language:
A casual, relatable definition of the concept.

Detailed Breakdown / Explanation:
Expanding how the concept works, why it matters, what it actually does. Use analogies and examples.

Examples:
At least two examples - one simple and relatable, one more complex.

Reasons Why This Concept Matters:
Explains the significance and relevance to daily life.

Problems or Paradoxes:
Shows interesting contradictions or thought-provoking aspects.

How to Apply or Recognize It:
Practical applications or signs to look for.

Final Thoughts:
Reflective conclusion that ties back to the overall theme.

Additional Guidelines:
- Ensure the overall tone is calm, reflective, and slightly mysterious, perfect for a video designed to help viewers fall asleep or unwind
- Each segment should be self-contained but contribute to the overarching theme
- Use storytelling, logical reasoning, and philosophical inquiry
- Include occasional pauses or breaks (e.g., 'Take a moment to reflect on this...')
- Make content immersive and intellectually stimulating but relaxing"""

    with open("Input_prompts/firstprompt.txt", "w", encoding="utf-8") as f:
        f.write(firstprompt_content)
    print("✅ Created Input_prompts/firstprompt.txt")
    
    # Example imageprompt.txt
    imageprompt_content = """You are an expert at creating detailed image prompts for AI image generation. Your task is to analyze a paragraph from a script and create a single, detailed image prompt that visually represents the key concepts, mood, and themes.

Instructions:
1. Read the provided paragraph carefully
2. Identify the main concepts, themes, and mood
3. Create ONE detailed image prompt (maximum 200 words)
4. Focus on visual elements that would work well for a calm, sleep-inducing video
5. Use descriptive language that captures both concrete and abstract elements

Style Guidelines:
- Calm, dreamy, and slightly mysterious aesthetic
- Soft lighting and muted colors
- Elements that evoke contemplation and tranquility
- Abstract or symbolic representations are encouraged
- Avoid overly busy or stimulating visuals

Output Format:
Provide only the image prompt description without labels or formatting. Make it detailed enough for an AI to generate a compelling visual that matches the paragraph's content and mood.

Example tone: "A serene landscape at twilight with floating geometric shapes representing different life choices, soft purple and blue hues, minimal lighting, contemplative atmosphere, abstract figures in the distance suggesting human connection..."

Now analyze this paragraph and create an image prompt:"""

    with open("Input_prompts/imageprompt.txt", "w", encoding="utf-8") as f:
        f.write(imageprompt_content)
    print("✅ Created Input_prompts/imageprompt.txt")
    
    # Create requirements.txt
    requirements_content = """streamlit>=1.28.0
openai>=1.3.0
requests>=2.31.0
tqdm>=4.66.0
pathlib2>=2.3.7
"""
    
    with open("requirements.txt", "w", encoding="utf-8") as f:
        f.write(requirements_content)
    print("✅ Created requirements.txt")

def create_readme():
    """Create README with instructions"""
    
    readme_content = """# YouTube Script & Image Generator

Automated system for generating long-form YouTube video scripts with AI-generated images.

## Setup Instructions

1. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure API Keys:**
   - Edit the main script and add your OpenAI API key
   - Edit the main script and add your Leonardo AI API key

3. **Prepare Input Files:**
   - Edit `topic.txt` with your video topic
   - Customize `Input_prompts/firstprompt.txt` for your script style
   - Customize `Input_prompts/imageprompt.txt` for your image style

4. **Run the Application:**
   ```bash
   streamlit run main.py
   ```

## How It Works

1. **Manual Chat Phase:** First, chat with GPT to train it on your desired output style
2. **Automation Trigger:** Type `--AUTO` to start the automated generation
3. **Script Generation:** Generates 15 script segments (1500 words each)
4. **Image Generation:** Creates images for paragraphs with 50+ words (max 40 images)
5. **Intro Creation:** Generates intro text and 6 intro images
6. **Final Assembly:** Combines everything into a final script file

## Directory Structure

```
📁 PROJECT_FOLDER/
├── main.py
├── topic.txt
├── requirements.txt
├── <topic_name>(generated_script).txt
├── 📁 Input_prompts/
│   ├── firstprompt.txt
│   └── imageprompt.txt
├── 📁 individual_scripts/
│   ├── intro.txt
│   ├── 1.txt
│   ├── 2.txt
│   └── ... (up to 15.txt)
└── 📁 Images/
    ├── 📁 Script_Images/
    │   ├── 1_first_ten_words.png
    │   └── ... (up to 40 images)
    └── 📁 intro_Images/
        ├── 1_intro_image.png
        └── ... (6 intro images)
```

## Features

- ✅ Interactive GPT training phase
- ✅ Automated script generation (15 segments)
- ✅ Smart paragraph filtering (50+ words)
- ✅ Leonardo AI image generation
- ✅ Automatic intro generation
- ✅ Progress tracking and error handling
- ✅ File organization and naming

## Notes

- The system generates exactly 40 images for the main script content
- Images are only generated for paragraphs with 50+ words
- All conversations happen in the same GPT thread to maintain consistency
- Leonardo AI settings are optimized for calm, contemplative imagery
"""

    with open("README.md", "w", encoding="utf-8") as f:
        f.write(readme_content)
    print("✅ Created README.md")

if __name__ == "__main__":
    print("🚀 Setting up YouTube Script & Image Generator...")
    print("=" * 50)
    
    create_directory_structure()
    print()
    create_example_files()
    print()
    create_readme()
    
    print("=" * 50)
    print("✅ Setup complete!")
    print("\nNext steps:")
    print("1. Install dependencies: pip install -r requirements.txt")
    print("2. Edit API keys in main.py")
    print("3. Customize topic.txt and prompt files")
    print("4. Run: streamlit run main.py")
