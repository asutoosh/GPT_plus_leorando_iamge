# YouTube Script & Image Generator


Run the setup file first 
python setup.py

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

