import gradio as gr
from transformers import BlipProcessor, BlipForConditionalGeneration, MarianMTModel, MarianTokenizer
from PIL import Image

# 1. Load the BLIP model for image captioning
caption_model_name = "Salesforce/blip-image-captioning-base"
processor = BlipProcessor.from_pretrained(caption_model_name)
caption_model = BlipForConditionalGeneration.from_pretrained(caption_model_name)

# 2. Load the translation model (English to French)
translation_model_name = "Helsinki-NLP/opus-mt-en-fr"
tokenizer = MarianTokenizer.from_pretrained(translation_model_name)
translation_model = MarianMTModel.from_pretrained(translation_model_name)

# Function 1: Generate a caption for the image
def generate_caption(image):
    inputs = processor(image, return_tensors="pt")
    outputs = caption_model.generate(**inputs)
    caption = processor.decode(outputs[0], skip_special_tokens=True)
    return caption

# Function 2: Translate the caption into French
def translate_caption(caption):
    inputs = tokenizer(caption, return_tensors="pt", padding=True)
    translated = translation_model.generate(**inputs)
    translated_caption = tokenizer.decode(translated[0], skip_special_tokens=True)
    return translated_caption

# Function 3: Combine captioning and translation
def caption_and_translate(image):
    caption = generate_caption(image)  # Generate caption in English
    translated_caption = translate_caption(caption)  # Translate to French
    return caption, translated_caption

# Gradio Interface
with gr.Blocks() as demo:
    gr.Markdown("# AI Image Captioning with Translation")
    gr.Markdown("Upload an image, and the AI will generate a caption in English and translate it into French.")

    with gr.Row():
        with gr.Column():
            image_input = gr.Image(type="pil", label="Upload Image")
        with gr.Column():
            english_output = gr.Textbox(label="Caption (English)")
            french_output = gr.Textbox(label="Caption (French)")

    generate_button = gr.Button("Generate Caption")
    generate_button.click(
        caption_and_translate,
        inputs=[image_input],
        outputs=[english_output, french_output]
    )

# Launch the app
if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
