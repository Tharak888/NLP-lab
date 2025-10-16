from transformers import MarianMTModel, MarianTokenizer
import torch

# Step 1: Define the model name (pretrained model for English to Hindi)
model_name = "Helsinki-NLP/opus-mt-en-hi"

# Step 2: Load tokenizer and model
tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name)

# Step 3: Define translation function
def translate_english_to_hindi(text):
    # Preprocessing
    text = text.strip()

    # Tokenization
    tokens = tokenizer([text], return_tensors="pt", padding=True, truncation=True)

    # Generate translation
    with torch.no_grad():
        translated_tokens = model.generate(**tokens)

    # Decode translated text
    translated_text = tokenizer.decode(translated_tokens[0], skip_special_tokens=True)
    return translated_text

# Step 4: Take user input (Expandable: can connect to file, API, or GUI)
english_sentences = [
    "I love my country.",
    "Machine translation is a part of natural language processing.",
    "How are you?",
    "We are learning Artificial Intelligence."
]

print("🔤 English → Hindi Translation:\n")
for sentence in english_sentences:
    hindi_output = translate_english_to_hindi(sentence)
    print(f"ENGLISH: {sentence}")
    print(f"HINDI:   {hindi_output}\n")
