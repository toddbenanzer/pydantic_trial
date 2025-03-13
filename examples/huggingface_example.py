from llm_interface import create_provider

# Create HuggingFace provider
provider = create_provider("huggingface", api_token="YOUR_API_TOKEN")

# Generate text
response = await provider.generate_text(
    prompt="Explain the theory of relativity in simple terms.",
    model="mistralai/Mistral-7B-Instruct-v0.2"
)
print(response.content)

# Generate code
code_response = await provider.generate_code(
    prompt="Write a Python function to find the n-th Fibonacci number using dynamic programming.",
    model="codellama/CodeLlama-7b-hf"
)
print(code_response.content)

# Text-to-speech
tts_response = await provider.text_to_speech(
    text="Hello world! This is text-to-speech using HuggingFace models.",
    model="espnet/kan-bayashi_ljspeech_vits"
)
# Save audio to file
with open("huggingface_tts.wav", "wb") as f:
    f.write(tts_response.audio_data)