from transformers import pipeline

# Load the text generation model
generator = pipeline("text-generation", model="microsoft/DialoGPT-medium")

# Generate a response
response = generator("Olá, como você está?", max_length=50, num_return_sequences=1)
print(response)


