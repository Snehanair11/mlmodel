from transformers import pipeline

print("Pre-downloading emotion model...")
pipeline(
    "text-classification",
    model="j-hartmann/emotion-english-distilroberta-base",
    top_k=3
)
print("Model cached.")
