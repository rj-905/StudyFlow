import os
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()

client = genai.Client(api_key=os.environ.get("GOOGLE_API_KEY"))

# Test generation
resp = client.models.generate_content(
    model='gemini-3.5-flash',
    contents='What is 2+2?',
    config=types.GenerateContentConfig(
        temperature=0.0
    )
)
print("Gen:", resp.text)

# Test JSON
resp2 = client.models.generate_content(
    model='gemini-3.5-flash',
    contents='{"test": 123} return this in json.',
    config=types.GenerateContentConfig(
        response_mime_type="application/json"
    )
)
print("JSON:", resp2.text)

# Test Embedding
resp3 = client.models.embed_content(
    model='text-embedding-004',
    contents=['Hello world'],
    config=types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT")
)
print("Embed len:", len(resp3.embeddings[0].values))
