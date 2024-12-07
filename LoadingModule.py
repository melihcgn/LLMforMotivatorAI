from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline, GPT2LMHeadModel, GPT2Tokenizer

# Initialize FastAPI app
app = FastAPI()

# Load model and tokenizer using pipeline
model_path = "./model_files"  # Path to your model files
generator = pipeline("text-generation", 
                     model=GPT2LMHeadModel.from_pretrained(model_path), 
                     tokenizer=GPT2Tokenizer.from_pretrained(model_path))

# Define input structure
class QueryRequest(BaseModel):
    query: str

# Define endpoint
@app.post("/generate/")
async def generate_quote(request: QueryRequest):
    mood = request.query

    # Generate motivational quote
    prompt = f"Motivational quote for someone feeling {mood}:"
    result = generator(prompt, max_length=50, num_return_sequences=1)[0]["generated_text"]

    return {"response": result}
