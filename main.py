# app/main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from utils import search_competitors, generate_brand_summary, load_data, find_competitors
from graph_handler import load_graph, save_graph

app = FastAPI()

# Load the graph at startup
G = load_graph()

# Initialize the sentence transformer model for embedding creation
model = SentenceTransformer('all-MiniLM-L6-v2')

# Define a proper Pydantic model
class BrandQuery(BaseModel):
    brand: str

    class Config:
        arbitrary_types_allowed = True  # Allows handling of custom types like PromptTemplate

@app.post("/find-competitors/")
async def get_competitors(query: BrandQuery):
    brand = query.brand

    # Check if competitors exist in the graph
    competitors = find_competitors(brand)
    if competitors:
        return competitors

    try:
        # Search for competitors
        brand_competitors = search_competitors(brand, num_results=1)
        brand_list = brand_competitors[brand] + [brand]

        # Generate brand profiles
        brand_profiles = {b: str(generate_brand_summary(b)) for b in brand_list}  # Ensure string output

        # Encode brand profiles
        brand_embeddings = {b: model.encode(profile) for b, profile in brand_profiles.items()}

        # Load data into the graph (Ensures no duplicate nodes/edges)
        load_data(brand_embeddings, G)

        # Save the updated graph
        save_graph(G)

        # Find and return competitors
        return find_competitors(brand)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    return {"message": "Welcome to the Brand Competitor Finder API"}
