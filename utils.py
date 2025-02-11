from googlesearch import search
from urllib.parse import urlparse
import requests
from bs4 import BeautifulSoup
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq

import json
import re
import ast
import spacy
import re
import random

import os
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file

api_key = os.getenv("GROQ_API_KEY")

# Load spaCy's English NER model
nlp = spacy.load("en_core_web_sm")

llm = ChatGroq(
    model="llama3-70b-8192",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2
)


# Function to fetch and parse content from a URL
def fetch_content(url):
    try:
        response = requests.get(url)
        response.raise_for_status()  # Check for HTTP errors
        soup = BeautifulSoup(response.content, 'html.parser')
        # Extract text content; adjust as needed
        return soup.get_text(separator=' ', strip=True)
    except requests.RequestException as e:
        print(f"Error fetching {url}: {e}")
        return None



def brand_search(query,num_results):
    # Perform the search with a higher limit to ensure 10 valid entries
    search_results = search(query, pause=2)

    # Sets to track unique main websites & URLs
    unique_domains = set()
    unique_urls = set()

    # List to store content from each page
    content_list = []

    # Fetch and store content from each unique URL
    for url in search_results:
        main_domain = urlparse(url).netloc  # Extract the main website (domain)

        if main_domain not in unique_domains:  # Ensure domain is unique
            try:
                content = fetch_content(url)
                if len(content)>100:
                    content_list.append(content)
                    unique_domains.add(main_domain)  # Store main domain
                    unique_urls.add(url)  # Store full URL

                # Stop once we have 10 unique results
                if len(content_list) >= num_results:
                    break
            except:
                continue

    return content_list  # Return the list with 10 unique results


from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
# from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq





def identify_competitors(brand,brand_information):
  # Define a simple prompt for the agent
  template = """
  You are a brand analyst. You are given extracted information from from across the web regarding a given brand
  and it's competitors.

  Below are the details :-
  brand: {brand_name}
  extracted information : {extracted_info}

  Analyze the extracted information and then identify all the competitors brands for the given brand.

  Your output should just be in json format wherein  key should be the traget brand and
  value should be the list of global competitors brand names.  
  

  """

  # Set up the prompt and LLM chain
  prompt = PromptTemplate(template=template, input_variables=["brand_name","extracted_info"])
  chain = LLMChain(prompt=prompt, llm=llm)
  response = chain.run(brand_name=brand,extracted_info=brand_information)
  return response

def extract_brand_summary(brand,summary):
# Define a simple prompt for the agent
  template = """
  You are a brand analyst and researcher. You are given list of crucial information regarding the given brand

  Below are the details :-
  brand: {brand_name}
  crucial information : {summarized_list}

  Your task is to carefully analyze the extracted information from the aspect of target brand 
  Extract all the key financial figures,list of countries in which the brand sells it's products, and type of products/services the brands offers.
  Give an output that is concise and has only relevant information.

  Your output should just be a crisp paragraph based summary based on above guidelines.Do not give any other text in output.

  """

  # Set up the prompt and LLM chain
  prompt = PromptTemplate(template=template, input_variables=["brand_name","summarized_list"])
  chain = LLMChain(prompt=prompt, llm=llm)
  response = chain.run(brand_name=brand,summarized_list=summary)

  return response





def extract_json_from_text(llm_output):
    # Regular expression to find JSON content (matches `{...}`)
    json_match = re.search(r"\{.*\}", llm_output, re.DOTALL)

    if json_match:
        json_string = json_match.group(0)  # Extract the matched JSON part
        try:
            data = json.loads(json_string)  # Convert JSON string to dictionary
            return data  # Return clean dictionary
        except json.JSONDecodeError:
            return ast.literal_eval(json_string)
    else:
        return "Error: No valid JSON found in the LLM output."
    
    

def extract_brands(text):
    """Extracts brand/organization names from text using NER."""
    doc = nlp(text)
    return [ent.text for ent in doc.ents if ent.label_ == "ORG"]  # Extract ORG entities

def extract_countries(text):
    """Extracts country names from text using NER."""
    doc = nlp(text)
    return [ent.text for ent in doc.ents if ent.label_ == "GPE"]

def count_countries(text):
    """Counts occurrences of brand names (NER) and numeric figures in a text."""

    # Extract brands using NER
    brands = extract_brands(text)

    # Count unique brand mentions
    brand_count = len(set(brands))

    # Count unique country mentions
    countries = extract_countries(text)

    country_count = len(set(countries))

    # Count numeric figures (money, percentages, large numbers)
    number_count = len(re.findall(r"\d+[\.,]?\d*\s?[BMK]?", text))

    return brand_count  # Higher score = higher priority

def prioritized_shuffle(text_list):
    """Sorts and shuffles texts based on detected brand & number occurrences."""

    # Compute scores for each text
    scored_texts = [(text, count_countries(text)) for text in text_list]

    # Sort texts based on score (higher score = higher priority)
    scored_texts.sort(key=lambda x: x[1], reverse=True)

    # Extract sorted texts
    sorted_texts = [text for text, _ in scored_texts]

    return sorted_texts



def count_brands_and_numbers(text):
    """Counts occurrences of brand names (NER) and numeric figures in a text."""

    # Extract brands using NER
    brands = extract_brands(text)

    # Count unique brand mentions
    brand_count = len(set(brands))

    # Count unique country mentions
    countries = extract_countries(text)

    country_count = len(set(countries))

    # Count numeric figures (money, percentages, large numbers)
    number_count = len(re.findall(r"\d+[\.,]?\d*\s?[BMK]?", text))

    return brand_count+brand_count+ number_count # Higher score = higher priority

def prioritized_shuffle_v1(text_list):
    """Sorts and shuffles texts based on detected brand & number occurrences."""

    # Compute scores for each text
    scored_texts = [(text, count_brands_and_numbers(text)) for text in text_list]

    # Sort texts based on score (higher score = higher priority)
    scored_texts.sort(key=lambda x: x[1], reverse=True)

    # Extract sorted texts
    sorted_texts = [text for text, _ in scored_texts]

    return sorted_texts


def split_text_by_sentence(text, max_size=500):
    """Splits text into chunks of approximately `max_size` characters, ensuring splits happen at full stops (.)"""

    sentences = text.split(". ")  # Split by full stops
    chunks = []
    current_chunk = ""

    for sentence in sentences:
        if len(current_chunk) + len(sentence) + 1 <= max_size:  # +1 for the period
            current_chunk += sentence + ". "  # Append sentence
        else:
            chunks.append(current_chunk.strip())  # Store completed chunk
            current_chunk = sentence + ". "  # Start new chunk

    if current_chunk:  # Add last chunk if non-empty
        chunks.append(current_chunk.strip())

    return chunks


def re_rank_chunks(chunk_list,criteria):
  re_ranked_chunk_list=[]
  for chunk in chunk_list:
    re_ranked_chunk_list.extend(split_text_by_sentence(chunk, max_size=1500))
  output =  rerank_text_chunks(re_ranked_chunk_list,criteria)
  return output



import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from itertools import combinations



# Create an undirected graph
# G = nx.Graph()

def load_data(brand_vectors,G):
# Load brand nodes into graph
  for brand in brand_vectors.keys() :
    if G.has_node(brand):
      G.add_node(brand)

  # Compute cosine similarity and add edges (ensuring each pair is considered only once)
  for brand_a, brand_b in combinations(brand_vectors.keys(), 2):
    if not G.has_edge(brand_a, brand_b):
        similarity = cosine_similarity([brand_vectors[brand_a]], [brand_vectors[brand_b]])[0][0]
        if similarity > 0.2:  # Threshold for adding an edge
            G.add_edge(brand_a, brand_b, relationship="COMPETES_WITH", confidence=round(similarity, 2))
 
# Function to find competitors of a given brand with confidence scores
def find_competitors(brand):
    if brand not in G:
        print(f"Brand '{brand}' not found in the graph.")
        return []
    return sorted([(comp, G[brand][comp]['confidence']) for comp in G.neighbors(brand)], key=lambda x: x[1], reverse=True)

# Visualize the knowledge graph
def visualize_graph(G):
    plt.figure(figsize=(10, 8))
    pos = nx.spring_layout(G)
    edges = G.edges(data=True)
    
    edge_colors = ["red" if data["confidence"] > 0.7 else "blue" for _, _, data in edges]
    
    nx.draw(G, pos, with_labels=True, node_color='lightblue', edge_color=edge_colors, node_size=3000, font_size=10)
    labels = nx.get_edge_attributes(G, 'confidence')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
    plt.title("Brand Competitor Knowledge Graph")
    plt.show()




# Import necessary libraries (if not already imported)
import json
from pprint import pprint

# G = nx.Graph()

def search_competitors(brand,num_results = 1):

  # information = brand_search(f"list of all the global competitors/peers/alternatives for the brand {brand} with similar market segment/product offerings",num_results)
  information = brand_search(f'Top competitors of {brand}',num_results)
  # pprint(information)

  information_chunks = []

  for chunk in information:
    information_chunks.extend(split_text_by_sentence(chunk, max_size=1500))
  
  information_chunks_sorted = prioritized_shuffle(information_chunks)

  combined_chunks = ' '.join(information_chunks_sorted[:4])
  
  
  brand_competitors = {}
  llm_output = identify_competitors(brand, combined_chunks)

  # print(llm_output)
  output = extract_json_from_text(llm_output)

  return output



def generate_brand_summary(brand):
  information = brand_search(f"{brand} market analysis ",1)
  information_chunks = split_text_by_sentence(information[0],1000)
  information_chunks_re_ranked = prioritized_shuffle_v1(information_chunks)
  processed_information = ' '.join(information_chunks_re_ranked[:3])
  brand_summary = extract_brand_summary(brand,processed_information)
  return brand_summary


