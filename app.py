# app.py
import streamlit as st
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from PIL import Image
import requests
import re
from thefuzz import process
import torch.nn.functional as F
import base64

# Page Configuration
st.set_page_config(
    page_title="Fruision",
    page_icon="🍓",
    layout="wide",
)

# Constants and Mappings
MODEL_PATH = "resnet50_trained_CV.pth"
CLASS_NAMES_PATH = "class_names.txt"
THEMEALDB_API_INGREDIENTS_URL = "https://www.themealdb.com/api/json/v1/1/list.php?i=list"
THEMEALDB_API_FILTER_URL = "https://www.themealdb.com/api/json/v1/1/filter.php?i="

# Load custom CSS
def local_css(file_name):
    try:
        with open(file_name) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        st.warning(f"CSS file '{file_name}' not found. Using default styles.")

# Caching Functions
@st.cache_data
def load_class_names():
    with open(CLASS_NAMES_PATH, 'r') as f:
        return [line.strip() for line in f.readlines()]

@st.cache_resource
def load_model(num_classes):
    model = models.resnet50()
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
    model.eval()
    return model

# @st.cache_data
# def get_all_ingredients():
#     try:
#         response = requests.get(THEMEALDB_API_INGREDIENTS_URL)
#         response.raise_for_status()
#         data = response.json()
#         ingredients = [item['strIngredient'] for item in data['meals']]
#         ingredients.extend(["Apples", "Bananas", "Peaches", "Pears"])
#         return sorted(list(set(ingredients)))
#     except requests.RequestException:
#         return []

@st.cache_data
def get_all_ingredients():
    try:
        response = requests.get(THEMEALDB_API_INGREDIENTS_URL)
        response.raise_for_status()
        data = response.json()
        return sorted(
            item["strIngredient"].strip().lower()
            for item in data["meals"]
            if item["strIngredient"]
        )
    except requests.RequestException:
        return []

# Processing Functions
def transform_image(image_bytes):
    image = Image.open(image_bytes).convert("RGB")
    transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
    return transform(image).unsqueeze(0)

# def get_best_api_ingredient(predicted_class, all_ingredients):
#     if not all_ingredients: return None
#     cleaned_prediction = re.sub(r'\s*\d+$', '', predicted_class).strip()
#     best_match, score = process.extractOne(cleaned_prediction, all_ingredients)
#     if score > 88: return best_match
#     general_term = predicted_class.split(' ')[0]
#     if general_term in all_ingredients: return general_term
#     if f"{general_term}s" in all_ingredients: return f"{general_term}s"
#     return None

def get_best_api_ingredient(predicted_class, api_ingredients):
    if not api_ingredients:
        return None

    pred = predicted_class.lower().strip()

    # Exact match first (best case)
    if pred in api_ingredients:
        return pred

    # Plural handling
    if f"{pred}s" in api_ingredients:
        return f"{pred}s"

    # Fuzzy match ONLY if close enough
    match, score = process.extractOne(pred, api_ingredients)
    if score >= 92:
        return match

    return None


def fetch_recipes(ingredient):
    try:
        response = requests.get(f"{THEMEALDB_API_FILTER_URL}{ingredient}")
        response.raise_for_status()
        data = response.json()
        return data['meals']
    except requests.RequestException:
        return None

# UI Logic
local_css("style.css")

def get_base64_image(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

img_base64 = get_base64_image("wallpaper.jpg")

st.markdown(f"""
    <style>
    .stApp {{
        background-image: url("data:image/jpg;base64,{img_base64}");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }}
    </style>
    """, unsafe_allow_html=True)

class_names = load_class_names()
model = load_model(len(class_names))
valid_api_ingredients = get_all_ingredients()

st.markdown(
    """
    <div class="big-container">
    """,
    unsafe_allow_html=True
)

st.title("Welcome to Fruision!")
st.markdown("Upload an image of a fruit and we'll find recipes for it!")

uploaded_file = st.file_uploader(
    "Choose an image...",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is None:
    st.info("Upload an image to get started!")
    st.stop()

col1, col2 = st.columns([0.8, 1])

with col1:
    with st.container():
        st.image(uploaded_file, caption="Your Upload", use_column_width=True)

with col2:
    with st.spinner("🧠 Analyzing image..."):
        image_tensor = transform_image(uploaded_file)
        
        # Prediction logic
        with torch.no_grad():
            outputs = model(image_tensor)
            # Apply Softmax to get probabilities
            probabilities = F.softmax(outputs, dim=1)
            # Get the top probability and its index
            confidence, predicted_idx = torch.max(probabilities, 1)

        predicted_class_name = class_names[predicted_idx.item()]
        confidence_percent = confidence.item() * 100
        api_ingredient = get_best_api_ingredient(predicted_class_name, valid_api_ingredients)
    
    with st.container():
        st.subheader("🔍 Analysis Result")
        
        # UI to show confidence in prediction
        st.markdown(f"**Model Prediction:** `{predicted_class_name}`")
        st.progress(confidence.item(), text=f"Confidence: {confidence_percent:.2f}%")

        if api_ingredient:
            st.success(f"**Found Ingredient:** `{api_ingredient}`")
        else:
            st.error(f"Could not find a matching ingredient for `{predicted_class_name}`.")

st.markdown("---")

if api_ingredient:
    st.header(f"🍳 Recipes with {api_ingredient}")
    with st.spinner(f"Searching for recipes..."):
        recipes = fetch_recipes(api_ingredient)

    if recipes:
        num_recipes = len(recipes)
        max_recipes_to_show = 12
        for i in range(0, min(num_recipes, max_recipes_to_show), 4):
            cols = st.columns(4)
            for j in range(4):
                if i + j < num_recipes:
                    recipe = recipes[i+j]
                    with cols[j]:
                        st.image(recipe['strMealThumb'])
                        recipe_url = f"https://www.themealdb.com/meal/{recipe['idMeal']}"
                        st.markdown(f"**[{recipe['strMeal']}]({recipe_url})**", unsafe_allow_html=True)
    else:
        st.warning(
            f"No cooking recipes found for `{predicted_class_name}`.\n"
            "This ingredient may not be commonly used as a main dish."
        )

if not valid_api_ingredients:
    st.error("Critical: Could not connect to TheMealDB API. Please check your internet connection and refresh.")
