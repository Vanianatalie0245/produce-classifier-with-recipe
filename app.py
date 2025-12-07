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

# Page Configuration
st.set_page_config(
    page_title="Recipe Finder",
    page_icon="🍓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Constants and Mappings
MODEL_PATH = "resnet50_trained_CV.pth"
CLASS_NAMES_PATH = "class_names.txt"
THEMEALDB_API_INGREDIENTS_URL = "https://www.themealdb.com/api/json/v1/1/list.php?i=list"
THEMEALDB_API_FILTER_URL = "https://www.themealdb.com/api/json/v1/1/filter.php?i="

MANUAL_OVERRIDE_MAP = {
    # 1. Synonyms / Alternate Names
    "Cocos 1": "Coconut",
    "Eggplant 1": "Aubergine",
    "Eggplant long 1": "Aubergine",
    "Maracuja 1": "Passion Fruit",
    "Limes 1": "Lime",
    "Blackberrie 1": "Blackberries",
    "Blackberrie 2": "Blackberries",
    "Cabbage red 1": "Red Cabbage",

    # 2. Critical Overrides to prevent wrong generalization
    "Potato Sweet 1": "Sweet Potatoes",

    # 3. Mapping generic terms to specific, useful ones
    "Nut Forest 1": "Hazelnuts",
    "Nut Pecan 1": "Pecan Nuts",

    # 4. Mapping terms to their closest equivalents
    "Onion Red 1": "Red Onions",
    "Pepper Green 1": "Green Pepper",
    "Pepper Orange 1": "Red Pepper",
    "Pepper Red 1": "Red Pepper",
    "Pepper Yellow 1": "Yellow Pepper",
    "Zucchini 1": "Zucchini",
    "Zucchini dark 1": "Zucchini",
    
    # 5. Mapping API misses to a reasonable substitute for recipe ideas
    "Kaki 1": "Apple",
    "Pepino 1": "Melon Piel de Sapo",
    "Physalis 1": "Gooseberry",
    "Tamarillo 1": "Tomatoes",
}

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

@st.cache_data
def get_all_ingredients():
    try:
        response = requests.get(THEMEALDB_API_INGREDIENTS_URL)
        response.raise_for_status()
        data = response.json()
        ingredients = [item['strIngredient'] for item in data['meals']]
        ingredients.extend(["Apples", "Bananas", "Peaches", "Pears"])
        return sorted(list(set(ingredients)))
    except requests.RequestException:
        return []

# Processing Functions
def transform_image(image_bytes):
    image = Image.open(image_bytes).convert("RGB")
    transform = transforms.Compose([transforms.Resize((100, 100)), transforms.ToTensor()])
    return transform(image).unsqueeze(0)

def get_best_api_ingredient(predicted_class, all_ingredients):
    if not all_ingredients: return None
    if predicted_class in MANUAL_OVERRIDE_MAP: return MANUAL_OVERRIDE_MAP[predicted_class]
    cleaned_prediction = re.sub(r'\s*\d+$', '', predicted_class).strip()
    best_match, score = process.extractOne(cleaned_prediction, all_ingredients)
    if score > 88: return best_match
    general_term = predicted_class.split(' ')[0]
    if general_term in all_ingredients: return general_term
    if f"{general_term}s" in all_ingredients: return f"{general_term}s"
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

class_names = load_class_names()
model = load_model(len(class_names))
valid_api_ingredients = get_all_ingredients()

with st.sidebar:
    st.image("https://www.themealdb.com/images/logo-small.png")
    st.title("Recipe Finder")
    st.markdown("Upload an image of a fruit, vegetable, or nut, and we'll find recipes for it!")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"], label_visibility="collapsed")

st.header("🍓 Your Culinary Assistant 🥕")

if uploaded_file is None:
    st.info("👈 Upload an image in the sidebar to get started!")
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
        st.warning(f"Sorry, no recipes found for `{api_ingredient}` in TheMealDB.")

if not valid_api_ingredients:
    st.error("Critical: Could not connect to TheMealDB API. Please check your internet connection and refresh.")