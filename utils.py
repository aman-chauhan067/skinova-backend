import cv2
import numpy as np
from sklearn.cluster import KMeans
from statistics import mode
from collections import defaultdict

def detect_face(image):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    if len(faces) == 0:
        return None
    (x, y, w, h) = faces[0]
    return image[y:y+h, x:x+w]

def analyze_skin_color(face_image):
    pixels = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB).reshape(-1, 3)
    kmeans = KMeans(n_clusters=3, n_init=10)
    kmeans.fit(pixels)
    counts = np.bincount(kmeans.labels_)
    dominant_color = kmeans.cluster_centers_[np.argmax(counts)]
    return dominant_color

def determine_undertone(color):
    r, g, b = color
    if r > b and (r - g) > 10:
        return 'warm'
    elif b > r and (b - g) > 10:
        return 'cool'
    return 'neutral'

def predict_skin_concerns(face_image, model):
    # Mock prediction for demonstration (replace with trained model)
    concern_labels = ['Acne', 'Aging', 'Hyperpigmentation', 'Dryness', 'Sensitive Skin']
    severity_labels = ['mild', 'moderate', 'severe']
    detected_concerns = []
    mock_probs = np.random.rand(15)  # 5 concerns * 3 severities
    for i, prob in enumerate(mock_probs):
        if prob > 0.5:
            concern_idx = i // 3
            severity_idx = i % 3
            detected_concerns.append({"name": concern_labels[concern_idx], "severity": severity_labels[severity_idx]})
    return detected_concerns

def filter_contraindications(ingredients):
    filtered = []
    for ing in ingredients:
        if not any(ing["name"] in existing["avoid_with"] or existing["name"] in ing["avoid_with"] for existing in filtered):
            filtered.append(ing)
    return filtered

def split_am_pm(ingredients):
    am = [ing for ing in ingredients if "AM" in ing["routine_stage"]]
    pm = [ing for ing in ingredients if "PM" in ing["routine_stage"]]
    return am, pm

def add_essential_steps(routine, time, db):
    if not any(ing["product_type"] == "Cleanser" for ing in routine):
        routine.insert(0, next(ing for ing in db if ing["name"] == "Generic Cleanser"))
    if not any(ing["product_type"] == "Moisturizer" for ing in routine):
        routine.append(next(ing for ing in db if ing["name"] == "Generic Moisturizer"))
    if time == "AM" and not any(ing["product_type"] == "Sunscreen" for ing in routine):
        routine.append(next(ing for ing in db if ing["name"] == "Generic Sunscreen"))
    return routine

def order_products(ingredients, time):
    routine_order = {"AM": ["Cleanser", "Toner", "Serum", "Moisturizer", "Sunscreen"], "PM": ["Cleanser", "Toner", "Serum", "Moisturizer", "Treatment"]}
    order = routine_order.get(time, [])
    def order_key(ing):
        try:
            return order.index(ing["product_type"])
        except ValueError:
            return len(order)
    return sorted(ingredients, key=order_key)

def format_routine(routine, products, detected_concerns):
    formatted = []
    for step in routine:
        step_name = step["name"] if step["name"].startswith("Generic") else f"{step['name']} ({step['product_type']})"
        description = step.get("description", "Essential skincare step.")
        if "max_use" in step and not step["name"].startswith("Generic"):
            max_use = step["max_use"] if isinstance(step["max_use"], str) else step["max_use"].get(detected_concerns[0]["severity"].lower(), "as directed")
            step_name += f" - {max_use}"
        formatted.append({"step": step_name, "description": description, "products": products.get(step["name"], {})})
    return formatted

def get_recommended_ingredients(detected_concerns, db):
    recommended_names = set()
    for ing in db:
        for concern in detected_concerns:
            for ing_concern in ing["concerns"]:
                if (ing_concern["name"].lower() == concern["name"].lower() and 
                    concern["severity"].lower() in [s.lower() for s in ing_concern["severity"]]):
                    recommended_names.add(ing["name"])
    return [ing for ing in db if ing["name"] in recommended_names]

def recommend_products(ingredients):
    from product_db import product_db
    product_recs = defaultdict(lambda: defaultdict(list))
    for ing in ingredients:
        matches = [p for p in product_db if p["ingredient"].lower() == ing["name"].lower() and p["product_type"].lower() == ing["product_type"].lower()]
        if matches:
            for category in ["budget", "mid", "premium"]:
                product_recs[ing["name"]][category] = matches[0].get(category, [])[:2]
    return product_recs

def generate_skincare_routine(detected_concerns):
    from ingredients_db import ingredients_db
    if not detected_concerns:
        print("No concerns detected. Generating basic routine.")
    recommended = get_recommended_ingredients(detected_concerns, ingredients_db)
    am_rec, pm_rec = split_am_pm(recommended)
    am_filtered = filter_contraindications(am_rec)
    pm_filtered = filter_contraindications(pm_rec)
    am = add_essential_steps(am_filtered, "AM", ingredients_db)
    pm = add_essential_steps(pm_filtered, "PM", ingredients_db)
    am_ordered = order_products(am, "AM")
    pm_ordered = order_products(pm, "PM")
    product_recs = recommend_products(am_ordered + pm_ordered)
    return {
        "concerns": detected_concerns,
        "am_routine": format_routine(am_ordered, product_recs, detected_concerns),
        "pm_routine": format_routine(pm_ordered, product_recs, detected_concerns)
    }