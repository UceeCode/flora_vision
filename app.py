from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import io
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

model = load_model("flora_vision_model.h5")

CLASS_NAMES = [
    "Pink Primrose", "Hard-Leaved Pocket Orchid", "Canterbury Bells", "Sweet Pea",
    "English Marigold", "Tiger Lily", "Moon Orchid", "Bird of Paradise", "Monkshood",
    "Globe Thistle", "Snapdragon", "Colt's Foot", "King Protea", "Spear Thistle",
    "Yellow Iris", "Globe Flower", "Purple Coneflower", "Peruvian Lily", "Balloon Flower",
    "Giant White Arum Lily", "Fire Lily", "Pincushion Flower", "Fritillary", "Red Ginger",
    "Grape Hyacinth", "Corn Poppy", "Prince of Wales Feathers", "Stemless Gentian",
    "Artichoke", "Sweet William", "Carnation", "Garden Phlox", "Love in the Mist",
    "Mexican Aster", "Alpine Sea Holly", "Ruby-Lipped Cattleya", "Cape Flower",
    "Great Masterwort", "Siam Tulip", "Lenten Rose", "Barbeton Daisy", "Daffodil",
    "Sword Lily", "Poinsettia", "Bolero Deep Blue", "Wallflower", "Marigold",
    "Buttercup", "Oxeye Daisy", "Common Dandelion", "Petunia", "Wild Pansy",
    "Primula", "Sunflower", "Pelargonium", "Bishop of Llandaff", "Gaura",
    "Geranium", "Orange Dahlia", "Pink-Yellow Dahlia", "Cautleya Spicata",
    "Japanese Anemone", "Black-Eyed Susan", "Silverbush", "Californian Poppy",
    "Osteospermum", "Spring Crocus", "Bee Balm", "Pink-Bristled Thistle",
    "Ball Moss", "Foxglove", "Bougainvillea", "Camellia", "Mallow", "Mexican Petunia",
    "Bromelia", "Blanket Flower", "Trumpet Creeper", "Blackberry Lily", "Common Tulip",
    "Wild Rose", "Primrose", "Cornflower", "Lily of the Valley", "Bishop's Flower",
    "Globe Amaranth", "Cape Daisy", "Stemless Gentian", "Fairy Lantern", "Japanese Maple",
    "Hippeastrum", "Water Lily", "Lotus", "Ginger Flower", "Desert Rose", "Fox-And-Cubs",
    "Magnolia", "Buttercup", "Bladder Campion", "Tree Poppy", "Mountain Avens",
    "Snowdrop", "Shooting Star", "Common Snowdrop", "Wild Raspberry", "Common Bluebell",
    "Winterberry", "American Elderberry"
]


@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No files uploaded"}), 400
    file = request.files["file"]
    image = Image.open(io.BytesIO(file.read()))

    image = image.resize((224, 224))
    image = np.array(image) / 255.0

    image = np.expand_dims(image, axis=0)

    prediction = model.predict(image)
    predicted_class = CLASS_NAMES[np.argmax(prediction)]
    confidence = float(np.max(prediction))

    return jsonify({"class": predicted_class, "confidence": confidence})

if __name__ == "__main__":
    app.run(debug=True)



