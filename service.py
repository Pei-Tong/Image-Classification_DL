# Define BentoML service, for API deploy and usage.
import bentoml
from bentoml.io import Image, JSON
from PIL import Image as PILImage
import numpy as np

# Load model
runner = bentoml.tensorflow.get("image_classifier:latest").to_runner()

# Build BentoML service
svc = bentoml.Service("image_classifier_service", runners=[runner])

@svc.api(input=Image(), output=JSON())
def predict(img: PILImage.Image):
    img = img.resize((150, 150))  
    img_array = np.array(img) / 255.0 
    img_array = np.expand_dims(img_array, axis=0)  # increase batchd dims
    prediction = runner.run(img_array)
    return {"prediction": prediction.tolist()}
