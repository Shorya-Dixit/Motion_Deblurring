from flask import Flask, request, render_template
import numpy as np
from PIL import Image
from io import BytesIO
from generator import generator_model
import base64


app = Flask(__name__)

# Loading GAN model
model = generator_model()
model.load_weights(r"weights\generator.h5")

def deblur(model, input, original_size):
    image = input.resize((256, 256))  # Resize image to match model input size
    image_resized = np.array(image)  # Convert to numpy array
    image_resized = np.expand_dims(image_resized, axis=0)  # Add batch dimension
    image_resized = (image_resized / 127.5) - 1  # Normalize image to [-1, 1]
    deblurred = model.predict(image_resized)
    deblurred = (deblurred + 1) * 127.5  # De-normalize image to [0, 255]
    deblurred = deblurred.astype(np.uint8)
    deblurred = np.squeeze(deblurred, axis=0)  # Remove batch dimension
    deblurred_resized = Image.fromarray(deblurred).resize(original_size)  # Resize back to original size
    return np.array(deblurred_resized)  # Convert back to numpy array

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        print("Received POST request")
        image = request.files['image']
        print("Image file:", image)
        image_pil = Image.open(image)
        print("Image opened")
        original_size = image_pil.size  # Save the original size
        deblurred_image = deblur(model, image_pil, original_size)
        print("Deblurred image created")
        
        # Store the images in memory
        uploaded_image_stream = BytesIO()
        image_pil.save(uploaded_image_stream, format='PNG')
        uploaded_image_base64 = 'data:image/png;base64,' + base64.b64encode(uploaded_image_stream.getvalue()).decode('utf-8')
        deblurred_image_stream = BytesIO()
        Image.fromarray(deblurred_image).save(deblurred_image_stream, format='PNG')
        deblurred_image_base64 = 'data:image/png;base64,' + base64.b64encode(deblurred_image_stream.getvalue()).decode('utf-8')
        return render_template('index.html', uploaded_image_base64=uploaded_image_base64, deblurred_image_base64=deblurred_image_base64)
    else:
        return render_template('index.html', uploaded_image_base64=None, deblurred_image_base64=None)

if __name__ == '__main__':
    app.run(debug=False)