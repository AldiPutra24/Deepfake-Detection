import os
import numpy as np
from PIL import Image, ImageChops, ImageEnhance
import tempfile

def build_model():
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, InputLayer, Dropout
    model = Sequential([
        InputLayer(input_shape=(224, 224, 3)),
        Conv2D(32, (3,3), activation='relu'),
        MaxPooling2D(),
        Conv2D(64, (3,3), activation='relu'),
        MaxPooling2D(),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(2, activation='softmax')
    ])
    return model


def convert_to_ela_image(path, quality=90, resize_to=(224, 224)):
    from PIL import Image, ImageChops, ImageEnhance
    import numpy as np
    import tempfile
    import os
    # Open original image
    original = Image.open(path).convert('RGB')
    # Create temp file
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
    temp_path = temp_file.name
    temp_file.close()
    # Save reduced quality image
    original.save(temp_path, 'JPEG', quality=quality)
    # Open compressed image
    with Image.open(temp_path) as compressed:
        ela_image = ImageChops.difference(original, compressed)
    # Close original
    original.close()
    # Scale brightness
    extrema = ela_image.getextrema()
    max_diff = max([ex[1] for ex in extrema]) or 1
    scale = 255.0 / max_diff
    ela_image = ImageEnhance.Brightness(ela_image).enhance(scale)
    # Resize
    ela_image = ela_image.resize(resize_to)
    # Convert to numpy array and normalize
    ela_array = np.array(ela_image) / 255.0
    # Clean up temporary file
    try:
        os.remove(temp_path)
    except OSError:
        pass
    return ela_array