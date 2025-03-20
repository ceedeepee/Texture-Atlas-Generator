import cv2
import json
import sys
import numpy as np
from PIL import Image
import tensorflow as tf


def box_inside(boxA, boxB):
    ax, ay, aw, ah = boxA
    bx, by, bw, bh = boxB
    # Check if boxA is even partially inside boxB
    return not (ax + aw <= bx or ax >= bx + bw or ay + ah <= by or ay >= by + bh)


# Load pre-trained MobileNet model
model = tf.keras.applications.MobileNetV2(weights='imagenet')
input_size = (224, 224)


# Function to classify cropped sprite
def classify_sprite(sprite_image):
    img = Image.fromarray(sprite_image).convert('RGB')
    img = img.resize(input_size)
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)

    predictions = model.predict(img_array)
    decoded = tf.keras.applications.mobilenet_v2.decode_predictions(predictions, top=1)
    return decoded[0][0][1]  # returns top label name


def generate_bounding_boxes(image_path, alpha_threshold=10, min_width=10, min_height=10):
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    alpha_channel = image[:, :, 3]

    _, binary = cv2.threshold(alpha_channel, alpha_threshold, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    raw_boxes = [cv2.boundingRect(contour) for contour in contours]

    boxes_filtered = [
        box for box in raw_boxes if box[2] >= min_width and box[3] >= min_height
    ]

    boxes_filtered.sort(key=lambda b: b[2] * b[3], reverse=True)

    final_boxes = []
    for current_box in boxes_filtered:
        if not any(box_inside(current_box, bigger_box) for bigger_box in final_boxes):
            final_boxes.append(current_box)

    boxes = {}
    for idx, (x, y, w, h) in enumerate(final_boxes):
        sprite_crop = image[y:y + h, x:x + w]
        label = classify_sprite(sprite_crop)
        boxes[f'sprite_{idx}'] = {'x': x, 'y': y, 'width': w, 'height': h, 'label': label}

    return boxes


def draw_bounding_boxes(image_path, boxes, output_path):
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    output_image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)

    for box in boxes.values():
        x, y, w, h = box['x'], box['y'], box['width'], box['height']
        label = box['label']
        cv2.rectangle(output_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(output_image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 0, 0), 1, cv2.LINE_AA)

    cv2.imwrite(output_path, output_image)


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python3 generate-texture-atlas.py <sprite_sheet.png>")
        sys.exit(1)

    sprite_sheet_path = sys.argv[1]
    boxes = generate_bounding_boxes(sprite_sheet_path)

    json_output_path = sprite_sheet_path.replace('.png', '_atlas.json')
    image_output_path = sprite_sheet_path.replace('.png', '_bbox.png')

    with open(json_output_path, 'w') as f:
        json.dump(boxes, f, indent=4)

    draw_bounding_boxes(sprite_sheet_path, boxes, image_output_path)

    print(f'Bounding boxes JSON generated successfully: {json_output_path}')
    print(f'Bounding boxes image generated successfully: {image_output_path}')
