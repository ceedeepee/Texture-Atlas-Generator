import cv2
import json
import sys

def box_inside(boxA, boxB):
    ax, ay, aw, ah = boxA
    bx, by, bw, bh = boxB
    # Check if boxA is even partially inside boxB
    return not (ax+aw <= bx or ax >= bx+bw or ay+ah <= by or ay >= by+bh)

def generate_bounding_boxes(image_path, alpha_threshold=10, min_width=10, min_height=10):
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    alpha_channel = image[:, :, 3]

    _, binary = cv2.threshold(alpha_channel, alpha_threshold, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    raw_boxes = [cv2.boundingRect(contour) for contour in contours]

    # Filter boxes by minimum size first
    boxes_filtered = [
        box for box in raw_boxes if box[2] >= min_width and box[3] >= min_height
    ]

    # Sort boxes by area descending (large boxes first)
    boxes_filtered.sort(key=lambda b: b[2]*b[3], reverse=True)

    final_boxes = []
    for current_box in boxes_filtered:
        if not any(box_inside(current_box, bigger_box) for bigger_box in final_boxes):
            final_boxes.append(current_box)

    # Output boxes
    boxes = {}
    for idx, (x, y, w, h) in enumerate(final_boxes):
        boxes[f'sprite_{idx}'] = {'x': x, 'y': y, 'width': w, 'height': h}

    return boxes

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python3 generate-texture-atlas.py <sprite_sheet.png>")
        sys.exit(1)

    sprite_sheet_path = sys.argv[1]
    boxes = generate_bounding_boxes(sprite_sheet_path)

    json_output_path = sprite_sheet_path.replace('.png', '_atlas.json')
    with open(json_output_path, 'w') as f:
        json.dump(boxes, f, indent=4)

    print(f'Cleaned bounding boxes JSON generated: {json_output_path}')
