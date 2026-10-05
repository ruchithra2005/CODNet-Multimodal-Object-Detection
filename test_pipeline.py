import torch
from ultralytics import YOLO
from transformers import BlipProcessor, BlipForConditionalGeneration
from transformers import BlipForQuestionAnswering
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import urllib.request
import os

print("GPU Available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("Device:", torch.cuda.get_device_name(0))

device = "cuda" if torch.cuda.is_available() else "cpu"

def detect_objects(image_path, conf=0.3):
    print("\n-- Running Object Detection --")
    model = YOLO("yolov8n.pt")
    results = model(image_path, conf=conf)
    detections = []
    for r in results:
        for box in r.boxes:
            detections.append({
                "class": r.names[int(box.cls)],
                "confidence": float(box.conf),
                "bbox": box.xyxy[0].tolist()
            })
    print(f"Detected {len(detections)} objects")
    for d in detections:
        print(f"  {d['class']}: {d['confidence']:.2f}")
    return results, detections

def generate_caption(image_path):
    print("\n-- Generating Caption --")
    processor = BlipProcessor.from_pretrained(
        "Salesforce/blip-image-captioning-base"
    )
    model = BlipForConditionalGeneration.from_pretrained(
        "Salesforce/blip-image-captioning-base"
    ).to(device)
    image = Image.open(image_path).convert("RGB")
    inputs = processor(image, return_tensors="pt").to(device)
    out = model.generate(**inputs, max_new_tokens=50)
    caption = processor.decode(out[0], skip_special_tokens=True)
    print(f"Caption: {caption}")
    return caption

def answer_question(image_path, question):
    processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
    model = BlipForQuestionAnswering.from_pretrained(
        "Salesforce/blip-vqa-base"
    ).to(device)
    image = Image.open(image_path).convert("RGB")
    inputs = processor(image, question, return_tensors="pt").to(device)
    out = model.generate(**inputs, max_new_tokens=20)
    answer = processor.decode(out[0], skip_special_tokens=True)
    return answer

def visualize_results(image_path, detections, caption, answers):
    print("\n-- Saving Visualization --")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    image = Image.open(image_path)

    axes[0].imshow(image)
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        rect = patches.Rectangle(
            (x1, y1), x2-x1, y2-y1,
            linewidth=2, edgecolor="red", facecolor="none"
        )
        axes[0].add_patch(rect)
        axes[0].text(
            x1, y1-5,
            f"{det['class']} {det['confidence']:.2f}",
            color="red", fontsize=8,
            bbox=dict(facecolor="white", alpha=0.5)
        )
    axes[0].set_title("YOLOv8 Object Detection")
    axes[0].axis("off")

    axes[1].imshow(image)
    axes[1].axis("off")
    result_text = f"Caption:\n{caption}\n\n"
    for q, a in answers.items():
        result_text += f"Q: {q}\nA: {a}\n\n"
    axes[1].set_title(result_text, fontsize=9, loc="left", wrap=True)

    plt.tight_layout()
    plt.savefig("results.png", dpi=150, bbox_inches="tight")
    print("Results saved to results.png")

if __name__ == "__main__":
    img_path = "car.jpeg"
    if not os.path.exists(img_path):
        print("Downloading sample image...")
        urllib.request.urlretrieve(
            "http://images.cocodataset.org/val2017/000000039769.jpg",
            img_path
        )
        print("Image downloaded!")

    results, detections = detect_objects(img_path)
    caption = generate_caption(img_path)

    print("\n-- Answering Questions --")
    questions = {
        "What is in the image?": "",
        "How many people are there?": "",
        "Which color is appeared more?": ""
    }
    for q in questions:
        questions[q] = answer_question(img_path, q)
        print(f"Q: {q}")
        print(f"A: {questions[q]}")

    visualize_results(img_path, detections, caption, questions)
    print("\nAll Done!")


