import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from ultralytics import YOLO
from transformers import BlipProcessor
from transformers import BlipForConditionalGeneration
from transformers import BlipForQuestionAnswering
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import warnings
warnings.filterwarnings('ignore')

# ── DEVICE ───────────────────────────────────────────────────────
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)
print(f"{'='*60}")
print(f"CODNet Final Combined Pipeline")
print(f"{'='*60}")
print(f"Device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# ── COCO CLASSES ─────────────────────────────────────────────────
COCO_CLASSES = [
    'person','bicycle','car','motorcycle',
    'airplane','bus','train','truck','boat',
    'traffic light','fire hydrant','stop sign',
    'parking meter','bench','bird','cat','dog',
    'horse','sheep','cow','elephant','bear',
    'zebra','giraffe','backpack','umbrella',
    'handbag','tie','suitcase','frisbee',
    'skis','snowboard','sports ball','kite',
    'baseball bat','baseball glove','skateboard',
    'surfboard','tennis racket','bottle',
    'wine glass','cup','fork','knife','spoon',
    'bowl','banana','apple','sandwich','orange',
    'broccoli','carrot','hot dog','pizza',
    'donut','cake','chair','couch',
    'potted plant','bed','dining table',
    'toilet','tv','laptop','mouse','remote',
    'keyboard','cell phone','microwave','oven',
    'toaster','sink','refrigerator','book',
    'clock','vase','scissors','teddy bear',
    'hair drier','toothbrush'
]

# ── CODNet MODEL ─────────────────────────────────────────────────
class CodNet(nn.Module):
    def __init__(self, num_classes=80):
        super(CodNet, self).__init__()
        vgg = models.vgg19(weights=None)
        self.features = vgg.features
        for param in self.features.parameters():
            param.requires_grad = False
        self.pool = nn.AdaptiveAvgPool2d((7, 7))
        self.lstm = nn.LSTM(
            input_size=512,
            hidden_size=256,
            num_layers=1,
            batch_first=True
        )
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = x.flatten(2).transpose(1, 2)
        _, (h_n, _) = self.lstm(x)
        return self.fc(h_n[-1])

# ── IMAGE TRANSFORM ──────────────────────────────────────────────
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ── LOAD MODELS ──────────────────────────────────────────────────
print("\nLoading CODNet (118K trained - 96.46%)...")
codnet = CodNet(num_classes=80).to(device)
codnet.load_state_dict(
    torch.load('codnet_best.pth', map_location=device)
)
codnet.eval()
print("CODNet loaded! ✅")

print("\nLoading YOLOv8...")
yolo = YOLO("yolov8n.pt")
print("YOLOv8 loaded! ✅")

print("\nLoading BLIP Caption model...")
blip_processor = BlipProcessor.from_pretrained(
    "Salesforce/blip-image-captioning-base"
)
blip_caption = BlipForConditionalGeneration.from_pretrained(
    "Salesforce/blip-image-captioning-base"
).to(device)
print("BLIP Caption loaded! ✅")

print("\nLoading BLIP VQA model...")
vqa_processor = BlipProcessor.from_pretrained(
    "Salesforce/blip-vqa-base"
)
vqa_model = BlipForQuestionAnswering.from_pretrained(
    "Salesforce/blip-vqa-base"
).to(device)
print("BLIP VQA loaded! ✅")

print("\nAll models loaded successfully!")

# ── FUNCTIONS ────────────────────────────────────────────────────
def get_codnet_predictions(image_path, threshold=0.3):
    image = Image.open(image_path).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = codnet(tensor)
        probs = torch.sigmoid(outputs)[0]
    predictions = []
    for idx, score in enumerate(probs):
        if score > threshold and idx < len(COCO_CLASSES):
            predictions.append({
                'class': COCO_CLASSES[idx],
                'confidence': float(score)
            })
    predictions = sorted(
        predictions,
        key=lambda x: x['confidence'],
        reverse=True
    )[:10]
    return predictions

def detect_objects_yolo(image_path, conf=0.3):
    results = yolo(image_path, conf=conf, verbose=False)
    detections = []
    for r in results:
        for box in r.boxes:
            detections.append({
                'class': r.names[int(box.cls)],
                'confidence': float(box.conf),
                'bbox': box.xyxy[0].tolist()
            })
    return detections

def generate_caption(image_path):
    image = Image.open(image_path).convert("RGB")
    inputs = blip_processor(
        image, return_tensors="pt"
    ).to(device)
    out = blip_caption.generate(
        **inputs, max_new_tokens=50
    )
    return blip_processor.decode(
        out[0], skip_special_tokens=True
    )

def answer_question(image_path, question):
    image = Image.open(image_path).convert("RGB")
    inputs = vqa_processor(
        image, question, return_tensors="pt"
    ).to(device)
    out = vqa_model.generate(
        **inputs, max_new_tokens=20
    )
    return vqa_processor.decode(
        out[0], skip_special_tokens=True
    )

def visualize_results(
    image_path, yolo_dets,
    codnet_preds, caption, answers, save_name
):
    fig = plt.figure(figsize=(20, 12))
    image = Image.open(image_path)

    # Plot 1: YOLOv8 Detection
    ax1 = fig.add_subplot(2, 3, 1)
    ax1.imshow(image)
    colors = ['red','blue','green',
              'yellow','purple','orange']
    for i, det in enumerate(yolo_dets):
        x1, y1, x2, y2 = det['bbox']
        color = colors[i % len(colors)]
        rect = patches.Rectangle(
            (x1, y1), x2-x1, y2-y1,
            linewidth=2,
            edgecolor=color,
            facecolor='none'
        )
        ax1.add_patch(rect)
        ax1.text(
            x1, y1-5,
            f"{det['class']} {det['confidence']:.2f}",
            color=color, fontsize=8,
            bbox=dict(facecolor='white', alpha=0.7)
        )
    ax1.set_title(
        f'YOLOv8 Detection\n({len(yolo_dets)} objects)',
        fontweight='bold'
    )
    ax1.axis('off')

    # Plot 2: CODNet Predictions
    ax2 = fig.add_subplot(2, 3, 2)
    ax2.imshow(image)
    ax2.axis('off')
    text = "CODNet (118K, 96.46%):\n\n"
    for p in codnet_preds[:6]:
        text += f"• {p['class']}: {p['confidence']:.2f}\n"
    ax2.set_title(text, fontsize=9, loc='left')

    # Plot 3: Caption
    ax3 = fig.add_subplot(2, 3, 3)
    ax3.imshow(image)
    ax3.axis('off')
    ax3.set_title(
        f"BLIP Caption:\n{caption}",
        fontsize=10, fontweight='bold'
    )

    # Plot 4: VQA
    ax4 = fig.add_subplot(2, 3, 4)
    ax4.axis('off')
    qa_text = "Visual Question Answering:\n"
    qa_text += "="*30 + "\n\n"
    for q, a in answers.items():
        qa_text += f"Q: {q}\nA: {a}\n\n"
    ax4.text(
        0.05, 0.95, qa_text,
        transform=ax4.transAxes,
        fontsize=9, verticalalignment='top',
        bbox=dict(
            boxstyle='round',
            facecolor='lightblue',
            alpha=0.6
        )
    )
    ax4.set_title('VQA Results', fontweight='bold')

    # Plot 5: Combined
    ax5 = fig.add_subplot(2, 3, 5)
    ax5.axis('off')
    codnet_cls = set([p['class'] for p in codnet_preds])
    yolo_cls = set([d['class'] for d in yolo_dets])
    agreed = codnet_cls & yolo_cls
    combined_text = "Combined Results:\n"
    combined_text += "="*30 + "\n\n"
    combined_text += f"✅ Both agreed ({len(agreed)}):\n"
    for c in list(agreed)[:5]:
        combined_text += f"  • {c}\n"
    combined_text += f"\n📌 Total: {len(codnet_cls|yolo_cls)}\n"
    ax5.text(
        0.05, 0.95, combined_text,
        transform=ax5.transAxes,
        fontsize=9, verticalalignment='top',
        bbox=dict(
            boxstyle='round',
            facecolor='lightgreen',
            alpha=0.6
        )
    )
    ax5.set_title('Combined', fontweight='bold')

    # Plot 6: Comparison Chart
    ax6 = fig.add_subplot(2, 3, 6)
    models_list = ['VLM\n(CLIP)', 'GPT-3\n(CLIP)',
                   'Paper\nCODNet', 'Our\nCODNet']
    accuracies = [76.00, 86.00, 95.12, 96.46]
    colors_bar = ['#9E9E9E','#757575',
                  '#FF9800','#2196F3']
    bars = ax6.bar(
        models_list, accuracies,
        color=colors_bar, alpha=0.8
    )
    for bar, acc in zip(bars, accuracies):
        ax6.text(
            bar.get_x() + bar.get_width()/2.,
            bar.get_height() + 0.5,
            f'{acc:.1f}%',
            ha='center', fontweight='bold',
            fontsize=9
        )
    ax6.set_ylim(0, 105)
    ax6.set_ylabel('Accuracy (%)')
    ax6.set_title(
        'SOTA Comparison',
        fontweight='bold'
    )
    ax6.grid(True, alpha=0.3, axis='y')

    plt.suptitle(
        'CODNet: Complete Pipeline Results\n'
        'CNN-mLSTM (96.46%) + YOLOv8 + BLIP',
        fontsize=14, fontweight='bold'
    )
    plt.tight_layout()
    plt.savefig(save_name, dpi=150, bbox_inches='tight')
    print("✅ Saved to results_final_combined.png")

# ── MAIN ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    # The specific list of images you want to test
    test_images = [
        "sample.jpg", "sample2.jpg", "sample3.jpg", 
        "sastra.jpeg", "girls.jpeg", "mycls.jpeg", 
        "lib.jpeg", "car.jpeg", "office.jpg"
    ]
    
    print(f"🚀 Starting test for {len(test_images)} specific images...")

    for img_path in test_images:
        if not os.path.exists(img_path):
            print(f"⚠️ Skipping {img_path} - file not found in directory.")
            continue
            
        print(f"\nProcessing: {img_path}...")
        
        # 1. Get predictions
        codnet_preds = get_codnet_predictions(img_path)
        yolo_dets = detect_objects_yolo(img_path)
        caption = generate_caption(img_path)
        
        # 2. Setup VQA Questions
        questions = {
            "What is in the image?": answer_question(img_path, "What is in the image?"),
            "Is it indoors or outdoors?": answer_question(img_path, "Is it indoors or outdoors?")
        }
        
        # 3. Create unique save name
        # This saves them as result_sastra.png, result_girls.png, etc.
        save_name = f"result_{img_path.replace('.', '_')}.png"
        
        # 4. Visualize (Ensure your function accepts save_name!)
        visualize_results(img_path, yolo_dets, codnet_preds, caption, questions, save_name)
        print(f"✅ Saved to {save_name}")

    print("\n" + "="*60)
    print("Batch Processing Complete! 🎉")
    print("="*60)
