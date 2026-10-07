"""RetinaCare AI backend. Run: uvicorn main:app --reload"""
import io
import torch
import torch.nn as nn
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from torchvision import models, transforms

app = FastAPI(title="RetinaCare AI")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

model = models.efficientnet_b0(weights=None)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, 5)
# model.load_state_dict(torch.load("efficientnet_b0_dr.pth", map_location="cpu"))  # <- your trained weights
model.eval()

prep = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    img = Image.open(io.BytesIO(await file.read())).convert("RGB")
    with torch.no_grad():
        probs = torch.softmax(model(prep(img).unsqueeze(0)), dim=1)[0].tolist()
    return {"probabilities": probs}
