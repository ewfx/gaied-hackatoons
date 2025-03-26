from fastapi import FastAPI, Request, HTTPException
from transformers import BertTokenizer, BertModel
import torch
import pdfplumber
from io import BytesIO
from PIL import Image
import pytesseract

app = FastAPI()

# Load pre-trained BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# Classification model
class ClassificationModel(torch.nn.Module):
    def __init__(self, model, num_classes=8):
        super(ClassificationModel, self).__init__()
        self.bert = model
        self.classification_layer = torch.nn.Linear(model.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask=None):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        logits = self.classification_layer(pooled_output)
        return logits

classification_model = ClassificationModel(model, num_classes=8)

# Text extraction for PDFs
def extract_text_from_pdf(pdf_file: BytesIO) -> str:
    try:
        with pdfplumber.open(pdf_file) as pdf:
            text = ""
            for page in pdf.pages:
                text += page.extract_text() or ""
        return text
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing PDF: {str(e)}")

# Text extraction for images
def extract_text_from_image(image_file: BytesIO) -> str:
    try:
        image = Image.open(image_file)
        text = pytesseract.image_to_string(image)
        return text
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing image: {str(e)}")

# Email classification endpoint
@app.post("/classify_email/")
async def classify_email(request: Request):
    try:
        body = await request.json()  # Parse the incoming JSON body
        print(f"Raw Input: {body}")

        # Extract fields from raw input
        email_text = body.get("email_text", "")
        attachment_data = body.get("attachment", None)  # Optional

        attachment_text = ""
        if attachment_data:
            file_bytes = BytesIO(bytes(attachment_data))
            if body.get("attachment_type", "").lower() == "pdf":
                attachment_text = extract_text_from_pdf(file_bytes)
            elif body.get("attachment_type", "").lower() in ["jpg", "jpeg", "png"]:
                attachment_text = extract_text_from_image(file_bytes)
            else:
                raise HTTPException(status_code=400, detail="Unsupported attachment type")

        # Combine email text and extracted text from attachment
        combined_text = (email_text or "") + " " + (attachment_text or "")

        # Tokenize text
        inputs = tokenizer(combined_text, return_tensors="pt", truncation=True, padding=True, max_length=512)

        # Predict class
        with torch.no_grad():
            logits = classification_model(inputs['input_ids'], attention_mask=inputs['attention_mask'])
            predicted_class = torch.argmax(logits, dim=1).item()

        classification = f"Class {predicted_class + 1}"  # Adjust for 1-based indexing
        return {"classification": classification}

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing request: {str(e)}")