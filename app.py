from flask import Flask, render_template, request
import cv2
import pytesseract
from PIL import Image
from textblob import TextBlob
import os

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Path to Tesseract executable (modify for your system)
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Keywords for evaluation (customize based on question)
keywords = ['keyword1', 'keyword2', 'keyword3']

def preprocess_image(image_path):
    """Preprocess image to improve OCR accuracy."""
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
    
    processed_image_path = os.path.join(app.config['UPLOAD_FOLDER'], 'processed.png')
    cv2.imwrite(processed_image_path, thresh)
    
    return processed_image_path

def extract_text(image_path):
    """Extract text using Tesseract OCR."""
    image = Image.open(image_path)
    return pytesseract.image_to_string(image, lang='eng')

def evaluate_text(text):
    """Evaluate text based on keywords and grammar."""
    score = sum(2 for keyword in keywords if keyword.lower() in text.lower())

    blob = TextBlob(text)
    corrected_text = str(blob.correct())

    if text != corrected_text:
        score -= 1  # Deduct for grammar/spelling mistakes

    return corrected_text, score

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    file = request.files['answer_sheet']
    
    if file.filename == '':
        return "No selected file"

    if file:
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(image_path)

        # Process image and extract text
        processed_image = preprocess_image(image_path)
        extracted_text = extract_text(processed_image)
        corrected_text, score = evaluate_text(extracted_text)

        return render_template('result.html', extracted_text=extracted_text, corrected_text=corrected_text, score=score)

if __name__ == '__main__':
    app.run(debug=True)
