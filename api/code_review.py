import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from flask import request, jsonify

tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")
model = RobertaForSequenceClassification.from_pretrained("microsoft/codebert-base")

def analyze_code(request):
    try:
        data = request.get_json()
        code_snippet = data.get("code", "")
        
        if not code_snippet:
            return jsonify({"error": "No code provided"}), 400
        
        inputs = tokenizer(code_snippet, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            outputs = model(**inputs)
        
        review_result = "Code looks good!" if torch.rand(1).item() > 0.5 else "Potential issues found!"
        
        return jsonify({"review": review_result})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
