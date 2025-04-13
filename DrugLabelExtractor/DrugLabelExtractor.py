import base64
import json
import requests
from openai import OpenAI
from dotenv import load_dotenv
import os
import cv2
import numpy as np
import re
from .ExtractFromFeature import extract

class DrugLabelExtractor:
    def __init__(self):
        load_dotenv()
        self.api_key = os.getenv("UPSTAGE_API_KEY")
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        
        if not self.api_key or not self.openai_api_key:
            raise ValueError("API keys not found in .env file. Please set UPSTAGE_API_KEY and OPENAI_API_KEY.")
        
        self.url = "https://api.upstage.ai/v1/document-digitization"
        self.headers = {"Authorization": f"Bearer {self.api_key}"}
        self.openai_client = OpenAI(api_key=self.openai_api_key)

    def __call__(self, image_path: str) -> dict:
        ocr_result = self.extract_from_file(image_path)
        if(len(re.sub(r"<[^>]+>", "", ocr_result["content"]["html"])) < 30):
            _ = extract(ocr_result, image_path = image_path)
            ocr_result = _ if _ != None else ocr_result["content"]["html"]
        else:
            ocr_result = ocr_result["content"]["html"]
        drug_info = self.extract_drug_info(ocr_result)
        return drug_info
    
    def preprocess_image(self, image_path: str) -> str:
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        
        # CLAHE (Contrast Limited Adaptive Histogram Equalization) 적용
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        contrast_img = clahe.apply(img)

        # 임계값 처리 (선택적으로 적용)
        _, thresh_img = cv2.threshold(contrast_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # 임시 파일로 저장
        processed_path = image_path.replace(".", "_processed.")
        cv2.imwrite(processed_path, thresh_img)

        return processed_path
    
    def encode_img_to_base64(self, img_path: str) -> str:
        processed_path = self.preprocess_image(img_path)
        with open(processed_path, 'rb') as img_file:
            img_bytes = img_file.read()
            base64_data = base64.b64encode(img_bytes).decode('utf-8')
            return base64_data

    def extract_from_file(self, image_path: str) -> dict:
        files = {"document": open(image_path, "rb")}
        
        data = {
            "ocr": "force",
            "base64_encoding": "['table']",
            "model": "document-parse"
        }
        response = requests.post(
            self.url,
            headers=self.headers,
            files=files,
            data=data
        )
        
        return response.json()

    def extract_drug_info(self, ocr_result: dict) -> dict:
        
        
        response = self.openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content": """You are an expert in extracting information from drug labels. 
                    Your task is to extract information from the given HTML content and return it in English ONLY.
                    IMPORTANT: You MUST translate ALL Korean text to English. The response MUST NOT contain any Korean characters.
                    
                    Extract the following information:
                    - active ingredients (scientific names and their amounts if available, in the format [ingredient] or [ingredient, amount])
                    - efficacy and effects
                    - usage and dosage
                    - precautions
                    - other additives
                    
                    Rules:
                    1. ALL text MUST be in English
                    2. If the original text is in Korean, you MUST translate it to English
                    3. If any information cannot be extracted or is unclear, omit that field
                    4. The response MUST be in JSON format
                    5. DO NOT include any Korean characters in the response
                    6. For active ingredients:
                       - If amount is available, extract in format [ingredient, amount]
                       - If amount is not available, extract only [ingredient]"""
                },
                {
                    "role": "user",
                    "content": ocr_result
                }
            ],
            functions=[
                {
                    "name": "extract_drug_info",
                    "description": "Extract information from drug labels in English.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "active_ingredients": {
                                "type": "array",
                                "items": {
                                    "type": "array",
                                    "items": {
                                        
                                        "type": "string"
                                    },
                                    "minItems": 1,
                                    "maxItems": 2,
                                    "description": "List of active ingredients and their amounts in the format [ingredient] or [ingredient, amount]"
                                },
                                "description": "List of active ingredients with their amounts (if available)"
                            },
                            "efficacy": {
                                "type": "string",
                                "description": "Efficacy and effects of the drug in English"
                            },
                            "usage": {
                                "type": "string",
                                "description": "Usage and dosage in English"
                            },
                            "precautions": {
                                "type": "string",
                                "description": "Precautions for use in English"
                            },
                            "other_additives": {
                                "type": "string",
                                "description": "Other additives in English"
                            }
                        },
                        "required": []
                    }
                }
            ],
            function_call={"name": "extract_drug_info"}
        )
        
        result = json.loads(response.choices[0].message.function_call.arguments)
        return self._validate_result(result)

    def _validate_result(self, result: dict) -> dict:
        valid_result = {}
        for key, value in result.items():
            if value is not None and value != "" and value != []:
                if key == "active_ingredients":
                    valid_ingredients = self._validate_ingredients(value)
                    if valid_ingredients:
                        valid_result[key] = valid_ingredients
                elif isinstance(value, str):
                    if self._is_valid_text(value):
                        valid_result[key] = value
        return valid_result

    def _validate_ingredients(self, ingredients: list) -> list:
        valid_ingredients = []
        for ing in ingredients:
            if isinstance(ing, list):
                if ing and isinstance(ing[0], str) and ing[0].strip():
                    valid_ingredients.append(ing)
            elif isinstance(ing, str):
                if ing.strip() and not any(x in ing.lower() for x in ["unknown", "not found", "n/a", "none"]):
                    valid_ingredients.append([ing])
        return valid_ingredients

    def _is_valid_text(self, text: str) -> bool:
        return not any(x in text.lower() 
                      for x in ["unknown", "not found", "n/a", "none", "not applicable"])
