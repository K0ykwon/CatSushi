import openai
import os
import json
from dotenv import load_dotenv
from pydantic import BaseModel
from typing import List
from openai import OpenAI
from torchvision import models, transforms
import torch
import torch.nn as nn
from PIL import Image
import numpy as np
import requests
from bs4 import BeautifulSoup
model = models.efficientnet_b0(pretrained=True)
model.fc = nn.Linear(model.classifier[1].in_features, 11)  # 11개의 클래스 (원형 ~ 기타)

    # GPU 사용 가능 여부 확인
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.load_state_dict(torch.load("./DrugLabelExtractor/models/best_model.pth", map_location=device))
model.eval()
class PillIdentifiers(BaseModel):
    identifiers: List[str]

def extract(identifiers:str, image_path: str = "./download.jpeg") -> str:
    headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    url = "https://m.terms.naver.com/" + getinfolink(identifiers, image_path)
    response = requests.get(url, headers=headers)

    soup = BeautifulSoup(response.text, 'html.parser')
    if soup.select_one('#hview-container > div') != None:
        return soup.select_one('#hview-container > div').text
    else:
        return ""

def getinfolink(identifiers:str, image_path: str = "./download.jpeg") -> str:
    headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    url = generate_link(identifiers, image_path)
    # 웹 페이지 요청
    response = requests.get(url, headers=headers)
    response.raise_for_status()  # 요청 실패 시 예외 발생

    # BeautifulSoup 객체 생성
    soup = BeautifulSoup(response.text, 'html.parser')

    # CSS 셀렉터로 링크 추출
    link = soup.select_one('#list > li:nth-child(1) > div > a')
    
    # 링크가 존재하는 경우
    if link:
        href = link.get('href')
        return href
    else:
        url = generate_link(identifiers, image_path, 1)
        response = requests.get(url, headers=headers)
        soup = BeautifulSoup(response.text, 'html.parser')
        link = soup.select_one('#list > li:nth-child(1) > div > a')
        
        if link:
            href = link.get('href')
            return href
        else:
            return ""

def generate_link(identifier: str, image_path: str = "./download.jpeg", status = 0) -> str:
    if status == 0: return f"https://m.terms.naver.com/medicineSearch.naver?mode=exteriorSearch&query=&so=st4.dsc&shape={extract_from_feature_from_image(image_path)}&color=&dosageForm=&divisionLine=&identifier={extract_from_feature(identifier)}"
    else: return f"https://m.terms.naver.com/medicineSearch.naver?mode=exteriorSearch&query=&so=st4.dsc&shape=&color=&dosageForm=&divisionLine=&identifier={extract_from_feature(identifier)}"

def extract_from_feature(prompt: str) -> PillIdentifiers:
    load_dotenv()
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    response = client.responses.create(
        model="gpt-4o-mini",
        input=[
            {
                "role": "system",
                "content": "You are an expert at analyzing drug descriptions. Extract all pill identifiers except for the number (alphanumeric characters printed on pills) from the text."
            },
            {
                "role": "user", 
                "content": str(prompt)
            }
        ],
        text={
            "format": {
                "type": "json_schema",
                "name": "PillIdentifiers",
                "schema": {
                    "type": "object",
                    "properties": {
                        "identifiers": {
                            "type": "array",
                            "items": {"type": "string"}
                        }
                    },
                    "required": ["identifiers"],
                    "additionalProperties": False
                },
                "strict": True
            }
        }
    )
    # Parse JSON output directly
    result = json.loads(response.output_text)
    return ('%2520'.join(PillIdentifiers(**result).identifiers))

def extract_from_feature_from_image(image_path: str = "./download.jpeg") -> str:
    
    def predict_image(image_path):
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])
        image = Image.open(image_path).convert("RGB")
        image = transform(add_noise_to_background(image)).unsqueeze(0).to(device)  # 이미지를 텐서로 변환하고 배치 차원 추가
        
        with torch.no_grad():  # 예측 시 기울기 계산을 하지 않음
            output = model(image)  # 예측
            _, predicted = torch.max(output, 1)  # 가장 큰 값을 갖는 클래스 인덱스 추출
        return predicted.item()
    result = predict_image(image_path)
    return result

def add_noise_to_background(image, noise_intensity=0.2, size=(224, 224)):
    np_image = np.array(image)  # PIL 이미지를 numpy 배열로 변환
    noise = np.random.normal(0, noise_intensity * 255, np_image.shape).astype(np.int32)  # 노이즈 생성
    noisy_image = np.clip(np_image + noise, 0, 255)  # 노이즈를 원본 이미지에 추가하고 값의 범위는 0~255로 제한
    noisy_image = Image.fromarray(noisy_image.astype(np.uint8))  # 다시 PIL 이미지로 변환
    return noisy_image
    # Example usage

