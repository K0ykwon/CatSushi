import requests
import json
from urllib.parse import quote_plus
from openai import OpenAI
from dotenv import load_dotenv
import os

class ParsingFromFDA:
    def __init__(self):
        self.base_url = "https://api.fda.gov/drug/label.json"
        load_dotenv()
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        if not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY not found in .env file")
        self.openai_client = OpenAI(api_key=self.openai_api_key)
        
    def search_drug(self, drug_name: str) -> dict:
        try:
            params = {
                'search': f'openfda.brand_name:"{drug_name}"',
                'limit': 1
            }
            
            
            response = requests.get(
                self.base_url,
                params=params,
                timeout=10
            )
            
            
            if response.status_code == 200:
                data = response.json()
                if data.get('results'):
                    return self._parse_drug_info(data['results'][0])
                else:
                    # 일반명으로도 검색 시도
                    params['search'] = f'openfda.generic_name:"{drug_name}"'
                    response = requests.get(self.base_url, params=params, timeout=10)
                    data = response.json()
                    
                    if data.get('results'):
                        return self._parse_drug_info(data['results'][0])
                    else:
                        return None
            else:
                return None
                
        except Exception as e:
            return None
            
    def _parse_ingredients(self, ingredients: list) -> list:
        try:
            if not ingredients or ingredients[0] == 'Unknown':
                return []
                
            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": """You are an expert in parsing drug ingredient information.
                        Your task is to extract ingredient names and their amounts from the given text.
                        Return the information in the format [ingredient, amount] or [ingredient] if amount is not available.
                        Example: "Acetaminophen 500mg" -> ["Acetaminophen", "500mg"]
                                "Aspirin" -> ["Aspirin"]"""
                    },
                    {
                        "role": "user",
                        "content": str(ingredients)
                    }
                ],
                functions=[
                    {
                        "name": "parse_ingredients",
                        "description": "Parse drug ingredients and their amounts",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "ingredients": {
                                    "type": "array",
                                    "items": {
                                        "type": "array",
                                        "items": {
                                            "type": "string"
                                        },
                                        "minItems": 1,
                                        "maxItems": 2
                                    },
                                    "description": "List of ingredients in format [ingredient, amount] or [ingredient]"
                                }
                            },
                            "required": ["ingredients"]
                        }
                    }
                ],
                function_call={"name": "parse_ingredients"}
            )
            
            result = json.loads(response.choices[0].message.function_call.arguments)
            return result["ingredients"]
            
        except Exception as e:
            return [[ing] for ing in ingredients if ing != 'Unknown']
            
    def _parse_drug_info(self, drug_data: dict) -> dict:
        try:
            info = {
                'active_ingredients': self._parse_ingredients(drug_data.get('active_ingredient', ['Unknown'])),
                'efficacy': drug_data.get('indications_and_usage', ['Unknown'])[0],
                'usage': drug_data.get('dosage_and_administration', ['Unknown'])[0],
                'precautions': drug_data.get('warnings', ['Unknown'])[0]
            }
            return info
        except Exception as e:
            return None

    def __call__(self, drug_name: str) -> dict:
        return self.search_drug(drug_name)
