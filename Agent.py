import json
from openai import OpenAI
from dotenv import load_dotenv
import os
from ParsingFromFDA import ParsingFromFDA
from ParsingFromWHO import ParsingFromWHO
from DrugLabelExtractor import DrugLabelExtractor  # DrugLabelExtractor 클래스 임포트

class MedicationAgent:
    def __init__(self):
        load_dotenv()
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY not found in environment.")

        self.client = OpenAI(api_key=self.api_key)
        self.fda_parser = ParsingFromFDA()
        self.who_parser = ParsingFromWHO()

        self.drug_label_extractor = DrugLabelExtractor()

        self.system_prompt = """
        당신은 전문 약사이자 의료 지식 전문가입니다. 아래 정보를 기반으로 사용자의 아이에게 약물 사용이 적절한지에 대한 정보를 제공해 주세요.
        - 기본적으로 extracted_info를 사용해 정보를 구성해 주세요.
        - 만약 빠진 정보나 추가적으로 필요한 정보가 있으면 다음과 같이 두 함수를 자유롭게 이용할 수 있습니다.

            1. get_fda_info:
            - FDA 라벨 데이터베이스에서 약물의 성분, 효능, 용법, 주의사항 등의 정보를 제공합니다.
            - 약물이 항생제가 아니거나, 일반적인 약물이라면 이 정보를 **최우선적으로** 활용하세요.

            2. get_who_info:
            - 항생제의 경우, WHO가 제공하는 표준 용량 정보를 조회합니다.
            - 약물이 항생제이고, FDA 정보가 부족하거나 명확하지 않을 경우에만 **보조적으로** 사용하세요.
        
        
        답변 스타일:
        - 300자 이내로 대답해주되, 사용자에 따라 사용하기 적절한지 아닌지 여부부터 사용 방법의 순서로 먼저 두괄식으로 정리해 주세요.
        - 특수문자는 절대 사용하면 안됩니다. 또한, 답변은 하나의 자연스러운 단락으로 줄글로 작성하세요. 
        - 사용자 질문에 따라 필요한 정보만 간결하고 정확하게 전달해 주세요."""

        self.functions = [
            {
                "name": "get_fda_info",
                "description": (
                    "FDA의 약물 라벨 데이터베이스에서 해당 약물의 정보를 검색합니다. "
                    "성분, 효능, 용법, 주의사항 등 방대한 정보를 제공합니다. "
                    "가능한 경우 이 정보를 기본으로 사용하세요."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "drug_name": {
                            "type": "string",
                            "description": "검색할 약물 이름 (예: Tylenol, Ibuprofen)"
                        }
                    },
                    "required": ["drug_name"]
                }
            },
            {
                "name": "get_who_info",
                "description": (
                    "WHO에서 제공하는 항생제에 대한 표준 용량 정보를 조회합니다. "
                    "약물이 항생제이고 FDA 정보가 부족할 경우에만 보조적으로 사용하세요."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "antibiotic_name": {
                            "type": "string",
                            "description": "검색할 항생제 이름 (예: Amikacin, Gentamicin)"
                        }
                    },
                    "required": ["antibiotic_name"]
                }
            }
        ]

    def __call__(self, user_prompt: str):
        image_path = 'image.jpeg' 
        return self.run(image_path, user_prompt)
    def get_fda_info(self, drug_name: str):
        return self.fda_parser(drug_name)

    def get_who_info(self, antibiotic_name: str):
        return self.who_parser(antibiotic_name)

    def _query_openai_for_additional_info(self, extracted_info: dict, user_prompt: str):
        prompt = f"""
        주어진 약물 정보와 사용자의 질문을 바탕으로 필요한 추가 정보를 다음 중에서 고르고, 그 정보가 무엇인지 알려주세요.
        
        - 약물 정보: {json.dumps(extracted_info, ensure_ascii=False)}
        - 질문: {user_prompt}
        - 필요한 정보:
            - 나이
            - 체중
            - 증상
            - 알레르기 정보 (꼭 필요한 경우)        
        이 외의 정보는 요청하지 않도록 하세요.

        만약 위에서 언급한 추가적인 정보가 모두 제공되었거나, 필요하지 않다면 필요한 추가 정보가 없다고 해 주세요.
        """

        response = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt + "300토큰 이내로 대답해줘."}],
            max_tokens=300,
            temperature=0.7
        )

        additional_info_needed = response.choices[0].message.content.strip()
        return additional_info_needed

    def extract_drug_info(self, image_path: str):
        extracted_info = self.drug_label_extractor(image_path)
        return extracted_info

    def run(self, image_path: str, user_prompt: str):
        extracted_info = self.extract_drug_info(image_path)

        additional_info_request = self._query_openai_for_additional_info(extracted_info, user_prompt)


        messages = [
            {"role": "system", "content": self.system_prompt+'\n\n사용자에게 요청해야 할 추가적인 정보:'+ additional_info_request+'(제가 방금 추가적인 정보가 필요 없다고 했으면 그냥 무시하고 답변 생성에 반영하지 말아 주세요.)'+\
             "약물 기본 정보:" + json.dumps(extracted_info, ensure_ascii=False) },
            {"role": "user", "content": user_prompt},
        ]

        response = self.client.chat.completions.create(
            model="gpt-4",
            messages=messages,
            functions=self.functions,
            function_call="auto"
        )

        message = response.choices[0].message

        if message.function_call:
            function_name = message.function_call.name
            arguments = json.loads(message.function_call.arguments)

            if function_name == "get_fda_info":
                result = self.get_fda_info(arguments["drug_name"])
            elif function_name == "get_who_info":
                result = self.get_who_info(arguments["antibiotic_name"])
            else:
                result = {}

            follow_up = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[*messages, {"role": "function", "name": function_name, "content": json.dumps(result, ensure_ascii=False)}]
            )

            return follow_up.choices[0].message.content

        return message.content
