from Agent import MedicationAgent
if __name__ == "__main__":
    # 사용 예시야!!!!! 이거보구 만들면돼!!!!!!!!!!! 팟띵!!!!
    agent = MedicationAgent()
    user_prompt = "13세 남자아이이고, 체중은 40kg이에요, 열이 심하게 나는데 이 약을 사용하면 괜찮나요? 알레르기는 따로 없어요. parsingwho를 반드시 사용하여 답해주세요"
    print(agent(user_prompt))

