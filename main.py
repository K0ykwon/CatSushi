import streamlit as st
from openai import OpenAI
from PIL import Image
from dotenv import load_dotenv
import os
import io
from Agent import MedicationAgent
from gtts import gTTS
from audio_recorder_streamlit import audio_recorder

# 환경 변수 로딩
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)
agent = MedicationAgent()

st.title("💊 고양이 초밥")

# 세션 상태 초기화 (세션이 새로 시작된 경우)
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "image_uploaded" not in st.session_state:
    st.session_state.image_uploaded = False

if "question_count" not in st.session_state:
    st.session_state.question_count = 0

if "new_question" not in st.session_state:
    st.session_state.new_question = False

# 인터넷 창 새로고침 또는 접속 시 초기화: 세션 상태 체크
if len(st.session_state.chat_history) == 0 and st.session_state.question_count == 0:
    st.session_state.chat_history = []  # 질문 히스토리 초기화
    st.session_state.question_count = 0  # 질문 수 초기화
    st.session_state.new_question = False  # 새 질문 플래그 초기화
    st.session_state.image_uploaded = False  # 이미지 업로드 상태 초기화
    st.info("새로운 세션을 시작합니다!")

# 이미지 업로드 부분
if not st.session_state.image_uploaded:
    image_file = st.file_uploader("약 이미지 업로드 ", type=["jpg", "jpeg", "png"])
    if image_file:
        image = Image.open(image_file)
        image.save("image.png")
        st.image(image, caption="업로드한 이미지", use_container_width=True)
        st.success("이미지가 저장되었습니다.")
        st.session_state.image_uploaded = True
        st.session_state.new_question = True
else:
    st.image("image.png", caption="현재 이미지", use_container_width=True)

# 질문 루프 트리거
if st.button("➕ 새 질문 추가하기"):
    st.session_state.new_question = True

# 아이에게 약을 먹여도 되는지 물어보는 상황을 처리
if st.session_state.new_question:
    st.markdown(f"### 🎤 질문 {st.session_state.question_count + 1} 녹음")

    # 음성 녹음 컴포넌트
    audio_bytes = audio_recorder(
        energy_threshold=(-1.0, 1.0),
        pause_threshold=10.0,
        sample_rate=16000,
        key=f"recorder_{st.session_state.question_count}"
    )

    if audio_bytes:
        # 녹음 오디오 재생
        st.audio(audio_bytes, format="audio/wav")
        st.info("Whisper STT 수행 중...")

        # BytesIO로 변환
        audio_file = io.BytesIO(audio_bytes)
        audio_file.name = f"question_{st.session_state.question_count}.wav"

        # 기존 STT 처리 로직
        with st.spinner("음성에서 텍스트 추출 중..."):
            transcript = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file
            )
            raw_text = transcript.text
            st.markdown(f"📝 인식된 질문: `{raw_text}`")

        # 질문을 명확히 정리
        with st.spinner("GPT가 질문을 명확하게 정리 중..."):
            clarify_prompt = f"""
            다음은 사용자의 말로 된 질문입니다. 
            사용자는 아이에게 약을 사용해도 괜찮은지에 관한 내용을 질문하거나 아이의 정보를 제공하고 있습니다.
            이를 더 명확하고 간결하게 정리해주세요:
            "{raw_text}"
            """
            gpt_response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "당신은 모호한 질문을 명확히 정리하는 조수입니다."},
                    {"role": "user", "content": clarify_prompt}
                ]
            )
            clarified_text = gpt_response.choices[0].message.content.strip()

        st.markdown(f"🔍 명확한 질문: `{clarified_text}`")
        st.session_state.chat_history.append({"question": clarified_text})

        # 전체 질문 히스토리 생성하여 Agent에 제공 (과거 질문임을 명시)
        full_prompt = ""
        for i, item in enumerate(st.session_state.chat_history):
            if i < st.session_state.question_count:  # 과거 질문만 '과거 질문'으로 레이블링
                full_prompt += f"과거 입력 {i+1}: {item['question']}\n 현재 입력 {i+1}: {item['answer']}\n"
            else:
                full_prompt += f"현재 입력: {item['question']}\n"

        with st.spinner("💊 Agent가 응답 중..."):
            answer = agent("과거의 입력은 참고만 하되, 현재의 입력에 명확하게 답변해주세요."+full_prompt, image_path="image.png")
            st.session_state.chat_history[-1]["answer"] = answer

        st.text_area("🤖 Agent 응답", answer, height=200)

        # TTS 생성
        with st.spinner("🔊 TTS 생성 중..."):
            tts = gTTS(text=answer, lang="ko")
            audio_path = f"response_{st.session_state.question_count}.mp3"
            tts.save(audio_path)
            st.audio(audio_path, format="audio/mp3")

        # 다음 질문을 받을 준비
        st.session_state.question_count += 1
        st.session_state.new_question = False

# 이전 대화 히스토리
if st.session_state.chat_history:
    st.markdown("### 📚 대화 히스토리")
    for i, item in enumerate(st.session_state.chat_history):
        st.markdown(f"**질문 {i+1}:** {item['question']}")
        st.markdown(f"**응답 {i+1}:** {item['answer']}")
