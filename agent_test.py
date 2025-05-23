from agents import Agent, Runner, set_tracing_disabled
from dotenv import load_dotenv
import os



# Tắt tracing để tránh cảnh báo OPENAI_API_KEY
set_tracing_disabled(True)

# Load biến môi trường từ file .env
load_dotenv()

# Lấy mô hình từ biến môi trường hoặc mặc định
MODEL = os.getenv("LLM_MODEL", "litellm/groq/llama3-8b-8192")

# Định nghĩa các agent ngôn ngữ
spanish_agent = Agent(
    name="Spanish Agent",
    instructions="Respond in Spanish to user queries.",
    model=MODEL,
)

english_agent = Agent(
    name="English Agent",
    instructions="Respond in English to user queries.",
    model=MODEL,
)

# Định nghĩa triage agent để chuyển giao dựa trên ngôn ngữ
triage_agent = Agent(
    name="Triage Agent",
    instructions="Analyze the language of the request and hand off to the appropriate agent (Spanish or English).",
    model=MODEL,
    handoffs=[spanish_agent, english_agent],
)

def main():
    # Chạy triage agent với một yêu cầu mẫu
    user_input = "What is the weather today?"
    result = Runner.run_sync(triage_agent, user_input)
    print("Kết quả:", result.final_output)

if __name__ == "__main__":
    main()
