from agents import Agent, Runner, set_tracing_disabled
from dotenv import load_dotenv
# from litellm import completion
from groq import Groq
import base64
import os

# Load biến môi trường từ file .env
load_dotenv()

# Tắt tracing để tránh cảnh báo OPENAI_API_KEY
set_tracing_disabled(True)
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

# Lấy mô hình từ biến môi trường hoặc mặc định
# GROQ_LLM_MODEL = os.getenv("litellm/groq/llama3-70b-8192")
# GROQ_VISION_MODEL = os.getenv("litellm/groq/llama-3.2-11b-vision-preview")
GROQ_LLM_MODEL = "litellm/groq/llama3-70b-8192"
GROQ_VISION_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"

# chuyển đổi ảnh thành base64
def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')

# Định nghĩa model cho Vision Agent
def vision_agent(image_path_input, instructions, model):
    base64_image = encode_image(image_path_input)
    text_prompt = instructions
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", 
                   "content": [
                      {"type": "text", "text": text_prompt}, 
                      {"type": "image_url", 
                       "image_url": {"url": f"data:image/jpg;base64,{base64_image}"}
                    #     "image_url": {
                    #     "url": "https://upload.wikimedia.org/wikipedia/commons/f/f2/LPU-v1-die.jpg"
                    # }
                       }]
                       }],
        temperature=1,
        max_completion_tokens=1024,
        top_p=1,
        stream=False,
        stop=None,
    )
    return response.choices[0].message.content

# Định nghĩa các agent ngôn ngữ

# agent cho các câu hỏi không tìm thấy
notf_agent = Agent(
    name="Not Found Agent",
    instructions="Notification of question not found.",
    model=GROQ_LLM_MODEL,
)

# agent cho các câu hỏi đã tìm thấy
found_agent = Agent(
    name="Found Agent",
    instructions="JUST give 1 or more corresponding answers for each question and keep the language of the question and answer intact",
    model=GROQ_LLM_MODEL,
)

# agent cho quyết định chuyển giao
triage_agent = Agent(
    name="Triage Agent",
    instructions="Chuyển giao cho agent phù hợp.",
    model=GROQ_LLM_MODEL,
    handoffs=[notf_agent, found_agent],
)



def main():
    # Chup ảnh man hình

    # Test với một ảnh mẫu
    image_path_input = "LPU-v1-die.jpg"
    # Chạy vision agent
    input_data = vision_agent(image_path_input=image_path_input,
                        instructions="Find all the questions and answers below the question in the image. Just give the question and answer, no need to answer the question. If there is no question, report that the question cannot be found.",
                        model=GROQ_VISION_MODEL)
    
    # kiem tra xem có câu hỏi nào không
    # print(input_data)
    # Chạy triage agent
    result = Runner.run_sync(triage_agent, input_data)
    print("Kết quả:", result.final_output)
if __name__ == "__main__":
    main()
