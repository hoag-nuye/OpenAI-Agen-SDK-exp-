import tkinter as tk
from PIL import ImageGrab
from agents import Agent, Runner, set_tracing_disabled
from dotenv import load_dotenv
from groq import Groq
import base64
import os
from io import BytesIO

# Load biến môi trường từ file .env
load_dotenv()
set_tracing_disabled(True)
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

# Định nghĩa các mô hình
GROQ_LLM_MODEL = "litellm/groq/llama3-70b-8192"
GROQ_VISION_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"

# Hàm mã hóa ảnh từ đường dẫn file
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

# Hàm mã hóa ảnh từ đối tượng PIL Image
def encode_image_pil(image):
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

# Hàm vision_agent hỗ trợ cả đường dẫn file và đối tượng PIL Image
def vision_agent(image_input, instructions, model):
    if isinstance(image_input, str):  # Nếu là đường dẫn file
        base64_image = encode_image(image_input)
    else:  # Nếu là đối tượng PIL Image
        base64_image = encode_image_pil(image_input)
    text_prompt = instructions
    response = client.chat.completions.create(
        model=model,
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": text_prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/jpg;base64,{base64_image}"}}
            ]
        }],
        temperature=1,
        max_completion_tokens=1024,
        top_p=1,
        stream=False,
        stop=None,
    )
    return response.choices[0].message.content

# Định nghĩa các agent
notf_agent = Agent(
    name="Not Found Agent",
    instructions="Notification of question not found.",
    model=GROQ_LLM_MODEL,
)

found_agent = Agent(
    name="Found Agent",
    instructions="JUST give 1 or more corresponding answers for each question and keep the language of the question and answer intact",
    model=GROQ_LLM_MODEL,
)

triage_agent = Agent(
    name="Triage Agent",
    instructions="Chuyển giao cho agent phù hợp.",
    model=GROQ_LLM_MODEL,
    handoffs=[notf_agent, found_agent],
)

# Lớp ứng dụng
class ScreenshotApp:
    def __init__(self, root):
        self.root = root
        # Tạo cửa sổ nút
        self.button_window = tk.Toplevel(root)
        self.button_window.overrideredirect(True)  # Xóa viền cửa sổ
        self.button_window.attributes('-topmost', True)  # Luôn trên cùng
        self.button_window.geometry('60x60+100+100')  # Kích thước 60x60 pixel
        self.canvas = tk.Canvas(self.button_window, width=60, height=60, bg='white', highlightthickness=0)
        self.canvas.pack()
        self.canvas.create_oval(5, 5, 55, 55, fill='green')  # Vẽ hình tròn xanh
        self.button_window.attributes('-transparentcolor', 'white')

        # Bind sự kiện drag cho toàn window
        self.button_window.bind("<Button-1>", self.start_drag)
        self.button_window.bind("<B1-Motion>", self.on_drag)

        # Thêm nút tắt ứng dụng (hình tròn đỏ nhỏ)
        self.close_button = tk.Canvas(self.button_window, width=20, height=20, bg='white', highlightthickness=0)
        self.close_button.create_oval(0, 0, 20, 20, fill='red')
        self.close_button.place(x=40, y=0)  # Đặt gần button xanh
        self.close_button.bind('<Button-1>', self.close_app)

        # Thêm nút chup man hinh (hình tròn xanh la cay)
        self.snap_button = tk.Canvas(self.button_window, width=20, height=20, bg='white', highlightthickness=0)
        self.snap_button.create_oval(0, 0, 20, 20, fill='blue')
        self.snap_button.place(x=0, y=0)  # Đặt gần button xanh
        self.snap_button.bind('<Button-1>', self.on_button_click)  # Bind sự kiện nhấp chuột để chụp ảnh

    def start_drag(self, event):
        # Lưu toạ độ chuột (trên màn hình) và toạ độ window ban đầu
        self.drag_start_x = event.x_root
        self.drag_start_y = event.y_root
        self.win_start_x = self.button_window.winfo_x()
        self.win_start_y = self.button_window.winfo_y()
    
    def on_drag(self, event):
        # Tính delta và cập nhật vị trí window
        dx = event.x_root - self.drag_start_x
        dy = event.y_root - self.drag_start_y
        new_x = self.win_start_x + dx
        new_y = self.win_start_y + dy
        self.button_window.geometry(f"+{new_x}+{new_y}")
        self.button_window.update_idletasks()

    def stop_drag(self, event):
        self.is_dragging = False

    def close_app(self, event):
        self.root.quit()

    def on_button_click(self, event):
        # Chụp màn hình
        screenshot = ImageGrab.grab()
        # Gọi vision_agent với ảnh chụp màn hình
        input_data = vision_agent(
            screenshot,
            "Find all the questions and answers below the question in the image. Just give the question and answer, no need to answer the question. If there is no question, report that the question cannot be found.",
            GROQ_VISION_MODEL
        )
        # Chạy triage_agent
        result = Runner.run_sync(triage_agent, input_data)
        # Hiển thị kết quả
        self.display_output(result.final_output)

    def display_output(self, text):
        # Tạo cửa sổ kết quả
        output_window = tk.Toplevel(self.root)
        output_window.title("Answer")
        # Định vị bên cạnh nút
        button_x = self.button_window.winfo_x()
        button_y = self.button_window.winfo_y()
        button_width = self.button_window.winfo_width()
        output_window.geometry(f"400x300+{button_x + button_width + 10}+{button_y}")
        # Thêm widget Text để hiển thị kết quả
        text_widget = tk.Text(output_window, wrap=tk.WORD)
        text_widget.insert(tk.END, text)
        text_widget.config(state=tk.DISABLED)  # Chỉ đọc
        text_widget.pack(expand=True, fill='both')
        # Thêm nút "x" để đóng
        close_button = tk.Button(output_window, text='x', command=output_window.destroy)
        close_button.place(relx=1.0, rely=0.0, anchor='ne')

# Chạy ứng dụng
if __name__ == "__main__":
    root = tk.Tk()
    root.withdraw()  # Ẩn cửa sổ gốc
    app = ScreenshotApp(root)
    # # Bind sự kiện nhấp chuột cho button xanh
    # app.canvas.bind('<Button-1>', app.on_button_click)
    root.mainloop()