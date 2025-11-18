from fastapi import FastAPI, File, UploadFile, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import os
from PIL import Image, ImageDraw, ImageFont
import io
import base64
import matplotlib.pyplot as plt
import numpy as np
import random
import string
import time

# Создаем приложение
app = FastAPI(title="Image Blender")

# Исправляем путь к шаблонам
templates = Jinja2Templates(directory=".")

# Хранилище для CAPTCHA
captcha_store = {}

@app.get("/style.css")
async def get_css():
    """Отдаем CSS файл напрямую"""
    try:
        with open("style.css", "r", encoding="utf-8") as f:
            css_content = f.read()
        return HTMLResponse(content=css_content, media_type="text/css")
    except FileNotFoundError:
        return HTMLResponse(content="/* CSS file not found */", media_type="text/css")

def generate_captcha():
    """Генерация CAPTCHA"""
    text = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(4))
    
    # Создаем изображение CAPTCHA
    width, height = 150, 50
    image = Image.new('RGB', (width, height), color='white')
    draw = ImageDraw.Draw(image)
    
    try:
        font = ImageFont.truetype("arial.ttf", 24)
    except:
        try:
            font = ImageFont.truetype("cour.ttf", 24)
        except:
            font = ImageFont.load_default()
    
    # Рисуем текст
    draw.text((20, 10), text, font=font, fill='black')
    
    # Добавляем шум
    for _ in range(50):
        x = random.randint(0, width)
        y = random.randint(0, height)
        draw.point((x, y), fill='gray')
    
    # Конвертируем в base64
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    
    # Сохраняем CAPTCHA
    captcha_id = str(time.time())
    captcha_store[captcha_id] = text
    
    return captcha_id, f"data:image/png;base64,{img_str}"

def verify_captcha(captcha_id: str, user_input: str) -> bool:
    """Проверка CAPTCHA"""
    if captcha_id in captcha_store:
        correct_text = captcha_store[captcha_id]
        del captcha_store[captcha_id]
        return user_input.upper() == correct_text.upper()
    return False

def blend_images(img1, img2, blend_level):
    """Смешивание двух изображений"""
    # Приводим к RGB и одинаковому размеру
    img1 = img1.convert('RGB')
    img2 = img2.convert('RGB')
    
    size1 = img1.size
    size2 = img2.size
    target_size = (min(size1[0], size2[0]), min(size1[1], size2[1]))
    
    img1 = img1.resize(target_size, Image.LANCZOS)
    img2 = img2.resize(target_size, Image.LANCZOS)
    
    # Конвертируем в numpy и смешиваем
    arr1 = np.array(img1, dtype=np.float32)
    arr2 = np.array(img2, dtype=np.float32)
    blended = arr1 * blend_level + arr2 * (1 - blend_level)
    blended = np.clip(blended, 0, 255).astype(np.uint8)
    
    return Image.fromarray(blended)

def create_histogram(image, title):
    """Создание гистограммы цветов"""
    image = image.convert('RGB')
    img_array = np.array(image)
    
    # Используем agg backend для matplotlib чтобы избежать GUI ошибок
    plt.switch_backend('Agg')
    
    # Создаем гистограмму
    plt.figure(figsize=(8, 4))
    colors = ['red', 'green', 'blue']
    color_names = ['Red', 'Green', 'Blue']
    
    for i, color in enumerate(colors):
        hist, bins = np.histogram(img_array[:,:,i].flatten(), bins=128, range=[0,256])
        plt.plot(bins[:-1], hist, color=color, alpha=0.7, label=color_names[i])
    
    plt.title(f'Гистограмма - {title}')
    plt.xlabel('Значение цвета')
    plt.ylabel('Частота')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Конвертируем в изображение
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=80, bbox_inches='tight')
    buf.seek(0)
    plt.close()
    
    return Image.open(buf)

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Главная страница"""
    captcha_id, captcha_image = generate_captcha()
    return templates.TemplateResponse("index.html", {
        "request": request,
        "captcha_id": captcha_id,
        "captcha_image": captcha_image
    })

@app.post("/blend", response_class=HTMLResponse)
async def blend_images_endpoint(
    request: Request,
    image1: UploadFile = File(...),
    image2: UploadFile = File(...),
    blend_level: float = Form(...),
    captcha_solution: str = Form(...),
    captcha_id: str = Form(...)
):
    """Обработка смешивания изображений"""
    # Проверка CAPTCHA
    if not verify_captcha(captcha_id, captcha_solution):
        new_captcha_id, new_captcha_image = generate_captcha()
        return templates.TemplateResponse("index.html", {
            "request": request,
            "error": "Неверная CAPTCHA! Попробуйте еще раз.",
            "captcha_id": new_captcha_id,
            "captcha_image": new_captcha_image
        })
    
    try:
        # Читаем и проверяем изображения
        img1_data = await image1.read()
        img2_data = await image2.read()
        
        img1 = Image.open(io.BytesIO(img1_data))
        img2 = Image.open(io.BytesIO(img2_data))
        
        # Смешиваем изображения
        blended_img = blend_images(img1, img2, blend_level)
        
        # Создаем гистограммы
        hist1 = create_histogram(img1, "Изображение 1")
        hist2 = create_histogram(img2, "Изображение 2")
        hist_blended = create_histogram(blended_img, "Результат")
        
        # Конвертируем в base64 для отображения в HTML
        def img_to_base64(img):
            buffered = io.BytesIO()
            img.save(buffered, format="PNG")
            return base64.b64encode(buffered.getvalue()).decode()
        
        result_data = {
            "request": request,
            "blend_level": blend_level,
            "image1": img_to_base64(img1),
            "image2": img_to_base64(img2),
            "blended_image": img_to_base64(blended_img),
            "hist1": img_to_base64(hist1),
            "hist2": img_to_base64(hist2),
            "hist_blended": img_to_base64(hist_blended)
        }
        
    except Exception as e:
        new_captcha_id, new_captcha_image = generate_captcha()
        return templates.TemplateResponse("index.html", {
            "request": request,
            "error": f"Ошибка обработки изображений: {str(e)}",
            "captcha_id": new_captcha_id,
            "captcha_image": new_captcha_image
        })
    
    # Генерируем новую CAPTCHA для следующего запроса
    new_captcha_id, new_captcha_image = generate_captcha()
    result_data["captcha_id"] = new_captcha_id
    result_data["captcha_image"] = new_captcha_image
    
    return templates.TemplateResponse("result.html", result_data)

@app.get("/health")
async def health_check():
    """Эндпоинт для проверки здоровья приложения"""
    return {"status": "healthy", "message": "Image Blender is running"}

# Для совместимости с Render.com
if __name__ == "__main__":
    import uvicorn
    import os
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port, reload=False)
