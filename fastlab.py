import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import fastapi.responses
import numpy
from pydantic import BaseModel
import os
import io
from PIL import Image
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi import Form, File, UploadFile
from typing import List
import matplotlib.pyplot as plt
import hashlib
from PIL import ImageDraw

app = FastAPI()
def sum_two_args(x,y):
 return x+y
# Hello World route
@app.get("/")
def read_root():
 return {"Hello": "World"}
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")
os.makedirs("static/images", exist_ok=True)
@app.get("/some_url/{something}", response_class=HTMLResponse)
async def read_something(request: Request, something: str):
 return templates.TemplateResponse("some.html", {"request": request,
                                   "something": something})
def create_some_image(some_difs):
 imx = 200
 imy = 200
 image = numpy.zeros((imx,imy, 3), dtype=numpy.int8)
 image[0:imy//2,0:imx//2,0] = some_difs
 image[imy//2:,imx//2:,2] = 240
 image[imy//2:,0:imx//2, 1] = 240
 return image
# возврат изображения в виде потока медиаданных по URL
@app.get("/bimage", response_class=fastapi.responses.StreamingResponse)
async def b_image(request: Request):
 # рисуем изображение, сюда можете вставить GAN, WGAN сети и т. д
 # взять изображение из массива в Image PIL
 image = create_some_image(100)
 im = Image.fromarray(image, mode="RGB")
 # сохраняем изображение в буфере оперативной памяти
 imgio = io.BytesIO()
 im.save(imgio, 'JPEG')
 imgio.seek(0)
 # Возвращаем изображение в виде mime типа image/jpeg
 return fastapi.responses.StreamingResponse(content=imgio, media_type="image/jpeg")
 # возврат двух изображений в таблице html, одна ячейка ссылается на url bimage
# другая ячейка указывает на файл из папки static по ссылке
# при этом файл туда предварительно сохраняется после генерации из массива
@app.get("/image", response_class=HTMLResponse)
async def make_image(request: Request):
    image_n = "image.jpg"
    image_dyn = request.base_url.path + "bimage"
    image_st = request.url_for("static", path=f'/{image_n}')
    image = create_some_image(250)
    im = Image.fromarray(image, mode="RGB")
    im.save(f"./static/{image_n}")
    # передаем в шаблон две переменные, к которым сохранили url
    return templates.TemplateResponse("image.html", {"request": request,
                                                     "im_st": image_st, "im_dyn": image_dyn})

@app.post("/image_form", response_class=HTMLResponse)
async def make_image(request: Request,
 name_op:str = Form(),
 number_op:int = Form(),
 r:int = Form(),
 g:int = Form(),
 b:int = Form(),
 files: List[UploadFile] = File(description="Multiple files as UploadFile")):
 # устанавливаем готовность прорисовки файлов, можно здесь проверить, что файлы вообще есть
 # лучше использовать исключения
 ready = False
 print(len(files))
 if(len(files)>0):
    if(len(files[0].filename)>0):
        ready = True
        images = []
        if ready:
            print([file.filename.encode('utf-8') for file in files])
        # преобразуем имена файлов в хеш -строку
        images = ["static/" + hashlib.sha256(file.filename.encode('utf-8')).hexdigest()
                  for file in files]
        # берем содержимое файлов
        content = [await file.read() for file in files]
        # создаем объекты Image типа RGB размером 200 на 200
        p_images = [Image.open(io.BytesIO(con)).convert("RGB").resize((200, 200))
                    for con in content]
        # сохраняем изображения в папке static
        for i in range(len(p_images)):
            draw = ImageDraw.Draw(p_images[i])
        # Рисуем красный эллипс с черной окантовкой
        draw.ellipse((100, 100, 150, 200 + number_op), fill=(r, g, b), outline=(0, 0, 0))
        p_images[i].save("./" + images[i], 'JPEG')
        # возвращаем html с параметрами-ссылками на изображения, которые позже будут
        # извлечены браузером запросами get по указанным ссылкам в img src
    return templates.TemplateResponse("forms.html", {"request": request,
                                                     "ready": ready, "images": images})

class User(BaseModel):
    name: str
    age: int

@app.get('/users/{user_id}')
def get_user(user_id):
    return User(name="John Doe", age=20)

def apply_function(img_array, period, mode, direction):
    h, w, c = img_array.shape
    img_array = img_array.astype(numpy.float32) / 255.
    if direction == "horizontal":
        x = numpy.linspace(0, 2 * numpy.pi, w)
        factor = numpy.sin(x * period) if mode == "sin" else numpy.cos(x * period)
        factor = factor.reshape(1, w, 1)
    else:
        y = numpy.linspace(0, 2 * numpy.pi, h)
        factor = numpy.sin(y * period) if mode == "sin" else numpy.cos(y * period)
        factor = factor.reshape(h, 1, 1 )
    new_img = numpy.clip(img_array * factor, 0, 1)
    return (new_img * 255).astype(numpy.uint8)

def save_plot(image_array, filename):
    plt.figure(figsize=(6, 3))
    plt.hist(image_array.ravel(), bins=256, color='gray', alpha=0.7)
    plt.xlabel("Pixel value")
    plt.ylabel("Frequency")
    plt.title("Color Distribution")
    plt.savefig(filename)
    plt.close()

@app.get("/image_process", response_class=HTMLResponse)
async def image_process_form(request: Request):
    return templates.TemplateResponse("image_process.html",
                                       {"request": request, "original": None, "modified": None, "plot_original": None,
                                        "plot_modified": None})

#Задание для 18 варианта 1 Л.Р.
@app.post("/image_process_task")
async def process_image(
         request: Request,
         file: UploadFile = File(...),
         period: float = Form(...),
         mode: str = Form(...),
         direction: str = Form(...)
 ):
    image = Image.open(io.BytesIO(file.file.read())).convert("RGB")
    img_array = numpy.array(image)
    modified_img_array = apply_function(img_array, period, mode, direction)
    modified_image = Image.fromarray(modified_img_array)
    original_path = f"static/images/original_{file.filename}"
    modified_path = f"static/images/modified_{file.filename}"
    plot_original = f"static/images/original_plot_{file.filename}.png"
    plot_modified = f"static/images/modified_plot_{file.filename}.png"

    image.save(original_path)
    modified_image.save(modified_path)
    save_plot(img_array, plot_original)
    save_plot(modified_img_array, plot_modified)
    return templates.TemplateResponse(
         "image_process.html",
         {
             "request": request,
             "original": original_path,
             "modified": modified_path,
             "plot_original": plot_original,
             "plot_modified": plot_modified,
         }
     )

@app.put('/users/{user_id}')
async def update_user(user_id, user: User):
     # поместите сюда код для обновления данных
     return user


