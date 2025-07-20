from fastapi import FastAPI, Form, UploadFile, File, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import os

app = FastAPI()

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Load HTML templates
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
def form(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/submit")
async def submit(
    api_key: str = Form(...),
    telegram_key: str = Form(...),
    knowledge_file: UploadFile = File(...)
):
    with open(".env", "w") as f:
        f.write(f"GOOGLE_API_KEY={api_key}\n")
        f.write(f"TELEGRAM_API_KEY={telegram_key}\n")
    with open("uploads/company_doc.txt", "wb") as f:
        f.write(await knowledge_file.read())
    return {"message": "Configuration saved successfully by Veridata."}
