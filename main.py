from fastapi import FastAPI, Form, UploadFile, File, Request
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from dotenv import dotenv_values
import os

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
def form(request: Request):
    env = dotenv_values(".env")
    google_api_key = env.get("GOOGLE_API_KEY", "")
    telegram_api_key = env.get("TELEGRAM_API_KEY", "")
    uploaded_file = "company_doc.txt" if os.path.exists("uploads/company_doc.txt") else ""

    return templates.TemplateResponse("index.html", {
        "request": request,
        "google_api_key": google_api_key,
        "telegram_api_key": telegram_api_key,
        "uploaded_file": uploaded_file
    })

@app.post("/submit")
async def submit(
    request: Request,
    api_key: str = Form(""),
    telegram_key: str = Form(""),
    knowledge_file: UploadFile = File(None)
):
    env = dotenv_values(".env")
    google_api_key = api_key if api_key else env.get("GOOGLE_API_KEY", "")
    telegram_api_key = telegram_key if telegram_key else env.get("TELEGRAM_API_KEY", "")

    with open(".env", "w") as f:
        f.write(f"GOOGLE_API_KEY={google_api_key}\n")
        f.write(f"TELEGRAM_API_KEY={telegram_api_key}\n")

    if knowledge_file:
        with open("uploads/company_doc.txt", "wb") as f:
            f.write(await knowledge_file.read())

    return templates.TemplateResponse("index.html", {
        "request": request,
        "google_api_key": google_api_key,
        "telegram_api_key": telegram_api_key,
        "uploaded_file": "company_doc.txt" if knowledge_file else "",
        "success": True  # This triggers the success message in the template
    })

@app.get("/download-file")
def download_file():
    return FileResponse("uploads/company_doc.txt", filename="company_doc.txt")

