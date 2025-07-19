from fastapi import FastAPI, Form, UploadFile, File
from fastapi.responses import HTMLResponse
import os

app = FastAPI()

@app.get("/", response_class=HTMLResponse)
def form():
    return """
    <form action="/submit" method="post" enctype="multipart/form-data">
        <label>Gemini API Key:</label><br>
        <input type="text" name="api_key" /><br><br>
        <label>Upload Base Knowledge File:</label><br>
        <input type="file" name="knowledge_file" /><br><br>
        <input type="submit" />
    </form>
    """

@app.post("/submit")
async def submit(api_key: str = Form(...), knowledge_file: UploadFile = File(...)):
    with open(".env", "w") as f:
        f.write(f"GEMINI_API_KEY={api_key}\n")
    with open("company_doc.txt", "wb") as f:
        f.write(await knowledge_file.read())
    return {"message": "Configuration saved successfully."}
