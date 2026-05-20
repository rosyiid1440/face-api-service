from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from app.api.routes import router
import os

app = FastAPI(title="Face Recognition API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

os.makedirs("dataset/known_faces", exist_ok=True)
app.include_router(router, prefix="/api/v1")

# Endpoint Baru: Menyajikan halaman utama kamera untuk HP / Browser
@app.get("/", response_class=HTMLResponse)
def index_page():
    # Membaca file index.html yang berada di dalam folder app/
    html_path = os.path.join(os.path.dirname(__index__ if '__index__' in locals() else __file__), "index.html")
    with open(html_path, "r", encoding="utf-8") as f:
        return f.read()

@app.get("/admin", response_class=HTMLResponse)
def admin_page():
    # Membaca file admin.html yang berada di dalam folder app/
    html_path = os.path.join(os.path.dirname(__index__ if '__index__' in locals() else __file__), "admin.html")
    with open(html_path, "r", encoding="utf-8") as f:
        return f.read()