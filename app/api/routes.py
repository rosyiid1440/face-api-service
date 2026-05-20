from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Depends, Security, status
from fastapi.security.api_key import APIKeyHeader
from app.services.face_engine import face_engine
from pydantic import BaseModel
import pymysql

router = APIRouter()

# 1. Tentukan nama Header dan Kunci Rahasia Anda
API_KEY_NAME = "X-API-Key"
# Di aplikasi profesional, nilai ini biasanya diambil dari environment variable (os.environ)
SANGAT_RAHASIA_KEY = "KunciRahasiaMicroserviceWajah123!"

api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

# 2. Buat fungsi pengecekan kunci (Dependency Injection)
async def verifikasi_api_key(header_key: str = Security(api_key_header)):
    if header_key == SANGAT_RAHASIA_KEY:
        return header_key
    
    raise HTTPException(
        status_code=status.HTTP_403_FORBIDDEN,
        detail="Akses Ditolak: API Key tidak valid atau tidak ditemukan di Header"
    )

# 3. Kunci Endpoint Tambah Wajah (Tambahkan Depends)
@router.post("/train")
async def upload_training_data(
    student_id: int = Form(...), # Sekarang menerima student_id (angka), bukan name lagi
    file: UploadFile = File(...),
    _ : str = Depends(verifikasi_api_key)
):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File harus berupa gambar")
        
    image_bytes = await file.read()
    
    # Kirim student_id ke face_engine
    success = face_engine.train_new_face(image_bytes, student_id)
    
    if not success:
        raise HTTPException(status_code=400, detail="Tidak ada wajah terdeteksi atau gagal menyimpan")
        
    return {"status": "success", "message": f"Data wajah untuk Student ID {student_id} berhasil ditambahkan."}
    
class StudentCreate(BaseModel):
    nisn: str
    name: str
    gender: str
    student_class: str
    major: str
    class_number: str

@router.post("/students")
async def create_student_and_get_id(student: StudentCreate):
    # Akses koneksi database langsung dari face_engine untuk menghemat baris kode
    connection = face_engine.get_db_connection()
    try:
        with connection.cursor() as cursor:
            # Pastikan NISN belum terdaftar
            cursor.execute("SELECT id FROM students WHERE nisn = %s", (student.nisn,))
            if cursor.fetchone():
                raise HTTPException(status_code=400, detail="Siswa dengan NISN ini sudah terdaftar.")

            # Insert data siswa baru
            sql = """
                INSERT INTO students (nisn, name, gender, class, major, class_number, status) 
                VALUES (%s, %s, %s, %s, %s, %s, 'aktif')
            """
            cursor.execute(sql, (
                student.nisn, student.name, student.gender, 
                student.student_class, student.major, student.class_number
            ))
            student_id = cursor.lastrowid
        connection.commit()
        return {"status": "success", "student_id": student_id, "message": "Profil siswa berhasil dibuat."}
    except pymysql.MySQLError as e:
        raise HTTPException(status_code=500, detail=f"Database error: {e}")
    finally:
        if connection.open:
            connection.close()

# 4. Biarkan Endpoint Deteksi Tetap Terbuka (Public) demi keamanan frontend
@router.post("/recognize")
async def recognize_uploaded_face(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File harus berupa gambar")
        
    image_bytes = await file.read()
    result = face_engine.recognize_face(image_bytes)
    return result

# 5. Kunci Endpoint Hapus Wajah (Tambahkan Depends)
@router.delete("/faces/{face_id}")
async def delete_face_data(
    face_id: int,
    _ : str = Depends(verifikasi_api_key) # Mengunci rute ini
):
    success = face_engine.delete_face(face_id)
    
    if not success:
        raise HTTPException(status_code=404, detail=f"Wajah ID {face_id} tidak ditemukan")
        
    return {"status": "success", "message": f"Data wajah ID {face_id} berhasil dihapus."}