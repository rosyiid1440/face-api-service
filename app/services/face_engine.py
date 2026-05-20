import face_recognition
import cv2
import numpy as np
import pymysql
import json

class FaceEngine:
    def __init__(self):
        self.db_host = "localhost"
        self.db_user = "root"
        self.db_password = ""
        self.db_name = "face_db"
        
        # Cache RAM dioptimalkan untuk menampung data akademik lengkap
        self.known_face_encodings = []
        self.known_face_ids = []
        self.known_face_names = []
        self.known_face_classes = [] # Tambahkan cache baru khusus kelas
        
        self._load_encodings_from_db()

    def get_db_connection(self):
        return pymysql.connect(
            host=self.db_host,
            user=self.db_user,
            password=self.db_password,
            database=self.db_name,
            cursorclass=pymysql.cursors.DictCursor
        )

    def _load_encodings_from_db(self):
        """Memuat data wajah DAN data kelas dari MySQL ke RAM saat server nyala."""
        try:
            connection = self.get_db_connection()
            with connection.cursor() as cursor:
                # Ambil data wajah sekaligus JOIN data kelas siswa yang aktif
                sql = """
                    SELECT sf.id, sf.student_id, s.name, s.class, s.major, s.class_number, sf.encoding 
                    FROM student_faces sf
                    JOIN students s ON sf.student_id = s.id
                    WHERE s.status = 'aktif'
                """
                cursor.execute(sql)
                rows = cursor.fetchall()
                
                # Kosongkan cache lama sebelum reload
                self.known_face_encodings = []
                self.known_face_ids = []
                self.known_face_names = []
                self.known_face_classes = []
                
                for row in rows:
                    self.known_face_ids.append(row['student_id']) # Simpan Student ID asli
                    self.known_face_names.append(row['name'])
                    
                    # Gabungkan format kelas: XII RPL 1
                    format_kelas = f"{row['class']} {row['major']} {row['class_number']}"
                    self.known_face_classes.append(format_kelas)
                    
                    # Decode data vektor wajah AI
                    encoding_array = np.array(json.loads(row['encoding']))
                    self.known_face_encodings.append(encoding_array)
                    
            print(f"🔥 [RAM CACHE] Berhasil memuat {len(self.known_face_names)} data wajah siswa ke memori.")
        except Exception as e:
            print(f"Gagal memuat database ke cache: {e}")
        finally:
            if 'connection' in locals() and connection.open:
                connection.close()

    def train_new_face(self, image_bytes: bytes, student_id: int) -> bool:
        """Mengekstrak wajah dan menyimpannya ke MySQL berdasarkan student_id."""
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        boxes = face_recognition.face_locations(rgb_img)
        encodings = face_recognition.face_encodings(rgb_img, boxes)
        
        if not encodings:
            return False

        encoding_to_save = encodings[0]
        encoding_json = json.dumps(encoding_to_save.tolist())

        try:
            connection = self.get_db_connection()
            with connection.cursor() as cursor:
                # Menggunakan tabel baru 'student_faces' dan kolom 'student_id'
                sql = "INSERT INTO student_faces (student_id, encoding) VALUES (%s, %s)"
                cursor.execute(sql, (student_id, encoding_json))
                inserted_id = cursor.lastrowid
            connection.commit()
            
            # Ambil data akademik terbaru siswa untuk dimasukkan ke RAM Cache
            with connection.cursor() as cursor:
                sql_get = "SELECT name, class, major, class_number FROM students WHERE id = %s"
                cursor.execute(sql_get, (student_id,))
                student_data = cursor.fetchone()
                
            if student_data:
                # Inject langsung ke dalam Cache RAM aktif
                self.known_face_ids.append(student_id)
                self.known_face_names.append(student_data['name'])
                
                format_kelas = f"{student_data['class']} {student_data['major']} {student_data['class_number']}"
                self.known_face_classes.append(format_kelas)
                self.known_face_encodings.append(encoding_to_save)
            
        except Exception as e:
            print(f"Gagal menyimpan ke MySQL: {e}")
            return False
        finally:
            if 'connection' in locals() and connection.open:
                connection.close()
                
        return True

    def recognize_face(self, image_bytes: bytes) -> dict:
        """Mendeteksi wajah secara instan menggunakan 100% data dari RAM Cache."""
        if not self.known_face_encodings:
            return {"status": "success", "faces_detected": 0, "data": []}

        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        boxes = face_recognition.face_locations(rgb_img)
        encodings = face_recognition.face_encodings(rgb_img, boxes)

        results = []
        for encoding in encodings:
            matches = face_recognition.compare_faces(self.known_face_encodings, encoding, tolerance=0.5)
            face_distances = face_recognition.face_distance(self.known_face_encodings, encoding)
            
            if len(face_distances) == 0:
                continue
                
            best_match_index = np.argmin(face_distances)
            
            if matches[best_match_index]:
                # Ambil data instan langsung dari RAM Cache (Super Cepat!)
                db_id = self.known_face_ids[best_match_index]
                name = self.known_face_names[best_match_index]
                student_class = self.known_face_classes[best_match_index]
                confidence = round((1 - face_distances[best_match_index]) * 100, 2)
                
                results.append({
                    "id": db_id,
                    "name": name,
                    "student_class": student_class,
                    "confidence": f"{confidence}%"
                })
            else:
                results.append({
                    "id": None,
                    "name": "Unknown",
                    "student_class": "-",
                    "confidence": "0%"
                })

        return {"status": "success", "faces_detected": len(results), "data": results}

    def delete_face(self, db_id: int) -> bool:
        """Menghapus wajah dari MySQL dan membuangnya dari Cache RAM."""
        # 1. Hapus dari Database MySQL
        try:
            connection = self.get_db_connection()
            with connection.cursor() as cursor:
                # Cek apakah data ada
                cursor.execute("SELECT id FROM faces WHERE id = %s", (db_id,))
                if not cursor.fetchone():
                    return False
                
                cursor.execute("DELETE FROM faces WHERE id = %s", (db_id,))
            connection.commit()
        except Exception as e:
            print(f"Gagal menghapus dari MySQL: {e}")
            return False
        finally:
            if 'connection' in locals() and connection.open:
                connection.close()

        # 2. Hapus dari Cache RAM (Agar sinkron seketika)
        if db_id in self.known_face_ids:
            index = self.known_face_ids.index(db_id)
            self.known_face_ids.pop(index)
            self.known_face_names.pop(index)
            self.known_face_encodings.pop(index)
            
        return True

face_engine = FaceEngine()