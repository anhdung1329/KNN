# ==============================================================================
# BÀI TẬP LỚN FACE ID - SCRIPT CHẠY SONG SONG TRÊN NHIỀU MÁY (SHARDING)
# ==============================================================================

# --- 1. CẤU HÌNH MÁY (BẠN CHỈ CẦN SỬA DÒNG NÀY) ---
TOTAL_MACHINES = 5      # Tổng số máy bạn định dùng
CURRENT_PART = 0        # <--- SỬA SỐ NÀY: Máy 1 điền 0, Máy 2 điền 1, ..., Máy 5 điền 4

# --- 2. CÀI ĐẶT MÔI TRƯỜNG ---
import os
import time
import math
import glob
import random
import numpy as np
from tqdm import tqdm

# Kết nối Google Drive để lưu kết quả
from google.colab import drive
drive.mount('/content/drive')

# Cài thư viện
print("⏳ Đang cài đặt thư viện...")
!pip install -q deepface kagglehub

from deepface import DeepFace
import kagglehub

# Tạo thư mục lưu trữ trên Drive
SAVE_DIR = '/content/drive/MyDrive/Colab Notebooks/Hoc May CH37/BTL'
if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)
    print(f"✅ Đã tạo/tìm thấy thư mục lưu trữ: {SAVE_DIR}")

# --- 3. TẢI DATASET TỰ ĐỘNG ---
print("\n⏳ Đang tải Dataset từ Kaggle (có thể mất 1-2 phút)...")
try:
    path = kagglehub.dataset_download("yakhyokhuja/ms1m-arcface-dataset")
    print("✅ Dataset đã tải tại:", path)

    # Xử lý đường dẫn (Fix lỗi cấu trúc thư mục)
    possible_path = os.path.join(path, "ms1m-arcface")
    if os.path.exists(possible_path):
        DATASET_ROOT = possible_path
    else:
        DATASET_ROOT = path # Fallback
    print(f"📂 Thư mục gốc chứa ảnh là: {DATASET_ROOT}")

except Exception as e:
    print(f"❌ Lỗi tải Dataset: {e}")
    # Dừng chương trình nếu không có data
    raise e

# --- 4. CHIA DỮ LIỆU (SHARDING LOGIC) ---
print(f"\n🤖 ĐANG KHỞI TẠO MÁY SỐ {CURRENT_PART + 1} / {TOTAL_MACHINES}...")

# Lấy tất cả thư mục ID và SẮP XẾP (Bắt buộc sort để đồng bộ giữa các máy)
all_folders = sorted(glob.glob(os.path.join(DATASET_ROOT, "*")))
total_ids = len(all_folders)

if total_ids == 0:
    raise ValueError("❌ Không tìm thấy thư mục ảnh nào! Kiểm tra lại đường dẫn.")

# Tính toán phần việc của máy này
chunk_size = math.ceil(total_ids / TOTAL_MACHINES)
start_index = CURRENT_PART * chunk_size
end_index = min((CURRENT_PART + 1) * chunk_size, total_ids)

my_folders = all_folders[start_index : end_index]

print(f"📌 NHIỆM VỤ: Xử lý từ ID thứ {start_index} đến {end_index}")
print(f"📊 Tổng số ID máy này cần làm: {len(my_folders)}")

# --- 5. LỌC ẢNH VÀ CHUẨN BỊ LIST ---
MIN_IMGS = 6
GALLERY_SIZE = 5

gallery_paths, gallery_labels = [], []
probe_paths, probe_labels = [], []

print("⏳ Đang quét và lọc ảnh...")
for folder_path in my_folders:
    img_files = glob.glob(os.path.join(folder_path, "*.jpg"))

    if len(img_files) >= MIN_IMGS:
        id_name = os.path.basename(folder_path)
        # Shuffle để lấy ngẫu nhiên
        random.shuffle(img_files)

        # Gallery: Lấy đúng 5 ảnh
        g_imgs = img_files[:GALLERY_SIZE]
        gallery_paths.extend(g_imgs)
        gallery_labels.extend([id_name] * len(g_imgs))

        # Probe: Lấy tối đa 2 ảnh còn lại để test (Tránh lấy quá nhiều gây chậm)
        p_imgs = img_files[GALLERY_SIZE : GALLERY_SIZE + 2]
        probe_paths.extend(p_imgs)
        probe_labels.extend([id_name] * len(p_imgs))

print(f"✅ Đã chuẩn bị xong list ảnh:")
print(f"   - Gallery: {len(gallery_paths)} ảnh")
print(f"   - Probe:   {len(probe_paths)} ảnh")
print(f"   - Tổng cộng: {len(gallery_paths) + len(probe_paths)} ảnh")
print(f"⏱️ Ước tính thời gian chạy: ~{(len(gallery_paths) + len(probe_paths))/36000:.1f} giờ (với tốc độ 10it/s)")

# --- 6. HÀM TẠO VECTOR VÀ LƯU ---
def l2_normalize(x):
    norm = np.linalg.norm(x)
    return x / norm if norm != 0 else x

def process_and_save(img_paths, labels, name_prefix):
    if len(img_paths) == 0: return

    print(f"\n🚀 Đang xử lý tập {name_prefix} (Part {CURRENT_PART})...")
    vectors = []
    valid_labels = []

    # Batch processing loop
    for path, label in tqdm(zip(img_paths, labels), total=len(img_paths)):
        try:
            # --- CORE LOGIC ---
            # detector_backend='skip' : Tăng tốc tối đa
            obj = DeepFace.represent(
                img_path=path,
                model_name="ArcFace",
                enforce_detection=False,
                detector_backend="skip"
            )
            vec = obj[0]["embedding"]
            vectors.append(l2_normalize(np.array(vec)))
            valid_labels.append(label)
        except:
            continue

    # Lưu file
    save_name_vec = f'{name_prefix}_vectors_part_{CURRENT_PART}.npy'
    save_name_lbl = f'{name_prefix}_labels_part_{CURRENT_PART}.npy'

    np.save(os.path.join(SAVE_DIR, save_name_vec), np.array(vectors))
    np.save(os.path.join(SAVE_DIR, save_name_lbl), np.array(valid_labels))

    print(f"🎉 ĐÃ LƯU THÀNH CÔNG: {save_name_vec}")

# --- 7. CHẠY THỰC TẾ ---
# Chạy Probe trước (Nhanh - để test)
process_and_save(probe_paths, probe_labels, "probe")

# Chạy Gallery sau (Lâu)
print("\n💤 Nghỉ 5 giây trước khi chạy Gallery...")
time.sleep(5)
process_and_save(gallery_paths, gallery_labels, "gallery")

print("\n🎯 MÁY NÀY ĐÃ HOÀN TẤT! BẠN CÓ THỂ ĐÓNG TAB.")