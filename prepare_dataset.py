import os
import cv2
import pandas as pd
import random

video_root = "DAiSEE/DataSet/Train"
label_file = "DAiSEE/Labels/TrainLabels.csv"

output_root = "dataset"
attentive_path = os.path.join(output_root, "attentive")
not_attentive_path = os.path.join(output_root, "not_attentive")

os.makedirs(attentive_path, exist_ok=True)
os.makedirs(not_attentive_path, exist_ok=True)

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

df = pd.read_csv(label_file)
label_dict = dict(zip(df['ClipID'], df['Engagement']))

fps_extract = 5
TARGET_PER_CLASS = 5000

attentive_count = 0
not_attentive_count = 0

for person in os.listdir(video_root):

    person_path = os.path.join(video_root, person)

    for clip_folder in os.listdir(person_path):
        clip_path = os.path.join(person_path, clip_folder)

        for file in os.listdir(clip_path):
            if not file.endswith(".avi") and not file.endswith(".mp4"):
                continue

            video_path = os.path.join(clip_path, file)
            clip_id = file 
            
            if clip_id not in label_dict:
                continue

            engagement = label_dict[clip_id]
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = 0

            print(f"Processing {clip_id}...")
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                if int(frame_count % (fps / fps_extract)) == 0:
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
                    if len(faces) > 0:
                        # take largest face
                        x, y, w, h = max(faces, key=lambda b: b[2]*b[3])
                        face = frame[y:y+h, x:x+w]
                        face = cv2.resize(face, (160, 160))

                        if engagement >= 2:
                            if attentive_count >= TARGET_PER_CLASS:
                                break
                            save_dir = attentive_path
                            filename = f"A_{attentive_count}.jpg"
                            attentive_count += 1
                        else:
                            if not_attentive_count >= TARGET_PER_CLASS:
                                break
                            save_dir = not_attentive_path
                            filename = f"N_{not_attentive_count}.jpg"
                            not_attentive_count += 1
                        
                        cv2.imwrite(os.path.join(save_dir, filename), face)
                frame_count += 1
            cap.release()
            
    if attentive_count >= TARGET_PER_CLASS and not_attentive_count >= TARGET_PER_CLASS:
        break

# Data Augmentation to balance classes
def augment_image(img):
    ops = []
    ops.append(cv2.flip(img, 1))

    bright = cv2.convertScaleAbs(img, alpha=1.2, beta=20)
    ops.append(bright)

    blur = cv2.GaussianBlur(img, (5, 5), 0)
    ops.append(blur)
    return random.choice(ops)


def balance_folder(folder, target_count):
    files = os.listdir(folder)

    while len(files) < target_count:
        file = random.choice(files)
        img = cv2.imread(os.path.join(folder, file))

        if img is None:
            continue

        aug = augment_image(img)
        new_name = f"aug_{len(files)}.jpg"
        cv2.imwrite(os.path.join(folder, new_name), aug)
        files.append(new_name)

balance_folder(attentive_path, TARGET_PER_CLASS)
balance_folder(not_attentive_path, TARGET_PER_CLASS)

print("Final Dataset:")
print("Attentive:", len(os.listdir(attentive_path)))
print("Not Attentive:", len(os.listdir(not_attentive_path)))