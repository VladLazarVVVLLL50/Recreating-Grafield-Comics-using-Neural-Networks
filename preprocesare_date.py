import os
import numpy as np
import cv2
from tqdm import tqdm

DIRECTOR_IMAGINI = "garfield_comics"
LATIME_IMAGINE = 768
INALTIME_IMAGINE = 256
FOLOSESTE_MARGINI = False
ESTE_DIM_MICA = False
if ESTE_DIM_MICA:
    LATIME_IMAGINE //= 4
    INALTIME_IMAGINE //= 4

toate_comicurile = []
nr_comicuri = 0

print("Uploadarea comicurilor...")
for file in tqdm(os.listdir(DIRECTOR_IMAGINI)):
    cale = os.path.join(DIRECTOR_IMAGINI, file)

    # incarcam doar imagini valide
    if not file.lower().endswith(('bmp', 'gif', 'png', 'jpg', 'jpeg')):
        print(f"Fisierul nu este imagine: {file}")
        continue

    # incarcam imaginea folosind OpenCV
    img = cv2.imread(cale)
    if img is None:
        print(f"Eroare la incarcarea imaginii: {file}")
        continue

    if len(img.shape) != 3 or img.shape[2] != 3:
        print(f"Imaginea nu este RGB (3 canale de culoare): {file}")
        continue

    ratio = float(img.shape[1]) / float(img.shape[0])
    if ratio < 3.17 or ratio > 3.60:
        print(f"Imaginea nu are proportia potrivita (3 panouri): {file}, proportie: {ratio}")
        continue

    # redimensionare imagine
    img = cv2.resize(img, (LATIME_IMAGINE, INALTIME_IMAGINE), interpolation=cv2.INTER_LINEAR)

    # detectare margini 
    if FOLOSESTE_MARGINI:
        img = np.where(np.amax(img, axis=2) < 128, 255, 0).astype(np.uint8)

    # impartire in 3 panouri
    panel_1 = img[:, 0 * LATIME_IMAGINE // 3 : 1 * LATIME_IMAGINE // 3]
    panel_2 = img[:, 1 * LATIME_IMAGINE // 3 : 2 * LATIME_IMAGINE // 3]
    panel_3 = img[:, 2 * LATIME_IMAGINE // 3 : 3 * LATIME_IMAGINE // 3]

    if FOLOSESTE_MARGINI:
        comic_curent = np.empty((3, 1, INALTIME_IMAGINE, LATIME_IMAGINE // 3), dtype=np.uint8)
        comic_curent[0][0] = panel_1
        comic_curent[1][0] = panel_2
        comic_curent[2][0] = panel_3
    else:
        comic_curent = np.empty((3, 3, INALTIME_IMAGINE, LATIME_IMAGINE // 3), dtype=np.uint8)
        comic_curent[0] = np.transpose(panel_1, (2, 0, 1))
        comic_curent[1] = np.transpose(panel_2, (2, 0, 1))
        comic_curent[2] = np.transpose(panel_3, (2, 0, 1))

    toate_comicurile.append(comic_curent)
    nr_comicuri += 1

print(f"Am uploadat {nr_comicuri} comicuri!")

# salvare in fisier .npy
if nr_comicuri > 0:
    print("Salvare...")
    os.makedirs("data", exist_ok=True)

    nume_fisier = "data/comicuri"
    if FOLOSESTE_MARGINI:
        nume_fisier += "_margini"
    if ESTE_DIM_MICA:
        nume_fisier += "_mic"
    nume_fisier += ".npy"

    np.save(nume_fisier, np.stack(toate_comicurile, axis=0))
    print(f"Salvare completa in {nume_fisier}.")
else:
    print("Nicio imagine nu a fost procesata!")
