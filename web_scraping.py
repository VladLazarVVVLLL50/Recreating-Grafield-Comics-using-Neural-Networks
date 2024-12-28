import os
import requests
from bs4 import BeautifulSoup
from PIL import Image
from io import BytesIO
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Constante
BASE_URL = "http://pt.jikos.cz/garfield/"
SAVE_DIR = "garfield_comics"
START_YEAR = 1981
END_YEAR = 1985
MAX_COMICS_PER_YEAR = 250

# Verifica daca directorul pentru salvare exista
if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

# Creeaza o sesiune de requests cu retry
def create_session():
    session = requests.Session()
    retries = Retry(total=5, backoff_factor=1, status_forcelist=[500, 502, 503, 504])
    adapter = HTTPAdapter(max_retries=retries)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session

session = create_session()

# Functie pentru a verifica daca o banda este de duminica (pe baza raportului de aspect)
def is_sunday_comic(img_url):
    try:
        response = session.get(img_url, stream=True)
        if response.status_code == 200:
            img = Image.open(BytesIO(response.content))
            width, height = img.size
            aspect_ratio = width / height
            return aspect_ratio < 3.0  # Benzi de duminica sunt mai late
        else:
            print(f"Nu s-a putut accesa imaginea pentru verificare: {img_url}")
    except Exception as e:
        print(f"Eroare la verificarea benzii de duminica: {e}")
    return False

# Functie pentru a descarca o banda
def download_comic(img_url, year, comic_number):
    try:
        response = session.get(img_url, stream=True)
        if response.status_code == 200:
            filename = os.path.join(SAVE_DIR, f"{year}_{comic_number:03d}_{os.path.basename(img_url)}")
            with open(filename, "wb") as f:
                for chunk in response.iter_content(1024):
                    f.write(chunk)
            print(f"Descarcat: {filename}")
        else:
            print(f"Nu s-a putut descarca banda: {img_url}")
    except Exception as e:
        print(f"Eroare la descarcarea benzii: {e}")

# Functie pentru a descarca benzi dintr-o luna specifica
def scrape_month(year, month):
    month_url = f"{BASE_URL}{year}/{month}/"
    comic_count = 0

    print(f"Se descarca benzi pentru luna {month} {year} de la {month_url}")
    response = session.get(month_url)
    if response.status_code != 200:
        print(f"Nu s-a putut accesa: {month_url}")
        return comic_count

    soup = BeautifulSoup(response.content, "html.parser")
    comic_imgs = soup.find_all("img", {"src": True, "alt": True})

    for img_tag in comic_imgs:
        img_url = img_tag["src"]
        alt_text = img_tag["alt"].lower()

        if img_url.startswith("/"):
            img_url = BASE_URL.rstrip("/") + img_url

        # Descarca doar benzi Garfield
        if "garfield" in alt_text and img_url.endswith((".gif", ".jpg", ".png")):
            if not is_sunday_comic(img_url):
                comic_count += 1
                print(f"Banda gasita: {img_url} (alt: {alt_text})")
                download_comic(img_url, year, comic_count)

    return comic_count

# Functie pentru a descarca benzi dintr-un an specific
def scrape_year(year):
    months = [
        "1", "2", "3", "4", "5", "6", 
        "7", "8", "9", "10", "11", "12"
    ]
    total_comics = 0

    for month in months:
        total_comics += scrape_month(year, month)
        if total_comics >= MAX_COMICS_PER_YEAR:
            break

    print(f"Anul {year} finalizat: {total_comics} benzi descarcate.")

# Script principal
for year in range(START_YEAR, END_YEAR + 1):
    print(f"Se incepe descarcarea pentru anul {year}")
    scrape_year(year)

print("Descarcare completa!")
