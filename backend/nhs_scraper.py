import requests
from bs4 import BeautifulSoup

url = "https://www.nhs.uk/conditions/breast-cancer/symptoms/"
response = requests.get(url)
soup = BeautifulSoup(response.content, "html.parser")

# You may need to adjust the selector based on NHS page structure
symptoms = []
for li in soup.select("ul li"):
    text = li.get_text().strip()
    if text and "breast" in text.lower():
        symptoms.append(text)

with open("nhs_symptoms.txt", "w", encoding="utf-8") as f:
    for s in symptoms:
        f.write(s + "\n")

print("Extracted symptoms:", symptoms)
