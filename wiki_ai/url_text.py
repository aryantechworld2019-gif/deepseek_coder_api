# import re
# import requests
# from bs4 import BeautifulSoup

# prompt = "how to use excel basics opearation help content url https://argoslabs.atlassian.net/wiki/spaces/LCpy/pages/1247739/Excel+Basics"

# url_match = re.search(r'https?://[^\s]+', prompt)

# if url_match:
#     url = url_match.group(0)
#     print("✅ URL found:", url)

#     try:
#         res = requests.get(url, timeout=10)
#         res.raise_for_status()

#         print(res.text)

#         soup = BeautifulSoup(res.text, 'html.parser')
#         for tag in soup(['script', 'style']):
#             tag.decompose()

#         content = soup.get_text(separator=' ', strip=True)
#         print("\n📝 Extracted Text:\n", content[:500])  # Preview first 500 chars

#     except Exception as e:
#         print("❌ Error fetching content:", e)

# else:
#     print("❌ No URL found in prompt.")

# from selenium import webdriver
# from selenium.webdriver.chrome.options import Options
# from webdriver_manager.chrome import ChromeDriverManager
# from bs4 import BeautifulSoup
# import time

# def extract_dynamic_content(url):
#     chrome_options = Options()
#     chrome_options.add_argument("--headless")
#     chrome_options.add_argument("--disable-gpu")
#     chrome_options.add_argument("--no-sandbox")

#     driver = webdriver.Chrome(ChromeDriverManager().install(), options=chrome_options)
#     driver.get(url)

#     time.sleep(3)  # wait for JS to render

#     soup = BeautifulSoup(driver.page_source, 'html.parser')
#     driver.quit()

#     for tag in soup(['script', 'style']):
#         tag.decompose()

#     text = soup.get_text(separator=' ', strip=True)
#     return ' '.join(text.split())[:2000]

# # Test it
# url = "https://argoslabs.atlassian.net/wiki/spaces/LCpy/pages/1247739/Excel+Basics"
# print("📝 Extracted:", extract_dynamic_content(url))

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup
import time

def extract_dynamic_content(url):
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--no-sandbox")

    # Fix: Use 'Service' object instead of passing executable path directly
    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=chrome_options)

    driver.get(url)
    time.sleep(3)  # Wait for JS to load/render

    soup = BeautifulSoup(driver.page_source, 'html.parser')
    driver.quit()

    for tag in soup(['script', 'style']):
        tag.decompose()

    text = soup.get_text(separator=' ', strip=True)
    return ' '.join(text.split())
    # [:2000]  # Truncate if needed

def read_links(file_path="links.txt"):
    try:
        with open(file_path, "r") as f:
            links = [line.strip() for line in f if line.strip()]
        return links
    except FileNotFoundError:
        return []

print(
    
)



url = "https://argoslabs.atlassian.net/wiki/spaces/LCpy/pages/1272956/Dynamic+Python-3.402.913"
print("📝 Extracted:", extract_dynamic_content(url))

