# -*- coding: utf-8 -*-

import requests
from bs4 import BeautifulSoup

url = "https://www.aozora.gr.jp/cards/000035/files/275_13903.html"

response = requests.get(url)

soup = BeautifulSoup(response.content,"html.parser")

for script_or_style in soup(["script","style"]):
    script_or_style.extract()

text = soup.get_text()

lines = (line.strip() for line in  text.splitlines())

chunks = (phrase.strip() for line in lines for phrase in line.split("  "))

text = "\n".join(chunk for chunk in chunks if chunk)

outtext = "joseito.txt"
with open(outtext,"w",encoding="utf-8") as file:
    file.write(text)

print(url,"の内容を",outtext,"に出力しました")