with open("joseito.txt","r",encoding="utf-8") as f:
    text = f.read()

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 100,
    chunk_overlap = 0,
)
texts = text_splitter.split_text(text)

#print(type(texts))
#print(len(texts))
#print(texts[0])
#print(texts[1])

embeddings = HuggingFaceEmbeddings(
    model_name = "intfloat/multilingual-e5-large",
    model_kwargs = {"device":"cuda:0"},
)