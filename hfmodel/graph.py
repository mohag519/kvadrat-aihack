from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from nomic import atlas
import numpy as np
from sentence_transformers import SentenceTransformer
from langchain.document_loaders.csv_loader import CSVLoader


# sentences = ["This is an example sentence", "Each sentence is converted"]
loader = CSVLoader(
    file_path="../examtypes/anamneses.csv",
    csv_args={
        "delimiter": ",",
        "quotechar": '"',
        "fieldnames": ["Anamnes", "Examination Type"],
    },
    encoding="utf8",
)
data = loader.load()
data.pop(0)

sentences = []
for index, x in enumerate(data):
    sentences.append(x.page_content)

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
embeddings = model.encode(sentences)

plotdata = [
    {"sentence": sentences[i % len(sentences)], "id": i} for i in range(len(embeddings))
]

project = atlas.map_embeddings(
    embeddings=embeddings,
    data=plotdata,
    name="flawless-cofactor",
    id_field="id",
    topic_label_field="sentence",
    reset_project_if_exists=True,
)
print("Done")
