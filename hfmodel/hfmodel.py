from langchain.document_loaders.csv_loader import CSVLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS

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

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = FAISS.from_documents(data, embeddings)
db.save_local("faiss_index")
