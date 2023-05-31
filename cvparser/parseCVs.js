import { OpenAI } from 'langchain/llms/openai'
import { OpenAIEmbeddings } from 'langchain/embeddings/openai'
import { HNSWLib } from 'langchain/vectorstores/hnswlib'
import { PDFLoader } from 'langchain/document_loaders/fs/pdf'
import { DocxLoader } from 'langchain/document_loaders/fs/docx'
import { TextLoader } from 'langchain/document_loaders/fs/text'
import { DirectoryLoader } from 'langchain/document_loaders/fs/directory'
import { loadQAStuffChain } from 'langchain/chains'
import { RetrievalQAChain } from 'langchain/chains'
import { RecursiveCharacterTextSplitter } from 'langchain/text_splitter'
import dotenv from 'dotenv'

dotenv.config()

const embeddings = new OpenAIEmbeddings({
  modelName: 'text-embedding-ada-002',
})

/* const embeddings = new HuggingFaceInferenceEmbeddings({
  model: 'birgermoell/roberta-swedish',
  //apiKey: 'YOUR-API-KEY', // In Node.js defaults to process.env.HUGGINGFACEHUB_API_KEY
}) */

const loadDocuments = async (directory) => {
  const loader = new DirectoryLoader(directory, {
    '.pdf': (path) => new PDFLoader(path, { splitPages: false }),
    '.txt': (path) => new TextLoader(path),
    '.docx': (path) => new DocxLoader(path),
  })
  const text_splitter = new RecursiveCharacterTextSplitter({ chunkSize: 1000 })
  return text_splitter.splitDocuments(await loader.load())
}

const createStore = async () => {
  var docs = await loadDocuments('./cvs')
  var store = await HNSWLib.fromDocuments(docs, embeddings)

  store.save('./cv-store')

  return store
}

const loadStore = async () => {
  return HNSWLib.load('./cv-store', embeddings).catch(() => {
    return createStore()
  })
}

const run = async (query) => {
  var openai = new OpenAI()

  var store = await loadStore()

  var qaChain2 = RetrievalQAChain.fromLLM(openai, store.asRetriever(4), {
    verbose: false,
    returnSourceDocuments: true,
  })

  return qaChain2.call({
    query: query,
  })
}

/*

  const store = await loadStore()
  const retriever = await store.asRetriever(1)
  const qaChain = new RetrievalQAChain()
  return retriever.getRelevantDocuments('Vem är bra på Kubernetes?') */

var stdin = process.openStdin()
stdin.addListener('data', function (d) {
  // note:  d is an object, and when converted to a string it will
  // end with a linefeed.  so we (rather crudely) account for that
  // with toString() and then trim()
  run(d.toString().trim()).then((result) => {
    console.log(result)
  })
})
