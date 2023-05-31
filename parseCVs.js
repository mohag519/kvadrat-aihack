import { OpenAI } from 'langchain/llms/openai'
import { OpenAIEmbeddings } from 'langchain/embeddings/openai'
import { HNSWLib } from 'langchain/vectorstores/hnswlib'
import { PDFLoader } from 'langchain/document_loaders/fs/pdf'
import { DocxLoader } from 'langchain/document_loaders/fs/docx'
import { TextLoader } from 'langchain/document_loaders/fs/text'
import { DirectoryLoader } from 'langchain/document_loaders/fs/directory'
import { ConversationalRetrievalQAChain } from 'langchain/chains'
import { RecursiveCharacterTextSplitter } from 'langchain/text_splitter'
import { BufferMemory } from 'langchain/memory'
import dotenv from 'dotenv'
import http from 'http'
import url from 'url'
import fs from 'fs'

dotenv.config()
const server = http
  .createServer(async (req, res) => {
    res.statusCode = 200

    if (req.url === '/index.html') {
      res.writeHead(200, { 'content-type': 'text/html; charset=utf-8' })
      fs.createReadStream('index.html').pipe(res)
    } else if (req.url.startsWith('/query?input=')) {
      res.setHeader('Content-Type', 'text/plain; charset=utf-8')

      var url_parts = url.parse(req.url, true)
      var query = url_parts.query

      var result = await run(query.input)
      res.end(result.text)
    } else {
      res.statusCode = 501
      res.end()
    }
  })
  .listen(8123, '127.0.0.1', () => {})

const embeddings = new OpenAIEmbeddings({
  modelName: 'text-embedding-ada-002',
})

const loadDocuments = async (directory) => {
  const loader = new DirectoryLoader(directory, {
    '.pdf': (path) => new PDFLoader(path, { splitPages: false }),
    '.txt': (path) => new TextLoader(path),
    '.docx': (path) => new DocxLoader(path),
  })
  const text_splitter = new RecursiveCharacterTextSplitter({
    chunkSize: 1000,
    chunkOverlap: 10,
  })
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

  var qaChain2 = ConversationalRetrievalQAChain.fromLLM(
    openai,
    store.asRetriever(4),
    {
      verbose: false,
      returnSourceDocuments: false,
      memory: new BufferMemory({
        memoryKey: 'chat_history', // Must be set to "chat_history"
      }),
    }
  )

  return qaChain2.call({
    question: query,
  })
}

/*

  const store = await loadStore()
  const retriever = await store.asRetriever(1)
  const qaChain = new RetrievalQAChain()
  return retriever.getRelevantDocuments('Vem är bra på Kubernetes?') */

/* var stdin = process.openStdin()
stdin.addListener('data', function (d) {
  // note:  d is an object, and when converted to a string it will
  // end with a linefeed.  so we (rather crudely) account for that
  // with toString() and then trim()
  run(d.toString().trim()).then((result) => {
    console.log(result)
  })
}) */
