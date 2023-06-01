import { OpenAI } from 'langchain/llms/openai'
import { OpenAIEmbeddings } from 'langchain/embeddings/openai'
import { HNSWLib } from 'langchain/vectorstores/hnswlib'
import { PDFLoader } from 'langchain/document_loaders/fs/pdf'
import { DocxLoader } from 'langchain/document_loaders/fs/docx'
import { TextLoader } from 'langchain/document_loaders/fs/text'
import { DirectoryLoader } from 'langchain/document_loaders/fs/directory'
import { ConversationalRetrievalQAChain } from 'langchain/chains'
import { RecursiveCharacterTextSplitter } from 'langchain/text_splitter'
import {
  ChatPromptTemplate,
  SystemMessagePromptTemplate,
  HumanMessagePromptTemplate,
} from 'langchain/prompts'
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

const openai = new OpenAI({
  temperature: 0.15,
  modelName: 'gpt-3.5-turbo',
})

const embeddings = new OpenAIEmbeddings({
  modelName: 'text-embedding-ada-002',
})

const memory = new BufferMemory({
  memoryKey: 'chat_history', // Must be set to "chat_history"
})

const loadDocuments = async (directory) => {
  const loader = new DirectoryLoader(directory, {
    '.pdf': (path) => new PDFLoader(path, { splitPages: false }),
    '.txt': (path) => new TextLoader(path),
    '.docx': (path) => new DocxLoader(path),
  })
  const text_splitter = new RecursiveCharacterTextSplitter({
    chunkSize: 256,
    chunkOverlap: 16,
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

const store = await loadStore()
/* 
const systemTemplate = `
  You are a snarky butler.
  Your goal is to find suiting consultants among the CVs provided, Try to respond by giving their first and last names and then why this person is a match.

  % RESPONSE TONE

  - Your tone should be snarky and rude
  
  %RESPONSE FORMAT

  - If a match is found among documents always provide which person it is by saying their full name first.
  - Try to include emojis
  - If you don't know the answer, say that you don't know
  - Answer in Swedish
`

const systemMessagePrompt =
  SystemMessagePromptTemplate.fromTemplate(systemTemplate)

const humanTemplate = '{query}'
const humanPrompt = HumanMessagePromptTemplate.fromTemplate(humanTemplate)
const chatPrompt = ChatPromptTemplate.fromPromptMessages([
  systemMessagePrompt,
  humanPrompt,
]) */

const qaChain2 = ConversationalRetrievalQAChain.fromLLM(
  openai,
  store.asRetriever(8),
  {
    verbose: false,
    returnSourceDocuments: false,
    memory: memory,
  }
)

const run = async (query) => {
  return qaChain2.call({
    question: query,
  })
}
