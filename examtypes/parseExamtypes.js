import { OpenAI } from 'langchain/llms/openai'
import { OpenAIEmbeddings } from 'langchain/embeddings/openai'
import { HuggingFaceInferenceEmbeddings } from 'langchain/embeddings/hf'
import { HNSWLib } from 'langchain/vectorstores/hnswlib'
import { CSVLoader } from 'langchain/document_loaders/fs/csv'
import { loadQAMapReduceChain } from 'langchain/chains'

import { ConversationalRetrievalQAChain } from 'langchain/chains'
import { BufferMemory } from 'langchain/memory'

import dotenv from 'dotenv'
import http from 'http'
import url from 'url'
import fs from 'fs'

dotenv.config()
const storeName = './examtypes-store'
const openai = new OpenAI({ temperature: 0 })
const memory = new BufferMemory({
  memoryKey: 'chat_history', // Must be set to "chat_history"
})
/* const embeddings = new OpenAIEmbeddings({ modelName: 'text-embedding-ada-002' }) */
const embeddings = new HuggingFaceInferenceEmbeddings({
  model: 'facebook/dino-vitb16',
  maxRetries: 3,
})
const chain = loadQAMapReduceChain(openai, {
  returnIntermediateSteps: true,
  verbose: true,
})

const loadDocuments = async () => {
  const loader = new CSVLoader('./anamneses.csv')
  const docs = await loader.load()
  return docs
}

const createStore = async () => {
  var docs = await loadDocuments()
  var store = await HNSWLib.fromDocuments(docs, embeddings)
  console.log('Store created')
  store.save(storeName)

  return store
}

const loadStore = async () => {
  console.log('Loading vector store')
  return HNSWLib.load(storeName, embeddings).catch(() => {
    console.log('Store did not exist. Creating store...')
    return createStore()
  })
}

const store = await loadStore()

const memChain = ConversationalRetrievalQAChain.fromLLM(
  openai,
  store.asRetriever(8),
  {
    verbose: false,
    returnSourceDocuments: false,
    memory: memory,
  }
)

await memChain.call({
  question: `You are a bot that will receive descriptions of hospital patients and their symptoms.
            Try to find types of examinations to conduct for the patients anamnes if possible. 
            If you can't find a suiting exam type then say "I don't know"`,
})

const run = async (query) => {
  /* const result = await chain.call({
    input_documents: await store.similaritySearch(query),
    question: `
      Answer as shortly as possible without omitting any Examination Type information 
      
      What exam type is recommended if the description of the anamnes is as follows?
      ${query}`,
  })
  console.log(result)
  return result */

  return memChain.call({
    question: `${query}`,
  })
}

//Simplest node server
const server = http
  .createServer(async (req, res) => {
    res.statusCode = 200

    if (req.url === '/examtypes/index.html') {
      res.writeHead(200, { 'content-type': 'text/html; charset=utf-8' })
      fs.createReadStream('index.html').pipe(res)
    } else if (req.url.startsWith('/examtypes/query?input=')) {
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
