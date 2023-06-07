import { OpenAI } from 'langchain/llms/openai'
import { FaissStore } from 'langchain/vectorstores/faiss'
import { OpenAIEmbeddings } from 'langchain/embeddings/openai'
import { CSVLoader } from 'langchain/document_loaders/fs/csv'
import { loadQAMapReduceChain } from 'langchain/chains'
import { HuggingFaceInferenceEmbeddings } from 'langchain/embeddings/hf'
import { ConversationalRetrievalQAChain } from 'langchain/chains'
import { BufferMemory } from 'langchain/memory'

import dotenv from 'dotenv'
import http from 'http'
import url from 'url'
import fs from 'fs'

dotenv.config()
const openai = new OpenAI({ temperature: 0 })
const memory = new BufferMemory({
  memoryKey: 'chat_history', // Must be set to "chat_history"
})

const chain = loadQAMapReduceChain(openai, {
  returnIntermediateSteps: true,
  verbose: true,
})

const loadStore = async () => {
  /* const embeddings = new OpenAIEmbeddings({ modelName: 'text-embedding-ada-002' }) */
  const embeddings = new HuggingFaceInferenceEmbeddings({
    model: 'diptanuc/all-mpnet-base-v2',
  })
  return FaissStore.loadFromPython('../hfmodel/faiss_index', embeddings)
}

console.log('Loading vector store')
const store = await loadStore()
console.log('Vector store loaded')

const run = async (query) => {
  const relevantDocs = await store.similaritySearch(query, 4)
  const result = await chain.call({
    input_documents: relevantDocs,
    question: `
      Answer as shortly as possible without omitting any Examination Type information 
      
      What exam type is recommended if the description of the anamnes is as follows?
      ${query}`,
  })
  console.log(result)
  return result

  /* return memChain.call({
    question: `${query}`,
  }) */
}

console.log('Starting server')
//Simplest node server
const server = http
  .createServer(async (req, res) => {
    if (req.url === '/examtypes/index.html') {
      res.statusCode = 200
      res.writeHead(200, { 'content-type': 'text/html; charset=utf-8' })
      fs.createReadStream('index.html').pipe(res)
    } else if (req.url.startsWith('/examtypes/query?input=')) {
      res.setHeader('Content-Type', 'text/plain; charset=utf-8')

      var url_parts = url.parse(req.url, true)
      var query = url_parts.query

      var result = await run(query.input)
      res.statusCode = 200
      res.end(result.text)
    } else {
      res.statusCode = 501
      res.end()
    }
  })
  .listen(8123, '127.0.0.1', () => {
    console.log('Server started. Listening on "127.0.0.1:8123')
  })
