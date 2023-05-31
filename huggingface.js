/* import { Chroma } from 'langchain/vectorstores/chroma'*/
import { OpenAIEmbeddings } from 'langchain/embeddings/openai'
import { HuggingFaceInferenceEmbeddings } from 'langchain/embeddings/hf'
import { HNSWLib } from 'langchain/vectorstores/hnswlib'
//import { Chroma } from 'langchain/vectorstores/chroma'
import { CSVLoader } from 'langchain/document_loaders/fs/csv'
import { RecursiveCharacterTextSplitter } from 'langchain/text_splitter'

import dotenv from 'dotenv'
dotenv.config()

const hnswStoreDir = './hnswstore'

const embeddings = new OpenAIEmbeddings({
  modelName: 'text-embedding-ada-002',
})

/* const embeddings = new HuggingFaceInferenceEmbeddings({
  model: 'birgermoell/roberta-swedish',
  //apiKey: 'YOUR-API-KEY', // In Node.js defaults to process.env.HUGGINGFACEHUB_API_KEY
})
 */
export const createVectorStore = async () => {
  // Create docs with a loader
  const loader = new CSVLoader('./inputshort.csv')

  const allDocs = await loader.load()

  //console.log(value)
  const textSplit = new RecursiveCharacterTextSplitter({
    chunkSize: 2000,
    chunkOverlap: 0,
  })

  const docs = await textSplit.splitDocuments(allDocs)

  // Create vector store and index the docs using openai
  const vectorStore = await HNSWLib.fromDocuments(
    docs,
    embeddings /* , {
      collectionName: 'a-test-collection-openai',
      url: 'http://127.0.0.1:8000/',
    } */
  )

  vectorStore.save(hnswStoreDir)

  return vectorStore
}

export const loadVectorStore = async () => {
  try {
    return await HNSWLib.load(hnswStoreDir, embeddings)
  } catch {
    return await createVectorStore()
  }
}

const vectorStore = await loadVectorStore()

var result = await vectorStore.similaritySearch(
  'MC olycka, kraftig blödning i vänster ben samt frakturer i skallen',
  2
)
console.log(result)
