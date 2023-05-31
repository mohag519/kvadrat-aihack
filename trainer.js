import fs from 'fs'
import { parse } from 'csv-parse'
import brain from 'brain.js'

const net = new brain.recurrent.LSTM()

new Promise((res, rej) => {
  const trainData = []
  fs.createReadStream('./BookShort.csv')
    .pipe(parse({ delimiter: ';', from_line: 2, quote: true }))
    .on('data', function (row) {
      trainData.push({ input: row[0], output: row[1] })
    })
    .on('end', () => {
      console.log('Finished reading file.')
      console.log(trainData)
      res(trainData)
    })
    .on('error', (err) => {
      console.log(err)
      rej(err)
    })
})
  .then((result) => {
    console.log('Starting to train')
    net.train(result)
    console.log('Training done. Saving to Function')
    fs.writeFileSync(
      'trained-net.cjs',
      `module.exports.trainedNet = ${net.toFunction().toString()};`
    )
  })
  .catch((reason) => console.log(reason))
