const tf = require('@tensorflow/tfjs')

const learningRate = 0.1

const input = tf.tensor2d([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])

const output = tf.tensor2d([
    [0],
    [1],
    [1],
    [0]
])

function createNetworkStructure() {
    model = tf.sequential()
    model.add(tf.layers.dense({
        inputShape: [2],
        units: 3,
        activation: 'sigmoid'
    }))
    model.add(tf.layers.dense({
        units: 1,
        activation: 'sigmoid'
    }))
    model.compile({
        optimizer: tf.train.adam(learningRate),
        loss: 'meanSquaredError'
    })
    return model
}

function trainNetwork() {
    return model.fit(input, output, {
        suffle: true,
        epochs: 1000
    })
}

neuralNetwork = createNetworkStructure()

trainNetwork().then( () => {
    result = neuralNetwork.predict(tf.tensor2d([[0, 1]]))
    resultValue = result.dataSync()
    result.dispose()
    resultValue > 0.5 ? resultValue = 1 : resultValue = 0
    console.log(resultValue) 
} )