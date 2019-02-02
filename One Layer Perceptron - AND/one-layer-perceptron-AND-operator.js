let input = [[0, 0], [0, 1], [1, 0], [1, 1]]
let output = [0, 0, 0, 1]
let weights = [0.0, 0.0]
let learningRate = 0.05

function stepFunction(sum) {
    return sum >= 1 ? 1 : 0
}

function calculateOutputFor(registry) {
    let sum = 0
    for (let i = 0; i < registry.length; i++) {
        sum += registry[i] * weights[i]
    }
    return stepFunction(sum)
}

function trainNetwork() {
    let totalError = 1
    while ( totalError != 0) {
        totalError = 0 
        for (let i = 0; i < input.length; i++) {
            //console.log(input[i])
            let calculatedOutput = calculateOutputFor(input[i])
            let error = Math.abs(output[i] - calculatedOutput)
            totalError += error
            for (let j = 0; j < weights.length; j++) {
                weights[j] = weights[j] + (learningRate * input[i][j] * error)
            }          
        }        
    }
}

function printWeights() {
    console.log('Optimal weights: ')
    weights.forEach( (elem, index) => {
        console.log('w' + (index + 1) + '->' + elem)
    })
}

function printOutputs() {
    console.log('Outputs: ')
    input.forEach( (elem) => {
        console.log(elem + ' -> ' + calculateOutputFor(elem))
    })
}

trainNetwork()
console.log('Network successfully trained!')
printWeights()
printOutputs()
