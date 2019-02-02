let xValues = []
let yValues = []

let a, b

const learningRate = 0.5
const optimizer = tf.train.sgd(learningRate)

function setup() {
    createCanvas(400, 400)
    background(0)

    a = tf.variable(tf.scalar(random(1)))
    b = tf.variable(tf.scalar(random(1)))
}

function mousePressed() {
    const x = map(mouseX, 0, width, -1, 1)
    const y = map(mouseY, 0, height, 1, -1)
    xValues.push(x)
    yValues.push(y)
}

function draw() {

    tf.tidy(() => {
        if (xValues.length > 0) {
            const tensorYs = tf.tensor1d(yValues)
            optimizer.minimize(() => loss(predict(xValues), tensorYs))
        }
    })

    background(0)
    stroke(255)
    strokeWeight(8)

    for (let i = 0; i < xValues.length; i++) {
        let px = map(xValues[i], -1, 1, 0, width)
        let py = map(yValues[i], -1, 1, height, 0)
        point(px, py)
    }

    const lineX = [-1, 1]
    const ys = tf.tidy(() => predict(lineX) )
    let lineY = ys.dataSync()
    ys.dispose()

    let x1 = map(lineX[0], -1, 1, 0, width)
    let x2 = map(lineX[1], -1, 1, 0, width)

    let y1 = map(lineY[0], -1, 1, height, 0)
    let y2 = map(lineY[1], -1, 1, height, 0)

    strokeWeight(2)
    line(x1, y1, x2, y2)

    //console.log(tf.memory().numTensors)
}

function loss(predictions, labels) {
    return predictions.sub(labels).square().mean()
}

function predict(points) {
    const tensorXs = tf.tensor1d(points)
    // y = ax + b
    const ys = tensorXs.mul(a).add(b)
    return ys
}
