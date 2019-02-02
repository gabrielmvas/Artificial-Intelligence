let xValues = []
let yValues = []

let a, b, c

let dragging = false

const learningRate = 0.05
const optimizer = tf.train.adam(learningRate)

function setup() {
    createCanvas(400, 400)
    background(0)

    a = tf.variable(tf.scalar(random(-1, 1)))
    b = tf.variable(tf.scalar(random(-1, 1)))
    c = tf.variable(tf.scalar(random(-1, 1)))
}

function mousePressed() {
    dragging = true
}
function mouseReleased() {
    dragging = false
}

/* function mouseDragged() {
    
} */

function draw() {

    if (dragging) {
        const x = map(mouseX, 0, width, -1, 1)
        const y = map(mouseY, 0, height, 1, -1)
        xValues.push(x)
        yValues.push(y)
    } else {
        tf.tidy(() => {
            if (xValues.length > 0) {
                const tensorYs = tf.tensor1d(yValues)
                optimizer.minimize(() => loss(predict(xValues), tensorYs))
            }
        })
    }
    
    background(0)
    stroke(255)
    strokeWeight(8)

    for (let i = 0; i < xValues.length; i++) {
        let px = map(xValues[i], -1, 1, 0, width)
        let py = map(yValues[i], -1, 1, height, 0)
        point(px, py)
    }

    const curveX = []
    for (let x = -1; x < 1.01; x += 0.05) {
        curveX.push(x)
    }

    const ys = tf.tidy(() => predict(curveX) )
    let curveY = ys.dataSync()
    ys.dispose()

    beginShape()
    noFill()
    stroke(255)
    strokeWeight(2)
    for (let i = 0; i < curveX.length; i++) {
        let x = map(curveX[i], -1, 1, 0, width)
        let y = map(curveY[i], -1, 1, height, 0)
        vertex(x, y)
    }
    endShape()
    
    //console.log(tf.memory().numTensors)
}

function loss(predictions, labels) {
    return predictions.sub(labels).square().mean()
}

function predict(points) {
    const tensorXs = tf.tensor1d(points)
    // y = ax^2 + bx + c
    const ys = tensorXs.square().mul(a).add(tensorXs.mul(b)).add(c)
    return ys
}
