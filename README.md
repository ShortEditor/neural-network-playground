# 🧠 Neural Network Playground

An interactive neural network simulator that runs entirely in your browser. Build a network, press **Train**, and watch it learn in real time - decision boundary, weights, activations and loss all update live.

**Live demo:** https://neural-network-playground.vercel.app

## What you can do

- **Shape the network** - add/remove hidden layers (up to 4), add/remove neurons per layer (1-8)
- **Pick input features** - X₁, X₂, X₁², X₂², X₁·X₂, sin(X₁), sin(X₂)
- **Tune hyperparameters** - activation (Tanh, ReLU, Leaky ReLU, Sigmoid), learning rate, L1/L2 regularization, batch size
- **Choose a dataset** - Circle, XOR, Moons, Spiral, Blobs, with noise and train/test split sliders
- **Watch it learn** - the background is the live decision boundary, edge color/thickness shows each weight's sign and magnitude, neuron fill shows its activation, and the loss chart tracks train vs test loss
- **Hover any neuron** to inspect its activation and bias

## How it works

It's a real multilayer perceptron trained with **backpropagation** and **mini-batch gradient descent**, written from scratch in plain JavaScript. No frameworks, no libraries, no build step, no server - just `index.html`, `style.css` and `app.js`.

- Output layer: sigmoid with binary cross-entropy loss
- Weights: Xavier-style initialization
- Rendering: three canvases (decision boundary + data, network graph, loss chart)

## Run it locally

Open `index.html` in a browser, or serve the folder:

```bash
npx serve .
```

## Deploy

It's a static site - drop it on Vercel, Netlify or GitHub Pages as-is. No build command needed.

---

Built by [Ganesh (ShortEditor)](https://github.com/ShortEditor)
