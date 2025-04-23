# RNN-based System Identification and Control

A hands-on project that simulates a mass-spring-damper system and uses simple RNN models to both identify and control it.

## Table of Contents
- [Description](#description)
- [Features](#features)
- [Prerequisites](#prerequisites)
- [Project Structure](#project-structure)
- [Results](#results)
- [Contributing](#contributing)

## Description

This repo is split into two main parts:

1. **System Identification**  
   Train a SimpleRNN model to predict how the system moves (position) when you push on it.
2. **Control**  
   Use that model to build a controller that drives the system toward a target position.

The dynamic system is a mass connected to a spring and damper, driven by different input signals (square, sawtooth, sine).

## Features

- Generate and visualize system responses to square, sawtooth, and sine wave inputs.
- Train a SimpleRNN to learn the plant (mass-spring-damper) behavior.
- Build a classic P controller and a neural-network-based controller.
- End-to-end training of controller + plant in one model.
- Save and load trained models for reuse.

## Prerequisites

- Python 3.8 or higher
- `numpy`, `pandas`, `matplotlib`, `scipy`, `scikit-learn`
- `tensorflow`, `keras`
- `control`

## Project Structure

```
/System Identification.py       # Data generation and RNN training for plant model
/Control.py                     # Load plant model, apply controllers, train final controller
/trained_SRNN_model.keras       # Saved RNN model for system identification
/ctrlxplantfinal.keras          # Saved combined controller + plant model
/ctrl_final.keras               # Saved standalone neural-network controller
/images/                        # Folder to store plot images for README
README.md                       # This file
```

## Results

Here are the main plots illustrating the system identification and control performance:

### System Response to Square Wave Input
![Square Response](images/square_response.png)

### System Response to Sawtooth Wave Input
![Sawtooth Response](images/sawtooth_response.png)

### System Response to Sine Wave Input
![Sine Response](images/sine_response.png)

### Position Control Performance
![Position Control](images/position_control.png)


## Contributing

Feel free to open issues or send pull requests. Contributions like fixing typos, improving docs, or adding new features are always welcome!
