RNN-based System Identification and Control

A hands-on project that simulates a mass-spring-damper system and uses simple RNN models to both identify and control it.

Table of Contents

Description

Features

Prerequisites

Installation

Project Structure

Usage

Results

Contributing

License

Description

This repo is split into two main parts:

System IdentificationTrain a SimpleRNN model to predict how the system moves (position) when you push on it.

ControlUse that model to build a controller that drives the system toward a target position.

The dynamic system is a mass connected to a spring and damper, driven by different input signals (square, sawtooth, sine).

Features

Generate and visualize system responses to square, sawtooth, and sine wave inputs.

Train a SimpleRNN to learn the plant (mass-spring-damper) behavior.

Build a classic P controller and a neural-network-based controller.

End-to-end training of controller + plant in one model.

Save and load trained models for reuse.

Prerequisites

Python 3.8 or higher

numpy, pandas, matplotlib, scipy, scikit-learn

tensorflow, keras

control

Install all with:

pip install control numpy pandas matplotlib scipy scikit-learn tensorflow keras

Installation

Clone this repo:



git clone https://github.com/yourusername/your-repo-name.git
cd your-repo-name

2. Install the prerequisites (see above).

## Project Structure


/System Identification.py       # Data gen & RNN training for plant model
/Control.py                     # Load plant model, apply controllers, train final controller
/trained_SRNN_model.keras       # Saved RNN model for system identification
/ctrlxplantfinal.keras          # Saved combined controller + plant model
/ctrl_final.keras               # Saved standalone neural-net controller
/model_checkpoint.keras         # Checkpoints saved during RNN training
/images/                        # Place to store plot images for README
README.md                       # This file
LICENSE                         # License info


## Usage

### 1. Identify the system (plant)

```bash
python "System Identification.py"

Uncomment the plotting section in that script if you want to see the training/validation curves.

2. Control the system

Make sure the paths to trained_SRNN_model.keras, ctrlxplantfinal.keras, and ctrl_final.keras in Control.py point to your model files.

Run:

python "Control.py"

You’ll get plots of how well the controller keeps the mass at the setpoint, plus error metrics (IAE, ISE, RMSE).

Results

Below are the key plots illustrating system identification and control performance.Make sure you generate and save these images into the /images folder and name them exactly as below.

System Response to Square Wave Input



System Response to Sawtooth Wave Input



System Response to Sine Wave Input



Position Control Performance



Contributing

Feel free to send pull requests or open issues. Whether it's fixing typos, improving docs, or adding features, your help is welcome!

License

This project is available under the MIT License. See LICENSE for details.
