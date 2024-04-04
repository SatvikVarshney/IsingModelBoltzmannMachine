# Boltzmann Machine for Ising Model Data

## Overview
This project focuses on the implementation and application of a Boltzmann Machine to the field of statistical mechanics, particularly the Ising model. By employing a Boltzmann Machine, we aim to uncover patterns and correlations within Ising model data, facilitating a deeper understanding of its statistical properties. This endeavor leverages numpy for numerical computations and matplotlib for visualizing the model's performance, emphasizing both theoretical insights and practical machine learning techniques.

## Features
- **Boltzmann Machine Implementation**: A custom-built Boltzmann Machine model using numpy, capable of learning and generating samples from Ising model data.
- **Custom Data Processing**: Includes a `DataParser` class for converting raw Ising model data into a format suitable for machine learning applications.
- **Hyperparameter Configuration**: Utilizes a JSON-based configuration system, allowing for easy experimentation with different model settings.
- **Performance Visualization**: Generates HTML reports and graphical representations of the training process, focusing on loss metrics and sample generation quality.

## Result Observations
### Training Loss Performance

![Performance_graphs](https://github.com/SatvikVarshney/IsingModelBoltzmannMachine/assets/114079530/fd7b9713-b9e5-49d1-8190-16a60f8e70f1)

The graph above illustrates the KL Divergence Loss performance over 100 training epochs for the dataset. Initially, we observe a steep decrease in loss value, dropping from around 4.5 to below 3 within the first few epochs. This indicates a rapid improvement in the model's ability to learn the statistical properties of the Ising model data at the beginning of the training process.

As training progresses, the loss continues to decline, though at a slower rate, eventually plateauing as it approaches convergence. The general downward trend in the graph suggests that the model consistently improves its sample generation to more accurately reflect the true distribution of the data.

Throughout the training, there are minor fluctuations in the loss value, which are typical during the optimization process as the model navigates the loss landscape. Despite these fluctuations, the overall trend remains decidedly downward, underscoring the effectiveness of the training regimen.

By the 100th epoch, the training loss stabilizes around a value of 1.0, indicative of the model's solid performance and its successful approximation of the Ising model's probability distribution. This result demonstrates the Boltzmann Machine's capability to learn complex patterns, paving the way for practical applications in the study of statistical mechanics.


## Getting Started

### Prerequisites
- Python 3.x
- numpy
- matplotlib

### Data Files
The project requires Ising model data, typically stored in a text format. Each line represents a state in the Ising model, with spins encoded as either `+` or `-`. The `DataParser` class includes methods for converting this data into a numerical matrix for training.

### Configuration Files
Model parameters and training settings are defined in a JSON file located in the `param` directory. This flexibility allows for quick adjustments to the model's learning rate, batch size, and training epochs.

Example `parameters.json`:

```json
{
	"learning rate": 0.01,
	"num iter": 100,
	"num_visible": 4,
	"num_hidden": 8,
	"batch_size" : 5
}
```

Clone this repository to your local machine:
```bash
git clone https://github.com/SatvikVarshney/IsingModelBoltzmannMachine.git
```
Navigate to the project directory:
```bash
cd IsingModelBoltzmannMachine

```
Install the required dependencies:
```bash
pip install -r requirements.txt
```
### Usage
To train the model and generate samples, run:

```bash
python Main.py data/in.txt param/params.json Result/Performance.html
```
