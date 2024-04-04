# Import libraries for data reading, writing, calculation, manipulation, and plotting
import json
import sys
import numpy as np
from math import log10
import matplotlib.pyplot as plt

class BoltzmannMachine: 
    def calculate_probability_distribution(self, spins, bias, weights):
        """Calculate Boltzmann Machine probability distribution."""
        energy = bias + np.dot(spins, weights)
        probabilities = 1 / (np.exp(-energy) + 1)
        random_sample = np.random.uniform(size=[spins.shape[0], weights.shape[1]])
        modified_data = np.zeros((spins.shape[0], weights.shape[1]), dtype='int')
        
        for row_idx in range(spins.shape[0]):
            for col_idx in range(weights.shape[1]):
                if random_sample[row_idx][col_idx] >= probabilities[row_idx][col_idx]:
                    modified_data[row_idx][col_idx] = 0
                else:
                    modified_data[row_idx][col_idx] = 1  
        return modified_data, probabilities

    def generate_samples(self, visible_units, visible_bias, hidden_bias, weight_matrix, omit_hidden_prime):
        """Generate samples using Boltzmann Machine."""
        hidden_units, hidden_prob = self.calculate_probability_distribution(visible_units, hidden_bias, weight_matrix)
        visible_prime, visible_prime_prob = self.calculate_probability_distribution(hidden_units, visible_bias, np.transpose(weight_matrix)) 
        if omit_hidden_prime:
            hidden_prime = np.zeros(hidden_units.shape) 
        else:
            hidden_prime, _ = self.calculate_probability_distribution(visible_prime, hidden_bias, weight_matrix)
        return visible_units, hidden_units, visible_prime, hidden_prime, visible_prime_prob

    def train(self, visible_units, visible_bias, hidden_bias, weight_matrix):
        """Train Boltzmann Machine model."""
        visible, hidden, visible_prime, hidden_prime, visible_prime_prob = self.generate_samples(visible_units, visible_bias, hidden_bias, weight_matrix, False)
        bias_delta_visible = np.mean(visible, axis=0) - np.mean(visible_prime, axis=0)
        bias_delta_hidden = np.mean(hidden, axis=0) - np.mean(hidden_prime, axis=0)
        weights_delta = np.mean(visible[:, :, None] * hidden[:, None, :], axis=0) - np.mean(visible_prime[:, :, None] * hidden_prime[:, None, :], axis=0)
        return bias_delta_visible, bias_delta_hidden, weights_delta, visible_prime_prob
    
    def calculate_KL_divergence(self, p_distribution, q_distribution):
        """Calculate KL divergence for model evaluation."""
        total_loss = 0
        for i in range(len(p_distribution)):
            if q_distribution[i] > 0: 
                total_loss += p_distribution[i] * log10(p_distribution[i] / q_distribution[i])
            else:
                total_loss += p_distribution[i]
        return total_loss 


class DataParser:
    def convert_ising_data_to_matrix(self, ising_data):
        """Convert raw Ising chain data into a numerical matrix."""
        num_rows = len(ising_data)
        num_cols = len(ising_data[0]) - 1  # Adjust for newline characters
        data_matrix = np.zeros((num_rows, num_cols))
        
        for row_idx, row in enumerate(ising_data):
            for col_idx, spin in enumerate(row[:-1]):  # Ignore newline at the end
                data_matrix[row_idx][col_idx] = 1 if spin == '+' else 0  # Encode '+' as 1, '-' as 0
        return data_matrix

    def read_input_file(self, filepath):
        """Read Ising model data from a file and convert to a numerical matrix."""
        with open(filepath, 'r') as file:
            ising_data_lines = file.readlines()
        data_matrix = self.convert_ising_data_to_matrix(ising_data_lines)
        return data_matrix 
    
    def split_data_into_batches(self, data, batch_size, batch_index):
        """Split the dataset into batches of a specified size."""
        start_index = batch_index * batch_size
        end_index = start_index + batch_size
        return data[start_index:end_index]


if __name__ == "__main__":
    # Display help message if "--help" is the first argument
    if sys.argv[1] == "--help":
        help_message = """
        Usage: python main.py [data_file] [params_json] [output_html]
        
        Parameters:
            - data_file: Path to the Ising chain input dataset (default: data/in.txt).
                * Description: Ising chain Input dataset file name with path
                * File type: txt
                
            - params_json: Path to the JSON file containing hyperparameters (default: param/params.json).
                * Description: JSON file to keep the hyper parameters (e.g., learning rate, batch size)
                * File type: JSON
                
            - output_html: Path to the HTML file for recording the output (default: result/Performance.html).
                * Description: HTML file name with path to record the output. This report summarizes the final results and performance.
                * File type: html
                
        Further details are available in the README.md file.
        """
        print(help_message.strip())

    else:
        # Set default values or parse command line arguments
        input_file = sys.argv[1] if len(sys.argv) > 1 else "data/in.txt"
        json_file = sys.argv[2] if len(sys.argv) > 2 else "param/params.json"
        output_html = sys.argv[3] if len(sys.argv) > 3 else "result/Performance.html"

        # Read hyperparameters from JSON file
        with open(json_file, 'r') as file:
            params = json.load(file)
        
        learning_rate = params.get('learning rate', 0.1)  # Example default value
        num_visible = params['num_visible']
        num_hidden = params['num_hidden']
        batch_size = params['batch_size']
        epochs = params['num iter']

        # Initialize HTML output
        with open(output_html, "w") as html_file:
            html_header = "<html><head><title>Performance Report</title></head><body><h1>Model Training Performance</h1></body></html>"
            html_file.write(html_header)
        
        # Read and process input data
        data_parser = DataParser()
        training_data = data_parser.read_input_file(input_file)

        # Initialize model and training parameters
        biases_visible = np.random.randn(num_visible)
        biases_hidden = np.random.randn(num_hidden)
        weights = np.random.randn(num_visible, num_hidden)
        boltzmann_machine = BoltzmannMachine()

        # Training loop
        for epoch in range(epochs):
            epoch_losses = []
            for batch_idx in range(0, len(training_data), batch_size):
                batch_data = training_data[batch_idx:batch_idx + batch_size]
                delta_biases_visible, delta_biases_hidden, delta_weights, probabilities = boltzmann_machine.train(batch_data, biases_visible, biases_hidden, weights)
                
                # Update model parameters
                biases_visible += learning_rate * delta_biases_visible
                biases_hidden += learning_rate * delta_biases_hidden
                weights += learning_rate * delta_weights
                
                # Calculate and store batch loss
                batch_loss = np.mean([boltzmann_machine.calculate_KL_divergence(probabilities[i], batch_data[i]) for i in range(len(batch_data))])
                epoch_losses.append(batch_loss)
            
            # Calculate epoch loss and log
            mean_epoch_loss = np.mean(epoch_losses)
            print(f"Epoch {epoch + 1}/{epochs} - Training Loss: {mean_epoch_loss:.4f}")
            with open(output_html, "a") as html_file:
                html_file.write(f"<p>Epoch {epoch + 1}/{epochs} - Training Loss: {mean_epoch_loss:.4f}</p>")

        # Finalize HTML report
        with open(output_html, "a") as html_file:
            html_file.write("</body></html>")

      # Plotting training loss
      plt.figure(figsize=(8, 6))
      plt.style.use('dark_background')
      plt.plot(range(1, epochs + 1), epoch_losses , label="Training Loss", color="orange", linewidth=2, marker='o')
      plt.title("KL Divergence Loss Performance for Training Dataset")
      plt.xlabel("Epoch")
      plt.ylabel("Loss Value")
      plt.legend()
      
      # Save the plot to a PNG file
      plot_filename = f"{output_html[:-5]}_training_loss_plot.png"
      plt.savefig(plot_filename)
      plt.show()
      
      # Embed the plot in the HTML report
      with open(output_html, "a") as html_file:
          html_file.write(f"<div><img src='{plot_filename.split('/')[-1]}' alt='Training Loss Plot'></div>")
      
      # Test phase: Summarizing coupler predictions
      n_test_batches = 50
      test_out_samples = np.zeros([n_test_batches, num_visible])
      
      with open(output_html, "a") as html_file:
          html_file.write("<h2>Coupler Dictionary Data</h2><table>")
      
          # Iterate to generate and log coupler predictions
          for j in range(n_test_batches):
              couplers_dict = {}
              visible, _, visible_prime, _, _ = boltzmann_machine.generate_samples(visible, biases_visible, biases_hidden, weights, True)
              test_out_samples[j, :] = visible[0, :]
              
              # Generate coupler predictions based on the sample
              for i in range(num_visible):
                  coupler_key = f"({i}, {(i + 1) % num_visible})"
                  couplers_dict[coupler_key] = 1 if test_out_samples[j, i] > 0 else -1
              
              # Log each batch's coupler dictionary
              html_file.write(f"<tr><td>Batch {j + 1}: {str(couplers_dict)}</td></tr>")
      
          html_file.write("</table>")
      
      print(f"Report writing to {output_html} completed. Model evaluation and coupler predictions are summarized.")
