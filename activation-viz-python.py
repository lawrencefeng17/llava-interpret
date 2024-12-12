import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Union, List
import pandas as pd

def plot_activations(activations1: np.ndarray, 
                    activations2: np.ndarray = None,
                    plot_type: str = 'line',
                    title: str = 'Neural Activations',
                    figsize: tuple = (15, 8)) -> None:
    """
    Plot neural network activations with various visualization options.
    
    Args:
        activations1: First activation tensor
        activations2: Optional second activation tensor for comparison
        plot_type: One of ['line', 'scatter', 'histogram', 'difference']
        title: Plot title
        figsize: Figure size tuple
    """
    plt.figure(figsize=figsize)
    
    if plot_type == 'line':
        plt.plot(activations1, label='Tensor 1', alpha=0.7)
        if activations2 is not None:
            plt.plot(activations2, label='Tensor 2', alpha=0.7)
        plt.xlabel('Neuron Index')
        plt.ylabel('Activation Value')
        
    elif plot_type == 'scatter':
        plt.scatter(range(len(activations1)), activations1, 
                   alpha=0.5, label='Tensor 1', s=1)
        if activations2 is not None:
            plt.scatter(range(len(activations2)), activations2, 
                       alpha=0.5, label='Tensor 2', s=1)
        plt.xlabel('Neuron Index')
        plt.ylabel('Activation Value')
        
    elif plot_type == 'histogram':
        sns.histplot(data=activations1, label='Tensor 1', alpha=0.5)
        if activations2 is not None:
            sns.histplot(data=activations2, label='Tensor 2', alpha=0.5)
        plt.xlabel('Activation Value')
        plt.ylabel('Count')
        
    elif plot_type == 'difference':
        if activations2 is None:
            raise ValueError("Second tensor required for difference plot")
        diff = activations1 - activations2
        plt.plot(diff, label='Difference', color='orange')
        plt.axhline(y=0, color='r', linestyle='--', alpha=0.3)
        plt.xlabel('Neuron Index')
        plt.ylabel('Activation Difference')
    
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

def print_activation_stats(activations1: np.ndarray, 
                         activations2: np.ndarray = None) -> None:
    """Print statistical information about the activations."""
    print("\nTensor 1 Statistics:")
    print(f"Mean: {np.mean(activations1):.4f}")
    print(f"Std:  {np.std(activations1):.4f}")
    print(f"Min:  {np.min(activations1):.4f}")
    print(f"Max:  {np.max(activations1):.4f}")
    
    if activations2 is not None:
        print("\nTensor 2 Statistics:")
        print(f"Mean: {np.mean(activations2):.4f}")
        print(f"Std:  {np.std(activations2):.4f}")
        print(f"Min:  {np.min(activations2):.4f}")
        print(f"Max:  {np.max(activations2):.4f}")
        
        diff = activations1 - activations2
        print("\nDifference Statistics:")
        print(f"Mean: {np.mean(diff):.4f}")
        print(f"Std:  {np.std(diff):.4f}")
        print(f"Min:  {np.min(diff):.4f}")
        print(f"Max:  {np.max(diff):.4f}")

# Example usage:
if __name__ == "__main__":
    # Generate sample data
    x1 = np.random.normal(0, 1, 16000)
    x2 = np.random.normal(0.5, 1.2, 16000)
    
    # Create all types of plots
    for plot_type in ['line', 'scatter', 'histogram', 'difference']:
        plot_activations(x1, x2, plot_type=plot_type, 
                        title=f'Neural Activations - {plot_type.title()} Plot')
    
    # Print statistics
    print_activation_stats(x1, x2)
