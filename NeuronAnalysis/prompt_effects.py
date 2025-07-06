from NeuronAnalysis.utils import get_neuron_activations, neuron_importance_analysis
import numpy as np

def compare_prompt_activations(model, tokenizer, question):
    activations_no_prompt = get_neuron_activations(model, tokenizer, question, prompt=None)
    activations_with_prompt = get_neuron_activations(model, tokenizer, question, prompt="假设你是一个专业的医生")

    kl_div, important_neurons = neuron_importance_analysis(activations_no_prompt, activations_with_prompt)

    return {
        "kl_divergence": float(kl_div) if kl_div is not None else None, 
        "important_neurons": important_neurons.tolist() if isinstance(important_neurons, np.ndarray) else important_neurons 
    }

