import tensorflow as tf
from keras.models import load_model

def load_trained_model(model_path: str):
    try:
        model = load_model(model_path)
        print(f"Model loaded from {model_path}")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None
    
def get_total_memory_usage(model, batch_size=32):
    try:
        # Calcola il numero totale di parametri
        total_params = model.count_params()
        parameters_mb = total_params * 4 / (1024 ** 2)  # 4 bytes per parametro

        # Calcola le attivazioni per ogni layer
        activations_mb = 0
        for layer in model.layers:
            if not hasattr(layer, 'output_shape'):
                continue
            output_shape = layer.output_shape
            if isinstance(output_shape, list):
                output_shape = output_shape[0]  # Prendi la prima uscita se ce ne sono più di una
            if output_shape is not None:
                activations = batch_size * tf.reduce_prod(output_shape[1:])  # Escludi la dimensione del batch
                activations_mb += activations * 4 / (1024 ** 2)  # 4 bytes per attivazione

        total_mb = parameters_mb + activations_mb
        return {
            "parameters_mb": parameters_mb,
            "activations_mb": activations_mb,
            "total_mb": total_mb
        }
    except Exception as e:
        print(f"Error calculating memory usage: {e}")
        return None
    

if __name__ == "__main__":
    model = load_trained_model("model.keras")
    if model is not None:
        memory = get_total_memory_usage(model, batch_size=32)
        if memory is not None:
            print(f"Parameters: {memory['parameters_mb']:.2f} MB")
            print(f"Activations: {memory['activations_mb']:.2f} MB")
            print(f"Total: {memory['total_mb']:.2f} MB")

            print(model.summary())