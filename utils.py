import matplotlib.pyplot as plt

def plot_predictions(real, predicted, title="Actual vs Predicted", xlabel="Time", ylabel="Stock Price"):
    plt.figure(figsize=(10, 5))
    plt.plot(real, label='Actual', marker='o')
    plt.plot(predicted, label='Predicted', marker='x')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    return plt
