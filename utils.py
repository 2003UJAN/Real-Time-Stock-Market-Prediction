import matplotlib.pyplot as plt

def plot_predictions(real, predicted):
    plt.figure(figsize=(10,5))
    plt.plot(real, label='Actual')
    plt.plot(predicted, label='Predicted')
    plt.legend()
    return plt
