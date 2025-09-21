import torch
import matplotlib.pyplot as plt
import pandas as pd
from transformers import Trainer


class CustomBaseTrainer(Trainer):
    """
    https://hf.qhduan.com/docs/transformers/trainer
    """

    def __init__(self, model, args, **kwargs):
        super().__init__(model, args, **kwargs)

    def plot_results(self):
        if not self.state.log_history:
            print("Нет данных для построения графиков. Проверьте, выполнялось ли обучение.")
            return
        
        log_data = pd.DataFrame(self.state.log_history)

        log_data = log_data.dropna(subset=["epoch"])
        log_data = log_data.groupby("epoch").last().reset_index()
        
        available_metrics = [col for col in log_data.columns if col.startswith("eval_") or col == "loss"]

        num_plots = len(available_metrics)
        plt.figure(figsize=(8, 4 * num_plots))

        for i, metric in enumerate(available_metrics, start=1):
            if metric == "report": 
                continue           
            plt.subplot(num_plots, 1, i)
            plt.plot(log_data["epoch"], log_data[metric], marker="o", label=metric)

            plt.xlabel("Эпохи")
            plt.ylabel(metric)
            plt.title(f"График {metric}")
            plt.legend()
            plt.grid(True)

        plt.tight_layout()
        plt.show()