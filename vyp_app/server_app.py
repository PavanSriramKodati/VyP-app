"""VyP-app: A Flower / PyTorch app."""

from flwr.app import ArrayRecord, ConfigRecord, Context
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedAvg
from vyp_app.task import Net

# Create ServerApp
app = ServerApp()

# --- CUSTOM AGGREGATION FUNCTION ---
def weighted_average(metrics):
    """Aggregates metrics from multiple clients."""
    # Multiply metric by number of examples on that client
    accuracies = [num_examples * m["eval_acc"] for num_examples, m in metrics]
    recalls = [num_examples * m["eval_recall"] for num_examples, m in metrics]
    f1s = [num_examples * m["eval_f1"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    
    # Aggregate and average
    total_examples = sum(examples)
    return {
        "accuracy": sum(accuracies) / total_examples,
        "recall": sum(recalls) / total_examples,
        "f1_score": sum(f1s) / total_examples,
    }

@app.main()
def main(grid: Grid, context: Context) -> None:
    fraction_train: float = context.run_config["fraction-train"]
    num_rounds: int = context.run_config["num-server-rounds"]
    lr: float = context.run_config["lr"]

    global_model = Net()
    arrays = ArrayRecord(global_model.state_dict())

    # Initialize FedAvg strategy with the custom aggregation function
    strategy = FedAvg(
        fraction_train=fraction_train,
        evaluate_metrics_aggregation_fn=weighted_average, # <--- KEY CHANGE
    )

    result = strategy.start(
        grid=grid,
        initial_arrays=arrays,
        train_config=ConfigRecord({"lr": lr}),
        num_rounds=num_rounds,
    )

    print("\nSaving final model to disk...")
    state_dict = result.arrays.to_torch_state_dict()
    import torch
    torch.save(state_dict, "final_model.pt")