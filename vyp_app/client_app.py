import torch
from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp
from sklearn.metrics import precision_recall_fscore_support

from vyp_app.task import Net, load_data
from vyp_app.task import test as test_fn
from vyp_app.task import train as train_fn

app = ClientApp()

# ----------------------------  TRAINING CLIENT MODEL ----------------------------
@app.train()
def train(msg: Message, context: Context):
    # Load the model and initialize it with the received weights
    model = Net()
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    
    # Use GPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Load data
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    trainloader, _ = load_data(partition_id, num_partitions)

    # Call training function
    train_loss = train_fn(
        model,
        trainloader,
        context.run_config["local-epochs"],
        msg.content["config"]["lr"],
        device,
    )

    # Construct reply Message
    model_record = ArrayRecord(model.state_dict())
    
    metrics = {
        "train_loss": train_loss,
        "num-examples": len(trainloader.dataset),
    }
    
    metric_record = MetricRecord(metrics)
    content = RecordDict({"arrays": model_record, "metrics": metric_record})
    
    return Message(content=content, reply_to=msg)


# ----------------------------  TESTING CLIENT MODEL ----------------------------
@app.evaluate()
def evaluate(msg: Message, context: Context):

    # Load the model and initialize it with the received weights
    model = Net()
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Load the data
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    _, valloader = load_data(partition_id, num_partitions)

    # Call the evaluation function
    eval_loss, eval_acc = test_fn(
        model,
        valloader,
        device,
    )

    # Construct and return reply Message
    metrics = {
        "eval_loss": eval_loss,
        "eval_acc": eval_acc,
        "num-examples": len(valloader.dataset),
    }
    
    metric_record = MetricRecord(metrics)
    content = RecordDict({"metrics": metric_record})
    
    return Message(content=content, reply_to=msg)
