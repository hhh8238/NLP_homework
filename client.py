import argparse

from typing import Dict, List
import Config
import flwr as fl
from client_init import Client, DEVICE


def main() -> None:
    """Load data, start CifarClient."""
    parser = argparse.ArgumentParser(description="Flower")
    parser.add_argument("--node-id", type=int, required=True, choices=range(0, 10))
    args = parser.parse_args()

    # Load data
    trainloader, testloader = Config.load_data(args.node_id)

    # Load model
    model = Config.Net().to(DEVICE).train()

    # Perform a single forward pass to properly initialize BatchNorm
    _ = model(next(iter(trainloader))[0].to(DEVICE))


    # Start client
    client = Client(model, trainloader, testloader)
    fl.client.start_numpy_client(server_address="127.0.0.1:8080", client=client)


if __name__ == "__main__":
    main()
