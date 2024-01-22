"""Start a Flower server.

Derived from Flower Android example.
"""

from flwr.server import ServerConfig, start_server
from flwr.server.strategy import FedAvgAndroid

from typing import Callable, Dict, List, Optional, Tuple, Union, cast
from flwr.common import Metrics , EvaluateRes, NDArrays, Scalar, FitRes, Parameters, EvaluateIns, FitIns, NDArray
import flwr.common
from flwr.server.client_proxy import ClientProxy
import flwr.server.strategy
import numpy as np

import tensorflow as tf

PORT = 8080

'''
class SaveModelStrategy(flwr.server.strategy.FedAvgAndroid):
    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[flwr.server.client_proxy.ClientProxy, flwr.common.FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:

        # Call aggregate_fit from base class (FedAvg) to aggregate parameters and metrics
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(server_round, results, failures)

        if aggregated_parameters is not None:
            # Convert `Parameters` to `List[np.ndarray]`
            aggregated_ndarrays: List[np.ndarray] = flwr.common.parameters_to_ndarrays(aggregated_parameters)

            # Save aggregated_ndarrays
            print(f"Saving round {server_round} aggregated_ndarrays...")
            np.savez(f"round-{server_round}-weights.npz", *aggregated_ndarrays)

        return aggregated_parameters, aggregated_metrics

class SaveModelStrategy(flwr.server.strategy.FedAvg):
    def aggregate_fit(
        self,
        rnd: int,
        results: List[Tuple[flwr.server.client_proxy.ClientProxy, flwr.common.FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(rnd, results, failures)
        if aggregated_parameters is not None:
            # Save weights
            #aggregated_ndarrays: List[np.ndarray] = flwr.common.parameters_to_ndarrays(aggregated_parameters)
            print(aggregated_metrics)
            print(aggregated_parameters)
            print(f"Saving round {rnd} aggregated_ndarrays...")
            np.savez(f"round-{rnd}-weights.npz", aggregated_parameters)
        return aggregated_parameters, aggregated_metrics


    #I dont use it
def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    # Multiply accuracy of each client by number of examples used
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    # Aggregate and return custom metric (weighted average)
    return {"accuracy": sum(accuracies) / sum(examples)}

'''

def main():
    strategy = FedAvgAndroid(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=5,
        min_evaluate_clients=5,
        min_available_clients=5,
        evaluate_fn=None,
        on_fit_config_fn=fit_config,
        initial_parameters=None,
    )
    
    try:
        # Start Flower server for 10 rounds of federated learning
        start_server(
            server_address=f"0.0.0.0:{PORT}",
            config=flwr.server.ServerConfig(num_rounds=10),
            strategy= strategy,
        ),
    except KeyboardInterrupt:
        return
    
def fit_config(server_round: int):
    config = {
        "local_epochs": 20,
    }
    return config

def evaluate(self, parameters, config):
        """Evaluate parameters on the locally held test set."""

        # Update local model with global parameters
        self.model.set_weights(parameters)

        # Evaluate global model parameters on the local test data
        loss, accuracy = self.model.evaluate(self.x_test, self.y_test)

        # Return results, including the custom accuracy metric
        num_examples_test = len(self.x_test)
        return loss, num_examples_test, {"accuracy": accuracy}

if __name__ == "__main__":
    main()

