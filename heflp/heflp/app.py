import flwr as fl
from pathlib import Path

def start_client(
        server_address: str,
        client: fl.client.Client,
        root_certificate: str,
):
    try:
        if isinstance(client, fl.client.NumPyClient):
            fl.client.start_numpy_client(
                server_address=server_address,
                client=client,
                root_certificates=Path(root_certificate).read_bytes(),
            )
        elif isinstance(client, fl.client.Client):
            fl.client.start_client(
                server_address=server_address,
                client=client,
                root_certificates=Path(root_certificate).read_bytes(),
            )
        else:
            raise ValueError(f"Invalid client type {type(client)}")
    except Exception as e:
        raise e

def start_server(
        server_address: str,
        strategy: fl.client.Client,
        config: fl.server.ServerConfig,
        root_certificate: str,
        priv_key_path: str,
        pub_key_path: str, 
):
    try:
        hist = fl.server.start_server(
            server_address=server_address,
            config=config,
            strategy=strategy,
            certificates=(
                Path(root_certificate).read_bytes(),
                Path(pub_key_path).read_bytes(),
                Path(priv_key_path).read_bytes(),
            )
        )
        return hist
    except Exception as e:
        raise e