import asyncio
import docker
import time
from dataclasses import dataclass
from benches.neo4j import Neo4jBenchamrk, wait_neo4j_ready

NEO4J = "neo4j:2026.04.0"
MONGO = "mongo:8.0.21"
FALCOR = "falkordb/falkordb-server:v4.18.3-alpine"
POSTGRES = "postgres:18.3-bookworm"
AGE = "apache/age:release_PG18_1.7.0"
SURREAL = "surrealdb/surrealdb:v3.0.5"
REDIS = "redis:8.6.2-trixie"
CHROMA = "chromadb/chroma:1.5.8"

client = docker.from_env()

images = [NEO4J, MONGO, FALCOR, POSTGRES, AGE, SURREAL, REDIS, CHROMA]


def get_docker_client():
    return client


async def pull_one_image(image_ref: str):
    print(f"Pulling {image_ref}...")
    image = await asyncio.to_thread(client.images.pull, image_ref)
    print(f"Pulled {image_ref}")
    return image


async def pull_images() -> list:
    tasks = [pull_one_image(image_ref) for image_ref in images]
    await asyncio.gather(*tasks)


def setup():
    asyncio.run(pull_images())


@dataclass
class DBSpec:
    image: str
    internal_port: int
    env: dict
    benchmark_cls: type
    wait_ready: callable
    volumes: dict = None


db_specs = [
    DBSpec(
        image=NEO4J,
        internal_port=7687,
        env={"NEO4J_AUTH": "neo4j/password"},
        benchmark_cls=Neo4jBenchamrk,
        wait_ready=wait_neo4j_ready,
    ),
    # DBSpec(image="postgres:16", ...),
    # DBSpec(image="mongo:7", ...),
]


def run_all_benchmarks():
    client = get_docker_client()

    for spec in db_specs:
        container = client.containers.run(
            spec.image,
            environment=spec.env,
            ports={f"{spec.internal_port}/tcp": None},
            detach=True,
        )
        container.reload()  # ensure attrs reflect the assigned port
        port_info = container.attrs["NetworkSettings"]["Ports"][
            f"{spec.internal_port}/tcp"
        ]

        counter = 0
        while port_info is None and counter < 2:
            time.sleep(0.1)
            container.reload()  # yes, I know it's a repeat. IGAF
            port_info = container.attrs["NetworkSettings"]["Ports"][
                f"{spec.internal_port}/tcp"
            ]
            counter += 1

        if port_info is None:
            print(
                f"Couldn't get port information from {spec.image} container. Skipping benchmarks."
            )
            container.stop()
            container.remove()
            continue

        port = int(port_info[0]["HostPort"])
        try:
            spec.wait_ready(port)
            with spec.benchmark_cls(port) as bench:
                bench.perform_benchmark()
        finally:
            container.stop()
            container.remove()
