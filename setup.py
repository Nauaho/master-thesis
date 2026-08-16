import asyncio
import docker
import time
from dataclasses import dataclass
from benches.neo4j import Neo4jBenchamrk, wait_neo4j_ready
from benches.redis import RedisBenchmark, wait_redis_ready
from benches.chroma import ChromaBenchmark, wait_chroma_ready
from benches.pgvector import PgVectorBenchmark, wait_pgvector_ready
from pathlib import Path

CPU_QUOTA = 4_000_000_000
MEM_QUOTA = "16g"

DATA_DIR = (Path(__file__).parent / "data").resolve()

NEO4J = "neo4j:2026.04.0"
MONGO = "mongo:8.0.21"
FALCOR = "falkordb/falkordb-server:v4.18.3-alpine"
POSTGRES = "postgres:18.3-bookworm"
AGE = "apache/age:release_PG18_1.7.0"
SURREAL = "surrealdb/surrealdb:v3.0.5"
REDIS = "redis:8.6.2-trixie"
CHROMA = "chromadb/chroma:1.5.8"
PGVECTOR = "pgvector/pgvector:0.8.6-pg18-trixie"

client = docker.from_env()

images = [NEO4J, MONGO, FALCOR, POSTGRES, AGE, SURREAL, REDIS, CHROMA, PGVECTOR]


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
        env={"NEO4J_AUTH": "neo4j/password", "NEO4J_dbms_directories_import": "/import",},
        benchmark_cls=Neo4jBenchamrk,
        wait_ready=wait_neo4j_ready,
        volumes={
            str(DATA_DIR): {"bind": "/import", "mode": "ro"}
        },
    ),
    DBSpec(
        image=REDIS,
        internal_port=6379,
        env={},
        benchmark_cls=RedisBenchmark,
        wait_ready=wait_redis_ready,
        volumes=None,
    ),
    DBSpec(
        image=CHROMA,
        internal_port=8000,
        env={"IS_PERSISTENT": "TRUE"},
        benchmark_cls=ChromaBenchmark,
        wait_ready=wait_chroma_ready,
        volumes=None,
    ),
    DBSpec(
        image=PGVECTOR,
        internal_port=5432,
        env={"POSTGRES_PASSWORD": "password"},
        benchmark_cls=PgVectorBenchmark,
        wait_ready=wait_pgvector_ready,
    )
]


def run_all_benchmarks():
    client = get_docker_client()

    for spec in db_specs:
        container = client.containers.run(
            spec.image,
            environment=spec.env,
            ports={f"{spec.internal_port}/tcp": None},
            volumes=spec.volumes,
            detach=True,
            nano_cpus=CPU_QUOTA,
            mem_limit=MEM_QUOTA
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
            print("Check container")
            container.stop()
            container.remove(v=True)
