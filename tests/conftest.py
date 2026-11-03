import pytest

from upstream_watch import config, pipeline
from upstream_watch.agent import build_agent


@pytest.fixture(scope="session")
def sample():
    return pipeline.load_curated(config.SAMPLE_DIR)


@pytest.fixture(scope="session")
def kit_and_model(sample):
    gauges, edges, stage = sample
    settings = config.Settings(llm_provider="rule", embedder="hash")
    return pipeline.build_toolkit(gauges, edges, stage, settings, embedder_name="hash")


@pytest.fixture(scope="session")
def kit(kit_and_model):
    return kit_and_model[0]


@pytest.fixture()
def agent(kit):
    return build_agent(kit)
