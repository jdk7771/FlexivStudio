"""CPU checks for changing DDP size without changing the effective batch."""

from copy import deepcopy

import pytest
from torch.utils.data import DistributedSampler

from train import configure_world_size, validate_resume_config


def test_resume_repartitions_batch_without_mutating_checkpoint():
    previous = {"batch_size": 8, "parallelism": {"dp_replicate": 4, "dp_shard": 1}, "steps": 50000}
    config = deepcopy(previous)
    configure_world_size(config, 2)
    validate_resume_config(previous, config)
    assert config["batch_size"] == 16
    assert previous["batch_size"] == 8
    assert previous["parallelism"]["dp_replicate"] == 4
    configure_world_size(config, 4)
    assert config == previous


@pytest.mark.parametrize("field,value", [("batch_size", 8), ("steps", 60000)])
def test_resume_rejects_other_changes(field, value):
    previous = {"batch_size": 8, "parallelism": {"dp_replicate": 4, "dp_shard": 1}, "steps": 50000}
    config = deepcopy(previous)
    configure_world_size(config, 2)
    config[field] = value
    with pytest.raises(ValueError):
        validate_resume_config(previous, config)


def test_sampler_preserves_global_samples_at_resumed_batch_offset():
    dataset = range(1024)
    selected = []
    for ranks, batch in ((4, 8), (2, 16)):
        samples = []
        for rank in range(ranks):
            sampler = DistributedSampler(dataset, num_replicas=ranks, rank=rank, seed=1000)
            sampler.set_epoch(7)
            indices = list(sampler)
            samples.extend(indices[11 * batch:12 * batch])
        assert len(set(samples)) == 32
        selected.append(sorted(samples))
    assert selected[0] == selected[1]
