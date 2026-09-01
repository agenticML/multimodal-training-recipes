import torch

from training_recipes.losses import compute_loss


def test_regression_composite_is_finite_and_differentiable():
    prediction = torch.randn(2, 1, 32, requires_grad=True)
    target = torch.randn_like(prediction)
    loss, metrics = compute_loss(
        {"prediction": prediction},
        {"targets": target},
        [{"name": "waveform_l1", "weight": 0.5}, {"name": "multi_resolution_stft", "weight": 0.5}],
    )
    loss.backward()
    assert torch.isfinite(loss)
    assert prediction.grad is not None
    assert len(metrics) == 2


def test_token_and_classification_losses_are_finite():
    token_logits = torch.randn(2, 5, 11, requires_grad=True)
    token_targets = torch.randint(0, 11, (2, 5))
    token_loss, _ = compute_loss(
        {"logits": token_logits},
        {"targets": token_targets},
        [{"name": "label_smoothed_cross_entropy"}],
    )
    class_logits = torch.randn(3, 4, requires_grad=True)
    class_targets = torch.tensor([0, 1, 2])
    class_loss, _ = compute_loss(
        {"logits": class_logits},
        {"targets": class_targets},
        [{"name": "focal"}],
    )
    assert torch.isfinite(token_loss + class_loss)


def test_every_task_loss_accepts_its_smoke_output():
    from torch.utils.data import DataLoader

    from training_recipes.datasets import create_dataset
    from training_recipes.models import create_smoke_model
    from training_recipes.registry import architecture_catalog, task_names

    for task in task_names():
        spec = next(iter(architecture_catalog(task).values()))
        config = {
            "data": {
                "length": 2,
                "sequence_length": 8,
                "vocab_size": 16,
                "num_classes": 4,
                "seed": 7,
            }
        }
        batch = next(iter(DataLoader(create_dataset(task, config["data"]), batch_size=2)))
        outputs = create_smoke_model(task, config)(batch)
        for name in spec["losses"]:
            loss, _ = compute_loss(outputs, batch, [{"name": name}])
            assert torch.isfinite(loss), f"non-finite {task}/{name}"
