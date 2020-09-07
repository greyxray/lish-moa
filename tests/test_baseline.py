import pytest
from models.baseline import read_data, build_model, cv_fit


@pytest.fixture
def xy(size=100):
    train = read_data("data/train_features.csv")[:size]
    targets = read_data("data/train_targets_scored.csv")[:size]
    return train, targets


def test_model(xy):
    X, y = xy
    model = build_model()
    model.fit(X, y)


def test_cv_model(xy):
    X, y = xy
    cv_fit(build_model(), X, y, X)
