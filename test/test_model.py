from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier


def test_model_training():
    iris = load_iris()
    X, y = iris.data, iris.target

    model = RandomForestClassifier()
    model.fit(X, y)

    assert model is not None


def test_prediction_shape():
    iris = load_iris()
    X, y = iris.data, iris.target

    model = RandomForestClassifier()
    model.fit(X, y)

    preds = model.predict(X)

    assert len(preds) == len(y)
