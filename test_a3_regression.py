# -*- coding: utf-8 -*-
"""
PyTests für a3_Regression.ipynb
- Lädt nur Funktions-/Klassendefinitionen aus dem Notebook (keine Trainings-/Plot-Zellen),
  damit Tests unabhängig von Demo-Code laufen.
- Prüft TODOs: Implementierungen, Lernkurven, Gradientenabstiege,
  Normalgleichungen (1D & multivariat).
"""

import io
import json
import math
import types
import numpy as np
import pytest

NB_PATH = "a3_Regression.ipynb"

# ---------------------------
# Hilfsfunktionen & Fixtures
# ---------------------------

def _load_code_cells_from_notebook(path):
    with io.open(path, "r", encoding="utf-8") as f:
        nb = json.load(f)
    # Sammle nur Codezellen
    for cell in nb.get("cells", []):
        if cell.get("cell_type") == "code":
            src = "".join(cell.get("source", []))
            yield src


def _exec_defs_only(namespace):
    """
    Führt nur Funktions-/Klassendefinitionen + simple Hilfsfunktionen aus dem NB aus.
    Vermeidet Zellen, die Training/Plots ausführen.
    """
    had_syntax_error = False
    syntax_errors = []

    for src in _load_code_cells_from_notebook(NB_PATH):
        # Wenn die Zelle Definitionen enthält, aber vermutlich keine Demos/Plots startet
        is_def = any(
            key in src
            for key in [
                "class SimpleGradientDescentModel",
                "class GradientDescentModel",
                "class GradientDescentModel_vector",
                "class OptimalLinearModel",
                "class OptimalLinearModel_vector",
                "def mean_squared_error",
            ]
        )
        # Skip offensichtliche "Demo"-Zellen: Training/Plots
        looks_like_demo = any(
            token in src
            for token in [
                "plt.show(", "model =", "plt.plot(", "plt.scatter(", "Axes3D",
                "plot_surface(", "print(\"MSE"
            ]
        )
        if is_def and not looks_like_demo:
            try:
                exec(src, namespace)
            except SyntaxError as e:
                had_syntax_error = True
                syntax_errors.append(str(e))

    if had_syntax_error:
        pytest.fail(
            "Dein Notebook enthält noch unvollständige TODOs / Syntaxfehler in den "
            "Klassendefinitionen. Bitte fülle die TODOs aus.\n\n"
            + "\n".join(syntax_errors)
        )


@pytest.fixture(scope="module")
def ns():
    """
    Namespace mit Definitions-Code aus dem Notebook.
    """
    namespace = {"np": np}
    _exec_defs_only(namespace)
    assert "mean_squared_error" in namespace, "mean_squared_error fehlt noch."
    return namespace


@pytest.fixture(scope="module")
def toy_data():
    # Repliziere die Daten aus dem Notebook
    data = [
        {"size": 19, "price": 380, "rooms": 1, "distance_to_center": 6,  "location": "Stieghorst"},
        {"size": 32, "price": 450, "rooms": 1, "distance_to_center": 4,  "location": "Schildesche"},
        {"size": 69, "price": 750, "rooms": 3, "distance_to_center": 3,  "location": "Heepen"},
        {"size": 60, "price": 800, "rooms": 2, "distance_to_center": 4.5,"location": "Dornberg"},
        {"size": 20, "price": 420, "rooms": 1, "distance_to_center": 1,  "location": "Mitte"},
        {"size": 18, "price": 580, "rooms": 1, "distance_to_center": 2,  "location": "Gadderbaum"},
        {"size": 52, "price": 700, "rooms": 3, "distance_to_center": 0.5,"location": "Innenstadt"},
        {"size": 45, "price": 1500,"rooms": 2, "distance_to_center": 1.5,"location": "Jöllenbeck"},
        {"size": 50, "price": 1600,"rooms": 2, "distance_to_center": 1,  "location": "Sennestadt"},
    ]
    sizes = np.array([d["size"] for d in data])
    dist  = np.array([d["distance_to_center"] for d in data])
    prices= np.array([d["price"] for d in data])
    rooms = np.array([d["rooms"] for d in data])
    X = np.column_stack((np.ones(sizes.shape[0]), sizes, dist))
    Y = np.column_stack((prices, rooms))
    return X, Y, sizes, dist, prices


# ---------------------------
# Einzeltests
# ---------------------------

def test_mean_squared_error(ns):
    mse = ns["mean_squared_error"]([1, 2, 3], [1, 2, 5])
    assert pytest.approx(mse, rel=1e-6) == 4/3  # ((0^2 + 0^2 + 2^2)/3)


def test_simple_gd_model_exists_and_learns(ns, toy_data):
    assert "SimpleGradientDescentModel" in ns, "Klasse SimpleGradientDescentModel fehlt."

    X, Y, sizes, dist, prices = toy_data
    Model = ns["SimpleGradientDescentModel"]
    m = Model(learning_rate=0.001, n_iterations=300)
    # Erwartung: fit() implementiert Gradientenabstieg und schreibt mse_history
    m.fit(sizes, prices)
    hist = m.get_mse_history()
    assert isinstance(hist, list) and len(hist) > 0, "mse_history wird nicht gefüllt."
    assert hist[0] > hist[-1], "MSE sollte im Verlauf sinken."
    preds = m.predict(sizes)
    assert len(preds) == len(sizes), "predict liefert falsche Länge."
    # Grober Qualitätscheck
    final_mse = ns["mean_squared_error"](prices, preds)
    assert final_mse < ns["mean_squared_error"](prices, [np.mean(prices)]*len(prices)), \
        "GD sollte besser sein als der Mittelwert-Baseline."


def test_gd_model_fit_and_fit_vector(ns, toy_data):
    assert "GradientDescentModel" in ns, "Klasse GradientDescentModel fehlt."

    X, Y, sizes, dist, prices = toy_data
    Model = ns["GradientDescentModel"]

    # Variante A: sample-weises Update
    m1 = Model(learning_rate=5e-6, n_iterations=600)
    m1.fit(sizes, prices)
    hist1 = m1.get_mse_history()
    assert len(hist1) > 0 and hist1[0] > hist1[-1], "fit: MSE sollte sinken."

    # Variante B: Vektorisiertes Update (Batch)
    m2 = Model(learning_rate=5e-6, n_iterations=600)
    assert hasattr(m2, "fit_vector"), "fit_vector fehlt."
    m2.fit_vector(sizes, prices)
    hist2 = m2.get_mse_history()
    assert len(hist2) > 0 and hist2[0] > hist2[-1], "fit_vector: MSE sollte sinken."

    # Beide sollten auf ähnliche Qualität kommen
    mse1 = ns["mean_squared_error"](prices, m1.predict(sizes))
    mse2 = ns["mean_squared_error"](prices, m2.predict(sizes))
    assert mse2 <= mse1 * 1.5 + 1e-6, "Batch-/Vektorversion sollte nicht deutlich schlechter sein."


def test_multidim_gradient_descent_vector(ns, toy_data):
    assert "GradientDescentModel_vector" in ns, "Klasse GradientDescentModel_vector fehlt."

    X, Y, sizes, dist, prices = toy_data
    Model = ns["GradientDescentModel_vector"]
    m = Model(learning_rate=5e-4, n_iterations=1500)
    m.fit(X, prices)
    preds = m.predict(X)
    assert preds.shape == prices.shape, "Vorhersageform passt nicht."
    hist = m.get_mse_history()
    assert len(hist) > 0 and hist[0] > hist[-1], "MSE sollte im Verlauf sinken."
    final_mse = ns["mean_squared_error"](prices, preds)
    assert final_mse < 2e5, "Fehler zu hoch – Gewichte/Gradienten vermutlich nicht korrekt."


def test_optimal_linear_model_closed_form(ns, toy_data):
    assert "OptimalLinearModel" in ns, "Klasse OptimalLinearModel fehlt."

    X, Y, sizes, dist, prices = toy_data
    Model = ns["OptimalLinearModel"]
    m = Model()
    m.fit(sizes, prices)

    # Referenz über numpy.polyfit (1. Grad)
    ref_slope, ref_intercept = np.polyfit(sizes, prices, 1)
    assert math.isfinite(m.slope) and math.isfinite(m.intercept), "m/b sind nicht gesetzt."
    assert m.slope == pytest.approx(ref_slope, rel=1e-4, abs=1e-2)
    assert m.intercept == pytest.approx(ref_intercept, rel=1e-4, abs=1e-1)

    # MSE gegen Referenz prüfen
    preds = m.predict(sizes)
    mse = ns["mean_squared_error"](prices, preds)
    ref_preds = ref_slope * sizes + ref_intercept
    ref_mse = ns["mean_squared_error"](prices, ref_preds)
    assert mse == pytest.approx(ref_mse, rel=1e-4, abs=1e-2)


def test_optimal_linear_model_vector_normal_equation(ns, toy_data):
    assert "OptimalLinearModel_vector" in ns, "Klasse OptimalLinearModel_vector fehlt."

    X, Y, sizes, dist, prices = toy_data
    Model = ns["OptimalLinearModel_vector"]
    m = Model()
    m.fit(X, prices)

    # Referenz: Normalgleichung
    w_ref = np.linalg.inv(X.T @ X) @ (X.T @ prices)
    assert m.weights is not None and m.weights.shape == (X.shape[1],), "Gewichte nicht korrekt initialisiert."
    assert np.allclose(m.weights, w_ref, rtol=1e-5, atol=1e-3), "Normalgleichung nicht korrekt umgesetzt."

    # MSE prüfen
    preds = m.predict(X)
    mse = ns["mean_squared_error"](prices, preds)
    ref_mse = ns["mean_squared_error"](prices, X @ w_ref)
    assert mse == pytest.approx(ref_mse, rel=1e-5, abs=1e-3)
