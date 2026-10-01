"""Conventional trained LR/MLP comparisons with validation-only model selection."""
from itertools import product
import time
import warnings
from pathlib import Path
import joblib
import torch
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from config import LABELS, file_hash

METHODS = ('logistic_regression', 'mlp')


def candidates(method):
    if method == 'logistic_regression':
        return [{'C': value, 'scaling': scaling} for value, scaling in product((.1, 1., 10.), ('fixed_range', 'standard'))]
    if method == 'mlp':
        return [{'width': width, 'alpha': alpha, 'scaling': scaling}
                for width, alpha, scaling in product((16, 32), (.001, .1), ('fixed_range', 'standard'))]
    raise ValueError(f'Unknown classifier: {method}')


def make_classifier(method, parameters, seed):
    if method == 'logistic_regression':
        estimator = LogisticRegression(C=parameters['C'], solver='lbfgs', max_iter=3000,
                                       tol=1e-7, random_state=seed)
    elif method == 'mlp':
        estimator = MLPClassifier(hidden_layer_sizes=(parameters['width'],),
            alpha=parameters['alpha'], activation='relu', solver='lbfgs',
            max_iter=3000, max_fun=100000, tol=1e-7, early_stopping=False,
            random_state=seed)
    else:
        raise ValueError(f'Unknown classifier: {method}')
    if parameters.get('scaling', 'standard') == 'standard':
        return make_pipeline(StandardScaler(), estimator)
    return make_pipeline(estimator)


def fit_classifier(method, parameters, seed, features, labels, selected):
    """NumPy conversion is the scikit-learn boundary; optimizers may use float64."""
    if set(int(labels[index]) for index in selected) != set(range(len(LABELS))):
        raise ValueError('Every class needs reviewed training examples')
    classifier = make_classifier(method, parameters, seed)
    started = time.perf_counter()
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter('always', ConvergenceWarning)
        classifier.fit(features[selected].numpy(), labels[selected].numpy())
    iterations = classifier[-1].n_iter_
    if hasattr(iterations, 'tolist'):
        iterations = iterations.tolist()
    return classifier, {'fit_ms': (time.perf_counter() - started) * 1000,
        'parameters': parameters, 'seed': seed, 'iterations': iterations,
        'convergence_warnings': [str(item.message) for item in captured if issubclass(item.category, ConvergenceWarning)],
        'other_warnings': [str(item.message) for item in captured if not issubclass(item.category, ConvergenceWarning)]}


def select_settings(features, labels, memory, validation, episodes, config):
    from learning import budget_indices, classification_metrics
    if any(episodes[i]['split'] != 'memory' for i in memory) or any(episodes[i]['split'] != 'validation' for i in validation):
        raise ValueError('Model selection needs memory training and validation evaluation only')
    selected_parameters, trials = {}, []
    for budget in config.budgets:
        selected = budget_indices(memory, labels, budget, episodes)
        selected_parameters[str(budget)] = {}
        for method in METHODS:
            scores = []
            for parameters in candidates(method):
                seed_scores, diagnostics = [], []
                for seed in config.encoder_seeds:
                    classifier, info = fit_classifier(method, parameters, seed, features, labels, selected)
                    predictions = classifier.predict(features[validation].numpy()).tolist()
                    seed_scores.append(classification_metrics(predictions, labels[validation].tolist())['macro_f1'])
                    diagnostics.append(info)
                score = sum(seed_scores) / len(seed_scores)
                scores.append(score)
                trials.append({'budget_per_class': budget, 'method': method, 'parameters': parameters,
                    'validation_macro_f1': score, 'seed_scores': seed_scores, 'fit_diagnostics': diagnostics,
                    'review_episode_ids': [episodes[i]['episode_id'] for i in selected]})
            # Candidate order breaks ties; test performance never participates.
            selected_parameters[str(budget)][method] = candidates(method)[max(range(len(scores)), key=lambda i: scores[i])]
        print(f'Frozen LR/MLP settings for {budget} reviews per class', flush=True)
    return {'parameters': selected_parameters, 'trials': trials,
        'selection': 'First data seed validation worlds only, per review budget; mean across three initialization seeds; frozen before final testing',
        'training': 'LR and one-hidden-layer ReLU MLP optimized with L-BFGS, max_iter=3000 and tol=1e-7; MLP max_fun=100000; warnings retained',
        'preprocessing': 'Validation chooses fixed physical range scaling or an additional StandardScaler fitted only to reviewed memory examples at each budget',
        'precision': 'Float32 input boundary; scikit-learn/SciPy optimizer and model state can use float64',
        'label_budget': 'Counts reviewed memory examples; separate labelled validation worlds are used for model selection',
        'hdc_settings': {'local_weight': config.local_weight, 'context_weight': config.context_weight,
            'handset_weight': config.handset_weight, 'levels': config.levels, 'dimension': config.dimension,
            'selection': 'Connected-context weight 2 selected using the archived validation diagnosis on data seeds 1001–1005; frozen before evaluation on fresh data seeds 1006–1010. No HDC setting search in the active suite.'}}


def persist_classifier(classifier, features, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(classifier, path, compress=0)
    restored = joblib.load(path)
    if not torch.equal(torch.as_tensor(classifier.predict(features.numpy())),
                       torch.as_tensor(restored.predict(features.numpy()))):
        raise AssertionError('Persisted classifier predictions changed')
    return {'path': path.name, 'sha256': file_hash(path), 'bytes': path.stat().st_size,
            'prediction_roundtrip_exact': True}
