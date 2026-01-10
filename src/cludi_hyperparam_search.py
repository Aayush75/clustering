"""
Hyperparameter search module for CLUDI clustering.

This module provides various methods for hyperparameter optimization
for the CLUDI clustering algorithm, including:
- Grid Search: Exhaustive search over specified parameter values
- Random Search: Random sampling from parameter distributions
- Bayesian Optimization: Sequential model-based optimization (using Optuna if available)

The module supports searching over all CLUDI-specific hyperparameters
and uses clustering metrics (accuracy, NMI, ARI) for evaluation.
"""

import os
import json
import itertools
import random
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
from dataclasses import dataclass, field, asdict
from copy import deepcopy

import torch
import numpy as np
from tqdm import tqdm

from src.cludi_clustering import CLUDIClusterer
from src.evaluation import evaluate_clustering


@dataclass
class CLUDIHyperparameterSpace:
    """
    Defines the hyperparameter search space for CLUDI.
    
    Each hyperparameter can be specified as:
    - A single value (fixed)
    - A list of values (for grid/random search)
    - A tuple (min, max) for continuous parameters in random search
    - A tuple (min, max, log_scale) for log-scale sampling
    
    Attributes:
        embedding_dim: Dimension of cluster embeddings
        learning_rate: Learning rate for optimizer
        diffusion_steps: Number of diffusion timesteps
        batch_diffusion: Batch size for diffusion sampling
        rescaling_factor: Factor for rescaling in diffusion
        ce_lambda: Weight for cross-entropy loss
        warmup_epochs: Number of warmup epochs
    """
    embedding_dim: Union[int, List[int]] = field(default_factory=lambda: [32, 64, 128])
    learning_rate: Union[float, List[float], Tuple[float, float]] = field(
        default_factory=lambda: (1e-5, 1e-3)  # Log scale by default
    )
    diffusion_steps: Union[int, List[int]] = field(default_factory=lambda: [500, 1000])
    batch_diffusion: Union[int, List[int]] = field(default_factory=lambda: [4, 8, 16])
    rescaling_factor: Union[float, List[float], Tuple[float, float]] = field(
        default_factory=lambda: [25.0, 49.0, 100.0]
    )
    ce_lambda: Union[float, List[float], Tuple[float, float]] = field(
        default_factory=lambda: [25.0, 50.0, 100.0]
    )
    warmup_epochs: Union[int, List[int]] = field(default_factory=lambda: [0, 1, 2])
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> 'CLUDIHyperparameterSpace':
        """Create from dictionary."""
        return cls(**{k: v for k, v in config.items() if k in cls.__dataclass_fields__})


@dataclass
class SearchResult:
    """
    Result of a single hyperparameter configuration trial.
    
    Attributes:
        params: The hyperparameter configuration used
        metrics: Evaluation metrics (accuracy, NMI, ARI)
        train_time: Time taken to train (seconds)
        trial_id: Unique identifier for this trial
    """
    params: Dict[str, Any]
    metrics: Dict[str, float]
    train_time: float
    trial_id: int
    
    @property
    def score(self) -> float:
        """Primary score for comparison (default: accuracy)."""
        return self.metrics.get('accuracy', 0.0)


class CLUDIHyperparameterSearch:
    """
    Hyperparameter search for CLUDI clustering.
    
    This class provides multiple search strategies for finding optimal
    hyperparameters for the CLUDI clustering algorithm.
    
    Supported search methods:
    - grid: Exhaustive grid search
    - random: Random search with specified number of trials
    - bayesian: Bayesian optimization using Optuna (if available)
    
    Example usage:
    ```python
    # Define search space
    search_space = CLUDIHyperparameterSpace(
        embedding_dim=[32, 64, 128],
        learning_rate=(1e-5, 1e-3),  # Log-scale sampling
        ce_lambda=[25.0, 50.0, 100.0]
    )
    
    # Create searcher
    searcher = CLUDIHyperparameterSearch(
        feature_dim=768,
        num_clusters=100,
        device="cuda"
    )
    
    # Run search
    best_params, results = searcher.search(
        features=train_features,
        labels=train_labels,
        search_space=search_space,
        method="random",
        n_trials=20,
        num_epochs=50
    )
    ```
    """
    
    def __init__(
        self,
        feature_dim: int,
        num_clusters: int,
        device: str = "cuda",
        metric: str = "accuracy",
        results_dir: str = "./hyperparam_search_results"
    ):
        """
        Initialize hyperparameter searcher.
        
        Args:
            feature_dim: Dimension of input features
            num_clusters: Number of clusters to create
            device: Device for computation (cuda/cpu)
            metric: Metric to optimize ('accuracy', 'nmi', 'ari')
            results_dir: Directory to save search results
        """
        self.feature_dim = feature_dim
        self.num_clusters = num_clusters
        self.device = device
        self.metric = metric
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        self.results: List[SearchResult] = []
        self.best_result: Optional[SearchResult] = None
    
    def _create_clusterer(self, params: Dict[str, Any]) -> CLUDIClusterer:
        """
        Create a CLUDIClusterer with the given hyperparameters.
        
        Args:
            params: Dictionary of hyperparameters
            
        Returns:
            Configured CLUDIClusterer instance
        """
        return CLUDIClusterer(
            feature_dim=self.feature_dim,
            num_clusters=self.num_clusters,
            device=self.device,
            embedding_dim=params.get('embedding_dim', 64),
            learning_rate=params.get('learning_rate', 0.0001),
            diffusion_steps=params.get('diffusion_steps', 1000),
            batch_diffusion=params.get('batch_diffusion', 8),
            rescaling_factor=params.get('rescaling_factor', 49.0),
            ce_lambda=params.get('ce_lambda', 50.0),
            use_v_prediction=params.get('use_v_prediction', True),
            warmup_epochs=params.get('warmup_epochs', 1)
        )
    
    def _evaluate_params(
        self,
        params: Dict[str, Any],
        features: torch.Tensor,
        labels: torch.Tensor,
        num_epochs: int,
        batch_size: int,
        trial_id: int,
        verbose: bool = False
    ) -> SearchResult:
        """
        Train and evaluate a single hyperparameter configuration.
        
        Args:
            params: Hyperparameter configuration
            features: Training features
            labels: Training labels
            num_epochs: Number of training epochs
            batch_size: Batch size for training
            trial_id: Unique trial identifier
            verbose: Whether to print progress
            
        Returns:
            SearchResult containing metrics and timing
        """
        if verbose:
            print(f"\n{'='*60}")
            print(f"Trial {trial_id}: {params}")
            print(f"{'='*60}")
        
        start_time = time.time()
        
        try:
            # Create and train clusterer
            clusterer = self._create_clusterer(params)
            
            history = clusterer.fit(
                features=features,
                num_epochs=num_epochs,
                batch_size=batch_size,
                verbose=verbose,
                save_checkpoints=False
            )
            
            # Get predictions and evaluate
            predictions = clusterer.predict(features, batch_size=batch_size)
            metrics = evaluate_clustering(labels, predictions)
            
            train_time = time.time() - start_time
            
            if verbose:
                print(f"Trial {trial_id} completed in {train_time:.1f}s")
                print(f"Accuracy: {metrics['accuracy']:.4f}, NMI: {metrics['nmi']:.4f}, ARI: {metrics['ari']:.4f}")
            
        except Exception as e:
            print(f"Trial {trial_id} failed: {e}")
            metrics = {'accuracy': 0.0, 'nmi': 0.0, 'ari': 0.0, 'error': str(e)}
            train_time = time.time() - start_time
        
        return SearchResult(
            params=params,
            metrics=metrics,
            train_time=train_time,
            trial_id=trial_id
        )
    
    def _generate_grid_configs(
        self,
        search_space: CLUDIHyperparameterSpace
    ) -> List[Dict[str, Any]]:
        """
        Generate all configurations for grid search.
        
        Args:
            search_space: Hyperparameter search space
            
        Returns:
            List of all parameter configurations
        """
        space_dict = search_space.to_dict()
        
        # Convert all values to lists
        param_lists = {}
        for key, value in space_dict.items():
            if isinstance(value, list):
                param_lists[key] = value
            elif isinstance(value, tuple):
                # For tuples, we need discrete values - sample 5 points
                if len(value) == 2:
                    if key == 'learning_rate':
                        # Log scale for learning rate
                        param_lists[key] = list(np.logspace(
                            np.log10(value[0]), np.log10(value[1]), 5
                        ))
                    else:
                        param_lists[key] = list(np.linspace(value[0], value[1], 5))
                else:
                    param_lists[key] = [value]
            else:
                param_lists[key] = [value]
        
        # Generate all combinations
        keys = list(param_lists.keys())
        values = list(param_lists.values())
        configs = []
        
        for combo in itertools.product(*values):
            config = dict(zip(keys, combo))
            # Round floats for cleaner representation
            for k, v in config.items():
                if isinstance(v, float):
                    if k == 'learning_rate':
                        config[k] = float(f"{v:.6f}")
                    else:
                        config[k] = round(v, 2)
                elif isinstance(v, (int, np.integer)):
                    config[k] = int(v)
            configs.append(config)
        
        return configs
    
    def _sample_random_config(
        self,
        search_space: CLUDIHyperparameterSpace
    ) -> Dict[str, Any]:
        """
        Sample a random configuration from the search space.
        
        Args:
            search_space: Hyperparameter search space
            
        Returns:
            Randomly sampled configuration
        """
        space_dict = search_space.to_dict()
        config = {}
        
        for key, value in space_dict.items():
            if isinstance(value, list):
                config[key] = random.choice(value)
            elif isinstance(value, tuple):
                if len(value) == 2:
                    low, high = value
                    if key == 'learning_rate':
                        # Log scale sampling
                        config[key] = float(np.exp(
                            random.uniform(np.log(low), np.log(high))
                        ))
                    elif key in ['embedding_dim', 'diffusion_steps', 'batch_diffusion', 'warmup_epochs']:
                        # Integer sampling
                        config[key] = random.randint(int(low), int(high))
                    else:
                        config[key] = random.uniform(low, high)
                elif len(value) == 3:
                    # (low, high, log_scale)
                    low, high, log_scale = value
                    if log_scale:
                        config[key] = float(np.exp(
                            random.uniform(np.log(low), np.log(high))
                        ))
                    else:
                        config[key] = random.uniform(low, high)
            else:
                config[key] = value
        
        # Ensure correct types
        if 'embedding_dim' in config:
            config['embedding_dim'] = int(config['embedding_dim'])
        if 'diffusion_steps' in config:
            config['diffusion_steps'] = int(config['diffusion_steps'])
        if 'batch_diffusion' in config:
            config['batch_diffusion'] = int(config['batch_diffusion'])
        if 'warmup_epochs' in config:
            config['warmup_epochs'] = int(config['warmup_epochs'])
        
        return config
    
    def grid_search(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
        search_space: CLUDIHyperparameterSpace,
        num_epochs: int = 50,
        batch_size: int = 256,
        verbose: bool = True,
        checkpoint_freq: int = 5
    ) -> Tuple[Dict[str, Any], List[SearchResult]]:
        """
        Perform exhaustive grid search over hyperparameter space.
        
        Args:
            features: Training features
            labels: Training labels
            search_space: Hyperparameter search space
            num_epochs: Training epochs per trial
            batch_size: Batch size for training
            verbose: Whether to print progress
            checkpoint_freq: Save results every N trials
            
        Returns:
            Tuple of (best_params, all_results)
        """
        configs = self._generate_grid_configs(search_space)
        total_configs = len(configs)
        
        print(f"\n{'='*60}")
        print(f"CLUDI Grid Search")
        print(f"{'='*60}")
        print(f"Total configurations: {total_configs}")
        print(f"Epochs per trial: {num_epochs}")
        print(f"Metric to optimize: {self.metric}")
        print(f"{'='*60}\n")
        
        self.results = []
        
        for i, config in enumerate(tqdm(configs, desc="Grid Search")):
            result = self._evaluate_params(
                params=config,
                features=features,
                labels=labels,
                num_epochs=num_epochs,
                batch_size=batch_size,
                trial_id=i,
                verbose=verbose
            )
            self.results.append(result)
            
            # Update best
            if self.best_result is None or result.metrics.get(self.metric, 0) > self.best_result.metrics.get(self.metric, 0):
                self.best_result = result
                if verbose:
                    print(f"New best! {self.metric}: {result.metrics.get(self.metric, 0):.4f}")
            
            # Checkpoint
            if (i + 1) % checkpoint_freq == 0:
                self._save_results("grid_search_checkpoint")
        
        self._save_results("grid_search_final")
        
        return self.best_result.params, self.results
    
    def random_search(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
        search_space: CLUDIHyperparameterSpace,
        n_trials: int = 20,
        num_epochs: int = 50,
        batch_size: int = 256,
        verbose: bool = True,
        seed: Optional[int] = None,
        checkpoint_freq: int = 5
    ) -> Tuple[Dict[str, Any], List[SearchResult]]:
        """
        Perform random search over hyperparameter space.
        
        Args:
            features: Training features
            labels: Training labels
            search_space: Hyperparameter search space
            n_trials: Number of random configurations to try
            num_epochs: Training epochs per trial
            batch_size: Batch size for training
            verbose: Whether to print progress
            seed: Random seed for reproducibility
            checkpoint_freq: Save results every N trials
            
        Returns:
            Tuple of (best_params, all_results)
        """
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        print(f"\n{'='*60}")
        print(f"CLUDI Random Search")
        print(f"{'='*60}")
        print(f"Number of trials: {n_trials}")
        print(f"Epochs per trial: {num_epochs}")
        print(f"Metric to optimize: {self.metric}")
        print(f"{'='*60}\n")
        
        self.results = []
        
        for i in tqdm(range(n_trials), desc="Random Search"):
            config = self._sample_random_config(search_space)
            
            result = self._evaluate_params(
                params=config,
                features=features,
                labels=labels,
                num_epochs=num_epochs,
                batch_size=batch_size,
                trial_id=i,
                verbose=verbose
            )
            self.results.append(result)
            
            # Update best
            if self.best_result is None or result.metrics.get(self.metric, 0) > self.best_result.metrics.get(self.metric, 0):
                self.best_result = result
                if verbose:
                    print(f"New best! {self.metric}: {result.metrics.get(self.metric, 0):.4f}")
            
            # Checkpoint
            if (i + 1) % checkpoint_freq == 0:
                self._save_results("random_search_checkpoint")
        
        self._save_results("random_search_final")
        
        return self.best_result.params, self.results
    
    def bayesian_search(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
        search_space: CLUDIHyperparameterSpace,
        n_trials: int = 30,
        num_epochs: int = 50,
        batch_size: int = 256,
        verbose: bool = True,
        seed: Optional[int] = None
    ) -> Tuple[Dict[str, Any], List[SearchResult]]:
        """
        Perform Bayesian optimization using Optuna.
        
        This method uses sequential model-based optimization to efficiently
        explore the hyperparameter space.
        
        Args:
            features: Training features
            labels: Training labels
            search_space: Hyperparameter search space
            n_trials: Number of optimization trials
            num_epochs: Training epochs per trial
            batch_size: Batch size for training
            verbose: Whether to print progress
            seed: Random seed for reproducibility
            
        Returns:
            Tuple of (best_params, all_results)
            
        Raises:
            ImportError: If Optuna is not installed
        """
        try:
            import optuna
            from optuna.samplers import TPESampler
        except ImportError:
            raise ImportError(
                "Optuna is required for Bayesian optimization. "
                "Install with: pip install optuna"
            )
        
        space_dict = search_space.to_dict()
        
        def objective(trial: optuna.Trial) -> float:
            """Optuna objective function."""
            params = {}
            
            for key, value in space_dict.items():
                if isinstance(value, list):
                    if all(isinstance(v, int) for v in value):
                        params[key] = trial.suggest_categorical(key, value)
                    elif all(isinstance(v, float) for v in value):
                        params[key] = trial.suggest_categorical(key, value)
                    else:
                        params[key] = trial.suggest_categorical(key, value)
                elif isinstance(value, tuple):
                    if len(value) == 2:
                        low, high = value
                        if key == 'learning_rate':
                            params[key] = trial.suggest_float(key, low, high, log=True)
                        elif key in ['embedding_dim', 'diffusion_steps', 'batch_diffusion', 'warmup_epochs']:
                            params[key] = trial.suggest_int(key, int(low), int(high))
                        else:
                            params[key] = trial.suggest_float(key, low, high)
                else:
                    params[key] = value
            
            result = self._evaluate_params(
                params=params,
                features=features,
                labels=labels,
                num_epochs=num_epochs,
                batch_size=batch_size,
                trial_id=trial.number,
                verbose=verbose
            )
            self.results.append(result)
            
            return result.metrics.get(self.metric, 0.0)
        
        print(f"\n{'='*60}")
        print(f"CLUDI Bayesian Optimization (Optuna)")
        print(f"{'='*60}")
        print(f"Number of trials: {n_trials}")
        print(f"Epochs per trial: {num_epochs}")
        print(f"Metric to optimize: {self.metric}")
        print(f"{'='*60}\n")
        
        self.results = []
        
        sampler = TPESampler(seed=seed) if seed is not None else TPESampler()
        study = optuna.create_study(
            direction="maximize",
            sampler=sampler,
            study_name="cludi_hyperparam_search"
        )
        
        study.optimize(
            objective,
            n_trials=n_trials,
            show_progress_bar=True
        )
        
        # Get best result
        best_params = study.best_params
        best_value = study.best_value
        
        # Find the corresponding result
        for result in self.results:
            if result.metrics.get(self.metric, 0) == best_value:
                self.best_result = result
                break
        
        if self.best_result is None and self.results:
            self.best_result = max(self.results, key=lambda r: r.metrics.get(self.metric, 0))
        
        self._save_results("bayesian_search_final")
        
        # Save Optuna study
        study_path = self.results_dir / "optuna_study.json"
        with open(study_path, 'w') as f:
            json.dump({
                'best_params': best_params,
                'best_value': best_value,
                'n_trials': len(study.trials)
            }, f, indent=2)
        
        return self.best_result.params if self.best_result else best_params, self.results
    
    def search(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
        search_space: Optional[CLUDIHyperparameterSpace] = None,
        method: str = "random",
        n_trials: int = 20,
        num_epochs: int = 50,
        batch_size: int = 256,
        verbose: bool = True,
        seed: Optional[int] = None,
        **kwargs
    ) -> Tuple[Dict[str, Any], List[SearchResult]]:
        """
        Perform hyperparameter search using the specified method.
        
        This is the main entry point for hyperparameter search.
        
        Args:
            features: Training features
            labels: Training labels
            search_space: Hyperparameter search space (uses default if None)
            method: Search method ('grid', 'random', 'bayesian')
            n_trials: Number of trials for random/bayesian search
            num_epochs: Training epochs per trial
            batch_size: Batch size for training
            verbose: Whether to print progress
            seed: Random seed for reproducibility
            **kwargs: Additional method-specific arguments
            
        Returns:
            Tuple of (best_params, all_results)
        """
        if search_space is None:
            search_space = CLUDIHyperparameterSpace()
        
        if method == "grid":
            return self.grid_search(
                features=features,
                labels=labels,
                search_space=search_space,
                num_epochs=num_epochs,
                batch_size=batch_size,
                verbose=verbose,
                **kwargs
            )
        elif method == "random":
            return self.random_search(
                features=features,
                labels=labels,
                search_space=search_space,
                n_trials=n_trials,
                num_epochs=num_epochs,
                batch_size=batch_size,
                verbose=verbose,
                seed=seed,
                **kwargs
            )
        elif method == "bayesian":
            return self.bayesian_search(
                features=features,
                labels=labels,
                search_space=search_space,
                n_trials=n_trials,
                num_epochs=num_epochs,
                batch_size=batch_size,
                verbose=verbose,
                seed=seed,
                **kwargs
            )
        else:
            raise ValueError(f"Unknown search method: {method}. Use 'grid', 'random', or 'bayesian'.")
    
    def _save_results(self, name: str):
        """Save search results to disk."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_path = self.results_dir / f"{name}_{timestamp}.json"
        
        results_data = {
            'feature_dim': self.feature_dim,
            'num_clusters': self.num_clusters,
            'metric': self.metric,
            'best_result': {
                'params': self.best_result.params,
                'metrics': self.best_result.metrics,
                'train_time': self.best_result.train_time,
                'trial_id': self.best_result.trial_id
            } if self.best_result else None,
            'all_results': [
                {
                    'params': r.params,
                    'metrics': r.metrics,
                    'train_time': r.train_time,
                    'trial_id': r.trial_id
                }
                for r in self.results
            ]
        }
        
        with open(results_path, 'w') as f:
            json.dump(results_data, f, indent=2)
        
        print(f"Results saved to {results_path}")
    
    def print_summary(self):
        """Print a summary of the search results."""
        if not self.results:
            print("No search results available.")
            return
        
        print(f"\n{'='*60}")
        print("Hyperparameter Search Summary")
        print(f"{'='*60}")
        print(f"Total trials: {len(self.results)}")
        print(f"Metric optimized: {self.metric}")
        
        if self.best_result:
            print(f"\nBest configuration:")
            for key, value in self.best_result.params.items():
                print(f"  {key}: {value}")
            print(f"\nBest metrics:")
            for key, value in self.best_result.metrics.items():
                if isinstance(value, float):
                    print(f"  {key}: {value:.4f}")
                else:
                    print(f"  {key}: {value}")
            print(f"\nTraining time: {self.best_result.train_time:.1f}s")
        
        # Top 5 results
        sorted_results = sorted(
            self.results,
            key=lambda r: r.metrics.get(self.metric, 0),
            reverse=True
        )[:5]
        
        print(f"\nTop 5 configurations by {self.metric}:")
        for i, result in enumerate(sorted_results, 1):
            print(f"  {i}. {self.metric}={result.metrics.get(self.metric, 0):.4f}, "
                  f"params={result.params}")
        
        print(f"{'='*60}\n")


def run_cludi_hyperparam_search(
    features: torch.Tensor,
    labels: torch.Tensor,
    feature_dim: int,
    num_clusters: int,
    search_method: str = "random",
    n_trials: int = 20,
    num_epochs: int = 50,
    batch_size: int = 256,
    device: str = "cuda",
    metric: str = "accuracy",
    results_dir: str = "./hyperparam_search_results",
    search_space: Optional[Dict[str, Any]] = None,
    seed: Optional[int] = None,
    verbose: bool = True
) -> Tuple[Dict[str, Any], List[SearchResult]]:
    """
    Convenience function to run CLUDI hyperparameter search.
    
    Args:
        features: Training features
        labels: Training labels
        feature_dim: Dimension of input features
        num_clusters: Number of clusters
        search_method: Search method ('grid', 'random', 'bayesian')
        n_trials: Number of trials
        num_epochs: Training epochs per trial
        batch_size: Batch size
        device: Computation device
        metric: Metric to optimize
        results_dir: Directory for results
        search_space: Custom search space dictionary (optional)
        seed: Random seed
        verbose: Whether to print progress
        
    Returns:
        Tuple of (best_params, all_results)
    """
    # Create search space
    if search_space is not None:
        space = CLUDIHyperparameterSpace.from_dict(search_space)
    else:
        space = CLUDIHyperparameterSpace()
    
    # Create searcher
    searcher = CLUDIHyperparameterSearch(
        feature_dim=feature_dim,
        num_clusters=num_clusters,
        device=device,
        metric=metric,
        results_dir=results_dir
    )
    
    # Run search
    best_params, results = searcher.search(
        features=features,
        labels=labels,
        search_space=space,
        method=search_method,
        n_trials=n_trials,
        num_epochs=num_epochs,
        batch_size=batch_size,
        verbose=verbose,
        seed=seed
    )
    
    # Print summary
    searcher.print_summary()
    
    return best_params, results
