"""
Model Registry for storing and retrieving trained models
"""

import json
import logging
import pickle
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)


class ModelRegistry:
    """Registry for storing and retrieving trained models"""

    def __init__(self, registry_path: str = "models/registry"):
        self.registry_path = Path(registry_path)
        self.registry_path.mkdir(parents=True, exist_ok=True)
        self.metadata_file = self.registry_path / "metadata.json"
        self.metadata = self._load_metadata()

    def _load_metadata(self) -> dict[str, Any]:
        """Load metadata from file"""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file) as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load metadata: {e}")
                return {}
        return {}

    def _save_metadata(self):
        """Save metadata to file"""
        try:
            with open(self.metadata_file, "w") as f:
                json.dump(self.metadata, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save metadata: {e}")

    async def save_model(
        self,
        model: Any,
        preprocessing_pipeline: Any,
        meta_features: dict[str, Any],
        leaderboard: pd.DataFrame,
        config: Any,
    ) -> str:
        """Save a trained model to the registry"""
        model_id = f"model_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        model_dir = self.registry_path / model_id
        model_dir.mkdir(exist_ok=True)

        try:
            # Save model
            model_path = model_dir / "model.pkl"
            with open(model_path, "wb") as f:
                pickle.dump(model, f)

            # Save preprocessing pipeline
            pipeline_path = model_dir / "pipeline.pkl"
            with open(pipeline_path, "wb") as f:
                pickle.dump(preprocessing_pipeline, f)

            # Save metadata
            metadata = {
                "model_id": model_id,
                "timestamp": datetime.now().isoformat(),
                "meta_features": meta_features,
                "config": {
                    "dataset_path": config.dataset_path,
                    "target_column": config.target_column,
                    "optimization_metric": config.optimization_metric,
                    "cross_validation_folds": config.cross_validation_folds,
                    "random_state": config.random_state,
                },
                "leaderboard": (
                    leaderboard.to_dict("records") if not leaderboard.empty else []
                ),
                "model_type": model.__class__.__name__,
                "model_path": str(model_path),
                "pipeline_path": str(pipeline_path),
            }

            metadata_path = model_dir / "metadata.json"
            with open(metadata_path, "w") as f:
                json.dump(metadata, f, indent=2)

            # Update global metadata
            self.metadata[model_id] = {
                "timestamp": metadata["timestamp"],
                "model_type": metadata["model_type"],
                "meta_features": meta_features,
                "performance": (
                    leaderboard.iloc[0].to_dict() if not leaderboard.empty else {}
                ),
            }
            self._save_metadata()

            logger.info(f"Model saved to registry: {model_id}")
            return model_id

        except Exception as e:
            logger.error(f"Failed to save model: {e}")
            raise

    async def load_model(self, model_id: str) -> dict[str, Any]:
        """Load a model from the registry"""
        model_dir = self.registry_path / model_id
        metadata_path = model_dir / "metadata.json"

        if not metadata_path.exists():
            raise ValueError(f"Model {model_id} not found in registry")

        try:
            with open(metadata_path) as f:
                metadata = json.load(f)

            # Load model
            model_path = model_dir / "model.pkl"
            with open(model_path, "rb") as f:
                model = pickle.load(f)

            # Load preprocessing pipeline
            pipeline_path = model_dir / "pipeline.pkl"
            with open(pipeline_path, "rb") as f:
                pipeline = pickle.load(f)

            return {"model": model, "pipeline": pipeline, "metadata": metadata}

        except Exception as e:
            logger.error(f"Failed to load model {model_id}: {e}")
            raise

    async def get_meta_learning_warm_starts(
        self, meta_features: dict[str, Any]
    ) -> dict[str, Any]:
        """Get meta-learning warm starts based on similar datasets"""
        warm_starts = {}

        try:
            # Find similar datasets based on meta-features
            similar_models = []
            for model_id, model_meta in self.metadata.items():
                similarity_score = self._calculate_similarity(
                    meta_features, model_meta.get("meta_features", {})
                )
                if similarity_score > 0.7:  # Threshold for similarity
                    similar_models.append((model_id, similarity_score))

            # Sort by similarity and get top models
            similar_models.sort(key=lambda x: x[1], reverse=True)

            for model_id, score in similar_models[:3]:  # Top 3 similar models
                try:
                    model_data = await self.load_model(model_id)
                    warm_starts[model_id] = {
                        "similarity_score": score,
                        "performance": model_data["metadata"].get("leaderboard", [{}])[
                            0
                        ],
                        "meta_features": model_data["metadata"].get(
                            "meta_features", {}
                        ),
                    }
                except Exception as e:
                    logger.warning(f"Failed to load warm start model {model_id}: {e}")
                    continue

            return warm_starts

        except Exception as e:
            logger.warning(f"Failed to get meta-learning warm starts: {e}")
            return {}

    def _calculate_similarity(
        self, features1: dict[str, Any], features2: dict[str, Any]
    ) -> float:
        """Calculate similarity between two sets of meta-features"""
        if not features1 or not features2:
            return 0.0

        # Simple similarity calculation based on common features
        common_features = set(features1.keys()) & set(features2.keys())
        if not common_features:
            return 0.0

        similarities = []
        for feature in common_features:
            val1 = features1[feature]
            val2 = features2[feature]

            if isinstance(val1, int | float) and isinstance(val2, int | float):
                # Numerical similarity
                if val1 == 0 and val2 == 0:
                    similarities.append(1.0)
                else:
                    similarities.append(
                        1.0 - abs(val1 - val2) / max(abs(val1), abs(val2), 1)
                    )
            elif isinstance(val1, str) and isinstance(val2, str):
                # String similarity
                similarities.append(1.0 if val1 == val2 else 0.0)
            else:
                similarities.append(0.0)

        return sum(similarities) / len(similarities) if similarities else 0.0

    def list_models(self) -> list[dict[str, Any]]:
        """List all models in the registry"""
        models = []
        for model_id, model_meta in self.metadata.items():
            models.append(
                {
                    "model_id": model_id,
                    "timestamp": model_meta.get("timestamp"),
                    "model_type": model_meta.get("model_type"),
                    "performance": model_meta.get("performance", {}),
                    "meta_features": model_meta.get("meta_features", {}),
                }
            )

        return sorted(models, key=lambda x: x["timestamp"], reverse=True)

    def delete_model(self, model_id: str) -> bool:
        """Delete a model from the registry"""
        try:
            model_dir = self.registry_path / model_id
            if model_dir.exists():
                import shutil

                shutil.rmtree(model_dir)

            if model_id in self.metadata:
                del self.metadata[model_id]
                self._save_metadata()

            logger.info(f"Model {model_id} deleted from registry")
            return True

        except Exception as e:
            logger.error(f"Failed to delete model {model_id}: {e}")
            return False

    def get_registry_stats(self) -> dict[str, Any]:
        """Get statistics about the registry"""
        return {
            "total_models": len(self.metadata),
            "registry_path": str(self.registry_path),
            "last_updated": max(
                [meta.get("timestamp", "") for meta in self.metadata.values()],
                default="Never",
            ),
        }
