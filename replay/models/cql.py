"""
Using CQL implementation from `d3rlpy` package.
"""
import io
import logging
import tempfile
import timeit
from typing import Optional, Dict, Any

import d3rlpy.algos.cql as CQL_d3rlpy
import numpy as np
import pandas as pd
import torch
from d3rlpy.argument_utility import (
    EncoderArg, QFuncArg, UseGPUArg, ScalerArg, ActionScalerArg,
    RewardScalerArg
)
from d3rlpy.base import ImplBase, LearnableBase, _serialize_params
from d3rlpy.constants import IMPL_NOT_INITIALIZED_ERROR
from d3rlpy.context import disable_parallel
from d3rlpy.dataset import MDPDataset
from d3rlpy.models.encoders import create_encoder_factory
from d3rlpy.models.optimizers import OptimizerFactory, AdamFactory
from d3rlpy.models.q_functions import create_q_func_factory
from d3rlpy.preprocessing import create_scaler, create_action_scaler, create_reward_scaler
from pyspark.sql import DataFrame, functions as sf, Window

from replay.constants import REC_SCHEMA
from replay.models.base_rec import Recommender
from replay.utils import assert_omp_single_thread


timer = timeit.default_timer


class CQL(Recommender):
    """Conservative Q-Learning algorithm.

    CQL is a SAC-based data-driven deep reinforcement learning algorithm, which
    achieves state-of-the-art performance in offline RL problems.

    CQL mitigates overestimation error by minimizing action-values under the
    current policy and maximizing values under data distribution for
    underestimation issue.

    .. math::
        L(\\theta_i) = \\alpha\\, \\mathbb{E}_{s_t \\sim D}
        \\left[\\log{\\sum_a \\exp{Q_{\\theta_i}(s_t, a)}} - \\mathbb{E}_{a \\sim D} \\big[Q_{\\theta_i}(s_t, a)\\big] - \\tau\\right]
        + L_\\mathrm{SAC}(\\theta_i)

    where :math:`\alpha` is an automatically adjustable value via Lagrangian
    dual gradient descent and :math:`\tau` is a threshold value.
    If the action-value difference is smaller than :math:`\tau`, the
    :math:`\alpha` will become smaller.
    Otherwise, the :math:`\alpha` will become larger to aggressively penalize
    action-values.

    In continuous control, :math:`\log{\sum_a \exp{Q(s, a)}}` is computed as
    follows.

    .. math::
        \\log{\\sum_a \\exp{Q(s, a)}} \\approx \\log{\\left(
        \\frac{1}{2N} \\sum_{a_i \\sim \\text{Unif}(a)}^N
            \\left[\\frac{\\exp{Q(s, a_i)}}{\\text{Unif}(a)}\\right]
        + \\frac{1}{2N} \\sum_{a_i \\sim \\pi_\\phi(a|s)}^N
            \\left[\\frac{\\exp{Q(s, a_i)}}{\\pi_\\phi(a_i|s)}\\right]\\right)}

    where :math:`N` is the number of sampled actions.

    An implementation of this algorithm is heavily based on the corresponding implementation
    in the d3rlpy library (see https://github.com/takuseno/d3rlpy/blob/master/d3rlpy/algos/cql.py)

    The rest of optimization is exactly same as :class:`d3rlpy.algos.SAC`.

    References:
        * `Kumar et al., Conservative Q-Learning for Offline Reinforcement
          Learning. <https://arxiv.org/abs/2006.04779>`_

    Args:
        n_epochs (int): the number of epochs to learn.
        mdp_dataset_builder (MdpDatasetBuilder): the MDP dataset builder from users' log.
        actor_learning_rate (float): learning rate for policy function.
        critic_learning_rate (float): learning rate for Q functions.
        temp_learning_rate (float): learning rate for temperature parameter of SAC.
        alpha_learning_rate (float): learning rate for :math:`\alpha`.
        actor_optim_factory (d3rlpy.models.optimizers.OptimizerFactory):
        optimizer factory for the actor.
        The available options are `[SGD, Adam or RMSprop]`.
        critic_optim_factory (d3rlpy.models.optimizers.OptimizerFactory):
        optimizer factory for the critic.
        The available options are `[SGD, Adam or RMSprop]`.
        temp_optim_factory (d3rlpy.models.optimizers.OptimizerFactory):
        optimizer factory for the temperature.
        The available options are `[SGD, Adam or RMSprop]`.
        alpha_optim_factory (d3rlpy.models.optimizers.OptimizerFactory):
        optimizer factory for :math:`\alpha`.
        The available options are `[SGD, Adam or RMSprop]`.
        actor_encoder_factory (d3rlpy.models.encoders.EncoderFactory or str):
        encoder factory for the actor.
        The available options are `['pixel', 'dense', 'vector', 'default']`.
        See d3rlpy.models.encoders.EncoderFactory for details.
        critic_encoder_factory (d3rlpy.models.encoders.EncoderFactory or str):
        encoder factory for the critic.
        The available options are `['pixel', 'dense', 'vector', 'default']`.
        See d3rlpy.models.encoders.EncoderFactory for details.
        q_func_factory (d3rlpy.models.q_functions.QFunctionFactory or str):
        Q function factory. The available options are `['mean', 'qr', 'iqn', 'fqf']`.
        See d3rlpy.models.q_functions.QFunctionFactory for details.
        batch_size (int): mini-batch size.
        n_frames (int): the number of frames to stack for image observation.
        n_steps (int): N-step TD calculation.
        gamma (float): discount factor.
        tau (float): target network synchronization coefficient.
        n_critics (int): the number of Q functions for ensemble.
        initial_temperature (float): initial temperature value.
        initial_alpha (float): initial :math:`\alpha` value.
        alpha_threshold (float): threshold value described as :math:`\tau`.
        conservative_weight (float): constant weight to scale conservative loss.
        n_action_samples (int): the number of sampled actions to compute
        :math:`\log{\sum_a \exp{Q(s, a)}}`.
        soft_q_backup (bool): flag to use SAC-style backup.
        use_gpu (bool, int or d3rlpy.gpu.Device):
        flag to use GPU, device ID or device.
        scaler (d3rlpy.preprocessing.Scaler or str): preprocessor.
        The available options are `['pixel', 'min_max', 'standard']`.
        action_scaler (d3rlpy.preprocessing.ActionScaler or str):
        action preprocessor. The available options are `['min_max']`.
        reward_scaler (d3rlpy.preprocessing.RewardScaler or str):
        reward preprocessor. The available options are
        `['clip', 'min_max', 'standard']`.
        impl (d3rlpy.algos.torch.cql_impl.CQLImpl): algorithm implementation.
    """

    n_epochs: int
    mdp_dataset_builder: 'MdpDatasetBuilder'
    model: CQL_d3rlpy.CQL

    can_predict_cold_users = True

    _observation_shape = (2, )
    _action_size = 1

    _search_space = {
        "actor_learning_rate": {"type": "loguniform", "args": [1e-5, 1e-3]},
        "critic_learning_rate": {"type": "loguniform", "args": [3e-5, 3e-4]},
        "n_epochs": {"type": "int", "args": [3, 20]},
        "temp_learning_rate": {"type": "loguniform", "args": [1e-5, 1e-3]},
        "alpha_learning_rate": {"type": "loguniform", "args": [1e-5, 1e-3]},
        "gamma": {"type": "loguniform", "args": [0.9, 0.999]},
        "n_critics": {"type": "int", "args": [2, 4]},
    }

    # pylint: disable=too-many-arguments, too-many-locals
    def __init__(
            self,
            mdp_dataset_builder: 'MdpDatasetBuilder',
            n_epochs: int = 1,

            # CQL inner params
            actor_learning_rate: float = 1e-4,
            critic_learning_rate: float = 3e-4,
            temp_learning_rate: float = 1e-4,
            alpha_learning_rate: float = 1e-4,
            actor_optim_factory: OptimizerFactory = AdamFactory(),
            critic_optim_factory: OptimizerFactory = AdamFactory(),
            temp_optim_factory: OptimizerFactory = AdamFactory(),
            alpha_optim_factory: OptimizerFactory = AdamFactory(),
            actor_encoder_factory: EncoderArg = "default",
            critic_encoder_factory: EncoderArg = "default",
            q_func_factory: QFuncArg = "mean",
            batch_size: int = 64,
            n_frames: int = 1,
            n_steps: int = 1,
            gamma: float = 0.99,
            tau: float = 0.005,
            n_critics: int = 2,
            initial_temperature: float = 1.0,
            initial_alpha: float = 1.0,
            alpha_threshold: float = 10.0,
            conservative_weight: float = 5.0,
            n_action_samples: int = 10,
            soft_q_backup: bool = False,
            use_gpu: UseGPUArg = False,
            scaler: ScalerArg = None,
            action_scaler: ActionScalerArg = None,
            reward_scaler: RewardScalerArg = None,
            **params
    ):
        super().__init__()
        self.n_epochs = n_epochs
        assert_omp_single_thread()

        if isinstance(actor_optim_factory, dict):
            self.logger.info('-- Desiarializing CQL parameters')
            actor_optim_factory = _deserialize_param('actor_optim_factory', actor_optim_factory)
            critic_optim_factory = _deserialize_param('critic_optim_factory', critic_optim_factory)
            temp_optim_factory = _deserialize_param('temp_optim_factory', temp_optim_factory)
            alpha_optim_factory = _deserialize_param('alpha_optim_factory', alpha_optim_factory)
            actor_encoder_factory = _deserialize_param(
                'actor_encoder_factory', actor_encoder_factory
            )
            critic_encoder_factory = _deserialize_param(
                'critic_encoder_factory', critic_encoder_factory
            )
            q_func_factory = _deserialize_param('q_func_factory', q_func_factory)
            scaler = _deserialize_param('scaler', scaler)
            action_scaler = _deserialize_param('action_scaler', action_scaler)
            reward_scaler = _deserialize_param('reward_scaler', reward_scaler)
            # non-model params
            mdp_dataset_builder = _deserialize_param('mdp_dataset_builder', mdp_dataset_builder)

        self.mdp_dataset_builder = mdp_dataset_builder

        self.model = CQL_d3rlpy.CQL(
            actor_learning_rate=actor_learning_rate,
            critic_learning_rate=critic_learning_rate,
            temp_learning_rate=temp_learning_rate,
            alpha_learning_rate=alpha_learning_rate,
            actor_optim_factory=actor_optim_factory,
            critic_optim_factory=critic_optim_factory,
            temp_optim_factory=temp_optim_factory,
            alpha_optim_factory=alpha_optim_factory,
            actor_encoder_factory=actor_encoder_factory,
            critic_encoder_factory=critic_encoder_factory,
            q_func_factory=q_func_factory,
            batch_size=batch_size,
            n_frames=n_frames,
            n_steps=n_steps,
            gamma=gamma,
            tau=tau,
            n_critics=n_critics,
            initial_temperature=initial_temperature,
            initial_alpha=initial_alpha,
            alpha_threshold=alpha_threshold,
            conservative_weight=conservative_weight,
            n_action_samples=n_action_samples,
            soft_q_backup=soft_q_backup,
            use_gpu=use_gpu,
            scaler=scaler,
            action_scaler=action_scaler,
            reward_scaler=reward_scaler,
            **params
        )

        # explicitly create the model's algorithm implementation at init stage
        # despite the lazy on-fit init convention in d3rlpy a) to avoid serialization
        # complications and b) to make model ready for prediction even before fitting
        self.model.create_impl(
            observation_shape=self._observation_shape,
            action_size=self._action_size
        )

    def _fit(
        self,
        log: DataFrame,
        user_features: Optional[DataFrame] = None,
        item_features: Optional[DataFrame] = None,
    ) -> None:
        mdp_dataset: MDPDataset = self.mdp_dataset_builder.build(log)
        self.model.fit(mdp_dataset, n_epochs=self.n_epochs)

    @staticmethod
    def _predict_pairs_inner(
        model: bytes,
        user_idx: int,
        items: np.ndarray,
    ) -> pd.DataFrame:
        user_item_pairs = pd.DataFrame({
            'user_idx': np.repeat(user_idx, len(items)),
            'item_idx': items
        })

        # deserialize model policy and predict items relevance for the user
        policy = CQL._deserialize_policy(model)
        items_batch = user_item_pairs.to_numpy()
        user_item_pairs['relevance'] = CQL._predict_relevance_with_policy(policy, items_batch)

        return user_item_pairs

    # pylint: disable=too-many-arguments
    def _predict(
        self,
        log: DataFrame,
        k: int,
        users: DataFrame,
        items: DataFrame,
        user_features: Optional[DataFrame] = None,
        item_features: Optional[DataFrame] = None,
        filter_seen_items: bool = True,
    ) -> DataFrame:
        available_items = items.toPandas()["item_idx"].values
        policy_bytes = self._serialize_policy()

        def grouped_map(log_slice: pd.DataFrame) -> pd.DataFrame:
            return CQL._predict_pairs_inner(
                model=policy_bytes,
                user_idx=log_slice["user_idx"][0],
                items=available_items,
            )[["user_idx", "item_idx", "relevance"]]

        # predict relevance for all available items and return them as is;
        # `filter_seen_items` and top `k` params are ignored
        self.logger.debug("Predict started")
        return users.groupby("user_idx").applyInPandas(grouped_map, REC_SCHEMA)

    def _predict_pairs(
        self,
        pairs: DataFrame,
        log: Optional[DataFrame] = None,
        user_features: Optional[DataFrame] = None,
        item_features: Optional[DataFrame] = None,
    ) -> DataFrame:
        policy_bytes = self._serialize_policy()

        def grouped_map(user_log: pd.DataFrame) -> pd.DataFrame:
            return CQL._predict_pairs_inner(
                model=policy_bytes,
                user_idx=user_log["user_idx"][0],
                items=np.array(user_log["item_idx_to_pred"][0]),
            )[["user_idx", "item_idx", "relevance"]]

        self.logger.debug("Calculate relevance for user-item pairs")
        return (
            pairs
            .groupBy("user_idx")
            .agg(sf.collect_list("item_idx").alias("item_idx_to_pred"))
            .join(log.select("user_idx").distinct(), on="user_idx", how="inner")
            .groupby("user_idx")
            .applyInPandas(grouped_map, REC_SCHEMA)
        )

    @property
    def _init_args(self) -> Dict[str, Any]:
        return dict(
            # non-model hyperparams
            n_epochs=self.n_epochs,
            mdp_dataset_builder=self.mdp_dataset_builder.init_args(),

            # model internal hyperparams
            **self._get_model_hyperparams()
        )

    def _save_model(self, path: str) -> None:
        self.logger.info('-- Saving model to %s', path)
        self.model.save_model(path)

    def _load_model(self, path: str) -> None:
        self.logger.info('-- Loading model from %s', path)
        self.model.load_model(path)

    def _get_model_hyperparams(self) -> Dict[str, Any]:
        """Get model hyperparams as dictionary.

        NB: The code is taken from a `d3rlpy.LearnableBase.save_params(logger)` method as
        there's no method to just return such params without saving them.
        """
        assert self.model._impl is not None, IMPL_NOT_INITIALIZED_ERROR

        # pylint: disable=invalid-name
        # get hyperparameters without impl
        params = {}
        with disable_parallel():
            for k, v in self.model.get_params(deep=False).items():
                if isinstance(v, (ImplBase, LearnableBase)):
                    continue
                params[k] = v

        # save algorithm name
        params["algorithm"] = self.model.__class__.__name__

        # serialize objects
        params = _serialize_params(params)
        return params

    def _serialize_policy(self) -> bytes:
        # store using temporary file and immediately read serialized version
        with tempfile.NamedTemporaryFile(suffix='.pt') as tmp:
            # noinspection PyProtectedMember
            self.model._impl.save_policy(tmp.name)
            with open(tmp.name, 'rb') as policy_file:
                return policy_file.read()

    @staticmethod
    def _deserialize_policy(policy: bytes) -> torch.jit.ScriptModule:
        with io.BytesIO(policy) as buffer:
            return torch.jit.load(buffer, map_location=torch.device('cpu'))

    @staticmethod
    def _predict_relevance_with_policy(
            policy: torch.jit.ScriptModule, items: np.ndarray
    ) -> np.ndarray:
        items = torch.from_numpy(items).float().cpu()
        with torch.no_grad():
            return policy.forward(items).numpy()


def _deserialize_param(name: str, value: Any) -> Any:
    if not isinstance(value, dict):
        # not a serialized object
        return value

    if name == "scaler":
        value = create_scaler(value["type"], **value["params"])
    elif name == "action_scaler":
        value = create_action_scaler(value["type"], **value["params"])
    elif name == "reward_scaler":
        value = create_reward_scaler(value["type"], **value["params"])
    elif "optim_factory" in name:
        value = OptimizerFactory(**value)
    elif "encoder_factory" in name:
        value = create_encoder_factory(value["type"], **value["params"])
    elif name == "q_func_factory":
        value = create_q_func_factory(value["type"], **value["params"])
    elif name == "mdp_dataset_builder":
        value = MdpDatasetBuilder(**value)
    return value


class MdpDatasetBuilder:
    r"""
    Markov Decision Process Dataset builder.
    This class transforms datasets with user logs, which is natural for recommender systems,
    to datasets consisting of users' decision-making session logs, which is natural for RL methods.

    Args:
        top_k (int): the number of top user items to learn predicting.
        action_randomization_scale (float): the scale of action randomization gaussian noise.
    """
    logger: logging.Logger
    top_k: int
    action_randomization_scale: float

    def __init__(self, top_k: int, action_randomization_scale: float = 1e-3):
        self.logger = logging.getLogger("replay")
        self.top_k = top_k
        # cannot set zero scale as then d3rlpy will treat transitions as discrete
        assert action_randomization_scale > 0
        self.action_randomization_scale = action_randomization_scale

    def build(self, log: DataFrame) -> MDPDataset:
        """Builds and returns MDP dataset from users' log."""

        start_time = timer()
        # reward top-K watched movies with 1, the others - with 0
        reward_condition = sf.row_number().over(
            Window
            .partitionBy('user_idx')
            .orderBy([sf.desc('relevance'), sf.desc('timestamp')])
        ) <= self.top_k

        # every user has his own episode (the latest item is defined as terminal)
        terminal_condition = sf.row_number().over(
            Window
            .partitionBy('user_idx')
            .orderBy(sf.desc('timestamp'))
        ) == 1

        user_logs = (
            log
            .withColumn("reward", sf.when(reward_condition, sf.lit(1)).otherwise(sf.lit(0)))
            .withColumn("terminal", sf.when(terminal_condition, sf.lit(1)).otherwise(sf.lit(0)))
            .withColumn(
                "action",
                sf.col("relevance").cast("float") + sf.randn() * self.action_randomization_scale
            )
            .orderBy(['user_idx', 'timestamp'], ascending=True)
            .select(['user_idx', 'item_idx', 'action', 'reward', 'terminal'])
            .toPandas()
        )
        train_dataset = MDPDataset(
            observations=np.array(user_logs[['user_idx', 'item_idx']]),
            actions=user_logs['action'].to_numpy()[:, None],
            rewards=user_logs['reward'].to_numpy(),
            terminals=user_logs['terminal'].to_numpy()
        )

        prepare_time = timer() - start_time
        self.logger.info('-- Building MDP dataset took %.2f seconds', prepare_time)
        return train_dataset

    # pylint: disable=missing-function-docstring
    def init_args(self):
        return dict(
            top_k=self.top_k,
            action_randomization_scale=self.action_randomization_scale
        )
