from pathlib import Path

from nerfstudio.cameras.camera_optimizers import CameraOptimizerConfig
from nerfstudio.configs.base_config import ViewerConfig, LoggingConfig
from nerfstudio.engine.optimizers import AdamOptimizerConfig
from nerfstudio.engine.schedulers import ExponentialDecaySchedulerConfig
from nerfstudio.plugins.types import MethodSpecification

from .nerf_trainer import NerfTrainerConfig, RMSpropOptimizerConfig
from .nerf_pipeline import NerfPipelineConfig
from .nerf_datamanager import NerfDataManagerConfig
from .nerf_model import NerfModelConfig

from .spin_dataparser import SpinDataParserConfig
from .llff_dataparser import LLFFDataParserConfig
# from nerfstudio.data.dataparsers.colmap_dataparser import ColmapDataParserConfig
# from nerfstudio.data.dataparsers.nerfstudio_dataparser import NerfstudioDataParserConfig
# from .scannetpp_dataparser import ScannetppDataParserConfig

nerf_method = MethodSpecification(
    config=NerfTrainerConfig(
        method_name="ours",
        steps_pretrain=5000,
        steps_reftrain=6000,
        steps_per_train_image=100000,
        steps_per_eval_image=10000,
        steps_per_save=1000,
        steps_per_video=1000,
        max_num_iterations=10000,
        logging=LoggingConfig(steps_per_log=100),
        mixed_precision=False,
        use_grad_scaler=True,
        pipeline=NerfPipelineConfig(
            datamanager=NerfDataManagerConfig(
                dataparser=SpinDataParserConfig(),
                train_num_rays_per_batch=4096,
                eval_num_rays_per_batch=4096,
                patch_size=256,
                # camera_optimizer=CameraOptimizerConfig(
                #     mode="off",
                #     optimizer=AdamOptimizerConfig(lr=6e-4, eps=1e-8, weight_decay=1e-2),
                #     scheduler=ExponentialDecaySchedulerConfig(lr_final=6e-6, max_steps=200000),
                # ),
            ),
            model=NerfModelConfig(
                eval_num_rays_per_chunk=2 << 15,
                camera_optimizer=CameraOptimizerConfig(mode="SO3xR3"),
                # camera_optimizer=CameraOptimizerConfig(mode="off"),
                average_init_density=0.1,
            ),
        ),
        optimizers={
            "proposal_networks": {
                "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=0.0001, max_steps=200000),
            },
            "fields": {
                "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=0.0001, max_steps=200000),
            },
            # "embedding_appearance": {
            #     "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
            #     "scheduler": ExponentialDecaySchedulerConfig(lr_final=0.0001, max_steps=200000),
            # },
            "camera_opt": {
                        "optimizer": AdamOptimizerConfig(lr=1e-3, eps=1e-15),
                        "scheduler": ExponentialDecaySchedulerConfig(lr_final=1e-4, max_steps=5000),
                    },
            "discriminator": {
                "optimizer": RMSpropOptimizerConfig(lr=1e-3),
                "scheduler": None,
            },
        },
        viewer=ViewerConfig(num_rays_per_chunk=2 << 15),
        vis="tensorboard",
),
    description="Base config for NeRF 1st training stage",
)