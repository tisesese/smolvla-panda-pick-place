# Third-party notices

## MuJoCo Menagerie Panda

The Panda model and mesh assets under `simulation/vendor/mujoco_menagerie/` originate from Google DeepMind MuJoCo Menagerie, commit `f054586a8e90465d49ee5be15335c4a0c7f57caf`.

The upstream Apache-2.0 license is retained in the `franka_emika_panda/LICENSE` file. Upstream vendor files are preserved. The project scene composes these assets with a tabletop, red cube, tray, fixed camera, lights, base placement and a grasp reference site. The publication copy changes only scene mesh-directory references to relative paths.

Source: https://github.com/google-deepmind/mujoco_menagerie

## Runtime dependencies and models

LeRobot / SmolVLA, MuJoCo, PyTorch, Transformers and other installed dependencies remain subject to their respective licenses. Full pretrained and fine-tuned weight binaries are not distributed here. Small model and tokenizer configuration assets are retained for compatibility with the original checkpoints.

Sources: https://github.com/huggingface/lerobot and https://huggingface.co/HuggingFaceTB/SmolVLM2-500M-Video-Instruct

This notice does not relicense third-party material or assign an open-source license to the author's course-project code.
