"""Audit raw-to-LeRobot actions, identical views, labels, and eight training inputs."""

import copy
import io
import json
from pathlib import Path

import cv2
import h5py
import numpy as np
from PIL import Image
import pyarrow.parquet as pq


def main(*, data_only: bool = False):
    root = Path("/iris/u/tiangao/lerobot_datasets/skybhh19")
    folders = {space: root / f"droid_cup_hanging_09132026_{space}" for space in ("jointpos", "jointvel")}
    manifests = {
        space: json.loads((folder / "meta/droid_source_manifest.json").read_text()) for space, folder in folders.items()
    }
    assert manifests["jointpos"]["episodes"] == manifests["jointvel"]["episodes"]
    for space, folder in folders.items():
        manifest = manifests[space]
        info = json.loads((folder / "meta/info.json").read_text())
        assert info["total_episodes"] == len(manifest["episodes"]) == 103
        assert info["total_frames"] == sum(e["frames"] for e in manifest["episodes"])
        assert manifest["ignored_camera_ids"] == ["23404442"]
        tasks = [json.loads(line) for line in (folder / "meta/tasks.jsonl").read_text().splitlines()]
        assert tasks == [{"task_index": 0, "task": "Hang the cup on the hook"}]
        action_key = "joint_position" if space == "jointpos" else "joint_velocity"
        for episode in manifest["episodes"]:
            path = folder / f"data/chunk-000/episode_{episode['episode_index']:06d}.parquet"
            rows = pq.read_table(
                path, columns=["actions", "joint_position", "gripper_position", "task_index"]
            ).to_pydict()
            indices = episode["raw_frame_indices"]
            assert len(rows["actions"]) == episode["frames"]
            with h5py.File(episode["trajectory_path"]) as raw:
                targets = np.concatenate(
                    [
                        raw[f"action/{action_key}"][:][indices],
                        raw["action/gripper_position"][:][indices].reshape(-1, 1),
                    ],
                    axis=1,
                )
                np.testing.assert_allclose(rows["actions"], targets, atol=1e-6)
                np.testing.assert_allclose(
                    rows["joint_position"], raw["observation/robot_state/joint_positions"][:][indices], atol=1e-6
                )
            assert set(rows["task_index"]) == {0}
        print(f"Verified every {space} action against raw HDF5 ({info['total_frames']} frames)", flush=True)
    for index in (0, 51, 102):
        columns = ["exterior_image_1_left", "wrist_image_left"]
        images = [
            pq.read_table(folder / f"data/chunk-000/episode_{index:06d}.parquet", columns=columns).to_pylist()
            for folder in folders.values()
        ]
        assert images[0] == images[1]
        episode = manifests["jointpos"]["episodes"][index]
        for camera, column in (("31078156", columns[0]), ("17471093", columns[1])):
            video = cv2.VideoCapture(str(Path(episode["raw_episode_dir"]) / "recordings/MP4" / f"{camera}.mp4"))
            video.set(cv2.CAP_PROP_POS_FRAMES, episode["raw_frame_indices"][0])
            ok, bgr = video.read()
            video.release()
            assert ok
            expected_image = np.asarray(Image.fromarray(bgr[..., ::-1]).resize((320, 180), Image.Resampling.BICUBIC))
            saved_image = np.asarray(Image.open(io.BytesIO(images[0][0][column]["bytes"])))
            np.testing.assert_array_equal(saved_image, expected_image)
    filter_dir = Path(__file__).parent / "lerobot_filtering_keys"
    selections = {}
    for subset in ("randompct60", "observabilitypct60", "observabilitypct30"):
        selections[subset] = json.loads(
            (filter_dir / f"cup_hanging_09132026_{subset}_episode_indices.json").read_text()
        )
        assert len(selections[subset]) == (31 if subset.endswith("30") else 62)
        assert len(set(selections[subset])) == len(selections[subset])
    assert set(selections["observabilitypct30"]) <= set(selections["observabilitypct60"])
    print("Raw actions, camera identities, paired images, provenance, and filters passed", flush=True)
    if data_only:
        return
    from openpi.training import config as configs
    from openpi.training import data_loader

    for checkpoint in ("pi05_base", "pi05_droid"):
        for subset in ("full", "randompct60", "observabilitypct60", "observabilitypct30"):
            name = f"{checkpoint}_cup_hanging_09132026_{subset}_low_mem_finetune"
            cfg = configs.get_config(name)
            data = cfg.data.create(Path("assets"), cfg.model)
            assert cfg.weight_loader.params_path == f"gs://openpi-assets/checkpoints/{checkpoint}/params"
            assert data.norm_stats is not None
            ds = data_loader.create_torch_dataset(data, cfg.model.action_horizon, cfg.model)
            sample = ds[0]
            assert sample["prompt"] == "Hang the cup on the hook"
            expected = np.asarray(sample["actions"]).copy()
            if checkpoint == "pi05_base":
                expected[..., :7] -= np.asarray(sample["joint_position"])
            transformed = copy.deepcopy(sample)
            for transform in (*data.repack_transforms.inputs, *data.data_transforms.inputs):
                transformed = transform(transformed)
            np.testing.assert_allclose(transformed["actions"], expected, atol=1e-6)
            final = data_loader.transform_dataset(ds, data)[0]
            assert final["actions"].shape == (16, 32)
            assert np.isfinite(final["actions"]).all()
            if subset != "full":
                assert len(ds) == sum(manifests["jointpos"]["episodes"][i]["frames"] for i in selections[subset])
            print(f"Validated {name}: {len(ds)} frames, normalized actions {final['actions'].shape}", flush=True)


if __name__ == "__main__":
    main()
