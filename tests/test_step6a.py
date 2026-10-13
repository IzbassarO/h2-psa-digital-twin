"""Pre-registered sampling checks; no simulation or response labels are used."""
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.stats import qmc

from h2psa import design


ROOT = Path(__file__).resolve().parents[1]
DECISIONS = (
    "t_feed", "t_blow", "purge_to_feed", "P_high", "P_low", "f_zeolite",
    "n_equalisations", "feed_flow",
)
DISTURBANCES = ("phi", "y_CH4", "y_CO")
BOUNDS = {
    "t_feed": (40.0, 150.0), "t_blow": (60.0, 200.0),
    "purge_to_feed": (0.02, 0.30), "P_high": (4.0, 8.5),
    "P_low": (0.3, 1.5), "feed_flow": (0.5, 1.5),
    "phi": (0.70, 1.00), "y_CH4": (0.0168, 0.0231),
    "y_CO": (0.005, 0.015),
}


@pytest.fixture(scope="module")
def step6a():
    spec = importlib.util.spec_from_file_location("run_step6a", ROOT / "scripts" / "run_step6a.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def blocks(step6a):
    return step6a.sample_blocks()


def test_registered_variable_order(step6a):
    assert tuple(step6a.DECISIONS) == DECISIONS
    assert tuple(step6a.DISTURBANCES) == DISTURBANCES


def test_registered_block_sizes_and_ids(blocks):
    sizes = {"D_train": 512, "D_cal": 256, "D_test": 256, "D_shift": 128}
    assert set(blocks) == set(sizes)
    for name, size in sizes.items():
        frame = blocks[name]
        assert len(frame) == size
        assert frame["id"].tolist() == [f"{name}_{i:04d}" for i in range(size)]
        assert frame["id"].is_unique
        assert frame["anchor"].eq("lopes").all()
    assert sum(map(len, blocks.values())) == 1152


def test_registered_random_coordinates_and_nested_train_prefixes(blocks):
    expected = {
        "D_train": qmc.Sobol(d=8, scramble=True, seed=20261011).random_base2(9),
        "D_cal": np.random.default_rng(1).random((256, 8)),
        "D_test": np.random.default_rng(2).random((256, 8)),
        "D_shift": np.random.default_rng(3).random((128, 11)),
    }
    for name, unit in expected.items():
        names = DECISIONS + (DISTURBANCES if name == "D_shift" else ())
        actual = blocks[name][[f"u_{variable}" for variable in names]].to_numpy()
        np.testing.assert_array_equal(actual, unit)
        assert ((0.0 <= actual) & (actual <= 1.0)).all()
    # The pilot and training-size prefixes reuse one 512-point draw without extra runs.
    train = blocks["D_train"][[f"u_{variable}" for variable in DECISIONS]].to_numpy()
    for size in (32, 128, 256):
        np.testing.assert_array_equal(train[:size], expected["D_train"][:size])


def test_registered_physical_mapping_preserves_raw_coordinates(blocks):
    for name, frame in blocks.items():
        names = DECISIONS + (DISTURBANCES if name == "D_shift" else ())
        for variable in names:
            unit = frame[f"u_{variable}"].to_numpy()
            if variable == "f_zeolite":
                expected = np.minimum(np.floor(31 * unit), 30) / 30
            elif variable == "n_equalisations":
                expected = np.minimum(1 + np.floor(3 * unit), 3)
            else:
                low, high = BOUNDS[variable]
                expected = low + (high - low) * unit
            np.testing.assert_allclose(frame[variable].to_numpy(), expected, rtol=0, atol=1e-14)
        assert set(frame["n_equalisations"]) <= {1, 2, 3}
        np.testing.assert_allclose(frame["f_zeolite"] * 30, np.round(frame["f_zeolite"] * 30), atol=1e-14)


def test_nominal_disturbances_and_shift_composition(blocks):
    for name in ("D_train", "D_cal", "D_test"):
        frame = blocks[name]
        for variable, value in {"phi": 1.0, "y_CH4": 0.021, "y_CO": 0.012}.items():
            np.testing.assert_array_equal(frame[variable].to_numpy(), np.full(len(frame), value))
    for row in blocks["D_shift"].to_dict("records"):
        feed = design.feed_composition("lopes", row["y_CH4"], row["y_CO"])
        assert sum(feed.values()) == pytest.approx(1.0, abs=1e-15)
        assert feed["CH4"] + feed["CO"] + feed["CO2"] == pytest.approx(0.203, abs=1e-15)
        assert feed["N2"] == 0.007
        assert feed["H2"] == pytest.approx(0.79, abs=1e-15)
        assert min(feed.values()) >= 0


def test_roles_have_distinct_design_draws(blocks):
    seen = set()
    for name in ("D_train", "D_cal", "D_test", "D_shift"):
        points = {tuple(row) for row in blocks[name][[f"u_{variable}" for variable in DECISIONS]].to_numpy()}
        assert not seen.intersection(points)
        seen.update(points)


@pytest.fixture
def synthetic_exports(step6a, blocks, tmp_path, monkeypatch):
    """Synthetic labels exercise formatting only; no generated labels are opened."""
    work = tmp_path / "work"
    (work / "inputs").mkdir(parents=True)
    (work / "D_train").mkdir()
    inputs = blocks["D_train"].iloc[:5].copy()
    inputs.to_csv(work / "inputs" / "D_train_design.csv", index=False)
    results = pd.DataFrame({
        "id": inputs["id"].tolist(),
        "status": ["css", "censored", "cap", "failed", "css"],
        "ppm": [0.0, 2000.0, 50.0, 9999.0, 10000.0],
        "recovery": [70.0, 60.0, 50.0, 99.0, 40.0],
        "productivity": [10.0, 20.0, 30.0, 40.0, 50.0],
        "cycles": [11, 100, 2000, 30, 20],
        "start": ["cold", "D_train_0000", "cold", "checkpoint", "D_train_0002"],
        "wall_s": [100.0, 200.0, 300.0, 40.0, 50.0],
        "solver_failures": [0, 0, 0, 1, 1],
        "attempts": [1, 1, 1, 2, 2],
    })
    # Completion order differs from draw order.
    results.iloc[::-1].to_csv(work / "D_train" / "results.csv", index=False)
    monkeypatch.setitem(step6a.SIZES, "D_train", len(inputs))
    return work, tmp_path / "generated", inputs, results


def test_export_registered_fields_censoring_clipping_and_failed_labels(step6a, synthetic_exports):
    work, generated, inputs, results = synthetic_exports
    path = step6a.export_block("D_train", work_root=work, generated=generated)
    assert path == generated / "D_train.csv"
    frame = pd.read_csv(path, float_precision="round_trip")
    pd.testing.assert_frame_equal(frame[inputs.columns], inputs, check_exact=True)
    assert set(("impurity_ppm", "z", "recovery", "productivity", "status", "cycles",
                "wall_time_s", "warm_start_source")) <= set(frame)
    np.testing.assert_allclose(frame["z"], [-1.0, 3.0, np.log10(50.0), np.nan, 3.0], equal_nan=True)
    assert frame.loc[3, ["impurity_ppm", "z", "recovery", "productivity"]].isna().all()
    assert frame["objectives_at_css"].tolist() == [True, False, False, False, True]
    assert frame["cycles"].tolist() == results["cycles"].tolist()
    assert frame["wall_time_s"].tolist() == results["wall_s"].tolist()
    assert frame["warm_start_source"].tolist() == results["start"].tolist()
    assert not path.with_suffix(".csv.part").exists()


@pytest.mark.parametrize("invalid", ["missing", "duplicate", "foreign"])
def test_export_rejects_incomplete_or_mismatched_points(step6a, synthetic_exports, invalid):
    work, generated, _, results = synthetic_exports
    if invalid == "missing":
        results = results.iloc[:-1]
    elif invalid == "duplicate":
        results.loc[4, "id"] = results.loc[0, "id"]
    else:
        results.loc[4, "id"] = "D_train_unregistered"
    results.to_csv(work / "D_train" / "results.csv", index=False)
    with pytest.raises(ValueError, match="incomplete or non-unique"):
        step6a.export_block("D_train", work_root=work, generated=generated)
    assert not (generated / "D_train.csv").exists()


def test_monitoring_reader_returns_only_status_timing_and_provenance(step6a, synthetic_exports):
    work, _, _, _ = synthetic_exports
    frame = step6a.read_metadata("D_train", work)
    assert not set(("ppm", "recovery", "productivity", "z")).intersection(frame)
    assert set(frame) == {"id", "status", "cycles", "start", "wall_s", "solver_failures", "attempts"}
    assert step6a.metadata_counts(frame) == {
        "n": 5,
        "status_counts": {"css": 2, "censored": 1, "cap": 1, "failed": 1},
        "solver_failure_points": 2,
    }


@pytest.fixture
def fake_publication(step6a, tmp_path, monkeypatch):
    """Model both Git histories; every subprocess is intercepted before execution."""
    import types

    class FakeGit:
        root = tmp_path
        work = tmp_path / "work"
        publishing = work / "csv-publish"
        relative = "data/generated/D_train.csv"
        base = "remote_base_sha"
        local = "local_csv_sha"
        published = "published_csv_sha"
        blob = "csv_blob_sha"

        def __init__(self):
            self.calls = []
            self.remote = self.base
            self.head = self.base
            self.outgoing = []
            self.staged_change = True
            self.has_blob = False
            self.files = {self.local: [self.relative], self.published: [self.relative]}
            self.messages = {sha: "Step 6a: D_train (512 points)" for sha in self.files}

        def existing_worktree(self, head):
            self.publishing.mkdir(parents=True, exist_ok=True)
            step6a.atomic_json(self.work / "csv-publish.json", {"base": self.base, "remote": self.base})
            self.head = head
            self.outgoing = [head] if head != self.base else []

        def run(self, command, **kwargs):
            cwd = kwargs["cwd"]
            self.calls.append((command, cwd))
            args = command[1:]
            returncode = 0
            if args[:3] == ["diff", "--cached", "--quiet"]:
                returncode = int(self.staged_change)
            elif args[0] == "commit":
                assert cwd == self.root
                self.staged_change = False
            elif args[:3] == ["worktree", "add", "--detach"]:
                assert args[-1] == self.base
                Path(args[-2]).mkdir(parents=True)
            elif args[:2] == ["cat-file", "-e"]:
                returncode = 0 if self.has_blob else 1
            elif args[0] == "cherry-pick":
                assert cwd == self.publishing and args == ["cherry-pick", "--ff", "--no-edit", self.local]
                self.head = self.published
                self.outgoing.append(self.published)
                self.has_blob = True
            elif args[:4] == ["-c", "push.followTags=false", "push", "origin"]:
                assert cwd == self.publishing and args[-1] == "HEAD:main"
                self.remote = self.head
            elif args[0] not in ("add", "fetch", "merge-base"):
                raise AssertionError(f"Unexpected mocked git command: {command}")
            return types.SimpleNamespace(returncode=returncode)

        def text(self, command, **kwargs):
            cwd = kwargs["cwd"]
            self.calls.append((command, cwd))
            args = command[1:]
            if args == ["diff", "--cached", "--name-only"]:
                return self.relative + "\n"
            if args[0] == "log":
                return (self.local if cwd == self.root else self.published) + "\n"
            if args[0] == "diff-tree":
                return "\n".join(self.files[args[-1]]) + "\n"
            if args[0] == "show":
                return self.messages[args[-1]] + "\n"
            if args == ["rev-parse", "refs/remotes/origin/main"]:
                return self.remote + "\n"
            if args == ["rev-parse", "HEAD"]:
                return self.head + "\n"
            if args[0] == "rev-parse" and ":" in args[-1]:
                return self.blob + "\n"
            if args == ["status", "--porcelain"]:
                return ""
            if args[0] == "rev-list":
                assert args[-1] == self.base + "..HEAD"
                return "\n".join(self.outgoing)
            raise AssertionError(f"Unexpected mocked git output request: {command}")

    git = FakeGit()
    path = tmp_path / git.relative
    path.parent.mkdir(parents=True)
    path.write_text("id,status\nsynthetic,css\n")
    monkeypatch.setattr(step6a, "ROOT", git.root)
    monkeypatch.setattr(step6a, "WORK", git.work)
    monkeypatch.setattr(step6a.subprocess, "run", git.run)
    monkeypatch.setattr(step6a.subprocess, "check_output", git.text)
    return git, path


def test_data_commit_and_push_publish_only_csv_history(step6a, fake_publication):
    git, path = fake_publication
    assert step6a.commit_and_push("D_train", path) == {"local": git.local, "published": git.published}
    root_commits = [command for command, cwd in git.calls if command[1] == "commit" and cwd == git.root]
    assert root_commits == [["git", "commit", "-m", "Step 6a: D_train (512 points)"]]
    assert (["git", "add", "--", git.relative], git.root) in git.calls
    assert (["git", "worktree", "add", "--detach", str(git.publishing), git.base], git.root) in git.calls
    assert (["git", "cherry-pick", "--ff", "--no-edit", git.local], git.publishing) in git.calls
    pushes = [(command, cwd) for command, cwd in git.calls if "push" in command]
    assert pushes == [(["git", "-c", "push.followTags=false", "push", "origin", "HEAD:main"], git.publishing)]
    assert git.outgoing == [git.published]
    assert git.files[git.published] == [git.relative]
    import json
    assert json.loads((git.work / "csv-publish.json").read_text()) == {
        "base": git.base, "remote": git.published,
    }


def test_publication_restart_with_identical_blob_skips_duplicate_cherry_pick(step6a, fake_publication):
    git, path = fake_publication
    first = step6a.commit_and_push("D_train", path)
    second = step6a.commit_and_push("D_train", path)
    assert first == second == {"local": git.local, "published": git.published}
    assert sum(command[1] == "commit" for command, _ in git.calls) == 1
    assert sum(command[1] == "cherry-pick" for command, _ in git.calls) == 1
    assert sum("push" in command for command, _ in git.calls) == 2
    assert git.outgoing == [git.published]


@pytest.mark.parametrize("payload", [["scripts/run_step6a.py"],
                                     ["data/generated/D_train.csv", "tests/test_step6a.py"]])
def test_publication_rejects_outgoing_source_or_test_payload(step6a, fake_publication, payload):
    git, path = fake_publication
    git.existing_worktree("unexpected_commit")
    git.files["unexpected_commit"] = payload
    git.messages["unexpected_commit"] = "Step 6a: D_train (512 points)"
    with pytest.raises(RuntimeError, match="anything beyond one registered CSV"):
        step6a.commit_and_push("D_train", path)
    assert not any(command[1] == "cherry-pick" or "push" in command for command, _ in git.calls)


def test_publication_pauses_when_remote_changes_unexpectedly(step6a, fake_publication):
    git, path = fake_publication
    git.existing_worktree(git.base)
    git.remote = "unexpected_remote_commit"
    with pytest.raises(RuntimeError, match="Remote main changed unexpectedly"):
        step6a.commit_and_push("D_train", path)
    assert not any(command[1] == "cherry-pick" or "push" in command for command, _ in git.calls)


def test_publication_rejects_wrong_local_commit_message(step6a, fake_publication):
    git, path = fake_publication
    git.messages[git.local] = "Unregistered simulation and analysis changes"
    with pytest.raises(RuntimeError, match="commit message differs"):
        step6a.commit_and_push("D_train", path)
    assert not any(command[1] in ("fetch", "cherry-pick") or "push" in command for command, _ in git.calls)


def test_data_commit_rejects_unrelated_staged_files(step6a, tmp_path, monkeypatch):
    import types

    path = tmp_path / "data" / "generated" / "D_test.csv"
    monkeypatch.setattr(step6a, "ROOT", tmp_path)
    commands = []

    def fake_run(command, **kwargs):
        commands.append(command)
        return types.SimpleNamespace(returncode=1 if command[:3] == ["git", "diff", "--cached"] else 0)

    monkeypatch.setattr(step6a.subprocess, "run", fake_run)
    monkeypatch.setattr(step6a.subprocess, "check_output",
                        lambda *a, **k: "data/generated/D_test.csv\nscripts/unrelated.py\n")
    with pytest.raises(RuntimeError, match="Unrelated staged files"):
        step6a.commit_and_push("D_test", path)
    assert not any(command[1] in ("commit", "push") for command in commands)
