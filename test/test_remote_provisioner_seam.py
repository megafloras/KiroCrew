"""The CPP ``remote_provisioners`` seam: descriptor, Default adapter, composition.

The HTTP half (listing, ``provider_id`` on a launch, engine routing) lives in
``test_cloud_handlers.py::TestProvisionerSeam``; the job-side half in
``test_cloud_launch_job.py::TestProvisionerOnTheJob``. This file pins the
contract objects themselves and that the default context composes the seam.
"""

from __future__ import annotations

import dataclasses

import pytest

from kiro_crew.cloud.launch_engine import RealLaunchEngine
from kiro_crew.platform.defaults import (
    BUILTIN_REMOTE_PROVISIONER,
    FARGATE_PROVISIONER_ID,
    FARGATE_REMOTE_PROVISIONER,
    DefaultRemoteProvisionerProvider,
)
from kiro_crew.platform.interfaces import (
    BUILTIN_PROVISIONER_ID,
    RemoteProvisioner,
    RemoteProvisionerProvider,
)


class TestFargateLane:
    """The second lane the core ships, offered only when ``cloud.json`` names it.

    Written against the PROVIDER rather than the HTTP surface, because the property
    is which engines exist -- the listing and routing halves are pinned in
    ``test_cloud_handlers.py::TestProvisionerSeam``.
    """

    @staticmethod
    def _write_config(home, block) -> None:
        import json

        path = home / "cloud.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"profile": "p", "region": "us-east-1"}
        if block is not None:
            payload["fargate"] = block
        path.write_text(json.dumps(payload), encoding="utf-8")

    @staticmethod
    def _complete_block() -> dict:
        # The credential secret is a conforming pair: the canonical name
        # kirocrew/crew/<crew>/<ENV> and the ARN that is that name plus one
        # six-character service suffix. The account id is fictional. Without a
        # secret named for the model credential the block is INCOMPLETE, because
        # the engine would refuse every launch made through the lane.
        return {
            "cluster": "kirocrew-crew-prod",
            "subnets": ["subnet-a"],
            "security_groups": ["sg-1"],
            "image": "public.ecr.aws/example/kirocrew-crew-base@sha256:" + "b" * 64,
            "secrets": [
                [
                    "kirocrew/crew/demo/KIRO_API_KEY",
                    "arn:aws:secretsmanager:us-east-1:123456789012:"
                    "secret:kirocrew/crew/demo/KIRO_API_KEY-AbCdEf",
                ]
            ],
            "cpu_architecture": "X86_64",
        }

    def test_unconfigured_offers_only_ec2_and_refuses_the_fargate_engine(
        self, monkeypatch, tmp_path
    ):
        monkeypatch.setenv("KIROCREW_HOME", str(tmp_path))
        provider = DefaultRemoteProvisionerProvider()
        assert [p.id for p in provider.provisioners()] == [BUILTIN_PROVISIONER_ID]
        with pytest.raises(KeyError):
            provider.engine_for(FARGATE_PROVISIONER_ID)

    def test_configured_offers_the_lane_and_builds_the_engine_from_the_block(
        self, monkeypatch, tmp_path
    ):
        monkeypatch.setenv("KIROCREW_HOME", str(tmp_path))
        from kiro_crew.config.loader import config_dir

        block = self._complete_block()
        self._write_config(config_dir(), block)

        provider = DefaultRemoteProvisionerProvider()
        assert [p.id for p in provider.provisioners()] == [
            BUILTIN_PROVISIONER_ID,
            FARGATE_PROVISIONER_ID,
        ]
        engine = provider.engine_for(FARGATE_PROVISIONER_ID)
        # The spec's fields come from the block, not from a default -- the engine
        # refuses to guess any of them, so a wrong value here is a wrong launch.
        spec = engine._require_spec()
        assert spec.placement.cluster == block["cluster"]
        assert spec.placement.subnets == tuple(block["subnets"])
        assert spec.placement.security_groups == tuple(block["security_groups"])
        assert spec.image == block["image"]
        assert spec.cpu_architecture == block["cpu_architecture"]

    def test_the_ec2_lane_is_unchanged_either_way(self, monkeypatch, tmp_path):
        monkeypatch.setenv("KIROCREW_HOME", str(tmp_path))
        from kiro_crew.config.loader import config_dir

        provider = DefaultRemoteProvisionerProvider()
        assert isinstance(provider.engine_for(BUILTIN_PROVISIONER_ID), RealLaunchEngine)
        self._write_config(config_dir(), self._complete_block())
        assert isinstance(provider.engine_for(BUILTIN_PROVISIONER_ID), RealLaunchEngine)

    def test_an_incomplete_block_leaves_the_lane_unoffered(self, monkeypatch, tmp_path):
        """Offered-and-refusing is the state this avoids."""
        monkeypatch.setenv("KIROCREW_HOME", str(tmp_path))
        from kiro_crew.config.loader import config_dir

        self._write_config(config_dir(), {"cluster": "only-a-cluster"})
        provider = DefaultRemoteProvisionerProvider()
        assert [p.id for p in provider.provisioners()] == [BUILTIN_PROVISIONER_ID]
        with pytest.raises(KeyError):
            provider.engine_for(FARGATE_PROVISIONER_ID)

    def test_a_block_missing_only_the_credential_secret_leaves_the_lane_unoffered(
        self, monkeypatch, tmp_path
    ):
        """The most likely way to reach offered-and-refusing, so pin it at the lane.

        Every placement field is present and only the model-credential secret is
        gone. The engine would build a task definition and refuse it, so this block
        must leave the lane unregistered rather than registered and rejecting. The
        config-level boundary asserts the same thing; without this case a change
        that drops the requirement passes the seam suite untouched.
        """
        monkeypatch.setenv("KIROCREW_HOME", str(tmp_path))
        from kiro_crew.config.loader import config_dir

        self._write_config(config_dir(), {**self._complete_block(), "secrets": []})
        provider = DefaultRemoteProvisionerProvider()
        assert [p.id for p in provider.provisioners()] == [BUILTIN_PROVISIONER_ID]
        with pytest.raises(KeyError):
            provider.engine_for(FARGATE_PROVISIONER_ID)

    def test_a_config_read_that_raises_does_not_take_the_selector_down(self, monkeypatch):
        """One lane's misconfiguration must not hide the other lane.

        ``provisioners()`` builds the Set-up list, so raising here would render the
        tab empty -- turning a Fargate problem into no lanes at all.
        """
        import kiro_crew.cloud.config as cloud_config

        def boom(*_args, **_kwargs):
            raise OSError("unreadable")

        monkeypatch.setattr(cloud_config.CloudConfig, "load", classmethod(boom))
        provider = DefaultRemoteProvisionerProvider()
        assert [p.id for p in provider.provisioners()] == [BUILTIN_PROVISIONER_ID]

    def test_the_block_is_read_per_call_not_cached(self, monkeypatch, tmp_path):
        """An operator who edits the file gets the answer from the next request."""
        monkeypatch.setenv("KIROCREW_HOME", str(tmp_path))
        from kiro_crew.config.loader import config_dir

        provider = DefaultRemoteProvisionerProvider()
        assert [p.id for p in provider.provisioners()] == [BUILTIN_PROVISIONER_ID]
        self._write_config(config_dir(), self._complete_block())
        assert FARGATE_PROVISIONER_ID in [p.id for p in provider.provisioners()]
        (config_dir() / "cloud.json").unlink()
        assert [p.id for p in provider.provisioners()] == [BUILTIN_PROVISIONER_ID]

    def test_the_descriptor_names_its_own_kind_so_the_dashboard_skips_it(self):
        """``kind`` equals the id, deliberately.

        The dashboard draws ``aws_ec2`` with the core's own form and SKIPS a kind it
        has no renderer for. Naming a kind of its own keeps this lane absent from
        the selector until a renderer exists, rather than handing the EC2 form a
        lane that takes no instance type.
        """
        assert FARGATE_REMOTE_PROVISIONER.id == FARGATE_PROVISIONER_ID
        assert FARGATE_REMOTE_PROVISIONER.kind == FARGATE_PROVISIONER_ID
        assert FARGATE_REMOTE_PROVISIONER.kind != BUILTIN_PROVISIONER_ID

    def test_importing_the_platform_does_not_pull_the_cloud_config_module(self):
        """The deferral is load-bearing twice over, so it is pinned.

        ``kiro_crew.cloud`` reaches ``kiro_crew.sandbox``, which imports
        ``kiro_crew.platform.current_context`` -- a module-level import here raises
        ``ImportError: cannot import name 'current_context' from partially
        initialized module``, because this module loads during ``platform`` init.
        It is also 105 ms and 122 modules against a 126 ms init, for a lane most
        deployments have not configured.
        """
        import os
        import subprocess
        import sys
        from pathlib import Path

        from kiro_crew.subprocess_utf8 import UTF8_TEXT

        probe = (
            "import sys, kiro_crew.platform.defaults as d;"
            "print('cloud.config' if 'kiro_crew.cloud.config' in sys.modules else 'deferred')"
        )
        # Run from the repository root with src/ on the path, so the child imports
        # this checkout rather than whatever is installed.
        root = Path(__file__).resolve().parents[1]
        env = dict(os.environ, PYTHONPATH=str(root / "src"))
        result = subprocess.run(
            [sys.executable, "-c", probe],
            capture_output=True,
            cwd=root,
            env=env,
            timeout=120,
            **UTF8_TEXT,
        )
        assert result.returncode == 0, result.stderr
        assert result.stdout.strip() == "deferred", result.stdout


class TestDescriptor:
    def test_builtin_descriptor_is_the_ec2_lane(self):
        assert BUILTIN_PROVISIONER_ID == "aws_ec2"
        assert BUILTIN_REMOTE_PROVISIONER.id == BUILTIN_PROVISIONER_ID
        # id and kind coincide for the built-in: the core's own form draws it.
        assert BUILTIN_REMOTE_PROVISIONER.kind == BUILTIN_PROVISIONER_ID
        assert BUILTIN_REMOTE_PROVISIONER.posix_only is True
        assert BUILTIN_REMOTE_PROVISIONER.step_labels == ()

    def test_descriptor_is_frozen_and_hashable(self):
        p = RemoteProvisioner(id="x", kind="k", label="X")
        with pytest.raises(dataclasses.FrozenInstanceError):
            p.id = "y"  # type: ignore[misc]
        assert hash(p)  # step_labels is a tuple, not a dict, so this holds

    def test_step_labels_are_key_value_pairs(self):
        p = RemoteProvisioner(
            id="devspace",
            kind="amazon_devspace",
            label="Amazon DevSpace",
            posix_only=False,
            step_labels=(("provision", "Create the DevSpace"),),
        )
        assert dict(p.step_labels) == {"provision": "Create the DevSpace"}


class TestDefaultProvider:
    def test_lists_exactly_the_builtin(self):
        rows = DefaultRemoteProvisionerProvider().provisioners()
        assert rows == [BUILTIN_REMOTE_PROVISIONER]

    def test_engine_for_the_builtin_is_the_ec2_engine(self):
        eng = DefaultRemoteProvisionerProvider().engine_for(BUILTIN_PROVISIONER_ID)
        assert isinstance(eng, RealLaunchEngine)

    def test_engine_for_anything_else_is_a_key_error(self):
        with pytest.raises(KeyError):
            DefaultRemoteProvisionerProvider().engine_for("devspace")

    def test_satisfies_the_protocol_shape(self):
        p: RemoteProvisionerProvider = DefaultRemoteProvisionerProvider()
        assert callable(p.provisioners) and callable(p.engine_for)


class TestComposition:
    def test_default_context_composes_the_seam(self, monkeypatch, tmp_path):
        monkeypatch.setenv("KIROCREW_HOME", str(tmp_path))
        from kiro_crew.config.loader import KiroCrewConfig
        from kiro_crew.platform.bootstrap import build_default_context

        ctx = build_default_context(KiroCrewConfig.load(), profile="standalone")
        assert isinstance(ctx.remote_provisioners, DefaultRemoteProvisionerProvider)

    def test_companion_can_replace_it_with_dataclasses_replace(self, monkeypatch, tmp_path):
        """The companion's composition root is ``dataclasses.replace`` on the base
        context; a new required field must not break that path."""
        monkeypatch.setenv("KIROCREW_HOME", str(tmp_path))
        from kiro_crew.config.loader import KiroCrewConfig
        from kiro_crew.platform.bootstrap import build_default_context

        class Two:
            def provisioners(self):
                return [BUILTIN_REMOTE_PROVISIONER, RemoteProvisioner("d", "amazon_devspace", "D")]

            def engine_for(self, provisioner_id):
                raise KeyError(provisioner_id)

        base = build_default_context(KiroCrewConfig.load(), profile="standalone")
        ctx = dataclasses.replace(base, remote_provisioners=Two())
        assert [p.id for p in ctx.remote_provisioners.provisioners()] == ["aws_ec2", "d"]
