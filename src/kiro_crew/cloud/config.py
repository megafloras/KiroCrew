"""Persisted cloud-launcher config — **profile name only, never credentials**.

Stores the AWS *profile name*, region, and the most-recent instance tag under
``~/.kiro/crew/cloud.json``. AWS credentials are never written here — they are
resolved by the ``aws`` CLI's own provider chain from the profile.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

from kiro_crew.atomic_write import atomic_write
from kiro_crew.config.loader import config_dir

_FILENAME = "cloud.json"
DEFAULT_REGION = "us-east-1"

# Cap at 51 (not 63) to match ec2._TAG_RE / validate_tag: a longer last_tag
# would pass THIS sanitizer but then raise ValidationError on resume (the IAM
# role name kirocrew-ec2-<tag> maxes at 64), defeating the "just treat it as no
# last launch" intent. Keep in lockstep with ec2._TAG_RE.
_TAG_RE = re.compile(r"^[a-zA-Z0-9-]{1,51}$")

#: A digest-pinned image reference, the only form ``cloud/fargate/taskdef.py``
#: accepts. Checked here so a hand-edited ``cloud.json`` naming a movable tag is
#: treated as no Fargate configuration at all, rather than becoming a refusal at
#: the first launch -- which is where the operator has least context for it.
_DIGEST_IMAGE_RE = re.compile(r"^[^\s@]+@sha256:[0-9a-f]{64}$")

#: The two architectures Fargate runs. Spelled here rather than imported from
#: ``cloud/fargate/taskdef.py`` on purpose: this module is loaded to read a config
#: file and must not pull in the ``cloud`` package's AWS surface to do it. The
#: values are held to the imported set by a test, so the two cannot drift.
_CPU_ARCHITECTURES = frozenset({"X86_64", "ARM64"})

#: The environment variable the model credential must reach the container under,
#: respelled from ``cloud/fargate/taskdef.py`` for the same reason as the
#: architectures and pinned to it by the same test. The engine refuses a task
#: definition that delivers no secret under this name, so a block naming no such
#: secret is a block whose lane would refuse every launch.
_MODEL_CREDENTIAL_ENV = "KIRO_API_KEY"


@dataclass(frozen=True)
class FargateConfig:
    """Where an operator writes the Fargate lane's placement, image and secrets.

    **Identifiers only, never a secret value.** A crew secret is named by its
    canonical name and its ARN; the value is fetched by the task's execution role
    from Secrets Manager before the container starts, so nothing here is a
    credential and this file's no-secrets contract holds unchanged.

    The engine takes these four fields as a ``FargateLaunchSpec`` and refuses to
    guess any of them -- "an unnamed subnet or security group is the same class of
    error as deleting a task on a guess". This is the place they are written down.
    """

    cluster: str = ""
    subnets: tuple[str, ...] = ()
    security_groups: tuple[str, ...] = ()
    image: str = ""
    #: ``(canonical name, ARN)`` pairs. A pair, not a bare ARN: an ARN alone
    #: cannot say where the secret's NAME ends, because the service appends a
    #: six-character suffix and nothing marks the boundary.
    secrets: tuple[tuple[str, str], ...] = ()
    cpu_architecture: str = "X86_64"
    #: False is the safe direction, and the flag is not the boundary -- a task in
    #: a public subnet with no NAT gateway cannot pull its image without one.
    assign_public_ip: bool = False

    def is_complete(self) -> bool:
        """True when every field the engine requires is present and well-formed.

        INCOMPLETE MEANS ABSENT, and that is the whole design of this method. A
        half-written block must leave the lane unregistered rather than registered
        and refusing: a lane that exists and rejects every launch spends the
        operator's attention at launch time on a mistake that was visible when
        they saved the file.

        The secrets must include one named for the model credential, because the
        engine refuses a task definition that delivers none: a block with every
        placement field and no credential secret is the offered-and-refusing state
        in its most likely form. Only the NAME is judged here. Whether the ARN is
        that name plus one service suffix is the engine's verification, and it is
        not repeated in this module.
        """
        return bool(
            self.cluster
            and self.subnets
            and self.security_groups
            and _DIGEST_IMAGE_RE.match(self.image or "")
            and self.cpu_architecture in _CPU_ARCHITECTURES
            and _names_model_credential(self.secrets)
        )

    @classmethod
    def from_mapping(cls, data: object) -> Optional["FargateConfig"]:
        """Read one block, or ``None`` for anything that is not usable.

        Every rejection returns ``None`` rather than a partially-populated object,
        so a caller cannot hold a config that looks present and is not. A secret
        entry that is not a two-string pair drops the WHOLE block, not just that
        entry: silently launching with one fewer secret than the operator wrote is
        how a task starts and then fails on a missing variable.

        ``assign_public_ip`` is read the same way: absent means ``False``, and a
        present value that is not a JSON boolean drops the block. The field decides
        network exposure, and coercing it would read the string ``"false"`` as
        true, which is the one direction this field must never be guessed in.
        """
        if not isinstance(data, dict):
            return None
        secrets: list[tuple[str, str]] = []
        raw_secrets = data.get("secrets", [])
        if not isinstance(raw_secrets, list):
            return None
        for entry in raw_secrets:
            if not (isinstance(entry, (list, tuple)) and len(entry) == 2):
                return None
            name, arn = entry
            if not (isinstance(name, str) and isinstance(arn, str) and name and arn):
                return None
            secrets.append((name, arn))
        assign_public_ip = data.get("assign_public_ip", False)
        if not isinstance(assign_public_ip, bool):
            return None
        candidate = cls(
            cluster=str(data.get("cluster", "")),
            subnets=_string_tuple(data.get("subnets")),
            security_groups=_string_tuple(data.get("security_groups")),
            image=str(data.get("image", "")),
            secrets=tuple(secrets),
            cpu_architecture=str(data.get("cpu_architecture", "X86_64")),
            assign_public_ip=assign_public_ip,
        )
        return candidate if candidate.is_complete() else None


def _names_model_credential(secrets: tuple[tuple[str, str], ...]) -> bool:
    """True when some secret's name ends in the model credential's variable.

    The engine derives a secret's destination variable from the final segment of
    its canonical name, so a block with no name ending in that segment cannot
    deliver the credential however its ARNs read. This catches the nameable case,
    where no credential secret is written down at all. It does NOT verify the
    name's prefix or its pairing with the ARN: that check belongs to the engine,
    and a second copy of it here would be a second place for it to drift.
    """
    return any(name.rsplit("/", 1)[-1] == _MODEL_CREDENTIAL_ENV for name, _arn in secrets)


def _string_tuple(value: object) -> tuple[str, ...]:
    """A list of non-empty strings, or empty. A non-list is empty, not an error.

    ``is_complete`` is the single place emptiness is judged, so this returns the
    empty tuple for every unusable shape instead of raising -- one refusal point
    rather than two that can disagree.
    """
    if not isinstance(value, list):
        return ()
    return tuple(item for item in value if isinstance(item, str) and item)


@dataclass
class CloudConfig:
    """The launcher's saved state (no secrets)."""

    profile: str = ""
    region: str = DEFAULT_REGION
    last_tag: str = ""
    #: The ``fargate`` block EXACTLY as read from the file, or ``None`` when the
    #: file has none. It is kept raw, not judged, so that ``save()`` writes back
    #: whatever the operator wrote: this object is loaded and re-saved to record
    #: ``last_tag`` after an ordinary EC2 launch, and a field that held only a
    #: judged value would erase a block the operator is half-way through writing.
    #: Whether the block is usable is :meth:`fargate_config`'s question.
    fargate: Any = None

    def fargate_config(self) -> Optional[FargateConfig]:
        """The Fargate block as a typed config, or ``None`` when it is not complete.

        ``None`` is what keeps the lane UNREGISTERED, so an operator who has not
        filled the block in is never offered a lane that would refuse them. This
        judges the raw block on every call rather than once at load, so the file
        round-trips untouched and the seam still sees complete-or-absent.
        """
        return FargateConfig.from_mapping(self.fargate)

    @classmethod
    def load(cls, path: Optional[Path] = None) -> "CloudConfig":
        p = path or (config_dir() / _FILENAME)
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return cls()
        # A hand-edited cloud.json may parse to valid JSON that is NOT an object
        # (e.g. `"hello"`, `[1,2]`, `42`, `null`); the .get() calls below would
        # then raise AttributeError and escape load() (handle_cloud only catches
        # AWS/validation errors), giving a raw traceback on every cloud command.
        # Honor the docstring's tolerate-a-corrupt-file promise: fall back to
        # defaults on any non-object shape.
        if not isinstance(data, dict):
            return cls()
        # Sanitize last_tag at the boundary: a hand-edited/corrupt cloud.json
        # must not carry a malformed tag into the resume path (downstream
        # validate_tag would raise; an empty tag just means "no last launch").
        last_tag = str(data.get("last_tag", ""))
        if last_tag and not _TAG_RE.match(last_tag):
            last_tag = ""
        return cls(
            profile=str(data.get("profile", "")),
            region=str(data.get("region", "") or DEFAULT_REGION),
            last_tag=last_tag,
            # Deliberately NOT sanitized here, unlike last_tag: an incomplete
            # block must survive a save, so it is carried as written and judged
            # by fargate_config() at the point of use.
            fargate=data.get("fargate"),
        )

    def save(self, path: Optional[Path] = None) -> None:
        p = path or (config_dir() / _FILENAME)
        p.parent.mkdir(parents=True, exist_ok=True)
        # Unique temp name per writer: concurrent cloud invocations must not
        # race on a shared .tmp path (see atomic_write's rationale).
        atomic_write(p, json.dumps(asdict(self), indent=2))
