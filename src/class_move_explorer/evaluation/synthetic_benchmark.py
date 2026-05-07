"""
Synthetic benchmark generator for ClassMoveExplorer.

This repository does not "train" a model end-to-end; it uses heuristics plus pretrained
embedding/LLM components. To estimate performance in a repeatable way, we generate many
small Java-like projects with well-placed classes, then intentionally corrupt packages
and measure recovery (precision/recall/F1).
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


PACKAGE_TYPES: Tuple[str, ...] = ("controller", "service", "model", "repository", "util", "config", "exception")


@dataclass(frozen=True)
class SyntheticProjectSpec:
    classes_per_type: int = 3
    base_package: str = "com.example"


def _mk_methods(names: List[str]) -> List[Dict]:
    return [
        {"name": n, "parameters": [], "return_type": "void", "annotations": []}
        for n in names
    ]


def _class_dict(
    *,
    class_name: str,
    package: str,
    annotations: List[str],
    imports: Optional[List[str]] = None,
    fields: Optional[List[Dict]] = None,
    methods: Optional[List[Dict]] = None,
    extends: str = "",
    implements: Optional[List[str]] = None,
    class_type: str = "class",
) -> Dict:
    imports = imports or []
    fields = fields or []
    methods = methods or []
    implements = implements or []

    return {
        "class_name": class_name,
        "package": package,
        "file_path": f"/synthetic/{package.replace('.', '/')}/{class_name}.java",
        "relative_path": f"src/main/java/{package.replace('.', '/')}/{class_name}.java",
        "imports": imports,
        "methods_detailed": methods,
        "methods": [m["name"] for m in methods],
        "fields_detailed": fields,
        "fields": [f["name"] for f in fields],
        "annotations": annotations,
        "extends": extends,
        "implements": implements,
        "class_type": class_type,
        "content_preview": f"package {package}; public class {class_name} {{}}",
    }


def generate_synthetic_project(spec: SyntheticProjectSpec, *, seed: int) -> List[Dict]:
    """
    Create a synthetic "project" as a list of class-info dicts matching JavaProjectAnalyzer output.
    """
    rng = random.Random(seed)
    base = spec.base_package

    # Core domain model
    model_classes = []
    for i in range(spec.classes_per_type):
        model_classes.append(
            _class_dict(
                class_name=f"User{i}Entity" if i == 0 else f"Order{i}Entity",
                package=f"{base}.model",
                annotations=["Entity", "Table"],
                imports=["javax.persistence.Entity", "javax.persistence.Table"],
                fields=[
                    {"name": "id", "type": "Long"},
                    {"name": "name", "type": "String"},
                ],
                methods=_mk_methods(["getId", "setId", "getName", "setName"]),
            )
        )

    # Repository layer depends on model
    repo_classes = []
    for i in range(spec.classes_per_type):
        repo_classes.append(
            _class_dict(
                class_name=f"User{i}Repository",
                package=f"{base}.repository",
                annotations=["Repository"],
                imports=[f"{base}.model.User0Entity", "org.springframework.stereotype.Repository"],
                fields=[{"name": "db", "type": "String"}],
                methods=_mk_methods(["findById", "save", "delete"]),
            )
        )

    # Service layer depends on repository/model
    svc_classes = []
    for i in range(spec.classes_per_type):
        svc_classes.append(
            _class_dict(
                class_name=f"User{i}Service",
                package=f"{base}.service",
                annotations=["Service"],
                imports=[f"{base}.repository.User0Repository", f"{base}.model.User0Entity", "org.springframework.stereotype.Service"],
                fields=[{"name": "repo", "type": "User0Repository"}],
                methods=_mk_methods(["getUser", "createUser", "updateUser", "deleteUser"]),
            )
        )

    # Controller depends on service
    ctrl_classes = []
    for i in range(spec.classes_per_type):
        ctrl_classes.append(
            _class_dict(
                class_name=f"User{i}Controller",
                package=f"{base}.controller",
                annotations=["RestController", "RequestMapping"],
                imports=[f"{base}.service.User0Service", "org.springframework.web.bind.annotation.RestController"],
                fields=[{"name": "service", "type": "User0Service"}],
                methods=_mk_methods(["getUser", "postUser", "putUser", "deleteUser"]),
            )
        )

    util_classes = [
        _class_dict(
            class_name="DateUtil",
            package=f"{base}.util",
            annotations=["Component"],
            imports=["org.springframework.stereotype.Component"],
            methods=_mk_methods(["format", "parse"]),
        )
    ]

    config_classes = [
        _class_dict(
            class_name="AppConfig",
            package=f"{base}.config",
            annotations=["Configuration"],
            imports=["org.springframework.context.annotation.Configuration"],
            methods=_mk_methods(["beanA", "beanB"]),
        )
    ]

    exc_classes = [
        _class_dict(
            class_name="NotFoundException",
            package=f"{base}.exception",
            annotations=[],
            imports=[],
            extends="RuntimeException",
            methods=_mk_methods(["NotFoundException"]),
        )
    ]

    # Mix them up a bit to avoid always same ordering
    classes = model_classes + repo_classes + svc_classes + ctrl_classes + util_classes + config_classes + exc_classes
    rng.shuffle(classes)
    return classes


def corrupt_packages(
    classes_data: List[Dict], *, misplace_ratio: float, seed: int, base_package: str = "com.example"
) -> Tuple[List[Dict], Dict[str, str]]:
    """
    Return a corrupted copy of classes_data and a ground-truth mapping:
      ground_truth[class_name] = original_package
    """
    rng = random.Random(seed)
    if not classes_data:
        return [], {}

    n = len(classes_data)
    k = max(1, int(n * misplace_ratio))
    idxs = rng.sample(range(n), k)

    possible_targets = [f"{base_package}.{t}" for t in PACKAGE_TYPES]

    ground_truth: Dict[str, str] = {}
    corrupted: List[Dict] = []

    for i, c in enumerate(classes_data):
        c2 = c.copy()
        if i in idxs:
            original_pkg = c["package"]
            wrong_pkg = rng.choice([p for p in possible_targets if p != original_pkg])
            ground_truth[c["class_name"]] = original_pkg
            c2["package"] = wrong_pkg
        corrupted.append(c2)

    return corrupted, ground_truth

