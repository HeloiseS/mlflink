""" CAUTION: This code was written by AI with minimal human oversight. 
Needs testing and review. 
"""
import ast
import sys
from pathlib import Path
from typing import Set, Dict, List
import importlib.metadata as importlib_md
import yaml   # pyyaml

STD_LIB_MODULES = None  # filled lazily


def _collect_top_level_imports(pkg_dir: Path) -> Set[str]:
    """
    Walks .py files under pkg_dir and returns top-level module names imported,
    e.g. 'numpy', 'sklearn', 'finkvra' (we will ignore imports that map to this package itself).
    """
    imports: Set[str] = set()
    for py in pkg_dir.rglob("*.py"):
        try:
            src = py.read_text(encoding="utf-8")
        except Exception:
            continue
        try:
            tree = ast.parse(src, filename=str(py))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for n in node.names:
                    top = n.name.split(".")[0]
                    imports.add(top)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    top = node.module.split(".")[0]
                    imports.add(top)
    return imports


def _get_stdlib_names() -> Set[str]:
    global STD_LIB_MODULES
    if STD_LIB_MODULES is None:
        # crude stdlib blacklist: use sys.stdlib_module_names if available (py3.10+)
        try:
            STD_LIB_MODULES = set(sys.stdlib_module_names)  # py3.10+
        except AttributeError:
            # fallback: small safe set to filter common stdlib names
            STD_LIB_MODULES = {
                "os", "sys", "json", "math", "typing", "pathlib", "logging",
                "itertools", "functools", "collections", "datetime", "subprocess",
            }
    return STD_LIB_MODULES


def _map_modules_to_distributions(mod_names: Set[str]) -> Dict[str, str]:
    """
    Use importlib.metadata.packages_distributions() to map top-level module names
    to distribution package names. Returns mapping mod_name -> dist_name.
    """
    pkg_map = importlib_md.packages_distributions()
    mapping: Dict[str, str] = {}
    for mod in mod_names:
        dists = pkg_map.get(mod)
        if dists:
            # prefer first listed distribution
            mapping[mod] = dists[0]
    return mapping


def _versions_for_distributions(dist_names: List[str]) -> Dict[str, str]:
    versions = {}
    for dist in dist_names:
        try:
            versions[dist] = importlib_md.version(dist)
        except importlib_md.PackageNotFoundError:
            # maybe different naming; skip
            continue
    return versions

def generate_requirements_txt_from_imports(
    pkg_dir: str | Path,
    output_path: str | Path,
    include_self: bool = False,
    extra_pip: List[str] | None = None,
    add_python_version_comment: bool = True,
) -> str:
    """
    Generate a requirements.txt inferred from imports found under pkg_dir.

    - pkg_dir: directory of your package (e.g. ~/software/finkvra)
    - output_path: where to save the requirements.txt
    - include_self: include the package itself as an editable install (-e file://...)
    - extra_pip: optional list of extra pip strings to include (e.g. ["mlflow==2.4.0"])
    - add_python_version_comment: if True, write a comment with the local python version at top
    """
    pkg_dir = Path(pkg_dir).expanduser().resolve()
    output_path = Path(output_path).expanduser().resolve()
    if not pkg_dir.exists():
        raise FileNotFoundError(pkg_dir)

    # collect top-level imports and filter stdlib + local package name
    top_imports = _collect_top_level_imports(pkg_dir)
    stdlib = _get_stdlib_names()
    top_imports = {m for m in top_imports if m not in stdlib}

    package_name = pkg_dir.name
    if not include_self and package_name in top_imports:
        top_imports.remove(package_name)

    # map to distributions and get versions (same logic as conda generator)
    mod_to_dist = _map_modules_to_distributions(top_imports)
    dist_names = sorted(set(mod_to_dist.values()))
    dist_versions = _versions_for_distributions(dist_names)

    # build pip lines
    pip_lines: List[str] = []
    for dist in dist_names:
        ver = dist_versions.get(dist)
        if ver:
            pip_lines.append(f"{dist}=={ver}")
        else:
            pip_lines.append(dist)

    if extra_pip:
        pip_lines.extend(extra_pip)

    if include_self:
        # requirements.txt uses editable install syntax for local package
        pip_lines.append(f"-e file://{pkg_dir}")

    # ensure output dir exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # write requirements.txt
    with open(output_path, "w", encoding="utf-8") as fh:
        if add_python_version_comment:
            pyver = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
            fh.write(f"# Generated for Python {pyver}\n")
        for line in pip_lines:
            fh.write(line.rstrip() + "\n")

    return str(output_path)

def generate_conda_yaml_from_imports(
    pkg_dir: str | Path,
    output_path: str | Path,
    include_self: bool = False,
    extra_pip: List[str] | None = None,
):
    """
    Generates a conda-style YAML file that pins python and pip packages inferred from imports.

    - pkg_dir: directory of your package (e.g. ~/software/finkvra)
    - output_path: where to save the conda yaml
    - include_self: include the package itself as a pip install if you want (usually False)
    - extra_pip: optional list of extra pip strings to include (e.g. ["mlflow==2.4.0"])
    """
    pkg_dir = Path(pkg_dir).expanduser().resolve()
    output_path = Path(output_path).expanduser().resolve()
    if not pkg_dir.exists():
        raise FileNotFoundError(pkg_dir)

    top_imports = _collect_top_level_imports(pkg_dir)
    stdlib = _get_stdlib_names()
    # remove stdlib and local package name
    top_imports = {m for m in top_imports if m not in stdlib}
    # remove the package itself (so we don't try to require finkvra unless requested)
    package_name = pkg_dir.name
    if not include_self and package_name in top_imports:
        top_imports.remove(package_name)

    # map top-level module -> distribution name
    mod_to_dist = _map_modules_to_distributions(top_imports)

    # If mapping misses something, we keep it out (can't resolve)
    dist_names = sorted(set(mod_to_dist.values()))
    dist_versions = _versions_for_distributions(dist_names)

    pip_lines: List[str] = []
    for dist in dist_names:
        ver = dist_versions.get(dist)
        if ver:
            pip_lines.append(f"{dist}=={ver}")
        else:
            pip_lines.append(dist)  # best effort unpinned

    if extra_pip:
        pip_lines.extend(extra_pip)

    # Optionally include the package itself as a pip editable/ wheel instruction
    if include_self:
        pip_lines.append(f"-e file://{pkg_dir}")  # editable install

    # build conda yaml structure
    pyver = f"{sys.version_info.major}.{sys.version_info.minor}"
    env = {
        "name": f"finkvra-generated-env",
        "channels": ["defaults", "conda-forge"],
        "dependencies": [
            f"python={pyver}",
            {"pip": pip_lines},
        ],
    }

    # write YAML
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fh:
        yaml.safe_dump(env, fh, sort_keys=False)

    return str(output_path)
