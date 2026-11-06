import os
import sys
import ast
import yaml
import copy
import types
import inspect
import argparse
import warnings
from typing import Optional, Tuple, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

import models.shape_completion as shape_completion


# ============================================================
# Config helpers
# ============================================================

def load_config(config_path: str) -> argparse.Namespace:
    with open(config_path, "r") as f:
        data = yaml.safe_load(f) or {}
    return argparse.Namespace(**data)


def _load_yaml_dict(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}


def _namespace_to_dict(ns):
    if isinstance(ns, argparse.Namespace):
        return {k: _namespace_to_dict(v) for k, v in vars(ns).items()}
    if isinstance(ns, dict):
        return {k: _namespace_to_dict(v) for k, v in ns.items()}
    if isinstance(ns, list):
        return [_namespace_to_dict(x) for x in ns]
    return ns


# ============================================================
# Checkpoint helpers
# ============================================================

def _strip_module_prefix(state_dict):
    if not any(k.startswith("module.") for k in state_dict.keys()):
        return state_dict
    return {k.replace("module.", "", 1): v for k, v in state_dict.items()}


def _extract_state_dict(ckpt, preferred_keys):
    if isinstance(ckpt, dict):
        for k in preferred_keys:
            if k in ckpt and isinstance(ckpt[k], dict):
                return ckpt[k]
    return ckpt


def _load_weights(
    model: nn.Module,
    ckpt_path: str,
    device,
    preferred_keys=("model_state_dict", "model_state", "state_dict"),
    strict: bool = True,
):
    ckpt = torch.load(ckpt_path, map_location=device)
    state = _extract_state_dict(ckpt, preferred_keys)
    state = _strip_module_prefix(state)
    missing, unexpected = model.load_state_dict(state, strict=strict)
    return missing, unexpected


def _auto_select_checkpoint(output_path: str, suffix: str = ".pt") -> Optional[str]:
    if not output_path or not os.path.isdir(output_path):
        return None

    files = [
        os.path.join(output_path, f)
        for f in os.listdir(output_path)
        if f.endswith(suffix)
    ]

    if not files:
        return None

    files.sort()
    return files[-1]


# ============================================================
# Dynamic class loader for saved model.py / experiment_code.py
# ============================================================

def _load_single_class_from_py(
    py_path: str,
    class_name: str,
    *,
    inject: Optional[Dict[str, Any]] = None,
) -> type:
    py_path = os.path.abspath(py_path)

    with open(py_path, "r", encoding="utf-8") as f:
        src = f.read()

    tree = ast.parse(src, filename=py_path)

    kept_nodes = []
    for node in tree.body:
        if isinstance(
            node,
            (
                ast.Import,
                ast.ImportFrom,
                ast.ClassDef,
                ast.FunctionDef,
                ast.AsyncFunctionDef,
            ),
        ):
            kept_nodes.append(node)

    mod_ast = ast.Module(body=kept_nodes, type_ignores=[])
    code = compile(mod_ast, py_path, "exec")

    module_name = f"_loaded_{class_name}_{abs(hash(py_path))}"
    mod = types.ModuleType(module_name)
    mod.__file__ = py_path

    py_dir = os.path.dirname(py_path)
    old_sys_path = list(sys.path)

    try:
        if py_dir not in sys.path:
            sys.path.insert(0, py_dir)

        if inject:
            mod.__dict__.update(inject)

        exec(code, mod.__dict__, mod.__dict__)

    finally:
        sys.path[:] = old_sys_path

    cls = mod.__dict__.get(class_name, None)
    if cls is None:
        available = [k for k, v in mod.__dict__.items() if isinstance(v, type)]
        raise AttributeError(
            f"Class '{class_name}' not found in {py_path}. "
            f"Available classes: {available}"
        )

    return cls


# ============================================================
# DINO prior loading: exactly one DINO
# ============================================================

def _get_single_dino_spec(args) -> Dict[str, Any]:
    priors = getattr(args, "dino_priors", None)

    if priors is None:
        raise ValueError(
            "Expected args.dino_priors with exactly one entry:\n"
            "dino_priors:\n"
            "  - name: dino\n"
            "    ckpt_path: /path/to/run\n"
            "    ckpt: checkpoint.pt"
        )

    if not isinstance(priors, list):
        raise ValueError(f"args.dino_priors must be a list, got {type(priors)}")

    if len(priors) != 1:
        raise ValueError(
            f"This simplified builder supports exactly one DINO prior, got {len(priors)}"
        )

    spec = priors[0]
    if not isinstance(spec, dict):
        raise ValueError(f"dino_priors[0] must be a dict, got {type(spec)}")

    return spec


def _resolve_dino_files(spec: Dict[str, Any]):
    run_dir = spec.get("ckpt_path", None)
    ckpt_name = spec.get("ckpt", None)

    if run_dir is None:
        raise ValueError("dino_priors[0] is missing 'ckpt_path'")
    if ckpt_name is None:
        raise ValueError("dino_priors[0] is missing 'ckpt'")

    model_py = os.path.join(run_dir, "model.py")
    config_path = os.path.join(run_dir, "config_distillation.yaml")
    ckpt_path = os.path.join(run_dir, ckpt_name)

    missing = []
    if not os.path.isfile(model_py):
        missing.append(model_py)
    if not os.path.isfile(config_path):
        missing.append(config_path)
    if not os.path.isfile(ckpt_path):
        missing.append(ckpt_path)

    if missing:
        raise FileNotFoundError(
            "Missing required DINO prior files:\n" + "\n".join(missing)
        )

    return run_dir, model_py, config_path, ckpt_path


def _instantiate_dino_class(dino_cls: type, dino_config: Dict[str, Any], device):
    """
    Preferred constructor:
        cls(config_dict)

    Extra fallbacks are kept only to avoid crashes if an older saved model.py used
    another signature.
    """
    attempts = [
        lambda: dino_cls(dino_config),
        lambda: dino_cls(dino_config, device),
    ]

    last_error = None
    for fn in attempts:
        try:
            return fn().to(device)
        except TypeError as e:
            last_error = e

    try:
        sig = str(inspect.signature(dino_cls.__init__))
    except Exception:
        sig = "<unknown>"

    raise TypeError(
        f"Could not instantiate DINO class '{dino_cls.__name__}'. "
        f"Constructor signature: {sig}. Last error: {last_error}"
    )


def load_dino_prior_net(args, device, verbose=True) -> nn.Module:
    """
    Loads exactly one DINO prior from:

        args.dino_priors[0]["ckpt_path"]/model.py
        args.dino_priors[0]["ckpt_path"]/config_distillation.yaml
        args.dino_priors[0]["ckpt_path"]/<ckpt>

    config_distillation.yaml must contain:
        model_name: SomeClassName
    """
    spec = _get_single_dino_spec(args)
    run_dir, model_py, dino_cfg_path, dino_ckpt_path = _resolve_dino_files(spec)

    dino_config = _load_yaml_dict(dino_cfg_path)

    model_name = dino_config.get("model_name", None)
    if model_name is None:
        raise ValueError(
            f"{dino_cfg_path} must contain 'model_name', e.g. "
            "model_name: DinoDenseHybridStudentPartTokens"
        )

    inject = {}

    # useful symbols for model.py if it references project classes
    for k, v in shape_completion.__dict__.items():
        if not k.startswith("_"):
            inject.setdefault(k, v)

    inject.update(
        {
            "torch": torch,
            "nn": nn,
            "F": F,
        }
    )

    dino_cls = _load_single_class_from_py(
        model_py,
        model_name,
        inject=inject,
    )

    dino_net = _instantiate_dino_class(dino_cls, dino_config, device)

    missing, unexpected = _load_weights(
        dino_net,
        dino_ckpt_path,
        device=device,
        preferred_keys=("model_state", "model_state_dict", "state_dict"),
        strict=True,
    )

    if verbose:
        print(f"✅ DINO prior loaded")
        print(f"   run   : {run_dir}")
        print(f"   class : {model_name}")
        print(f"   ckpt  : {os.path.basename(dino_ckpt_path)}")

        if missing:
            print("   DINO missing keys:", missing)
        if unexpected:
            print("   DINO unexpected keys:", unexpected)

    freeze_dino = bool(getattr(args, "freeze_dino", False))

    if freeze_dino:
        for p in dino_net.parameters():
            p.requires_grad_(False)
        dino_net.eval()
        if verbose:
            print("   DINO mode: frozen/eval")
    else:
        dino_net.train()
        if verbose:
            print("   DINO mode: trainable/train")

    return dino_net


# ============================================================
# Main model loading
# ============================================================

def _load_main_model_class(args, eval: bool, verbose=True):
    """
    During eval, prefer output_path/experiment_code.py if it exists.
    During training, use models.shape_completion.
    """
    exp_code_path = os.path.join(args.output_path, "experiment_code.py")

    if eval and os.path.isfile(exp_code_path):
        print("Loading main model class from:", exp_code_path)

        inject = {}
        for k, v in shape_completion.__dict__.items():
            if not k.startswith("_"):
                inject.setdefault(k, v)

        inject.update(
            {
                "torch": torch,
                "nn": nn,
                "F": F,
            }
        )

        return _load_single_class_from_py(
            exp_code_path,
            args.model_class,
            inject=inject,
        )

    if verbose:
        print("Loading main model class from models.shape_completion:", args.model_class)
    return getattr(shape_completion, args.model_class)


def _instantiate_main_model(model_cls: type, args, device, dino_net: Optional[nn.Module]):
    """
    Supported main-model constructors:

        Model(args, device, dino_net)
        Model(args, device)
        Model(args)

    If args.use_dino_prior is true, we try the DINO constructor first.
    """
    use_dino = bool(getattr(args, "use_dino_prior", True))

    attempts = []

    if use_dino:
        attempts.append(lambda: model_cls(args, device, dino_net))

    attempts.extend(
        [
            lambda: model_cls(args, device),
            lambda: model_cls(args),
        ]
    )

    last_error = None

    for fn in attempts:
        try:
            return fn().to(device)
        except TypeError as e:
            last_error = e

    try:
        sig = str(inspect.signature(model_cls.__init__))
    except Exception:
        sig = "<unknown>"

    raise TypeError(
        f"Could not instantiate main model '{model_cls.__name__}'. "
        f"Constructor signature: {sig}. Last error: {last_error}"
    )


def build_hybrid_model_from_config(
    args: argparse.Namespace,
    eval: bool = False,
    ckpt_path: Optional[str] = None,
    device: Optional[torch.device] = None,
    verbose=True,
) -> Tuple[nn.Module, argparse.Namespace]:
    """
    Simplified builder.

    What it does:
      1. Load args from config_path.
      2. Load exactly one DINO prior if args.use_dino_prior is true.
      3. Instantiate main model.
      4. If eval=True, load main checkpoint.

    """

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Using device:", device)
    if verbose:
        print("Main model class:", args.model_class)
        print("Output path:", args.output_path)

    use_dino = bool(getattr(args, "use_dino_prior", True))

    dino_net = None
    if use_dino:
        dino_net = load_dino_prior_net(args, device, verbose=verbose)

    model_cls = _load_main_model_class(args, eval=eval, verbose=verbose)
    model = _instantiate_main_model(model_cls, args, device, dino_net)

    if eval:
        model.eval()

        if ckpt_path is None:
            ckpt_path = _auto_select_checkpoint(args.output_path, suffix=".pt")

            if ckpt_path is None:
                print("⚠️ No main checkpoint found in:", args.output_path)
                print("Returning randomly initialized main model.")
                return model, args

            print("Auto-selected main checkpoint:", ckpt_path)
        else:
            print("Using main checkpoint:", ckpt_path)

        missing, unexpected = _load_weights(
            model,
            ckpt_path,
            device=device,
            preferred_keys=("model_state_dict", "model_state", "state_dict"),
            strict=True,
        )

        print("✅ Main model weights loaded.")

        if missing:
            print("   MAIN missing keys:", missing)
        if unexpected:
            print("   MAIN unexpected keys:", unexpected)

    return model, args


# ============================================================
# Fine-tuning helper
# ============================================================

def configure_fine_tuning(model, fine_tuned_modules=None, verbose=True):
    """
    Freeze all parameters except modules listed in fine_tuned_modules.

    Example:
        configure_fine_tuning(
            model,
            ["encoder", "decoder.up32"]
        )

    If fine_tuned_modules is None or empty, everything is trainable.
    """
    raw_model = model.module if hasattr(model, "module") else model

    if fine_tuned_modules is None or len(fine_tuned_modules) == 0:
        for p in raw_model.parameters():
            p.requires_grad_(True)
        if verbose:
            print("Fine-tuning mode: all model parameters are trainable.")
        return

    for p in raw_model.parameters():
        p.requires_grad_(False)

    named_modules = dict(raw_model.named_modules())

    not_found = []

    for module_name in fine_tuned_modules:
        if module_name not in named_modules:
            not_found.append(module_name)
            continue

        for p in named_modules[module_name].parameters(recurse=True):
            p.requires_grad_(True)

    if not_found:
        raise ValueError(
            f"fine_tuned_modules not found: {not_found}\n"
            f"Available top-level modules: {list(raw_model._modules.keys())}"
        )

    if verbose:
        print("Fine-tuning only:")
        for name in fine_tuned_modules:
            print(f"  - {name}")

        total = sum(p.numel() for p in raw_model.parameters())
        trainable = sum(p.numel() for p in raw_model.parameters() if p.requires_grad)
        print(f"Trainable params: {trainable} / {total}")