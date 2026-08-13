"""Real LoRA-stack inference backend for the GPU host.

This module implements the ``backend_fn`` seam of ``LocalLoraExecutor``:
given a frozen request payload and input image bytes it loads
Illustrious-XL + style LoRA + character LoRA (+ optional ControlNet OpenPose)
and returns PNG bytes deterministically for a fixed seed.

Design rules:

- Heavy imports (torch/diffusers/controlnet_aux/PIL) are lazy; importing this
  module and constructing the backend never requires the GPU stack.
- If the dependencies or model files are missing, the backend raises
  ``EnvironmentCapabilityError`` with install guidance instead of failing
  deep inside a library call.
- The diffusers API surface is validated on the GPU host by the remote
  bootstrap self-test before any real run (kohya/diffusers versions can
  drift; the frozen sampling variables live in the request payload, not in
  this module).
"""

from __future__ import annotations

from collections.abc import Callable
from io import BytesIO
from pathlib import Path

from anime_remix.errors import EnvironmentCapabilityError, InputValidationError
from anime_remix.services.execution.local_lora_executor import (
    _parse_size_pixels,
)


class LocalLoraInferenceBackend:
    """Diffusers-based backend: base model + dual LoRA + optional ControlNet.

    ``operation`` dispatch matches the Composer contract:

    - ``character_synthesis``: text-to-image (identity/style from the LoRA
      weights), with optional OpenPose ControlNet when a pose reference is
      present.
    - ``first_frame_fusion``: image-to-image anchored on the first condition
      slot (canvas), with the same optional ControlNet.
    - ``local_inpaint``: SDXL inpaint on ``composite_image`` +
      ``inpaint_mask``.
    """

    def __init__(
        self,
        *,
        base_model_path: str | Path,
        controlnet_model_path: str | Path | None = None,
        device: str | None = None,
        cpu_offload: bool = True,
        controlnet_conditioning_scale: float = 0.8,
        img2img_strength: float = 0.65,
    ) -> None:
        base_model_path = Path(base_model_path)
        if not base_model_path.exists():
            raise EnvironmentCapabilityError(
                f"Illustrious-XL base model not found: {base_model_path}"
            )
        self.base_model_path = base_model_path
        self.controlnet_model_path = (
            Path(controlnet_model_path)
            if controlnet_model_path is not None
            else None
        )
        self.device = device
        self.cpu_offload = cpu_offload
        if not 0 < float(controlnet_conditioning_scale) <= 2:
            raise InputValidationError(
                "controlnet_conditioning_scale must be in (0, 2]"
            )
        self.controlnet_conditioning_scale = float(
            controlnet_conditioning_scale
        )
        if not 0 < float(img2img_strength) <= 1:
            raise InputValidationError(
                "img2img_strength must be in (0, 1]"
            )
        self.img2img_strength = float(img2img_strength)
        self._pipeline_cache: dict[str, object] = {}

    def __call__(
        self,
        request_payload: dict,
        operation: str,
        inputs: dict[str, bytes],
    ) -> bytes:
        parameters = request_payload["parameters"]
        controlnet_params = parameters["controlnet"]
        seed = parameters["seed"]
        steps = parameters["steps"]
        cfg = parameters["cfg_scale"]
        negative_prompt = parameters["negative_prompt"]
        width, height = _parse_size_pixels(parameters["size"])
        prompt = self._prompt_with_triggers(
            request_payload["prompt"], parameters
        )

        torch, _diffusers = self._require_deps()
        torch_device = self._torch_device(torch)
        torch.manual_seed(seed)
        generator = torch.Generator(device=torch_device).manual_seed(seed)

        _pipeline_key, pipeline = self._get_pipeline(
            operation=operation,
            controlnet_params=controlnet_params,
            inputs=inputs,
            parameters=parameters,
        )
        kwargs = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "num_inference_steps": steps,
            "guidance_scale": cfg,
            "generator": generator,
            "width": width,
            "height": height,
            "num_images_per_prompt": 1,
        }
        if operation == "local_inpaint":
            kwargs["image"] = self._pil(inputs["composite_image"], "composite")
            kwargs["mask_image"] = self._pil(inputs["inpaint_mask"], "mask")
        else:
            control = self._control_image(
                controlnet_params, inputs, torch, pipeline
            )
            if control is not None:
                kwargs["control_image"] = control
                kwargs["controlnet_conditioning_scale"] = (
                    self.controlnet_conditioning_scale
                )
            init = self._init_image(operation, request_payload, inputs)
            if init is not None:
                kwargs["image"] = init
                kwargs["strength"] = self.img2img_strength
        result = pipeline(**kwargs)
        return self._png_bytes(result.images[0])

    def _prompt_with_triggers(self, prompt: str, parameters: dict) -> str:
        triggers = []
        for label in ("style_lora", "character_lora"):
            config = parameters.get(label) or {}
            trigger = config.get("trigger")
            if trigger:
                triggers.append(str(trigger))
        if not triggers:
            return prompt
        return ", ".join(triggers) + ", " + prompt

    def _require_deps(self) -> tuple[object, object]:
        try:
            import diffusers
            import torch
        except ImportError as exc:
            raise EnvironmentCapabilityError(
                "local LoRA inference requires torch + diffusers; install "
                f"them on the GPU host (import error: {exc})"
            ) from exc
        return torch, diffusers

    def _torch_device(self, torch: object) -> str:
        if self.device is not None:
            return self.device
        cuda = torch.cuda.is_available()  # type: ignore[attr-defined]
        return "cuda" if cuda else "cpu"

    def _dtype(self, torch: object, torch_device: str) -> object:
        return (
            torch.float16  # type: ignore[attr-defined]
            if torch_device == "cuda"
            else torch.float32  # type: ignore[attr-defined]
        )

    def _get_pipeline(
        self,
        *,
        operation: str,
        controlnet_params: dict,
        inputs: dict[str, bytes],
        parameters: dict,
    ) -> tuple[str, object]:
        use_control = (
            operation != "local_inpaint"
            and self._control_image_bytes(controlnet_params, inputs) is not None
        )
        key = f"{operation}:control={use_control}"
        if key in self._pipeline_cache:
            return key, self._pipeline_cache[key]
        torch, diffusers = self._require_deps()
        torch_device = self._torch_device(torch)
        dtype = self._dtype(torch, torch_device)
        pipeline = self._build_pipeline(
            diffusers,
            torch_dtype=dtype,
            operation=operation,
            use_control=use_control,
        )
        self._apply_loras(pipeline, parameters)
        if torch_device == "cuda" and self.cpu_offload:
            pipeline.enable_model_cpu_offload()  # type: ignore[attr-defined]
        else:
            pipeline.to(torch_device)  # type: ignore[attr-defined]
        self._pipeline_cache[key] = pipeline
        return key, pipeline

    def _build_pipeline(
        self,
        diffusers: object,
        *,
        torch_dtype: object,
        operation: str,
        use_control: bool,
    ) -> object:
        if operation == "local_inpaint":
            return diffusers.StableDiffusionXLInpaintPipeline.from_pretrained(  # type: ignore[attr-defined]
                self.base_model_path,
                torch_dtype=torch_dtype,
                use_safetensors=True,
            )
        if use_control:
            if self.controlnet_model_path is None:
                raise EnvironmentCapabilityError(
                    "ControlNet pose conditioning requires "
                    "controlnet_model_path (download an SDXL OpenPose "
                    "ControlNet on the GPU host)"
                )
            if not self.controlnet_model_path.exists():
                raise EnvironmentCapabilityError(
                    "ControlNet model not found: "
                    f"{self.controlnet_model_path}"
                )
            controlnet = diffusers.ControlNetModel.from_pretrained(  # type: ignore[attr-defined]
                self.controlnet_model_path,
                torch_dtype=torch_dtype,
                use_safetensors=True,
            )
            return diffusers.StableDiffusionXLControlNetPipeline.from_pretrained(  # type: ignore[attr-defined]
                self.base_model_path,
                controlnet=controlnet,
                torch_dtype=torch_dtype,
                use_safetensors=True,
            )
        if operation == "first_frame_fusion":
            return diffusers.StableDiffusionXLImg2ImgPipeline.from_pretrained(  # type: ignore[attr-defined]
                self.base_model_path,
                torch_dtype=torch_dtype,
                use_safetensors=True,
            )
        return diffusers.StableDiffusionXLPipeline.from_pretrained(  # type: ignore[attr-defined]
            self.base_model_path,
            torch_dtype=torch_dtype,
            use_safetensors=True,
        )

    def _apply_loras(self, pipeline: object, parameters: dict) -> None:
        for label in ("style_lora", "character_lora"):
            config = parameters.get(label) or {}
            path = config.get("path")
            if not path:
                continue
            pipeline.load_lora_weights(  # type: ignore[attr-defined]
                str(path)
            )

    def _control_image_bytes(
        self, controlnet_params: dict, inputs: dict[str, bytes]
    ) -> bytes | None:
        ref = controlnet_params.get("pose_condition_ref")
        if isinstance(ref, str) and ref in inputs:
            return inputs[ref]
        pose_source = controlnet_params.get("pose_source")
        if isinstance(pose_source, str) and Path(pose_source).exists():
            return Path(pose_source).read_bytes()
        return None

    def _control_image(
        self,
        controlnet_params: dict,
        inputs: dict[str, bytes],
        torch: object,
        pipeline: object,
    ):
        raw = self._control_image_bytes(controlnet_params, inputs)
        if raw is None:
            return None
        try:
            from controlnet_aux import OpenposeDetector
        except ImportError as exc:
            raise EnvironmentCapabilityError(
                "OpenPose extraction requires controlnet_aux; install it on "
                f"the GPU host (import error: {exc})"
            ) from exc
        detector = OpenposeDetector.from_pretrained(
            "lllyasviel/Annotators"
        )
        pose = detector(self._pil(raw, "pose reference"))
        image = pose.convert("RGB")
        torch_device = self._torch_device(torch)
        dtype = self._dtype(torch, torch_device)
        return pipeline.prepare_control_image(  # type: ignore[attr-defined]
            image=image,
            width=int(pipeline.unet.config.sample_size) * 8,  # type: ignore[attr-defined]
            height=int(pipeline.unet.config.sample_size) * 8,  # type: ignore[attr-defined]
            batch_size=1,
            num_images_per_prompt=1,
            device=torch_device,
            dtype=dtype,
            do_classifier_free_guidance=True,
            guess_mode=False,
        )

    def _init_image(
        self,
        operation: str,
        request_payload: dict,
        inputs: dict[str, bytes],
    ):
        if operation != "first_frame_fusion":
            return None
        conditions = request_payload.get("conditions") or []
        if not conditions:
            return None
        first_ref = conditions[0].get("condition_ref")
        if not isinstance(first_ref, str) or first_ref not in inputs:
            return None
        return self._pil(inputs[first_ref], "init canvas")

    @staticmethod
    def _pil(data: bytes, label: str):
        try:
            from PIL import Image
        except ImportError as exc:
            raise EnvironmentCapabilityError(
                f"Pillow is required to decode {label} (import error: {exc})"
            ) from exc
        try:
            return Image.open(BytesIO(data)).convert("RGB")
        except Exception as exc:
            raise InputValidationError(
                f"cannot decode {label} as an image"
            ) from exc

    @staticmethod
    def _png_bytes(image: object) -> bytes:

        output = BytesIO()
        image.convert("RGB").save(output, format="PNG")  # type: ignore[attr-defined]
        return output.getvalue()


LocalLoraInferenceBackendFn = Callable[[dict, str, dict[str, bytes]], bytes]
