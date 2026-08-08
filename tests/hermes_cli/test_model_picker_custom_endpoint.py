from types import SimpleNamespace
import unittest
from unittest.mock import patch

import cli as cli_mod
from hermes_cli.model_switch import ModelSwitchResult


def _shell(**overrides):
    shell = cli_mod.HermesCLI.__new__(cli_mod.HermesCLI)
    defaults = {
        "_app": None,
        "_model_picker_state": None,
        "provider": "openrouter",
        "model": "old-model",
        "base_url": "https://openrouter.ai/api/v1",
        "api_key": "old-runtime-key",
        "api_mode": "chat_completions",
        "_modal_input_snapshot": None,
        "_invalidate": lambda **_kwargs: None,
        "_restore_modal_input_snapshot": lambda: None,
        "_confirm_and_apply_model_switch_result": lambda *_args: None,
    }
    defaults.update(overrides)
    for name, value in defaults.items():
        setattr(shell, name, value)
    return shell


class CustomEndpointModelPickerTest(unittest.TestCase):
    def test_provider_picker_keeps_native_choices_and_opens_custom_editor(self):
        shell = _shell(
            _model_picker_state={
                "stage": "provider",
                "providers": [{"slug": "openrouter", "name": "OpenRouter"}],
                "selected": 1,
            }
        )
        started = []
        with patch.object(shell, "_start_custom_model_picker", lambda: started.append(True)):
            shell._handle_model_picker_selection(persist_global=True)

        self.assertEqual(started, [True])
        self.assertIsNotNone(shell._model_picker_state)

    def test_custom_editor_is_first_choice_when_no_provider_is_authenticated(self):
        shell = _shell(
            _model_picker_state={
                "stage": "provider",
                "providers": [],
                "selected": 0,
            }
        )
        started = []
        with patch.object(shell, "_start_custom_model_picker", lambda: started.append(True)):
            shell._handle_model_picker_selection(persist_global=True)

        self.assertEqual(started, [True])

    def test_existing_provider_model_selection_still_dispatches_switch(self):
        result = ModelSwitchResult(
            success=True,
            new_model="openai/gpt-5.5-pro",
            target_provider="nous",
        )
        captured = {}

        class _Thread:
            def __init__(self, *, target, args, daemon):
                captured.update(target=target, args=args, daemon=daemon)

            def start(self):
                captured["started"] = True

        shell = _shell(
            _app=object(),
            _model_picker_state={
                "stage": "model",
                "provider_data": {"slug": "nous"},
                "model_list": ["openai/gpt-5.5-pro"],
                "selected": 0,
                "user_provs": None,
                "custom_provs": None,
            },
            provider="nous",
            model="openai/gpt-5.5",
            base_url="",
            api_key="",
        )
        with (
            patch("hermes_cli.model_switch.switch_model", lambda **_kwargs: result),
            patch.object(cli_mod.threading, "Thread", _Thread),
        ):
            shell._handle_model_picker_selection(persist_global=True)

        self.assertIsNone(shell._model_picker_state)
        self.assertTrue(captured["started"])
        self.assertTrue(captured["daemon"])
        self.assertEqual(captured["args"], (result, True))

    def test_existing_picker_viewport_keeps_long_catalog_selection_visible(self):
        offset = 0
        for cursor in list(range(36)) + list(range(35, -1, -1)):
            offset, visible = cli_mod.HermesCLI._compute_model_picker_viewport(
                cursor, offset, n=36, term_rows=30
            )
            self.assertIn(cursor, range(offset, offset + visible))

    def test_custom_endpoint_wizard_masks_key_and_validates_required_fields(self):
        shell = _shell(
            _model_picker_state={
                "stage": "custom_url",
                "selected": 0,
                "custom": {
                    "base_url": "",
                    "model": "",
                    "api_mode": "",
                    "name": "",
                    "existing_api_key": "",
                    "new_api_key": "",
                    "has_new_api_key": False,
                },
            }
        )

        shell._handle_custom_model_picker_selection("not-a-url", True)
        self.assertEqual(shell._model_picker_state["stage"], "custom_url")
        self.assertIn("http://", shell._model_picker_state["error"])

        shell._handle_custom_model_picker_selection("https://models.example/v1/", True)
        self.assertEqual(shell._model_picker_state["stage"], "custom_key")
        self.assertTrue(shell._model_picker_secret_input_active())

        shell._handle_custom_model_picker_selection("secret-value", True)
        self.assertEqual(shell._model_picker_state["stage"], "custom_model")
        self.assertEqual(
            shell._model_picker_state["custom"]["new_api_key"], "secret-value"
        )

        shell._handle_custom_model_picker_selection("", True)
        self.assertEqual(shell._model_picker_state["stage"], "custom_model")
        self.assertEqual(shell._model_picker_state["error"], "Model name is required.")

    def test_save_custom_endpoint_keeps_api_key_out_of_yaml(self):
        saved = {
            "config": {
                "model": {"default": "old-model", "provider": "openrouter"},
                "custom_providers": [],
            },
            "env": {},
        }

        def _load_config():
            return {
                "model": dict(saved["config"]["model"]),
                "custom_providers": [
                    dict(entry) for entry in saved["config"]["custom_providers"]
                ],
            }

        captured = {}

        def _switch_model(**kwargs):
            captured.update(kwargs)
            return SimpleNamespace(success=True)

        with (
            patch("hermes_cli.config.load_config", _load_config),
            patch(
                "hermes_cli.config.save_config",
                lambda config: saved.__setitem__("config", config),
            ),
            patch(
                "hermes_cli.config.save_env_value",
                lambda key, value: saved["env"].__setitem__(key, value),
            ),
            patch("hermes_cli.auth.deactivate_provider", lambda: None),
            patch(
                "hermes_cli.inventory.load_picker_context",
                lambda: SimpleNamespace(
                    user_providers={},
                    custom_providers=saved["config"]["custom_providers"],
                ),
            ),
            patch("hermes_cli.model_switch.switch_model", _switch_model),
        ):
            shell = _shell(api_key="")
            result = shell._save_custom_model_endpoint(
                {
                    "base_url": "https://models.example/v1",
                    "original_url": "",
                    "model": "gpt-test",
                    "api_mode": "codex_responses",
                    "name": "Lab Endpoint",
                    "key_env": "",
                    "existing_api_key": "",
                    "new_api_key": "api-secret-value",
                    "has_new_api_key": True,
                },
                persist_global=True,
            )

        self.assertTrue(result.success)
        key_env = "HERMES_CUSTOM_LAB_ENDPOINT_API_KEY"
        self.assertEqual(saved["env"], {key_env: "api-secret-value"})
        self.assertNotIn("api-secret-value", repr(saved["config"]))
        provider = saved["config"]["custom_providers"][0]
        self.assertEqual(provider["key_env"], key_env)
        self.assertEqual(provider["api_mode"], "codex_responses")
        self.assertEqual(provider["models"], {"gpt-test": {}})
        self.assertEqual(saved["config"]["model"]["provider"], "custom:lab-endpoint")
        self.assertNotIn("api_key", saved["config"]["model"])
        self.assertEqual(captured["explicit_provider"], "custom:lab-endpoint")

    def test_remote_plaintext_http_requires_confirmation_stage(self):
        shell = _shell(
            _model_picker_state={
                "stage": "custom_url",
                "selected": 0,
                "custom": {"base_url": "", "model": "m", "name": "Local"},
            }
        )

        shell._handle_custom_model_picker_selection(
            "http://192.168.1.20:8000/v1", True
        )

        self.assertEqual(shell._model_picker_state["stage"], "custom_http_warning")
        shell._model_picker_state["selected"] = 1
        shell._handle_custom_model_picker_selection("", True)
        self.assertEqual(shell._model_picker_state["stage"], "custom_url")


if __name__ == "__main__":
    unittest.main()
