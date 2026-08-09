"""Telegram private-DM rescue phrase and temporary profile routing tests."""

from types import SimpleNamespace

import pytest

from gateway.config import Platform
from gateway.platforms.base import (
    MessageEvent,
    coerce_plaintext_gateway_command,
)
from gateway.profile_routing import ProfileRoute
from gateway.rescue import verify_rescue_phrase
from gateway.run import GatewayRunner
from gateway.session import SessionSource


_TEST_PHRASE = "test-rescue-phrase"


@pytest.fixture(scope="module")
def rescue_hash():
    from plugins.dashboard_auth.basic import hash_password

    return hash_password(_TEST_PHRASE)


@pytest.fixture
def configured_rescue(monkeypatch, rescue_hash):
    monkeypatch.setenv("GATEWAY_RESCUE_SECRET_HASH", rescue_hash)


def _source(*, platform=Platform.TELEGRAM, chat_type="dm") -> SessionSource:
    return SessionSource(
        platform=platform,
        chat_id="10001",
        chat_type=chat_type,
        user_id="20002",
        user_name="owner",
    )


def _event(text: str, **source_kwargs) -> MessageEvent:
    return MessageEvent(text=text, source=_source(**source_kwargs))


def _runner(*, multiplex=True) -> GatewayRunner:
    runner = object.__new__(GatewayRunner)
    runner.config = SimpleNamespace(
        multiplex_profiles=multiplex,
        profile_routes=[
            ProfileRoute(
                name="telegram-butler",
                platform="telegram",
                profile="tg-butler",
            )
        ],
    )
    return runner


def test_rescue_hash_verification_is_exact_and_case_sensitive(configured_rescue):
    assert verify_rescue_phrase(_TEST_PHRASE)
    assert not verify_rescue_phrase(_TEST_PHRASE.upper())
    assert not verify_rescue_phrase(f"{_TEST_PHRASE}x")


def test_rescue_hash_reads_active_profile_secret_scope(monkeypatch, rescue_hash):
    from agent.secret_scope import reset_secret_scope, set_secret_scope

    monkeypatch.delenv("GATEWAY_RESCUE_SECRET_HASH", raising=False)
    token = set_secret_scope({"GATEWAY_RESCUE_SECRET_HASH": rescue_hash})
    try:
        assert verify_rescue_phrase(_TEST_PHRASE)
    finally:
        reset_secret_scope(token)


def test_rescue_hash_rejects_unbounded_scrypt_parameters(monkeypatch, rescue_hash):
    parts = rescue_hash.split("$")
    parts[1] = str(2**20)
    monkeypatch.setenv("GATEWAY_RESCUE_SECRET_HASH", "$".join(parts))

    assert not verify_rescue_phrase(_TEST_PHRASE)


def test_authorized_plaintext_phrase_becomes_trusted_rescue_command(configured_rescue):
    event = _event(f"  {_TEST_PHRASE}  ")

    coerce_plaintext_gateway_command(event, allow_rescue=True)

    assert event.text == "/rescue"
    assert event._gateway_rescue_verified is True


@pytest.mark.parametrize(
    ("event", "allow_rescue"),
    [
        (_event(_TEST_PHRASE), False),
        (_event(_TEST_PHRASE, chat_type="group"), True),
        (_event(_TEST_PHRASE, platform=Platform.DISCORD), True),
    ],
)
def test_plaintext_phrase_is_not_coerced_outside_authorized_telegram_dm(
    configured_rescue, event, allow_rescue
):
    coerce_plaintext_gateway_command(event, allow_rescue=allow_rescue)

    assert event.text == _TEST_PHRASE
    assert not hasattr(event, "_gateway_rescue_verified")


@pytest.mark.asyncio
async def test_rescue_temporarily_overrides_static_route_and_off_restores_it(
    configured_rescue, monkeypatch
):
    import gateway.slash_commands as slash_commands

    now = [100.0]
    monkeypatch.setattr(slash_commands.time, "monotonic", lambda: now[0])
    runner = _runner()
    source = _source()
    event = MessageEvent(text="/rescue", source=source)
    event._gateway_rescue_verified = True

    reply = await runner._handle_rescue_command(event)

    assert "15 minutes" in reply
    assert "approvals remain enabled" in reply
    assert runner._profile_name_for_source(source) == "default"

    status = await runner._handle_rescue_command(
        MessageEvent(text="/rescue status", source=source)
    )
    assert "active" in status
    assert "900 seconds" in status

    off = await runner._handle_rescue_command(
        MessageEvent(text="/rescue off", source=source)
    )
    assert "disabled" in off
    assert runner._profile_name_for_source(source) == "tg-butler"


@pytest.mark.asyncio
async def test_rescue_expires_back_to_static_route(configured_rescue, monkeypatch):
    import gateway.slash_commands as slash_commands

    now = [100.0]
    monkeypatch.setattr(slash_commands.time, "monotonic", lambda: now[0])
    runner = _runner()
    source = _source()

    reply = await runner._handle_rescue_command(
        MessageEvent(text=f"/rescue {_TEST_PHRASE}", source=source)
    )
    assert "enabled" in reply
    assert runner._profile_name_for_source(source) == "default"

    now[0] = 1001.0
    assert runner._profile_name_for_source(source) == "tg-butler"


@pytest.mark.asyncio
async def test_rescue_rejects_wrong_phrase_group_and_non_multiplex(configured_rescue):
    runner = _runner()
    assert "Invalid" in await runner._handle_rescue_command(
        MessageEvent(text="/rescue wrong", source=_source())
    )
    assert "private chats" in await runner._handle_rescue_command(
        MessageEvent(text=f"/rescue {_TEST_PHRASE}", source=_source(chat_type="group"))
    )
    assert "multiplex_profiles" in await _runner(multiplex=False)._handle_rescue_command(
        MessageEvent(text=f"/rescue {_TEST_PHRASE}", source=_source())
    )


def test_rescue_command_is_registered_for_busy_dispatch():
    from hermes_cli.commands import resolve_command

    command = resolve_command("rescue")
    assert command is not None
    assert command.gateway_only is True
    assert command.busy_policy == "dispatch"
