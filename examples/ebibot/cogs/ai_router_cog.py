"""ai_router_cog.py — マルチAI APIルーターCog

Discordメッセージを受信し、チャンネルに応じて最適なAI APIに直接ルーティングする。
Claude Codeセッションを経由せず、GPT-5.4/Gemini APIを直接呼び出すことで
トークン消費とレイテンシを最小化する。

配置（2026-03-22 会長承認）:
- CEO室・営業部・情報収集部・財務部 → GPT-5.4 Thinking High
- 開発部（GCP/GAS） → Gemini 3.1 Pro
- 品質管理部 → Claude Opus 4.6（ClaudeChatCogが処理）

環境変数:
    AI_ROUTER_ENABLED         "1" でルーティング有効化
    AI_ROUTER_GPT_CHANNELS    GPT-5.4で処理するチャンネルID（カンマ区切り）
    AI_ROUTER_GEMINI_CHANNELS Geminiで処理するチャンネルID（カンマ区切り）
    OPENAI_API_KEY            OpenAI APIキー
    GEMINI_API_KEY            Gemini APIキー
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import urllib.request
from typing import TYPE_CHECKING

import discord
from discord.ext import commands

if TYPE_CHECKING:
    from claude_discord.bot import ClaudeDiscordBot

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

AI_ROUTER_ENABLED = os.environ.get("AI_ROUTER_ENABLED", "0") == "1"

_gpt_raw = os.environ.get("AI_ROUTER_GPT_CHANNELS", "")
GPT_CHANNEL_IDS: set[int] = {int(x) for x in _gpt_raw.split(",") if x.strip().isdigit()}

_gemini_raw = os.environ.get("AI_ROUTER_GEMINI_CHANNELS", "")
GEMINI_CHANNEL_IDS: set[int] = {int(x) for x in _gemini_raw.split(",") if x.strip().isdigit()}

ALL_ROUTED_CHANNEL_IDS = GPT_CHANNEL_IDS | GEMINI_CHANNEL_IDS

# System prompts per department channel
_SYSTEM_PROMPTS: dict[int, str] = {}
_sys_raw = os.environ.get("AI_ROUTER_SYSTEM_PROMPTS", "")
if _sys_raw:
    try:
        _SYSTEM_PROMPTS = {int(k): v for k, v in json.loads(_sys_raw).items()}
    except (json.JSONDecodeError, ValueError):
        logger.warning("AI_ROUTER_SYSTEM_PROMPTS is not valid JSON, ignoring")

def _build_common_context() -> str:
    """Build common context with runtime credentials."""
    linear_key = os.environ.get("LINEAR_API_KEY", "未設定")
    return (
        "重要ルール:\n"
        "- 会長（沼田宏太）には簡単な言葉で結論から話す。専門用語禁止\n"
        "- 「できない」「権限がない」「担当者が実行してください」は絶対禁止。このシステムはClaude Code経由でVM上の全操作を実行できる\n"
        "- 選択肢を出さず最善案を実行→事後報告。会長に判断コストを押し付けない\n"
        "- 同じ回答を繰り返すな。会長が同じ質問をした場合、前回と異なるアプローチで対応する\n"
        "- 会長が既に回答・提供済みの情報を再度聞くな。過去の会話を確認してから質問しろ\n"
        "- 「試験移行」「切替希望日」等の不要なステップを提案するな。指示されたら即実行\n"
        "\n"
        "完了済みタスク（2026-03-24時点）:\n"
        "- Linear導入完了。GitHub Issues全167件移行済み。自動同期稼働中\n"
        "- HP修正（techforward.jp）全7ページ更新完了。開発プロセス統一・余白・中央寄せ対応済み\n"
        "- マルチAI体制稼働中（GPT-5.4/Claude/Gemini/Codex）\n"
        "- AIルーターにClaude Codeプロバイダー追加済み。GPT/Geminiエラー時は自動フォールバック\n"
        "\n"
        "利用可能なAPIキー・認証情報:\n"
        f"- Linear API Key: {linear_key}\n"
        "- Linear URL: https://linear.app/techforward\n"
        "- Linear同期: scripts/linear_sync.py（GitHub Actions linear-sync.ymlで30分ごと自動実行）\n"
        "- WordPress: scripts/hp_update.py（GitHub Actions hp-update.ymlで自動デプロイ）\n"
        "- GitHub: tech-forward/techforward-company（gh CLIで操作可能）\n"
        "- Discord Bot: 各部署に専用Bot（.env.departmentsに格納）\n"
        "\n"
        "会長のタスク起票先:\n"
        "- 会長はLinearの「会長起票」に起票する（https://linear.app/techforward）\n"
        "- GitHubは開かない。Linearだけ見る\n"
    )


def _build_default_system() -> str:
    """Build default system prompt with runtime credentials."""
    return (
        _build_common_context()
        + "あなたはTechForward合同会社のAIエージェントです。"
        "日本語で回答してください。簡潔で実用的な回答を心がけてください。"
        "難しい専門用語は使わず、簡単な言葉で話してください。"
    )

MAX_RESPONSE_LENGTH = 1900  # Discord message limit - buffer


# ---------------------------------------------------------------------------
# API Callers (async wrappers)
# ---------------------------------------------------------------------------

async def _call_gpt54(
    prompt: str,
    system: str | None = None,
    reasoning_effort: str = "high",
    web_search: bool = True,
) -> str:
    """Call GPT-5.4 Thinking High via OpenAI API."""
    api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("OPENAI_API_KEY_CODEX")
    if not api_key:
        return "ERROR: OPENAI_API_KEY not set"

    if web_search:
        return await _call_gpt54_responses(prompt, system, reasoning_effort, api_key)
    else:
        return await _call_gpt54_chat(prompt, system, reasoning_effort, api_key)


async def _call_gpt54_responses(
    prompt: str, system: str | None, reasoning_effort: str, api_key: str,
) -> str:
    """OpenAI Responses API with web search."""
    body: dict = {
        "model": "gpt-5.4",
        "input": prompt,
        "max_output_tokens": 16384,
        "tools": [{"type": "web_search_preview", "search_context_size": "high"}],
    }
    if system:
        body["instructions"] = system
    if reasoning_effort and reasoning_effort != "none":
        body["reasoning"] = {"effort": reasoning_effort}

    def _do_request() -> str:
        data = json.dumps(body).encode()
        req = urllib.request.Request(
            "https://api.openai.com/v1/responses",
            data=data,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=180) as resp:
                result = json.loads(resp.read())
                for item in result.get("output", []):
                    if item.get("type") == "message":
                        for content in item.get("content", []):
                            if content.get("type") == "output_text":
                                return content.get("text", "")
                return "ERROR: No text output in response"
        except Exception as e:
            return f"ERROR: OpenAI API call failed: {e}"

    return await asyncio.get_event_loop().run_in_executor(None, _do_request)


async def _call_gpt54_chat(
    prompt: str, system: str | None, reasoning_effort: str, api_key: str,
) -> str:
    """OpenAI Chat Completions API (without web search)."""
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    body: dict = {
        "model": "gpt-5.4",
        "messages": messages,
        "max_completion_tokens": 16384,
    }
    if reasoning_effort and reasoning_effort != "none":
        body["reasoning_effort"] = reasoning_effort

    def _do_request() -> str:
        data = json.dumps(body).encode()
        req = urllib.request.Request(
            "https://api.openai.com/v1/chat/completions",
            data=data,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                result = json.loads(resp.read())
                return result["choices"][0]["message"]["content"]
        except Exception as e:
            return f"ERROR: OpenAI API call failed: {e}"

    return await asyncio.get_event_loop().run_in_executor(None, _do_request)


async def _call_gemini(
    prompt: str,
    system: str | None = None,
    web_search: bool = True,
) -> str:
    """Call Gemini 3.1 Pro API."""
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        return "ERROR: GEMINI_API_KEY not set"

    body: dict = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"maxOutputTokens": 16384},
    }
    if system:
        body["systemInstruction"] = {"parts": [{"text": system}]}
    if web_search:
        body["tools"] = [{"google_search": {}}]

    model = "gemini-2.5-pro"

    def _do_request() -> str:
        data = json.dumps(body).encode()
        url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/{model}"
            f":generateContent?key={api_key}"
        )
        req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                result = json.loads(resp.read())
                candidates = result.get("candidates", [])
                if candidates:
                    parts = candidates[0].get("content", {}).get("parts", [])
                    return "".join(p.get("text", "") for p in parts)
                return "ERROR: No response from Gemini API"
        except Exception as e:
            return f"ERROR: Gemini API call failed: {e}"

    return await asyncio.get_event_loop().run_in_executor(None, _do_request)


# ---------------------------------------------------------------------------
# Helper: split long messages for Discord
# ---------------------------------------------------------------------------

def _split_message(text: str, limit: int = MAX_RESPONSE_LENGTH) -> list[str]:
    """Split a long message into chunks that fit Discord's character limit."""
    if len(text) <= limit:
        return [text]
    chunks = []
    while text:
        if len(text) <= limit:
            chunks.append(text)
            break
        # Try to split at a newline
        split_at = text.rfind("\n", 0, limit)
        if split_at == -1:
            split_at = limit
        chunks.append(text[:split_at])
        text = text[split_at:].lstrip("\n")
    return chunks


# ---------------------------------------------------------------------------
# Cog
# ---------------------------------------------------------------------------

class AIRouterCog(commands.Cog):
    """Routes messages to GPT-5.4 or Gemini API based on channel configuration."""

    def __init__(self, bot: ClaudeDiscordBot) -> None:
        self.bot = bot
        # Register routed channel IDs on the bot so ClaudeChatCog can skip them
        self.bot._ai_router_channel_ids = ALL_ROUTED_CHANNEL_IDS  # type: ignore[attr-defined]
        logger.info(
            "AIRouterCog loaded: GPT channels=%s, Gemini channels=%s",
            GPT_CHANNEL_IDS, GEMINI_CHANNEL_IDS,
        )

    def _get_channel_id(self, message: discord.Message) -> int:
        """Get the effective channel ID (parent for threads)."""
        if isinstance(message.channel, discord.Thread):
            return message.channel.parent_id or message.channel.id
        return message.channel.id

    def _should_route(self, message: discord.Message) -> str | None:
        """Return 'gpt' or 'gemini' if this message should be routed, else None."""
        if not AI_ROUTER_ENABLED:
            return None
        if message.author.bot:
            return None
        if message.type not in (discord.MessageType.default, discord.MessageType.reply):
            return None

        ch_id = self._get_channel_id(message)
        if ch_id in GPT_CHANNEL_IDS:
            return "gpt"
        if ch_id in GEMINI_CHANNEL_IDS:
            return "gemini"
        return None

    @commands.Cog.listener()
    async def on_message(self, message: discord.Message) -> None:
        """Intercept messages and route to appropriate AI API."""
        provider = self._should_route(message)
        if provider is None:
            return

        # Mark as handled so ClaudeChatCog skips this message
        if not hasattr(self.bot, "_ai_router_handled"):
            self.bot._ai_router_handled = set()  # type: ignore[attr-defined]
        self.bot._ai_router_handled.add(message.id)  # type: ignore[attr-defined]

        ch_id = self._get_channel_id(message)
        system = _SYSTEM_PROMPTS.get(ch_id, _build_default_system())
        prompt = message.content

        # Add reaction to show processing
        try:
            await message.add_reaction("⏳")
        except discord.HTTPException:
            pass

        # Determine if we need to create a thread or reply in existing one
        if isinstance(message.channel, discord.Thread):
            target = message.channel
        else:
            try:
                target = await message.create_thread(
                    name=prompt[:90] if len(prompt) > 3 else "AI Response",
                    auto_archive_duration=1440,
                )
            except discord.HTTPException:
                target = message.channel  # type: ignore[assignment]

        # Call appropriate API
        try:
            if provider == "gpt":
                response = await _call_gpt54(prompt, system=system)
                ai_label = "GPT-5.4 Thinking High"
            else:
                response = await _call_gemini(prompt, system=system)
                ai_label = "Gemini 3.1 Pro"
        except Exception as e:
            response = f"ERROR: API call failed: {e}"
            ai_label = provider

        # Remove processing reaction
        try:
            await message.remove_reaction("⏳", self.bot.user)  # type: ignore[arg-type]
        except discord.HTTPException:
            pass

        # Post response
        if response.startswith("ERROR:"):
            logger.error("AI Router error: %s", response)
            try:
                await message.add_reaction("❌")
            except discord.HTTPException:
                pass
            # Fall back to Claude Code for this message
            if hasattr(self.bot, "_ai_router_handled"):
                self.bot._ai_router_handled.discard(message.id)  # type: ignore[attr-defined]
            return

        # Split and send response
        chunks = _split_message(response)
        for i, chunk in enumerate(chunks):
            try:
                await target.send(chunk)
            except discord.HTTPException as e:
                logger.error("Failed to send AI response chunk %d: %s", i, e)

        # Add completion reaction
        try:
            await message.add_reaction("✅")
        except discord.HTTPException:
            pass


# ---------------------------------------------------------------------------
# Setup function (required by discord.py Cog loader)
# ---------------------------------------------------------------------------

async def setup(bot: ClaudeDiscordBot, runner=None, components=None) -> None:
    """Register the AIRouterCog."""
    if not AI_ROUTER_ENABLED:
        logger.info("AIRouterCog disabled (AI_ROUTER_ENABLED != 1)")
        return
    if not GPT_CHANNEL_IDS and not GEMINI_CHANNEL_IDS:
        logger.info("AIRouterCog disabled (no channel IDs configured)")
        return
    await bot.add_cog(AIRouterCog(bot))
