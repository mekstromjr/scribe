# Slack app setup

The Slack CLI (`slack login`) is **not usable here** — it returns:

> This workspace is not eligible for the next generation Slack platform.

That gate is about Slack's Deno-hosted platform (Run On Slack), which requires a paid
plan. **Socket Mode, bot users, and bot tokens are all available on free plans**, and
scribe uses only those — it runs in the cluster and dials out. So the app is created
through the web UI instead; `manifest.json` in this repo is the source of truth either
way.

## 1. Create the app from the manifest

1. Go to <https://api.slack.com/apps>
2. **Create New App** -> **From an app manifest**
3. Pick the workspace
4. Paste the contents of [`manifest.json`](manifest.json)
5. Create

## 1a. If DMs show "Sending messages to this app has been turned off"

The manifest now sets `features.app_home.messages_tab_read_only_enabled: false`, but an
app created before that was added defaults the Messages tab to **read-only** — the DM box
is disabled and the bot can never be reached.

Fix without recreating the app: **App Home** -> *Show Tabs* -> **Messages Tab** -> enable
it and check **"Allow users to send Slash commands and messages from the messages tab"**.

## 2. Install it and collect the two tokens

Socket Mode needs **two** tokens, and the manifest can only produce the first:

| Token | Where | Notes |
|---|---|---|
| `xoxb-...` | **Install App** -> Install to Workspace | Bot token. Grants the manifest's scopes. |
| `xapp-...` | **Basic Information** -> **App-Level Tokens** -> Generate | Needs the `connections:write` scope. **Not creatable from a manifest** — this step is manual. |

## 3. Store them

```
secret/infra/scribe
  slack-bot-token   xoxb-...
  slack-app-token   xapp-...
  gitlab-vault-token  <project access token, write_repository on mekadmin/mekvault>
```

Locally, `SCRIBE_SLACK_BOT_TOKEN` / `SCRIBE_SLACK_APP_TOKEN` / `SCRIBE_GITLAB_TOKEN`.

## Why these scopes

Deliberately minimal. `im:history` plus `app_mentions:read` means scribe sees **only**
DMs sent to it and messages that explicitly @-mention it. There is no
`channels:history`, so it cannot read channel traffic it was not addressed in — a real
privacy boundary for a bot that forwards content into a personal vault.

`socket_mode_enabled: true` is load-bearing: Slack cannot reach this network (the
cluster is behind Tailscale CGNAT with no public ingress), so an outbound WebSocket is
what makes a Slack integration possible at all without exposing anything.
