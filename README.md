# OpenClaw Intelligent Agent

一个 Node.js IoT 智能中枢项目，集成了：

- 设备注册与能力管理（`data/iot-devices.json` 持久化）
- 原子化设备控制/多步骤执行/自动化任务
- Telegram 机器人交互（文本 + 图片）
- 本地或远端 OpenAI 兼容 LLM（用于聊天与工具编排）
- 本地视觉模型链路（Ollama + Python 脚本）

---

## 1. 项目结构（核心）

- `src/iot/`：IoT 能力、设备存储、自动化、校验 schema
- `src/agent/`：LLM 调用、工具白名单、编排器、会话与审计
- `src/bot/telegramAgentBot.ts`：Telegram 消息入口
- `src/routes/httpRoute.ts`：HTTP API 路由
- `script/Image_Recognition/describe_image.py`：本地视觉识别脚本
- `data/`：
  - `iot-devices.json`：设备与语义信息
  - `agent-audit/*.jsonl`：按天审计日志
  - `agent-sessions/*.json`：按 session 存储聊天会话

---

## 2. 快速开始

### 2.1 环境准备

```bash
cp .env.example .env
```

至少配置：

- `TELEGRAM_BOT_TOKEN`
- `NOTIFY_TARGET_USER_ID`（多个用逗号分隔）
- `AGENT_LLM_BASE_URL`
- `AGENT_LLM_MODEL`
- `AGENT_LLM_API_KEY`

### 2.2 启动

```bash
npm run compose:up
```

或：

```bash
docker compose up -d
```

停止：

```bash
npm run compose:down
```

---

## 3. LLM 配置（聊天/编排）

当前聊天与编排统一走 OpenAI 兼容接口：

- `AGENT_LLM_BASE_URL`（例如 `https://api.openai.com/v1` 或 `http://host.docker.internal:11434/v1`）
- `AGENT_LLM_MODEL`
- `AGENT_LLM_API_KEY`
- `AGENT_LLM_TIMEOUT_SEC`
- `AGENT_LLM_MAX_RETRIES`

### 3.1 使用本地 Ollama 作为聊天模型

`.env` 参考：

```env
AGENT_LLM_BASE_URL=http://host.docker.internal:11434/v1
AGENT_LLM_MODEL=qwen2.5:1.5b
AGENT_LLM_API_KEY=local-ollama
AGENT_LLM_TIMEOUT_SEC=600
AGENT_LLM_MAX_RETRIES=2
```

---

## 4. 本地视觉模型配置（图片识别）

`.env` 参考：

```env
OPENCLAW_IMAGE_VIA_SCRIPT=1
OLLAMA_HOST=http://host.docker.internal:11434
OLLAMA_MODEL=blaifa/InternVL3_5:8B
OPENCLAW_LOCAL_VISION_TIMEOUT_SEC=240
```

说明：

- Telegram 图片消息会走本地视觉脚本链路
- 图片 caption 会作为 prompt 传入视觉模型

---

## 5. Telegram 机器人

关键变量：

- `TELEGRAM_BOT_TOKEN`
- `NOTIFY_TARGET_USER_ID`
- `AGENT_TELEGRAM_ENABLED`（默认 1）
- `AGENT_TELEGRAM_STARTUP_SUMMARY`（默认 1）

行为：

- 文本消息：先回 `Accepted. Processing...`，后台异步处理后再发送结果
- 图片消息：下载图片 -> 本地视觉推理 -> 回发描述
- 白名单：`NOTIFY_TARGET_USER_ID` 非空时，仅白名单用户可调用

---

## 6. 常用 API

### 6.1 设备管理

- `POST /iot/devices/register`
- `GET /iot/devices`
- `GET /iot/devices/:chipId`
- `PATCH /iot/devices/:chipId/semantic`
- `POST /iot/openclaw/update-semantic`
- `POST /iot/openclaw/update-semantic-template-atomic`

### 6.2 Agent 与执行

- `GET /iot/openclaw/device-summary`
- `GET /iot/openclaw/llm-device-context`
- `POST /iot/openclaw/resolve-and-execute`
- `POST /iot/openclaw/relative-write-and-verify`
- `POST /iot/openclaw/execute-plan-atomic`
- `POST /iot/openclaw/automations/create-and-verify`
- `POST /iot/openclaw/automations/delete-and-verify`
- `GET /iot/openclaw/automations/list-and-verify`

### 6.3 图片与流式

- `POST /iot/openclaw/describe-image`
- `POST /iot/openclaw/describe-image-stream`（SSE）
- `POST /iot/openclaw/agent-reply-stream`（SSE）

---

## 7. 运行与评估脚本

```bash
npm run build
npm run dev
npm run test
npm run eval:nlu
npm run eval:automation:interval-multi
```

---

## 8. 日志与排查

### 8.1 审计日志

- 文件：`data/agent-audit/YYYY-MM-DD.jsonl`
- 关键事件示例：
  - `tg_update_received`
  - `tg_text_async_started` / `tg_text_async_sent` / `tg_text_async_failed`
  - `agent_chain_step`
  - `llm_call_start` / `llm_call_success` / `llm_call_error`

### 8.2 快速判断“是否走本地聊天模型”

在审计日志里看 `llm_call_start.payload`：

- `baseUrl` 是否为 `http://host.docker.internal:11434/v1`
- `model` 是否为你配置的本地模型名

---

## 9. 常见问题

### Q1: Telegram 发消息没反应

检查：

1. `TELEGRAM_BOT_TOKEN` 是否有效
2. `NOTIFY_TARGET_USER_ID` 是否包含当前 chat id
3. 容器是否已重启并加载新 `.env`
4. 审计日志是否出现 `tg_update_received`

### Q2: 本地模型偶发超时或拒绝连接

- 先验证容器内能访问 Ollama：
  - `docker compose exec web sh -lc 'curl -sS http://host.docker.internal:11434/api/tags'`
- 查看审计中的 `llm_call_error` 细节（`ECONNREFUSED` / timeout / HTTP status）

### Q3: 为什么闲聊也触发工具调用

当前编排支持 `final`（纯聊天）与 `tool_call`（设备动作）双路由；  
若仍触发工具，请查看 `agent_chain_step` 和 `llm_plan` 原始输出，确认模型路由是否偏移。