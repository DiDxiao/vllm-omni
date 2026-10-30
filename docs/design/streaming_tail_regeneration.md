# 流式视频尾部重生成（Streaming Tail Regeneration）设计文档

## 1. 设计动机

流式扩散视频生成（如 Helios）按 chunk 递进：每个 chunk 的去噪结果会累积进
`history_latents`，后续 chunk 以历史帧为条件继续生成。如果用户对已生成视频的
**尾部**不满意（例如 prompt 改变后期望不同走向），重新生成整段视频需要把前缀
重新去噪一遍，代价大且前缀本身没有任何变化。

本特性允许客户端引用一次**已完成的流式会话**：

- 指定起始帧 `regenerate_from_frame`，起始帧之前的所有帧**直接复用快照 latent，
  一次 VAE 解码输出，不做任何去噪**；
- 只有起始帧之后的尾部用新 prompt 重新去噪；
- 前缀与新尾部之间通过 chunk 级 prompt 线性插值平滑过渡，避免画风突变。

**一句话总结**：用「保存的历史 latent + 直接解码前缀 + 只去噪尾部」把尾部重生成
的成本从 O(全片) 降到 O(尾部)。

## 2. 架构总览

特性拆成「模型无关的存储层」和「模型相关的快照/历史恢复逻辑」两层，便于未来其它
流式视频模型复用：

```text
┌─────────────────────────────────────────────────────────────────┐
│ API 层 (OpenAI-compatible video API)                            │
│   protocol/videos.py          新增 3 个请求字段                  │
│   serving_video.py            sync 路径透传 extra_args           │
│   serving_video_output_stream  streaming 路径透传 + 快照 key 透传 │
└──────────────────────────────┬──────────────────────────────────┘
                               │ extra_args
┌──────────────────────────────▼──────────────────────────────────┐
│ Runner 层 (diffusion_model_runner.py) —— 模型无关               │
│   GenerationSessionStore   LRU 快照仓库（latent 留在 GPU）       │
│   _resolve_regeneration()  新请求 → 查快照 → 注入 state.extra   │
│   _save_generation_snapshot() 成功完成的流式请求 → 存快照        │
└──────────────────────────────┬──────────────────────────────────┘
                               │ state.extra["regenerate_from_snapshot"]
┌──────────────────────────────▼──────────────────────────────────┐
│ Pipeline 层 (pipeline_helios.py) —— 模型相关                    │
│   snapshot_generation_state()    导出最终 history_latents 等     │
│   _seed_history_from_snapshot()  由快照恢复历史：前缀解码/窗口回填 │
│   post_decode: 首个重生成 chunk 拼接前缀视频帧                    │
└─────────────────────────────────────────────────────────────────┘
```

**一句话总结**：存储在 Runner 层模型无关，快照/历史恢复在 Pipeline 层模型相关，
两层通过 `state.extra` 中不透明的 `payload` 解耦。

## 3. 核心概念

| 概念 | 说明 |
| --- | --- |
| 会话（session） | 一次成功完成的流式生成，以 `external_request_id`（客户端可见的 request id）标识 |
| 快照（snapshot） | 会话结束时保存的不透明 payload：最终 `history_latents`、结束 prompt embedding、chunk 结构元数据 |
| 起始帧（start frame） | `regenerate_from_frame`，0-based 输出像素帧下标；之前的前缀帧不再参与去噪 |
| 前缀（prefix） | `[0, start_frame)` 的帧。其 latent 直接 VAE 解码，一次性拼到首个重生成 chunk 的输出前面 |
| 尾部（tail） | `[start_frame, num_frames)`，用新 prompt 重新去噪，按整 chunk 补齐 |
| prompt 过渡 | 前缀沿用旧会话 prompt，尾部从旧 prompt 到新 prompt 在 `transition_chunks` 个 chunk 内线性插值（embedding 空间） |

**一句话总结**：核心心智模型是「前缀=只读回放，尾部=重算，边界=插值」。

## 4. 数据结构

### 4.1 快照 payload（模型相关，由 Pipeline 产出）

见 `vllm_omni/diffusion/models/helios/pipeline_helios.py:991`：

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `history_latents` | `Tensor[B,C,L,H,W]` | 已归一化 latent，`detach()` 后**留在生成设备**，不拷 CPU |
| `prompt_embeds` | `Tensor \| None` | 结束时的 prompt embedding，供重生成时做旧→新插值 |
| `num_latent_frames_per_chunk` | `int` | 源会话每 chunk 的 latent 帧数（含 keep_first 帧） |
| `keep_first_frame` | `bool` | 源会话是否保留首帧 |
| `history_sizes` | `list[int]` | long/mid/short 窗口划分 |
| `total_generated_latent_frames` | `int` | 实际去噪生成的 latent 帧总数 |
| `num_frames` / `width` / `height` | `int` | 源会话输出规格（校验分辨率一致性用） |

### 4.2 快照外壳（模型无关）

`GenerationSessionSnapshot`（`vllm_omni/diffusion/generation_session.py:37`）
持有 `session_id / num_frames / width / height / fps / model / payload` 与
`created_at / last_access`；`GenerationSessionStore`（同文件 `:80`）用
`OrderedDict` 实现 LRU，默认容量 32，可由 `od_config.generation_session_max_entries`
配置。**容量满时驱逐最久未访问项，latent 随之释放显存。**

**一句话总结**：外壳只管「按 id 存取 + LRU 驱逐」，对 payload 内容零感知。

## 5. 数据流跟踪

### 5.1 保存流（会话正常结束时）

```text
streaming 请求完成
  └─ runner denoise loop: finished && streaming && result.error is None
       └─ _save_generation_snapshot()            (diffusion_model_runner.py:656)
            ├─ pipeline.snapshot_generation_state(state)  → payload
            └─ key = extra_args["external_request_id"] or state.request_id
                 └─ GenerationSessionStore.put(snapshot)
```

`external_request_id` 由 streaming 入口在 `serving_video_output_stream.py:706`
写入 `extra_args`——即 **API 层收到的原始 request id**，避开引擎内部加的唯一性
后缀，客户端回传什么就匹配什么。

### 5.2 重生成流

```text
客户端: session_id=S, frame=F, new prompt
  └─ API 层校验成对出现，写入 extra_args
       └─ runner._prepare_batch_inputs() 对新请求调用 _resolve_regeneration()  (:691)
            ├─ store.get(S) → snapshot（未命中/被驱逐 → 400）
            └─ state.extra["regenerate_from_snapshot"]     = snapshot.payload
               state.extra["regenerate_from_start_frame"]  = F
       └─ pipeline.prepare_encode(state)
            ├─ _seed_history_from_snapshot()               (pipeline_helios.py:1019)
            │    ├─ 前缀 latent → VAE 一次性解码 → prefix_video
            │    ├─ image_latents ← 最后保留 chunk 的第一帧（视觉连续性）
            │    └─ history_latents ← 窗口(19)尾部回填最近 k 帧真实 latent，前导补零
            ├─ 若快照带 prompt_embeds → start_prompt_transition()（首 chunk 即混入第一插值步）
            └─ state.chunk_index = start_chunk（直接跳到起始 chunk）
       └─ 逐 chunk 去噪尾部 …
       └─ post_decode: 首个重生成 chunk 输出时
            output = cat([prefix_video, current_video])    (:958)，随后 pop 该键只拼一次
```

### 5.3 帧号 → chunk 映射（历史恢复的核心几何）

设源会话每 chunk 生成 `new = num_latent_frames_per_chunk - keep_first` 个新
latent 帧，时间下采样 `t = vae_scale_factor_temporal`，每 chunk 解码出的像素帧数
`chunk_px = (num_latent_frames_per_chunk - 1) * t + 1`：

```text
new_latent_per_chunk = num_latent_per_chunk - (1 if keep_first else 0)
pixel_per_chunk      = new_latent_per_chunk * t
start_chunk          = clamp(F // pixel_per_chunk, 0, src_total_chunks - 1)
prefix_real          = 0                              (start_chunk == 0)
                     = num_latent_per_chunk + (start_chunk-1) * new_latent_per_chunk
prefix_px            = (prefix_real - 1) * t + 1
tail_px              = max(num_frames - prefix_px, chunk_px)   # 总长以新请求 num_frames 为准
tail_chunks          = ceil(tail_px / chunk_px)
total_chunks         = start_chunk + tail_chunks
```

注意 `history_latents` 窗口长度**始终固定为训练窗口 19 个 latent 帧**
（long 16 + mid 2 + short 1），只回填最近 `k = min(prefix_real, 19)` 帧，不做拼接生长。

**一句话总结**：保存流在会话收尾处旁路挂载，重生成流在 `prepare_encode` 处一次性
完成「解码前缀 + 回填窗口 + 跳 chunk」，之后走完全正常的逐 chunk 生成路径。

## 6. 实现细节（分层改动清单）

| 层 | 文件 | 关键点 |
| --- | --- | --- |
| API | `vllm_omni/entrypoints/openai/protocol/videos.py:149,158,167` | 新增 `regenerate_from_session_id` / `regenerate_from_frame`(ge=0) / `regenerate_prompt_transition_chunks`(ge=0) |
| API | `vllm_omni/entrypoints/openai/serving_video.py:196` | sync 路径：两参数必须成对，否则 400 |
| API | `vllm_omni/entrypoints/openai/serving_video_output_stream.py:552` | streaming 路径：成对校验 + `regenerate_prompt_transition_chunks` 透传 |
| Runner | `vllm_omni/diffusion/worker/diffusion_model_runner.py:656` | 仅 `streaming_output=True` 且成功结束时才存快照（try/except 兜底，存失败只告警） |
| Runner | `vllm_omni/diffusion/worker/diffusion_model_runner.py:691` | 未提供参数直接返回；只给其一抛错；快照不存在抛「已驱逐或未完成」 |
| Pipeline | `vllm_omni/diffusion/models/helios/pipeline_helios.py:1019` | 分辨率不一致直接拒绝；`state.chunk_index = start_chunk` 跳过前缀去噪（函数名中的 seed 指「用快照预置历史状态」，非随机种子） |
| Pipeline | `vllm_omni/diffusion/models/helios/pipeline_helios.py:539` | 快照带 prompt 时首 chunk 立即应用第一插值步（不等 chunk 边界） |
| Pipeline | `vllm_omni/diffusion/models/helios/pipeline_helios.py:958` | `regeneration_prefix_video` 只在首个重生成 chunk 拼接一次，用完即 `pop` |
| Prompt | `vllm_omni/diffusion/prompt_update.py:106` | `start_prompt_transition`：`transition_chunks<=0` 立即切换；否则构造 `_PromptUpdateState` 并预先 `advance_transition()` 一步 |
| Client | `examples/online_serving/streaming_video_generation/streaming_video_client.py` | `--regenerate-from-session`（显式 dest 到 `regenerate_from_session_id`）/ `--regenerate-from-frame` / `--regenerate-prompt-transition-chunks` |

另外本 PR 顺带包含 streaming 入口的 video-to-video 支持
（`video_reference` → `[1,C,T,H,W]` 归一化张量，见
`serving_video_output_stream.py` 中 `_decode_video_reference` /
`_video_frames_to_tensor`），它走的正是上面 5.2 中未被重生成分支覆盖的
`video_latents` 历史条件构建路径。

**一句话总结**：API 层只做校验透传，Runner 层管存储与注入，Pipeline 层管几何与
历史恢复，客户端一个 flag 即可使用。

## 7. 参数示例

### 7.1 HTTP 请求（streaming session.interaction）

```json
{
  "type": "session.start",
  "model": "Helios-Distilled",
  "prompt": "a cat turning its head to the camera",
  "num_frames": 121,
  "regenerate_from_session_id": "video-abc123",
  "regenerate_from_frame": 60,
  "regenerate_prompt_transition_chunks": 3
}
```

- `regenerate_from_frame=60`：前 60 帧直接复用旧会话 latent 解码，从第 61 帧开始重算；
- `regenerate_prompt_transition_chunks=3`：重生成的前 3 个 chunk 内，prompt 从旧会话
  prompt 线性过渡到新 prompt；`0` 表示立即切换；缺省为 3。

### 7.2 命令行（示例 client，需 `pip install av`）

```bash
# 第一次生成（记住返回的 request id）
python streaming_video_client.py --prompt "一只猫回头看镜头" --num-frames 121

# 从该会话第 60 帧开始换 prompt 重生成尾部
python streaming_video_client.py \
    --regenerate-from-session <request_id> \
    --regenerate-from-frame 60 \
    --prompt "一只猫跳上桌子" \
    --num-frames 121
```

**一句话总结**：两个必填参数成对出现，一个可选参数控制过渡节奏，其余与普通
流式请求完全一致。

## 8. 设计亮点

1. **存储与模型解耦**：`GenerationSessionStore` 不认识 Helios；任何流式视频模型只要
   实现 `snapshot_generation_state()` 钩子即可获得重生成能力（Runner 侧用
   `hasattr` 探测钩子决定是否保存）。
2. **零拷贝显存复用**：快照 latent 全程留在 GPU，重生成时 device-to-device
   切片回填，无 CPU 往返；驱逐即释放显存。
3. **前缀逐字节稳定**：前缀不做任何「重新去噪+截断」近似，直接解码原始 latent，
   保证与原会话视觉一致；`image_latents` 取最后保留 chunk 的第一帧，维持
   短历史条件与原图连续。
4. **总量守恒**：重生成视频总像素长度由**新请求的 `num_frames`** 决定
   （`tail_px = num_frames - prefix_px`），而非沿用原会话时长，尾部按整 chunk 补齐。
5. **复用既有插值机制**：prompt 过渡直接复用 mid-stream prompt update 的
   `_PromptUpdateState` 插值器，只是把「从下一个 chunk 边界生效」改为「当前 chunk
   立即生效」。
6. **键对齐客户端视角**：快照 key 用 API 层原始 request id（`external_request_id`），
   客户端无需感知引擎内部的 id 改写。

**一句话总结**：亮点集中在「解耦、零拷贝、前缀稳定、总量守恒、机制复用」五点。

## 9. 常见误解

1. **「重生成会重算前缀的 noise step」** —— 不会。前缀 latent 原样解码，唯一代价是
   一次 VAE decode；`state.chunk_index` 直接从 `start_chunk` 开始。
2. **「`history_latents` 会随前缀变长」** —— 不会。它恒为训练窗口大小（19 latent
   帧），只回填最近 `k` 帧真实 latent，前导补零；增长的是 `total_chunks`。
3. **「快照永久保存」** —— 不是。LRU 上限默认 32 个会话（可配
   `generation_session_max_entries`），被驱逐后重生成请求返回明确的
   「unknown or expired session」错误；且**只有 streaming 模式**会保存快照。
4. **「两参数可以只传一个」** —— 不行。API 层、Runner 层双重校验必须成对提供。
5. **「换分辨率重生成也行」** —— 不行。`_seed_history_from_snapshot` 校验
   width/height 必须与源会话一致，否则直接拒绝。
6. **「多 worker 下任何 worker 都能接重生成请求」** —— 不行。快照在进程内
   （in-process），多 worker 部署需要 session→worker 亲和路由，本 PR 明确将其列为
   范围外。

**一句话总结**：最容易踩的坑是「以为前缀会重算 / 快照永久保存 / 可换分辨率」，
三者都与实现相反。
