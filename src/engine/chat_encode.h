// The container's chat encoding: roles, thinking modes, tool calls and the output parser.
//
// A transcription of the checkpoint's own `encoding/encoding_dsv4.py`, which is the authority — this
// release ships NO Jinja chat_template and points at that folder instead. The spec, its sources and
// the byte-exact fixtures are under `tests/fixtures/chat/`; `tests/test_chat_encode.cpp` drives
// the same four conversations the reference does and compares byte for byte.
//
// Deliberately free of HTTP and GPU types: this is string in, string out, so it is testable on a
// host with no device and the fixtures are the acceptance criteria.

#pragma once

#include <string>
#include <vector>

namespace aff {

// One entry of the OpenAI `tools` array, already unwrapped from {"type":"function","function":{…}}.
struct ToolDef {
  // The function object's raw JSON text. Re-emitted with Python's separators rather than rebuilt,
  // so key order and every value's spelling survive — see json_compact in the .cpp for why that is
  // what byte-matching the reference requires.
  std::string function_json;
};

struct ToolCall {
  std::string id;         // OpenAI's call id. Only used to order tool results; may be empty.
  std::string name;
  std::string arguments;  // a JSON object, as a string, exactly as the OpenAI schema wants
};

struct ChatMsg {
  std::string role;               // system | user | assistant | tool | latest_reminder | developer
  std::string content;
  std::string reasoning_content;  // assistant only
  std::string tool_call_id;       // tool only, and only to sort results by call order
  std::string task;               // quick-instruction token: action|query|authority|domain|title|read_url
  std::vector<ToolCall> tool_calls;  // assistant only
  std::vector<ToolDef> tools;        // system or developer only
  std::string response_format;       // system or developer only; the raw JSON object
};

enum class ThinkingMode { Chat, Thinking };
enum class ReasoningEffort { Low, High, Max };

struct EncodeOpts {
  ThinkingMode    thinking_mode    = ThinkingMode::Thinking;
  // Only in thinking mode, and only as a text prefix at the very start. `Low` adds nothing.
  ReasoningEffort reasoning_effort = ReasoningEffort::Low;
  // Strip reasoning from assistant turns before the last user message. FORCED OFF when any message
  // carries tools, because a tool-calling conversation needs the whole chain.
  bool            drop_thinking    = true;
  // The prompt string carries its own BOS, so the tokenizer must NOT add another. Callers pass
  // add_bos=false to Tokenizer::encode.
  bool            add_bos          = true;
  // DeepSeek-V4.1's `encoding.py`: effort is a NUMBER, rendered as
  // `<｜System｜>Reasoning Effort: N (range 1-100, ...)\n\n` before the first message in thinking
  // mode, and system messages carry the `<｜System｜>` token (V4 renders a leading system message
  // bare). `reasoning_effort` above is ignored when this is set. Tools still render V4's DSML tags —
  // V4.1 renamed them ("<｜DSML｜ calls>" etc.) and that is NOT ported yet.
  bool            v41              = false;
  int             effort_budget    = 100;
};

// The prompt for a conversation, ending in the assistant prefix and its think marker.
std::string encode_messages(const std::vector<ChatMsg>& msgs, const EncodeOpts& opts);

struct ParsedMessage {
  std::string content;
  std::string reasoning_content;
  std::vector<ToolCall> tool_calls;
  // The reference parser raises on malformed output and its own docstring says a server wants more
  // than that. So this reports instead: `ok` false leaves `content` holding the raw text, which is
  // the only honest thing to return when the markup did not parse.
  bool        ok = true;
  std::string error;
};

// One assistant turn of model output, with or without its trailing EOS.
ParsedMessage parse_completion(const std::string& text, ThinkingMode mode);

// Exposed for the tests and for the streaming path, which has to hold back any prefix that might
// still turn into a tool-call marker.
extern const char* const kToolCallsOpen;   // "\n\n<｜DSML｜tool_calls"
extern const char* const kThinkEnd;        // "</think>"
extern const char* const kEosStr;          // "<｜end▁of▁sentence｜>"

} // namespace aff
