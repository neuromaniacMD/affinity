// OpenAI-compatible HTTP server (requirement R4).
//
// Deliberately NOT a vLLM-style continuous-batching throughput engine: per decision D1 the
// scheduler is latency-first at B=1-4, because the MoE union law makes wide batching a bad trade
// on this hardware: batching buys a little aggregate throughput for much worse per-request
// latency.
#pragma once

#include <atomic>
#include <cstdint>
#include <functional>
#include <mutex>
#include <string>
#include <vector>

namespace aff {

struct ToolCallIO {
  std::string id;         // OpenAI's call id; only used to order tool results
  std::string name;
  std::string arguments;  // a JSON object, as a string, which is what the schema wants
};

struct ChatMessage {
  std::string role, content;
  std::string reasoning_content;          // assistant turns, echoed back on the next request
  std::string tool_call_id;               // role == "tool"
  std::vector<ToolCallIO> tool_calls;     // role == "assistant"
};

struct CompletionRequest {
  std::string model;
  std::vector<ChatMessage> messages;   // chat endpoint
  std::string prompt;                  // legacy completions endpoint
  // Absent means "as much as the context allows", which is what OpenAI and llama.cpp both do; a
  // fixed default here would silently truncate a long answer at a number the caller never chose.
  uint32_t max_tokens = 0;
  bool has_max_tokens = false;
  uint32_t n = 1;                          // completions to return; only 1 is served
  float temperature = 1.0f;
  float top_p = 1.0f;
  uint32_t top_k = 0;
  float min_p = 0.0f;
  // No `repetition_penalty`: the served path samples on the device, and the device sampler takes
  // temperature, top_p, top_k and min_p only. Parsing it here and dropping it would leave a client
  // believing a knob it sent had an effect.
  bool stream = false;
  // stream_options.include_usage: OpenAI appends one extra chunk carrying usage and no choices.
  bool include_usage = false;
  std::vector<std::string> stop;
  uint64_t seed = 0;
  // The `function` object of each entry of `tools`, as raw JSON. Kept as text rather than parsed
  // because the encoder re-emits it verbatim into the prompt — see chat_encode.h.
  std::vector<std::string> tools;
  std::string tool_choice;                // "auto" | "none" | "required"; advisory
  std::string response_format;            // the raw JSON object, rendered into the system message
  std::string reasoning_effort;           // "low" | "high" | "max"
  std::string thinking_mode;              // "thinking" | "chat"
  bool has_temperature = false;           // so a server default is distinguishable from a request
  bool has_top_p = false;
};

// What one generation produced, once its markup has been parsed.
struct GenResult {
  std::string content;
  std::string reasoning_content;
  std::vector<ToolCallIO> tool_calls;
  std::string finish_reason = "stop";     // "stop" | "tool_calls" | "length"
  // `usage` is not optional in the OpenAI schema, and a client that meters context or cost reads
  // it on every reply. Counted in tokens, so reasoning counts toward completion as it does upstream.
  uint32_t prompt_tokens = 0;
  uint32_t completion_tokens = 0;
  // Set when the request cannot be served as asked. The server answers 400 rather than an empty
  // completion, so the caller sees a refusal instead of a model that had nothing to say.
  std::string error;
};

// Which channel a streamed piece belongs to. Reasoning is not the answer, and a client that shows
// the two the same way shows the model thinking out loud as if it had replied — so the stream says
// which is which and lets the client decide, exactly as `reasoning_content` does on the completed
// message.
enum class Delta { Content, Reasoning };

// Emits one piece of output. Returning false asks generation to stop (client disconnected).
using TokenSink = std::function<bool(const std::string& piece, Delta chan, bool done)>;

// Supplied by the engine; the server owns no model state.
// The sink streams text as it appears; the result carries what the markup meant, which is only
// known once the turn is complete. A tool call is not text and cannot be streamed as content.
using GenerateFn = std::function<void(const CompletionRequest&, const TokenSink&, GenResult*)>;

class Server {
public:
  bool start(const std::string& host, uint16_t port, GenerateFn gen, std::string* err);
  void stop();
  void set_model_name(const std::string& n) { model_name_ = n; }
  uint16_t port() const { return port_; }

private:
  void serve_one(int fd);
  int listen_fd_ = -1;
  uint16_t port_ = 0;
  std::string model_name_ = "DeepSeek-V4-Flash";
  // The generator closes over one engine: one KV cache, one set of scratch buffers. Threads are per
  // connection, so without this two requests interleave inside that state and both come out fluent
  // and wrong. Serving is one request at a time by construction; the rest queue here.
  std::mutex gen_mu_;
  GenerateFn gen_;
  std::atomic<bool> running_{false};
};

// Exposed for testing.
bool parse_completion_request(const std::string& body, bool chat, CompletionRequest* out);
std::string json_escape(const std::string& s);

} // namespace aff
