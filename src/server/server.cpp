#include "server.h"

#include <arpa/inet.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/socket.h>
#include <unistd.h>
#include <atomic>
#include <cctype>
#include <cstring>
#include <cstdlib>
#include <cstdio>
#include <ctime>
#include <thread>

namespace aff {

namespace {
const char* skip_ws(const char* p) { while (*p == ' ' || *p == '\n' || *p == '\t' || *p == '\r') ++p; return p; }

// A key at the TOP level of `s`. The plain search matches the same name anywhere, nested objects
// included: a get_weather tool with a "temperature" property would otherwise be read as the
// sampler's temperature, parse to 0, and pin the whole server to greedy decoding.
const char* find_top_key(const std::string& s, const char* key) {
  const std::string pat = std::string("\"") + key + "\"";
  size_t depth = 0;
  bool in_str = false, esc = false;
  for (size_t i = 0; i < s.size(); ++i) {
    const char c = s[i];
    if (in_str) {
      if (esc) esc = false;
      else if (c == '\\') esc = true;
      else if (c == '"') in_str = false;
      continue;
    }
    if (c == '"') {
      // Only a key is followed by a colon; a value that happens to read the same is not one.
      if (depth == 1 && s.compare(i, pat.size(), pat) == 0) {
        const char* q = skip_ws(s.c_str() + i + pat.size());
        if (*q == ':') return q + 1;
      }
      in_str = true;
      continue;
    }
    if (c == '{' || c == '[') ++depth;
    else if (c == '}' || c == ']') { if (depth) --depth; }
  }
  return nullptr;
}

std::string read_json_string(const char*& p) {
  p = skip_ws(p);
  if (*p != '"') return "";
  ++p;
  std::string out;
  while (*p && *p != '"') {
    if (*p == '\\' && p[1]) {
      ++p;
      switch (*p) {
        case 'n': out.push_back('\n'); break;
        case 't': out.push_back('\t'); break;
        case 'r': out.push_back('\r'); break;
        case '"': out.push_back('"'); break;
        case '\\': out.push_back('\\'); break;
        case 'u': {
          uint32_t cp = (uint32_t)std::strtoul(std::string(p + 1, p + 5).c_str(), nullptr, 16);
          p += 4;
          // A character outside the BMP is escaped as a surrogate PAIR, which is what any client
          // sending ASCII-safe JSON emits — Python's json.dumps does by default. Decoding the halves
          // separately gives two sequences that are not valid UTF-8, so an emoji in a tool result
          // reaches the model as mojibake and comes back out the same way.
          const auto hex4 = [](const char* q) {
            for (int i = 0; i < 4; ++i)
              if (!std::isxdigit((unsigned char)q[i])) return false;
            return true;
          };
          if (cp >= 0xD800 && cp < 0xDC00 && p[1] == '\\' && p[2] == 'u' && hex4(p + 3)) {
            const uint32_t lo = (uint32_t)std::strtoul(std::string(p + 3, p + 7).c_str(), nullptr, 16);
            if (lo >= 0xDC00 && lo < 0xE000) {
              cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
              p += 6;
            }
          }
          if (cp < 0x80) out.push_back((char)cp);
          else if (cp < 0x800) { out.push_back((char)(0xC0|(cp>>6))); out.push_back((char)(0x80|(cp&0x3F))); }
          else if (cp < 0x10000) { out.push_back((char)(0xE0|(cp>>12))); out.push_back((char)(0x80|((cp>>6)&0x3F)));
                 out.push_back((char)(0x80|(cp&0x3F))); }
          else { out.push_back((char)(0xF0|(cp>>18))); out.push_back((char)(0x80|((cp>>12)&0x3F)));
                 out.push_back((char)(0x80|((cp>>6)&0x3F))); out.push_back((char)(0x80|(cp&0x3F))); }
          break;
        }
        default: out.push_back(*p);
      }
    } else out.push_back(*p);
    ++p;
  }
  if (*p == '"') ++p;
  return out;
}
bool send_all(int fd, const char* d, size_t n) {
  while (n) {
    const ssize_t w = ::send(fd, d, n, MSG_NOSIGNAL);
    if (w <= 0) return false;
    d += w; n -= (size_t)w;
  }
  return true;
}
} // namespace

std::string json_escape(const std::string& s) {
  std::string o; o.reserve(s.size() + 16);
  for (unsigned char c : s) {
    switch (c) {
      case '"': o += "\\\""; break;
      case '\\': o += "\\\\"; break;
      case '\n': o += "\\n"; break;
      case '\r': o += "\\r"; break;
      case '\t': o += "\\t"; break;
      default:
        if (c < 0x20) { char b[8]; std::snprintf(b, sizeof(b), "\\u%04x", c); o += b; }
        else o.push_back((char)c);
    }
  }
  return o;
}


static std::string usage_json(const GenResult& g) {
  const uint64_t t = (uint64_t)g.prompt_tokens + g.completion_tokens;
  return "\"usage\":{\"prompt_tokens\":" + std::to_string(g.prompt_tokens) +
         ",\"completion_tokens\":" + std::to_string(g.completion_tokens) +
         ",\"total_tokens\":" + std::to_string(t) + "}";
}

// The raw text of the value at `key`, braces and all. The parser above reads scalars; tools and
// tool_calls are structures that get handed on verbatim, so they need their span rather than their
// meaning.
static std::string raw_value(const std::string& s, const char* key) {
  const char* p = find_top_key(s, key);
  if (!p) return {};
  const char* q = skip_ws(p);
  const size_t start = (size_t)(q - s.c_str());
  if (*q != '{' && *q != '[') return {};
  const char open = *q, close = open == '{' ? '}' : ']';
  size_t depth = 0;
  bool in_str = false, esc = false;
  for (size_t i = start; i < s.size(); ++i) {
    const char c = s[i];
    if (in_str) {
      if (esc) esc = false;
      else if (c == '\\') esc = true;
      else if (c == '"') in_str = false;
      continue;
    }
    if (c == '"') { in_str = true; continue; }
    if (c == open) ++depth;
    else if (c == close && --depth == 0) return s.substr(start, i - start + 1);
  }
  return {};
}

// Top-level elements of a JSON array, as raw text.
static std::vector<std::string> raw_elements(const std::string& arr) {
  std::vector<std::string> out;
  size_t depth = 0, start = std::string::npos;
  bool in_str = false, esc = false;
  for (size_t i = 0; i < arr.size(); ++i) {
    const char c = arr[i];
    if (in_str) {
      if (esc) esc = false;
      else if (c == '\\') esc = true;
      else if (c == '"') in_str = false;
      continue;
    }
    if (c == '"') { in_str = true; continue; }
    if (c == '{' || c == '[') { if (depth++ == 1) start = i; }
    else if (c == '}' || c == ']') {
      if (--depth == 1 && start != std::string::npos) { out.push_back(arr.substr(start, i - start + 1)); start = std::string::npos; }
    }
  }
  return out;
}

bool parse_completion_request(const std::string& body, bool chat, CompletionRequest* out) {
  if (const char* p = find_top_key(body, "model")) { const char* q = p; out->model = read_json_string(q); }
  // `max_completion_tokens` is the current OpenAI spelling and `max_tokens` the deprecated one;
  // clients send either, and reading only one silently ignores the caller's limit.
  for (const char* k : {"max_tokens", "max_completion_tokens"})
    if (const char* p = find_top_key(body, k)) {
      out->max_tokens = (uint32_t)std::strtoul(p, nullptr, 10);
      out->has_max_tokens = true;
    }
  if (const char* p = find_top_key(body, "temperature")) { out->temperature = std::strtof(p, nullptr); out->has_temperature = true; }
  if (const char* p = find_top_key(body, "top_p")) { out->top_p = std::strtof(p, nullptr); out->has_top_p = true; }
  if (const char* p = find_top_key(body, "top_k")) out->top_k = (uint32_t)std::strtoul(p, nullptr, 10);
  if (const char* p = find_top_key(body, "min_p")) out->min_p = std::strtof(p, nullptr);
  if (const char* p = find_top_key(body, "seed")) out->seed = std::strtoull(p, nullptr, 10);
  if (const char* p = find_top_key(body, "n")) out->n = (uint32_t)std::strtoul(p, nullptr, 10);
  if (const char* p = find_top_key(body, "stream")) { p = skip_ws(p); out->stream = std::strncmp(p, "true", 4) == 0; }
  // include_usage lives inside stream_options, so it is one level down and the top-level search
  // cannot see it. Pull the sub-object out and search that.
  const std::string stream_options = raw_value(body, "stream_options");
  if (const char* p = find_top_key(stream_options, "include_usage")) {
    p = skip_ws(p);
    out->include_usage = std::strncmp(p, "true", 4) == 0;
  }
  if (const char* p = find_top_key(body, "reasoning_effort")) { const char* q = p; out->reasoning_effort = read_json_string(q); }
  if (const char* p = find_top_key(body, "thinking_mode")) { const char* q = p; out->thinking_mode = read_json_string(q); }
  if (const char* p = find_top_key(body, "tool_choice")) { const char* q = p; out->tool_choice = read_json_string(q); }
  out->response_format = raw_value(body, "response_format");

  // tools[] -> the `function` object of each, verbatim. The wrapper is dropped here because that is
  // what the container's encoder renders (`tools_from_openai_format`).
  for (const std::string& t : raw_elements(raw_value(body, "tools"))) {
    std::string fn = raw_value(t, "function");
    if (fn.empty()) fn = t;                  // already unwrapped, which some clients send
    out->tools.push_back(fn);
  }

  if (const char* p = find_top_key(body, "stop")) {
    const char* q = skip_ws(p);
    if (*q == '[') { ++q; while (*q && *q != ']') { q = skip_ws(q); if (*q == '"') out->stop.push_back(read_json_string(q)); else ++q; if (*q == ',') ++q; } }
    else if (*q == '"') out->stop.push_back(read_json_string(q));
  }

  if (chat) {
    size_t p = body.find("\"messages\"");
    if (p == std::string::npos) return false;
    p = body.find('[', p);
    const char* q = body.c_str() + p + 1;
    while (*q && *q != ']') {
      if (*q == '{') {
        ChatMessage m;
        const size_t obj = (size_t)(q - body.c_str());
        size_t depth = 0, e = obj;
        for (size_t i = obj; i < body.size(); ++i) {
          if (body[i] == '{') ++depth;
          else if (body[i] == '}') { if (--depth == 0) { e = i; break; } }
        }
        const std::string sub = body.substr(obj, e - obj + 1);
        if (const char* r = find_top_key(sub, "role")) { const char* z = r; m.role = read_json_string(z); }
        // OpenAI renamed `system` to `developer`, and llama.cpp maps it back the same way. The
        // container happens to define a `developer` role of its own that renders as a USER turn --
        // taking the name at face value would put the caller's system instructions in the user's
        // mouth. `function` is the retired spelling of `tool`.
        if (m.role == "developer") m.role = "system";
        else if (m.role == "function") m.role = "tool";
        // `content` is a string OR an array of typed blocks. Every agent harness sends the array
        // form, and reading it as a string yields "" — a message the model receives as empty, which
        // it answers by saying nobody has asked it anything.
        if (const char* c = find_top_key(sub, "content")) {
          const char* z = skip_ws(c);
          if (*z == '[') {
            std::string joined;
            for (const std::string& blk : raw_elements(raw_value(sub, "content"))) {
              const char* t = find_top_key(blk, "text");
              if (!t) continue;
              const char* w = t;
              const std::string piece = read_json_string(w);
              if (piece.empty()) continue;
              if (!joined.empty()) joined += "\n\n";
              joined += piece;
            }
            m.content = joined;
          } else {
            const char* w = c;
            m.content = read_json_string(w);
          }
        }
        if (const char* c = find_top_key(sub, "reasoning_content")) { const char* z = c; m.reasoning_content = read_json_string(z); }
        if (const char* c = find_top_key(sub, "tool_call_id")) { const char* z = c; m.tool_call_id = read_json_string(z); }
        for (const std::string& tc : raw_elements(raw_value(sub, "tool_calls"))) {
          ToolCallIO c2;
          if (const char* z = find_top_key(tc, "id")) { const char* w = z; c2.id = read_json_string(w); }
          const std::string fn = raw_value(tc, "function");
          const std::string& src = fn.empty() ? tc : fn;
          if (const char* z = find_top_key(src, "name")) { const char* w = z; c2.name = read_json_string(w); }
          if (const char* z = find_top_key(src, "arguments")) { const char* w = z; c2.arguments = read_json_string(w); }
          if (!c2.name.empty()) m.tool_calls.push_back(std::move(c2));
        }
        if (!m.role.empty()) out->messages.push_back(std::move(m));
        q = body.c_str() + e + 1;
      } else ++q;
    }
    return !out->messages.empty();
  }
  if (const char* p = find_top_key(body, "prompt")) { const char* q = p; out->prompt = read_json_string(q); }
  return !out->prompt.empty();
}

// Reassembles a chunked body: each chunk is a hex length, CRLF, that many bytes, CRLF, ending at a
// zero-length chunk. Without this a streamed request reads as empty and is answered as malformed.
static std::string dechunk(const std::string& in) {
  std::string out;
  size_t i = 0;
  while (i < in.size()) {
    const size_t eol = in.find("\r\n", i);
    if (eol == std::string::npos) break;
    const size_t len = (size_t)std::strtoul(in.substr(i, eol - i).c_str(), nullptr, 16);
    if (len == 0) break;
    if (eol + 2 + len > in.size()) break;
    out.append(in, eol + 2, len);
    i = eol + 2 + len + 2;
  }
  return out;
}

void Server::serve_one(int fd) {
  int one = 1;
  ::setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));
  std::string req;
  char buf[8192];
  size_t header_end = std::string::npos, content_len = 0;
  bool chunked = false;
  for (;;) {
    const ssize_t n = ::recv(fd, buf, sizeof(buf), 0);
    if (n <= 0) { ::close(fd); return; }
    req.append(buf, (size_t)n);
    if (header_end == std::string::npos) {
      header_end = req.find("\r\n\r\n");
      if (header_end != std::string::npos) {
        // Header names are case-insensitive (RFC 7230). A client or proxy that sends
        // `content-length:` would otherwise read as a zero-length body, and the request would be
        // answered as malformed rather than served. Lowercasing preserves offsets.
        std::string h = req.substr(0, header_end);
        for (char& c : h) c = (char)std::tolower((unsigned char)c);
        const size_t cl = h.find("content-length:");
        if (cl != std::string::npos)
          content_len = (size_t)std::strtoul(req.c_str() + cl + 15, nullptr, 10);
        // A body of unknown length arrives chunked, which is what an HTTP client sends whenever it
        // streams the request. There is no Content-Length to wait for, so the terminator is.
        chunked = h.find("transfer-encoding:") != std::string::npos && h.find("chunked") != std::string::npos;
      }
    }
    if (header_end == std::string::npos) continue;
    if (chunked) { if (req.find("\r\n0\r\n", header_end) != std::string::npos) break; continue; }
    if (req.size() >= header_end + 4 + content_len) break;
  }
  const std::string head = req.substr(0, header_end);
  std::string body = req.substr(header_end + 4);
  if (chunked) body = dechunk(body);
  const bool is_chat = head.find("/v1/chat/completions") != std::string::npos;
  const bool is_comp = head.find("/v1/completions") != std::string::npos;
  const bool is_models = head.find("/v1/models") != std::string::npos;

  // A browser sends OPTIONS before any cross-origin POST and will not send the request itself
  // unless the preflight is answered. Without this every web client fails before it starts.
  if (head.rfind("OPTIONS ", 0) == 0) {
    const char* h = "HTTP/1.1 204 No Content\r\nAccess-Control-Allow-Origin: *\r\n"
                    "Access-Control-Allow-Methods: GET, POST, OPTIONS\r\n"
                    "Access-Control-Allow-Headers: *\r\nAccess-Control-Max-Age: 86400\r\n"
                    "Content-Length: 0\r\nConnection: close\r\n\r\n";
    send_all(fd, h, std::strlen(h));
    ::close(fd);
    return;
  }

  auto reply = [&](const std::string& ctype, const std::string& payload) {
    char h[512];
    const int n = std::snprintf(h, sizeof(h),
        "HTTP/1.1 200 OK\r\nContent-Type: %s\r\nContent-Length: %zu\r\n"
        "Access-Control-Allow-Origin: *\r\nConnection: close\r\n\r\n", ctype.c_str(), payload.size());
    send_all(fd, h, (size_t)n);
    send_all(fd, payload.data(), payload.size());
  };

  if (is_models) {
    const std::string j = "{\"object\":\"list\",\"data\":[{\"id\":\"" + model_name_ +
                          "\",\"object\":\"model\",\"created\":" +
                          std::to_string((long long)::time(nullptr)) +
                          ",\"owned_by\":\"affinity\"}]}";
    reply("application/json", j);
    ::close(fd);
    return;
  }
  if (!is_chat && !is_comp) {
    const std::string j = "{\"error\":{\"message\":\"not found\",\"type\":\"invalid_request_error\"}}";
    char h[256];
    const int n = std::snprintf(h, sizeof(h),
        "HTTP/1.1 404 Not Found\r\nContent-Type: application/json\r\nContent-Length: %zu\r\n\r\n", j.size());
    send_all(fd, h, (size_t)n); send_all(fd, j.data(), j.size());
    ::close(fd);
    return;
  }

  CompletionRequest r;
  if (!parse_completion_request(body, is_chat, &r)) {
    const std::string j = "{\"error\":{\"message\":\"bad request\",\"type\":\"invalid_request_error\"}}";
    char h[256];
    const int n = std::snprintf(h, sizeof(h),
        "HTTP/1.1 400 Bad Request\r\nContent-Type: application/json\r\nContent-Length: %zu\r\n\r\n", j.size());
    send_all(fd, h, (size_t)n); send_all(fd, j.data(), j.size());
    ::close(fd);
    return;
  }

  // `id` and `created` are required by the OpenAI schema on both the completion and every chunk of a
  // stream, and a client that validates its responses rejects the whole turn without them. One id
  // per request, repeated on each chunk, is what identifies the chunks as one completion.
  static std::atomic<uint64_t> seq{0};
  const std::string cmpl_id = "chatcmpl-" + std::to_string(seq.fetch_add(1)) + "-" +
                              std::to_string((uint64_t)::time(nullptr));
  const std::string envelope = "\"id\":\"" + cmpl_id + "\",\"created\":" +
                               std::to_string((long long)::time(nullptr)) + ",\"object\":\"";

  if (r.stream) {
    const char* h = "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\n"
                    "Cache-Control: no-cache\r\nAccess-Control-Allow-Origin: *\r\nConnection: close\r\n\r\n";
    send_all(fd, h, std::strlen(h));
    bool alive = true;
    const std::string obj = is_chat ? "chat.completion.chunk" : "text_completion";
    GenResult res;
    std::unique_lock<std::mutex> gen_lock(gen_mu_);
    // The first chat delta of a stream carries the role and nothing else to say it opens an
    // assistant message; clients key the new message on it, and OpenAI itself sends it.
    bool opened = false;
    gen_(r, [&](const std::string& piece, Delta chan, bool done) -> bool {
      if (!alive || done || piece.empty()) return alive;
      std::string j = "data: {" + envelope + obj + "\",\"model\":\"" + model_name_ +
                      "\",\"choices\":[{\"index\":0,";
      if (is_chat) {
        const std::string role = opened ? "" : "\"role\":\"assistant\",";
        opened = true;
        j += chan == Delta::Reasoning
                 ? "\"delta\":{" + role + "\"reasoning_content\":\"" + json_escape(piece) + "\"}"
                 : "\"delta\":{" + role + "\"content\":\"" + json_escape(piece) + "\"}";
        j += ",\"finish_reason\":null";
      } else {
        // The legacy endpoint has no channel to put reasoning in, so it carries the answer only.
        if (chan == Delta::Reasoning) return alive;
        j += "\"text\":\"" + json_escape(piece) + "\",\"finish_reason\":null";
      }
      j += "}]}\n\n";
      alive = send_all(fd, j.data(), j.size());
      return alive;
    }, &res);
    gen_lock.unlock();
    // The 200 and its headers are long gone by the time generation reports a problem, so a stream
    // carries the refusal as an `error` event — which is how OpenAI reports a mid-stream failure.
    if (alive && !res.error.empty()) {
      const std::string e = "data: {\"error\":{\"message\":\"" + json_escape(res.error) +
                            "\",\"type\":\"invalid_request_error\"}}\n\n";
      send_all(fd, e.data(), e.size());
      send_all(fd, "data: [DONE]\n\n", 14);
      ::close(fd);
      return;
    }
    // Tool calls arrive as one delta at the end rather than as text: they are not content, and a
    // client that concatenated them as content would have to re-parse markup this server already
    // parsed. The final chunk also carries the finish_reason, which is what tells the client a tool
    // round trip is expected rather than a finished answer.
    if (alive && is_chat && !res.tool_calls.empty()) {
      std::string j = "data: {" + envelope + obj + "\",\"model\":\"" + model_name_ +
                      "\",\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[";
      for (size_t i = 0; i < res.tool_calls.size(); ++i) {
        if (i) j += ",";
        j += "{\"index\":" + std::to_string(i) + ",\"id\":\"" + json_escape(res.tool_calls[i].id) +
             "\",\"type\":\"function\",\"function\":{\"name\":\"" +
             json_escape(res.tool_calls[i].name) + "\",\"arguments\":\"" +
             json_escape(res.tool_calls[i].arguments) + "\"}}";
      }
      j += "]},\"finish_reason\":null}]}\n\n";
      alive = send_all(fd, j.data(), j.size());
    }
    if (alive) {
      std::string j = "data: {" + envelope + obj + "\",\"model\":\"" + model_name_ +
                      "\",\"choices\":[{\"index\":0,";
      j += is_chat ? "\"delta\":{}" : "\"text\":\"\"";
      j += ",\"finish_reason\":\"" + res.finish_reason + "\"}]}\n\n";
      alive = send_all(fd, j.data(), j.size());
    }
    // OpenAI puts usage in one extra chunk with an EMPTY choices array, only when asked for it.
    if (alive && r.include_usage) {
      const std::string j = "data: {" + envelope + obj + "\",\"model\":\"" + model_name_ +
                            "\",\"choices\":[]," + usage_json(res) + "}\n\n";
      alive = send_all(fd, j.data(), j.size());
    }
    if (alive) send_all(fd, "data: [DONE]\n\n", 14);
    ::close(fd);
    return;
  }

  GenResult res;
  {
    std::lock_guard<std::mutex> gen_lock(gen_mu_);
    gen_(r, [](const std::string&, Delta, bool) -> bool { return true; }, &res);
  }
  if (!res.error.empty()) {
    const std::string e = "{\"error\":{\"message\":\"" + json_escape(res.error) +
                          "\",\"type\":\"invalid_request_error\"}}";
    char h[256];
    const int n = std::snprintf(h, sizeof(h),
        "HTTP/1.1 400 Bad Request\r\nContent-Type: application/json\r\nContent-Length: %zu\r\n"
        "Access-Control-Allow-Origin: *\r\nConnection: close\r\n\r\n", e.size());
    send_all(fd, h, (size_t)n); send_all(fd, e.data(), e.size());
    ::close(fd);
    return;
  }
  std::string j = "{" + envelope + std::string(is_chat ? "chat.completion" : "text_completion") +
                  "\",\"model\":\"" + model_name_ + "\",\"choices\":[{\"index\":0,";
  if (is_chat) {
    j += "\"message\":{\"role\":\"assistant\",\"content\":\"" + json_escape(res.content) + "\"";
    // Alongside content, as DeepSeek's own API returns it. Not in the OpenAI schema, and clients
    // that do not know it ignore it; a client that does gets the reasoning without having to strip
    // <think> out of the text itself.
    if (!res.reasoning_content.empty())
      j += ",\"reasoning_content\":\"" + json_escape(res.reasoning_content) + "\"";
    if (!res.tool_calls.empty()) {
      j += ",\"tool_calls\":[";
      for (size_t i = 0; i < res.tool_calls.size(); ++i) {
        if (i) j += ",";
        j += "{\"id\":\"" + json_escape(res.tool_calls[i].id) +
             "\",\"type\":\"function\",\"function\":{\"name\":\"" +
             json_escape(res.tool_calls[i].name) + "\",\"arguments\":\"" +
             json_escape(res.tool_calls[i].arguments) + "\"}}";
      }
      j += "]";
    }
    j += "}";
  } else {
    j += "\"text\":\"" + json_escape(res.content) + "\"";
  }
  j += ",\"finish_reason\":\"" + res.finish_reason + "\"}]," + usage_json(res) + "}";
  reply("application/json", j);
  ::close(fd);
}

bool Server::start(const std::string& host, uint16_t port, GenerateFn gen, std::string* err) {
  gen_ = std::move(gen);
  listen_fd_ = ::socket(AF_INET, SOCK_STREAM, 0);
  if (listen_fd_ < 0) { if (err) *err = "socket failed"; return false; }
  int one = 1;
  ::setsockopt(listen_fd_, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));
  sockaddr_in a{};
  a.sin_family = AF_INET;
  a.sin_port = htons(port);
  a.sin_addr.s_addr = (host == "0.0.0.0") ? INADDR_ANY : inet_addr(host.c_str());
  if (::bind(listen_fd_, (sockaddr*)&a, sizeof(a)) != 0) { if (err) *err = "bind failed"; return false; }
  if (::listen(listen_fd_, 64) != 0) { if (err) *err = "listen failed"; return false; }
  socklen_t sl = sizeof(a);
  ::getsockname(listen_fd_, (sockaddr*)&a, &sl);
  port_ = ntohs(a.sin_port);
  running_ = true;
  std::thread([this] {
    while (running_) {
      const int fd = ::accept(listen_fd_, nullptr, nullptr);
      if (fd < 0) { if (!running_) break; continue; }
      // One thread per connection. With a latency-first scheduler at B<=4 the connection count is
      // small by design, so a thread pool would be complexity without benefit.
      std::thread([this, fd] { serve_one(fd); }).detach();
    }
  }).detach();
  return true;
}

void Server::stop() {
  running_ = false;
  if (listen_fd_ >= 0) { ::shutdown(listen_fd_, SHUT_RDWR); ::close(listen_fd_); listen_fd_ = -1; }
}

} // namespace aff
