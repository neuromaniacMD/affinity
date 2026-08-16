#include "engine/chat_encode.h"

#include <algorithm>
#include <cstddef>
#include <set>

namespace aff {

namespace {

const char* const kBos        = "<｜begin▁of▁sentence｜>";
const char* const kUser       = "<｜User｜>";
const char* const kAssistant  = "<｜Assistant｜>";
const char* const kReminder   = "<｜latest_reminder｜>";
const char* const kThinkStart = "<think>";
const char* const kDsml       = "｜DSML｜";

// Verbatim from REASONING_EFFORT_PROMPTS. The trailing blank line is part of the prefix.
const char* const kEffortHigh =
    "Reasoning Effort: Absolute maximum with no shortcuts permitted.\n"
    "You MUST be very thorough in your thinking and comprehensively decompose the problem to resolve the root cause, rigorously stress-testing your logic against all potential paths, edge cases, and adversarial scenarios.\n"
    "Explicitly write out your entire deliberation process, documenting every intermediate step, considered alternative, and rejected hypothesis to ensure absolutely no assumption is left unchecked.\n\n";
const char* const kEffortMax =
    "Reasoning Effort: Beyond maximum — exhaustive, relentless, and uncompromising.\n"
    "You MUST reason with the utmost depth and rigor, leaving absolutely nothing to chance: exhaustively decompose the problem into its most fundamental components, trace every causal chain to its root, and resolve the underlying cause rather than any surface symptom.\n"
    "Do not stop reasoning until you have independently verified the solution from multiple angles and are certain that no assumption remains unchecked and no error remains undiscovered.\n\n";

// TOOLS_TEMPLATE with dsml_token/think tokens already substituted. Every space and newline here is
// load-bearing: the fixtures compare byte for byte.
std::string tools_block(const std::vector<ToolDef>& tools);
std::string response_format_block(const std::string& schema_json);

// ---- JSON, only as much as the format needs -----------------------------------------------------
//
// The reference emits tool schemas with `json.dumps(ensure_ascii=False)`, which keeps insertion
// order and uses ", " and ": " as separators. Rebuilding that from a parsed value model would mean
// re-spelling every number and re-escaping every string the way Python happens to, and any
// disagreement is a byte the fixture catches.
//
// So this REFORMATS the caller's JSON text instead: it walks the tokens, drops whitespace between
// them, and puts back exactly Python's separators. Key order, string escapes and the literal text of
// every number survive untouched, because none of them is ever re-derived.
//
// The one thing it cannot reproduce is Python renormalising a number's spelling (1e5 -> 100000.0).
// Tool schemas are objects, strings and arrays; no fixture contains a float, and a schema that did
// would differ only in the digits of a number the model reads as text.
std::string json_compact(const std::string& s) {
  std::string out;
  out.reserve(s.size());
  bool in_str = false, esc = false;
  for (size_t i = 0; i < s.size(); ++i) {
    const char ch = s[i];
    if (in_str) {
      out += ch;
      if (esc)            esc = false;
      else if (ch == '\\') esc = true;
      else if (ch == '"')  in_str = false;
      continue;
    }
    switch (ch) {
      case ' ': case '\t': case '\n': case '\r': continue;   // whitespace between tokens
      case '"': in_str = true; out += ch; break;
      case ',': out += ", "; break;
      case ':': out += ": "; break;
      default:  out += ch; break;
    }
  }
  return out;
}

// Whether a JSON value's text is a string literal, which is what decides `string="true|false"`.
bool json_is_string(const std::string& v) {
  size_t i = 0;
  while (i < v.size() && (v[i] == ' ' || v[i] == '\t' || v[i] == '\n' || v[i] == '\r')) ++i;
  return i < v.size() && v[i] == '"';
}

// The text of a JSON string literal, unescaped. Only the escapes the format can produce.
std::string json_unquote(const std::string& v) {
  size_t i = v.find('"');
  if (i == std::string::npos) return v;
  std::string out;
  for (++i; i < v.size(); ++i) {
    if (v[i] == '"') break;
    if (v[i] != '\\') { out += v[i]; continue; }
    if (++i >= v.size()) break;
    switch (v[i]) {
      case 'n': out += '\n'; break;
      case 't': out += '\t'; break;
      case 'r': out += '\r'; break;
      case 'b': out += '\b'; break;
      case 'f': out += '\f'; break;
      case 'u': {
        // \uXXXX. Only the BMP, and surrogate pairs are joined, which is what the encoder's own
        // ensure_ascii=False output round-trips to.
        if (i + 4 >= v.size()) return out;
        unsigned cp = (unsigned)std::stoul(v.substr(i + 1, 4), nullptr, 16);
        i += 4;
        if (cp >= 0xD800 && cp <= 0xDBFF && i + 6 < v.size() && v[i + 1] == '\\' && v[i + 2] == 'u') {
          const unsigned lo = (unsigned)std::stoul(v.substr(i + 3, 4), nullptr, 16);
          cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
          i += 6;
        }
        if (cp < 0x80) out += (char)cp;
        else if (cp < 0x800) { out += (char)(0xC0 | (cp >> 6)); out += (char)(0x80 | (cp & 0x3F)); }
        else if (cp < 0x10000) {
          out += (char)(0xE0 | (cp >> 12)); out += (char)(0x80 | ((cp >> 6) & 0x3F));
          out += (char)(0x80 | (cp & 0x3F));
        } else {
          out += (char)(0xF0 | (cp >> 18)); out += (char)(0x80 | ((cp >> 12) & 0x3F));
          out += (char)(0x80 | ((cp >> 6) & 0x3F)); out += (char)(0x80 | (cp & 0x3F));
        }
        break;
      }
      default: out += v[i]; break;
    }
  }
  return out;
}

// A JSON string literal for `s`, matching json.dumps(ensure_ascii=False): only the mandatory
// escapes, UTF-8 passed through.
std::string json_quote(const std::string& s) {
  std::string out = "\"";
  for (unsigned char ch : s) {
    switch (ch) {
      case '"':  out += "\\\""; break;
      case '\\': out += "\\\\"; break;
      case '\n': out += "\\n";  break;
      case '\t': out += "\\t";  break;
      case '\r': out += "\\r";  break;
      case '\b': out += "\\b";  break;
      case '\f': out += "\\f";  break;
      default:
        if (ch < 0x20) {
          static const char* hex = "0123456789abcdef";
          out += "\\u00"; out += hex[(ch >> 4) & 0xF]; out += hex[ch & 0xF];
        } else {
          out += (char)ch;
        }
    }
  }
  return out + "\"";
}

// Split a JSON object's top-level members into (key, raw value text) in order. Not a validator: the
// arguments it is handed came from the model or from a client, and a malformed one is reported by
// the caller rather than diagnosed here.
std::vector<std::pair<std::string, std::string>> json_members(const std::string& obj) {
  std::vector<std::pair<std::string, std::string>> out;
  size_t i = obj.find('{');
  if (i == std::string::npos) return out;
  ++i;
  auto skip_ws = [&] { while (i < obj.size() && (obj[i]==' '||obj[i]=='\t'||obj[i]=='\n'||obj[i]=='\r')) ++i; };
  while (i < obj.size()) {
    skip_ws();
    if (i < obj.size() && obj[i] == '}') break;
    if (i >= obj.size() || obj[i] != '"') break;
    const size_t ks = i;
    bool esc = false;
    for (++i; i < obj.size(); ++i) {
      if (esc) { esc = false; continue; }
      if (obj[i] == '\\') { esc = true; continue; }
      if (obj[i] == '"') break;
    }
    if (i >= obj.size()) break;
    const std::string key = json_unquote(obj.substr(ks, ++i - ks));
    skip_ws();
    if (i >= obj.size() || obj[i] != ':') break;
    ++i;
    skip_ws();
    // The value: one token, or a balanced run for a container or string.
    const size_t vs = i;
    int depth = 0;
    bool in_str = false;
    esc = false;
    for (; i < obj.size(); ++i) {
      const char ch = obj[i];
      if (in_str) {
        if (esc) esc = false;
        else if (ch == '\\') esc = true;
        else if (ch == '"') in_str = false;
        continue;
      }
      if (ch == '"') { in_str = true; continue; }
      if (ch == '{' || ch == '[') { ++depth; continue; }
      if (ch == '}' || ch == ']') { if (depth == 0) break; --depth; continue; }
      if (ch == ',' && depth == 0) break;
    }
    std::string val = obj.substr(vs, i - vs);
    while (!val.empty() && (val.back()==' '||val.back()=='\t'||val.back()=='\n'||val.back()=='\r'))
      val.pop_back();
    out.emplace_back(key, val);
    if (i < obj.size() && obj[i] == ',') ++i;
  }
  return out;
}

// response_format_template from the reference: the schema, verbatim, under a fixed heading.
std::string response_format_block(const std::string& schema_json) {
  return "## Response Format:\n\nYou MUST strictly adhere to the following schema to reply:\n" +
         json_compact(schema_json);
}

std::string tools_block(const std::vector<ToolDef>& tools) {
  std::string schemas;
  for (size_t i = 0; i < tools.size(); ++i) {
    if (i) schemas += "\n";
    schemas += json_compact(tools[i].function_json);
  }
  std::string s = "## Tools\n\nYou have access to a set of tools to help answer the user's question. "
                  "You can invoke tools by writing a \"<";
  s += kDsml; s += "tool_calls>\" block like the following:\n\n<";
  s += kDsml; s += "tool_calls>\n<";
  s += kDsml; s += "invoke name=\"$TOOL_NAME\">\n<";
  s += kDsml; s += "parameter name=\"$PARAMETER_NAME\" string=\"true|false\">$PARAMETER_VALUE</";
  s += kDsml; s += "parameter>\n...\n</";
  s += kDsml; s += "invoke>\n<";
  s += kDsml; s += "invoke name=\"$TOOL_NAME2\">\n...\n</";
  s += kDsml; s += "invoke>\n</";
  s += kDsml; s += "tool_calls>\n\nString parameters should be specified as is and set "
                  "`string=\"true\"`. For all other types (numbers, booleans, arrays, objects), "
                  "pass the value in JSON format and set `string=\"false\"`.\n\nIf thinking_mode is "
                  "enabled (triggered by ";
  s += kThinkStart; s += "), you MUST output your complete reasoning inside ";
  s += kThinkStart; s += "..."; s += kThinkEnd;
  s += " BEFORE any tool calls or final response.\n\nOtherwise, output directly after ";
  s += kThinkEnd; s += " with tool calls or final response.\n\n### Available Tool Schemas\n\n";
  s += schemas;
  s += "\n\nYou MUST strictly follow the above defined tool name and parameter schemas to invoke "
       "tool calls.\n";
  return s;
}

// One assistant turn's DSML block, from `encode_arguments_to_dsml` and `tool_call_template`.
std::string render_tool_calls(const std::vector<ToolCall>& tcs) {
  std::string out = "\n\n<";
  out += kDsml; out += "tool_calls>\n";
  for (size_t i = 0; i < tcs.size(); ++i) {
    if (i) out += "\n";
    out += "<"; out += kDsml; out += "invoke name=\"" + tcs[i].name + "\">\n";
    // Arguments that are not a JSON object are wrapped under an "arguments" key, exactly as the
    // reference's except-branch does, rather than dropped.
    auto members = json_members(tcs[i].arguments);
    std::vector<std::pair<std::string, std::string>> fallback;
    if (members.empty() && !tcs[i].arguments.empty() &&
        tcs[i].arguments.find('{') == std::string::npos) {
      fallback.emplace_back("arguments", json_quote(tcs[i].arguments));
      members = fallback;
    }
    for (size_t m = 0; m < members.size(); ++m) {
      if (m) out += "\n";
      const bool is_str = json_is_string(members[m].second);
      out += "<"; out += kDsml;
      out += "parameter name=\"" + members[m].first + "\" string=\"";
      out += is_str ? "true" : "false";
      out += "\">";
      out += is_str ? json_unquote(members[m].second) : json_compact(members[m].second);
      out += "</"; out += kDsml; out += "parameter>";
    }
    out += "\n</"; out += kDsml; out += "invoke>";
  }
  out += "\n</"; out += kDsml; out += "tool_calls>";
  return out;
}

int last_user_index(const std::vector<ChatMsg>& m) {
  for (int i = (int)m.size() - 1; i >= 0; --i)
    if (m[(size_t)i].role == "user" || m[(size_t)i].role == "developer") return i;
  return -1;
}

const char* task_token(const std::string& t) {
  if (t == "action")    return "<｜action｜>";
  if (t == "query")     return "<｜query｜>";
  if (t == "authority") return "<｜authority｜>";
  if (t == "domain")    return "<｜domain｜>";
  if (t == "title")     return "<｜title｜>";
  if (t == "read_url")  return "<｜read_url｜>";
  return nullptr;
}

// merge_tool_messages: the format has no standalone tool role, so a run of tool results becomes one
// user message whose content is the results joined by "\n\n", each in a <tool_result> wrapper. A
// user message following them joins the same block.
std::vector<ChatMsg> merge_tool_messages(const std::vector<ChatMsg>& in) {
  std::vector<ChatMsg> out;
  bool last_was_merged = false;
  for (const ChatMsg& m : in) {
    if (m.role == "tool") {
      std::string block = "<tool_result>" + m.content + "</tool_result>";
      if (last_was_merged && !out.empty()) out.back().content += "\n\n" + block;
      else {
        ChatMsg u; u.role = "user"; u.content = block;
        out.push_back(u);
        last_was_merged = true;
      }
      continue;
    }
    if (m.role == "user" && last_was_merged && !out.empty() && out.back().task.empty()) {
      out.back().content += "\n\n" + m.content;
      out.back().task = m.task;
      continue;
    }
    out.push_back(m);
    last_was_merged = false;
  }
  return out;
}

}  // namespace

const char* const kToolCallsOpen = "\n\n<｜DSML｜tool_calls";
const char* const kThinkEnd      = "</think>";
const char* const kEosStr        = "<｜end▁of▁sentence｜>";

std::string encode_messages(const std::vector<ChatMsg>& in, const EncodeOpts& opts) {
  std::vector<ChatMsg> msgs = merge_tool_messages(in);

  // Tools anywhere force the whole conversation to keep its reasoning.
  bool has_tools = false;
  for (const ChatMsg& m : msgs) has_tools = has_tools || !m.tools.empty();
  const bool thinking = opts.thinking_mode == ThinkingMode::Thinking;
  const bool drop = opts.drop_thinking && !has_tools;

  const int last_user = last_user_index(msgs);

  std::string out = opts.add_bos ? kBos : "";
  for (size_t idx = 0; idx < msgs.size(); ++idx) {
    const ChatMsg& m = msgs[idx];

    if (idx == 0 && thinking) {
      if (opts.reasoning_effort == ReasoningEffort::High) out += kEffortHigh;
      else if (opts.reasoning_effort == ReasoningEffort::Max) out += kEffortMax;
    }

    if (m.role == "system") {
      out += m.content;
      if (!m.tools.empty()) out += "\n\n" + tools_block(m.tools);
      if (!m.response_format.empty()) out += "\n\n" + response_format_block(m.response_format);
    } else if (m.role == "developer") {
      out += kUser;
      out += m.content;
      if (!m.tools.empty()) out += "\n\n" + tools_block(m.tools);
      if (!m.response_format.empty()) out += "\n\n" + response_format_block(m.response_format);
    } else if (m.role == "user") {
      out += kUser;
      out += m.content;
    } else if (m.role == "latest_reminder") {
      out += kReminder;
      out += m.content;
    } else if (m.role == "assistant") {
      // A turn that answers a quick-instruction task carries no thinking at all.
      const bool prev_task = idx > 0 && !msgs[idx - 1].task.empty();
      if (thinking && !prev_task && (!drop || (int)idx > last_user))
        out += m.reasoning_content + kThinkEnd;
      out += m.content;
      if (!m.tool_calls.empty()) out += render_tool_calls(m.tool_calls);
      out += kEosStr;
    }

    // The transition, which only fires when an assistant turn (or the end) follows.
    if (idx + 1 < msgs.size() && msgs[idx + 1].role != "assistant" &&
        msgs[idx + 1].role != "latest_reminder")
      continue;

    if (const char* tt = task_token(m.task)) {
      if (m.task != "action") {
        out += tt;
      } else {
        out += kAssistant;
        out += thinking ? kThinkStart : kThinkEnd;
        out += tt;
      }
    } else if (m.role == "user" || m.role == "developer") {
      out += kAssistant;
      if (thinking && (!drop || (int)idx >= last_user)) out += kThinkStart;
      else                                              out += kThinkEnd;
    }
  }
  return out;
}

// ---- parsing ------------------------------------------------------------------------------------

namespace {

// The reference's _read_until_stop: the EARLIEST of the stops wins, not the first listed.
size_t read_until(size_t i, const std::string& t, const std::vector<std::string>& stops,
                  std::string* content, std::string* matched) {
  size_t best = t.size();
  const std::string* hit = nullptr;
  for (const std::string& s : stops) {
    const size_t p = t.find(s, i);
    if (p != std::string::npos && p < best) { best = p; hit = &s; }
  }
  *content = t.substr(i, best - i);
  if (hit) { *matched = *hit; return best + hit->size(); }
  matched->clear();
  return t.size();
}

}  // namespace

ParsedMessage parse_completion(const std::string& text, ThinkingMode mode) {
  ParsedMessage out;
  const std::string dsml = kDsml;
  const std::string tc_open  = std::string("\n\n<") + dsml + "tool_calls";
  const std::string tc_close = std::string("</") + dsml + "tool_calls>";
  const std::string inv_open = std::string("<") + dsml + "invoke";
  const std::string inv_close= std::string("</") + dsml + "invoke";
  const std::string par_open = std::string("<") + dsml + "parameter";
  const std::string par_close= std::string("/") + dsml + "parameter";

  auto fail = [&](const char* why) {
    out.ok = false;
    out.error = why;
    out.content = text;              // the raw text, which is all a caller can honestly be given
    out.reasoning_content.clear();
    out.tool_calls.clear();
    return out;
  };

  size_t i = 0;
  std::string chunk, stop;

  if (mode == ThinkingMode::Thinking) {
    i = read_until(i, text, {kThinkEnd, tc_open}, &chunk, &stop);
    if (stop != kThinkEnd) return fail("missing </think>");
    out.reasoning_content = chunk;
  }

  i = read_until(i, text, {kEosStr, tc_open}, &chunk, &stop);
  out.content = chunk;

  if (stop == tc_open) {
    // Inside the block: alternating `invoke` headers and `parameter` bodies until the close.
    while (i < text.size()) {
      i = read_until(i, text, {inv_open, tc_close}, &chunk, &stop);
      if (chunk != ">\n") return fail("tool call format: expected '>\\n'");
      if (stop == tc_close) break;
      if (stop.empty()) return fail("unterminated tool_calls block");

      i = read_until(i, text, {par_open, inv_close}, &chunk, &stop);
      // ` name="X">\n`
      const size_t nq = chunk.find("name=\"");
      const size_t ne = nq == std::string::npos ? std::string::npos : chunk.find("\">\n", nq + 6);
      if (nq == std::string::npos || ne == std::string::npos) return fail("tool name format");
      ToolCall tc;
      tc.name = chunk.substr(nq + 6, ne - nq - 6);

      std::string args = "{";
      std::set<std::string> seen;
      bool first = true;
      while (stop == par_open) {
        i = read_until(i, text, {par_close}, &chunk, &stop);
        // ` name="k" string="true|false">value<`
        const size_t kq = chunk.find("name=\"");
        const size_t k_end = kq == std::string::npos ? std::string::npos : chunk.find("\" string=\"", kq + 6);
        if (kq == std::string::npos || k_end == std::string::npos) return fail("parameter format");
        const std::string key = chunk.substr(kq + 6, k_end - kq - 6);
        const size_t sv = k_end + 10;
        const size_t se = chunk.find("\">", sv);
        if (se == std::string::npos) return fail("parameter format");
        const std::string is_str = chunk.substr(sv, se - sv);
        if (is_str != "true" && is_str != "false") return fail("parameter string= must be true|false");
        std::string val = chunk.substr(se + 2);
        if (!val.empty() && val.back() == '<') val.pop_back();

        // The reference rejects a repeated parameter; a silently duplicated JSON key would otherwise
        // reach the caller and be resolved differently by whichever parser reads it.
        if (seen.count(key)) return fail("duplicate parameter name");
        seen.insert(key);
        if (!first) args += ", ";
        first = false;
        args += json_quote(key) + ": " + (is_str == "true" ? json_quote(val) : val);

        i = read_until(i, text, {par_open, inv_close}, &chunk, &stop);
        if (chunk != ">\n") return fail("parameter format: expected '>\\n'");
      }
      args += "}";
      tc.arguments = args;
      out.tool_calls.push_back(tc);
    }
    i = read_until(i, text, {kEosStr}, &chunk, &stop);
    if (!chunk.empty()) return fail("content after tool calls");
  } else if (stop != kEosStr && !stop.empty()) {
    return fail("unexpected stop token");
  }

  // A special token loose in the text means the parse mis-framed something.
  for (const char* sp : {kBos, kEosStr, kThinkStart, kThinkEnd, kDsml})
    if (out.content.find(sp) != std::string::npos ||
        out.reasoning_content.find(sp) != std::string::npos)
      return fail("special token in content");

  return out;
}

}  // namespace aff
